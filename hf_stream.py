# version: 2026-09-05.2 - SWAP-STORM GUARD: a tight box does not always raise
#   MemoryError - Windows slides into a swap-storm (allocations succeed via
#   the pagefile, wall-clock dies, the run looks frozen). Two guards: (1)
#   BlockStreamRunner re-checks io-cache ram at EVERY stage and downgrades to
#   disk when the packed copy would not fit (MOE_FORCE_IO_RAM=1 overrides);
#   (2) a RAM watchdog thread drops the ram cache the moment free memory
#   crosses the floor, before the thrash starts (MOE_NO_RAM_WATCHDOG=1 off).
# version: 2026-09-05.1 - LOW-RAM FIX: a MemoryError during a block load no
#   longer kills the run - the io-cache ram GGUF copy (~= packed file size) is
#   dropped and the block load retries from disk/mmap (BlockStreamRunner
#   ._degrade_ram_cache; prefetch is disabled alongside, it doubles the peak).
"""Streaming MoE model run at low RAM: the full model is NEVER materialized.

Why: at stages 3-4 the pipeline used to load the whole model (~14 GB bf16 for
OLMoE-7B) - on 8-16 GB RAM machines Windows silently killed the process
(commit limit exhaustion). That phase no longer exists.

How: the model is built on a meta device (0 bytes of data); only the backbone
(~1-1.5 GB: embeddings, attention, routers, norms, head) is materialized in
RAM. Each MoE block's experts (~0.8 GB for OLMoE) are read from disk exactly
for the duration of THEIR layer's forward pass (forward hooks) and freed
immediately after. Layers execute through the stock transformers code - the
numeric result matches the full model.

Peak RAM: backbone + one expert block + activations ~ 2-3 GB for OLMoE-7B
(versus ~15 GB). Cost: experts are read from disk on every pass (~13 GB per
chunk run for OLMoE) - an SSD is required; generation is slow, so the
pipeline shortens demo generation in streaming mode.

PREFETCH (overlap load with compute): a background thread dequantizes the
NEXT block while the current one computes, hiding the ~5-7 s per-block dequant
latency. Cost: +1 expert block in RAM (~0.2-0.8 GB depending on the model).
prefetch=0 disables the thread (purely synchronous, the old behavior); any
prefetch error falls back to the synchronous path automatically.

Two weight source variants (numerically equivalent):
  1. full dequant checkpoint (safetensors in src) - fp16 read ~0.8 GB/block;
  2. the GGUF itself (gguf=... or the _gguf_source.json marker in src) -
     ON-THE-FLY dequant, the Q4 packing is read from disk (~0.23 GB/block,
     ~3.5x less traffic, no 14-GB checkpoint), cost ~5 s CPU per block load
     (OLMoE).

The calibration activation pool is collected in the same pass: (MoE input ->
output) pairs are captured by hooks and depend on neither text order nor
context - the fit samples them as independent vectors (bootstrap inference).
"""
import gc
import os
import queue
import threading
import time

import torch
import torch.nn as nn

from hf_field_transform import find_moe_blocks


def free_ram_gb():
    """GB of AVAILABLE RAM; None when it cannot be measured. psutil ->
    /proc/meminfo -> Windows GlobalMemoryStatusEx; never raises."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0]) * 1024
        return info.get("MemAvailable", info.get("MemTotal", 0)) / 1e9
    except Exception:
        pass
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 1e9
    except Exception:
        pass
    return None


def ram_cache_fits(avail_gb, gguf_gb):
    """Whether the io-cache ram packed copy may be armed (2026-09-05.2).

    The cache is ~= the packed GGUF size, and it grows WHILE the working set
    (backbone + capture chunks + dequant scratch) is live. Seen in the wild:
    3.2 GB free + a 2.8 GB GGUF -> around block 13 of the first refine window
    Windows entered a swap-storm - allocations kept succeeding through the
    pagefile, so no MemoryError ever fired and the run just froze. The bar is
    therefore deliberately generous: 1.5x the GGUF + 1 GB headroom. An
    unmeasurable amount of RAM never second-guesses the user."""
    if avail_gb is None:
        return True
    return avail_gb >= gguf_gb * 1.5 + 1.0


class BlockStreamRunner:
    """Full-model replacement for the base/calibration stages.

    Delegates everything unknown to self.model (config, generate,
    named_modules...), so pipeline code barely changes: eval_logits_cache_disk,
    collect_pairs, find_moe_blocks work as with a regular model.
    """

    def __init__(self, src, dtype=torch.bfloat16, device="cpu", progress=True,
                 gguf=None, prefetch=1, io_workers=1, io_cache="disk"):
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

        self.src, self.dtype = src, dtype
        self.device = torch.device(device)
        self.progress = bool(progress)
        self._io_workers = max(1, int(io_workers))   # GGUF dequant threads
        self._io_cache = str(io_cache)               # disk | ram (packed cache)
        self._oom_hint = False        # ram cache already dropped on an OOM
        self._gguf = self._open_gguf(gguf)   # weight source: None -> safetensors
        cfg = AutoConfig.from_pretrained(src)
        with torch.device("meta"):
            try:
                self.model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
            except TypeError:  # older transformers versions
                self.model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        # service attrs - BEFORE any calls (otherwise __getattr__ goes to the model)
        self._handles = {}          # file -> safe_open
        self._pinned = None         # block prefix pinned by with_block
        self._loaded = None
        self._hooks = []
        self._stream_n = 0
        self._stream_bytes = 0
        self._load_time = 0.0
        self._slow_hint = False
        self._expert_keys = {}
        self._moe_prefixes = []
        # prefetch state
        self._pf_depth = max(0, int(prefetch))
        self._pf_buf = {}           # prefix -> {key: tensor} ready to assign
        self._pf_lock = threading.Lock()
        self._pf_cv = threading.Condition(self._pf_lock)
        self._pf_queue = queue.Queue(maxsize=4)
        self._pf_thread = None
        self._pf_dead = False       # worker failed -> sync fallback
        self._pf_hint = False
        # RAM watchdog (2026-09-05.2) - swap-storm guard
        self._wd_stop = threading.Event()
        self._wd_thread = None
        self._wd_poll = 2.0         # seconds between free-RAM samples
        self._fix_buffers()
        self._load_backbone(src)
        self._map_expert_keys(src)
        self._install_hooks()
        self._start_prefetch()
        self._start_ram_watchdog()
        gp = os.path.join(src, "generation_config.json")
        if os.path.exists(gp):
            self.model.generation_config = GenerationConfig.from_pretrained(src)

    # ------------------------------------------------------------ delegation
    def __getattr__(self, name):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def __call__(self, *a, **kw):
        return self.model(*a, **kw)

    # ------------------------------------------------------------ backbone
    def _open_gguf(self, gguf):
        """GGUF weight source: explicit path or the _gguf_source.json marker in
        src. A full dequant checkpoint (safetensors in src) takes priority: it
        is already on disk and reads without dequant."""
        if gguf is None and os.path.isdir(self.src):
            if any(f.endswith(".safetensors") for f in os.listdir(self.src)):
                return None
            mp = os.path.join(self.src, "_gguf_source.json")
            if os.path.isfile(mp):
                import json as _json
                with open(mp, encoding="utf-8") as f:
                    gguf = _json.load(f).get("gguf")
                if gguf and os.path.isfile(gguf):
                    print(f"streaming: reading weights straight from the GGUF "
                          f"({os.path.basename(gguf)}), per-block on-the-fly "
                          f"dequant", flush=True)
        if not gguf:
            return None
        from hf_gguf_to_hf import GgufHfSource
        if self._io_cache == "ram":
            # 2026-09-05.2: re-check at EVERY stage (free RAM measured at
            # stage 0 says nothing about stage 5b) - an explicit request is
            # still honored only when it cannot swap-storm the machine.
            use_ram = True
            try:
                size_gb = os.path.getsize(gguf) / 1e9
            except OSError:
                size_gb = None
            avail_gb = free_ram_gb()
            if size_gb is not None and not ram_cache_fits(avail_gb, size_gb) \
                    and os.environ.get("MOE_FORCE_IO_RAM") != "1":
                use_ram = False
                self._io_cache = "disk"
                print(f"streaming: io-cache ram DOES NOT FIT this machine "
                      f"({size_gb:.1f} GB GGUF, {avail_gb:.1f} GB free - it "
                      f"would swap-storm) -> reading from disk instead "
                      f"(MOE_FORCE_IO_RAM=1 overrides)", flush=True)
            if use_ram:
                print(f"streaming: io-cache ram - raw GGUF tensors are copied to "
                      f"RAM on first touch (~= the packed file size, ~2.7 GB for a "
                      f"1B Q8 MoE); later passes read nothing from disk", flush=True)
        return GgufHfSource(gguf, io_workers=self._io_workers,
                            cache_ram=(self._io_cache == "ram"))

    @staticmethod
    def _module_names(model):
        return {id(m): n for n, m in model.named_modules()}

    def _decoder_layers(self):
        core = getattr(self.model, "model", self.model)
        layers = getattr(core, "layers", None)
        if layers is None:
            raise RuntimeError("decoder layers not found (model.layers)")
        return list(layers)

    def _fix_buffers(self):
        """On a meta model buffers (rotary inv_freq) are empty - recreate the
        rotary buffers on CPU and move real buffers into the backbone."""
        fixed = 0
        for name, mod in self.model.named_modules():
            inv = getattr(mod, "inv_freq", None)
            if isinstance(inv, torch.Tensor) and inv.device.type == "meta":
                ref = type(mod)(mod.config)  # real module on CPU (light)
                with torch.no_grad():
                    for bn, buf in ref.named_buffers(recurse=False):
                        t = buf.detach().clone().to(self.device)
                        if bn in dict(mod.named_buffers(recurse=False)):
                            mod.register_buffer(bn, t, persistent=False)
                        else:
                            setattr(mod, bn, t)
                    if hasattr(ref, "attention_scaling"):
                        mod.attention_scaling = ref.attention_scaling
                fixed += 1
                del ref
        if fixed:
            print(f"streaming: recreated rotary buffers: {fixed}", flush=True)

    def _assign(self, key, t):
        parts = key.split(".")
        obj = self.model
        for p in parts[:-1]:
            obj = getattr(obj, p)  # AttributeError -> key not in the model
        leaf = parts[-1]
        if isinstance(obj, nn.Module):
            par = obj._parameters.get(leaf)
            if par is not None:
                if t.device != self.device:
                    t = t.to(self.device)
                # swap, not .data=: between meta and a real tensor set_data is
                # forbidden, while swap keeps the Parameter object (tied links
                # survive)
                torch.utils.swap_tensors(par, t)
                return
            if leaf in obj._buffers and obj._buffers[leaf] is not None:
                obj._buffers[leaf] = t.to(self.device)
                return
        raise KeyError(key)

    def _read_tensor(self, fn, key):
        f = self._handles.get(fn)
        if f is None:
            from safetensors import safe_open
            f = safe_open(os.path.join(self.src, fn), framework="pt")
            self._handles[fn] = f
        t = f.get_tensor(key)
        if t.dtype != self.dtype:
            t = t.to(self.dtype)
        else:
            t = t.clone()  # safetensors mmap view - read-only
        return t

    def _get_weight(self, fn, key):
        """Tensor from disk: safetensors shard or on-the-fly dequant from GGUF.
        Thread-safe: used by both the main thread and the prefetch worker."""
        if self._gguf is not None:
            t = self._gguf.get(key)          # already fp16 torch on CPU
            if t.dtype != self.dtype:
                t = t.to(self.dtype)
            return t
        return self._read_tensor(fn, key)

    def _load_backbone(self, src):
        if self._gguf is not None:
            all_keys = self._gguf.keys()
            skipped = sum(1 for k in all_keys if ".experts." in k)
            jobs = [("gguf", k) for k in all_keys if ".experts." not in k]
            print(f"streaming: reading backbone from the quantized GGUF "
                  f"(on-the-fly dequant, {len(jobs)} tensors)...", flush=True)
        else:
            files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
            if not files:
                raise RuntimeError(f"no safetensors files in {src}")
            skipped = 0
            jobs = []
            for fn in files:
                for key in self._open_shard(fn).keys():
                    if ".experts." in key:
                        skipped += 1
                        continue        # experts load block by block!
                    jobs.append((fn, key))
        if skipped:
            print(f"streaming: expert tensors skipped: {skipped} "
                  f"- they are read block by block on their layer's pass",
                  flush=True)
        gb = 0.0
        t0 = time.time()
        for i, (fn, key) in enumerate(jobs):
            t = self._get_weight(fn, key)
            try:
                self._assign(key, t)
            except (AttributeError, KeyError):
                print(f"\n  (backbone has no '{key}' - skipped)", flush=True)
            gb += t.numel() * t.element_size()
            del t
            if self.progress and (i + 1) % 8 == 0:
                print(f"\r    backbone: {i + 1}/{len(jobs)} tensors, "
                      f"{gb / 1e9:.2f} GB, {time.time() - t0:.0f} s",
                      end="", flush=True)
        if self.progress and len(jobs) >= 8:
            print(flush=True)
        src_txt = ("from the quantized GGUF (on-the-fly dequant)"
                   if self._gguf is not None else "from disk block by block")
        print(f"streaming: backbone in RAM ({gb / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f} s), experts will be read {src_txt}",
              flush=True)
        bad = [n for n, p in self.model.named_parameters(remove_duplicate=False)
               if p.device.type == "meta" and ".experts." not in n]
        if bad:
            raise RuntimeError("backbone tensors not materialized: "
                               + ", ".join(bad[:8]))
        # invariant: experts must NOT land in the backbone (RAM would hit ~14 GB)
        bad_exp = [n for n, p in self.model.named_parameters(remove_duplicate=False)
                   if ".experts." in n and p.device.type != "meta"]
        if bad_exp:
            raise RuntimeError(f"experts leaked into the backbone "
                               f"({len(bad_exp)} tensors) - RAM would blow up to "
                               f"~14 GB; the .experts. filter failed")

    def _open_shard(self, fn):
        from safetensors import safe_open
        f = self._handles.get(fn)
        if f is None:
            f = safe_open(os.path.join(self.src, fn), framework="pt")
            self._handles[fn] = f
        return f

    # ------------------------------------------------------------ experts
    def _map_expert_keys(self, src):
        """block prefix -> [(file, key), ...] over the source keys."""
        self._expert_keys = {}
        if self._gguf is not None:
            for key in self._gguf.keys():
                if ".experts." not in key:
                    continue
                prefix = key.rsplit(".experts.", 1)[0] + ".experts"
                self._expert_keys.setdefault(prefix, []).append(("gguf", key))
        else:
            for fn in sorted(f for f in os.listdir(src)
                             if f.endswith(".safetensors")):
                f = self._open_shard(fn)
                for key in f.keys():
                    if ".experts." not in key:
                        continue
                    prefix = key[:key.index(".experts.") + len(".experts")] \
                        if key.endswith(".experts") else \
                        key.rsplit(".experts.", 1)[0] + ".experts"
                    self._expert_keys.setdefault(prefix, []).append((fn, key))
        self._moe_prefixes = [n + ".experts" for n, _ in find_moe_blocks(self.model)]
        missing = [p for p in self._moe_prefixes if p not in self._expert_keys]
        if missing:
            raise RuntimeError(f"no block weights in the source: {missing[:4]}")

    def _install_hooks(self):
        names = self._module_names(self.model)
        handles = []
        for layer in self._decoder_layers():
            lname = names.get(id(layer))
            if lname is None:
                continue
            mine = [p for p in self._moe_prefixes
                    if p.startswith(lname + ".") and p != lname]
            for pf in mine[:1]:  # one MoE block per layer (contract)
                handles.append(layer.register_forward_pre_hook(self._make_pre(pf)))
                handles.append(layer.register_forward_hook(self._make_post(pf)))
        if not handles:
            raise RuntimeError("no MoE blocks found in decoder layers")
        return handles

    # ------------------------------------------------------------ prefetch
    def _start_prefetch(self):
        """Background worker: dequantizes upcoming expert blocks while the
        current layer computes. Any failure silently downgrades to the sync
        path (prefetch is a pure optimization)."""
        if self._pf_depth <= 0:
            return
        if os.environ.get("MOE_NO_PREFETCH") == "1":
            return
        try:
            self._pf_thread = threading.Thread(
                target=self._pf_loop, name="moe-prefetch", daemon=True)
            self._pf_thread.start()
            print(f"streaming: prefetch ON (depth {self._pf_depth}, "
                  f"+~1 block in RAM)", flush=True)
        except Exception:
            self._pf_thread = None

    def _pf_next_prefixes(self, after):
        """Next block prefixes in execution order (layers run 0..N cyclically)."""
        if after not in self._moe_prefixes:
            return []
        i = self._moe_prefixes.index(after)
        n = len(self._moe_prefixes)
        return [self._moe_prefixes[(i + k) % n] for k in range(1, n + 1)]

    def _pf_request(self, prefix):
        """Ask the worker to have the NEXT blocks ready after `prefix`."""
        if self._pf_thread is None or self._pf_dead or self._pinned is not None:
            return
        nxt = self._pf_next_prefixes(prefix)[:self._pf_depth]
        for p in nxt:
            with self._pf_lock:
                if p in self._pf_buf:
                    continue
            try:
                self._pf_queue.put_nowait(p)
            except queue.Full:
                pass

    def _pf_loop(self):
        while True:
            try:
                prefix = self._pf_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if prefix is None:
                return
            with self._pf_lock:
                already = prefix in self._pf_buf
            if already:
                continue
            try:
                buf = {}
                for fn, key in self._expert_keys[prefix]:
                    buf[key] = self._get_weight(fn, key)  # dequant (expensive)
                with self._pf_lock:
                    self._pf_buf[prefix] = buf
                    while len(self._pf_buf) > max(1, self._pf_depth):
                        self._pf_buf.pop(next(iter(self._pf_buf)))
            except Exception as e:  # noqa: BLE001
                # downgrading to the synchronous path - never fatal
                with self._pf_lock:
                    self._pf_buf.clear()
                    self._pf_dead = True
                if not self._pf_hint:
                    self._pf_hint = True
                    print(f"\n    (prefetch disabled, continuing synchronously: "
                          f"{type(e).__name__}: {e})", flush=True)
                return

    # ------------------------------------------------------------ load/free
    def _degrade_ram_cache(self, reason="OOM"):
        """OOM recovery (2026-09-05.1; reason param 2026-09-05.2): the
        packed-GGUF RAM cache (io-cache ram) is a pure optimization - when an
        allocation fails (or the RAM watchdog sees free memory crossing the
        floor), drop it and read from disk/mmap for the rest of the run
        instead of crashing or swap-storming. Seen in the wild: a 2.8 GB GGUF
        with 3.2 GB free - backbone + ram cache + one dequant scratch do not
        fit, stage 4 died with "Unable to allocate 128 MiB". Prefetch is
        disabled alongside (it keeps a whole extra block in RAM)."""
        dropped = False
        if self._gguf is not None and getattr(self._gguf, "cache_ram", False):
            try:
                self._gguf.drop_ram_cache()
                dropped = True
            except AttributeError:      # a weight source without a ram cache
                self._gguf.cache_ram = False
            except Exception:  # noqa: BLE001
                pass
        with self._pf_lock:
            self._pf_buf.clear()
            self._pf_dead = True
        gc.collect()
        if dropped and not self._oom_hint:
            self._oom_hint = True
            print(f"\n    ({reason}: dropped the io-cache ram GGUF copy - "
                  f"continuing from disk/mmap; the ram cache does not fit this "
                  f"machine, next time run with --io-cache disk)", flush=True)
        return dropped

    # ------------------------------------------------------------ watchdog
    def _start_ram_watchdog(self):
        """2026-09-05.2: the io-cache ram copy grows block by block, and on a
        tight box Windows may respond with a swap-storm instead of a
        MemoryError (the refine capture pass froze there in the wild: 13 block
        loads, then nothing). A daemon thread samples free RAM and drops the
        cache the moment it crosses the floor - BEFORE the thrash starts."""
        if self._gguf is None or not getattr(self._gguf, "cache_ram", False):
            return
        if os.environ.get("MOE_NO_RAM_WATCHDOG") == "1":
            return
        try:
            self._wd_thread = threading.Thread(
                target=self._ram_watchdog_loop, name="moe-ram-watchdog",
                daemon=True)
            self._wd_thread.start()
        except Exception:
            self._wd_thread = None

    def _ram_watchdog_loop(self):
        while not self._wd_stop.wait(self._wd_poll):
            gguf = self._gguf
            if gguf is None or not getattr(gguf, "cache_ram", False):
                return                  # cache already gone - nothing to guard
            avail = free_ram_gb()
            if avail is not None and avail < 0.6:
                self._degrade_ram_cache(
                    reason=f"watchdog: only {avail:.1f} GB RAM left")
                return

    def _make_pre(self, prefix):
        def pre(module, args):
            if self._pinned is None and self._loaded != prefix:
                self._load_block(prefix)
        return pre

    def _make_post(self, prefix):
        def post(module, args, output):
            if self._pinned is None and self._loaded == prefix:
                self._free_block(prefix)
        return post

    @torch.no_grad()
    def _load_block(self, prefix):
        nb = 0
        t0 = time.time()
        is_gguf = self._gguf is not None

        def bytes_of(fn, key, t):
            # count DISK READS: for GGUF it is the Q4 packing, for shards the
            # stored tensor size
            return self._gguf.disk_bytes(key) if is_gguf \
                else t.numel() * t.element_size()

        # fast path: tensors already dequantized by the prefetch worker
        with self._pf_lock:
            pre = self._pf_buf.pop(prefix, None) if self._pf_thread is not None \
                else None
        try:
            if pre is None:
                for fn, key in self._expert_keys[prefix]:
                    t = self._get_weight(fn, key)
                    self._assign(key, t)
                    nb += bytes_of(fn, key, t)
                    del t
            else:
                for fn, key in self._expert_keys[prefix]:
                    t = pre.get(key)
                    if t is None:                   # partial buffer - fallback
                        t = self._get_weight(fn, key)
                    self._assign(key, t)
                    nb += bytes_of(fn, key, t)
                    del t
        except MemoryError:
            # io-cache ram does not fit this machine: drop the packed GGUF
            # copy (+~2-3 GB) and retry the whole block from disk/mmap. The
            # partial _assign above is safe to redo: swap_tensors is idempotent.
            self._degrade_ram_cache()
            for fn, key in self._expert_keys[prefix]:
                t = self._get_weight(fn, key)
                self._assign(key, t)
                nb += bytes_of(fn, key, t)
                del t
        self._loaded = prefix
        self._stream_n += 1
        self._stream_bytes += nb
        dt = time.time() - t0
        self._load_time += dt
        if self.progress:
            print(f"\r    ... experts from disk: {self._stream_n} block loads, "
                  f"{self._stream_bytes / 1e9:.0f} GB read (block {dt:.1f} s)",
                  end="", flush=True)
        if dt > 25 and not self._slow_hint:
            self._slow_hint = True
            print("\n    (slow: weak CPU and/or HDD; this pass happens once - "
                  "the pool is cached. Faster: --full-dequant, at the cost of "
                  "+14 GB temporarily on disk)", flush=True)
        self._pf_request(prefix)

    @torch.no_grad()
    def _free_block(self, prefix):
        obj = self.model.get_submodule(prefix)
        for _, p in obj.named_parameters(recurse=True):
            empty = torch.empty(p.shape, dtype=p.dtype, device="meta")
            torch.utils.swap_tensors(p, empty)
        self._loaded = None

    # ------------------------------------------------------------ public
    def with_block(self, i):
        """Context: experts of the i-th MoE block are pinned in RAM
        (for expert_means/geometry outside forward)."""
        return self._Pin(self, self._moe_prefixes[i])

    class _Pin:
        def __init__(self, runner, prefix):
            self.r, self.p = runner, prefix

        def __enter__(self):
            self.r._pinned = self.p
            if self.r._loaded != self.p:
                self.r._load_block(self.p)
            return self.r

        def __exit__(self, *exc):
            self.r._free_block(self.p)
            self.r._pinned = None
            return False

    def close(self):
        self._wd_stop.set()          # stop the RAM watchdog
        self._wd_thread = None
        if self._pf_thread is not None:
            try:
                self._pf_queue.put_nowait(None)
            except queue.Full:
                pass
            self._pf_thread = None
        with self._pf_lock:
            self._pf_buf.clear()
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for f in self._handles.values():
            del f
        self._handles = {}
        if self.progress:
            avg = self._load_time / self._stream_n if self._stream_n else 0.0
            print(f"\n    streaming finished: {self._stream_n} block loads, "
                  f"{self._stream_bytes / 1e9:.1f} GB read from disk, "
                  f"block-load time {self._load_time:.0f} s "
                  f"(avg {avg:.2f} s/block)", flush=True)
        if self._gguf is not None:
            try:
                n, gb, hits = self._gguf.cache_stats()
                if n:
                    print(f"    io-cache ram: {n} tensors ({gb:.2f} GB) held in "
                          f"RAM, {hits} served-from-RAM hits", flush=True)
            except AttributeError:
                pass
        self._gguf = None            # release the GGUF mmap
