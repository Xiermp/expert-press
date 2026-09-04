# version: 2026-09-05.2 - swap-storm guard tests: runner-level io-cache ram
# fit re-check, MOE_FORCE_IO_RAM override, RAM watchdog, refine flush at.
# version: 2026-09-05.1 - LOW-RAM FIX tests: convert_tensor memory safety,
# keep-smaller pool semantics, OOM degrade of io-cache ram.
"""Toy bench for the LOW-RAM FIX (2026-09-05.1) + SWAP-STORM GUARD (05.2).

Sections:
  A. convert_tensor: bit-exact vs the OLD single-shot/split implementation on
     the exact crash shape (64, 512, 1024) Q8_0 + 2D + F16 + the old-split
     shape; peak RSS of the new path must be ~= the output, not 2-3x.
  B. pool_is_complete: a usable smaller cached pool is KEPT (keep_smaller),
     --pool-recalibrate forces re-collection, notices print once per run.
  C. BlockStreamRunner._load_block survives a MemoryError: drops the ram
     cache, retries from disk (fake runner, no model needed).
  D. GgufHfSource.drop_ram_cache clears the process-wide raw cache.
  E. ensure_prereqs end-to-end: fit-only plan with a smaller usable pool does
     not exit; --pool-recalibrate restores the hard exit.
  F. swap-storm guard (2026-09-05.2): ram_cache_fits boundaries, the runner
     re-check downgrades ram->disk on a tight box (and MOE_FORCE_IO_RAM=1
     overrides), the watchdog drops the cache on low free RAM, refine_flush_at.

Run:  python test_lowram_fix.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from contextlib import redirect_stdout

import numpy as np
import torch

import hf_gguf_to_hf as g2h
import hf_pipeline as hp
import hf_stream as hs

PASS = 0
FAIL = []


def ok(cond, name):
    global PASS
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}")


def capture():
    return io.StringIO()


# ------------------------------------------------------------------ section A

def pack_q8_0(vals):
    """fp32 values -> Q8_0 packed bytes (2B fp16 scale + 32 x int8 per block)."""
    v = np.ascontiguousarray(vals, dtype=np.float32).reshape(-1, 32)
    scale = np.abs(v).max(axis=1).astype(np.float32) / np.float32(127.0)
    scale[scale == 0] = np.float32(1e-30)
    q = np.rint(v / scale[:, None]).astype(np.int8)
    blk = np.concatenate(
        [scale.astype(np.float16).reshape(-1, 1).view(np.uint8),
         q.view(np.uint8)], axis=1)
    assert blk.shape[1] == 34            # Q8_0 block size
    return np.ascontiguousarray(blk.reshape(-1))


def old_convert_tensor(rt, g, out_dtype, workers=1):
    """Verbatim copy of the pre-2026-09-05.1 implementation (reference)."""
    from gguf import GGMLQuantizationType, dequantize
    from gguf.constants import GGML_QUANT_SIZES
    data = rt.data
    q = rt.tensor_type
    target = tuple(int(x) for x in np.asarray(rt.shape)[::-1])
    if q in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        arr = np.ascontiguousarray(data, dtype=np.float32)
    else:
        ne = [int(x) for x in rt.shape]
        n_last = ne[-1] if ne else 1
        blk_elems = GGML_QUANT_SIZES[q][0]
        slab_elems = int(np.prod(ne[:-1])) if len(ne) > 1 else int(ne[0])
        can_split = (workers > 1 and n_last >= 2
                     and data.ndim == 1 and data.shape[0] % n_last == 0
                     and slab_elems >= blk_elems > 0
                     and slab_elems % blk_elems == 0 and slab_elems > 4_000_000)
        if can_split:
            import concurrent.futures
            nbytes = data.shape[0] // n_last
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(
                    lambda i: dequantize(data[i * nbytes:(i + 1) * nbytes], q),
                    range(n_last)))
            arr = np.concatenate(parts).astype(np.float32).reshape(target)
        else:
            arr = np.ascontiguousarray(dequantize(data, q)).astype(np.float32)
            if arr.shape != target:
                arr = arr.reshape(target)
    return arr


def peak_rss(fn):
    """(fn(), peak_rss_bytes) sampled in a 2 ms background thread."""
    import psutil
    p = psutil.Process()
    base = p.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], p.memory_info().rss)
            time.sleep(0.002)

    th = threading.Thread(target=sample)
    th.start()
    try:
        out = fn()
    finally:
        stop.set()
        th.join()
    return out, peak[0] - base


def section_a():
    print("A. convert_tensor: bit-exact + peak RSS")
    from gguf import GGMLQuantizationType
    rng = np.random.default_rng(0xC0FFEE)

    # the exact crash shape from the user's traceback
    ne = [1024, 512, 64]
    target = tuple(ne[::-1])                       # (64, 512, 1024)
    vals = rng.standard_normal(target).astype(np.float32)
    rt = types.SimpleNamespace(data=pack_q8_0(vals), tensor_type=GGMLQuantizationType.Q8_0,
                               shape=tuple(ne))

    ref = old_convert_tensor(rt, None, np.float32, workers=1)
    new1 = g2h.convert_tensor(rt, None, np.float32, workers=1)
    new2 = g2h.convert_tensor(rt, None, np.float32, workers=2)
    ok(np.array_equal(new1, ref), "Q8_0 (64,512,1024): workers=1 bit-exact")
    ok(np.array_equal(new2, ref), "Q8_0 (64,512,1024): workers=2 bit-exact")
    ok(new1.shape == target, "shape == HF target")

    del ref, new1, new2
    (_, pk_new), (_, pk_old) = peak_rss(
        lambda: g2h.convert_tensor(rt, None, np.float32, workers=1)), \
        peak_rss(lambda: old_convert_tensor(rt, None, np.float32, workers=1))
    out_mb = int(np.prod(target)) * 4 / 2 ** 20
    print(f"    peak RSS: new {pk_new / 2 ** 20:.0f} MiB vs old "
          f"{pk_old / 2 ** 20:.0f} MiB (output {out_mb:.0f} MiB)")
    ok(pk_new < pk_old - 60 * 2 ** 20,
       "new peak is >=60 MiB below the old path")
    ok(pk_new < out_mb * 2 ** 20 + 48 * 2 ** 20,
       "new peak ~= the output (+48 MiB budget)")

    # a shape where the OLD threaded split path fired (slab > 4M elems)
    ne2 = [1024, 4096, 8]
    vals2 = rng.standard_normal(tuple(ne2[::-1])).astype(np.float32)
    rt2 = types.SimpleNamespace(data=pack_q8_0(vals2),
                                tensor_type=GGMLQuantizationType.Q8_0,
                                shape=tuple(ne2))
    ref2a = old_convert_tensor(rt2, None, np.float32, workers=2)
    ref2b = old_convert_tensor(rt2, None, np.float32, workers=1)
    new2a = g2h.convert_tensor(rt2, None, np.float32, workers=2)
    ok(np.array_equal(new2a, ref2a), "Q8_0 old-split shape: workers=2 bit-exact")
    ok(np.array_equal(ref2a, ref2b), "sanity: old split == old single-shot")
    del ref2a, ref2b, new2a

    # 2D Q8_0 (lm_head-like, 32768 rows) - many slabs, serial + threaded
    ne3 = [1024, 32768]
    vals3 = rng.standard_normal(tuple(ne3[::-1])).astype(np.float32)
    rt3 = types.SimpleNamespace(data=pack_q8_0(vals3),
                                tensor_type=GGMLQuantizationType.Q8_0,
                                shape=tuple(ne3))
    ref3 = old_convert_tensor(rt3, None, np.float32, workers=1)
    ok(np.array_equal(g2h.convert_tensor(rt3, None, np.float32, workers=1), ref3),
       "Q8_0 (32768,1024) 2D: workers=1 bit-exact")
    ok(np.array_equal(g2h.convert_tensor(rt3, None, np.float32, workers=2), ref3),
       "Q8_0 (32768,1024) 2D: workers=2 bit-exact")
    del ref3

    # F16 stacked tensor (the .astype(np.float32) double-copy path)
    ne4 = [512, 4096, 16]
    src = (rng.standard_normal(tuple(ne4[::-1])) * 3).astype(np.float16)
    rt4 = types.SimpleNamespace(data=src, tensor_type=GGMLQuantizationType.F16,
                                shape=tuple(ne4))
    ref4 = old_convert_tensor(rt4, None, np.float32, workers=1)
    ok(np.array_equal(g2h.convert_tensor(rt4, None, np.float32, workers=2), ref4),
       "F16 (16,4096,512): bit-exact")
    del ref4

    # tiny 3D Q8_0 -> single-job branch
    ne5 = [1024, 8, 2]
    vals5 = rng.standard_normal(tuple(ne5[::-1])).astype(np.float32)
    rt5 = types.SimpleNamespace(data=pack_q8_0(vals5),
                                tensor_type=GGMLQuantizationType.Q8_0,
                                shape=tuple(ne5))
    ref5 = old_convert_tensor(rt5, None, np.float32, workers=1)
    ok(np.array_equal(g2h.convert_tensor(rt5, None, np.float32, workers=1), ref5),
       "tiny Q8_0 single-job: bit-exact")


# ------------------------------------------------------------------ section B

def make_pool(d, n_pairs=100, n_layers=2, inits=False):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "art_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_layers": n_layers}, f)
    for i in range(n_layers):
        torch.save({"X": torch.zeros(n_pairs, 8), "Y": torch.zeros(n_pairs, 8)},
                   os.path.join(d, f"pairs_blk{i}.pt"))
        if inits:
            torch.save({"dummy": True}, os.path.join(d, f"init_blk{i}.pt"))


def section_b():
    print("B. pool_is_complete keep-smaller semantics")
    tmp = tempfile.mkdtemp(prefix="pool_")
    try:
        d = os.path.join(tmp, "cache")
        make_pool(d, n_pairs=100)
        hp._POOL_NOTICE_SHOWN[0] = False
        ok(hp.pool_is_complete(d) == 2, "no min_pairs -> complete (2 blocks)")
        ok(hp.pool_is_complete(d, min_pairs=50) == 2, "have >= cap -> complete")

        out = capture()
        with redirect_stdout(out):
            r1 = hp.pool_is_complete(d, min_pairs=65536)
            r2 = hp.pool_is_complete(d, min_pairs=65536)
        ok(r1 == 0 and r2 == 0, "smaller pool, keep_smaller=False -> 0")
        ok(out.getvalue().count("recalibrating") == 1,
           "recalibrate notice printed ONCE for 2 calls")

        hp._POOL_NOTICE_SHOWN[0] = False
        out = capture()
        with redirect_stdout(out):
            r = hp.pool_is_complete(d, min_pairs=65536, keep_smaller=True,
                                    min_useful=50)
        ok(r == 2, "smaller pool >= floor + keep_smaller -> KEPT (2)")
        ok("keeping the cached pool" in out.getvalue(),
           "keep notice printed")

        # the user's exact numbers: 49152 cached, cap 65536, floor 32768
        d2 = os.path.join(tmp, "cache_big")
        make_pool(d2, n_pairs=49152)
        hp._POOL_NOTICE_SHOWN[0] = False
        out = capture()
        with redirect_stdout(out):
            r = hp.pool_is_complete(d2, min_pairs=65536, keep_smaller=True,
                                    min_useful=32768)
        ok(r == 2, "user case: 49152-pool vs cap 65536 -> KEPT")
        ok(hp.pool_is_complete(d2, min_pairs=65536, keep_smaller=True,
                               min_useful=60000) == 0,
           "pool below the floor -> recalibrate (0)")

        shutil.rmtree(d, ignore_errors=True)
        ok(hp.pool_is_complete(d, min_pairs=65536, keep_smaller=True,
                               min_useful=50) == 0,
           "keep_smaller without art_meta/pairs -> 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ section C

def fake_runner(src, keys=("k1", "k2"), always_fail=False):
    r = hs.BlockStreamRunner.__new__(hs.BlockStreamRunner)
    import threading as _th
    r._pf_lock = _th.Lock()
    r._pf_buf = {}
    r._pf_thread = None
    r._pf_dead = False
    r._pf_depth = 1
    r._pf_hint = False
    r._pf_queue = None
    r._loaded = None
    r._pinned = None
    r._expert_keys = {"blk.experts": [("gguf", k) for k in keys]}
    r._gguf = src
    r._stream_n = 0
    r._stream_bytes = 0
    r._load_time = 0.0
    r._slow_hint = False
    r._oom_hint = False
    r.progress = False
    r._io_workers = 1
    calls = {"n": 0}

    def get_weight(fn, key):
        calls["n"] += 1
        if calls["n"] == 1 or always_fail:
            raise MemoryError("Unable to allocate 128. MiB")
        return torch.zeros(4)

    r._get_weight = get_weight
    assigned = []
    r._assign = lambda key, t: assigned.append(key)
    return r, assigned, calls


def section_c():
    print("C. _load_block survives MemoryError (ram cache dropped)")
    with g2h._RAW_CACHE_LOCK:
        g2h._RAW_CACHE["junk"] = np.zeros(16, dtype=np.uint8)
    src = g2h.GgufHfSource.__new__(g2h.GgufHfSource)
    src.cache_ram = True
    src.io_workers = 1
    src.disk_bytes = lambda key: 16        # fake: packed bytes per tensor
    r, assigned, calls = fake_runner(src)
    out = capture()
    with redirect_stdout(out):
        r._load_block("blk.experts")
    ok(assigned == ["k1", "k2"], "block loaded after the OOM retry")
    ok(r._loaded == "blk.experts", "_loaded set")
    ok(src.cache_ram is False, "ram cache disabled on the source")
    ok(r._pf_dead and not r._pf_buf, "prefetch disabled + buffer cleared")
    ok(r._oom_hint, "oom hint set")
    ok("OOM: dropped the io-cache ram" in out.getvalue(), "notice printed")
    with g2h._RAW_CACHE_LOCK:
        ok(len(g2h._RAW_CACHE) == 0, "process-wide raw cache cleared")

    r2, _, _ = fake_runner(src, always_fail=True)
    try:
        r2._load_block("blk.experts")
        ok(False, "double OOM re-raises")
    except MemoryError:
        ok(True, "double OOM re-raises MemoryError")


# ------------------------------------------------------------------ section D

def section_d():
    print("D. GgufHfSource.drop_ram_cache")
    with g2h._RAW_CACHE_LOCK:
        g2h._RAW_CACHE["junk"] = np.zeros(8, dtype=np.uint8)
    src = g2h.GgufHfSource.__new__(g2h.GgufHfSource)
    src.cache_ram = True
    src.drop_ram_cache()
    ok(src.cache_ram is False, "cache_ram -> False")
    with g2h._RAW_CACHE_LOCK:
        ok(len(g2h._RAW_CACHE) == 0, "raw cache empty")
    n, b = g2h.raw_cache_stats()
    ok(n == 0 and b == 0, "raw_cache_stats agrees")


# ------------------------------------------------------------------ section E

def section_e():
    print("E. ensure_prereqs with a usable smaller pool")
    tmp = tempfile.mkdtemp(prefix="prereq_")
    try:
        pool_dir = os.path.join(tmp, "cache_tiny")
        lp_dir = os.path.join(pool_dir, "lp_base")
        fit_dir = os.path.join(pool_dir, "fit_r128")
        make_pool(pool_dir, n_pairs=49152, n_layers=2, inits=True)
        os.makedirs(lp_dir, exist_ok=True)

        class A:
            rank = 128
            pool_recalibrate = False

        plan = ["fit"]
        plan = hp.ensure_prereqs(plan, A(), pool_dir, lp_dir, fit_dir,
                                 os.path.join(tmp, "out"), min_pairs=65536)
        ok("calibrate" not in plan and "fit" in plan,
           "usable smaller pool: fit-only plan accepted (no re-collection)")

        A.pool_recalibrate = True
        hp._POOL_NOTICE_SHOWN[0] = False
        try:
            hp.ensure_prereqs(["fit"], A(), pool_dir, lp_dir, fit_dir,
                              os.path.join(tmp, "out"), min_pairs=65536)
            ok(False, "--pool-recalibrate forces re-collection (exit)")
        except SystemExit:
            ok(True, "--pool-recalibrate forces re-collection (exit)")

        # a pool below the usable floor still fails a fit-only plan
        A.pool_recalibrate = False
        make_pool(os.path.join(tmp, "c2"), n_pairs=100, n_layers=2, inits=True)
        try:
            hp.ensure_prereqs(["fit"], A(), os.path.join(tmp, "c2"), lp_dir,
                              fit_dir, os.path.join(tmp, "out"),
                              min_pairs=65536)
            ok(False, "pool below floor: fit-only plan exits with a hint")
        except SystemExit as e:
            ok("lower --per-layer-cap" in str(e),
               "exit hint mentions lowering the cap")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ section F

def section_f():
    print("F. swap-storm guard: ram-fit re-check + watchdog")
    v = hs.free_ram_gb()
    ok(v is None or (isinstance(v, float) and v > 0),
       "free_ram_gb: float>0 or None (portability)")

    ok(not hs.ram_cache_fits(3.2, 2.75),
       "3.2 GB free vs 2.75 GB GGUF -> ram NO (the user's box)")
    ok(hs.ram_cache_fits(6.0, 2.75), "6.0 GB free vs 2.75 GB GGUF -> ram ok")
    ok(hs.ram_cache_fits(None, 100.0),
       "unmeasurable RAM -> do not second-guess the user")
    ok(not hs.ram_cache_fits(0.5, 0.0),
       "0-size GGUF still needs the +1 GB headroom")

    ok(hp.refine_flush_at(3.2) == 1024, "low RAM -> capture flush at 1024")
    ok(hp.refine_flush_at(64.0) == 8192, "roomy RAM -> capture flush at 8192")
    ok(hp.refine_flush_at(None) == 8192,
       "unmeasurable RAM -> conservative 8192")

    gguf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tiny.gguf")
    if not os.path.isfile(gguf_path):
        print("    (tiny.gguf not found - runner-level checks skipped)")
        return

    real_free = hs.free_ram_gb

    def bare_runner():
        r = hs.BlockStreamRunner.__new__(hs.BlockStreamRunner)
        r.src = os.path.dirname(gguf_path)
        r._io_cache = "ram"
        r._io_workers = 1
        r.progress = False
        return r

    # the runner re-check: a tight box gets ram->disk at _open_gguf time
    hs.free_ram_gb = lambda: 0.4
    r = bare_runner()
    out = capture()
    with redirect_stdout(out):
        src_ = r._open_gguf(gguf_path)
    hs.free_ram_gb = real_free
    ok(src_.cache_ram is False, "tight box: GGUF opened with cache_ram=False")
    ok(r._io_cache == "disk", "runner io_cache state downgraded to disk")
    ok("DOES NOT FIT" in out.getvalue(), "downgrade notice printed")
    del src_

    # ...and the explicit override still works
    os.environ["MOE_FORCE_IO_RAM"] = "1"
    hs.free_ram_gb = lambda: 0.4
    r2 = bare_runner()
    with redirect_stdout(capture()):
        src2 = r2._open_gguf(gguf_path)
    hs.free_ram_gb = real_free
    del os.environ["MOE_FORCE_IO_RAM"]
    ok(src2.cache_ram is True, "MOE_FORCE_IO_RAM=1 keeps the ram cache")
    with redirect_stdout(capture()):
        src2.drop_ram_cache()
    del src2

    # the watchdog: free RAM below the floor -> cache dropped, notice printed
    src3 = g2h.GgufHfSource(gguf_path, io_workers=1, cache_ram=True)
    ok(src3.cache_ram is True, "sanity: source opened with cache_ram=True")
    r3 = hs.BlockStreamRunner.__new__(hs.BlockStreamRunner)
    r3._pf_lock = threading.Lock()
    r3._pf_buf = {}
    r3._pf_thread = None
    r3._pf_dead = False
    r3._pf_hint = False
    r3._oom_hint = False
    r3._gguf = src3
    r3._wd_stop = threading.Event()
    r3._wd_thread = None
    r3._wd_poll = 0.05
    hs.free_ram_gb = lambda: 0.3
    out = capture()
    with redirect_stdout(out):
        r3._start_ram_watchdog()
        if r3._wd_thread is not None:
            # the watchdog loop returns right AFTER printing the notice
            # (drop_ram_cache -> gc -> print) - joining avoids a race where
            # the check below runs before the print lands in the capture
            r3._wd_thread.join(timeout=5.0)
    hs.free_ram_gb = real_free
    r3._wd_stop.set()
    ok(src3.cache_ram is False, "watchdog dropped the ram cache on low RAM")
    ok("watchdog: only 0.3 GB RAM left" in out.getvalue(),
       "watchdog notice carries the measured amount")
    with g2h._RAW_CACHE_LOCK:
        ok(len(g2h._RAW_CACHE) == 0, "process-wide raw cache cleared")


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    print(f"\n{PASS} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", *FAIL, sep="\n  - ")
        sys.exit(1)
    print("ALL GREEN")
