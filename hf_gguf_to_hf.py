#!/usr/bin/env python3
"""GGUF (incl. Q4_K_M) -> a plain HF checkpoint OLMoE (safetensors fp16/bf16).

Why: transformers can load GGUF directly only for a few architectures
(qwen2_moe/qwen3_moe); olmoe is not among them. This converter has two modes:
  A. FULL dequant (convert): downloads a ready Q4 file from HF (default
     mradermacher/OLMoE-1B-7B-0924-GGUF, Q4_K_M ~4.4 GB) or takes a local
     .gguf; dequantizes tensors (Q4_K / Q6_K / Q8_0 / F16 / F32) via the gguf
     package; maps llama.cpp names -> HF (transposed, fused gate_up for
     experts); writes a standard checkpoint ~14 GB: config.json +
     model-XXXXX.safetensors + index; tokenizer: copy from the base repo
     (exact) first, fallback - from the GGUF; self-check against a skeleton
     on a meta-device.
  B. LIGHT catalog (prepare_light_dir) + GgufHfSource: only config.json +
     tokenizer + a marker with the GGUF path land on disk; weights are read
     straight from the GGUF with ON-THE-FLY dequant (bit-identical result).
     Saves ~14 GB of disk and SSD writes; costs ~5 s of CPU per block load
     (stages 3-4 run once, afterwards the pool cache is reused).

Usage:
  python3 hf_gguf_to_hf.py                                   # Q4_K_M from mradermacher
  python3 hf_gguf_to_hf.py --quant Q4_K_S                    # smaller/simpler
  python3 hf_gguf_to_hf.py --repo mradermacher/OLMoE-1B-7B-0924-i1-GGUF  # imatrix version
  python3 hf_gguf_to_hf.py --gguf /path/model.Q4_K_M.gguf   # already downloaded file
Output: <out>/  - a checkpoint folder ready for AutoModelForCausalLM.from_pretrained(out)
The full dequant is NOT required by the pipeline: hf_pipeline.py by default
builds a light catalog (prepare_light_dir) and reads weights from the GGUF
block by block.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import types

import hf_env  # noqa: F401  - HF cache inside the project; before transformers/hub
import numpy as np
from gguf import GGUFReader

REPO_DEFAULT = "mradermacher/OLMoE-1B-7B-0924-GGUF"
BASE_REPO = "allenai/OLMoE-1B-7B-0924"
SHARD_BYTES = 3_800_000_000  # ~3.8 GB per safetensors shard
SUPPORTED_ARCH = ("olmoe", "hy_v3")   # hy_v3: NanoColibri and relatives (HunYuan-v3)
# auto-quant preference (best compromise first; the base comparison model IS
# the quantized file itself, so a lower quant = easier to match but a lower
# quality floor)
QUANT_PREFERENCE = ("Q4_K_M", "Q5_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q6_K",
                    "Q8_0", "BF16", "F16")


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ download

def gguf_in_cache(repo, gguf_file=None, quant="Q4_K_M"):
    """Find an already-downloaded .gguf of `repo` in the local HF cache
    (offline; --skip download / reuse-only mode). -> path or None."""
    import glob
    from huggingface_hub.constants import HF_HUB_CACHE
    root = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"),
                        "snapshots")
    cands = sorted(glob.glob(os.path.join(root, "*", "*.gguf")))
    if gguf_file:
        cands = [c for c in cands if os.path.basename(c) == gguf_file]
    elif str(quant).lower() != "auto":
        want = f".{quant}.gguf".lower()
        cands = [c for c in cands if c.lower().endswith(want)]
    else:  # auto: best available by the same preference order as online mode
        cands.sort(key=lambda c: next(
            (i for i, q in enumerate(QUANT_PREFERENCE)
             if c.lower().endswith(f".{q.lower()}.gguf")), len(QUANT_PREFERENCE)))
    return cands[0] if cands else None


def resolve_gguf(args, local_only=False):
    """-> (path to .gguf, human-readable source name).
    local_only: never hit the network - only an already-downloaded file
    (local --gguf path or the HF cache) is accepted."""
    if args.gguf:
        p = os.path.abspath(args.gguf)
        if not os.path.isfile(p):
            sys.exit(f"file not found: {p}")
        return p, os.path.basename(p)
    repo = args.repo
    if local_only:
        p = gguf_in_cache(repo, gguf_file=args.gguf_file, quant=args.quant)
        if p is None:
            sys.exit(f"--skip download: no .gguf of {repo} in the local HF cache\n"
                     "  -> drop 'download' from --skip, or pass --gguf /path/file.gguf")
        log(f"local cache hit: {os.path.basename(p)} (no network)")
        return p, f"{repo}/{os.path.basename(p)} (local cache)"
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(repo) if f.endswith(".gguf")]
    if not files:
        sys.exit(f"no .gguf files in {repo}")
    target = None
    if args.gguf_file:
        target = next((f for f in files if f == args.gguf_file), None)
    else:
        if str(args.quant).lower() == "auto":
            # pick the best available quant by preference order
            def quant_of(f):
                low = f.lower()
                for q in QUANT_PREFERENCE:
                    if low.endswith(f".{q.lower()}.gguf"):
                        return q
                return low.rsplit(".", 2)[-2].upper()
            available = {quant_of(f) for f in files}
            pick = next((q for q in QUANT_PREFERENCE if q in available), None)
            if pick is None:
                sys.exit(f"auto-quant: nothing recognizable in {repo}; "
                         f"available: {', '.join(sorted(available))} "
                         f"(use --gguf-file for an exact name)")
            log(f"quant auto -> {pick} (available: {', '.join(sorted(available))})")
            args.quant = pick
        want = f".{args.quant}.gguf"
        cand = [f for f in files if f.lower().endswith(want.lower())]
        if not cand:
            quants = sorted({f.rsplit(".", 2)[-2] for f in files})
            sys.exit(f"no {args.quant} in {repo}; available: {', '.join(quants)}")
        target = sorted(cand)[0]
    log(f"downloading {repo}/{target} (resumes on a re-run)...")
    p = hf_hub_download(repo, target)
    return p, f"{repo}/{target}"


# ------------------------------------------------------------------ header

def read_meta(rd):
    """Key GGUF metadata as a flat dict."""
    val = {}

    def field_value(f):
        t = f.types[0]
        if t == 8:
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        if t == 9:
            et, items = f.types[1], f.data
            if et == 8:
                return [bytes(f.parts[i]).decode("utf-8", "replace") for i in items]
            cast = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
                    5: np.int32, 6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64,
                    12: np.float64}[et]
            return np.concatenate([np.frombuffer(f.parts[i], dtype=cast) for i in items])
        cast = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
                5: np.int32, 6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64,
                12: np.float64}[t]
        return np.frombuffer(f.parts[f.data[-1]], dtype=cast)[0]

    for k, f in rd.fields.items():
        try:
            val[k] = field_value(f)
        except Exception:
            pass
    return val


def auto_base_repo(gguf_path_or_meta):
    """Base HF repo from GGUF metadata (general.base_model.0.repo_url /
    general.source.url) - exact config+tokenizer without manual flags."""
    if isinstance(gguf_path_or_meta, dict):
        meta = gguf_path_or_meta
    else:
        try:
            meta = read_meta(GGUFReader(gguf_path_or_meta, mode="r"))
        except Exception:
            return None
    for k in ("general.base_model.0.repo_url", "general.source.url"):
        u = str(meta.get(k, "") or "")
        if "huggingface.co/" in u:
            rid = u.split("huggingface.co/", 1)[1].strip("/")
            if rid and "/" in rid and not rid.endswith(".gguf"):
                return rid
    return None


def geom_from_meta(val):
    arch = str(val.get("general.architecture", "olmoe"))
    if arch not in SUPPORTED_ARCH:
        sys.exit(f"the converter supports {', '.join(SUPPORTED_ARCH)}, "
                 f"file arch={arch}")
    a = lambda k: val.get(f"{arch}.{k}")
    g = dict(
        arch=arch,
        n_layers=int(a("block_count")), d=int(a("embedding_length")),
        n_heads=int(a("attention.head_count")),
        n_kv=int(a("attention.head_count_kv") or 0),
        n_exp=int(a("expert_count") or 0),
        top_k=int(a("expert_used_count") or 0),
        vocab=int(val.get("tokenizer.ggml.vocab_size",
                          0) or len(val.get("tokenizer.ggml.tokens", []) or []) or 0),
        ctx=int(a("context_length")), rope=float(a("rope.freq_base") or 10000.0),
        eps=float(a("attention.layer_norm_rms_epsilon") or 1e-5),
    )
    # d_ff: for olmoe - feed_forward_length (experts); for hy_v3 - the expert
    # dff (expert_feed_forward_length), while feed_forward_length is the dense
    # layer 0
    if arch == "hy_v3":
        g["dff"] = int(a("expert_feed_forward_length") or 0)
        g["dff_dense"] = int(a("feed_forward_length") or 0)
        g["dff_shexp"] = int(a("expert_shared_feed_forward_length") or 0)
        g["hd"] = int(a("attention.key_length") or 0) or g["d"] // max(g["n_heads"], 1)
        g["router_scale"] = float(a("expert_weights_scale") or 1.0)
    else:
        g["dff"] = int(a("feed_forward_length"))
        g["dff_dense"] = g["dff"]
        g["dff_shexp"] = 0
        g["hd"] = g["d"] // max(g["n_heads"], 1)
    return g


# ------------------------------------------------------------------ mapping

def gguf_to_hf_name(name, g):
    """llama.cpp name -> (HF name, expected shape). None - skip.
    The map is shared for olmoe/hy_v3: extra entries simply never occur in
    the file."""
    if name == "token_embd.weight":
        return "model.embed_tokens.weight", (g["vocab"], g["d"])
    if name == "output.weight":
        return "lm_head.weight", (g["vocab"], g["d"])
    if name == "output_norm.weight":
        return "model.norm.weight", (g["d"],)
    if not name.startswith("blk."):
        return None, None
    rest = name.split(".", 2)[2].removesuffix(".weight")
    n = int(name.split(".")[1])
    p = f"model.layers.{n}"
    d, hd = g["d"], g.get("hd") or g["d"]
    kvd = (g.get("n_kv") or (d // hd)) * hd          # k/v projection outputs (GQA)
    dffd = g.get("dff_dense") or g["dff"]            # dense FFN (hy_v3 layer 0)
    dffs = g.get("dff_shexp") or 0                    # shared experts (hy_v3)
    # q/k-norm: hy_v3 - on head_dim; olmoe - on hidden (bases disagree)
    qkn = (hd,) if g["arch"] == "hy_v3" else (d,)
    m = {
        "attn_norm": (f"{p}.input_layernorm.weight", (d,)),
        "ffn_norm": (f"{p}.post_attention_layernorm.weight", (d,)),
        "attn_q_norm": (f"{p}.self_attn.q_norm.weight", qkn),
        "attn_k_norm": (f"{p}.self_attn.k_norm.weight", qkn),
        "attn_q": (f"{p}.self_attn.q_proj.weight", (d, d)),
        "attn_k": (f"{p}.self_attn.k_proj.weight", (kvd, d)),
        "attn_v": (f"{p}.self_attn.v_proj.weight", (kvd, d)),
        "attn_output": (f"{p}.self_attn.o_proj.weight", (d, d)),
        "ffn_gate_inp": (f"{p}.mlp.gate.weight", (g["n_exp"], d)),
        "ffn_gate_exps": (f"{p}.mlp.experts.gate_up_proj",
                          (g["n_exp"], g["dff"], d)),   # gate before fusing
        "ffn_up_exps": ("__up__", (g["n_exp"], g["dff"], d)),
        "ffn_down_exps": (f"{p}.mlp.experts.down_proj",
                          (g["n_exp"], d, g["dff"])),
        "exp_probs_b": (f"{p}.mlp.e_score_correction_bias", (g["n_exp"],)),
        "ffn_gate": (f"{p}.mlp.gate_proj.weight", (dffd, d)),
        "ffn_up": (f"{p}.mlp.up_proj.weight", (dffd, d)),
        "ffn_down": (f"{p}.mlp.down_proj.weight", (d, dffd)),
    }
    if dffs:
        m.update(
            ffn_gate_shexp=(f"{p}.mlp.shared_experts.gate_proj.weight", (dffs, d)),
            ffn_up_shexp=(f"{p}.mlp.shared_experts.up_proj.weight", (dffs, d)),
            ffn_down_shexp=(f"{p}.mlp.shared_experts.down_proj.weight", (d, dffs)),
        )
    return m.get(rest, (None, None))


def convert_tensor(rt, g, out_dtype, workers=1):
    """ReaderTensor -> fp32 numpy/HF layout.

    In gguf-py rt.shape holds GGUF ne [ne0, ne1, ...] (ne0 fastest), while
    rt.data / the dequantize result are in numpy order (reversed), i.e.
    already (out, in) = HF layout. After dequantizing block types we reshape
    to reversed(rt.shape) when needed.

    workers > 1: for expert tensors the SLOWEST ne axis (the expert index) is
    contiguous in the byte stream, so the packed data splits into per-expert
    slabs that dequantize in a thread pool (numpy releases the GIL - real
    parallelism). Falls back to single-thread when the split is unsafe."""
    from gguf import GGMLQuantizationType, dequantize
    from gguf.constants import GGML_QUANT_SIZES
    data = rt.data
    q = rt.tensor_type
    target = tuple(int(x) for x in np.asarray(rt.shape)[::-1])   # numpy-native
    if q in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        arr = np.ascontiguousarray(data, dtype=np.float32)
    else:
        ne = [int(x) for x in rt.shape]
        n_last = ne[-1] if ne else 1
        blk_elems = GGML_QUANT_SIZES[q][0]
        slab_elems = int(np.prod(ne[:-1])) if len(ne) > 1 else int(ne[0])
        can_split = (workers > 1 and n_last >= 2
                     and data.ndim == 1 and data.shape[0] % n_last == 0
                     and slab_elems >= block_elems > 0
                     and slab_elems % block_elems == 0 and slab_elems > 4_000_000)
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


def fit_layout(arr, expected):
    """Bring the GGUF layout to the expected HF shape (by transposing)."""
    if arr.shape == expected:
        return arr
    if arr.shape == tuple(reversed(expected)) and arr.ndim == 2:
        return arr.T
    if arr.ndim == 3 and arr.shape == (expected[2], expected[1], expected[0]):
        return arr.transpose(2, 1, 0)          # (d, x, N) -> (N, x, d)
    if arr.ndim == 3 and arr.shape == (expected[1], expected[0], expected[2]) \
            and expected[0] != expected[1]:
        return arr.transpose(1, 0, 2)
    if arr.ndim == 2 and arr.shape[1] == expected[1] and arr.shape[0] != expected[0] \
            and expected[0] == arr.shape[1]:
        return arr.T
    raise RuntimeError(f"shape {arr.shape} does not fit the expected {expected}")


# ------------------------------------------------------------------ shard writer

class ShardWriter:
    """Writes model-0000X-of-0000N.safetensors + model.safetensors.index.json."""

    def __init__(self, out_dir, plan_bytes, shard_bytes=SHARD_BYTES):
        from safetensors.torch import save_file
        self._save = save_file
        self.out, self.shard_bytes = out_dir, shard_bytes
        self.n_shards = max(1, -(-plan_bytes // shard_bytes))
        self.weight_map, self.total = {}, 0
        self.buf, self.cur, self.cur_bytes = {}, 1, 0

    def add(self, name, tensor):
        import torch
        t = torch.from_numpy(tensor)
        self.buf[name] = t
        self.cur_bytes += t.numel() * t.element_size()
        self.weight_map[name] = f"model-{self.cur:05d}-of-{self.n_shards:05d}.safetensors"
        self.total += t.numel() * t.element_size()
        if self.cur_bytes >= self.shard_bytes:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        path = os.path.join(self.out,
                            f"model-{self.cur:05d}-of-{self.n_shards:05d}.safetensors")
        # shards with fewer than planned are fine - the index is authoritative
        self._save(self.buf, path)
        log(f"  shard {self.cur}/{self.n_shards}: {os.path.basename(path)} "
            f"({self.cur_bytes / 1e9:.2f} GB)")
        self.buf, self.cur_bytes = {}, 0
        self.cur += 1

    def finalize(self):
        self.flush()
        # the actual shard count may differ from the plan - rename
        used = self.cur - 1
        if used != self.n_shards:
            for name in list(self.weight_map):
                old = self.weight_map[name]
                no = int(old.split("-")[1])
                self.weight_map[name] = f"model-{no:05d}-of-{used:05d}.safetensors"
            for f in os.listdir(self.out):
                if f.startswith("model-") and f.endswith(".safetensors"):
                    no = int(f.split("-")[1])
                    new = f"model-{no:05d}-of-{used:05d}.safetensors"
                    if f != new:
                        os.rename(os.path.join(self.out, f), os.path.join(self.out, new))
        idx = dict(metadata=dict(total_size=self.total), weight_map=self.weight_map)
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(idx, f, indent=2)
        return used


# ------------------------------------------------------------------ config

def _load_base_json(base_repo, name):
    """JSON from the base repo: local folder or HF (raw/main)."""
    if not base_repo:
        return None
    lp = os.path.join(base_repo, name)
    if os.path.isdir(base_repo):
        if not os.path.isfile(lp):
            return None
        with open(lp, encoding="utf-8") as f:
            return json.load(f)
    try:
        import requests
        r = requests.get(f"https://huggingface.co/{base_repo}/raw/main/{name}",
                         timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        log(f"{name} from {base_repo} unavailable ({e})")
        return None


def build_config(g, meta, base_repo, tied=None, moe_blocks=None):
    """Source config: geometry from GGUF is authoritative; base_repo adds the
    rest. tied: None -> as in base_repo; True/False - known tied-head fact
    (output.weight missing from the GGUF -> True)."""
    if g["arch"] == "hy_v3":
        return _build_config_hyv3(g, meta, base_repo, tied=tied,
                                  moe_blocks=moe_blocks)
    return _build_config_olmoe(g, meta, base_repo, tied=tied)


def _special_tokens(meta, g, params):
    for mk, pk in (("tokenizer.ggml.eos_token_id", "eos_token_id"),
                   ("tokenizer.ggml.bos_token_id", "bos_token_id"),
                   ("tokenizer.ggml.padding_token_id", "pad_token_id")):
        v = meta.get(mk)
        if v is not None and int(v) < g["vocab"]:
            params[pk] = int(v)


def _build_config_olmoe(g, meta, base_repo, tied=None):
    from transformers import OlmoeConfig
    params = dict(vocab_size=g["vocab"], hidden_size=g["d"],
                  intermediate_size=g["dff"], num_hidden_layers=g["n_layers"],
                  num_attention_heads=g["n_heads"], num_key_value_heads=g["n_kv"],
                  num_experts=g["n_exp"], num_experts_per_tok=g["top_k"],
                  rms_norm_eps=g["eps"], rope_theta=g["rope"],
                  max_position_embeddings=g["ctx"], hidden_act="silu",
                  norm_topk_prob=False, attention_bias=False,
                  tie_word_embeddings=False, model_type="olmoe")
    if tied is not None:
        params["tie_word_embeddings"] = bool(tied)
    _special_tokens(meta, g, params)
    src = _load_base_json(base_repo, "config.json")
    if src:
        # geometry from the GGUF itself is authoritative; from the base repo we
        # take only what the GGUF lacks (dropout, special tokens, aux coefs...)
        for k, v in src.items():
            if k not in params and k != "model_type":
                params[k] = v
        log("config: geometry from GGUF + extra keys from " + str(base_repo))
    else:
        log("config: built from GGUF metadata")
    params.pop("model_type", None)
    cfg = OlmoeConfig(**params)
    cfg.architectures = ["OlmoeForCausalLM"]
    return cfg


def _build_config_hyv3(g, meta, base_repo, tied=None, moe_blocks=None):
    from transformers import HYV3Config
    src = _load_base_json(base_repo, "config.json") or {}
    params = dict(src)
    params.update(
        model_type="hy_v3", vocab_size=g["vocab"], hidden_size=g["d"],
        num_hidden_layers=g["n_layers"], num_attention_heads=g["n_heads"],
        num_key_value_heads=g["n_kv"] or g["n_heads"], head_dim=g["hd"],
        intermediate_size=g.get("dff_dense") or g["dff"],
        moe_intermediate_size=g["dff"],
        num_experts=g["n_exp"], num_experts_per_tok=g["top_k"],
        rms_norm_eps=g["eps"], max_position_embeddings=g["ctx"],
        hidden_act="silu",
        router_scaling_factor=g.get("router_scale", 2.826),
    )
    params["rope_parameters"] = dict(src.get("rope_parameters") or {},
                                     rope_type="default", rope_theta=g["rope"])
    if g.get("dff_shexp"):
        params["num_shared_experts"] = max(1, g["dff_shexp"] // g["dff"])
        params["enable_moe_fp32_combine"] = bool(
            src.get("enable_moe_fp32_combine", True))
    if tied is not None:
        params["tie_word_embeddings"] = bool(tied)
    if moe_blocks is not None and "mlp_layer_types" not in src:
        params["mlp_layer_types"] = [
            "sparse" if i in moe_blocks else "dense" for i in range(g["n_layers"])]
    _special_tokens(meta, g, params)
    # drop unknown HYV3Config keys from a foreign config.json (@strict)
    known = set(HYV3Config.__dict__) | {"architectures", "transformers_version",
                                        "dtype", "torch_dtype", "_attn_implementation"}
    params = {k: v for k, v in params.items() if k in known}
    cfg = HYV3Config(**params)
    cfg.architectures = ["HYV3ForCausalLM"]
    log(f"hy_v3 config: layers {g['n_layers']} (MoE {len(moe_blocks) if moe_blocks else '?'})"
        f", d {g['d']}, expert dff {g['dff']}, shexp {g.get('dff_shexp', 0)}"
        f", experts {g['n_exp']} (top-{g['top_k']}), scale {g.get('router_scale')}")
    return cfg


# ------------------------------------------------------------------ tokenizer

def build_tokenizer_from_gguf(meta, out_dir):
    """Fallback: BPE tokenizer straight from the GGUF (tokens+merges), skipping
    the transformers architecture table. Works for the gpt2 family (incl. OLMo).
    Pre-tokenizer is byte-level (OLMo regex differs slightly - acceptable for
    a fallback; the main path is the tokenizer from the base repo)."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens is None or len(tokens) == 0:
        raise RuntimeError("no tokenizer.ggml.tokens in the GGUF")
    merges = meta.get("tokenizer.ggml.merges", []) or []
    vocab = {t: i for i, t in enumerate(tokens)}
    if len(vocab) != len(tokens):
        raise RuntimeError("duplicate tokens in the GGUF")
    pairs = []
    for m in merges:
        try:
            l, r = m.rsplit(" ", 1)
        except ValueError:
            continue
        if l in vocab and r in vocab and l + r in vocab:   # guard vs broken merges
            pairs.append((l, r))
    bpe = models.BPE(vocab=vocab, merges=pairs)
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=True)
    tok.decoder = decoders.ByteLevel()
    kwargs = {}
    eos = meta.get("tokenizer.ggml.eos_token_id")
    pad = meta.get("tokenizer.ggml.padding_token_id")
    if eos is not None:
        kwargs["eos_token"] = tokens[int(eos)]
    if pad is not None:
        kwargs["pad_token"] = tokens[int(pad)]
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, **kwargs)
    fast.save_pretrained(out_dir)
    return "gguf_bpe"


def save_tokenizer(out_dir, gguf_path, base_repo):
    from transformers import AutoTokenizer
    gguf_vocab = None
    try:
        m = read_meta(GGUFReader(gguf_path))
        gguf_vocab = int(m.get("tokenizer.ggml.vocab_size", 0)) or None
        if gguf_vocab is None:
            toks = m.get("tokenizer.ggml.tokens")
            gguf_vocab = len(toks) if toks is not None else None
    except Exception:  # noqa: BLE001
        pass
    if base_repo:
        try:
            tok = AutoTokenizer.from_pretrained(base_repo)
            if gguf_vocab and len(tok) > gguf_vocab:
                # such tokenizer ids would not fit the model embeddings -
                # small/custom GGUFs carry their own vocab, base_repo is foreign
                log(f"tokenizer from {base_repo} ({len(tok)}) is larger than the "
                    f"GGUF vocab ({gguf_vocab}) - taking the GGUF tokenizer, "
                    f"otherwise IndexError")
            else:
                tok.save_pretrained(out_dir)
                log(f"tokenizer: from {base_repo} (exact)")
                return "base_repo"
        except Exception as e:  # noqa: BLE001
            log(f"tokenizer from {base_repo} failed ({e}); fallback - GGUF")
    try:
        src = build_tokenizer_from_gguf(read_meta(GGUFReader(gguf_path)), out_dir)
        log("tokenizer: built from GGUF metadata (byte-level BPE)")
        return src
    except Exception as e:  # noqa: BLE001
        log(f"GGUF tokenizer failed too: {e} - checkpoint without tokenizer")
        return "none"


# ------------------------------------------------------------------ self-check

def self_check(cfg, sd_shapes):
    """Model skeleton on a meta-device: verify weight names/shapes."""
    try:
        import torch
        mt = str(getattr(cfg, "model_type", "olmoe"))
        if mt == "hy_v3":
            from transformers import HYV3ForCausalLM as Cls
        else:
            from transformers import OlmoeForCausalLM as Cls
        with torch.device("meta"):
            model = Cls._from_config(cfg)
        msd = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        extra = set(sd_shapes) - set(msd)
        missing = set(msd) - set(sd_shapes)
        bad = [k for k in set(sd_shapes) & set(msd) if sd_shapes[k] != msd[k]]
        del model
        if extra or missing or bad:
            log(f"SELF-CHECK: mismatches extra={len(extra)} missing={len(missing)} "
                f"shapes={len(bad)}")
            for k in list(extra)[:3]:
                log(f"  extra {k}")
            for k in list(missing)[:3]:
                log(f"  missing {k}")
            for k in bad[:3]:
                log(f"  shape {k}: file {sd_shapes[k]} vs model {msd[k]}")
            return False
        log(f"SELF-CHECK: all {len(msd)} weights matched by name and shape")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"self-check skipped: {type(e).__name__}: {e}")
        return True


# ------------------------------------------------------------------ GGUF source

# Process-wide cache of RAW (packed) GGUF tensor bytes, shared by every
# GgufHfSource instance created in this process (stream runner, refit runner,
# artifact writer...). Key: (gguf abspath, tensor name) -> np.ndarray copy.
# Effect: after the first pass over the layers every subsequent block load is
# served from RAM - no repeated disk reads on the calibration/refit passes.
# RAM cost: ~= the size of the packed GGUF (Q4/Q8), lazily, tensor by tensor.
_RAW_CACHE = {}
_RAW_CACHE_LOCK = threading.Lock()
_RAW_CACHE_HITS = [0]


def raw_cache_stats():
    """(n_tensors, bytes) currently held in the process-wide raw cache."""
    with _RAW_CACHE_LOCK:
        return len(_RAW_CACHE), sum(t.nbytes for t in _RAW_CACHE.values())


def raw_cache_clear():
    with _RAW_CACHE_LOCK:
        _RAW_CACHE.clear()


class GgufHfSource:
    """GGUF as a weight source with HF names: ON-THE-FLY dequant, no checkpoint.

    get(hf_name) returns a tensor bit-identical to what convert() writes: the
    same convert_tensor -> fit_layout -> fp16 cast. Casting is elementwise, so
    "cast before concat" gives the same result as "concat before cast" in
    convert(). RAM: one tensor at a time (for OLMoE at most ~0.8 GB per fused
    gate_up). Disk: only the GGUF itself (~4.4 GB instead of 4.4 + 13.8 GB of
    dequant).
    """

    def __init__(self, gguf_path, dtype="float16", io_workers=1, cache_ram=False):
        self.gguf_path = os.path.abspath(gguf_path)
        self.io_workers = max(1, int(io_workers))
        self.cache_ram = bool(cache_ram)   # keep raw packed tensors in RAM
        self.rd = GGUFReader(self.gguf_path, mode="r")
        self.meta = read_meta(self.rd)
        self.g = geom_from_meta(self.meta)
        self.np_dtype = np.float16 if str(dtype).endswith("16") else np.float32
        self.by = {t.name: t for t in self.rd.tensors}
        self.hf2gguf, self.expected = {}, {}
        for name in self.by:
            hf, exp = gguf_to_hf_name(name, self.g)
            if hf is None or hf == "__up__":
                continue
            self.hf2gguf[hf] = name
            self.expected[hf] = exp
        if not self.hf2gguf:
            raise RuntimeError(f"no known {', '.join(SUPPORTED_ARCH)} tensors "
                               f"in {self.gguf_path}")
        tot = sum(int(t.data.nbytes) for t in self.rd.tensors)
        print(f"GGUF source: {len(self.by)} tensors, "
              f"{tot / 1e9:.2f} GB of packed data", flush=True)

    def keys(self):
        return sorted(self.hf2gguf)

    def __contains__(self, hf):
        return hf in self.hf2gguf

    def _rt(self, name):
        """ReaderTensor for `name`; with cache_ram=True the packed bytes are
        copied into RAM on first touch and served from the process-wide cache
        afterwards (thread-safe: the prefetch worker calls this too)."""
        rt = self.by[name]
        if not self.cache_ram:
            return rt
        key = (self.gguf_path, name)
        with _RAW_CACHE_LOCK:
            buf = _RAW_CACHE.get(key)
            if buf is None:
                buf = np.array(rt.data)       # copy: memmap slice -> RAM
                _RAW_CACHE[key] = buf
            else:
                _RAW_CACHE_HITS[0] += 1
        if buf is rt.data:
            return rt
        return types.SimpleNamespace(data=buf, tensor_type=rt.tensor_type,
                                     shape=rt.shape)

    def _convert(self, name, expected):
        """GGUF tensor -> numpy in HF layout and the output dtype."""
        arr = convert_tensor(self._rt(name), self.g, self.np_dtype,
                             workers=self.io_workers)  # fp32
        arr = fit_layout(arr, expected)
        return np.ascontiguousarray(arr.astype(self.np_dtype))

    def get(self, hf):
        """HF name -> torch tensor (fp16 by default) on CPU."""
        import torch
        name = self.hf2gguf[hf]
        out = self._convert(name, self.expected[hf])
        if name.endswith("ffn_gate_exps.weight"):     # fused gate_up from two
            up_name = name.replace("ffn_gate_exps", "ffn_up_exps")
            up = self._convert(up_name, self.expected[hf])
            out = np.concatenate([out, up], axis=1)
        return torch.from_numpy(out)

    def disk_bytes(self, hf):
        """How many GGUF bytes are actually read from disk for this tensor."""
        name = self.hf2gguf[hf]
        n = int(self.by[name].data.nbytes)
        if name.endswith("ffn_gate_exps.weight"):
            n += int(self.by[name.replace("ffn_gate_exps", "ffn_up_exps")].data.nbytes)
        return n

    def cache_stats(self):
        """(n_tensors, GB, hits) of raw GGUF bytes held in the process-wide
        RAM cache (shared by every GgufHfSource in this process)."""
        n, b = raw_cache_stats()
        return n, b / 1e9, _RAW_CACHE_HITS[0]


def has_full_weights(d):
    """Does the folder hold a full dequant checkpoint (safetensors)?"""
    return os.path.isdir(d) and any(
        f.endswith(".safetensors") for f in os.listdir(d))


def _tensor_facts(rd):
    """(tied, moe_blocks) from the tensor list: tied = no output.weight;
    moe_blocks = blocks with ffn_*_exps."""
    tied, moe = True, set()
    for t in rd.tensors:
        if t.name == "output.weight":
            tied = False
        if t.name.endswith(("ffn_gate_exps.weight", "ffn_down_exps.weight")):
            moe.add(int(t.name.split(".")[1]))
    return tied, moe


def prepare_light_dir(gguf_path, out_dir, base_repo):
    """Light source folder WITHOUT weights: config.json + tokenizer + a marker
    _gguf_source.json (path to the GGUF). GgufHfSource reads weights from it -
    the full 14 GB dequant checkpoint is never created."""
    os.makedirs(out_dir, exist_ok=True)
    rd = GGUFReader(gguf_path, mode="r")
    meta = read_meta(rd)
    g = geom_from_meta(meta)
    tied, moe_blocks = _tensor_facts(rd)
    cfg = build_config(g, meta, base_repo, tied=tied, moe_blocks=moe_blocks)
    cfg.save_pretrained(out_dir)
    tok = save_tokenizer(out_dir, gguf_path, base_repo)
    with open(os.path.join(out_dir, "_gguf_source.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(gguf=os.path.abspath(gguf_path), tokenizer=tok,
                       dtype="float16"), f, ensure_ascii=False, indent=2)
    return out_dir


# ------------------------------------------------------------------ main

def convert(gguf_path, out_dir, dtype, base_repo, max_shard_mb=None):
    from gguf import GGUFReader
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "_converted_ok.json")
    if os.path.exists(done):
        log(f"{out_dir} already converted ({done}) - skipping")
        return out_dir

    rd = GGUFReader(gguf_path, mode="r")
    meta = read_meta(rd)
    g = geom_from_meta(meta)
    log(f"{g['arch']}: layers {g['n_layers']}, d {g['d']}, dff {g['dff']}, "
        f"experts {g['n_exp']} (top-{g['top_k']}), vocab {g['vocab']}")

    # byte plan: elements x dtype; fused gate+up counts two GGUF tensors
    planned = 0
    items = []
    by = {t.name: t for t in rd.tensors}
    for name, rt in by.items():
        hf, expected = gguf_to_hf_name(name, g)
        if hf is None:
            log(f"  (skipping {name})")
            continue
        if hf == "__up__":                       # up goes into the fused gate_up
            continue
        n_el = int(np.prod([int(x) for x in rt.shape]))
        # fused gate_up holds both gate and up (up is not written separately)
        mult = 2 if name.endswith("ffn_gate_exps.weight") else 1
        planned += n_el * mult
        items.append((name, rt, hf, expected))

    dtype_np = np.float16 if str(dtype).endswith("16") else np.float32
    writer = ShardWriter(out_dir, planned * np.dtype(dtype_np).itemsize,
                         shard_bytes=(max_shard_mb or SHARD_BYTES // (1024 * 1024))
                         * 1024 * 1024)
    sd_shapes = {}
    fused_done = set()
    for i, (name, rt, hf, expected) in enumerate(items):
        if hf in fused_done:
            continue
        arr = convert_tensor(rt, g, dtype_np)
        arr = fit_layout(arr, expected)
        if name.endswith("ffn_gate_exps.weight"):   # fused = [gate; up] by outputs
            up_rt = by[name.replace("ffn_gate_exps", "ffn_up_exps")]
            up = fit_layout(convert_tensor(up_rt, g, dtype_np), expected)
            arr = np.concatenate([arr, up], axis=1)   # (N, dff, d) -> (N, 2dff, d)
            fused_done.add(hf)
        out = arr.astype(dtype_np)
        writer.add(hf, out)
        sd_shapes[hf] = out.shape
        del arr, out
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(items)} tensors, {time.time() - t0:.0f} s")
    writer.finalize()
    log(f"weights written: {writer.total / 1e9:.2f} GB in {time.time() - t0:.0f} s")

    cfg = build_config(g, meta, base_repo,
                       tied=_tensor_facts(rd)[0], moe_blocks=_tensor_facts(rd)[1])
    cfg.save_pretrained(out_dir)
    tok_src = save_tokenizer(out_dir, gguf_path, base_repo)
    ok = dict(gguf=gguf_path, bytes=os.path.getsize(gguf_path),
              config="base" if base_repo else "gguf_meta", tokenizer=tok_src,
              total_bytes=writer.total, n_weights=len(sd_shapes))
    with open(done, "w", encoding="utf-8") as f:
        json.dump(ok, f, ensure_ascii=False, indent=2)
    self_check(cfg, {k: tuple(v) for k, v in sd_shapes.items()})
    log(f"DONE: {out_dir} ({time.time() - t0:.0f} s)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_DEFAULT, help="HF repo with GGUF")
    ap.add_argument("--quant", default="Q4_K_M",
                    help="Q4_K_M | Q4_K_S | Q3_K_M | Q8_0 | ... or 'auto' to pick "
                         "the best available (if --gguf-file is not set)")
    ap.add_argument("--gguf-file", default=None, help="exact file name in the repo")
    ap.add_argument("--gguf", default=None, help="local .gguf (skip downloading)")
    ap.add_argument("--out", default=None, help="folder for the HF checkpoint")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--base-repo", default=BASE_REPO,
                    help="where to take the exact config/tokenizer from")
    args = ap.parse_args()
    gguf_path, src_name = resolve_gguf(args)
    sz = os.path.getsize(gguf_path) / 1e9
    log(f"source: {src_name} ({sz:.2f} GB)")
    out = args.out or os.path.splitext(gguf_path)[0] + "-hf"
    if os.path.isdir(out) and os.listdir(out) and not os.path.exists(
            os.path.join(out, "_converted_ok.json")):
        shutil.rmtree(out)
        log(f"cleaned the unfinished {out}")
    convert(gguf_path, out, args.dtype, args.base_repo)


if __name__ == "__main__":
    main()
