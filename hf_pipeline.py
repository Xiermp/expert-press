#!/usr/bin/env python3
"""Pipeline for a REAL MoE model from HuggingFace -> "field engine".

The model is downloaded ALREADY QUANTIZED - by default the ready Q4_K_M GGUF
from mradermacher/OLMoE-1B-7B-0924-GGUF (~4.4 GB, works without a GPU).
What one run does:
  0. bootstrap   - installs missing pip packages
  1. download    - Q4_K_M.gguf from the HF hub (resumes on a re-run);
                   NO dequant checkpoint is created: a light catalog (config +
                   tokenizer) is built and weights are read straight from the
                   GGUF block by block (on-the-fly dequant, saves ~14 GB of
                   disk; --full-dequant brings back the old path with a ~14 GB
                   checkpoint)
  2. texts       - calibration/eval text with no overlap (leak fix)
  3. base eval   - perplexity/log-prob cache (ON DISK)/generation;
                   THE FULL MODEL NEVER LOADS: STREAMING - backbone in RAM
                   (~1.5 GB), each block's experts are read from disk exactly
                   for their layer's pass (hf_stream.py, with background
                   prefetch of the next block)
  4. calibrate   - (MoE input -> output) pairs via hooks + block centroids,
                   also streaming. The pair pool on disk IS the calibration
                   artifact: the fit samples vectors independently, text
                   order does not matter - a re-run SKIPS STAGES 3-4
  5. fit         - fits the field r=32 on pairs from disk: the model is NOT
                   in RAM, peak RAM ~1-2 GB; blocks are independent and can
                   be fitted in parallel (--fit-workers); several optimizer
                   methods available (--fit-method, --fit-preset)
  6. save        - the artifact is assembled STREAMINGLY from disk (backbone
                   tensors copied one by one, experts skipped, field - from
                   fit files); the full model never loads
  7. verify      - the artifact loads as a NORMAL model (~1-2 GB):
                   ppl/KL vs the quantized base + demo generation
  8. report      - report into the artifact and results/; after success the
                   dequant checkpoint (if any) deletes itself: what remains
                   is GGUF + pool cache + fit + artifact (~8.6 GB vs ~22 GB)

Local run:
  python3 hf_pipeline.py                       # OLMoE Q4_K_M GGUF, rank 32
  python3 hf_pipeline.py --auto                # zero-config: auto quant + balanced preset
  python3 hf_pipeline.py --low-mem             # memory-frugal metrics mode
  python3 hf_pipeline.py --threads 4           # keep some CPU cores free
  python3 hf_pipeline.py --cleanup             # erase the GGUF ~4.4 GB after success
  python3 hf_pipeline.py --full-dequant        # full dequant checkpoint ~14 GB after all
  python3 hf_pipeline.py --gguf-quant Q4_K_S   # smaller file
  python3 hf_pipeline.py --gguf-quant auto     # pick the best available quant
  python3 hf_pipeline.py --gguf /path/model.Q4_K_M.gguf  # an already downloaded GGUF
  python3 hf_pipeline.py --model allenai/OLMoE-1B-7B-0924  # bf16 source (~15 GB)
  python3 hf_pipeline.py --model RichardErkhov/allenai_-_OLMoE-1B-7B-0924-4bits  # needs CUDA

Fit tuning:
  --fit-preset fast|balanced|quality           # steps/batch/lr/method bundles
  --fit-method adam|adamw|adam-cosine|rmsprop  # optimizer type
  --fit-workers 2                              # parallel fit of independent blocks
  --prefetch 0                                 # disable background block prefetch
                                               # (saves ~1 block of RAM)
  --io-cache ram                               # keep packed GGUF tensors in RAM
                                               # (first pass fills the cache,
                                               # later passes read no disk)

Stage toggles (the auto-pipeline is a chain of 9 stages; any can be switched
on/off - the plan is printed before the run):
  python3 hf_pipeline.py --list-stages             # the stage table
  python3 hf_pipeline.py --stages fit,save,verify  # run ONLY these (from cache)
  python3 hf_pipeline.py --skip base,verify,report # run all EXCEPT these
  python3 hf_pipeline.py --skip download           # reuse-only: never touch the network
  python3 hf_pipeline.py --stages fit --rank 64    # a new rank from the cached pool
  python3 hf_pipeline.py --stages refine --refine-rounds 1  # refine implies rounds>=1
  --gen-tokens 0        # no demo generations (saves a streaming pass)
  --no-cache-verify     # skip the 2-chunk log-prob cache self-check
Cheap missing stages (texts/download) are auto-added with a notice; expensive
ones (base/calibrate/fit/verify) fail fast with a hint instead of surprising
you with a multi-hour pass. A re-run also auto-skips whatever is cached.

Windows: the same commands via double click - step1_compress.bat / step2_chat.bat.
"""
import argparse
import gc
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

import hf_env  # noqa: F401  - HF cache inside the project; BEFORE transformers/hub

BASE = os.path.dirname(os.path.abspath(__file__))
DL = os.environ.get("MOE_OUT_DIR", os.path.join(BASE, "results"))
os.makedirs(DL, exist_ok=True)
CORPUS_URLS = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
)
REQUIRED = [("transformers", "transformers>=5"), ("huggingface_hub", "huggingface_hub"),
            ("safetensors", "safetensors"), ("accelerate", "accelerate")]
# fit presets: bundles of steps/batch/lr/method (explicit flags always win)
FIT_PRESETS = {
    "fast":     dict(fit_steps=120, fit_bs=4096, fit_lr=3e-3, fit_method="adam-cosine"),
    "balanced": dict(fit_steps=300, fit_bs=4096, fit_lr=2e-3, fit_method="adam-cosine"),
    "quality":  dict(fit_steps=600, fit_bs=8192, fit_lr=2e-3, fit_method="adam-cosine"),
}
FIT_LEGACY = dict(fit_steps=300, fit_bs=4096, fit_lr=2e-3, fit_method="adam")
T = {}

# Pipeline stages: togglable via --stages / --skip (names in run order).
STAGE_ORDER = ["download", "texts", "base", "calibrate", "fit", "refine",
               "save", "verify", "report"]
STAGE_DESCR = {
    "download":  "download   1   source: GGUF resolve/download + light catalog (config+tokenizer)",
    "texts":     "texts      2   calibration/eval text split + tokenization",
    "base":      "base       3   base ppl/log-prob cache/demo generation (STREAMING)",
    "calibrate": "calibrate  4   pair pool + block centroids/geometry (STREAMING)",
    "fit":       "fit        5   field fit per block (no model in RAM)",
    "refine":    "refine     5b  self-distillation refit rounds (--refine-rounds; default off)",
    "save":      "save       6   assemble the artifact STREAMINGLY",
    "verify":    "verify     7   reload artifact: KL/ppl vs base + demo generation",
    "report":    "report     8   write reports (artifact README + results/)",
}


def banner(msg):
    print("\n" + "=" * 66 + f"\n== {msg}\n" + "=" * 66, flush=True)


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def ensure_package(pkg, pip_name=None):
    if not have(pkg):
        print(f"installing package: {pip_name or pkg}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               pip_name or pkg])


def stage_bootstrap(args):
    need = [pip for m, pip in REQUIRED if not have(m)]
    if args.calib_dataset and not have("datasets"):
        need.append("datasets")
    if need:
        print(f"installing packages: {', '.join(need)}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *need])
    import transformers, torch  # noqa: E401
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
          f"cuda: {torch.cuda.is_available()}", flush=True)


def download_text_url(url, dst):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def resolve_texts(args):
    """Calibration and eval text: a file / wikitext / local corpus.txt."""
    calib = evalt = None
    if args.calib_file:
        calib = open(args.calib_file, encoding="utf-8", errors="ignore").read()
    elif args.calib_dataset:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", args.calib_dataset,
                          split="train" if "raw" in args.calib_dataset else "train")
        calib = "\n".join(t for t in ds["text"] if t.strip())
    if args.eval_file:
        evalt = open(args.eval_file, encoding="utf-8", errors="ignore").read()
    if calib is None:  # fallback: local corpus from the mini PoC
        for name in ("corpus.txt", "corpus_raw.txt", "corpus_ru.txt"):
            p = os.path.join(BASE, name)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                calib = open(p, encoding="utf-8", errors="ignore").read()
                print(f"calibration: local {name}", flush=True)
                break
        if calib is None:
            dst = os.path.join(BASE, "corpus_raw.txt")
            for url in CORPUS_URLS:
                try:
                    download_text_url(url, dst)
                    calib = open(dst, encoding="utf-8", errors="ignore").read()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  {url}: {e}", flush=True)
            if calib is None:
                sys.exit("No calibration text: pass --calib-file or --calib-dataset")
    if evalt is None:
        # leak fix: eval = the tail of the text, calibration - everything
        # BEFORE it (no overlap). Previously eval was cut OUT of the
        # calibration text, and collect_pairs sampled windows from the same
        # piece the KL was measured on.
        k = max(1, int(len(calib) * 0.9))
        calib, evalt = calib[:k], calib[k:]
        print("split: calibration 90% | eval 10% - no overlap (leak fix)",
              flush=True)
    calib = calib[:args.text_cap]
    evalt = evalt[:args.text_cap]
    return calib, evalt


def load_source_model(args, src, dtype, device, quantized):
    from transformers import AutoModelForCausalLM
    if quantized:
        return AutoModelForCausalLM.from_pretrained(
            src, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    m = AutoModelForCausalLM.from_pretrained(src, dtype=dtype,
                                             low_cpu_mem_usage=True)
    return m.to(device).eval()


def release_model(model, device):
    """Unload the model from RAM (the main saving: the fit runs without it)."""
    del model
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def dir_size_gb(p):
    if os.path.isfile(p):
        return os.path.getsize(p) / 1e9
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return tot / 1e9


def do_cleanup(targets):
    banner("Cleaning intermediate files (--cleanup)")
    freed = 0.0
    for p in targets:
        if not p or not os.path.exists(p):
            continue
        sz = dir_size_gb(p)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            freed += sz
            print(f"  removed: {p} ({sz:.2f} GB)", flush=True)
        except OSError as e:
            print(f"  could not remove {p}: {e}", flush=True)
    print(f"freed {freed:.2f} GB", flush=True)


def cache_is_complete(cache_dir, lp_dir, min_pairs=0, pairs_only=False):
    """Pair pool + base log-probs already on disk? -> block count (0 = no).
    Pairs/centroids/log-probs depend on neither rank nor text order - such a
    cache survives a --rank/--fit-* change, and --cleanup does NOT touch it.
    min_pairs: when the cached pool holds FEWER pairs per block than the
    requested --per-layer-cap, the cache is treated as incomplete (the caller
    recalibrates with the bigger cap).
    pairs_only: check ONLY the pair pool + centroids (art_meta/pairs/init),
    without the base log-prob cache - enough for fit/save/refine, which never
    read the log-probs; the full check is needed by base/verify."""
    import torch as _t
    am = os.path.join(cache_dir, "art_meta.json")
    if not os.path.isfile(am):
        return 0
    try:
        with open(am, encoding="utf-8") as f:
            n = int(json.load(f)["n_layers"])
    except Exception:
        return 0
    if not all(os.path.isfile(os.path.join(cache_dir, f"pairs_blk{i}.pt"))
               for i in range(n)):
        return 0
    if not all(os.path.isfile(os.path.join(cache_dir, f"init_blk{i}.pt"))
               for i in range(n)):
        return 0
    if not pairs_only:
        tp = os.path.join(cache_dir, "eval_tokens.pt")
        if not os.path.isfile(tp):
            return 0
        try:
            d = _t.load(tp, map_location="cpu")
        except Exception:
            return 0
        if not all(os.path.isfile(os.path.join(lp_dir, f"lp_{i:03d}.pt"))
                   for i in range(len(d["X"]))):
            return 0
    if min_pairs:
        try:
            x0 = _t.load(os.path.join(cache_dir, "pairs_blk0.pt"),
                         map_location="cpu")["X"]
            if x0.shape[0] < min_pairs:
                print(f"cached pool holds {x0.shape[0]} pairs/block < requested "
                      f"cap {min_pairs} - recalibrating with the bigger pool",
                      flush=True)
                return 0
        except Exception:
            return 0
    return n


def resolve_fit_args(args):
    """Preset + explicit flags -> effective fit params (legacy values when
    nothing is set). Returns (fit_params dict, preset_name)."""
    fit = dict(FIT_LEGACY)
    preset = args.fit_preset
    if preset is None and args.auto:
        preset = "balanced"
    if preset:
        fit.update(FIT_PRESETS[preset])
    for k in ("fit_steps", "fit_bs", "fit_lr", "fit_method"):
        v = getattr(args, k)
        if v is not None:
            fit[k] = v
    return fit, preset


def resolve_plan(args):
    """--stages / --skip / --skip-reload-check -> a validated stage list
    (STAGE_ORDER subset). Also handles --list-stages and the refine defaults."""
    if args.list_stages:
        print("\nPipeline stages (run order):")
        for s in STAGE_ORDER:
            print(f"  {STAGE_DESCR[s]}")
        print("\ntoggles:")
        print("  --stages fit,save,verify   run ONLY these stages")
        print("  --skip base,report         run all EXCEPT these")
        print("  --skip download            reuse-only: no network/downloads")
        print("  (--skip-reload-check == --skip verify; a re-run also auto-skips")
        print("   whatever is already cached: pool, log-probs, fits)")
        sys.exit(0)
    if args.stages and args.skip:
        sys.exit("use either --stages or --skip, not both")
    if args.stages:
        wanted = {s.strip() for s in args.stages.split(",") if s.strip()}
    else:
        skipped = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
        wanted = set(STAGE_ORDER) - skipped
    bad = wanted - set(STAGE_ORDER)
    if bad:
        sys.exit(f"unknown stage(s): {', '.join(sorted(bad))}\n"
                 f"known stages: {', '.join(STAGE_ORDER)}  (--list-stages for details)")
    if not wanted:
        sys.exit("empty stage plan (--list-stages shows the stage table)")
    if args.skip_reload_check and "verify" in wanted:
        wanted.discard("verify")
        print("--skip-reload-check: 'verify' removed from the plan (== --skip verify)",
              flush=True)
    if "refine" in wanted and args.refine_rounds == 0 and not args.stages:
        wanted.discard("refine")   # refine is opt-in: the default plan leaves 5b off
    if "refine" in wanted and args.refine_rounds == 0 and args.stages:
        # refine is opt-in: an explicit --stages refine implies at least 1 round;
        # the default plan keeps rounds=0 (stage 5b off, as before)
        args.refine_rounds = 1
        print("'refine' requested via --stages without --refine-rounds: "
              "defaulting to 1 round", flush=True)
    if "refine" not in wanted and args.refine_rounds > 0:
        print(f"note: --refine-rounds {args.refine_rounds} ignored - 'refine' is "
              f"not in the plan", flush=True)
    return [s for s in STAGE_ORDER if s in wanted]


def ensure_prereqs(plan, args, pool_dir, lp_dir, fit_dir, out_dir, min_pairs):
    """Auto-add cheap missing stages with a notice; hard-fail with a hint when
    an expensive prerequisite is absent from disk AND from the plan. Returns
    the final ordered plan."""
    full = cache_is_complete(pool_dir, lp_dir, min_pairs=min_pairs)
    pairs = full or cache_is_complete(pool_dir, lp_dir, min_pairs=min_pairs,
                                      pairs_only=True)
    auto = []

    def add(s):
        if s not in plan:
            plan.append(s)
            auto.append(s)

    if plan != ["report"]:          # anything but a pure report rerun needs src
        add("download")
    if "refine" in plan:
        add("texts")
    if "calibrate" in plan and not pairs:
        add("texts")
    if "fit" in plan and not pairs and "calibrate" not in plan:
        hint = ""
        try:
            import torch as _t
            x0 = _t.load(os.path.join(pool_dir, "pairs_blk0.pt"),
                         map_location="cpu")["X"]
            hint = f"\n  -> or lower --per-layer-cap (the cached pool holds " \
                   f"{x0.shape[0]} pairs/block)"
        except Exception:  # noqa: BLE001
            pass
        sys.exit(f"stage 'fit': the pair pool is missing/incomplete in {pool_dir}\n"
                 "  -> run the full pipeline once (keep base+calibrate in the "
                 f"plan), or drop 'fit' from --stages{hint}")
    if ("save" in plan or "refine" in plan) and "fit" not in plan:
        n = pairs or 0
        if not n or not all(os.path.isfile(os.path.join(fit_dir, f"fit_blk{i}.pt"))
                            for i in range(n)):
            what = "stage 'refine': the fits" if "refine" in plan else \
                   "stage 'save': the fits"
            sys.exit(f"{what} are missing in {fit_dir}\n"
                     "  -> keep 'fit' in the plan, or finish a full run first")
    if "verify" in plan:
        if not full and "base" not in plan:
            sys.exit("stage 'verify': the base log-prob cache is missing/"
                     f"incomplete in {lp_dir}\n"
                     "  -> keep 'base' in the plan (one streaming pass builds "
                     "the cache), or --skip verify")
        if "save" not in plan and not os.path.isfile(os.path.join(out_dir,
                                                                 "config.json")):
            sys.exit(f"stage 'verify': no artifact to verify at {out_dir}\n"
                     "  -> keep 'save' in the plan, or pass --out <artifact dir>")
    if auto:
        plan = [s for s in STAGE_ORDER if s in plan]
        print(f"plan adjusted, cheap stages auto-added: +{', '.join(auto)}",
              flush=True)
    plan = [s for s in STAGE_ORDER if s in plan]
    print(f"\nPLAN: {' -> '.join(plan)}", flush=True)
    skipped = [s for s in STAGE_ORDER if s not in plan]
    if skipped:
        print(f"skipped: {', '.join(skipped)}", flush=True)
    T["plan"], T["plan_skipped"] = plan, skipped
    return plan


def print_profile(args, fit, preset):
    """One place to see everything the run will actually use."""
    rows = [
        ("model", args.model),
        ("quant", "auto" if str(args.gguf_quant).lower() == "auto" else args.gguf_quant),
        ("rank", args.rank),
        ("fit", f"method={fit['fit_method']} steps={fit['fit_steps']} "
                f"bs={fit['fit_bs']} lr={fit['fit_lr']}"
                + (f" preset={preset}" if preset else "")),
        ("fit_workers", args.fit_workers),
        ("fit_jitter", args.fit_jitter),
        ("fit_early_stop", args.fit_early_stop),
        ("refine_rounds", args.refine_rounds),
        ("io_threads", args.io_threads),
        ("prefetch", args.prefetch),
        ("cpu cores", os.cpu_count()),
        ("pairs/layer cap", args.per_layer_cap),
        ("eval chunks / kl chunks", f"{args.eval_chunks} / {args.kl_chunks}"),
    ]
    print("profile: " + " | ".join(f"{k}={v}" for k, v in rows), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    default="mradermacher/OLMoE-1B-7B-0924-GGUF",
                    help="HF id (default Q4_K_M GGUF mradermacher; "
                         "for bf16: allenai/OLMoE-1B-7B-0924)")
    ap.add_argument("--gguf-quant", default="Q4_K_M",
                    help="which quant to download from a GGUF repo: "
                         "Q4_K_M | Q4_K_S | Q3_K_M | Q8_0 | ... | auto "
                         "(pick the best available)")
    ap.add_argument("--gguf-file", default=None, help="exact .gguf name in the repo")
    ap.add_argument("--gguf", default=None,
                    help="local .gguf file (skip downloading)")
    ap.add_argument("--gguf-out", default=None,
                    help="folder for the light catalog / dequant checkpoint")
    ap.add_argument("--gguf-base-repo", default=None,
                    help="where to take the exact config/tokenizer from (default: "
                         "auto-detected from GGUF metadata; empty - build from GGUF)")
    ap.add_argument("--local-path", default=None, help="path to an already downloaded model")
    ap.add_argument("--auto", action="store_true",
                    help="zero-config mode: --gguf-quant auto + the balanced fit "
                         "preset (explicit flags still win)")
    ap.add_argument("--rank", type=int, default=32, help="field rank (default 32)")
    ap.add_argument("--out", default=None, help="artifact folder")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--calib-dataset", default=None,
                    help="e.g. wikitext-2-raw-v1 (requires datasets)")
    ap.add_argument("--text-cap", type=int, default=3_000_000, help="text characters")
    ap.add_argument("--calib-windows", type=int, default=3)
    ap.add_argument("--calib-bsz", type=int, default=8)
    ap.add_argument("--calib-ctx", type=int, default=512)
    ap.add_argument("--per-layer-cap", type=int, default=8192)
    ap.add_argument("--fit-steps", type=int, default=None,
                    help="fit steps per block (default: 300, or the preset value)")
    ap.add_argument("--fit-bs", type=int, default=None,
                    help="fit batch size (default: 4096, or the preset value)")
    ap.add_argument("--fit-lr", type=float, default=None,
                    help="fit learning rate (default: 2e-3, or the preset value)")
    ap.add_argument("--fit-method", default=None,
                    help="optimizer: adam | adamw | adam-cosine | rmsprop "
                         "(default: adam, or the preset value)")
    ap.add_argument("--fit-jitter", type=float, default=0.0,
                    help="Gaussian noise on fit inputs, per-dim std units "
                         "(variance reduction for a SMALL calibration pool: "
                         "0.2-0.3 at <8 pairs/dim; at 16+ pairs/dim prefer 0 - "
                         "the bias outweighs the win, and systematic deploy "
                         "shift is what --refine-rounds is for; 0 = off)")
    ap.add_argument("--fit-preset", default=None,
                    help="fit bundle: fast (~2.5x quicker) | balanced (default "
                         "quality/speed trade) | quality (slowest, lowest mse)")
    ap.add_argument("--fit-workers", type=int, default=1,
                    help="parallel fit workers for independent blocks (2-4 on a "
                         "multi-core CPU; 1 = sequential)")
    ap.add_argument("--fit-early-stop", type=int, default=0,
                    help="stop each block's fit after 2 consecutive flat mse "
                         "checkpoints (every N steps, e.g. 50; 0 = off). Saves "
                         "time on plateauing blocks")
    ap.add_argument("--refine-rounds", type=int, default=0,
                    help="self-distillation rounds after the first fit (try 1-2): "
                         "a streaming pass where the FIELD model feeds its own "
                         "outputs forward while the original GGUF experts provide "
                         "targets, then a warm-started refit. Fixes the compounding "
                         "error the first fit cannot see (it is calibrated on the "
                         "BASE model's activations)")
    ap.add_argument("--io-threads", type=int, default=1,
                    help="threads for GGUF dequant of expert tensors (2-4 speeds "
                         "up stage 3-4/6 block reads on a multi-core CPU; "
                         "1 = single-threaded)")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="background prefetch of the next expert block while the "
                         "current layer computes (default 1; 0 = off, saves ~1 "
                         "block of RAM)")
    ap.add_argument("--io-cache", choices=["disk", "ram"], default="disk",
                    help="ram: copy the packed GGUF tensors into RAM on first "
                         "touch (~= packed file size); later passes (pool "
                         "collection, refit, artifact write) read nothing from "
                         "disk - big win on Colab/Drive or HDD")
    ap.add_argument("--eval-chunks", type=int, default=50)
    ap.add_argument("--kl-chunks", type=int, default=16)
    ap.add_argument("--eval-ctx", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--gen-rep-pen", type=float, default=1.15,
                    help="repetition penalty for the report's demo generations "
                         "(applied to BOTH base and field; compressed models "
                         "loop under plain greedy; 1.0 = off)")
    ap.add_argument("--max-shard", default="4GB")
    ap.add_argument("--save-backbone", default="keep", choices=["keep", "bf16"],
                    help="keep - backbone as in the source (Q4 source -> Q4 artifact); "
                         "bf16 - dequant the backbone (CPU inference of the artifact)")
    ap.add_argument("--threads", type=int, default=None,
                    help="limit torch CPU threads (default: all cores; set e.g. 4 "
                         "to keep the machine responsive)")
    ap.add_argument("--low-mem", action="store_true",
                    help="memory-frugal metrics: smaller pair/chunk caps "
                         "(lower RAM, nearly the same metric quality)")
    ap.add_argument("--cleanup", action="store_true",
                    help="after success erase the GGUF too (~4.4 GB); the pool "
                         "cache and artifact stay (the pool serves new ranks "
                         "without recalibration)")
    ap.add_argument("--full-dequant", action="store_true",
                    help="build a full dequant checkpoint from the GGUF (~14 GB "
                         "on disk; block reads without dequant - faster on a fast "
                         "SSD). By default weights are read straight from the GGUF "
                         "block by block: -14 GB of disk, but ~5 s CPU per block load")
    ap.add_argument("--keep-dequant", action="store_true",
                    help="keep the dequant checkpoint after success (by default it "
                         "deletes itself: the artifact is assembled, and can be "
                         "rebuilt from the GGUF without re-downloading)")
    ap.add_argument("--skip-fit-guard", action="store_true",
                    help="do not abort the run if the field fit failed to beat the "
                         "centroid baseline (degradation guard)")
    ap.add_argument("--skip-reload-check", action="store_true",
                    help="skip the artifact check (same as --skip verify)")
    ap.add_argument("--stages", default=None, metavar="A,B,...",
                    help="run ONLY these stages, e.g. fit,save,verify (names: "
                         "--list-stages). Mutually exclusive with --skip")
    ap.add_argument("--skip", default=None, metavar="A,B,...",
                    help="run all stages EXCEPT these, e.g. base,verify,report")
    ap.add_argument("--list-stages", action="store_true",
                    help="print the stage table and exit")
    ap.add_argument("--no-cache-verify", action="store_true",
                    help="skip the 2-chunk log-prob cache self-check (stage 3)")
    ap.add_argument("--smoke", action="store_true",
                    help="mini wiring run: short fit/eval")
    args = ap.parse_args()
    plan = resolve_plan(args)          # may exit (--list-stages / bad names)
    if args.smoke:
        args.calib_windows, args.calib_bsz, args.calib_ctx = 2, 2, 128
        args.per_layer_cap, args.fit_steps, args.fit_bs = 2048, 60, 2048
        args.eval_chunks, args.kl_chunks, args.eval_ctx = 6, 4, 128
        args.gen_tokens = 24
    if args.low_mem:
        args.per_layer_cap = min(args.per_layer_cap, 4096)
        args.kl_chunks = min(args.kl_chunks, 8)
        args.eval_ctx = min(args.eval_ctx, 256)
        args.calib_windows = min(args.calib_windows, 2)
        args.calib_bsz = min(args.calib_bsz, 4)
        print("--low-mem mode: pair/chunk caps reduced (lower RAM, nearly the "
              "same metrics)", flush=True)
    fit, fit_preset = resolve_fit_args(args)
    args.fit_steps, args.fit_bs = fit["fit_steps"], fit["fit_bs"]
    args.fit_lr, args.fit_method = fit["fit_lr"], fit["fit_method"]
    print_profile(args, fit, fit_preset)

    t0 = time.time()
    banner("STAGE 0 - bootstrap (dependencies)")
    stage_bootstrap(args)
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.activations import ACT2FN
    sys.path.insert(0, BASE)
    from hf_field_transform import (base_metrics_from_cache, block_geometry,
                                    block_router_bias, block_shared_weights,
                                    collect_pairs, eval_logits_cache_disk,
                                    eval_vs_cache_disk, expert_means,
                                    field_accounting, find_moe_blocks,
                                    fit_field_module, generate_text,
                                    load_pairs_block, make_batches,
                                    router_weight, save_pairs_block,
                                    write_field_artifact,
                                    FieldSparseMoe)
    from hf_stream import BlockStreamRunner
    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"CPU threads limited to: {args.threads}", flush=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    is_gguf = (args.gguf is not None or (args.model or "").lower().endswith(".gguf")
               or "-gguf" in (args.model or "").lower())
    if args.dtype == "auto":
        # bf16 even on CPU: fp32 would double RAM (~28 GB for OLMoE) for nothing
        dtype = torch.bfloat16
    else:
        dtype = getattr(torch, args.dtype)
    T["device"], T["dtype"] = str(device), str(dtype)

    def base_label():
        sq = str(T.get("source_quant", "none"))
        if sq.startswith("gguf"):
            return f"quantized model {sq.split(':')[1]} (GGUF)"
        if sq in ("none", "None"):
            return "original model"
        return f"original model ({sq})"

    tag = re.sub(r"[^A-Za-z0-9_.-]", "_",
                 os.path.basename((args.gguf or args.model).rstrip("/")))[:60]
    pool_dir = os.path.join(DL, f"cache_{tag}")            # calibration pool shared by all ranks
    fit_dir = os.path.join(pool_dir, f"fit_r{args.rank}")  # fit of a specific rank
    lp_dir = os.path.join(pool_dir, "lp_base")
    os.makedirs(lp_dir, exist_ok=True)
    out_dir = args.out or os.path.join(DL, f"field_{tag}_r{args.rank}")
    T["cache_dir"], T["fit_dir"] = pool_dir, fit_dir
    mp = 0 if args.smoke else args.per_layer_cap
    plan = ensure_prereqs(plan, args, pool_dir, lp_dir, fit_dir, out_dir, mp)

    reuse_only = "download" not in plan
    banner(f"STAGE 1 - {'reuse-only (--skip download: no network)' if reuse_only else 'download'}: {args.model}")
    quantized = False
    conv_dir = gguf_path = None
    light_gguf = None          # weights are read straight from the GGUF (on-the-fly dequant)
    deconv_full_dir = None     # full dequant checkpoint, if present/created
    cleanup_targets = []
    if is_gguf:
        ensure_package("gguf")
        import hf_gguf_to_hf as g2h
        from types import SimpleNamespace
        gguf_path, src_name = g2h.resolve_gguf(SimpleNamespace(
            gguf=args.gguf, repo=args.model, quant=args.gguf_quant,
            gguf_file=args.gguf_file), local_only=reuse_only)
        base_repo = args.gguf_base_repo or g2h.auto_base_repo(gguf_path) \
            or "allenai/OLMoE-1B-7B-0924"
        if not args.gguf_base_repo:
            print(f"base repo (config+tokenizer) from GGUF metadata: "
                  f"{base_repo}", flush=True)
        T["source_quant"] = f"gguf:{args.gguf_quant}"
        print(f"source: {src_name} "
              f"({os.path.getsize(gguf_path) / 1e9:.2f} GB)", flush=True)
        conv_dir = args.gguf_out or os.path.join(
            DL, "gguf_hf", os.path.basename(gguf_path).removesuffix(".gguf") + "-hf")
        if args.full_dequant:
            if g2h.has_full_weights(conv_dir):
                banner("STAGE 1b - --full-dequant: dequant checkpoint already "
                       "on disk, using it")
            elif reuse_only:
                sys.exit("--skip download + --full-dequant: no dequant checkpoint "
                         f"at {conv_dir}\n  -> drop 'download' from --skip once to "
                         "build it, or drop --full-dequant")
            else:
                banner("STAGE 1b - dequant GGUF -> full HF checkpoint "
                       "(~14 GB on disk, --full-dequant)")
            src = g2h.convert(gguf_path, conv_dir, dtype="float16",
                              base_repo=base_repo)
            deconv_full_dir = conv_dir
        else:
            if g2h.has_full_weights(conv_dir) and args.gguf_out is None:
                sz_old = dir_size_gb(conv_dir)
                shutil.rmtree(conv_dir, ignore_errors=True)
                print(f"removed the previous run's dequant checkpoint "
                      f"({sz_old:.1f} GB): it is not needed - weights are read "
                      f"straight from the quantized GGUF (bring the old path "
                      f"back: --full-dequant)", flush=True)
            banner("STAGE 1b - light catalog (config+tokenizer): weights are "
                   "read straight from the GGUF, no dequant checkpoint is created")
            marker = os.path.join(conv_dir, "_gguf_source.json")
            ok_marker = False
            if os.path.isfile(marker):
                try:
                    with open(marker, encoding="utf-8") as f:
                        m = json.load(f)
                    ok_marker = (m.get("gguf") == os.path.abspath(gguf_path)
                                 and os.path.isfile(m.get("gguf", "")))
                except Exception:  # noqa: BLE001
                    ok_marker = False
            if ok_marker:
                src = conv_dir
                print("light catalog already in place for this GGUF - skipping",
                      flush=True)
            elif reuse_only:
                sys.exit("--skip download: the light catalog for this GGUF is not "
                         f"built yet ({conv_dir})\n  -> drop 'download' from --skip "
                         "once (it only reads config/tokenizer + writes a marker)")
            else:
                src = g2h.prepare_light_dir(gguf_path, conv_dir,
                                            base_repo=base_repo)
            light_gguf = gguf_path
            print("disk: only the GGUF itself (-~14 GB vs a full dequant); the "
                  "price is ~5 s of CPU dequant per block load, stages 3-4 run "
                  "once (the pool is cached)", flush=True)
        # clean only what we created ourselves: a local --gguf and explicit --gguf-out are untouched
        if args.gguf is None:
            cleanup_targets.append(gguf_path)
        if args.gguf_out is None:
            cleanup_targets.append(conv_dir)
        T["_gguf_path"] = gguf_path
    else:
        try:
            src = args.local_path or snapshot_download(
                args.model, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                                            "*.jinja"],
                local_files_only=reuse_only)
        except Exception as e:  # noqa: BLE001
            if reuse_only:
                sys.exit(f"--skip download: {args.model} is not in the local HF "
                         f"cache ({type(e).__name__})\n  -> drop 'download' from "
                         "--skip once, or pass --local-path")
            raise
        with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
            src_quant = (json.load(f).get("quantization_config") or {}).get("quant_method")
        quantized = src_quant in ("bitsandbytes", "bnb-4bit")
        T["source_quant"] = str(src_quant or "none")
        print(f"source: {src}\nsource quantization: {src_quant or 'none'}", flush=True)
        if src_quant == "gptq":
            sys.exit("GPTQ source is not supported: take a bnb-4bit repo, "
                     "e.g. RichardErkhov/allenai_-_OLMoE-1B-7B-0924-4bits")
        if quantized:
            ensure_package("bitsandbytes")
            if not torch.cuda.is_available():
                sys.exit("a Q4 model (bitsandbytes) requires a CUDA GPU. On CPU "
                         "use GGUF: --model mradermacher/OLMoE-1B-7B-0924-GGUF")
    tokenizer = AutoTokenizer.from_pretrained(src)
    if quantized:
        src_gb = sum(os.path.getsize(os.path.join(src, f))
                     for f in os.listdir(src) if f.endswith(".safetensors")) / 1e9
        base_total_b = int(src_gb * 1e9)

    if "texts" in plan:
        banner("STAGE 2 - texts: calibration/eval")
        calib_text, eval_text = resolve_texts(args)
        calib_ids = torch.tensor(tokenizer(calib_text)["input_ids"])
        eval_ids = torch.tensor(tokenizer(eval_text)["input_ids"])
        print(f"tokens: calib {len(calib_ids)}, eval {len(eval_ids)}", flush=True)
    else:
        banner("STAGE 2 - texts: skipped (--skip texts; token ids come from the "
               "run cache)")
        calib_ids, eval_ids = None, None

    # ========== PHASE A - base + calibration: STREAMING (model not in RAM) =====
    full_cache = cache_is_complete(pool_dir, lp_dir, min_pairs=mp)
    pairs_cache = full_cache or cache_is_complete(pool_dir, lp_dir, min_pairs=mp,
                                                  pairs_only=True)
    base_pass = "base" in plan and not full_cache
    calib_pass = "calibrate" in plan and not pairs_cache
    stream = None
    base_gen = None
    if not base_pass and not calib_pass:
        banner("PHASE A - no streaming pass needed: everything from the run cache")
        X = Y = eval_ids = None
        base_m = None
        if full_cache:
            d = torch.load(os.path.join(pool_dir, "eval_tokens.pt"),
                           map_location="cpu")
            X, Y, eval_ids = d["X"], d["Y"], d["eval_ids"]
            base_m = base_metrics_from_cache(lp_dir, X, Y)
            print(f"BASE ({base_label()}) (from the log-prob cache): "
                  f"ppl {base_m['ppl']:.2f}", flush=True)
        pairs = []
        for i in range(pairs_cache):
            p = os.path.join(pool_dir, f"pairs_blk{i}.pt")
            n = load_pairs_block(p)[0].shape[0]
            pairs.append((p, n))
        for i, (_, n) in enumerate(pairs):
            print(f"  block {i}: {n} pairs (from cache)", flush=True)
        cfg = AutoConfig.from_pretrained(src)
        act = ACT2FN[cfg.hidden_act]
    else:
        banner("STAGE 3 - base: ppl/log-probs/generation "
               "(STREAMING: the full model never loads)"
               if base_pass else
               "STAGE 4 prep - loading the source for calibration (base metrics "
               "are skipped: --skip base / cached)")
        stream = None
        if quantized:
            model = load_source_model(args, src, dtype, device, quantized)
        else:
            try:
                model = BlockStreamRunner(src, dtype=dtype, device=device,
                                          gguf=light_gguf, prefetch=args.prefetch,
                                          io_workers=args.io_threads,
                                          io_cache=args.io_cache)
                stream = model
            except Exception as e:  # noqa: BLE001
                if light_gguf:
                    print(f"streaming failed ({e}) - assembling a full dequant "
                          f"and loading the whole model (may not fit in RAM)",
                          flush=True)
                    src = g2h.convert(gguf_path, conv_dir, dtype="float16",
                                      base_repo=base_repo)
                    light_gguf = None
                    deconv_full_dir = conv_dir
                    model = load_source_model(args, src, dtype, device, False)
                else:
                    print(f"streaming failed ({e}) - loading the full model "
                          f"(may not fit in RAM)", flush=True)
                    model = load_source_model(args, src, dtype, device, quantized)
        if not quantized:
            base_total_b = sum(p.numel() * 2 for p in model.parameters())
            if stream:
                wt = ("quantized GGUF, per-block dequant - no checkpoint"
                      if light_gguf else "disk (dequant checkpoint)")
                print(f"backbone ready; weights: {wt}; full size "
                      f"{base_total_b / 1e9:.2f} GB (fp16 accounting of unpacked "
                      f"values - that IS the quantized model), in RAM only the "
                      f"backbone + one expert block (~2-3 GB)", flush=True)
            else:
                print(f"model loaded; total {base_total_b / 1e9:.2f} GB "
                      f"(fp16 accounting)", flush=True)
        cfg = model.config
        act = ACT2FN[cfg.hidden_act]      # also needed by the fit when the pairs
                                          # come from the cache (stage 4 skipped)

        if base_pass:
            if stream and args.gen_tokens > 12:
                args.gen_tokens = 12
                print("demo generation shortened to 12 tokens: in streaming every "
                      "step reads experts from disk", flush=True)

            X, Y = eval_logits_cache_disk(model, eval_ids, args.eval_ctx,
                                          args.kl_chunks, lp_dir)
            torch.save({"X": X, "Y": Y, "eval_ids": eval_ids},
                       os.path.join(pool_dir, "eval_tokens.pt"))
            if not args.no_cache_verify:
                # cache check: 2 chunks suffice for the base (the cache is its own
                # log-probs; a full check = wasted passes, expensive in streaming)
                chk = eval_vs_cache_disk(model, X, Y, lp_dir, n_max=2)
                print(f"log-prob cache verified on 2 chunks: "
                      f"KL {chk['kl_bits']:.4f} bits", flush=True)
            base_m = base_metrics_from_cache(lp_dir, X, Y)
            if args.gen_tokens > 0:
                base_gen = generate_text(model, eval_ids, tokenizer,
                                         n_new=args.gen_tokens,
                                         repetition_penalty=args.gen_rep_pen)
            print(f"BASE ({base_label()}): ppl {base_m['ppl']:.2f} "
                  f"(from the log-prob cache)", flush=True)
            T["base_total_mb"] = base_total_b / 1e6

        if calib_pass:
            banner("STAGE 4 - calibration: pairs to disk + centroids (STREAMING, "
                   "weights - quantized GGUF)")
            blocks = find_moe_blocks(model)
            geoms = [block_geometry(b, cfg) for _, b in blocks]
            print(f"MoE blocks: {len(blocks)}; first geometry: {geoms[0]}",
                  flush=True)
            gen = torch.Generator().manual_seed(11)
            batches = make_batches(calib_ids, args.calib_ctx, args.calib_bsz,
                                   args.calib_windows, device, gen)
            pairs = collect_pairs(model, blocks, batches, args.per_layer_cap,
                                  flush_dir=pool_dir)
            for i, (p, n) in enumerate(pairs):
                print(f"  block {i}: {n} pairs -> "
                      f"{os.path.basename(p) if p else 'EMPTY!'}", flush=True)
                if n == 0:
                    sys.exit("no pairs for a block - increase --calib-windows / text")
            for i, (_, block) in enumerate(blocks):
                if stream:
                    with stream.with_block(i):
                        mgu, mdn = expert_means(block)
                else:
                    mgu, mdn = expert_means(block)      # without an fp32 expert stack
                # hy_v3: selection bias + shared experts (backbone already in RAM)
                eb = block_router_bias(block)
                sh = block_shared_weights(block)
                if eb is not None or sh is not None:
                    print(f"  block {i}: bias {'yes' if eb is not None else 'no'}, "
                          f"shared {'yes' if sh is not None else 'no'}", flush=True)
                torch.save(dict(geom=geoms[i], gw=router_weight(block.gate).clone(),
                                mgu=mgu, mdn=mdn, eb=eb, shared=sh),
                           os.path.join(pool_dir, f"init_blk{i}.pt"))
            with open(os.path.join(pool_dir, "art_meta.json"), "w",
                      encoding="utf-8") as f:
                base_obj = model.model if stream else model
                json.dump(dict(base_cls=type(base_obj).__name__,
                               router_cls=type(blocks[0][1].gate).__name__,
                               router_mod=type(blocks[0][1].gate).__module__,
                               n_layers=len(blocks),
                               block_names=[n for n, _ in blocks]), f,
                          ensure_ascii=False, indent=2)

        print("unloading the backbone - the fit runs without it", flush=True)
        if stream:
            stream.close()
        release_model(model, device)
        stream = None
    T["stream_mode"] = bool(stream)
    T["reused_cache"] = bool(pairs_cache) and not calib_pass

    # ================= PHASE B - field fit: model NOT in RAM ==================

    n_blocks = len(pairs)
    if "fit" not in plan:
        banner(f"STAGE 5 - field fit r={args.rank}: skipped (--skip fit); "
               f"fit files are taken from {fit_dir}")
        fit_mses = [None] * n_blocks
        try:
            with open(os.path.join(fit_dir, "mse.json"), encoding="utf-8") as f:
                fit_mses = json.load(f)
            if len(fit_mses) != n_blocks:
                fit_mses = [None] * n_blocks
        except Exception:
            pass
    else:
        banner(f"STAGE 5 - field fit r={args.rank} on pairs from disk "
               f"({args.fit_method}, model NOT in RAM)")
        os.makedirs(fit_dir, exist_ok=True)
        fit_sig = dict(fit_steps=args.fit_steps, fit_bs=args.fit_bs, fit_lr=args.fit_lr,
                       fit_method=args.fit_method, fit_jitter=args.fit_jitter,
                       fit_early_stop=args.fit_early_stop,
                       preset=fit_preset or "none")
        fit_meta_p = os.path.join(fit_dir, "fit_meta.json")
        fit_done = all(os.path.isfile(os.path.join(fit_dir, f"fit_blk{i}.pt"))
                       for i in range(n_blocks)) and os.path.isfile(fit_meta_p)
        if fit_done:
            with open(fit_meta_p, encoding="utf-8") as f:
                fit_done = json.load(f) == fit_sig
        if fit_done:
            print(f"fit r={args.rank} with the same settings already cached - "
                  f"skipping (new fit: delete {fit_dir} or change --fit-steps/--fit-method)",
                  flush=True)
            try:
                with open(os.path.join(fit_dir, "mse.json"), encoding="utf-8") as f:
                    fit_mses = json.load(f)
            except Exception:
                fit_mses = [None] * n_blocks
        else:
            def fit_one(i):
                """Fit a single block (thread-safe: only local state + files)."""
                Xi, Yi = load_pairs_block(pairs[i][0])
                ini = torch.load(os.path.join(pool_dir, f"init_blk{i}.pt"),
                                 map_location="cpu")
                fit_mod = FieldSparseMoe(ini["geom"], args.rank, gate_w=ini["gw"],
                                         act_fn=act, gate_bias=ini.get("eb"),
                                         shared=ini.get("shared")).to(device)
                with torch.no_grad():
                    fit_mod.wgud.copy_(ini["mgu"])
                    fit_mod.wdnd.copy_(ini["mdn"])
                mse = fit_field_module(fit_mod, Xi, Yi, args.fit_steps, args.fit_bs,
                                       args.fit_lr, device,
                                       log_prefix=f"block {i}/{n_blocks}",
                                       guard=not args.skip_fit_guard,
                                       method=args.fit_method, seed=5 + i,
                                       jitter=args.fit_jitter,
                                       early_stop=args.fit_early_stop)
                torch.save({n: getattr(fit_mod, n).detach().clone()
                            for n in fit_mod.field_names},
                           os.path.join(fit_dir, f"fit_blk{i}.pt"))
                del fit_mod, Xi, Yi
                gc.collect()
                return mse

            fit_mses = [None] * n_blocks
            errors = []
            w = max(1, min(int(args.fit_workers), n_blocks))
            if w > 1:
                if not args.threads:
                    per = max(1, (os.cpu_count() or 2) // w)
                    torch.set_num_threads(per)
                    print(f"parallel fit: {w} workers x {per} cpu threads "
                          f"(--fit-workers)", flush=True)
                work = queue.Queue()
                for i in range(n_blocks):
                    work.put(i)
                lk = threading.Lock()

                def run_worker():
                    while True:
                        try:
                            i = work.get_nowait()
                        except queue.Empty:
                            return
                        try:
                            mse = fit_one(i)
                            with lk:
                                fit_mses[i] = mse
                        except Exception as e:  # noqa: BLE001
                            with lk:
                                errors.append((i, e))

                ths = [threading.Thread(target=run_worker, name=f"fit-{k}",
                                        daemon=True) for k in range(w)]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
                if errors:
                    i, e = errors[0]
                    raise RuntimeError(f"fit failed on block {i}: {e}") from e
            else:
                for i in range(n_blocks):
                    fit_mses[i] = fit_one(i)
            with open(os.path.join(fit_dir, "mse.json"), "w", encoding="utf-8") as f:
                json.dump(fit_mses, f)
            with open(fit_meta_p, "w", encoding="utf-8") as f:
                json.dump(fit_sig, f)


    # ========== STAGE 5b - refine rounds (self-distillation) ==================
    # The first fit is calibrated on the BASE model's activations, but at
    # deploy the field feeds ITS OWN outputs forward - errors compound layer
    # by layer and the fit never saw those shifted inputs. Each refine round:
    # one streaming pass where the field model feeds forward (hook-replaced
    # outputs) while the original GGUF experts provide the targets, then a
    # warm-started refit on those pairs.
    if "refine" in plan and args.refine_rounds > 0:
        if quantized:
            print("refine rounds need a GGUF/safetensors source (original "
                  "experts must be loadable per block) - skipping", flush=True)
        else:
            n_blocks = len(fit_mses)
            refit_dir = os.path.join(pool_dir, f"refine_r{args.rank}")
            os.makedirs(refit_dir, exist_ok=True)
            import glob as _glob
            for rnd in range(1, args.refine_rounds + 1):
                banner(f"STAGE 5b.{rnd} - refine round: the field feeds forward, "
                       f"the original experts teach")
                try:
                    # 1) field modules from the current fits
                    field_mods = []
                    for i in range(n_blocks):
                        ini = torch.load(os.path.join(pool_dir, f"init_blk{i}.pt"),
                                         map_location="cpu")
                        fm = FieldSparseMoe(ini["geom"], args.rank, gate_w=ini["gw"],
                                            act_fn=act, gate_bias=ini.get("eb"),
                                            shared=ini.get("shared"))
                        fit = torch.load(os.path.join(fit_dir, f"fit_blk{i}.pt"),
                                         map_location="cpu")
                        with torch.no_grad():
                            for k, v in fit.items():
                                getattr(fm, k).copy_(v.float())
                        field_mods.append(fm.to(device).eval())
                    # 2) streaming pass: capture (input -> ORIGINAL output) pairs,
                    #    feed the FIELD output forward
                    for f in _glob.glob(os.path.join(refit_dir, "pairs_blk*_p*.pt")):
                        os.remove(f)
                    runner2 = BlockStreamRunner(src, dtype=dtype, device=device,
                                                gguf=light_gguf,
                                                prefetch=args.prefetch,
                                                io_workers=args.io_threads,
                                                io_cache=args.io_cache)
                    blocks2 = find_moe_blocks(runner2)
                    stores = [[] for _ in range(n_blocks)]   # per-block list of (X,Y) chunk files

                    chunk_X = [[] for _ in range(n_blocks)]
                    chunk_Y = [[] for _ in range(n_blocks)]
                    chunk_n = [0] * n_blocks

                    def store_pair(i, x, y):
                        chunk_X[i].append(x)
                        chunk_Y[i].append(y)
                        chunk_n[i] += x.shape[0]

                    def flush_chunk(i, force=False):
                        # flush a chunk once it is big enough (bounded RAM),
                        # or whatever remains at the end of the pass
                        if chunk_n[i] and (force or chunk_n[i] >= 8192):
                            p = os.path.join(refit_dir,
                                             f"pairs_blk{i}_p{len(stores[i]):02d}.pt")
                            save_pairs_block(chunk_X[i], chunk_Y[i], p)
                            stores[i].append(p)
                            chunk_X[i], chunk_Y[i], chunk_n[i] = [], [], 0

                    hooks2 = []
                    for i, (_, b) in enumerate(blocks2):
                        def make_cap(i):
                            def cap_hook(m, a, output):
                                if chunk_n[i] < args.per_layer_cap:
                                    x = a[0].detach()
                                    y = output.detach() if torch.is_tensor(output) \
                                        else output[0].detach()
                                    store_pair(i,
                                               x.reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu(),
                                               y.reshape(-1, y.shape[-1]).to(torch.bfloat16).cpu())
                                with torch.no_grad():
                                    # fit modules are fp32; the stream is bf16
                                    fout = field_mods[i](a[0].float())
                                fout = fout.to(a[0].dtype)
                                if isinstance(output, tuple):
                                    return (fout,) + tuple(output[1:])
                                return fout
                            return cap_hook
                        hooks2.append(b.register_forward_hook(make_cap(i)))
                    gen2 = torch.Generator().manual_seed(11 + rnd)
                    batches2 = make_batches(calib_ids, args.calib_ctx, args.calib_bsz,
                                            args.calib_windows, device, gen2)
                    for batch in batches2:
                        runner2(**batch)
                        for i in range(n_blocks):
                            flush_chunk(i)
                        if all(chunk_n[i] >= args.per_layer_cap for i in range(n_blocks)):
                            break
                    for h in hooks2:
                        h.remove()
                    for i in range(n_blocks):
                        flush_chunk(i, force=True)
                    runner2.close()
                    del runner2
                    gc.collect()
                    for i, paths in enumerate(stores):
                        print(f"  block {i}: {len(paths)} chunks -> refine pairs",
                              flush=True)
                    # 3) warm-started refit on the refine pairs
                    ref_mses = []
                    for i in range(n_blocks):
                        Xs, Ys = [], []
                        for p in sorted(_glob.glob(os.path.join(
                                refit_dir, f"pairs_blk{i}_p*.pt"))):
                            d = torch.load(p, map_location="cpu")
                            Xs.append(d["X"])
                            Ys.append(d["Y"])
                        Xi = torch.cat(Xs)[:args.per_layer_cap]
                        Yi = torch.cat(Ys)[:args.per_layer_cap]
                        ini = torch.load(os.path.join(pool_dir, f"init_blk{i}.pt"),
                                         map_location="cpu")
                        prev = torch.load(os.path.join(fit_dir, f"fit_blk{i}.pt"),
                                          map_location="cpu")
                        fit_mod = FieldSparseMoe(ini["geom"], args.rank,
                                                 gate_w=ini["gw"], act_fn=act,
                                                 gate_bias=ini.get("eb"),
                                                 shared=ini.get("shared"),
                                                 init=prev).to(device)
                        mse = fit_field_module(fit_mod, Xi, Yi, args.fit_steps,
                                               args.fit_bs, args.fit_lr, device,
                                               log_prefix=f"block {i}/{n_blocks} r{rnd}",
                                               guard=not args.skip_fit_guard,
                                               method=args.fit_method,
                                               seed=1000 * rnd + 5 + i,
                                               jitter=args.fit_jitter,
                                               early_stop=args.fit_early_stop)
                        torch.save({n: getattr(fit_mod, n).detach().clone()
                                    for n in fit_mod.field_names},
                                   os.path.join(fit_dir, f"fit_blk{i}.pt"))
                        del fit_mod, Xi, Yi
                        gc.collect()
                        ref_mses.append(mse)
                    fit_mses = ref_mses
                    with open(os.path.join(fit_dir, "mse.json"), "w",
                              encoding="utf-8") as f:
                        json.dump(fit_mses, f)
                    print(f"refine round {rnd} done: fits updated "
                          f"(warm start from the previous round)", flush=True)
                    T["refine_rounds"] = rnd
                except Exception as e:  # noqa: BLE001
                    print(f"refine round {rnd} failed ({type(e).__name__}: {e}) "
                          f"- continuing with the current fits", flush=True)
                    break

    pairs = None
    gc.collect()

    # worst-block summary: outliers here are the compounding suspects (they
    # shift the residual stream for every layer after them -> loops in text)
    if fit_mses and all(m is not None for m in fit_mses):
        order = sorted(range(len(fit_mses)), key=lambda i: -fit_mses[i])[:3]
        print("worst blocks by final mse: "
              + ", ".join(f"blk{i} {fit_mses[i]:.5f}" for i in order)
              + " (outliers -> candidates for --refine-rounds)", flush=True)

    # ================= PHASE C - streaming artifact + verify ==================
    T["save_backbone"] = "keep"
    if "save" in plan:
        banner("STAGE 6 - building the artifact STREAMINGLY (the full model is not needed)")
        n_blocks = len(fit_mses)
        geoms = [torch.load(os.path.join(pool_dir, f"init_blk{i}.pt"),
                            map_location="cpu")["geom"] for i in range(n_blocks)]
        if quantized and args.save_backbone == "bf16":
            sys.exit("--save-backbone bf16 for a bnb source is not supported in the "
                     "streaming mode: take a GGUF source (it is light anyway)")
        profile = dict(model=args.model, quant=str(args.gguf_quant), rank=args.rank,
                       fit_method=args.fit_method, fit_steps=args.fit_steps,
                       fit_bs=args.fit_bs, fit_lr=args.fit_lr,
                       fit_jitter=args.fit_jitter, fit_early_stop=args.fit_early_stop,
                       fit_preset=fit_preset or "none", fit_workers=args.fit_workers,
                       io_threads=args.io_threads,
                       prefetch=args.prefetch, io_cache=args.io_cache,
                       per_layer_cap=args.per_layer_cap)
        write_field_artifact(src, out_dir, pool_dir, fit_dir, args.rank, dtype,
                             gguf=light_gguf, profile=profile,
                             io_workers=args.io_threads, io_cache=args.io_cache)
        full_b, field_b = field_accounting(geoms, args.rank)
        T.update(rank=args.rank, full_experts_mb=full_b / 1e6, field_mb=field_b / 1e6,
                 ratio=full_b / max(field_b, 1), fit_mses=fit_mses,
                 cache_dir=pool_dir, fit_dir=fit_dir, phased_flow=True,
                 low_mem=bool(args.low_mem), base_label=base_label(), profile=profile)
        print(f"\nFull experts: {full_b / 1e6:.0f} MB -> field: {field_b / 1e6:.0f} MB "
              f"(x{full_b / field_b:.1f} on experts)", flush=True)
        print(f"artifact -> {out_dir} ({dir_size_gb(out_dir):.2f} GB) - experts "
              f"discarded, backbone + field remain: SMALLER than the original Q4 GGUF",
              flush=True)
    else:
        print(f"STAGE 6 - save: skipped (--skip save); existing artifact: {out_dir}")
        T.update(rank=args.rank, fit_mses=fit_mses, low_mem=bool(args.low_mem),
                 base_label=base_label())

    if "verify" in plan:
        banner("STAGE 7 - verify: the artifact loads as a NORMAL model")
        with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
            art_q = (json.load(f).get("quantization_config") or {}).get("quant_method")
        if art_q:
            art = AutoModelForCausalLM.from_pretrained(
                out_dir, device_map={"": 0}, trust_remote_code=True,
                low_cpu_mem_usage=True).eval()
        else:
            art = AutoModelForCausalLM.from_pretrained(
                out_dir, dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True).to(device).eval()
        field_m = eval_vs_cache_disk(art, X, Y, lp_dir)
        field_gen = None
        if args.gen_tokens > 0:
            field_gen = generate_text(art, eval_ids, tokenizer,
                                      n_new=args.gen_tokens,
                                      repetition_penalty=args.gen_rep_pen)
        if base_m:
            dpct = 100 * (field_m["ppl"] - base_m["ppl"]) / base_m["ppl"]
            print(f"FIELD (artifact) r={args.rank}: KL {field_m['kl_bits']:.3f} bits/token, "
                  f"ppl {field_m['ppl']:.2f} ({dpct:+.1f}%)", flush=True)
        else:
            dpct = None
            print(f"FIELD (artifact) r={args.rank}: KL {field_m['kl_bits']:.3f} bits/token "
                  f"(base ppl not in this run - no delta)", flush=True)
        if field_gen:
            print("\nGeneration FROM THE ARTIFACT (same base/prompt as above):\n"
                  + field_gen, flush=True)
        print(f"\nInference from the saved artifact: `python3 hf_chat.py` "
              f"(or step2_chat.bat) - it finds the artifact itself", flush=True)
        T.update(base=base_m, field=field_m, ppl_delta=dpct,
                 gen_base=base_gen, gen_field=field_gen)
        release_model(art, device)

    banner("PIPELINE FINISHED")
    # the dequant checkpoint is no longer needed: the artifact is built and
    # verified; from the GGUF (if still around) everything rebuilds without
    # re-downloading
    if deconv_full_dir and not args.keep_dequant and args.gguf_out is None:
        banner("CLEANUP - the dequant checkpoint is no longer needed, erasing "
               "(the artifact is ready)")
        do_cleanup([deconv_full_dir])
        deconv_full_dir = None
    T["total_seconds"] = round(time.time() - t0, 1)
    if "report" in plan:
        write_report(args, out_dir)
    else:
        print("report skipped (--skip report)", flush=True)
    if args.cleanup:
        do_cleanup(cleanup_targets)
    print(f"total {T['total_seconds']} s\nartifact: {out_dir}"
          + (f"\nreport: {out_dir}/README.md + {DL}/moe_hf_pipeline_report.md"
             if "report" in plan else "")
          + f"\nchat: python3 hf_chat.py  (finds the artifact itself)", flush=True)
    print("left on disk:" + "\n" + "\n".join(
        f"  {name}: {dir_size_gb(p):.2f} GB  ({p})"
        for name, p in (("GGUF source", gguf_path),
                        ("pool cache (pairs/centroids/log-probs)", pool_dir),
                        ("field fit r=" + str(args.rank), fit_dir),
                        ("artifact (backbone + field)", out_dir))
        if p and os.path.exists(p)), flush=True)


def write_report(args, out_dir):
    from hf_field_transform import TEMPLATE_PATH  # noqa: F401
    md = ["# Model with a field instead of experts (field-engine)", ""]
    sq = str(T.get("source_quant", "none"))
    is_gguf_src = sq.startswith("gguf")
    q4 = (T.get("save_backbone") == "keep" and sq in ("bitsandbytes", "bnb-4bit"))
    if is_gguf_src:
        back = f"fp16 (dequant from GGUF {sq.split(':')[1]})"
    elif q4:
        back = "Q4 (bitsandbytes)"
    else:
        back = "bf16/fp32"
    md.append(f"Model: `{args.model}` | field rank: **r={args.rank}** | "
              f"artifact backbone: {back} | "
              f"device: {T['device']} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pr = T.get("profile")
    if pr:
        md.append(f"Profile: quant {pr.get('quant')} | fit "
                  f"{pr.get('fit_method')}/{pr.get('fit_steps')} steps/"
                  f"bs {pr.get('fit_bs')}/lr {pr.get('fit_lr')}"
                  f" (preset {pr.get('fit_preset')}) | workers "
                  f"{pr.get('fit_workers')} | prefetch {pr.get('prefetch')}")
    if T.get("plan"):
        md.append(f"Stages run: {' -> '.join(T['plan'])}"
                  + (f" | skipped: {', '.join(T['plan_skipped'])}"
                     if T.get("plan_skipped") else ""))
    md.append("")
    if T.get("reused_cache"):
        mem = ("Calibration taken from the previous run's cache: the pool of "
               "activation-vector pairs depends on neither text order nor field "
               "rank, so the base/calibration stages were skipped entirely. ")
    else:
        mem = ("The full model never loaded: the base/calibration stages ran "
               "STREAMING (backbone in RAM ~1.5 GB, each block's experts read "
               "from disk on their layer's pass), the fit ran without the model "
               "at all (peak ~1-2 GB). ")
    md.append("Memory: " + mem + "Run cache (pair pool + base log-probs): "
              f"`{T.get('cache_dir', 'results/cache_*')}` - reusable for a new "
              "rank/fit settings without recalibration.")
    md.append("")
    if T.get("full_experts_mb") is not None:
        md.append("## Expert compression")
        md.append("")
        md.append("| full experts | field | compression |")
        md.append("|---|---|---|")
        md.append(f"| {T['full_experts_mb']:.0f} MB | {T['field_mb']:.0f} MB | "
                  f"x{T['ratio']:.1f} |")
        md.append("")
        md.append("Explicit expert weights are NOT stored: centroids + low-rank "
                  "factors U,V + coordinates C (the movement seed is computed from "
                  "the router).")
        md.append("")
        md.append(f"Artifact size on disk: **{dir_size_gb(out_dir):.2f} GB** "
                  "(backbone + field). The experts (~12.9 GB fp16 for OLMoE) are "
                  "replaced by the field, so the artifact is smaller than even the "
                  "original Q4 GGUF; inference from it takes RAM roughly the size "
                  "of the artifact + ~1 GB overhead. Recompressing the artifact is "
                  "not needed.")
        md.append("")
    else:
        md.append("## Expert compression")
        md.append("")
        md.append("Not computed in this run (save stage was skipped; the artifact "
                  "from the previous run is used as is).")
        md.append("")
    left = [("GGUF source", T.get("_gguf_path")),
            ("calibration pool cache", T.get("cache_dir")),
            ("field fit", T.get("fit_dir")), ("artifact", out_dir)]
    rows = [f"| {n} | `{p}` | {dir_size_gb(p):.2f} |" for n, p in left
            if p and os.path.exists(p)]
    if rows:
        md.append("What remains on disk after the run:")
        md.append("")
        md.append("| what | where | GB |")
        md.append("|---|---|---|")
        md.extend(rows)
        md.append("")
        md.append("The pool cache survives rank/fit changes; the GGUF and the "
                  "light catalog can be erased (--cleanup) - everything "
                  "rebuilds without re-downloading the model.")
        md.append("")
    md.append("## Quality: comparison with the QUANTIZED model")
    md.append("")
    md.append("The comparison base is the quantized model from the GGUF itself: "
              "its weights are read straight from the .gguf (per-block dequant, "
              "unpacked values bit-identical to what the quantized model "
              "computes - not a separate model, the same numbers). A full "
              "dequant checkpoint is never created.")
    md.append("")
    md.append("| variant | ppl | KL vs base, bits/token |")
    md.append("|---|---|---|")
    if T.get("base"):
        md.append(f"| base ({T.get('base_label', 'quantized Q4_K_M (GGUF)')}) | "
                  f"{T['base']['ppl']:.2f} | 0 |")
    if T.get("field"):
        md.append(f"| field r={args.rank} (from the artifact) | {T['field']['ppl']:.2f} | "
                  f"{T['field']['kl_bits']:.3f} (Δppl {T['ppl_delta']:+.1f}%) |")
    else:
        md.append("| field | metrics skipped in this run (--skip verify) | |")
    md.append("")
    md.append("## Generation (greedy, same prompt)")
    md.append("")
    if T.get("gen_base"):
        md.append("**Base:**\n```text\n" + T["gen_base"][:300] + "\n```")
    else:
        md.append("Base: generation not saved in this run (cache-only or "
                  "--gen-tokens 0; see the first run's report for a sample).")
    if T.get("gen_field"):
        md.append(f"**Field r={args.rank} (generation from the built artifact):**\n"
                  "```text\n" + T["gen_field"][:300] + "\n```")
    md.append("## How to load locally")
    md.append("")
    md.append("```python")
    md.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
    md.append(f'm = AutoModelForCausalLM.from_pretrained("{os.path.basename(out_dir)}",')
    md.append('    trust_remote_code=True, device_map="auto")  '
              + ('# Q4 artifact: needs a GPU + bitsandbytes' if q4
                 else '# works on CPU too'))
    md.append("```")
    md.append("")
    md.append(f"Chat with the model: `python3 hf_chat.py --model {os.path.basename(out_dir)}`")
    md.append("")
    md.append("vLLM: the base model works directly; the field version needs a "
              "small adapter (the field forward is ~10 lines, see "
              "modeling_field.py).")
    if q4:
        md.append("The experts are already compressed by the field (see the "
                  "table) and stored in bf16; the artifact backbone is in the "
                  "original quantization (Q4 bitsandbytes, GPU inference). For "
                  "CPU inference rebuild with a GGUF source.")
    else:
        md.append("The experts are already compressed by the field (see the "
                  "table) and stored in bf16; the artifact backbone is plain "
                  "fp16/bf16 weights (CPU/GPU inference, no surprises).")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    os.makedirs(DL, exist_ok=True)
    for ext in (".md", ".json"):
        dst = os.path.join(DL, "moe_hf_pipeline_report" + ext)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(md) if ext == ".md" else
                    json.dumps({k: v for k, v in T.items()}, ensure_ascii=False,
                               indent=2, default=str))


if __name__ == "__main__":
    main()
