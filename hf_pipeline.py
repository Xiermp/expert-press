# version: 2026-09-05.5 - SVD-INIT SELF-HEAL: missing fit_dir/init_svd_blk*.pt
#   (pool cache from a pre-SVD build, or an earlier run that fell back to the
#   random init) are detected BEFORE the fit and rebuilt by a streaming pass
#   from the real expert deltas - the optimizer never starts blind again. The
#   stage-5 fallback is now a loud WARNING + report metadata, and refine
#   rounds key their signature on the fit state (a re-fit invalidates stale
#   refine caches: the captured pairs are the field's own forward outputs).
# version: 2026-09-05.4 - REFINE RESUME + PARALLEL REFIT: a completed refine
#   round is skipped on re-runs (done_r*.json markers in the run cache), a
#   half-done round reuses the captured pairs (pairs_sig.json) instead of
#   re-streaming the whole model for hours, and the refit loop now honors
#   --fit-workers (before: always one worker). --refresh-refine forces a
#   full redo.
# version: 2026-09-05.2 - SWAP-STORM GUARD (refine freeze fix): the capture
#   pass of the refine round kept per-block pair chunks of up to 8192 pairs in
#   RAM for ALL blocks at once (~1.1 GB) while the io-cache ram copy was also
#   growing - on a 3.2 GB-free box Windows slid into a swap-storm at ~block 13
#   (no MemoryError, the run just froze). Now: the flush threshold adapts to
#   free RAM (1024 pairs below 8 GB -> resident ~= one batch, ~0.4 GB), the
#   capture pass prints a per-window progress line ("frozen" vs "working" is
#   visible), and BlockStreamRunner re-checks io-cache ram at every stage
#   (see hf_stream).
# version: 2026-09-05.1 - LOW-RAM FIX: (1) a cached pair pool SMALLER than the
#   requested --per-layer-cap is now KEPT by default (>= usable floor of
#   max(4096, 16*rank, cap/2) pairs/block) instead of forcing the most
#   expensive re-collection, which OOM-crashed low-RAM boxes and looked like
#   "nothing was saved" - --pool-recalibrate forces the old behavior;
#   (2) the "cached pool holds ..." notice prints once per run, not 3-6x.
# version: 2026-09-04.3 - RESUME FIX: the run cache no longer resets to zero
#   after an interruption ("nothing was saved"): (1) art_meta.json is written
#   at the START of stage 4 (it was written last, so a kill during the silent
#   centroids+SVD loop invalidated the whole pair pool on restart); (2) the
#   stage-4 init loop is resumable per block (existing init_blk*.pt are
#   reused, only the missing ones are rebuilt) and prints per-block progress
#   + timing; (3) the stage-5 fit is resumable per block (fit_blk*.pt +
#   fit_partial.json with the exact fit_sig; only unfitted blocks re-run;
#   finished blocks are reused verbatim); (4) all cache jsons and pair/fit
#   saves are atomic (tmp + os.replace) - a kill mid-save cannot leave a torn
#   file that poisons the next run. Semantics of the stage-4 skip: the pair
#   pool is reused whenever it is on disk (the expensive, model-forward part);
#   only missing centroids/init files are rebuilt from the model.
# version: 2026-09-04.2 - UPDATE-10 speed rebuild + the user's three holes:
#   --fit-init {svd,random}: SVD init of U,V,C from the STACKED expert deltas
#   (shared-basis randomized SVD over the streamed experts, saved per-rank as
#   fit_dir/init_svd_blk*.pt; --refresh-init rebuilds those for an existing
#   pool without recalibrating); --fit-autocast {auto,on,off} with the honest
#   real-step probe (1 warmup + 3 timed steps per dtype arm, >=1.2x rule);
#   --fit-method muon|muon-cosine (BY-NAME split: U*/V* only, C*/gw never),
#   --muon-max-dim/--muon-ns-steps; jitter routing follows the clean anchor
#   row inside fit_field_module. Toy bench: svd-init same quality in ~3x
#   fewer steps, autocast 1.7-1.8x/step.
#   --fit-guard-warmup (-1=auto) and --strict-fit-guard (old hard error),
#   --fit-lr-warmup (linear Adam adaptation ramp)
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
import hashlib
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


def _save_json_atomic(path, obj):
    """Write a cache json atomically (tmp + os.replace). A kill mid-write
    leaves the previous version intact instead of a torn/empty file that
    would invalidate the whole cache on the next start."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


_POOL_NOTICE_SHOWN = [False]   # the pool-size notice prints once per run


def pool_is_complete(cache_dir, min_pairs=0, keep_smaller=False,
                     min_useful=0):
    """Pair pool already on disk? -> block count (0 = no).
    The PAIR POOL is the expensive part of stage 4 (a full model forward
    over the calibration windows) and it does not depend on rank nor on the
    fit settings - once it is on disk it must never be re-collected, even if
    the later (cheap, but silent) centroids/SVD-init loop did not finish.
    Requires art_meta.json + every pairs_blk{i}.pt (art_meta is written at
    the START of stage 4, so a kill anywhere later keeps the pool resumable).

    min_pairs: when the cached pool holds FEWER pairs per block than the
    requested --per-layer-cap it is treated as incomplete (the caller
    recalibrates with the bigger cap) - UNLESS keep_smaller and the pool is
    still usable (>= min_useful). 2026-09-05.1: re-collection is the most
    expensive thing the pipeline does and it OOM-crashed low-RAM boxes (and
    looked like "nothing was saved"); 49k pairs at rank 128 = 384 pairs/dim
    is already ample, so a usable smaller pool is KEPT with a one-line
    notice and --pool-recalibrate restores the old forced re-collection."""
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
    if min_pairs:
        try:
            x0 = _t.load(os.path.join(cache_dir, "pairs_blk0.pt"),
                         map_location="cpu")["X"]
            have = int(x0.shape[0])
            if have < min_pairs:
                if keep_smaller and have >= min_useful:
                    if not _POOL_NOTICE_SHOWN[0]:
                        _POOL_NOTICE_SHOWN[0] = True
                        print(f"cached pool holds {have} pairs/block < requested "
                              f"cap {min_pairs} - keeping the cached pool "
                              f"(>= the usable floor {min_useful}); pass "
                              f"--pool-recalibrate to force a bigger one",
                              flush=True)
                    return n
                if not _POOL_NOTICE_SHOWN[0]:
                    _POOL_NOTICE_SHOWN[0] = True
                    print(f"cached pool holds {have} pairs/block < requested "
                          f"cap {min_pairs} - recalibrating with the bigger pool",
                          flush=True)
                return 0
        except Exception:
            return 0
    return n


def lp_cache_is_complete(cache_dir, lp_dir):
    """eval_tokens.pt + every lp_XXX.pt chunk present/loadable? Prereq of the
    verify stage only (the base log-prob cache is independent of the pair
    pool / init files state)."""
    import torch as _t
    tp = os.path.join(cache_dir, "eval_tokens.pt")
    if not os.path.isfile(tp):
        return False
    try:
        d = _t.load(tp, map_location="cpu")
    except Exception:  # noqa: BLE001
        return False
    return all(os.path.isfile(os.path.join(lp_dir, f"lp_{i:03d}.pt"))
               for i in range(len(d["X"])))


def cache_is_complete(cache_dir, lp_dir, min_pairs=0, pairs_only=False,
                      keep_smaller=False, min_useful=0):
    """Pair pool + base log-probs already on disk? -> block count (0 = no).
    Pairs/centroids/log-probs depend on neither rank nor text order - such a
    cache survives a --rank/--fit-* change, and --cleanup does NOT touch it.
    min_pairs: when the cached pool holds FEWER pairs per block than the
    requested --per-layer-cap, the cache is treated as incomplete (the caller
    recalibrates with the bigger cap) - unless keep_smaller/min_useful say
    the smaller pool is usable (see pool_is_complete, 2026-09-05.1).
    pairs_only: check ONLY the pair pool + per-block init files (art_meta/
    pairs/init_blk), without the base log-prob cache - enough for
    fit/save/refine, which never read the log-probs; the full check is needed
    by base/verify. The pair-pool-only check is pool_is_complete()."""
    import torch as _t
    n = pool_is_complete(cache_dir, min_pairs=min_pairs,
                         keep_smaller=keep_smaller, min_useful=min_useful)
    if not n:
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


def fit_blocks_ok(fit_dir, n):
    """True when every fit_blkN.pt exists and is non-empty. A 0-byte file is
    a leftover of an interrupted write (seen in the wild): treat it as
    missing so the fit re-runs instead of dying later in torch.load."""
    return all(os.path.isfile(os.path.join(fit_dir, f"fit_blk{i}.pt"))
               and os.path.getsize(os.path.join(fit_dir, f"fit_blk{i}.pt")) > 0
               for i in range(n))


def ensure_prereqs(plan, args, pool_dir, lp_dir, fit_dir, out_dir, min_pairs):
    """Auto-add cheap missing stages with a notice; hard-fail with a hint when
    an expensive prerequisite is absent from disk AND from the plan. Returns
    the final ordered plan."""
    ks = not getattr(args, "pool_recalibrate", False)
    mu = max(4096, 16 * args.rank, (min_pairs or 0) // 2)
    full = cache_is_complete(pool_dir, lp_dir, min_pairs=min_pairs,
                             keep_smaller=ks, min_useful=mu)
    pairs = full or cache_is_complete(pool_dir, lp_dir, min_pairs=min_pairs,
                                      pairs_only=True, keep_smaller=ks,
                                      min_useful=mu)
    # pool = the pair files alone (the expensive part; init files are cheap
    # model-only passes that a resumed run rebuilds per block)
    pool = pairs or pool_is_complete(pool_dir, min_pairs=min_pairs,
                                     keep_smaller=ks, min_useful=mu)
    auto = []

    def add(s):
        if s not in plan:
            plan.append(s)
            auto.append(s)

    if plan != ["report"]:          # anything but a pure report rerun needs src
        add("download")
    if "refine" in plan:
        add("texts")
    if "calibrate" in plan and not pool:
        add("texts")
    if "fit" in plan and not pool and "calibrate" not in plan:
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
    if "fit" in plan and pool and "calibrate" not in plan:
        # the pool is on disk, but an interruption could have left some
        # init_blk*.pt unwritten (the fit needs ALL of them) - rebuild just
        # the missing init files instead of dying inside the fit
        have = pool
        if not all(os.path.isfile(os.path.join(pool_dir, f"init_blk{i}.pt"))
                   for i in range(have)):
            add("calibrate")
            add("texts")
            print("note: init_blk*.pt are incomplete in the cache - auto-adding "
                  "the 'calibrate' stage (the pair pool is reused, only the "
                  "missing init files are rebuilt)", flush=True)
    elif (("save" in plan or "refine" in plan) and pool
            and "calibrate" not in plan and "fit" not in plan):
        if not all(os.path.isfile(os.path.join(pool_dir, f"init_blk{i}.pt"))
                   for i in range(pool)):
            add("calibrate")
            add("texts")
            print("note: init_blk*.pt are incomplete in the cache - auto-adding "
                  "the 'calibrate' stage (the pair pool is reused, only the "
                  "missing init files are rebuilt)", flush=True)
    if ("save" in plan or "refine" in plan) and "fit" not in plan:
        n = pool or 0
        if not n or not fit_blocks_ok(fit_dir, n):
            what = "stage 'refine': the fits" if "refine" in plan else \
                   "stage 'save': the fits"
            sys.exit(f"{what} are missing in {fit_dir}\n"
                     "  -> keep 'fit' in the plan, or finish a full run first")
    if "verify" in plan:
        if not lp_cache_is_complete(pool_dir, lp_dir) and "base" not in plan:
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


def ram_gb(available=False):
    """GB of RAM (available/total); best effort - never raises."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (vm.available if available else vm.total) / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k] = int(v.strip().split()[0]) * 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", total)
        return (avail if available else total) / 1e9
    except Exception:
        return 8.0


def _cuda_ok():
    """Cheap CUDA probe that works before the bootstrap stage."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def apply_hardware_profile(args, device):
    """--profile auto|low|high: lift the conservative defaults on strong
    hardware, keep them 1:1 on a weak box. Explicit flags always win: only
    values still at their None placeholder are auto-filled."""
    cuda = (device == "cuda")
    total_gb, cores = ram_gb(), (os.cpu_count() or 1)
    if getattr(args, "profile", "auto") == "auto":
        args.profile = "high" if (cuda or (total_gb >= 32 and cores >= 8)) else "low"
    high = (args.profile == "high")
    low_mem = getattr(args, "low_mem", False)
    if getattr(args, "io_cache", None) is None:
        args.io_cache = "ram" if (high and not low_mem) else "disk"
        args._io_cache_auto = True
    if getattr(args, "io_threads", None) is None:
        args.io_threads = 4 if high else 1
    bsz16 = (high and cuda and not low_mem)
    if getattr(args, "calib_bsz", None) is None:
        args.calib_bsz = 16 if bsz16 else 8
    elif bsz16 and args.calib_bsz == 8:
        args.calib_bsz = 16
    if high and cuda and not low_mem:
        if getattr(args, "dtype", "auto") == "auto":
            try:
                import torch
                if tuple(torch.cuda.get_device_capability()) < (8, 0):
                    args.dtype = "float16"   # bf16 needs sm_80+ (T4 = sm_75)
                    print("[profile] pre-Ampere GPU: dtype auto -> float16 "
                          "(no native bfloat16)", flush=True)
            except Exception:
                pass
    print(f"[profile] {args.profile} (device={device}, ram={total_gb:.0f} GB, "
          f"cores={cores}) -> io-cache={args.io_cache}, "
          f"io-threads={args.io_threads}, calib-bsz={args.calib_bsz}"
          + (f", dtype={args.dtype}" if getattr(args, "dtype", "auto") != "auto"
             else ""), flush=True)


def check_io_cache_fit(args, gguf_path):
    """io-cache ram needs the GGUF (+ dequant headroom) in free RAM; an
    auto-selected ram cache downgrades to disk when it does not fit, an
    explicit one only warns."""
    if getattr(args, "io_cache", None) != "ram" or not gguf_path:
        return
    try:
        size_gb = os.path.getsize(gguf_path) / 1e9
    except OSError:
        return
    avail = ram_gb(available=True)
    if avail >= max(size_gb * 2.0, 4.0):
        return
    if getattr(args, "_io_cache_auto", False):
        print(f"[profile] io-cache ram: {avail:.1f} GB free is too tight for a "
              f"{size_gb:.1f} GB GGUF (+headroom) -> falling back to disk",
              flush=True)
        args.io_cache = "disk"
    else:
        print(f"[profile] WARNING: io-cache ram may not fit ({size_gb:.1f} GB "
              f"GGUF, {avail:.1f} GB free) - every streaming stage re-checks "
              f"and falls back to disk if it still does not fit "
              f"(MOE_FORCE_IO_RAM=1 overrides)", flush=True)


def refine_flush_at(available_gb):
    """Resident-pair flush threshold for the refine capture pass
    (2026-09-05.2). The old fixed 8192 pairs/block meant ~1.1 GB of chunks
    across 23 blocks on top of the backbone and (optionally) the io-cache ram
    copy - the last straw on a 3.2 GB-free box (it froze mid-window without
    any error). Below 8 GB free: flush at 1024 pairs -> each block holds at
    most the current window's batch (~0.4 GB resident for 23x4096x1024 bf16
    x2 sides); the cost is more, smaller chunk files, nothing else."""
    return 1024 if (available_gb is not None and available_gb < 8.0) else 8192


def print_profile(args, fit, preset):
    """One place to see everything the run will actually use."""
    rows = [
        ("profile", args.profile),
        ("io_cache", args.io_cache),
        ("model", args.model),
        ("quant", "auto" if str(args.gguf_quant).lower() == "auto" else args.gguf_quant),
        ("rank", args.rank),
        ("fit", f"method={fit['fit_method']} steps={fit['fit_steps']} "
                f"bs={fit['fit_bs']} lr={fit['fit_lr']}"
                + (f" preset={preset}" if preset else "")),
        ("fit_workers", args.fit_workers),
        ("fit_jitter", args.fit_jitter),
        ("fit_init/autocast", f"{args.fit_init}/{args.fit_autocast}"),
        ("muon", f"max_dim={args.muon_max_dim} ns={args.muon_ns_steps}"),
        ("fit_early_stop", args.fit_early_stop),
        ("refine_rounds", args.refine_rounds),
        ("io_threads", args.io_threads),
        ("prefetch", args.prefetch),
        ("cpu cores", os.cpu_count()),
        ("pairs/layer cap", args.per_layer_cap),
        ("eval chunks / kl chunks", f"{args.eval_chunks} / {args.kl_chunks}"),
    ]
    if getattr(args, "pool_recalibrate", False):
        rows.append(("pool", "forced recalibration (--pool-recalibrate)"))
    print("profile: " + " | ".join(f"{k}={v}" for k, v in rows), flush=True)


def _compat_check():
    """Fail fast with one clear message when the sibling .py files are older
    than this hf_pipeline.py (a partial update - hit in the wild: a folder
    with the new hf_pipeline.py but an old hf_stream.py without
    BlockStreamRunner). Every symbol below was added by a specific update
    commit, so a missing one dates the stale file precisely."""
    need = {
        "hf_env": ("CACHE",),
        "hf_stream": ("BlockStreamRunner",),
        "hf_field_transform": ("_mdev", "fit_field_module",
                               "block_router_bias", "block_shared_weights",
                               "expert_basis_init", "_resolve_fit_autocast"),
        "hf_gguf_to_hf": ("resolve_gguf",),
    }
    for mod, names in need.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[update] cannot import {mod}.py: {e}\n"
                     "[update] the folder has a broken/mixed file set - "
                     "re-copy ALL .py files from the update package")
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            sys.exit(f"[update] {mod}.py is OUTDATED "
                     f"(missing: {', '.join(missing)})\n"
                     "[update] this hf_pipeline.py is from the full update "
                     "package - re-copy ALL .py files from it, not just some")


def main():
    _compat_check()
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
    ap.add_argument("--profile", default="auto", choices=["auto", "low", "high"],
                    help="hardware profile: auto detects it (CUDA, or 32 GB+ RAM "
                         "and 8+ cores -> high, else low); high lifts the "
                         "conservative defaults on a strong box (io-cache ram, "
                         "io-threads 4, calib-bsz 16 on GPU, fp16 on pre-Ampere "
                         "GPUs); low = the classic cautious defaults. Explicit "
                         "--io-cache/--io-threads/--calib-bsz/--dtype always win")
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--calib-dataset", default=None,
                    help="e.g. wikitext-2-raw-v1 (requires datasets)")
    ap.add_argument("--text-cap", type=int, default=3_000_000, help="text characters")
    ap.add_argument("--calib-windows", type=int, default=3)
    ap.add_argument("--calib-bsz", type=int, default=None,
                    help="calibration batch size (default: 8, or 16 on a GPU "
                         "under the high profile)")
    ap.add_argument("--calib-ctx", type=int, default=512)
    ap.add_argument("--per-layer-cap", type=int, default=8192)
    ap.add_argument("--pool-recalibrate", action="store_true",
                    help="force re-collection when the cached pair pool holds "
                         "fewer pairs/block than --per-layer-cap (default: keep "
                         "the cached pool when it is still usable - >= "
                         "max(4096, 16*rank, cap/2) pairs/block - re-collection "
                         "is the most expensive stage and OOM-crashes low-RAM "
                         "boxes)")
    ap.add_argument("--fit-steps", type=int, default=None,
                    help="fit steps per block (default: 300, or the preset value)")
    ap.add_argument("--fit-bs", type=int, default=None,
                    help="fit batch size (default: 4096, or the preset value)")
    ap.add_argument("--fit-lr", type=float, default=None,
                    help="fit learning rate (default: 2e-3, or the preset value)")
    ap.add_argument("--fit-method", default=None,
                    help="optimizer: adam | adamw | adam-cosine | rmsprop | "
                         "muon | muon-cosine (muon: NS-orthogonalized updates "
                         "for the U*/V* factors, BY-NAME split; C*/router "
                         "always stay on Adam; default: adam, or the preset "
                         "value)")
    ap.add_argument("--fit-autocast", default="auto", choices=["auto", "on", "off"],
                    help="bf16-autocast fit (params stay fp32, matmuls run "
                         "bf16 via oneDNN - 1.7-1.8x/step on the toy): auto "
                         "= the honest real-step probe decides per geometry "
                         "(8 real steps total, cached; keeps fp32 when bf16 "
                         "is not faster, e.g. no AMX/AVX512-bf16 ISA)")
    ap.add_argument("--muon-max-dim", type=int, default=512,
                    help="muon split gate: a U*/V* factor goes through "
                         "Newton-Schulz only if min(shape) <= this (raise to "
                         "let the big centroid matrices join; costs ~40-50%% "
                         "of the step time on real models)")
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="Newton-Schulz iterations per muon update "
                         "(3 = cheaper/looser, 5 = default)")
    ap.add_argument("--fit-init", default="svd", choices=["svd", "random"],
                    help="U,V,C initialization: svd = shared basis of the "
                         "REAL expert deltas (captures ~50-70%% of the delta "
                         "energy at step 0 - the toy reaches the old fit's "
                         "quality in ~3x fewer steps; computed once per rank "
                         "from the streamed experts); random = the old "
                         "randn*0.02/zeros")
    ap.add_argument("--refresh-init", action="store_true",
                    help="rebuild ONLY the per-rank SVD init files "
                         "(init_svd_blk*.pt) for an existing pool cache and "
                         "exit - normally not needed: since 2026-09-05.5 the "
                         "pipeline detects missing files and rebuilds them "
                         "automatically before the fit; existing files are "
                         "kept (delete fit_dir/init_svd_blk*.pt to force)")
    ap.add_argument("--refresh-refine", action="store_true",
                    help="ignore the refine round cache (done markers + "
                         "captured pairs) and redo the refine rounds - use "
                         "after changing fit settings or when the current "
                         "fits must be re-taught from scratch")
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
                    help="parallel fit workers for independent blocks, the "
                         "refine refit uses them too (2-4 on a multi-core "
                         "CPU; 1 = sequential)")
    ap.add_argument("--fit-early-stop", type=int, default=0,
                    help="stop each block's fit after 2 consecutive flat mse "
                         "checkpoints (every N steps, e.g. 50; 0 = off). Saves "
                         "time on plateauing blocks")
    ap.add_argument("--fit-guard-warmup", type=int, default=-1,
                    help="the 2x divergence bail is armed only after this many "
                         "fit steps: -1 = auto (max(30, steps//10)), 0 = old "
                         "always-armed behavior. The old guard could abort a "
                         "block on Adam's normal early overshoot, before the "
                         "optimizer had time to adapt")
    ap.add_argument("--strict-fit-guard", action="store_true",
                    help="old hard end-guard: abort the run when a block's fit "
                         "mse did not drop at least 2%% below the centroid "
                         "baseline. Default: warn and ship the best state (a "
                         "fit that ended WORSE than the baseline still aborts)")
    ap.add_argument("--fit-lr-warmup", type=int, default=0,
                    help="linear lr ramp over the first N fit steps - gives "
                         "Adam time to adapt (0 = off). Useful when blocks "
                         "stall or diverge at the start")
    ap.add_argument("--fit-router", default="off", choices=["off", "after", "joint"],
                    help="let the ORIGINAL router join the rebuild (in place, "
                         "pairs from disk - no extra artifact memory): after = "
                         "short anchored polish once the field fit is done; "
                         "joint = the router trains alongside the field from "
                         "step 0. The tuned router replaces the gate weight in "
                         "the artifact. Toy-bench caveat: after a converged fit "
                         "the router is usually NOT the bottleneck - treat as a "
                         "cheap diagnostic; --refine-rounds is the stronger lever")
    ap.add_argument("--router-steps", type=int, default=80,
                    help="anchored router-polish steps for --fit-router after")
    ap.add_argument("--router-lr", type=float, default=None,
                    help="router polish lr (default: the fit lr)")
    ap.add_argument("--router-anchor", type=float, default=0.03,
                    help="L2 anchor pulling the router to the original "
                         "(0 = free router; higher = safer for LM-level drift)")
    ap.add_argument("--refine-rounds", type=int, default=0,
                    help="self-distillation rounds after the first fit (try 1-2): "
                         "a streaming pass where the FIELD model feeds its own "
                         "outputs forward while the original GGUF experts provide "
                         "targets, then a warm-started refit. Fixes the compounding "
                         "error the first fit cannot see (it is calibrated on the "
                         "BASE model's activations)")
    ap.add_argument("--io-threads", type=int, default=None,
                    help="threads for GGUF dequant of expert tensors (2-4 speeds "
                         "up stage 3-4/6 block reads on a multi-core CPU; "
                         "default: 1, or 4 under the high profile)")
    ap.add_argument("--prefetch", type=int, default=1,
                    help="background prefetch of the next expert block while the "
                         "current layer computes (default 1; 0 = off, saves ~1 "
                         "block of RAM)")
    ap.add_argument("--io-cache", choices=["disk", "ram"], default=None,
                    help="ram: copy the packed GGUF tensors into RAM on first "
                         "touch (~= packed file size); later passes (pool "
                         "collection, refit, artifact write) read nothing from "
                         "disk - big win on Colab/Drive or HDD (default: auto - "
                         "disk, or ram under the high profile when the GGUF "
                         "fits in free RAM)")
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
    device_hint = "cuda" if (args.device == "cuda"
                             or (args.device == "auto" and _cuda_ok())) else "cpu"
    apply_hardware_profile(args, device_hint)
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
    # effective router-polish lr (always in scope: refine reuses it even when
    # stage 5 itself was skipped as cached)
    router_lr = args.router_lr if args.router_lr is not None else fit["fit_lr"]
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
                                    eval_vs_cache_disk, expert_basis_init,
                                    expert_means, field_accounting,
                                    find_moe_blocks, fit_field_module,
                                    generate_text, load_pairs_block,
                                    make_batches, polish_router_module,
                                    router_weight, save_pairs_block,
                                    write_field_artifact, FieldSparseMoe)
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
    T["profile"], T["io_cache"] = args.profile, args.io_cache

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
    pool_ks = not getattr(args, "pool_recalibrate", False)
    pool_mu = max(4096, 16 * args.rank, mp // 2) if mp else 0
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
        check_io_cache_fit(args, gguf_path)
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
    pairs = None            # (path, n) per block; built in every branch below
    full_cache = cache_is_complete(pool_dir, lp_dir, min_pairs=mp,
                                   keep_smaller=pool_ks, min_useful=pool_mu)
    pairs_cache = full_cache or cache_is_complete(pool_dir, lp_dir, min_pairs=mp,
                                                  pairs_only=True,
                                                  keep_smaller=pool_ks,
                                                  min_useful=pool_mu)
    # the pair pool alone (the expensive model-forward part); the init files
    # are cheap model-only passes and are rebuilt per block when missing
    pool_cache = pairs_cache or pool_is_complete(pool_dir, min_pairs=mp,
                                                 keep_smaller=pool_ks,
                                                 min_useful=pool_mu)
    base_pass = "base" in plan and not full_cache
    calib_pass = ("calibrate" in plan
                  and (not pool_cache
                       or not all(os.path.isfile(os.path.join(pool_dir,
                                                f"init_blk{i}.pt"))
                                  for i in range(pool_cache))))
    # 2026-09-05.5 SVD-init self-heal: the fit (default --fit-init svd)
    # needs fit_dir/init_svd_blk*.pt; they are missing when the pool was
    # built by a pre-SVD build or an earlier run silently fell back to
    # the random init (its per-block resume then skipped the SVD part
    # forever). Detect the gap HERE and force the streaming pass that
    # rebuilds the files from the real expert deltas - the fit starts
    # informed instead of blind, and a stale random-init fit is re-fitted
    # automatically (its signature changes). Cost: one expert-weights
    # read per block, no model forward.
    svd_heal = False
    if ("fit" in plan and args.fit_init == "svd" and not args.refresh_init
            and pool_cache):
        svd_heal = not all(os.path.isfile(os.path.join(
            fit_dir, f"init_svd_blk{i}.pt")) for i in range(pool_cache))
        if svd_heal:
            print(f"SVD init check: init_svd_blk*.pt missing for rank "
                  f"{args.rank} ({pool_cache} blocks) - a streaming pass "
                  f"will build them from the expert weights (~minutes: one "
                  f"expert read per block, no model forward). An existing "
                  f"random-init fit for this rank is re-fitted afterwards.",
                  flush=True)
            calib_pass = True  # force the streaming pass (experts on disk)
    refresh_only = False
    if args.refresh_init:
        if not pairs_cache:
            sys.exit("--refresh-init needs the calibration pool cache (run the "
                     "pipeline once first); nothing to refresh")
        if args.fit_init != "svd":
            sys.exit("--refresh-init only makes sense with --fit-init svd")
        refresh_only = True
        base_pass = False
        calib_pass = True     # force the streaming pass: experts come from disk
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
        if calib_pass and not base_pass and full_cache:
            # 2026-09-05.5 SVD-init heal pass: stage 3/base is fully
            # cached - take the base metrics + eval tokens from the cache
            # so the verify stage keeps its ppl delta (the model below is
            # loaded for the expert weights only, no forward passes)
            d = torch.load(os.path.join(pool_dir, "eval_tokens.pt"),
                           map_location="cpu")
            X, Y, eval_ids = d["X"], d["Y"], d["eval_ids"]
            base_m = base_metrics_from_cache(lp_dir, X, Y)
            print(f"BASE ({base_label()}) (from the log-prob cache): "
                  f"ppl {base_m['ppl']:.2f}", flush=True)
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
            if refresh_only:
                banner("STAGE 4 (--refresh-init) - rebuilding the per-rank SVD "
                       "init files from the streamed experts")
            elif pool_cache:
                banner("STAGE 4 - resuming calibration: the pair pool is on "
                       "disk, rebuilding the missing init files (STREAMING)")
            else:
                banner("STAGE 4 - calibration: pairs to disk + centroids "
                       "(STREAMING, weights - quantized GGUF)")
            blocks = find_moe_blocks(model)
            geoms = [block_geometry(b, cfg) for _, b in blocks]
            # art_meta.json is written FIRST (it was written LAST before
            # 2026-09-04.3: any interruption during the silent centroids+SVD
            # loop then invalidated the whole pair pool). It carries only the
            # block inventory, which is known right here - write it now so
            # every resume check below can trust the cache on disk.
            am_path = os.path.join(pool_dir, "art_meta.json")
            if not os.path.isfile(am_path):
                base_obj = model.model if stream else model
                _save_json_atomic(am_path, dict(
                    base_cls=type(base_obj).__name__,
                    router_cls=type(blocks[0][1].gate).__name__,
                    router_mod=type(blocks[0][1].gate).__module__,
                    n_layers=len(blocks),
                    block_names=[n for n, _ in blocks]))
            if refresh_only:
                pairs = []
                for i in range(pairs_cache):
                    p = os.path.join(pool_dir, f"pairs_blk{i}.pt")
                    pairs.append((p, load_pairs_block(p)[0].shape[0]))
            else:
                # pre-2026-09-04.3 runs wrote art_meta.json LAST: a kill in the
                # silent init loop left a COMPLETE pair pool without art_meta.
                # Adopt such orphans (full consecutive set + cap verified) so
                # the expensive collection is not redone one more time.
                pool_n = pool_cache
                if not pool_n:
                    orphan = 0
                    while os.path.isfile(os.path.join(pool_dir,
                                                      f"pairs_blk{orphan}.pt")):
                        orphan += 1
                    if orphan == len(blocks) and orphan > 0:
                        try:
                            x0 = load_pairs_block(os.path.join(
                                pool_dir, "pairs_blk0.pt"))[0]
                            if (mp == 0 or int(x0.shape[0]) >= mp
                                    or (pool_ks
                                        and int(x0.shape[0]) >= pool_mu)):
                                pool_n = orphan
                                print(f"adopting {pool_n} pairs_blk*.pt files "
                                      f"from an interrupted run (art_meta was "
                                      f"missing) - no re-collection", flush=True)
                        except Exception:  # noqa: BLE001
                            pass
                if pool_n:
                    pairs = []
                    for i in range(pool_n):
                        p = os.path.join(pool_dir, f"pairs_blk{i}.pt")
                        pairs.append((p, load_pairs_block(p)[0].shape[0]))
                    if pool_cache:
                        print(f"pair pool reused from cache: {len(pairs)} "
                              f"blocks (no re-collection)", flush=True)
                else:
                    print(f"MoE blocks: {len(blocks)}; first geometry: {geoms[0]}",
                          flush=True)
                    gen = torch.Generator().manual_seed(11)
                    batches = make_batches(calib_ids, args.calib_ctx,
                                           args.calib_bsz, args.calib_windows,
                                           device, gen)
                    pairs = collect_pairs(model, blocks, batches,
                                          args.per_layer_cap, flush_dir=pool_dir)
                    for i, (p, n) in enumerate(pairs):
                        print(f"  block {i}: {n} pairs -> "
                              f"{os.path.basename(p) if p else 'EMPTY!'}",
                              flush=True)
                        if n == 0:
                            sys.exit("no pairs for a block - increase "
                                     "--calib-windows / text")
            os.makedirs(fit_dir, exist_ok=True)

            def _save_svd_init(i, block, mgu, mdn):
                basis = expert_basis_init(block, mgu, mdn, args.rank,
                                          log_prefix=f"block {i}/{len(blocks)}")
                ftmp = os.path.join(fit_dir, f"init_svd_blk{i}.pt.tmp")
                torch.save(basis, ftmp)   # atomic: no torn svd-init file
                os.replace(ftmp, os.path.join(fit_dir, f"init_svd_blk{i}.pt"))
                return 1

            svd_done = 0
            inits_reused = 0
            t_stage4 = time.time()
            for i, (_, block) in enumerate(blocks):
                ipath = os.path.join(pool_dir, f"init_blk{i}.pt")
                spath = os.path.join(fit_dir, f"init_svd_blk{i}.pt")
                svd_missing = (args.fit_init == "svd"
                               and not os.path.isfile(spath))
                if refresh_only and not svd_missing:
                    # resumable --refresh-init: existing files are kept
                    # (delete fit_dir/init_svd_blk*.pt to force a rebuild)
                    print(f"  block {i}/{len(blocks)}: init_svd_blk{i}.pt "
                          f"already exists - skipped", flush=True)
                    continue
                cached = (not refresh_only and os.path.isfile(ipath))
                if cached and not svd_missing:
                    # resumable stage 4: this block's init survived the last
                    # run (and its SVD init exists or is not wanted) - inits
                    # depend only on the model, never recompute
                    inits_reused += 1
                    print(f"  block {i}/{len(blocks)}: init cached - skipped",
                          flush=True)
                    continue
                t0 = time.time()
                # 2026-09-05.5: a cached init_blk also feeds the SVD-init
                # heal (its centroids are reusable) - no expert_means re-run
                ini = (torch.load(ipath, map_location="cpu")
                       if (refresh_only or cached) else None)
                print(f"  block {i}/{len(blocks)}: "
                      f"{'cached init + ' if cached else ''}"
                      f"{'SVD init' if args.fit_init == 'svd' else 'init'} "
                      f"(rank {args.rank})...", flush=True)
                if stream:
                    with stream.with_block(i):
                        if ini is not None:      # cached centroids (heal)
                            mgu, mdn = ini["mgu"], ini["mdn"]
                        else:
                            mgu, mdn = expert_means(block)
                        if svd_missing:
                            svd_done += _save_svd_init(i, block, mgu, mdn)
                else:
                    if ini is not None:          # cached centroids (heal)
                        mgu, mdn = ini["mgu"], ini["mdn"]
                    else:
                        mgu, mdn = expert_means(block)  # without an fp32 expert stack
                    if svd_missing:
                        svd_done += _save_svd_init(i, block, mgu, mdn)
                if refresh_only:
                    continue
                if cached:
                    # heal: the init file itself is complete - only the
                    # SVD part was missing
                    inits_reused += 1
                    print(f"  block {i}/{len(blocks)}: init cached, SVD "
                          f"init built in {time.time() - t0:.1f}s",
                          flush=True)
                    continue
                # hy_v3: selection bias + shared experts (backbone already in RAM)
                eb = block_router_bias(block)
                sh = block_shared_weights(block)
                if eb is not None or sh is not None:
                    print(f"  block {i}: bias {'yes' if eb is not None else 'no'}, "
                          f"shared {'yes' if sh is not None else 'no'}", flush=True)
                itmp = ipath + ".tmp"
                torch.save(dict(geom=geoms[i], gw=router_weight(block.gate).clone(),
                                mgu=mgu, mdn=mdn, eb=eb, shared=sh), itmp)
                os.replace(itmp, ipath)
                print(f"  block {i}/{len(blocks)}: done in "
                      f"{time.time() - t0:.1f}s", flush=True)
            if refresh_only:
                print(f"refresh-init done: {svd_done}/{len(blocks)} SVD init "
                      f"files written to {fit_dir}"
                      + ("" if svd_done == len(blocks) else
                         " (the rest already exist)")
                      + " - now re-run the fit "
                        "(the old fits are invalidated automatically)",
                      flush=True)
                if stream:
                    stream.close()
                release_model(model, device)
                sys.exit(0)
            print(f"stage 4 init files: {len(blocks) - inits_reused} rebuilt, "
                  f"{inits_reused} reused "
                  f"({time.time() - t_stage4:.1f}s total)", flush=True)

        print("unloading the backbone - the fit runs without it", flush=True)
        if stream:
            stream.close()
        release_model(model, device)
        stream = None
    T["stream_mode"] = bool(stream)
    T["reused_cache"] = bool(pairs_cache) and not calib_pass
    if pairs is None:
        # the base pass re-ran (e.g. the lp cache was invalidated) while the
        # pair pool + inits were already complete: stage 4 was skipped, so the
        # pairs list is built from the cache here
        pairs = []
        for i in range(pairs_cache):
            p = os.path.join(pool_dir, f"pairs_blk{i}.pt")
            pairs.append((p, load_pairs_block(p)[0].shape[0]))

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
               f"({args.fit_method}, "
               f"{T.get('fit_init_effective', args.fit_init)} init, "
               f"model NOT in RAM)")
        os.makedirs(fit_dir, exist_ok=True)
        svd_files = [os.path.join(fit_dir, f"init_svd_blk{i}.pt")
                     for i in range(n_blocks)]
        svd_ready = (args.fit_init == "svd"
                     and all(os.path.isfile(f) for f in svd_files))
        T["fit_init_effective"] = "svd" if svd_ready else "random"
        if args.fit_init == "svd" and not svd_ready:
            print("WARNING: --fit-init svd but init_svd_blk*.pt are missing "
                  "for this rank - fitting from the RANDOM init (~3x more "
                  "steps for the same quality; UPDATE-10). Since 2026-09-05.5 "
                  "the pipeline self-heals this before the fit - if you still "
                  "see this WARNING, check the log above for a failed "
                  "streaming pass.", flush=True)
        fit_sig = dict(fit_steps=args.fit_steps, fit_bs=args.fit_bs, fit_lr=args.fit_lr,
                       fit_method=args.fit_method, fit_jitter=args.fit_jitter,
                       fit_early_stop=args.fit_early_stop,
                       fit_router=args.fit_router,
                       router_steps=args.router_steps,
                       router_lr=(args.router_lr if args.router_lr is not None
                                  else fit["fit_lr"]),
                       router_anchor=args.router_anchor,
                       guard_warmup=args.fit_guard_warmup,
                       strict_guard=args.strict_fit_guard,
                       lr_warmup=args.fit_lr_warmup,
                       autocast=args.fit_autocast,
                       muon_max_dim=args.muon_max_dim,
                       muon_ns_steps=args.muon_ns_steps,
                       init=("svd" if svd_ready else "random"),
                       preset=fit_preset or "none")
        fit_meta_p = os.path.join(fit_dir, "fit_meta.json")
        fit_done = fit_blocks_ok(fit_dir, n_blocks) and os.path.isfile(fit_meta_p)
        if fit_done:
            with open(fit_meta_p, encoding="utf-8") as f:
                fit_done = json.load(f) == fit_sig
        # per-block resume: fit_partial.json carries {sig, mse{i}} for the
        # blocks whose fit_blk{i}.pt was fully written by an interrupted run;
        # only the missing blocks are re-fitted (a fit of hundreds of steps
        # per block must never restart from block 0 after a kill)
        part_mse = {}
        if not fit_done:
            partial_p = os.path.join(fit_dir, "fit_partial.json")

            def _blk_ok(i):
                p = os.path.join(fit_dir, f"fit_blk{i}.pt")
                return os.path.isfile(p) and os.path.getsize(p) > 0

            if os.path.isfile(partial_p):
                try:
                    with open(partial_p, encoding="utf-8") as f:
                        pj = json.load(f)
                    if pj.get("sig") == fit_sig and isinstance(pj.get("mse"), dict):
                        part_mse = {int(k): v for k, v in pj["mse"].items()
                                    if _blk_ok(int(k))}
                        if part_mse:
                            print(f"fit resume: {len(part_mse)}/{n_blocks} "
                                  f"blocks were already fitted with the same "
                                  f"settings - reusing their fit_blk*.pt",
                                  flush=True)
                except Exception:  # noqa: BLE001
                    part_mse = {}
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
                fit_init = dict(ini)
                if svd_ready:                       # hole-1: real-delta SVD init
                    fit_init.update(torch.load(svd_files[i], map_location="cpu"))
                fit_mod = FieldSparseMoe(ini["geom"], args.rank, gate_w=ini["gw"],
                                         act_fn=act, gate_bias=ini.get("eb"),
                                         shared=ini.get("shared"),
                                         init=fit_init).to(device)
                with torch.no_grad():
                    fit_mod.wgud.copy_(ini["mgu"])
                    fit_mod.wdnd.copy_(ini["mdn"])
                mse = fit_field_module(fit_mod, Xi, Yi, args.fit_steps, args.fit_bs,
                                       args.fit_lr, device,
                                       log_prefix=f"block {i}/{n_blocks}",
                                       guard=not args.skip_fit_guard,
                                       method=args.fit_method, seed=5 + i,
                                       jitter=args.fit_jitter,
                                       early_stop=args.fit_early_stop,
                                       guard_warmup=(None
                                                     if args.fit_guard_warmup < 0
                                                     else args.fit_guard_warmup),
                                       strict_guard=args.strict_fit_guard,
                                       lr_warmup=args.fit_lr_warmup,
                                       train_router=(args.fit_router == "joint"),
                                       router_anchor=args.router_anchor,
                                       autocast=args.fit_autocast,
                                       muon_max_dim=args.muon_max_dim,
                                       muon_ns_steps=args.muon_ns_steps)
                if args.fit_router == "after":
                    pr_stat = polish_router_module(
                        fit_mod, Xi, Yi, args.router_steps, args.fit_bs,
                        router_lr, device, anchor=args.router_anchor,
                        log_prefix=f"block {i}/{n_blocks} router", seed=5 + i)
                    router_stats[i] = pr_stat
                elif args.fit_router == "joint":
                    router_stats[i] = dict(drift=float(
                        (fit_mod.gw.detach() - ini["gw"].float()).norm()
                        / ini["gw"].float().norm().clamp_min(1e-12)))
                out = {n: getattr(fit_mod, n).detach().clone()
                       for n in fit_mod.field_names}
                if args.fit_router != "off":
                    out["gw_tuned"] = fit_mod.gw.detach().clone()
                fpath = os.path.join(fit_dir, f"fit_blk{i}.pt")
                ftmp = fpath + ".tmp"          # atomic: no torn fit file
                torch.save(out, ftmp)
                os.replace(ftmp, fpath)
                with lk:                        # crash-safe per-block resume
                    part_mse[i] = mse
                    _save_json_atomic(partial_p, {"sig": fit_sig,
                                                  "mse": {str(k): v
                                                          for k, v
                                                          in part_mse.items()}})
                del fit_mod, Xi, Yi
                gc.collect()
                return mse

            fit_mses = [part_mse.get(i) for i in range(n_blocks)]
            router_stats = {}
            errors = []
            lk = threading.Lock()
            todo = [i for i in range(n_blocks)
                    if part_mse.get(i) is None
                    or not os.path.isfile(os.path.join(fit_dir,
                                                       f"fit_blk{i}.pt"))]
            if len(todo) < n_blocks:
                for i in range(n_blocks):
                    if i not in todo:
                        print(f"  block {i}/{n_blocks}: fit cached (resume) "
                              f"- skipped", flush=True)
            w = max(1, min(int(args.fit_workers), len(todo) or 1))
            if w > 1 and len(todo) > 1:
                if not args.threads:
                    per = max(1, (os.cpu_count() or 2) // w)
                    torch.set_num_threads(per)
                    print(f"parallel fit: {w} workers x {per} cpu threads "
                          f"(--fit-workers)", flush=True)
                work = queue.Queue()
                for i in todo:
                    work.put(i)

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
                for i in todo:
                    fit_mses[i] = fit_one(i)
            _save_json_atomic(os.path.join(fit_dir, "mse.json"), fit_mses)
            if router_stats:
                _save_json_atomic(os.path.join(fit_dir, "router_meta.json"),
                                  router_stats)
            _save_json_atomic(fit_meta_p, fit_sig)
            try:                            # the fit is complete: partial
                os.remove(partial_p)        # resume bookkeeping is done
            except OSError:
                pass


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

            def _fit_state_sha():
                # 2026-09-05.5: refine caches depend on the fit they
                # warm-start from - the captured pairs are the FIELD's own
                # forward outputs and change with every re-fit.
                # fit_meta.json is rewritten exactly when the fit (re)runs
                # (refine itself never touches it), so its hash invalidates
                # stale refine rounds after a re-fit.
                p = os.path.join(fit_dir, "fit_meta.json")
                try:
                    with open(p, "rb") as f:
                        return hashlib.sha1(f.read()).hexdigest()[:16]
                except OSError:
                    return "nofit"
            for rnd in range(1, args.refine_rounds + 1):
                banner(f"STAGE 5b.{rnd} - refine round: the field feeds forward, "
                       f"the original experts teach")
                try:
                    # 2026-09-05.4: refine RESUME - the round is identified by
                    # a signature of its settings; a completed round is
                    # skipped, a half-done one reuses the captured pairs (the
                    # capture is the expensive half: a full streaming pass).
                    # --refresh-refine ignores both caches.
                    rtag = dict(stamp="2026-09-05.5", rank=args.rank, rnd=rnd,
                                fit_state=_fit_state_sha(),
                                per_layer_cap=args.per_layer_cap,
                                calib_ctx=args.calib_ctx,
                                calib_bsz=args.calib_bsz,
                                calib_windows=args.calib_windows,
                                fit_steps=args.fit_steps, fit_bs=args.fit_bs,
                                fit_lr=args.fit_lr,
                                fit_method=args.fit_method,
                                fit_jitter=args.fit_jitter,
                                fit_early_stop=args.fit_early_stop,
                                fit_guard_warmup=args.fit_guard_warmup,
                                fit_lr_warmup=args.fit_lr_warmup,
                                fit_router=args.fit_router,
                                router_anchor=args.router_anchor,
                                router_steps=args.router_steps,
                                router_lr=router_lr,
                                strict_fit_guard=args.strict_fit_guard,
                                fit_autocast=args.fit_autocast,
                                muon_max_dim=args.muon_max_dim,
                                muon_ns_steps=args.muon_ns_steps)
                    r_sig = hashlib.sha256(json.dumps(
                        rtag, sort_keys=True, default=str).encode()).hexdigest()[:16]
                    done_p = os.path.join(refit_dir, f"done_r{rnd}.json")
                    if not args.refresh_refine and os.path.isfile(done_p):
                        try:
                            with open(done_p, encoding="utf-8") as f:
                                dmk = json.load(f)
                        except Exception:  # noqa: BLE001 - torn marker
                            dmk = None
                        if dmk and dmk.get("sig") == r_sig and dmk.get("mse"):
                            fit_mses = [float(v) for v in dmk["mse"]]
                            T["refine_rounds"] = max(
                                int(T.get("refine_rounds") or 0), rnd)
                            print(f"  refine round {rnd}: cached (resume) - "
                                  f"skipped (--refresh-refine to force redo)",
                                  flush=True)
                            continue
                    psig_p = os.path.join(refit_dir, "pairs_sig.json")
                    reuse_pairs = False
                    if not args.refresh_refine and os.path.isfile(psig_p):
                        try:
                            with open(psig_p, encoding="utf-8") as f:
                                ps = json.load(f)
                            reuse_pairs = bool(
                                ps.get("sig") == r_sig and ps.get("files")
                                and all(os.path.isfile(os.path.join(refit_dir, fn))
                                        for fn in ps["files"]))
                        except Exception:  # noqa: BLE001 - torn sig
                            reuse_pairs = False
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
                    if reuse_pairs:
                        n_pf = len(_glob.glob(os.path.join(
                            refit_dir, "pairs_blk*_p*.pt")))
                        print(f"  refine pairs: {n_pf} chunk files reused "
                              f"from the cache - capture skipped", flush=True)
                    else:
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

                        chunk_flush_at = refine_flush_at(ram_gb(available=True))
                        if chunk_flush_at != 8192:
                            print(f"refine capture: flushing pair chunks at "
                                  f"{chunk_flush_at} pairs (low RAM - smaller "
                                  f"disk chunks, resident ~= one batch ~0.4 GB "
                                  f"instead of ~1.1 GB)", flush=True)

                        def flush_chunk(i, force=False):
                            # flush a chunk once it is big enough (bounded RAM),
                            # or whatever remains at the end of the pass
                            if chunk_n[i] and (force or chunk_n[i] >= chunk_flush_at):
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
                        n_windows2 = len(batches2)
                        for wi, batch in enumerate(batches2, 1):
                            runner2(**batch)
                            for i in range(n_blocks):
                                flush_chunk(i)
                            # 2026-09-05.2: per-window progress - in the wild this
                            # pass froze silently (swap-storm); a moving line
                            # makes "working" vs "stuck" distinguishable.
                            print(f"\r    ... refine capture: window {wi}/"
                                  f"{n_windows2}, pairs {min(chunk_n)}.."
                                  f"{max(chunk_n)} of {args.per_layer_cap}".
                                  ljust(100), end="", flush=True)
                            if all(chunk_n[i] >= args.per_layer_cap for i in range(n_blocks)):
                                break
                        print()
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
                        psig_files = sorted(
                            os.path.basename(p) for p in _glob.glob(
                                os.path.join(refit_dir, "pairs_blk*_p*.pt")))
                        with open(psig_p + ".tmp", "w", encoding="utf-8") as f:
                            json.dump({"sig": r_sig, "files": psig_files}, f)
                        os.replace(psig_p + ".tmp", psig_p)
                    # 3) warm-started refit on the refine pairs
                    # 2026-09-05.4: blocks are independent - the refit runs
                    # through the same worker pool as the stage-5a fit
                    # (--fit-workers; before this it was always one worker)
                    ref_mses = [None] * n_blocks
                    router_stats = {}
                    errors = []
                    lk = threading.Lock()

                    def refit_one(i):
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
                        if "gw_tuned" in prev:
                            with torch.no_grad():   # keep the tuned router
                                fit_mod.gw.copy_(prev["gw_tuned"])
                        mse = fit_field_module(fit_mod, Xi, Yi, args.fit_steps,
                                               args.fit_bs, args.fit_lr, device,
                                               log_prefix=f"block {i}/{n_blocks} r{rnd}",
                                               guard=not args.skip_fit_guard,
                                               method=args.fit_method,
                                               seed=1000 * rnd + 5 + i,
                                               jitter=args.fit_jitter,
                                               early_stop=args.fit_early_stop,
                                               guard_warmup=(None
                                                             if args.fit_guard_warmup < 0
                                                             else args.fit_guard_warmup),
                                               strict_guard=args.strict_fit_guard,
                                               lr_warmup=args.fit_lr_warmup,
                                               train_router=(args.fit_router == "joint"),
                                               router_anchor=args.router_anchor,
                                               autocast=args.fit_autocast,
                                               muon_max_dim=args.muon_max_dim,
                                               muon_ns_steps=args.muon_ns_steps)
                        if args.fit_router == "after":
                            pr_stat = polish_router_module(
                                fit_mod, Xi, Yi, args.router_steps, args.fit_bs,
                                router_lr, device, anchor=args.router_anchor,
                                log_prefix=f"block {i}/{n_blocks} r{rnd} router",
                                seed=1000 * rnd + 5 + i)
                            router_stats[i] = pr_stat
                        elif args.fit_router == "joint":
                            router_stats[i] = dict(drift=float(
                                (fit_mod.gw.detach() - ini["gw"].float()).norm()
                                / ini["gw"].float().norm().clamp_min(1e-12)))
                        out = {n: getattr(fit_mod, n).detach().clone()
                               for n in fit_mod.field_names}
                        if args.fit_router != "off":
                            out["gw_tuned"] = fit_mod.gw.detach().clone()
                        torch.save(out, os.path.join(fit_dir, f"fit_blk{i}.pt"))
                        if args.fit_router == "after":
                            router_stats[i] = pr_stat
                        elif args.fit_router == "joint":
                            router_stats[i] = dict(drift=float(
                                (fit_mod.gw.detach() - ini["gw"].float()).norm()
                                / ini["gw"].float().norm().clamp_min(1e-12)))
                        del fit_mod, Xi, Yi
                        gc.collect()
                        return mse

                    w = max(1, min(int(args.fit_workers), n_blocks or 1))
                    if w > 1 and n_blocks > 1:
                        if not args.threads:
                            per = max(1, (os.cpu_count() or 2) // w)
                            torch.set_num_threads(per)
                            print(f"refine refit: {w} workers x {per} cpu "
                                  f"threads (--fit-workers)", flush=True)
                        work = queue.Queue()
                        for i in range(n_blocks):
                            work.put(i)

                        def refit_worker():
                            while True:
                                try:
                                    i = work.get_nowait()
                                except queue.Empty:
                                    return
                                try:
                                    mse = refit_one(i)
                                    ref_mses[i] = mse
                                except Exception as e:  # noqa: BLE001
                                    with lk:
                                        errors.append((i, e))

                        ths = [threading.Thread(target=refit_worker,
                                                name=f"refit-{k}",
                                                daemon=True) for k in range(w)]
                        for t in ths:
                            t.start()
                        for t in ths:
                            t.join()
                        if errors:
                            i, e = errors[0]
                            raise RuntimeError(
                                f"refine refit failed on block {i}: {e}") from e
                    else:
                        for i in range(n_blocks):
                            ref_mses[i] = refit_one(i)
                    fit_mses = ref_mses
                    with open(os.path.join(fit_dir, "mse.json"), "w",
                              encoding="utf-8") as f:
                        json.dump(fit_mses, f)
                    if router_stats:
                        with open(os.path.join(fit_dir, "router_meta.json"),
                                  "w", encoding="utf-8") as f:
                            json.dump(router_stats, f)
                    with open(done_p + ".tmp", "w", encoding="utf-8") as f:
                        json.dump({"sig": r_sig, "mse": fit_mses}, f)
                    os.replace(done_p + ".tmp", done_p)
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
                       fit_router=args.fit_router,
                       fit_guard_warmup=args.fit_guard_warmup,
                       fit_lr_warmup=args.fit_lr_warmup,
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
        if X is None or Y is None:
            # resumed runs: phase A loaded the eval tokens only when the whole
            # cache was complete - the verify math needs just the tokens, and
            # they are deterministic (fixed-seed chunks saved on disk)
            d = torch.load(os.path.join(pool_dir, "eval_tokens.pt"),
                           map_location="cpu")
            X, Y, eval_ids = d["X"], d["Y"], d.get("eval_ids", eval_ids)
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
                  f"{pr.get('fit_workers')} | prefetch {pr.get('prefetch')}"
                  + (f" | router: {pr.get('fit_router')}"
                     if pr.get("fit_router", "off") != "off" else "")
                  + (f" | guard warmup {pr.get('fit_guard_warmup')}"
                     if pr.get("fit_guard_warmup") not in (None, -1) else "")
                  + (f" | lr warmup {pr.get('fit_lr_warmup')}"
                     if pr.get("fit_lr_warmup") else ""))
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
