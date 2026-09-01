#!/usr/bin/env python3
"""MoE -> field-engine pipeline: bootstrap -> transform -> verify.

On launch it FETCHES everything missing:
  * pip packages (torch, numpy, matplotlib);
  * a corpus (tinyshakespeare from GitHub / Gutenberg) - if no local texts;
  * the base-model checkpoint (trains via train.py) - if ckpt.pt is missing.

Stages:
  bootstrap   checks + auto-download of dependencies, corpus, checkpoint
  transform   converting experts into the field (field_eval.py): SVD anchors,
              fit, saving deploy artifacts, CSV/chart
  verify      a separate run: load the transformed model FROM THE ARTIFACT and
              verify (KL/ppl/memory/generation) -> moe_pipeline_report.md
  all         all stages in order (default)

Examples:
  python3 pipeline.py all                 # full run
  python3 pipeline.py all --smoke         # quick wiring run (60 steps, rank 8)
  python3 pipeline.py transform --ranks 8,16
  python3 pipeline.py verify --rank 8
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED = [("torch", "torch"), ("numpy", "numpy")]
OPTIONAL = [("matplotlib", "matplotlib")]

CORPUS_URLS = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
)

SUMMARY = {"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def banner(msg):
    print("\n" + "=" * 64 + f"\n== {msg}\n" + "=" * 64, flush=True)


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def pip_install(pkgs):
    print(f"installing packages: {', '.join(pkgs)}", flush=True)
    return subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs]) == 0


def ensure_deps(install_optional=True):
    need = [pip for mod, pip in REQUIRED if not have(mod)]
    if install_optional:
        need += [pip for mod, pip in OPTIONAL if not have(mod)]
    if need and not pip_install(need):
        print("pip failed - retrying without optional packages", flush=True)
        pip_install([pip for mod, pip in REQUIRED if not have(mod)])
    missing = [mod for mod, _ in REQUIRED if not have(mod)]
    if missing:
        print(f"CRITICAL: could not install {missing}; "
              f"manually: pip install {' '.join(pip for _, pip in REQUIRED)}", flush=True)
        sys.exit(1)
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    print(f"dependencies: torch {torch.__version__}, numpy {have('numpy')}, "
          f"matplotlib {have('matplotlib')}, threads {torch.get_num_threads()}", flush=True)


def ensure_corpus():
    for name in ("corpus.txt", "corpus_ru.txt", "corpus_raw.txt"):
        p = os.path.join(BASE, name)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print(f"corpus: have {name} ({os.path.getsize(p) / 1e6:.2f} MB)", flush=True)
            return True
    print("corpus not found - downloading...", flush=True)
    for url in CORPUS_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 100_000:
                raise RuntimeError(f"response too short: {len(data)} B")
            dst = os.path.join(BASE, "corpus_raw.txt")
            with open(dst, "wb") as f:
                f.write(data)
            print(f"downloaded {len(data) / 1e6:.2f} MB -> {dst}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001 - any network failure tries the next URL
            print(f"  {url}: {e}", flush=True)
    print("WARNING: download failed - common.py will substitute the built-in "
          "fallback text (the pipeline keeps going)", flush=True)
    return False


def ensure_ckpt(steps):
    ck = os.path.join(BASE, "ckpt.pt")
    if os.path.exists(ck):
        print(f"checkpoint: have ckpt.pt ({os.path.getsize(ck) / 1e6:.2f} MB)", flush=True)
        return True
    print(f"no checkpoint - training the base model ({steps} steps)...", flush=True)
    r = subprocess.call([sys.executable, "-u", "train.py", "--steps", str(steps)],
                        cwd=BASE)
    if r != 0 or not os.path.exists(ck):
        print("CRITICAL: training failed", flush=True)
        sys.exit(1)
    return True


def stage_bootstrap(args):
    t0 = time.time()
    banner("STAGE 0/2 - bootstrap: dependencies, corpus, checkpoint")
    ensure_deps(install_optional=not args.no_optional)
    corpus_ok = ensure_corpus()
    ckpt_ok = ensure_ckpt(60 if args.smoke else args.steps)
    from common import CFG  # smoke check of local modules
    print(f"model config: {CFG}", flush=True)
    SUMMARY["stages"]["bootstrap"] = dict(seconds=round(time.time() - t0, 1),
                                          corpus_ok=corpus_ok, ckpt_ok=ckpt_ok)


def stage_transform(args):
    t0 = time.time()
    banner("STAGE 1/2 - transform: converting experts into the field engine")
    ensure_deps(install_optional=False)
    ranks = args.ranks or ("8" if args.smoke else "8,16,32")
    fit_steps = 120 if args.smoke else 400
    cmd = [sys.executable, "-u", "field_eval.py",
           "--ranks", ranks, "--fit-steps", str(fit_steps)]
    print("running: " + " ".join(cmd), flush=True)
    if subprocess.call(cmd, cwd=BASE) != 0:
        print("CRITICAL: transform failed", flush=True)
        sys.exit(1)
    SUMMARY["stages"]["transform"] = dict(seconds=round(time.time() - t0, 1),
                                          ranks=ranks, fit_steps=fit_steps)


def stage_verify(args):
    t0 = time.time()
    banner("STAGE 2/2 - verify: checking the result on the transformed model")
    ensure_deps(install_optional=False)
    cmd = [sys.executable, "-u", "verify_transformed.py", "--artifacts-dir", OUT_DIR]
    if args.rank:
        cmd += ["--rank", str(args.rank)]
    print("running: " + " ".join(cmd), flush=True)
    if subprocess.call(cmd, cwd=BASE) != 0:
        print("CRITICAL: verify failed", flush=True)
        sys.exit(1)
    SUMMARY["stages"]["verify"] = dict(seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["bootstrap", "transform", "verify", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="quick wiring run: 60 training steps, rank 8, fit 120")
    ap.add_argument("--steps", type=int, default=1200,
                    help="training steps if there is no checkpoint (default 1200)")
    ap.add_argument("--ranks", default=None, help="field ranks for transform: 8,16,32")
    ap.add_argument("--rank", type=int, default=0,
                    help="rank for verify (0 = verify all artifacts)")
    ap.add_argument("--no-optional", action="store_true",
                    help="do not install matplotlib")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    if args.stage in ("bootstrap", "all"):
        stage_bootstrap(args)
    if args.stage in ("transform", "all"):
        stage_transform(args)
    if args.stage in ("verify", "all"):
        stage_verify(args)

    SUMMARY["total_seconds"] = round(time.time() - t0, 1)
    SUMMARY["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sp = os.path.join(OUT_DIR, "moe_pipeline_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2)
    banner("PIPELINE FINISHED")
    print(json.dumps(SUMMARY["stages"], ensure_ascii=False, indent=2), flush=True)
    print(f"summary -> {sp}", flush=True)
    print(f"report  -> {os.path.join(OUT_DIR, 'moe_pipeline_report.md')}", flush=True)


if __name__ == "__main__":
    main()
