"""Fit a sampling temperature that brings the compressed model's distribution
back to the base model's confidence level.

Field compression flattens the output distribution slightly (experts are
averaged/low-rank corrected, so confident logits lose a bit of margin). The
KL number (0.757 bits/token) measures that blur, but generation suffers more
than the average suggests: a flattened distribution samples long-tail tokens
too often, the reply drifts into a "neighbour mode" (for OLMoE - an archaic /
poetry register), and the drift compounds over the dialog turns.

A single scalar T < 1 re-sharpens the logits. This tool fits T by minimizing
KL(base || softmax(field_logits / T)) over the same kind of eval chunks the
pipeline uses (seeded windows over the eval tail - same split, same seed).

The base model is STREAMED from the GGUF (never fully loaded), so the tool
runs fine on Colab next to the compression itself.

Usage:
  python3 temp_calibrate.py --model results/field_xxx --gguf /content/model.Q4_0.gguf
  python3 temp_calibrate.py --model ... --gguf ... --calib-file corpus.txt --chunks 8

Output:
  <artifact>/sampling.json - picked up automatically by hf_chat.py
"""

import argparse
import datetime
import json
import math
import os
import sys

import hf_env  # noqa: F401  - HF cache inside the project; BEFORE transformers

import torch

BASE = os.path.dirname(os.path.abspath(__file__))


def resolve_eval_text(args):
    """Same text rules as hf_pipeline.resolve_texts: --calib-file, wikitext,
    or the local fallback corpus; eval = the tail 10% (leak fix)."""
    text = None
    if args.calib_file:
        text = open(args.calib_file, encoding="utf-8", errors="ignore").read()
    elif args.calib_dataset:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", args.calib_dataset,
                          split="train" if "raw" in args.calib_dataset else "train")
        text = "\n".join(t for t in ds["text"] if t.strip())
    else:
        for name in ("corpus.txt", "corpus_raw.txt", "corpus_ru.txt"):
            p = os.path.join(BASE, name)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                text = open(p, encoding="utf-8", errors="ignore").read()
                print(f"calibration text: local {name}", flush=True)
                break
        if text is None:
            sys.exit("no calibration text: pass --calib-file <txt> or "
                     "--calib-dataset wikitext-2-raw-v1")
    k = max(1, int(len(text) * 0.9))
    evalt = text[k:]                       # tail 10% - same as the pipeline
    if len(evalt.strip()) < 2000:
        sys.exit(f"eval tail is too small ({len(evalt)} chars); pass a bigger "
                 f"--calib-file (the pipeline split needs >= ~2k chars)")
    return evalt


def sample_windows(ids, ctx, n_chunks, seed=17):
    """Same windowing as hf_field_transform.eval_logits_cache_disk (seed 17)."""
    g = torch.Generator().manual_seed(seed)
    starts = []
    for _ in range(n_chunks):
        starts.append(int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g)))
    return starts


@torch.no_grad()
def collect_logits(model_fn, ids, starts, ctx, device):
    """Forward each window, return float32 log-softmax rows [N*ctx, V] (CPU)."""
    rows = []
    for s in starts:
        x = ids[s:s + ctx].unsqueeze(0).to(device)
        logits = model_fn(x).float().cpu()
        rows.append(torch.log_softmax(logits, dim=-1))
    return torch.cat(rows, dim=0)


def fit_temperature(lp_base, lp_field, tmin, tmax, step):
    """Grid-search T minimizing KL(p_base || softmax(field/T)); also returns
    entropy / agreement diagnostics."""
    p = lp_base.exp()
    lpb = lp_base
    ent_base = float(-(p * lpb).sum(-1).mean())
    # top-1 agreement (T-invariant, structural diagnostic)
    agree = float((lp_base.argmax(-1) == lp_field.argmax(-1)).float().mean())

    grid = torch.arange(tmin, tmax + 1e-9, step).tolist()
    best_t, best_kl, rows = 1.0, None, []
    for T in grid:
        q = torch.log_softmax(lp_field / T, dim=-1)
        kl = float((p * (lpb - q)).sum(-1).mean())
        ent = float(-(q.exp() * q).sum(-1).mean())
        rows.append((T, kl, ent))
        if best_kl is None or kl < best_kl:
            best_t, best_kl = T, kl
    q1 = torch.log_softmax(lp_field, dim=-1)
    kl_at_1 = float((p * (lpb - q1)).sum(-1).mean())
    ent_at_1 = float(-(q1.exp() * q1).sum(-1).mean())
    ent_best = {t: e for t, _, e in rows}[best_t]
    return dict(temperature=best_t, kl_at_1=kl_at_1, kl_best=best_kl,
                ent_base=ent_base, ent_field=ent_at_1, ent_field_T=ent_best,
                top1_agreement=agree, grid=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="artifact folder after hf_pipeline.py (default: newest field_*)")
    ap.add_argument("--gguf", required=True,
                    help="the same GGUF the artifact was built from (base is "
                         "streamed from it, never fully loaded)")
    ap.add_argument("--src", default=None,
                    help="config dir for the streamed base (default: "
                         "results/gguf_hf/<gguf basename>-hf)")
    ap.add_argument("--calib-file", default=None, help="corpus txt (same file as "
                    "during compression; eval tail 10%% is used)")
    ap.add_argument("--calib-dataset", default=None,
                    help="e.g. wikitext-2-raw-v1 (datasets)")
    ap.add_argument("--chunks", type=int, default=8, help="eval windows")
    ap.add_argument("--ctx", type=int, default=128, help="window length")
    ap.add_argument("--tmin", type=float, default=0.30)
    ap.add_argument("--tmax", type=float, default=1.50)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--io-threads", type=int, default=1, dest="io_threads",
                    help="GGUF dequant threads")
    ap.add_argument("--io-cache", default="ram", choices=["disk", "ram"],
                    help="ram = keep raw packed tensors in RAM (recommended)")
    ap.add_argument("--prefetch", type=int, default=1)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from hf_chat import load_model, pick_model

    path = pick_model(a.model)
    gguf = os.path.abspath(a.gguf)
    if not os.path.isfile(gguf):
        sys.exit(f"gguf not found: {gguf}")
    src = a.src or os.path.join(
        BASE, "results", "gguf_hf",
        os.path.basename(gguf).removesuffix(".gguf") + "-hf")
    if not os.path.isfile(os.path.join(src, "config.json")):
        sys.exit(f"config dir for the streamed base not found: {src}\n"
                 f"pass --src <gguf_hf conversion dir> (created by hf_pipeline.py)")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    dt = {"auto": torch.bfloat16, "bfloat16": torch.bfloat16,
          "float16": torch.float16, "float32": torch.float32}[a.dtype]

    tok = AutoTokenizer.from_pretrained(path)
    evalt = resolve_eval_text(a)
    ids = torch.tensor(tok(evalt)["input_ids"])
    if len(ids) <= a.ctx + 1:
        sys.exit(f"eval text is shorter than ctx={a.ctx}")
    starts = sample_windows(ids, a.ctx, a.chunks)
    print(f"eval windows: {a.chunks} x {a.ctx} tokens "
          f"(seed 17, same split as the pipeline)", flush=True)

    # ---- base (streamed from GGUF) -------------------------------------
    print(f"base pass (streaming: {gguf})", flush=True)
    from hf_stream import BlockStreamRunner
    runner = BlockStreamRunner(src, dtype=dt, device=dev, gguf=gguf,
                               prefetch=a.prefetch, io_workers=a.io_threads,
                               io_cache=a.io_cache)
    lp_base = collect_logits(lambda x: runner(input_ids=x).logits[0],
                             ids, starts, a.ctx, dev)
    runner.close()

    # ---- field artifact --------------------------------------------------
    print(f"field pass (artifact: {path})", flush=True)
    model, _, mdev = load_model(path, a.device if a.device != "auto" else
                                ("cuda" if dev == "cuda" else "cpu"), a.dtype)
    lp_field = collect_logits(lambda x: model(input_ids=x).logits[0],
                              ids, starts, a.ctx, mdev)
    del model

    # ---- fit --------------------------------------------------------------
    r = fit_temperature(lp_base, lp_field, a.tmin, a.tmax, a.step)
    T = r["temperature"]
    print("\nresult:", flush=True)
    print(f"  KL(base||field):   {r['kl_at_1']:.3f} bits/token  ->  "
          f"{r['kl_best']:.3f} bits/token at T={T:.2f}", flush=True)
    print(f"  entropy:  base {r['ent_base']:.3f} | field {r['ent_field']:.3f} "
          f"| field@T {r['ent_field_T']:.3f}  (lower = sharper; the base<->field "
          f"gap is the compression blur)", flush=True)
    print(f"  top-1 agreement:    {100 * r['top1_agreement']:.1f}% "
          f"(structural - temperature does not change it)", flush=True)
    if T >= a.tmax - 1e-9:
        print("  note: best T hit the grid ceiling - the field is SHARPER than "
              "the base; keep T=1.0 for sampling", flush=True)
        T = 1.0

    out = dict(temperature=T,
               kl_bits_at_1=round(r["kl_at_1"], 4),
               kl_bits_calibrated=round(r["kl_best"], 4),
               entropy_base=round(r["ent_base"], 4),
               entropy_field=round(r["ent_field"], 4),
               entropy_field_calibrated=round(r["ent_field_T"], 4),
               top1_agreement=round(r["top1_agreement"], 4),
               n_chunks=a.chunks, ctx=a.ctx,
               gguf=os.path.basename(gguf),
               created=datetime.datetime.now().isoformat(timespec="seconds"))
    dst = os.path.join(path, "sampling.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {dst}\nhf_chat.py will use temperature={T} automatically "
          f"(--temperature overrides it)", flush=True)


if __name__ == "__main__":
    main()
