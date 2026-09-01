"""Pipeline stage 3: INDEPENDENT verification on the transformed model.
Loads ONLY the compact field artifact (no explicit expert weights), builds the
deploy model via deploy.py and compares against the base:
  - KL(base || transformed), bits/token (the same eval sample as in the
    transformation);
  - perplexity and its change;
  - memory: claimed by formula / actual by artifact / file size;
  - text generation: base vs transformed (visual check).
Report: download/moe_pipeline_report.md + .json
Run: python3 verify_transformed.py [--rank 8] [--artifacts-dir DIR]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import torch

from common import CKPT, TinyMoE, prepare_data
from transform_eval import base_logits, eval_ce_kl, generate
from deploy import load_deployed, artifact_field_bytes, field_bytes_claimed, list_artifacts

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)
SEED_CHARS = 220


def full_expert_bytes(cfg):
    """Full size of the explicit experts (fp16): what we do NOT store."""
    d, dff, N, L = cfg["d_model"], cfg["d_ff"], cfg["n_exp"], cfg["n_layer"]
    return L * 2 * N * dff * d * 2


def load_csv_field_rows(path):
    """family=field rows from the transformation CSV - to cross-check the
    fp32 fit with the fp16 deploy. Accepts the legacy Russian label too."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["family"] in ("field", "поле"):
                r = int(row["variant"].split("=")[1])
                rows[r] = (float(row["kl_bits"]), float(row["ppl_delta"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default=OUT_DIR)
    ap.add_argument("--rank", type=int, default=0, help="verify a single rank (0 = all)")
    ap.add_argument("--report", default=os.path.join(OUT_DIR, "moe_pipeline_report.md"))
    args = ap.parse_args()

    t0 = time.time()
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos = ck["cfg"], ck["itos"]
    model_base = TinyMoE(cfg, len(itos))
    model_base.load_state_dict(ck["sd"])

    _, val_ids, _, _ = prepare_data()
    X, Y, LP = base_logits(model_base, val_ids)
    base_ce, _ = eval_ce_kl(model_base, X, Y, LP)
    base_ppl = math.exp(base_ce)
    print(f"BASE: ce={base_ce:.3f} ppl={base_ppl:.2f} "
          f"(checkpoint val_ppl={ck['val_ppl']:.2f})", flush=True)

    seed_ids = val_ids[:32]
    base_gen = generate(model_base, seed_ids, itos)
    full_mb = full_expert_bytes(cfg) / 1e6

    arts = list_artifacts(args.artifacts_dir)
    if args.rank:
        arts = [a for a in arts if a.endswith(f"_r{args.rank}.pt")]
    if not arts:
        sys.exit("Artifacts not found - run the transform stage first "
                 "(python3 pipeline.py transform)")
    print(f"Artifacts to verify: {len(arts)}\n", flush=True)

    csv_rows = load_csv_field_rows(os.path.join(args.artifacts_dir,
                                                "moe_field_results.csv"))
    results = []
    for path in arts:
        model_t, art = load_deployed(path)
        r = art["r"]
        ce, kl = eval_ce_kl(model_t, X, Y, LP)
        ppl = math.exp(ce)
        dpct = 100 * (ppl - base_ppl) / base_ppl
        claimed = field_bytes_claimed(cfg, r)
        actual = artifact_field_bytes(art)
        fsize = os.path.getsize(path)
        gen = generate(model_t, seed_ids, itos)
        same = sum(a == b for a, b in zip(gen, base_gen)) / max(len(base_gen), 1)
        row = dict(rank=r, artifact=os.path.basename(path),
                   claimed_mb=claimed / 1e6, actual_mb=actual / 1e6,
                   file_mb=fsize / 1e6, backbone_mb=(fsize - actual) / 1e6,
                   full_mb=full_mb, ratio=full_mb / max(claimed / 1e6, 1e-9),
                   kl_bits=kl, ppl=ppl, ppl_delta=dpct,
                   csv_kl=csv_rows.get(r, (None, None))[0],
                   csv_ppl_delta=csv_rows.get(r, (None, None))[1],
                   gen_match=100 * same, gen=gen)
        results.append(row)
        csv_note = (f" (CSV fp32 fit: KL {row['csv_kl']:.3f})"
                    if row["csv_kl"] is not None else "")
        print(f"[field r={r:2d}] field {row['claimed_mb']:.2f} MB (claimed) / "
              f"{row['actual_mb']:.2f} MB (actual), file {row['file_mb']:.2f} MB, "
              f"compression x{row['ratio']:.1f}", flush=True)
        print(f"             KL {kl:6.3f} bits/token{csv_note}  "
              f"ppl {ppl:6.2f} ({dpct:+.1f}%)  "
              f"generation match vs base {row['gen_match']:.0f}%", flush=True)

    report_md, report_json = build_report(args, base_ppl, base_ce, base_gen,
                                          results, len(val_ids))
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_md)
    jp = args.report.rsplit(".", 1)[0] + ".json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {args.report}\njson   -> {jp}", flush=True)
    print(f"verify took {time.time() - t0:.0f} s", flush=True)


def build_report(args, base_ppl, base_ce, base_gen, results, n_val):
    md = ["# Transformed model verification (field engine)", ""]
    md.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append("Stage: base MoE -> compact field (no explicit expert weights) -> "
              "loading from the artifact and verification.  ")
    md.append(f"Base: ppl **{base_ppl:.2f}** (ce {base_ce:.3f}), eval: {n_val} val tokens, "
              "120 chunks, protocol identical to the transformation.  ")
    md.append("Deploy: the field is stored in fp16, computed in fp32 "
              "(de-quantization at load).")
    md.append("")
    md.append("## Metrics on the transformed model")
    md.append("")
    md.append("| r | field, MB (claimed/actual) | expert compression | KL, bits/token | ppl | Δppl | "
              "fp32-fit KL (CSV) | gen. match |")
    md.append("|---|---|---|---|---|---|---|---|")
    for w in results:
        csv_cell = f"{w['csv_kl']:.3f}" if w["csv_kl"] is not None else "-"
        md.append(f"| {w['rank']} | {w['claimed_mb']:.2f} / {w['actual_mb']:.2f} | "
                  f"x{w['ratio']:.1f} | {w['kl_bits']:.3f} | {w['ppl']:.2f} | "
                  f"{w['ppl_delta']:+.1f}% | {csv_cell} | {w['gen_match']:.0f}% |")
    md.append("")
    md.append(f"Full size of the explicit experts (fp16): {results[0]['full_mb']:.2f} MB; "
              f"the r={results[0]['rank']} artifact file: {results[0]['file_mb']:.2f} MB "
              f"(of which backbone {results[0]['backbone_mb']:.2f} MB - the shared part "
              "of the model; in a real scenario it already exists and is not counted "
              "as compression).")
    md.append("")
    md.append("## Generation (greedy, same seed)")
    md.append("")
    md.append("**Base model (explicit experts):**")
    md.append("")
    md.append("```text")
    md.append(base_gen[:SEED_CHARS])
    md.append("```")
    for w in results:
        md.append(f"**Transformed (field r={w['rank']}, fp16 artifact):**")
        md.append("")
        md.append("```text")
        md.append(w["gen"][:SEED_CHARS])
        md.append("```")
    md.append("## Conclusion")
    md.append("")
    best = min(results, key=lambda w: w["kl_bits"])
    md.append(f"The transformed model deploys from the artifact without explicit "
              f"expert weights and works: the best rank r={best['rank']} gives KL "
              f"{best['kl_bits']:.3f} bits/token and Δppl {best['ppl_delta']:+.1f}% at "
              f"expert-memory compression x{best['ratio']:.1f}. The price of fp16 "
              f"storage vs the fp32 fit: "
              + (f"{best['kl_bits'] - best['csv_kl']:+.3f} bits/token by KL."
                 if best["csv_kl"] is not None else "no CSV data to cross-check."))
    md.append("")
    json_out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    base_ppl=base_ppl, base_ce=base_ce,
                    artifacts=[{k: v for k, v in w.items() if k != "gen"} for w in results],
                    generation_base=base_gen[:SEED_CHARS])
    return "\n".join(md), json_out


if __name__ == "__main__":
    main()
