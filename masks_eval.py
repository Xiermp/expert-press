"""The user's variant: an expert = a mask over the base layer.
Pareto-front comparison: SVD vs top-k mask vs ternary delta vs hybrid (SVD +
mask).
Run: python3 masks_eval.py
"""
import csv
import math
import os

import torch

from common import CFG, CKPT, TinyMoE, prepare_data
from transform_eval import base_logits, eval_ce_kl, precompute_svd, moe_weight_keys
from variants_eval import reconstruct_ranks, pca_variant

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)


def mask_variant(sd, cfg, density):
    """Expert = base + the top-k% largest deltas (fp16 value + uint16 index)."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = base_bytes = delta_bytes = 0.0
    for key in moe_weight_keys(cfg):
        W = sd[key]
        N = W.shape[0]
        C = W.mean(0, keepdim=True)
        D = W - C
        DH = D[0].numel()
        k = max(1, int(DH * density))
        full_bytes += D.numel() * 2
        base_bytes += C.numel() * 2
        for e in range(N):
            flat = D[e].reshape(-1)
            _, idx = torch.topk(flat.abs(), k)
            rec = torch.zeros_like(flat)
            rec[idx] = flat[idx]
            sd2[key][e] = C[0] + rec.reshape(W.shape[1], W.shape[2])
            delta_bytes += k * 4  # fp16 value (2B) + uint16 index (2B)
    return sd2, full_bytes, base_bytes + delta_bytes


def ternary_variant(sd, cfg, keep_p):
    """Delta -> {0, ±alpha}: 2 bits/weight, alpha = the mean magnitude of the
    kept entries."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = base_bytes = delta_bytes = 0.0
    for key in moe_weight_keys(cfg):
        W = sd[key]
        N = W.shape[0]
        C = W.mean(0, keepdim=True)
        D = W - C
        DH = D[0].numel()
        full_bytes += D.numel() * 2
        base_bytes += C.numel() * 2
        for e in range(N):
            flat = D[e].reshape(-1)
            tau = torch.quantile(flat.abs(), 1 - keep_p) if keep_p < 1.0 else 0.0
            m = flat.abs() > tau
            alpha = float(flat.abs()[m].mean()) if bool(m.any()) else 0.0
            rec = alpha * torch.sign(flat) * m.float()
            sd2[key][e] = C[0] + rec.reshape(W.shape[1], W.shape[2])
            delta_bytes += DH * 2 / 8 + 2  # 2 bits/weight + an fp16 scale
    return sd2, full_bytes, base_bytes + delta_bytes


def hybrid_variant(sd, cfg, factors, r, density):
    """Delta = an SVD of rank r + a top-q% mask on the residual (RoSA-style)."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = base_bytes = delta_bytes = 0.0
    for key in moe_weight_keys(cfg):
        C, D, facs = factors[key]
        N = C.shape[0]
        DH = D[0].numel()
        q = max(1, int(DH * density))
        full_bytes += D.numel() * 2
        base_bytes += C.numel() * 2
        for e, (U, S, Vh) in enumerate(facs):
            r_eff = min(r, S.numel())
            R = ((U[:, :r_eff] * S[:r_eff]) @ Vh[:r_eff, :]).reshape(D[e].shape)
            E = D[e] - R
            flat = E.reshape(-1)
            _, idx = torch.topk(flat.abs(), q)
            rec = R.clone().reshape(-1)
            rec[idx] += flat[idx]
            sd2[key][e] = C[0] + rec.reshape(W_shape(key, cfg))
            delta_bytes += r_eff * (U.shape[0] + U.shape[1]) * 2 + q * 4
    return sd2, full_bytes, base_bytes + delta_bytes


def W_shape(key, cfg):
    return cfg["d_ff"] if key.endswith("w1") else cfg["d_model"], \
           cfg["d_model"] if key.endswith("w1") else cfg["d_ff"]


def delta_tail_report(sd, cfg):
    """How heavy-tailed the deltas are: the share of energy in the top-1%
    weights."""
    print("\nDelta heavy-tailedness (the delta energy share in the top-1% "
          "weights):", flush=True)
    for key in moe_weight_keys(cfg):
        W = sd[key]
        D = W - W.mean(0, keepdim=True)
        shares = []
        for e in range(D.shape[0]):
            flat = D[e].reshape(-1)
            k = max(1, int(flat.numel() * 0.01))
            top = torch.topk(flat.abs(), k).values
            shares.append(float((top ** 2).sum() / max(float((flat ** 2).sum()), 1e-12)))
        print(f"  {key}: {100 * sum(shares) / len(shares):.1f}%", flush=True)


def main():
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos, sd = ck["cfg"], ck["itos"], ck["sd"]

    _, val_ids, _, _ = prepare_data()
    model = TinyMoE(cfg, len(itos))
    model.load_state_dict(sd)

    factors = precompute_svd(sd, cfg)
    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)
    print(f"BASE: ppl={base_ppl:.2f}", flush=True)

    delta_tail_report(sd, cfg)

    rows = []

    def add(family, label, sd2, fb, cb):
        model.load_state_dict(sd2)
        ce, kl = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        rows.append(dict(family=family, variant=label, mb=cb / 1e6,
                         ratio=fb / max(cb, 1), kl_bits=kl, ppl=ppl,
                         ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"[{family:8s}] {label:16s} mem {cb/1e6:5.2f} MB (x{fb/max(cb,1):4.1f}) "
              f"KL {kl:6.3f} bit  ppl {ppl:6.2f} ({100*(ppl-base_ppl)/base_ppl:+7.1f}%)",
              flush=True)

    # SVD baseline
    for r in (2, 4, 8, 16):
        sd2, fb, cb = reconstruct_ranks(sd, cfg, factors, r)
        add("SVD", f"r={r}", sd2, fb, cb)

    # top-k mask (the user's idea)
    for d in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        sd2, fb, cb = mask_variant(sd, cfg, d)
        add("mask", f"top-{int(d*100)}%", sd2, fb, cb)

    # ternary delta
    for p in (1.0, 0.5, 0.25, 0.1):
        sd2, fb, cb = ternary_variant(sd, cfg, p)
        add("ternary", f"keep-{int(p*100)}%", sd2, fb, cb)

    # hybrid SVD + residual mask
    for r, q in ((4, 0.10), (4, 0.05), (8, 0.05), (8, 0.02)):
        sd2, fb, cb = hybrid_variant(sd, cfg, factors, r, q)
        add("hybrid", f"r={r}+q={int(q*100)}%", sd2, fb, cb)

    # anchor: centroid only
    sd0, fb, cb = pca_variant(sd, cfg, 0)
    add("anchor", "centroid", sd0, fb, cb)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_masks_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {csv_path}", flush=True)

    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "SVD": dict(color="gray", marker="o"),
        "mask": dict(color="#B8860B", marker="s"),
        "ternary": dict(color="#8B0000", marker="^"),
        "hybrid": dict(color="#556B2F", marker="D"),
        "anchor": dict(color="black", marker="x"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    for fam, st in styles.items():
        pts = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["mb"])
        if not pts:
            continue
        ax.plot([p["mb"] for p in pts], [max(p["kl_bits"], 1e-7) for p in pts],
                linestyle="-" if fam in ("SVD", "mask") else "none",
                markersize=7, label=fam, **st)
        for p in pts:
            if fam in ("mask", "hybrid"):
                ax.annotate(f"x{p['ratio']:.0f}", (p["mb"], max(p["kl_bits"], 1e-7)),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color="#444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Full storage size (base + deltas), MB (log)")
    ax.set_ylabel("KL(base || variant), bits/token (log)")
    ax.set_title("Pareto front: mask vs SVD - the smaller the record, the better")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = os.path.join(OUT_DIR, "moe_masks_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    main()
