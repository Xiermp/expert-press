"""The expert field (the user's idea): no explicit experts - a continuous
field instead. The "engine" = base dynamics (centroid) + a low-rank additive
U-diag(c(z))-V^T, and the "movement seed" c(z) is a mixture of expert
coordinates by the router's soft weights: a token assembles its specialist on
the fly as a point in the field, in one FFN pass.

Variants:
  blend    - diagnostics: one-pass forward with WEIGHT mixing
             W(z)=sum_k w_k*W_e_k (full storage, but MoE FLOPs/2; checks the
             linearity of top-2 in the weights)
  field r=N - FieldMoE: centroid + a low-rank router-driven additive, fitted
             on real activations (MSE to the original MoE output)
Anchors: SVD r=8/16 (+ dense-MLP, wh-SVD, cascade from earlier CSVs on the
chart).
Run: python3 field_eval.py
"""
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, prepare_data3
from transform_eval import base_logits, eval_ce_kl, precompute_svd
from variants_eval import reconstruct_ranks, collect_moe_pairs
import deploy

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(11)


def soft_topk(probs, k):
    topw, topi = torch.topk(probs, k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    z = torch.zeros_like(probs).scatter_(-1, topi, topw)      # (B,T,N)
    return z


class BlendMoE(nn.Module):
    """Linearity diagnostics: one FFN pass with weights W(z)=sum_k w_k W_{e_k}."""

    def __init__(self, router, W1, W2):
        super().__init__()
        self.router = router
        self.W1 = W1                                          # (N, d_ff, d)
        self.W2 = W2                                          # (N, d, d_ff)
        self.k = CFG["top_k"]

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)
        z = soft_topk(probs, self.k)
        zw, zi = torch.topk(z, 2, dim=-1)                     # (B,T,2), descending
        e1, e2 = zi[..., 0:1], zi[..., 1:2]
        w1, w2 = zw[..., 0:1], zw[..., 1:2]
        W1z = w1.unsqueeze(-1) * self.W1[e1.squeeze(-1)] + \
              w2.unsqueeze(-1) * self.W1[e2.squeeze(-1)]                # (B,T,d_ff,d)
        W2z = w1.unsqueeze(-1) * self.W2[e1.squeeze(-1)] + \
              w2.unsqueeze(-1) * self.W2[e2.squeeze(-1)]                # (B,T,d,d_ff)
        h = F.gelu((x.unsqueeze(2) @ W1z.transpose(-1, -2)).squeeze(2))
        y = (h.unsqueeze(2) @ W2z.transpose(-1, -2)).squeeze(2)
        return y, probs


class FieldMoE(nn.Module):
    """Engine: centroid + U-diag(c(z))-V^T; c(z) = z @ C - coordinates in the
    field."""

    def __init__(self, router, w1d, w2d, U1, V1, U2, V2, C1, C2):
        super().__init__()
        self.router = router
        self.w1d, self.w2d = w1d, w2d                          # (d_ff,d), (d,d_ff)
        self.U1, self.V1, self.U2, self.V2 = U1, V1, U2, V2    # (d_ff,r),(d,r),(d,r),(d_ff,r)
        self.C1, self.C2 = C1, C2                              # (N,r)
        self.k = CFG["top_k"]

    def delta(self, acts, V, U, c):
        return (acts @ V * c) @ U.t()

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)
        z = soft_topk(probs, self.k)                           # (B,T,N)
        c1 = z @ self.C1                                       # (B,T,r) - movement seed
        c2 = z @ self.C2
        h = F.gelu(x @ self.w1d.t() + self.delta(x, self.V1, self.U1, c1))
        y = h @ self.w2d.t() + self.delta(h, self.V2, self.U2, c2)
        return y, probs


def fit_field(zin, yout, router_w, W1, W2, r, steps=400, bs=8192, lr=2e-3, seed=5):
    """Fit the field on real activations: MSE to the original MoE output
    (top-2, 2 passes)."""
    torch.manual_seed(seed)
    d, dff, N = CFG["d_model"], CFG["d_ff"], CFG["n_exp"]
    router = nn.Linear(d, N, bias=False)
    router.weight.data = router_w.clone()
    router.weight.requires_grad_(False)
    w1d = nn.Parameter(W1.mean(0).clone())                    # centroid start
    w2d = nn.Parameter(W2.mean(0).clone())
    U1 = nn.Parameter(torch.randn(dff, r) * 0.02)
    V1 = nn.Parameter(torch.randn(d, r) * 0.02)
    U2 = nn.Parameter(torch.randn(d, r) * 0.02)
    V2 = nn.Parameter(torch.randn(dff, r) * 0.02)
    C1 = nn.Parameter(torch.zeros(N, r))
    C2 = nn.Parameter(torch.zeros(N, r))
    mod = FieldMoE(router, w1d, w2d, U1, V1, U2, V2, C1, C2)
    opt = torch.optim.Adam([p for p in mod.parameters() if p.requires_grad], lr=lr)
    n = zin.shape[0]
    for s in range(steps):
        ix = torch.randint(0, n, (min(bs, n),))
        x, y = zin[ix], yout[ix]
        out, _ = mod(x)
        loss = F.mse_loss(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 100 == 0:
            print(f"    field r={r} step {s}: mse {loss.item():.5f}", flush=True)
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod


def field_bytes(cfg, r):
    d, dff, N, L = cfg["d_model"], cfg["d_ff"], cfg["n_exp"], cfg["n_layer"]
    per_matrix = dff * d + r * (dff + d) + N * r               # dense + U,V + coordinates
    return L * 2 * per_matrix * 2                              # fp16, w1+w2 per layer


def load_prev():
    extra = []
    spec = [
        (os.path.join(OUT_DIR, "moe_variants_results.csv"),
         lambda r: (r["variant"].startswith("dense-MLP"), r["variant"]), "dense-MLP"),
        (os.path.join(OUT_DIR, "moe_upgrade_results.csv"),
         lambda r: (r["family"] == "whitened" and r["variant"] in ("r=8", "r=16"),
                    f"wh-SVD {r['variant']}"), "wh-SVD"),
        (os.path.join(OUT_DIR, "moe_upgrade_results.csv"),
         lambda r: (r["family"] in ("cascade", "каскад"), r["variant"]), "cascade"),
        (os.path.join(OUT_DIR, "moe_bank_results.csv"),
         lambda r: (r["family"] == "PQ-W" and r["variant"] == "m=16", "PQ m=16"), "PQ-W"),
    ]
    for path, keep, fam in spec:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ok, label = keep(row)
                if ok:
                    extra.append(dict(family=fam, variant=label,
                                      mb=float(row["mb"]), ratio=float(row["ratio"]),
                                      kl_bits=float(row["kl_bits"]), ppl=float(row["ppl"]),
                                      ppl_delta=float(row["ppl_delta"])))
    return extra


def main(ranks=(8, 16, 32), fit_steps=400, save_dir=OUT_DIR):
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos, sd = ck["cfg"], ck["itos"], ck["sd"]

    _, calib_ids, val_ids, _, _ = prepare_data3()
    model = TinyMoE(cfg, len(itos))
    model.load_state_dict(sd)

    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)
    print(f"BASE: ppl={base_ppl:.2f}", flush=True)

    print("Collecting (MoE input -> MoE output) pairs on the calibration "
          "segment (a separate calib piece, eval untouched - leak fix)...",
          flush=True)
    pairs = collect_moe_pairs(model, calib_ids)

    original_moe = [b.moe for b in model.blocks]
    rows = []

    def add(family, label, cb, swapped=False, sd2=None):
        if not swapped:
            model.load_state_dict(sd2)
        ce, kl = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        fb = sum(sd[k].numel() for li in range(cfg["n_layer"])
                 for k in (f"blocks.{li}.moe.w1", f"blocks.{li}.moe.w2")) * 2
        rows.append(dict(family=family, variant=label, mb=cb / 1e6,
                         ratio=fb / max(cb, 1), kl_bits=kl, ppl=ppl,
                         ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"[{family:8s}] {label:10s} mem {cb / 1e6:5.2f} MB (x{fb / max(cb, 1):4.1f}) "
              f"KL {kl:6.3f} bit  ppl {ppl:6.2f} ({100 * (ppl - base_ppl) / base_ppl:+7.1f}%)",
              flush=True)
        for li, b in enumerate(model.blocks):
            b.moe = original_moe[li]

    # SVD anchors
    fac = precompute_svd(sd, cfg)
    for r in (8, 16):
        sd2, fb, cb = reconstruct_ranks(sd, cfg, fac, r)
        add("SVD", f"r={r}", cb, sd2=sd2)

    # 1) blend: weight mixing instead of function mixing (full storage, FLOPs/2)
    print("  blend forward (linearity diagnostics)...", flush=True)
    for li, b in enumerate(model.blocks):
        W1 = sd[f"blocks.{li}.moe.w1"]
        W2 = sd[f"blocks.{li}.moe.w2"]
        b.moe = BlendMoE(b.moe.router, W1, W2)
    full_cb = sum(sd[k].numel() for li in range(cfg["n_layer"])
                  for k in (f"blocks.{li}.moe.w1", f"blocks.{li}.moe.w2")) * 2
    add("blend", "top-2 weights", full_cb, swapped=True)

    # 2) field: centroid + a low-rank router-driven additive
    for r in ranks:
        print(f"  field fit r={r}...", flush=True)
        fmods = []
        for li, b in enumerate(model.blocks):
            router_w = sd[f"blocks.{li}.moe.router.weight"]
            W1 = sd[f"blocks.{li}.moe.w1"].float()
            W2 = sd[f"blocks.{li}.moe.w2"].float()
            mod = fit_field(pairs[li][0], pairs[li][1], router_w, W1, W2, r,
                            steps=fit_steps)
            b.moe = mod
            fmods.append(mod)
        art_path = os.path.join(save_dir, f"moe_transformed_field_r{r}.pt")
        deploy.save_deployed(art_path, cfg, itos, sd, fmods, r,
                             meta=dict(base_ppl=base_ppl, fit_steps=fit_steps))
        print(f"  artifact -> {art_path}", flush=True)
        add("field", f"r={r}", field_bytes(cfg, r), swapped=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_field_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {csv_path}", flush=True)

    try:
        plot(rows + load_prev())
    except ImportError:
        print("matplotlib unavailable - chart skipped", flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "SVD": dict(color="gray", marker="o"),
        "wh-SVD": dict(color="#B8860B", marker="s"),
        "dense-MLP": dict(color="#4B0082", marker="P"),
        "cascade": dict(color="#556B2F", marker="D"),
        "PQ-W": dict(color="#C2185B", marker="*"),
        "field": dict(color="#1565C0", marker="H"),
        "blend": dict(color="#00838F", marker="X"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    fig.suptitle("The expert field: the router assembles a specialist from a "
                 "continuous field")

    for fam, st in styles.items():
        pts = sorted([r for r in rows if r["family"] == fam], key=lambda p: p["mb"])
        if not pts:
            continue
        axes[0].plot([p["mb"] for p in pts], [max(p["kl_bits"], 1e-7) for p in pts],
                     linestyle="-" if fam in ("SVD", "wh-SVD", "field") else "none",
                     markersize=9, label=fam, **st)
        axes[1].plot([p["mb"] for p in pts], [p["ppl_delta"] for p in pts],
                     linestyle="none", markersize=9, label=fam, **st)
        for p in pts:
            axes[0].annotate(f"x{p['ratio']:.1f}", (p["mb"], max(p["kl_bits"], 1e-7)),
                             textcoords="offset points", xytext=(5, 4),
                             fontsize=7, color="#444")

    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Storage size, MB (log)")
    ax.set_ylabel("KL(base || variant), bits/token (log)")
    ax.set_title("Distribution accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_xscale("log")
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("Storage size, MB (log)")
    ax.set_ylabel("Perplexity change, %")
    ax.set_title("LM quality")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    out = os.path.join(OUT_DIR, "moe_field_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", default="8,16,32", help="field ranks, comma-separated")
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--save-dir", default=OUT_DIR)
    a = ap.parse_args()
    main(ranks=tuple(int(x) for x in a.ranks.split(",") if x.strip()),
         fit_steps=a.fit_steps, save_dir=a.save_dir)
