"""The formula bank (the user's idea): deduplication of expert pieces.
Cut weights (or deltas) into pieces - neuron rows or blocks of B neurons,
cluster the pieces of ALL experts of a layer into a codebook (bank); an
expert stores only references: (index in the bank, fp16 scale).
A similar piece is not created again - experts reference the same one.

Variants:
  bank-W     - pieces of weights W_e, no base (a pure bank)
  bank-D     - base + a bank of delta pieces (W_e = W_bar + gather(bank))
  bank-W4 / bank-D4 - pieces of 4 neurons (block granularity)
  bank-W wh  - k-means in the whitened input space (activation-aware)
  sign-mask  - BitDelta: D_e = alpha_e * sign(D_e), 1 bit/weight (the mask limit)
Anchors: SVD r=8/16 (+ whitened SVD, cascade, dense-MLP from earlier CSVs on
the chart).
Run: python3 bank_eval.py
"""
import csv
import math
import os

import torch
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, prepare_data3
from transform_eval import base_logits, eval_ce_kl, moe_weight_keys, precompute_svd
from variants_eval import reconstruct_ranks
from upgrade_eval import collect_routed_inputs

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(7)


# ----------------------------- piece utilities ------------------------------

def pieces_of(W, B):
    """(N, out, in) -> (N * out/B, B*in): pieces of B neuron rows."""
    N, out, in_ = W.shape
    assert out % B == 0, f"out={out} is not divisible by B={B}"
    return W.reshape(N, out // B, B * in_).reshape(N * (out // B), B * in_)


def unpieces(Xh, N, out, in_, B):
    return Xh.reshape(N, out // B, B, in_).reshape(N, out, in_)


# ------------------------- sign-symmetric k-means ---------------------------

def bank_kmeans(X, Cn, iters=30, seed=0):
    """k-means for the bank: piece ~ s * c, s - a free fp16 scale.
    Assignment: min over c of ||x - s c||  <=>  max <x,c>^2/||c||^2 (s optimal).
    Update: centroid = the mean of sign-aligned pieces (c ~ -c when s<0)."""
    g = torch.Generator().manual_seed(seed)
    M, D = X.shape
    Cn = min(Cn, M)
    cent = X[torch.randperm(M, generator=g)[:Cn]].clone()
    ar = torch.arange(M)
    for _ in range(iters):
        G = X @ cent.t()                                    # (M, C)
        cn = (cent ** 2).sum(1).clamp(min=1e-12)
        assign = (G ** 2 / cn[None, :]).argmax(1)
        sgn = torch.sign(G[ar, assign]).unsqueeze(1)
        Xa = X * sgn                                        # sign alignment
        cnt = torch.bincount(assign, minlength=Cn).float()
        newc = torch.zeros_like(cent)
        newc.index_add_(0, assign, Xa)
        empty = cnt == 0
        ne = int(empty.sum())
        if ne:
            rs = torch.randint(0, M, (ne,), generator=g)
            newc[empty] = X[rs]
            cnt[empty] = 1.0
        cent = newc / cnt.clamp(min=1.0)[:, None]
    # final assignment + optimal scales
    G = X @ cent.t()
    cn = (cent ** 2).sum(1).clamp(min=1e-12)
    assign = (G ** 2 / cn[None, :]).argmax(1)
    c = cent[assign]
    s = (X * c).sum(1) / ((c * c).sum(1) + 1e-12)
    Xh = c * s[:, None]
    expl = 1.0 - float(((X - Xh) ** 2).sum() / max(float((X ** 2).sum()), 1e-12))
    return cent, assign, s, expl


# ------------------------------ bank variants -------------------------------

def bank_variant(sd, cfg, Cn, B, use_base, whiten, routed=None, iters=30):
    """A piece bank. use_base=False -> a bank instead of weights (no base);
    use_base=True -> base + a bank of delta pieces. whiten -> k-means in the
    whitened input space (B=1, activation-aware)."""
    assert not (whiten and B != 1), "whitening only for B=1"
    sd2 = {k: v.clone() for k, v in sd.items()}
    fb = cb = 0.0
    expl_all = []
    for key in moe_weight_keys(cfg):
        W = sd[key].float()
        N, out, in_ = W.shape
        if use_base:
            Cb = W.mean(0, keepdim=True)
            P = pieces_of(W - Cb, B)
            cb += out * in_ * 2                              # base fp16
        else:
            Cb = None
            P = pieces_of(W, B)
        M, D = P.shape
        fb += N * out * in_ * 2                              # all experts fp16
        if whiten:
            # only live experts with matching dimensionality
            parts = [routed[(key, e)].float() for e in range(N)]
            parts = [p for p in parts
                     if p.shape[0] > 0 and p.shape[1] == P.shape[1]]
            n = sum(p.shape[0] for p in parts)
            if n >= 256:
                Xall = torch.cat(parts)
                Cx = Xall.t() @ Xall / n
                Cx += torch.eye(in_) * (1e-3 * float(Cx.diagonal().mean()) + 1e-8)
                L = torch.linalg.cholesky(Cx)
                Pw = P @ L                                   # into the whitened world
                cent, assign, s, expl = bank_kmeans(Pw, Cn, iters=iters)
                cent_o = torch.linalg.solve_triangular(L.t(), cent.t(),
                                                       upper=True).t()  # c' L^-1
                Ph = cent_o[assign] * s[:, None]             # back into weight space
            else:
                cent, assign, s, expl = bank_kmeans(P, Cn, iters=iters)
                Ph = cent[assign] * s[:, None]
        else:
            cent, assign, s, expl = bank_kmeans(P, Cn, iters=iters)
            Ph = cent[assign] * s[:, None]
        expl_all.append(expl)
        bits = max(1, math.ceil(math.log2(Cn)))
        cb += Cn * D * 2                                     # bank fp16
        cb += M * bits / 8                                   # indices
        cb += M * 2                                          # scales fp16
        rec = unpieces(Ph, N, out, in_, B).to(sd[key].dtype)
        sd2[key] = (Cb + rec) if use_base else rec
        print(f"    {key}: pieces {M}x{D} bank={Cn} explained {100 * expl:.1f}% energy",
              flush=True)
    return sd2, fb, cb, sum(expl_all) / len(expl_all)


def sign_variant(sd, cfg):
    """BitDelta: delta = alpha_e * sign(D_e) - 1 bit/weight + base."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    fb = cb = 0.0
    for key in moe_weight_keys(cfg):
        W = sd[key]
        N, out, in_ = W.shape
        Cb = W.mean(0, keepdim=True)
        Dlt = W - Cb
        fb += Dlt.numel() * 2
        cb += out * in_ * 2                                  # base
        alpha = Dlt.abs().mean(dim=(1, 2), keepdim=True)     # LS-optimal scale
        sd2[key] = Cb + alpha * Dlt.sign()
        cb += N * out * in_ / 8 + N * 2                      # 1 bit/weight + alpha
    return sd2, fb, cb


# --------------------------- product quantization ---------------------------

def kmeans_plain(X, Cn, iters=25, seed=0):
    """Plain k-means (Euclidean, no scale/sign) - for PQ without scale."""
    g = torch.Generator().manual_seed(seed)
    M, D = X.shape
    Cn = min(Cn, M)
    cent = X[torch.randperm(M, generator=g)[:Cn]].clone()
    for _ in range(iters):
        assign = torch.cdist(X, cent).argmin(1)
        cnt = torch.bincount(assign, minlength=Cn).float()
        newc = torch.zeros_like(cent)
        newc.index_add_(0, assign, X)
        empty = cnt == 0
        ne = int(empty.sum())
        if ne:
            rs = torch.randint(0, M, (ne,), generator=g)
            newc[empty] = X[rs]
            cnt[empty] = 1.0
        cent = newc / cnt.clamp(min=1.0)[:, None]
    assign = torch.cdist(X, cent).argmin(1)
    return cent, assign


def pq_variant(sd, cfg, m, K, use_scale, use_base):
    """PQ - a "bank of banks", the limiting case of the idea: a row is cut into
    m subvectors, each piece type has its OWN bank of K entries; a row = m
    codes of log2(K) bits. use_scale - an fp16 scale per (row, subvector)."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    fb = cb = 0.0
    bits = math.ceil(math.log2(K))
    for key in moe_weight_keys(cfg):
        W = sd[key].float()
        N, out, in_ = W.shape
        Cb = None
        if use_base:
            Cb = W.mean(0, keepdim=True)
            rows = (W - Cb).reshape(N * out, in_)
            cb += out * in_ * 2
        else:
            rows = W.reshape(N * out, in_)
        R, Din = rows.shape
        assert Din % m == 0
        d = Din // m
        fb += N * out * in_ * 2
        cb += m * K * d * 2                                   # m banks fp16
        cb += R * m * bits / 8                                # codes
        if use_scale:
            cb += R * m * 2                                   # scales fp16
        rec = torch.empty_like(rows)
        for j in range(m):
            sub = rows[:, j * d:(j + 1) * d]
            if use_scale:
                cent, assign, s, _ = bank_kmeans(sub, K, iters=25, seed=j)
                rec[:, j * d:(j + 1) * d] = cent[assign] * s[:, None]
            else:
                cent, assign = kmeans_plain(sub, K, seed=j)
                rec[:, j * d:(j + 1) * d] = cent[assign]
        rec = rec.reshape(N, out, in_)
        sd2[key] = (Cb + rec) if use_base else rec
    return sd2, fb, cb


# --------------------------- dedup diagnostics ------------------------------

def dedup_report(sd, cfg):
    """Piece similarity ACROSS experts: max cosine to foreign pieces.
    High similarity = there is something for the bank to deduplicate."""
    for key in moe_weight_keys(cfg):
        W = sd[key].float()
        N, out, in_ = W.shape
        eid = torch.arange(N).repeat_interleave(out)
        same = eid[:, None] == eid[None, :]
        for tag, X in (("W ", W.reshape(N * out, in_)),
                       ("d ", (W - W.mean(0, keepdim=True)).reshape(N * out, in_))):
            Xn = F.normalize(X, dim=1)
            S = Xn @ Xn.t()
            S = S.masked_fill(same, -1.0)
            top = S.max(1).values
            print(f"  {key} [{tag}] max cosine to foreign: "
                  f"mean {float(top.mean()):.3f}, p90 {float(top.quantile(0.9)):.3f}",
                  flush=True)


# ---------------------------------- main ------------------------------------

def main():
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos, sd = ck["cfg"], ck["itos"], ck["sd"]

    _, calib_ids, val_ids, _, _ = prepare_data3()
    model = TinyMoE(cfg, len(itos))
    model.load_state_dict(sd)

    print("Expert calibration inputs (for the whitened bank)...", flush=True)
    routed = collect_routed_inputs(model, calib_ids, sd)

    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)
    print(f"BASE: ppl={base_ppl:.2f}", flush=True)

    print("\nDedup potential: piece similarity across experts", flush=True)
    dedup_report(sd, cfg)

    fac = precompute_svd(sd, cfg)
    rows = []

    def add(family, label, sd2, fb, cb):
        model.load_state_dict(sd2)
        ce, kl = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        rows.append(dict(family=family, variant=label, mb=cb / 1e6,
                         ratio=fb / max(cb, 1), kl_bits=kl, ppl=ppl,
                         ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"[{family:10s}] {label:8s} mem {cb / 1e6:5.2f} MB (x{fb / max(cb, 1):4.1f}) "
              f"KL {kl:6.3f} bit  ppl {ppl:6.2f} ({100 * (ppl - base_ppl) / base_ppl:+7.1f}%)",
              flush=True)

    # anchor: SVD
    for r in (8, 16):
        sd2, fb, cb = reconstruct_ranks(sd, cfg, fac, r)
        add("SVD", f"r={r}", sd2, fb, cb)

    # bank-W: a pure bank instead of weights (no base)
    for Cn in (128, 256, 512):
        print(f"  bank-W C={Cn}...", flush=True)
        sd2, fb, cb, ex = bank_variant(sd, cfg, Cn, 1, False, False)
        add("bank-W", f"C={Cn}", sd2, fb, cb)

    # bank-D: base + a bank of delta pieces (as in the user's idea)
    for Cn in (128, 256, 512):
        print(f"  bank-D C={Cn}...", flush=True)
        sd2, fb, cb, ex = bank_variant(sd, cfg, Cn, 1, True, False)
        add("bank-D", f"C={Cn}", sd2, fb, cb)

    # block pieces of 4 neurons
    for Cn in (32, 64):
        print(f"  bank-W4 C={Cn}...", flush=True)
        sd2, fb, cb, ex = bank_variant(sd, cfg, Cn, 4, False, False)
        add("bank-W4", f"C={Cn}", sd2, fb, cb)
    for Cn in (32, 64):
        print(f"  bank-D4 C={Cn}...", flush=True)
        sd2, fb, cb, ex = bank_variant(sd, cfg, Cn, 4, True, False)
        add("bank-D4", f"C={Cn}", sd2, fb, cb)

    # PQ - the limiting bank granularity: subvectors, a bank per piece type
    for m, K, sc, base, fam, lab in (
            (8, 256, False, False, "PQ-W", "m=8"),
            (8, 256, True, False, "PQ-W", "m=8+s"),
            (16, 256, False, False, "PQ-W", "m=16"),
            (8, 256, False, True, "PQ-D", "m=8")):
        print(f"  {fam} m={m} K={K} scale={sc} base={base}...", flush=True)
        sd2, fb, cb = pq_variant(sd, cfg, m, K, sc, base)
        add(fam, lab, sd2, fb, cb)

    # whitened bank-W: clustering aware of real inputs
    for Cn in (128, 256, 512):
        print(f"  bank-W wh C={Cn}...", flush=True)
        sd2, fb, cb, ex = bank_variant(sd, cfg, Cn, 1, False, True, routed)
        add("bank-W wh", f"C={Cn}", sd2, fb, cb)

    # sign mask (BitDelta, 1 bit/weight)
    sd2, fb, cb = sign_variant(sd, cfg)
    add("sign-mask", "1 bit", sd2, fb, cb)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_bank_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {csv_path}", flush=True)

    prev = load_prev()
    plot(rows + prev)


def load_prev():
    """Anchors from earlier runs: dense-MLP, whitened SVD, cascade."""
    extra = []
    spec = [
        (os.path.join(OUT_DIR, "moe_variants_results.csv"),
         lambda r: (r["variant"].startswith("dense-MLP"), r["variant"]),
         "dense-MLP"),
        (os.path.join(OUT_DIR, "moe_upgrade_results.csv"),
         lambda r: (r["family"] == "whitened" and r["variant"] in ("r=8", "r=16"),
                    f"wh-SVD {r['variant']}"),
         "wh-SVD"),
        (os.path.join(OUT_DIR, "moe_upgrade_results.csv"),
         lambda r: (r["family"] in ("cascade", "каскад"), r["variant"]),
         "cascade"),
    ]
    for path, keep, fam in spec:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ok, label = keep(r)
                if ok:
                    extra.append(dict(family=fam, variant=label,
                                      mb=float(r["mb"]), ratio=float(r["ratio"]),
                                      kl_bits=float(r["kl_bits"]),
                                      ppl=float(r["ppl"]),
                                      ppl_delta=float(r["ppl_delta"])))
    return extra


def plot(rows, ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "SVD": dict(color="gray", marker="o"),
        "wh-SVD": dict(color="#B8860B", marker="s"),
        "dense-MLP": dict(color="#4B0082", marker="P"),
        "cascade": dict(color="#556B2F", marker="D"),
        "bank-W": dict(color="#1F6FB4", marker="o"),
        "bank-D": dict(color="#E4572E", marker="^"),
        "bank-W4": dict(color="#7FB3D5", marker="v"),
        "bank-D4": dict(color="#F5B041", marker="<"),
        "bank-W wh": dict(color="#0E6655", marker="P"),
        "PQ-W": dict(color="#C2185B", marker="*"),
        "PQ-D": dict(color="#795548", marker="X"),
        "sign-mask": dict(color="#8B0000", marker="x"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    fig.suptitle("The formula bank: deduplication of expert pieces vs SVD / masks")

    for fam, st in styles.items():
        pts = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["mb"])
        if not pts:
            continue
        ax = axes[0]
        ax.plot([p["mb"] for p in pts], [max(p["kl_bits"], 1e-7) for p in pts],
                linestyle="-" if fam in ("SVD", "wh-SVD", "bank-W", "bank-D", "bank-W wh", "PQ-W") else "none",
                markersize=8, label=fam, **st)
        ax = axes[1]
        ax.plot([p["mb"] for p in pts], [p["ppl_delta"] for p in pts],
                linestyle="none", markersize=8, label=fam, **st)
        for p in pts:
            axes[0].annotate(f"x{p['ratio']:.1f}", (p["mb"], max(p["kl_bits"], 1e-7)),
                             textcoords="offset points", xytext=(5, 4),
                             fontsize=7, color="#444")

    ax = axes[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Storage size (bank/base + references), MB (log)")
    ax.set_ylabel("KL(base || variant), bits/token (log)")
    ax.set_title("Distribution accuracy")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.set_xscale("log")
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("Storage size, MB (log)")
    ax.set_ylabel("Perplexity change, %")
    ax.set_title("LM quality")
    ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    out = os.path.join(OUT_DIR, "moe_bank_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    main()
