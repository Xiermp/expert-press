"""Comparing MoE transformation variants on the same model and eval sample.
Variants: centroid only / PCA basis + coefs. / uniform SVD / adaptive SVD /
a dense MLP fitted on activations.
Run: python3 variants_eval.py
"""
import csv
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, get_batch, prepare_data3
from transform_eval import base_logits, eval_ce_kl, precompute_svd, moe_weight_keys

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)


def reconstruct_ranks(sd, cfg, factors, ranks_map):
    """Reconstruction with per-expert ranks.
    ranks_map: key -> list of ranks per expert (or an int for uniform)."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    for key in moe_weight_keys(cfg):
        C, D, facs = factors[key]
        N = C.shape[0]
        full_bytes += D.numel() * 2
        comp_bytes += C.numel() * 2
        rec = torch.empty_like(D)
        for e, (U, S, Vh) in enumerate(facs):
            r = ranks_map[key] if isinstance(ranks_map, dict) else ranks_map
            r_e = r[e] if isinstance(r, (list, tuple)) else r
            r_e = max(0, min(r_e, S.numel()))
            if r_e > 0:
                rec[e] = (U[:, :r_e] * S[:r_e]) @ Vh[:r_e, :]
            else:
                rec[e] = 0.0
            comp_bytes += r_e * (U.shape[0] + U.shape[1]) * 2
        sd2[key] = C + rec
    return sd2, full_bytes, comp_bytes


def greedy_ranks(facs, budget):
    """Distribute ranks greedily: each step - to the expert with the largest
    next singular value."""
    ranks = [0] * len(facs)
    for _ in range(budget):
        best, best_sv = -1, -1.0
        for e, (_, S, _) in enumerate(facs):
            if ranks[e] < S.numel() and float(S[ranks[e]]) > best_sv:
                best, best_sv = e, float(S[ranks[e]])
        if best < 0:
            break
        ranks[best] += 1
    return ranks


def pca_variant(sd, cfg, K):
    """A 'shared basis + coefficients' variant: PCA over expert deltas.
    K=0 -> centroid only; K=N -> exact reconstruction."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    for key in moe_weight_keys(cfg):
        W = sd[key]                                  # (N, out, in)
        N = W.shape[0]
        C = W.mean(0, keepdim=True)
        D = (W - C).reshape(N, -1)                   # (N, DH)
        full_bytes += D.numel() * 2
        comp_bytes += C.numel() * 2
        K_eff = min(K, N)
        if K_eff > 0:
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            coef = (U[:, :K_eff] * S[:K_eff])        # (N,K) - the expert "formula"
            Drec = (coef @ Vh[:K_eff, :]).reshape(W.shape)
            comp_bytes += K_eff * C.numel() * 2 + coef.numel() * 2
        else:
            Drec = torch.zeros_like(W)
        sd2[key] = C + Drec
    return sd2, full_bytes, comp_bytes


class DenseWrapper(nn.Module):
    """One shared MLP instead of the MoE (M experts -> 1)."""

    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d))
        self.w2 = nn.Parameter(torch.empty(d, d_ff))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        return F.gelu(x @ self.w1.t()) @ self.w2.t(), None


@torch.no_grad()
def collect_moe_pairs(model, pool_ids, min_tokens=48_000):
    """Inputs (ln2) and outputs of every MoE layer on calibration tokens.
    pool_ids - the calibration pool; must NOT overlap the eval segment
    (leak fix: val_ids used to be passed here and the fit ran on eval)."""
    model.eval()
    g = torch.Generator().manual_seed(23)
    caps = [[] for _ in model.blocks]
    n = 0
    while n < min_tokens:
        x, _ = get_batch(pool_ids, 16, CFG["ctx"], gen=g)
        B, T = x.shape
        d = CFG["d_model"]
        h = model.emb(x)
        for li, b in enumerate(model.blocks):
            q, k, v = b.qkv(b.ln1(h)).split(d, dim=-1)
            q = q.view(B, T, b.h, d // b.h).transpose(1, 2)
            k = k.view(B, T, b.h, d // b.h).transpose(1, 2)
            v = v.view(B, T, b.h, d // b.h).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).reshape(B, T, d)
            h = h + b.proj(y)
            zin = b.ln2(h)
            moe_y, _ = b.moe(zin)
            caps[li].append((zin.reshape(-1, d).clone(), moe_y.reshape(-1, d).clone()))
            h = h + moe_y
        n += B * T
    model.train()
    return [(torch.cat([t[0] for t in layer]),
             torch.cat([t[1] for t in layer])) for layer in caps]


def fit_dense(zin, yout, steps=300, bs=8192, lr=2e-3, seed=5):
    torch.manual_seed(seed)
    d = zin.shape[1]
    mlp = DenseWrapper(d, CFG["d_ff"])
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    n = zin.shape[0]
    for s in range(steps):
        ix = torch.randint(0, n, (bs,))
        loss = F.mse_loss(mlp(zin[ix])[0], yout[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % 100 == 0:
            print(f"    dense fit step {s}: mse {loss.item():.5f}", flush=True)
    return mlp


def main():
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos, sd = ck["cfg"], ck["itos"], ck["sd"]

    _, calib_ids, val_ids, _, _ = prepare_data3()
    model = TinyMoE(cfg, len(itos))
    model.load_state_dict(sd)

    factors = precompute_svd(sd, cfg)
    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)
    n_exp = cfg["n_exp"]

    print(f"BASE: ppl={base_ppl:.2f}", flush=True)
    rows = []

    def add(name, sd2=None, fb=None, cb=None, dense=False):
        nonlocal model
        if dense:
            pass  # the model was already swapped outside
        else:
            model.load_state_dict(sd2)
        ce, kl = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        rows.append(dict(variant=name, mb=cb / 1e6, ratio=fb / max(cb, 1),
                         kl_bits=kl, ppl=ppl,
                         ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"{name:28s} mem {cb/1e6:6.2f} MB (x{fb/max(cb,1):4.1f}) "
              f"KL {kl:6.3f} bit  ppl {ppl:6.2f} ({100*(ppl-base_ppl)/base_ppl:+7.1f}%)",
              flush=True)

    # 1. centroid only (deltas discarded)
    sd0, fb, cb = pca_variant(sd, cfg, 0)
    add("centroid only", sd0, fb, cb)

    # 2. PCA basis + coefficients
    for K in (1, 2, 4):
        sdK, fb, cb = pca_variant(sd, cfg, K)
        add(f"PCA basis K={K}", sdK, fb, cb)

    # 3. uniform SVD (for comparison)
    for r in (8, 16):
        sdR, fb, cb = reconstruct_ranks(sd, cfg, factors, r)
        add(f"SVD r={r} uniform", sdR, fb, cb)

    # 4. adaptive SVD: same budget, ranks by delta energy
    for avg_r in (8, 16):
        ranks_map = {k: greedy_ranks(factors[k][2], n_exp * avg_r)
                     for k in moe_weight_keys(cfg)}
        ex = next(iter(ranks_map.values()))
        print(f"  adaptive avg_r={avg_r}: example ranks {ex}", flush=True)
        sdA, fb, cb = reconstruct_ranks(sd, cfg, factors, ranks_map)
        add(f"SVD adaptive ~r={avg_r}", sdA, fb, cb)

    # 5. dense MLP fitted on activations (M experts -> 1 shared MLP)
    print("  fitting the dense MLP on activations (calibration, eval untouched)...",
          flush=True)
    pairs = collect_moe_pairs(model, calib_ids)
    d = cfg["d_model"]
    full_bytes = sum(sd[k].numel() for k in moe_weight_keys(cfg)) * 2
    dense_bytes = cfg["n_layer"] * 2 * d * cfg["d_ff"] * 2
    for li, b in enumerate(model.blocks):
        mlp = fit_dense(*pairs[li])
        b.moe = mlp
    add("dense-MLP (M->1)", None, full_bytes, dense_bytes, dense=True)

    # table + export
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_variants_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {csv_path}", flush=True)

    plot(rows, base_ppl)


def plot(rows, base_ppl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [f"{r['variant']}\n(memory x{r['ratio']:.1f})" for r in rows]
    kls = [max(r["kl_bits"], 1e-7) for r in rows]
    pds = [r["ppl_delta"] for r in rows]
    ypos = list(range(len(rows)))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fig.suptitle("MoE transformation variants - what actually works")

    ax = axes[0]
    ax.barh(ypos, kls, color="#B8860B")
    ax.set_yticks(ypos, names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("KL(base || variant), bits/token (log)")
    ax.set_title("Distribution deviation")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    colors = ["#556B2F" if v >= 0 else "#2E8B57" for v in pds]
    ax.barh(ypos, pds, color=colors)
    ax.set_yticks(ypos, names, fontsize=8)
    ax.set_xlabel("Perplexity change, % (0 = same as base)")
    ax.set_title(f"LM quality (base ppl={base_ppl:.2f})")
    ax.grid(True, axis="x", alpha=0.3)

    out = os.path.join(OUT_DIR, "moe_variants_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    main()
