"""Blockwise SVD, whitened SVD (activation-aware, closed form) and the cascade
'weight formula + a repair formula on activations'.
Run: python3 upgrade_eval.py
"""
import csv
import math
import os

import torch
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, get_batch, prepare_data3
from transform_eval import base_logits, eval_ce_kl, moe_weight_keys
from variants_eval import reconstruct_ranks

OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)
CAP = 8000          # max calibration inputs per expert
TOKENS = 40000


@torch.no_grad()
def collect_routed_inputs(model, pool_ids, sd):
    """Real inputs of each expert (the tokens the router sent to it).
    pool_ids - the calibration pool; does NOT overlap the eval segment (leak
    fix: the whitening covariances and the cascade repair used to be collected
    from val_ids)."""
    model.eval()
    g = torch.Generator().manual_seed(31)
    n_exp, d = cfg()["n_exp"], CFG["d_model"]
    bufs = {f"blocks.{li}.moe.{w}": [[] for _ in range(n_exp)]
            for li in range(CFG["n_layer"]) for w in ("w1", "w2")}
    n = 0
    while n < TOKENS:
        x, _ = get_batch(pool_ids, 16, CFG["ctx"], gen=g)
        B = x.shape[0]
        h = model.emb(x)
        for li, b in enumerate(model.blocks):
            T = x.shape[1]
            q, k, v = b.qkv(b.ln1(h)).split(d, dim=-1)
            hd = d // b.h
            q = q.view(B, T, b.h, hd).transpose(1, 2)
            k = k.view(B, T, b.h, hd).transpose(1, 2)
            v = v.view(B, T, b.h, hd).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + b.proj(y.transpose(1, 2).reshape(B, T, d))
            zin = b.ln2(h)
            probs = F.softmax(b.moe.router(zin), dim=-1)
            am = probs.argmax(-1).reshape(-1)
            zf = zin.reshape(-1, d)
            for e in range(n_exp):
                sel = (am == e).nonzero().squeeze(-1)
                if sel.numel() == 0:
                    continue
                xe = zf[sel]
                bufs[f"blocks.{li}.moe.w1"][e].append(xe)
                bufs[f"blocks.{li}.moe.w2"][e].append(
                    F.gelu(xe @ sd[f"blocks.{li}.moe.w1"][e].t()))
            h = h + b.moe(zin)[0]
        n += B * CFG["ctx"]

    outs = {}
    for key, lst in bufs.items():
        for e in range(n_exp):
            X = torch.cat(lst[e]) if lst[e] else torch.zeros(0, d)
            if X.shape[0] > CAP:
                X = X[torch.randperm(X.shape[0])[:CAP]]
            outs[(key, e)] = X
    model.train()
    return outs


def whitened_recon(D_e, X, r):
    """The SVD optimal for inputs X: SVD(DL), where LL^T = the input
    covariance."""
    din = D_e.shape[1]
    if X.shape[0] < 16:  # a dead expert - plain SVD, no whitening
        U, S, Vh = torch.linalg.svd(D_e, full_matrices=False)
        r_eff = min(r, S.numel())
        U_r = U[:, :r_eff]
        Rf = S[:r_eff, None] * Vh[:r_eff, :]
        return U_r @ Rf, U_r, Rf
    n = X.shape[0]
    Cx = (X.t() @ X) / n
    Cx += torch.eye(din) * (1e-3 * float(Cx.diagonal().mean()) + 1e-8)
    Lc = torch.linalg.cholesky(Cx)
    M = D_e @ Lc
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r_eff = min(r, S.numel())
    U_r = U[:, :r_eff]
    Rf = S[:r_eff, None] * Vh[:r_eff, :]
    # we need Rf @ L^-1; solve L^T Z = Rf^T -> Z = L^-T Rf^T -> Rf = Z^T = Rf L^-1
    Rf = torch.linalg.solve_triangular(Lc.t(), Rf.t(), upper=True).t()
    return U_r @ Rf, U_r, Rf


def whitened_variant(sd, routed, r):
    """All deltas -> whitened SVD of rank r. Bytes same as plain SVD."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    for key in moe_weight_keys(CFG):
        W = sd[key]
        C = W.mean(0, keepdim=True)
        D = W - C
        full_bytes += D.numel() * 2
        comp_bytes += C.numel() * 2
        for e in range(W.shape[0]):
            X = routed[(key, e)]
            rec, U_r, Rf = whitened_recon(D[e], X, r)
            sd2[key][e] = C[0] + rec
            comp_bytes += (U_r.numel() + Rf.numel()) * 2
    return sd2, full_bytes, comp_bytes


def block_variant(sd, cfg, grid, r_block):
    """Blockwise SVD: a grid x grid layout, its own formula per block."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    go, gi = grid
    for key in moe_weight_keys(cfg):
        W = sd[key]
        N, out, inw = W.shape
        C = W.mean(0, keepdim=True)
        D = W - C
        full_bytes += D.numel() * 2
        comp_bytes += C.numel() * 2
        bo, bi = out // go, inw // gi
        for e in range(N):
            rec = torch.zeros_like(D[e])
            for i in range(go):
                for j in range(gi):
                    blk = D[e, i * bo:(i + 1) * bo, j * bi:(j + 1) * bi]
                    U, S, Vh = torch.linalg.svd(blk, full_matrices=False)
                    rb = min(r_block, S.numel())
                    rec[i * bo:(i + 1) * bo, j * bi:(j + 1) * bi] = \
                        (U[:, :rb] * S[:rb]) @ Vh[:rb, :]
                    comp_bytes += rb * (bo + bi) * 2
            sd2[key][e] = C[0] + rec
    return sd2, full_bytes, comp_bytes


def cascade_variant(sd, routed, factors, r1, r2, steps=300):
    """Double transformation: level 1 - whitened SVD r1 (weights), level 2 - a
    repair formula of rank r2 fitted on activations."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    for key in moe_weight_keys(CFG):
        W = sd[key]
        C = W.mean(0, keepdim=True)
        D = W - C
        full_bytes += D.numel() * 2
        comp_bytes += C.numel() * 2
        for e in range(W.shape[0]):
            X = routed[(key, e)]
            L1, U_r, Rf = whitened_recon(D[e], X, r1)
            if X.shape[0] < 16:  # dead expert: level 1 only
                sd2[key][e] = C[0] + L1
                comp_bytes += (U_r.numel() + Rf.numel()) * 2
                continue
            dout, din = D[e].shape
            n = X.shape[0]
            Tres = (D[e] - L1) @ X.t()                     # (dout, n)
            A = torch.randn(dout, r2) * 0.01
            Bm = torch.randn(r2, din) * 0.01
            A.requires_grad_(True)
            Bm.requires_grad_(True)
            opt = torch.optim.Adam([A, Bm], lr=1e-2)
            for _ in range(steps):
                idx = torch.randint(0, n, (min(4096, n),))
                loss = ((Tres[:, idx] - A @ (Bm @ X[idx].t())) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            A, Bm = A.detach(), Bm.detach()
            sd2[key][e] = C[0] + L1 + A @ Bm
            comp_bytes += (U_r.numel() + Rf.numel()) * 2 + (A.numel() + Bm.numel()) * 2
    return sd2, full_bytes, comp_bytes


def cfg():
    return CFG


def main():
    ck = torch.load(CKPT, weights_only=False)
    itos, sd = ck["itos"], ck["sd"]

    _, calib_ids, val_ids, _, _ = prepare_data3()
    model = TinyMoE(CFG, len(itos))
    model.load_state_dict(sd)

    print("Collecting expert calibration inputs (calib segment, eval untouched)...",
          flush=True)
    routed = collect_routed_inputs(model, calib_ids, sd)
    sizes = [routed[(("blocks.0.moe.w1"), e)].shape[0] for e in range(CFG["n_exp"])]
    print(f"  inputs per expert (layer 0, w1): {sizes}", flush=True)

    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)
    print(f"BASE: ppl={base_ppl:.2f}", flush=True)

    rows = []

    def add(family, label, sd2, fb, cb):
        model.load_state_dict(sd2)
        ce, kl = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        rows.append(dict(family=family, variant=label, mb=cb / 1e6,
                         ratio=fb / max(cb, 1), kl_bits=kl, ppl=ppl,
                         ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"[{family:9s}] {label:18s} mem {cb/1e6:5.2f} MB (x{fb/max(cb,1):4.1f}) "
              f"KL {kl:6.3f} bit  ppl {ppl:6.2f} ({100*(ppl-base_ppl)/base_ppl:+7.1f}%)",
              flush=True)

    # baseline: plain SVD
    from transform_eval import precompute_svd
    fac = precompute_svd(sd, CFG)
    for r in (8, 16):
        sd2, fb, cb = reconstruct_ranks(sd, CFG, fac, r)
        add("SVD", f"r={r}", sd2, fb, cb)

    # blockwise SVD (same memory as r=16)
    sd2, fb, cb = block_variant(sd, CFG, (2, 2), 8)
    add("blocks 2x2", "r_b=8", sd2, fb, cb)
    sd2, fb, cb = block_variant(sd, CFG, (4, 4), 4)
    add("blocks 4x4", "r_b=4", sd2, fb, cb)

    # whitened SVD - same memory, but the formula fits the real inputs
    for r in (4, 8, 16):
        sd2, fb, cb = whitened_variant(sd, routed, r)
        add("whitened", f"r={r}", sd2, fb, cb)

    # cascade: whitened r=8 + a repair formula r=8 on activations
    sd2, fb, cb = cascade_variant(sd, routed, fac, 8, 8)
    add("cascade", "wh8+repair8", sd2, fb, cb)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_upgrade_results.csv")
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
        "blocks 2x2": dict(color="#8B0000", marker="v"),
        "blocks 4x4": dict(color="#FF6347", marker="^"),
        "whitened": dict(color="#B8860B", marker="s"),
        "cascade": dict(color="#556B2F", marker="D"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    for fam, st in styles.items():
        pts = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["mb"])
        if not pts:
            continue
        ax.plot([p["mb"] for p in pts], [max(p["kl_bits"], 1e-7) for p in pts],
                linestyle="-" if fam in ("SVD", "whitened") else "none",
                markersize=8, label=fam, **st)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Full storage size (base + formulas), MB (log)")
    ax.set_ylabel("KL(base || variant), bits/token (log)")
    ax.set_title("Blockwise / whitened / cascade - double transformation under "
                 "activations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = os.path.join(OUT_DIR, "moe_upgrade_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    main()
