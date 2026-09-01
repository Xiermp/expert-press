"""Transforming a ready MoE: centroid base + SVD deltas, rank sweep.
Metrics: memory (fp16), delta error, KL(base || compressed) bits/token,
perplexity.
Run: python3 transform_eval.py
"""
import csv
import math
import os

import torch
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, get_batch, prepare_data

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
OUT_DIR = os.environ.get("MOE_OUT_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT_DIR, exist_ok=True)
CHUNKS = 120


def moe_weight_keys(cfg):
    return [f"blocks.{li}.moe.{w}" for li in range(cfg["n_layer"]) for w in ("w1", "w2")]


def precompute_svd(sd, cfg):
    """Compute the SVD of all deltas once. Returns the factors and the
    spectrum energy."""
    factors = {}
    for key in moe_weight_keys(cfg):
        W = sd[key]                                # (N, out, in)
        C = W.mean(0, keepdim=True)                # centroid base
        D = W - C                                  # deltas
        facs = []
        for e in range(W.shape[0]):
            U, S, Vh = torch.linalg.svd(D[e], full_matrices=False)
            facs.append((U, S, Vh))
        factors[key] = (C, D, facs)
    return factors


def reconstruct(sd, cfg, factors, r):
    """Assemble the compressed state_dict at rank r + compute memory and
    errors."""
    sd2 = {k: v.clone() for k, v in sd.items()}
    full_bytes = comp_bytes = 0.0
    err_num = err_den = 0.0
    energy = {}
    for key in moe_weight_keys(cfg):
        C, D, facs = factors[key]
        N = C.shape[0]
        full_bytes += D.numel() * 2                     # all experts, fp16
        comp_bytes += C.numel() * 2                     # the base stored once
        rec = torch.empty_like(D)
        cap = []
        for e, (U, S, Vh) in enumerate(facs):
            r_eff = min(r, S.numel())
            Ur, Sr, Vhr = U[:, :r_eff], S[:r_eff], Vh[:r_eff, :]
            rec[e] = (Ur * Sr) @ Vhr
            comp_bytes += (Ur.numel() + Vhr.numel()) * 2
            err_num += float(((D[e] - rec[e]) ** 2).sum())
            err_den += float((D[e] ** 2).sum())
            cap.append(float((Sr ** 2).sum() / max(float((S ** 2).sum()), 1e-12)))
        energy[key] = sum(cap) / len(cap)
        sd2[key] = C + rec
    return sd2, full_bytes, comp_bytes, err_num / max(err_den, 1e-12), energy


@torch.no_grad()
def base_logits(model, val_ids):
    """Base-model log-probs on a fixed set of chunks (for KL)."""
    model.eval()
    g = torch.Generator().manual_seed(11)
    X, Y, LP = [], [], []
    for _ in range(CHUNKS):
        x, y = get_batch(val_ids, 1, CFG["ctx"], gen=g)
        logits, _ = model(x, None)
        LP.append(F.log_softmax(logits[0], dim=-1))
        X.append(x[0])
        Y.append(y[0])
    model.train()
    return X, Y, LP


@torch.no_grad()
def eval_ce_kl(model, X, Y, LP):
    model.eval()
    ces, kls = [], []
    for x, y, lp in zip(X, Y, LP):
        logits, _ = model(x[None], None)
        lq = F.log_softmax(logits[0], dim=-1)
        ces.append(F.cross_entropy(logits[0], y).item())
        p = lp.exp()
        kls.append(float((p * (lp - lq)).sum(-1).mean()))
    model.train()
    ce = sum(ces) / len(ces)
    kl_bits = (sum(kls) / len(kls)) / math.log(2)
    return ce, kl_bits


@torch.no_grad()
def generate(model, seed_ids, itos, n=160):
    model.eval()
    out = list(seed_ids)
    for _ in range(n):
        idx = torch.tensor(out[-CFG["ctx"]:])[None]
        logits, _ = model(idx, None)
        out.append(int(logits[0, -1].argmax()))
    model.train()
    return "".join(itos[i] for i in out)


def main():
    ck = torch.load(CKPT, weights_only=False)
    cfg, itos = ck["cfg"], ck["itos"]
    base_ppl = ck["val_ppl"]

    _, val_ids, _, _ = prepare_data()
    model = TinyMoE(cfg, len(itos))
    model.load_state_dict(ck["sd"])

    factors = precompute_svd(ck["sd"], cfg)
    X, Y, LP = base_logits(model, val_ids)
    base_ce, _ = eval_ce_kl(model, X, Y, LP)
    base_ppl = math.exp(base_ce)   # reference on the same eval sample
    print(f"BASE: ce={base_ce:.3f} ppl={base_ppl:.2f} "
          f"(ckpt val_ppl={ck['val_ppl']:.2f})", flush=True)

    seed = val_ids[:32]
    rows = []
    for r in RANKS:
        sd2, fb, cb, werr, energy = reconstruct(ck["sd"], cfg, factors, r)
        model.load_state_dict(sd2)
        ce, kl_bits = eval_ce_kl(model, X, Y, LP)
        ppl = math.exp(ce)
        ratio = fb / max(cb, 1)
        rows.append(dict(r=r, mb_full=fb / 1e6, mb_comp=cb / 1e6, ratio=ratio,
                         werr_pct=100 * math.sqrt(werr), kl_bits=kl_bits,
                         ppl_after=ppl, ppl_delta=100 * (ppl - base_ppl) / base_ppl))
        print(f"r={r:3d} mem {fb/1e6:7.2f}->{cb/1e6:6.2f} MB (x{ratio:5.1f}) "
              f"werr {100*math.sqrt(werr):5.1f}% KL {kl_bits:6.3f} bit "
              f"ppl {ppl:7.2f} ({100*(ppl-base_ppl)/base_ppl:+6.1f}%)", flush=True)

    e16 = reconstruct(ck["sd"], cfg, factors, 16)[4]
    print("\nShare of delta energy captured at r=16 (per layer):", flush=True)
    for key, v in e16.items():
        print(f"  {key}: {100 * v:.1f}%", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "moe_svd_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv -> {csv_path}", flush=True)

    # generation: base vs compressed at r=16
    base_gen = generate(model_load(ck), seed, itos)
    sd16 = reconstruct(ck["sd"], cfg, factors, 16)[0]
    m2 = TinyMoE(cfg, len(itos))
    m2.load_state_dict(sd16)
    comp_gen = generate(m2, seed, itos)
    print("\n--- BASE MODEL ---")
    print(base_gen)
    print("\n--- COMPRESSED (r=16) ---")
    print(comp_gen)

    plot(rows, base_ppl, cfg)


def model_load(ck):
    m = TinyMoE(ck["cfg"], len(ck["itos"]))
    m.load_state_dict(ck["sd"])
    return m


def plot(rows, base_ppl, cfg):  # noqa: the base now uses the same sample
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [row["r"] for row in rows]
    kls = [max(row["kl_bits"], 1e-6) for row in rows]
    ppls = [row["ppl_after"] for row in rows]
    ratios = [row["ratio"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    fig.suptitle("MoE transformation: base + SVD deltas "
                 f"({cfg['n_layer']} layers x {cfg['n_exp']} experts, CPU PoC)")

    ax = axes[0]
    ax.plot(rs, kls, marker="o", color="#B8860B")
    ax.set_yscale("log")
    ax.set_xlabel("Approximation rank r")
    ax.set_ylabel("KL(base || compressed), bits/token")
    ax.grid(True, alpha=0.3)
    ax.set_title("Distribution reconstruction accuracy")

    ax = axes[1]
    ax.plot(rs, ppls, marker="s", color="#556B2F", label="after compression")
    ax.axhline(base_ppl, ls="--", color="gray", label=f"base ({base_ppl:.2f})")
    for x, y, c in zip(rs, ppls, ratios):
        ax.annotate(f"x{c:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#444")
    ax.set_yscale("log")
    ax.set_xlabel("Approximation rank r  (labels: x - expert memory compression)")
    ax.set_ylabel("Perplexity (val)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("LM quality after transformation")

    out = os.path.join(OUT_DIR, "moe_svd_chart.png")
    fig.savefig(out, dpi=160)
    print(f"chart -> {out}", flush=True)


if __name__ == "__main__":
    main()
