#!/usr/bin/env python3
"""Toy bench for UPDATE-10 (session 2026-09-04), user's report of three holes
plus the speed rebuild. All on synthetic MoE blocks fed through the REAL
FieldSparseMoe module (imported from the package; the FIT LOGIC below is a
local copy with switches - the repo gets patched only after this bench
validates it).

  A. SVD-INIT (hole 1): U,V,C from the stacked expert deltas instead of
     randn*0.02 / zeros. NOTE: SVD of (1/N) sum_e dW_e is the SVD OF ZERO
     (sum_e dW_e == 0 identically); the correct variant is the shared-basis
     SVD of the CONCATENATED deltas. Checks: step-0 variance capture, fit
     quality after equal steps, streaming-randomized vs exact Grams.
  B. MUON SPLIT (hole 2): Cgu/Cdn (coordinate tables) must NOT go through
     Newton-Schulz (it couples unrelated experts' coordinates). Compare
     final mse: C-in-muon vs C-in-adam, and adam-cosine reference.
  C. JITTER ROUTING (hole 3): targets were produced by the base model on
     CLEAN rows; routing must follow the clean anchor (z_all[ix]), not the
     noisy xb. Compare final clean held-out mse.
  D. AUTOCAST SPEED: per-step time fp32 vs bf16-autocast (real steps), the
     >=1.2x decision rule of the honest probe. This box may lack AMX - the
     probe exists exactly to detect that per machine/block.

Outputs: results_speed/ CSV + PNG next to this script + console table.
"""
import copy
import csv
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
_CAND = [BASE, os.path.join(BASE, "expert-press-update"),
         os.path.join(BASE, "..", "ep7_work", "expert-press-update")]
PKG = next((p for p in _CAND
            if os.path.isfile(os.path.join(p, "hf_field_transform.py"))), _CAND[0])
sys.path.insert(0, os.path.abspath(PKG))
from hf_field_transform import FieldSparseMoe  # the REAL module  # noqa: E402

OUT = os.path.join(BASE, "results_speed")
os.makedirs(OUT, exist_ok=True)
DEV = "cpu"
ROWS = []


def logrow(section, name, **kw):
    print(f"  [{section}] {name}: " + " ".join(f"{k}={v}" for k, v in kw.items()),
          flush=True)
    ROWS.append(dict(section=section, name=name, **kw))


# --------------------------------------------------------------- synthetic
def make_block(d=512, dff=256, n_exp=16, top_k=4, seed=100, r_true=12,
               shared_frac=0.7):
    """Experts = centroid + per-expert delta. The delta mixes a SHARED
    subspace component (all experts live on the same rank-r_true side basis,
    different coefficients) and a PRIVATE component (per-expert random
    subspace, decaying spectrum). shared_frac=0.7 mimics real MoE deltas;
    0.0 = pessimistic no-shared-structure case."""
    g = torch.Generator().manual_seed(seed)

    def ln(a, b):
        return torch.randn(a, b, generator=g) * (1.0 / math.sqrt(b))

    m_gu = ln(2 * dff, d) * 0.9
    m_dn = ln(d, dff) * 0.9
    # SHARED bases (fixed across experts)
    Us_gu = ln(2 * dff, r_true)
    Vs_gu = ln(d, r_true)
    Us_dn = ln(d, r_true)
    Vs_dn = ln(dff, r_true)
    Wgu = torch.zeros(n_exp, 2 * dff, d)
    Wdn = torch.zeros(n_exp, d, dff)
    spec = torch.exp(-torch.arange(r_true, dtype=torch.float32) / (r_true / 3.0))
    for e in range(n_exp):
        coef_gu = torch.randn(r_true, generator=g) * 0.5
        coef_dn = torch.randn(r_true, generator=g) * 0.5
        sh_gu = (Us_gu * coef_gu.unsqueeze(0)) @ Vs_gu.t()
        sh_dn = (Us_dn * coef_dn.unsqueeze(0)) @ Vs_dn.t()
        Ue = torch.randn(2 * dff, r_true, generator=g) * 0.05
        Ve = torch.randn(d, r_true, generator=g) * 0.05
        pv_gu = (Ue * spec.unsqueeze(0)) @ Ve.t()
        Ue2 = torch.randn(d, r_true, generator=g) * 0.05
        Ve2 = torch.randn(dff, r_true, generator=g) * 0.05
        pv_dn = (Ue2 * spec.unsqueeze(0)) @ Ve2.t()
        Wgu[e] = m_gu + shared_frac * sh_gu + (1 - shared_frac) * pv_gu
        Wdn[e] = m_dn + shared_frac * sh_dn + (1 - shared_frac) * pv_dn
    gw = torch.randn(n_exp, d, generator=g) * (1.0 / math.sqrt(d))
    # realistic expert-individuality level (delta RMS / weight RMS ~ 0.35)
    for W, m in ((Wgu, m_gu), (Wdn, m_dn)):
        cur = (W - m.unsqueeze(0)).norm() / m.norm().clamp_min(1e-12)
        W.copy_(m.unsqueeze(0) + (W - m.unsqueeze(0)) * (0.35 / cur.clamp_min(1e-9)))
    return dict(m_gu=m_gu, m_dn=m_dn, Wgu=Wgu, Wdn=Wdn, gw=gw,
                geom=dict(n_exp=n_exp, d_model=d, d_ff=dff, top_k=top_k,
                          norm_topk=True, hidden_act="silu"))


@torch.no_grad()
def base_forward(blk, X, bs=4096):
    ys = []
    for i in range(0, X.shape[0], bs):
        x = X[i:i + bs]
        logits = x @ blk["gw"].t()
        probs = F.softmax(logits.float(), dim=-1)
        scores, idx = torch.topk(probs, blk["geom"]["top_k"], dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores)
        gu = torch.einsum("td,ned->tne", x, blk["Wgu"])
        gg, uu = gu.chunk(2, dim=-1)
        h = F.silu(gg) * uu
        y_exp = torch.einsum("tne,nde->tnd", h, blk["Wdn"])
        ys.append((z.unsqueeze(-1) * y_exp).sum(1))
    return torch.cat(ys)


def make_pool(blk, n, seed, d):
    g = torch.Generator().manual_seed(seed)
    scales = torch.exp(torch.randn(d, generator=g) * 0.4)
    X = torch.randn(n, d, generator=g) * scales
    return X, base_forward(blk, X)


# ------------------------------------------------------------ SVD init (A)
def _topk_eigh(G, r, os=16, seed=917):
    """Top-r eigenvectors (descending eigenvalue) of a symmetric PSD matrix,
    via a randomized range finder (streaming-friendly; the repo version
    never forms G)."""
    n = G.shape[0]
    q = min(n, r + os)
    g = torch.Generator().manual_seed(seed)
    Y = G @ (torch.randn(n, q, generator=g) / math.sqrt(q))
    Q, _ = torch.linalg.qr(Y)
    S = Q.t() @ (G @ Q)
    S = 0.5 * (S + S.t())
    vals, vecs = torch.linalg.eigh(S)
    order = torch.argsort(vals, descending=True)[:r]
    return Q @ vecs[:, order]


def svd_init_field(blk, rank, exact_grams=False):
    """Shared-basis init from the stacked deltas (the CORRECT variant of the
    user's hole-1 proposal; the mean-delta SVD is degenerate: sum dW = 0).
    U = top-r left subspace of the delta stack, V = top-r input subspace,
    C_e,j = u_j^T dW_e v_j. exact_grams: form the full Grams (toy-only
    reference) instead of the streaming randomized route."""
    dWgu = blk["Wgu"] - blk["m_gu"].unsqueeze(0)      # (N, 2dff, d)
    dWdn = blk["Wdn"] - blk["m_dn"].unsqueeze(0)      # (N, d, dff)
    out = {}
    for side, dW in (("gu", dWgu), ("dn", dWdn)):
        o, i = dW.shape[1], dW.shape[2]
        if exact_grams:                                # toy reference only
            G_out = torch.einsum("eoi,emi->om", dW, dW)   # sum dW dW^T (out,out)
            G_in = torch.einsum("eoi,eoj->ij", dW, dW)    # sum dW^T dW (in,in)
            U = _topk_eigh_direct(G_out, rank)
            V = _topk_eigh_direct(G_in, rank)
        else:
            # streaming route (what the repo will do): 2 passes over experts
            q = min(o, rank + 16)
            g = torch.Generator().manual_seed(917)
            Om = torch.randn(i, q, generator=g) / math.sqrt(q)
            Y = torch.zeros(o, q)
            for e in range(dW.shape[0]):
                Y += dW[e] @ Om
            Q, _ = torch.linalg.qr(Y)                  # (o, q)
            B = torch.stack([Q.t() @ dW[e] for e in range(dW.shape[0])])  # (N,q,i)
            G_out_q = sum(B[e] @ B[e].t() for e in range(B.shape[0]))     # (q,q)
            G_in_p = sum(B[e].t() @ B[e] for e in range(B.shape[0]))      # (i,i)
            U = Q @ _topk_eigh(G_out_q, rank)
            V = _topk_eigh(G_in_p, rank)
        # coordinates: C_e = diag(U^T dW_e V)
        C = torch.stack([torch.diagonal(U.t() @ dW[e] @ V)
                         for e in range(dW.shape[0])])
        out[f"U{side}"], out[f"V{side}"], out[f"C{side}"] = U, V, C
    return out


def _topk_eigh_direct(G, r):
    vals, vecs = torch.linalg.eigh(G)
    return vecs[:, torch.argsort(vals, descending=True)[:r]]


def capture_report(blk, init):
    """Share of per-expert delta energy captured by the (U,V,C) basis."""
    rep = {}
    for side in ("gu", "dn"):
        dW = blk["Wgu" if side == "gu" else "Wdn"] - \
            blk["m_gu" if side == "gu" else "m_dn"].unsqueeze(0)
        U, V, C = init[f"U{side}"], init[f"V{side}"], init[f"C{side}"]
        num = den = 0.0
        for e in range(dW.shape[0]):
            den += float(dW[e].norm() ** 2)
            rec = (U * C[e].unsqueeze(0)) @ V.t()
            num += float(rec.norm() ** 2)
        rep[side] = num / max(den, 1e-12)
    return rep


# ------------------------------------------------------- fit loop (switches)
def _ns(G, steps=5):
    X = G.clone()
    X = X / (X.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.t()
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.t() if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, ns_steps=5, momentum=0.95):
        super().__init__(list(params), dict(lr=lr, ns_steps=ns_steps,
                                            momentum=momentum, bs=0.9))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            for p in gr["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(p)
                st["mom"].mul_(gr["momentum"]).add_(p.grad)
                upd = p.grad.lerp(st["mom"], gr["momentum"])
                u = _ns(upd, gr["ns_steps"]).to(p.dtype)
                p.add_(u, alpha=-gr["lr"] * math.sqrt(
                    max(1.0, p.size(0) / max(1, p.size(1)))))


def build_opt(kind, mod, p, lr, c_in_muon=False, w_in_muon=False):
    """Returns a list of optimizers. kind 'adam': one Adam on everything.
    kind 'muon': split BY NAME - U*/V* operator factors go to Muon; C*
    (coordinate tables; hole 2) and w* (centroids) stay on Adam by default.
    c_in_muon=True puts C* into Muon too (the user's reported bug, isolated:
    threshold 256 so w* stays Adam in the toy); w_in_muon=True puts w* into
    Muon (isolate the effect seen in the first run)."""
    if kind == "adam":
        return [torch.optim.Adam(p, lr=lr)]
    names = list(mod.field_names)
    npairs = list(zip(names, p))
    mu, ad = [], []
    for n, t in npairs:
        if n.startswith(("U", "V")):
            mu.append(t)
        elif n.startswith("C") and c_in_muon:
            mu.append(t)
        elif n.startswith("w") and w_in_muon:
            mu.append(t)
        else:
            ad.append(t)
    return [torch.optim.Adam(ad, lr=lr) if ad else None,
            Muon(mu, lr=lr) if mu else None]


def fit(mod, X, Y, steps, bs, lr, kind="adam", seed=5, jit=0.0,
        c_in_muon=False, w_in_muon=False, autocast=False, z_clean=False,
        cosine=True, eval_X=None, eval_Y=None, log_every=40):
    gen = torch.Generator().manual_seed(seed)
    p = mod.fit_params()
    for t in p:
        t.requires_grad_(True)
    opts = build_opt(kind, mod, p, lr, c_in_muon, w_in_muon)
    n_tok = X.shape[0]
    x_std = X.float().std(dim=0) if jit > 0 else None
    noise = None
    if x_std is not None:
        with torch.no_grad():
            noise = torch.randn(X.shape, generator=gen) * (x_std * jit)
    with torch.no_grad():
        Xf = X.float()
        z_all = mod._z(Xf)
        tgt = Y.float()
    ac = (torch.autocast("cpu", dtype=torch.bfloat16) if autocast
          else torch.autocast("cpu", enabled=False))
    for s in range(steps):
        if cosine:
            lrs = lr * 0.5 * (1.0 + math.cos(math.pi * s / max(1, steps)))
            for o in opts:
                if o is not None:
                    for g in o.param_groups:
                        g["lr"] = lrs
        ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
        xb = Xf[ix]
        yb = tgt[ix]
        if x_std is not None:
            jx = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
            xb = xb + noise[jx]
            # hole 3: "noisy routing" (current repo behavior) recomputes z on
            # the NOISY input; "clean anchor" takes z of the clean row - the
            # row the base model actually produced the target from
            zb = (z_all[ix] if z_clean else mod._z(xb)).detach()
        else:
            zb = z_all[ix]
        with ac:
            out = mod.forward_from_z(xb, zb)
        loss = F.mse_loss(out.float(), yb)
        for o in opts:
            if o is not None:
                o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            if o is not None:
                o.step()
        last = float(loss.item())
        if s % log_every == 0 or s == steps - 1:
            print(f"      step {s}: mse {last:.5f}", flush=True)
    res = dict(final=last)
    if eval_X is not None:
        res["heldout"] = eval_mse(mod, eval_X, eval_Y)
    return res


@torch.no_grad()
def eval_mse(mod, X, Y, bs=4096):
    tot, n = 0.0, 0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs]
        yb = Y[i:i + bs]
        out = mod.forward_from_z(xb, mod._z(xb))
        tot += float(F.mse_loss(out, yb)) * xb.shape[0]
        n += xb.shape[0]
    return tot / n


def new_mod(blk, rank, init=None):
    m = FieldSparseMoe(blk["geom"], rank, gate_w=blk["gw"],
                       act_fn=F.silu, dtype=torch.float32,
                       init=(init or {}))
    with torch.no_grad():
        m.wgud.copy_(blk["m_gu"])
        m.wdnd.copy_(blk["m_dn"])
    return m


def rand_init(blk, rank):
    g = torch.Generator().manual_seed(1234567)
    out = {}
    for side, o, i in (("gu", 2 * blk["geom"]["d_ff"], blk["geom"]["d_model"]),
                       ("dn", blk["geom"]["d_model"], blk["geom"]["d_ff"])):
        out[f"U{side}"] = torch.randn(o, rank, generator=g) * 0.02
        out[f"V{side}"] = torch.randn(i, rank, generator=g) * 0.02
        out[f"C{side}"] = torch.zeros(blk["geom"]["n_exp"], rank)
    return out


# ----------------------------------------------------------------- sections
def main():
    RANK = 32
    STEPS = 120
    LR = 2e-3
    BS = 1024
    D, DFF, NEXP, TOPK = 512, 256, 16, 4

    # ---------------- A. SVD init -----------------------------------------
    print("\n=== A. SVD init (hole 1) ===", flush=True)
    blk = make_block(d=D, dff=DFF, n_exp=NEXP, top_k=TOPK, seed=100,
                     r_true=12, shared_frac=0.7)
    Xtr, Ytr = make_pool(blk, 4096, seed=7, d=D)
    Xho, Yho = make_pool(blk, 2048, seed=8, d=D)
    dWgu = blk["Wgu"] - blk["m_gu"].unsqueeze(0)
    den_gu = float((dWgu.norm() ** 2))

    sv = svd_init_field(blk, RANK)
    svx = svd_init_field(blk, RANK, exact_grams=True)
    cap_stream, cap_exact = capture_report(blk, sv), capture_report(blk, svx)
    rnd = rand_init(blk, RANK)
    logrow("A", "capture@step0", stream_gu=round(cap_stream["gu"], 3),
           exact_gu=round(cap_exact["gu"], 3),
           stream_dn=round(cap_stream["dn"], 3),
           exact_dn=round(cap_exact["dn"], 3),
           random_gu=round(capture_report(blk, rnd)["gu"], 4))

    m0 = new_mod(blk, RANK, rnd)
    base0 = eval_mse(m0, Xho, Yho)
    logrow("A", "heldout_mse@init", random=round(base0, 6),
           svd=round(eval_mse(new_mod(blk, RANK, sv), Xho, Yho), 6))

    for tag, ini in (("random", rnd), ("svd", sv)):
        m = new_mod(blk, RANK, ini)
        r = fit(m, Xtr, Ytr, STEPS, BS, LR, kind="adam", cosine=True,
                seed=5, eval_X=Xho, eval_Y=Yho)
        logrow("A", f"fit_adamcos_{tag}", final=round(r["final"], 6),
               heldout=round(r["heldout"], 6))

    # ---------------- B. muon split (hole 2) ------------------------------
    print("\n=== B. muon split (hole 2) ===", flush=True)
    for tag, kw in (("adamcos_ref", dict(kind="adam")),
                    ("muoncos_UVonly", dict(kind="muon")),
                    ("muoncos_CinMuon_HOLE", dict(kind="muon", c_in_muon=True)),
                    ("muoncos_WinMuon", dict(kind="muon", w_in_muon=True))):
        m = new_mod(blk, RANK, sv)
        r = fit(m, Xtr, Ytr, STEPS, BS, LR, seed=5, eval_X=Xho, eval_Y=Yho, **kw)
        logrow("B", f"fit_{tag}", final=round(r["final"], 6),
               heldout=round(r["heldout"], 6))

    # ---------------- C. jitter routing (hole 3) --------------------------
    print("\n=== C. jitter routing (hole 3) ===", flush=True)
    Xs, Ys = make_pool(blk, 2048, seed=9, d=D)   # 4 pairs/dim: jitter regime
    for jt in (0.15, 0.6):
        for tag, zc in (("noisyRouting_BUG", False), ("cleanAnchor", True)):
            m = new_mod(blk, RANK, sv)
            r = fit(m, Xs, Ys, STEPS, BS, LR, seed=5, jit=jt, z_clean=zc,
                    eval_X=Xho, eval_Y=Yho, log_every=60)
            logrow("C", f"fit_j{jt}_{tag}", final=round(r["final"], 6),
                   heldout=round(r["heldout"], 6))
    m = new_mod(blk, RANK, sv)
    r = fit(m, Xs, Ys, STEPS, BS, LR, seed=5, jit=0.0, eval_X=Xho, eval_Y=Yho,
            log_every=60)
    logrow("C", "fit_noJitter_ref", final=round(r["final"], 6),
           heldout=round(r["heldout"], 6))

    # ---------------- D. autocast speed -----------------------------------
    print("\n=== D. autocast speed (honest probe logic) ===", flush=True)
    for arm, ac_on in (("fp32", False), ("bf16-autocast", True)):
        m = new_mod(blk, RANK, sv)
        p = m.fit_params()
        for t in p:
            t.requires_grad_(True)
        opt = torch.optim.Adam(p, lr=LR)
        Xf = Xtr.float()
        z_all = m._z(Xf)
        tgt = Ytr.float()
        ts = []
        for s in range(8):                     # 2 warmup + 6 timed
            ix = torch.randint(0, Xf.shape[0], (BS,), generator=None)
            xb, yb, zb = Xf[ix], tgt[ix], z_all[ix]
            t0 = time.perf_counter()
            with (torch.autocast("cpu", dtype=torch.bfloat16) if ac_on
                  else torch.autocast("cpu", enabled=False)):
                out = m.forward_from_z(xb, zb)
            loss = F.mse_loss(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if s >= 2:
                ts.append(time.perf_counter() - t0)
        tmin = min(ts)
        speed = 1.0
        logrow("D", f"step_time_{arm}", min_s=round(tmin, 4),
               mse_last=round(float(loss.item()), 5))
        if arm == "fp32":
            t_fp32 = tmin
        else:
            speed = t_fp32 / tmin
    decision = "autocast ON" if speed >= 1.2 else "autocast OFF (probe keeps fp32)"
    logrow("D", "probe_decision", speedup=round(speed, 2), decision=decision)

    # ---------------- E. steps-to-quality (the wall-clock story) ----------
    print("\n=== E. steps-to-quality (svd init vs random init) ===", flush=True)
    for tag, ini, st in (("svd", sv, 30), ("svd", sv, 60), ("svd", sv, 120),
                         ("random", rnd, 120), ("random", rnd, 240)):
        m = new_mod(blk, RANK, ini)
        r = fit(m, Xtr, Ytr, st, BS, LR, kind="adam", cosine=True, seed=5,
                eval_X=Xho, eval_Y=Yho, log_every=10**9)
        logrow("E", f"adamcos_{tag}_s{st}", final=round(r["final"], 6),
               heldout=round(r["heldout"], 6))

    # ---------------- save -------------------------------------------------
    keys = ["section", "name"] + sorted({k for r in ROWS for k in r} -
                                        {"section", "name"})
    with open(os.path.join(OUT, "toy_speed_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ROWS)
    print(f"\ncsv -> {os.path.join(OUT, 'toy_speed_results.csv')}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        for fp in ("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",):
            if os.path.isfile(fp):
                fm.fontManager.addfont(fp)
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axs[0]
        names = [r["name"] for r in ROWS if r["section"] == "A"
                 and r["name"].startswith("fit")]
        vals = [r["heldout"] for r in ROWS if r["section"] == "A"
                and r["name"].startswith("fit")]
        ax.bar(names, vals, color=["#999", "#2a7"])
        ax.set_title("A: SVD-init fit (heldout mse)")
        ax = axs[1]
        rows = [r for r in ROWS if r["section"] == "B"]
        ax.bar([r["name"] for r in rows], [r["heldout"] for r in rows],
               color=["#999", "#2a7", "#d33", "#7a5"])
        ax.set_title("B: muon split (heldout mse)")
        ax = axs[2]
        rows = [r for r in ROWS if r["section"] == "C"]
        ax.bar([r["name"] for r in rows], [r["heldout"] for r in rows],
               color=["#d33", "#2a7", "#999"])
        ax.set_title("C: jitter routing (heldout mse)")
        for a in axs:
            a.tick_params(axis="x", rotation=20, labelsize=8)
        fig.tight_layout()
        p = os.path.join(OUT, "toy_speed_chart.png")
        fig.savefig(p, dpi=130)
        print(f"chart -> {p}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"chart skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
