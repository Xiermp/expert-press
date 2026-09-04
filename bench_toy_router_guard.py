#!/usr/bin/env python3
"""Toy bench for three user questions (session 2026-09-04), all on ONE
synthetic MoE block fed through the REAL FieldSparseMoe module:

  1. GUARD  - "the fit guard is too aggressive at the start, it cuts the fit
              before Adam had time to adapt": reproduce the premature
              divergence-bail, then compare old-guard vs warmup-guard.
  2. ROUTER - "add the original router to the expert rebuild to shrink the
              post-conversion gap": (a) polish the (frozen-field) router on
              the same pairs AFTER the field fit, anchored to the original;
              (b) train router+field jointly.  Held-out MSE + how far the
              routing actually moves.
  3. ACT    - "the original activation hinders in those places": rank-budget
              split (gu-only vs dn-only) and learnable activation scales
              (gamma on g, tau scalar, gelu swap).

Outputs: results CSV + PNG next to this script (results_toy/) and a console
table.  Fit loop below is a faithful copy of fit_field_module (v7) with
switches; the module and forward math are the real ones.
"""
import copy
import math
import os
import sys

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
_CAND = [BASE,                                        # package layout (flat)
         os.path.join(BASE, "..", "ep7_work", "expert-press-update")]
PKG = next((p for p in _CAND
            if os.path.isfile(os.path.join(p, "hf_field_transform.py"))),
           _CAND[-1])
sys.path.insert(0, os.path.abspath(PKG))
from hf_field_transform import FieldSparseMoe  # the REAL module  # noqa: E402

OUT = os.path.join(BASE, "results_toy")
os.makedirs(OUT, exist_ok=True)
DEV = "cpu"


# --------------------------------------------------------------- synthetic
def make_block(d=256, dff=192, n_exp=16, top_k=4, seed=100, hard=False,
               dtype=torch.float32):
    """Experts with structured low-rank deltas from the centroid (the same
    picture real MoE deltas show: fast-decaying spectrum), random router.
    normal: delta spectrum r_true=12 (field-friendly, like real MoE),
    hard: r_true=20 + 4 fat experts (a tail the field can NOT fit)."""
    g = torch.Generator().manual_seed(seed)

    def ln(a, b):  # normal linear weight
        return torch.randn(a, b, generator=g) * (1.0 / math.sqrt(b))

    m_gu = ln(2 * dff, d) * 0.9
    m_dn = ln(d, dff) * 0.9
    r_true = 20 if hard else 12
    s_hi = 2.2 if hard else 1.4
    Wgu = torch.zeros(n_exp, 2 * dff, d)
    Wdn = torch.zeros(n_exp, d, dff)
    for e in range(n_exp):
        sc = s_hi if (hard and e < 4) else 1.0
        R = d
        spec = torch.exp(-torch.arange(R, dtype=torch.float32) / (r_true / 3.0)) * sc
        Ue = torch.randn(2 * dff, R, generator=g) * 0.05
        Ve = torch.randn(d, R, generator=g) * 0.05
        Wgu[e] = m_gu + (Ue * spec.unsqueeze(0)) @ Ve.t() / math.sqrt(R)
        R2 = dff
        spec2 = torch.exp(-torch.arange(R2, dtype=torch.float32) / (r_true / 3.0)) * sc
        Ue2 = torch.randn(d, R2, generator=g) * 0.05
        Ve2 = torch.randn(dff, R2, generator=g) * 0.05
        Wdn[e] = m_dn + (Ue2 * spec2.unsqueeze(0)) @ Ve2.t() / math.sqrt(R2)
    gw = torch.randn(n_exp, d, generator=g) * (1.0 / math.sqrt(d))  # router
    # rescale deltas to a REALISTIC expert-individuality level: in real MoE
    # the per-expert delta RMS is a decent fraction of the weight RMS (the
    # centroid alone is NOT enough); target ratio: normal ~0.35, hard ~0.55
    tgt_ratio = 0.55 if hard else 0.35
    for W, m in ((Wgu, m_gu), (Wdn, m_dn)):
        cur = (W - m.unsqueeze(0)).norm() / (m.norm().clamp_min(1e-12))
        W.copy_(m.unsqueeze(0) + (W - m.unsqueeze(0)) * (tgt_ratio / cur.clamp_min(1e-9)))
    return dict(m_gu=m_gu.to(dtype), m_dn=m_dn.to(dtype),
                Wgu=Wgu.to(dtype), Wdn=Wdn.to(dtype), gw=gw.to(dtype),
                geom=dict(n_exp=n_exp, d_model=d, d_ff=dff, top_k=top_k,
                          norm_topk=True, hidden_act="silu"))


@torch.no_grad()
def base_forward(blk, X, bs=4096):
    """Original MoE block: softmax top-k (norm), silu MLP, weighted sum."""
    ys = []
    for i in range(0, X.shape[0], bs):
        x = X[i:i + bs]
        logits = x @ blk["gw"].t()
        probs = F.softmax(logits.float(), dim=-1)
        scores, idx = torch.topk(probs, blk["geom"]["top_k"], dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores)     # (t,N)
        gu = torch.einsum("td,ned->tne", x, blk["Wgu"])
        gg, uu = gu.chunk(2, dim=-1)
        h = F.silu(gg) * uu
        y_exp = torch.einsum("tne,nde->tnd", h, blk["Wdn"])        # (t,N,d)
        ys.append((z.unsqueeze(-1) * y_exp).sum(1))
    return torch.cat(ys)


def make_pool(blk, n=16384, seed=7, d=256):
    g = torch.Generator().manual_seed(seed)
    scales = torch.exp(torch.randn(d, generator=g) * 0.4)
    X = torch.randn(n, d, generator=g) * scales
    return X, base_forward(blk, X)


# ------------------------------------------------------- toy fit loop (v7 copy)
def _target(mod, Y):
    return Y.float()                     # toy: no shared experts branch


def eval_mse(mod, X, Y, bs=4096):
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs].to(DEV)
            yb = Y[i:i + bs].to(DEV)
            out = mod.forward_from_z(xb, mod._z(xb))
            tot += float(F.mse_loss(out, yb)) * xb.shape[0]
            n += xb.shape[0]
    return tot / n


def routing_stats(mod, X, bs=4096):
    """How far the tuned router moved: top-k set change + score JS."""
    with torch.no_grad():
        chg, js, n = 0, 0.0, 0
        for i in range(0, X.shape[0], bs):
            x = X[i:i + bs].to(DEV)
            logits = x @ mod.gw.t()
            probs = F.softmax(logits.float(), dim=-1)
            s2, i2 = torch.topk(probs, mod.k, dim=-1)
            p0 = x @ gw0.t()
            q0 = F.softmax(p0.float(), dim=-1)
            s1, i1 = torch.topk(q0, mod.k, dim=-1)
            chg += (i1.sort(-1).values != i2.sort(-1).values).any(-1).sum().item()
            m = (q0 > 0) | (probs > 0)
            js += (0.5 * (q0 * (q0.clamp_min(1e-12) / probs.clamp_min(1e-12)).log()
                          * m) ).sum(-1).mean().item() * x.shape[0]
            n += x.shape[0]
    return chg / n, js / n


def toy_fit(mod, X, Y, steps, bs, lr, device, *, seed=5, method="adam-cosine",
            log_every=100, guard_mode="warmup", guard_warmup=None,
            end_guard="soft", lr_warmup=0, train_router=False,
            router_anchor=0.0, train_names=None, prefix=""):
    """Faithful copy of fit_field_module (guard/jitter parts kept) plus:
    guard_mode old|warmup, end_guard old|soft, lr_warmup, train_router."""
    gen = torch.Generator().manual_seed(int(seed))
    names = train_names if train_names is not None else list(mod.field_names)
    if train_router and "gw" not in names:
        names = names + ["gw"]
    p = [getattr(mod, n) for n in names]
    for t in p:
        t.requires_grad_(True)
    for n, t in mod.named_parameters():
        if n not in names:
            t.requires_grad_(False)
    gw0 = mod.gw.detach().clone() if train_router else None
    if method == "adamw":
        opt = torch.optim.AdamW(p, lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.Adam(p, lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
             if method == "adam-cosine" else None)
    base_lr = lr
    n_tok = X.shape[0]
    with torch.no_grad():
        Xf = X.float()
        tgt = _target(mod, Y)
        z_all = mod._z(Xf)
    warm_g = (max(30, steps // 10) if guard_warmup is None
              else int(guard_warmup)) if guard_mode == "warmup" else 0

    def snap():
        return [t.detach().to("cpu", copy=True) for t in p]

    def restore():
        with torch.no_grad():
            for t, s in zip(p, best_state):
                t.copy_(s.to(t.device))

    msgs, last, first = [], None, None
    ema = best_score = None
    best_state, last_snap = None, 0
    restored, bailed_at = False, None
    for s in range(steps):
        if lr_warmup and s < lr_warmup:
            for gpt in opt.param_groups:
                gpt["lr"] = base_lr * (s + 1) / lr_warmup
        ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
        xb = Xf[ix].to(device, non_blocking=True)
        yb = tgt[ix].to(device, non_blocking=True)
        if train_router:
            zb = mod._z(xb)                  # grads flow into the router
        else:
            zb = z_all[ix].to(device, non_blocking=True)
        out = mod.forward_from_z(xb, zb)
        loss = F.mse_loss(out, yb)
        if train_router and router_anchor > 0:
            loss = loss + router_anchor * (((mod.gw - gw0) ** 2).sum()
                                           / (gw0 ** 2).sum().clamp_min(1e-12))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None and (not lr_warmup or s >= lr_warmup):
            sched.step()
        last = float(loss.item())
        if first is None:
            first = last
        if not math.isfinite(last):
            msgs.append(f"non-finite at step {s} -> best restored")
            if best_state is not None:
                restore()
                restored = True
            break
        ema = last if ema is None else 0.9 * ema + 0.1 * last
        if best_state is None:
            best_score, best_state, last_snap = ema, snap(), s
        elif ema < best_score * 0.998 and s - last_snap >= 20:
            best_score, best_state, last_snap = ema, snap(), s
        elif best_score > 0 and ema > 2.0 * best_score:
            # the divergence bail: OLD fires from step 0, WARMUP waits
            if guard_mode == "old" or s >= warm_g:
                msgs.append(f"diverged at step {s} (ema {ema:.5f} vs best "
                            f"{best_score:.5f}) -> best restored, stopped")
                restore()
                restored = True
                bailed_at = s
                break
        if s % log_every == 0 or s == steps - 1:
            print(f"      {prefix} step {s}: mse {last:.5f}", flush=True)
    for t in p:
        t.requires_grad_(False)
    if best_state is not None and not restored and math.isfinite(ema) \
            and ema > best_score * 1.02:
        restore()
        restored = True
        msgs.append(f"late drift -> best restored (ema {ema:.5f} vs "
                    f"best {best_score:.5f})")
    if restored:
        last = best_score                 # shipped == best snapshot region
    verdict = "ok"
    if end_guard == "old":
        if first is not None and (not math.isfinite(last) or last > 0.98 * first):
            verdict = "RAISE"
            msgs.append(f"FIT GUARD: mse did not drop ({first:.5f} -> "
                        f"{last:.5f}) -> RuntimeError")
    else:
        if first is not None and not math.isfinite(last):
            verdict = "RAISE"
        elif first is not None and last > first:
            verdict = "warn-worse"
            msgs.append(f"guard: mse {first:.5f} -> {last:.5f} WORSE than "
                        f"centroid baseline")
        elif first is not None and last > 0.98 * first:
            verdict = "warn-flat"
            msgs.append(f"guard: mse {first:.5f} -> {last:.5f} no real gain "
                        f"(<2%) - shipping best state, continuing")
        else:
            msgs.append(f"guard: mse {first:.5f} -> {last:.5f} "
                        f"({100 * (first - last) / max(first, 1e-12):.1f}% "
                        f"below baseline)")
    return dict(first=first, last=last, shipped=last, verdict=verdict,
                bailed_at=bailed_at, restored=restored, msgs=msgs,
                n_steps=s + 1)


def polish_router(mod, X, Y, steps, bs, lr, anchor, *, seed=11, prefix=""):
    """after-mode: field frozen, ONLY the manual router gw trains; z is
    recomputed per step (grads flow through z into gw)."""
    Xf = X.float()
    tgt = _target(mod, Y)
    n_tok = X.shape[0]
    with torch.no_grad():
        mse0 = float(F.mse_loss(mod.forward_from_z(
            Xf[:4096].to(DEV), mod._z(Xf[:4096])).cpu(),
            tgt[:4096]))
    gw = mod.gw
    gw0 = gw.detach().clone()
    gw.requires_grad_(True)
    opt = torch.optim.Adam([gw], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    ema = best = None
    best_w = gw.detach().clone()
    for s in range(steps):
        ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
        xb = Xf[ix].to(DEV)
        yb = tgt[ix].to(DEV)
        out = mod.forward_from_z(xb, mod._z(xb))
        loss = F.mse_loss(out, yb)
        if anchor > 0:
            loss = loss + anchor * (((gw - gw0) ** 2).sum()
                                    / (gw0 ** 2).sum().clamp_min(1e-12))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        v = float(loss.item())
        ema = v if ema is None else 0.9 * ema + 0.1 * v
        if best is None or ema < best:
            best, best_w = ema, gw.detach().clone()
        if s % 20 == 0 or s == steps - 1:
            print(f"      {prefix} router step {s}: mse {v:.5f}", flush=True)
    with torch.no_grad():
        gw.copy_(best_w)
        gw.requires_grad_(False)
        mse1 = float(F.mse_loss(mod.forward_from_z(
            Xf[:4096].to(DEV), mod._z(Xf[:4096])).cpu(), tgt[:4096]))
    drift = float((gw.detach() - gw0).norm() / gw0.norm().clamp_min(1e-12))
    return dict(before=mse0, after=mse1, drift=drift)


def fresh_module(blk, rank, act_fn=F.silu):
    m = FieldSparseMoe(blk["geom"], rank, gate_w=blk["gw"], act_fn=act_fn,
                       dtype=torch.float32).to(DEV)
    with torch.no_grad():
        m.wgud.copy_(blk["m_gu"])
        m.wdnd.copy_(blk["m_dn"])
    return m


def eval_mse_temp(mod, X, Y, bs=4096):
    """Held-out mse for the temperature variant (own _z with temb)."""
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], bs):
            xb = X[i:i + bs].to(DEV)
            yb = Y[i:i + bs].to(DEV)
            out = mod.forward_from_z(xb, None)
            tot += float(F.mse_loss(out, yb)) * xb.shape[0]
            n += xb.shape[0]
    return tot / n


def rows_add(rows, group, name, **kv):
    r = dict(group=group, name=name, **kv)
    rows.append(r)
    print("  RESULT | " + " | ".join(f"{k}={v}" if not isinstance(v, float)
          else f"{k}={v:.5f}" for k, v in r.items()), flush=True)


# ------------------------------------------------------------------ experiments
def main():
    rows = []
    d = 256
    print("== pools ==", flush=True)
    blk_n = make_block(seed=100, hard=False)
    blk_h = make_block(seed=200, hard=True)
    pools = {}
    for tag, blk in (("normal", blk_n), ("hard", blk_h)):
        X, Y = make_pool(blk, n=16384, seed=7, d=d)
        pools[tag] = (X[:12288], X[12288:], Y[:12288], Y[12288:])
        m = fresh_module(blk, 8)
        print(f"  {tag}: centroid-only held-out mse "
              f"{eval_mse(m, pools[tag][1], pools[tag][3]):.5f}", flush=True)

    # ---- 1) GUARD: old vs warmup -----------------------------------------
    print("\n== 1) GUARD: old guard vs warmup guard (adam constant) ==",
          flush=True)
    for tag, blk in (("normal", blk_n), ("hard", blk_h)):
        Xf, Xe, Yf, Ye = pools[tag]
        for lr in (1e-3, 2e-3, 4e-3, 8e-3):
            for gm in ("old", "warmup"):
                m = fresh_module(blk, 8)
                r = toy_fit(m, Xf, Yf, 300, 512, lr, DEV, method="adam",
                            guard_mode=gm, end_guard="old" if gm == "old"
                            else "soft", seed=5, prefix=f"{tag}/lr{lr}/{gm}")
                ho = eval_mse(m, Xe, Ye)
                rows_add(rows, "guard", f"{tag}-lr{lr:g}-{gm}", lr=lr,
                         ho_mse=ho, fit_last=r["last"], verdict=r["verdict"],
                         bailed=r["bailed_at"] if r["bailed_at"] is not None
                         else -1, steps=r["n_steps"])
                for msg in r["msgs"]:
                    print(f"        [{tag}/lr{lr:g}/{gm}] {msg}", flush=True)
                if gm == "old" and r["verdict"] == "RAISE":
                    print("        -> pipeline would ABORT here (old guard)",
                          flush=True)
        # rescue variants: lr warmup gives Adam time to adapt (the user's
        # "did not have time to adapt" case) - no mid-fit bail expected
        for lr, lw in ((2e-3, 40), (4e-3, 40), (8e-3, 40)):
            m = fresh_module(blk, 8)
            r = toy_fit(m, Xf, Yf, 300, 512, lr, DEV, method="adam",
                        guard_mode="warmup", lr_warmup=lw, seed=5,
                        prefix=f"{tag}/lr{lr}/warm-lr{lw}")
            ho = eval_mse(m, Xe, Ye)
            rows_add(rows, "guard", f"{tag}-lr{lr:g}-warmup-lrw{lw}", lr=lr,
                     ho_mse=ho, fit_last=r["last"], verdict=r["verdict"],
                     bailed=r["bailed_at"] if r["bailed_at"] is not None else -1,
                     steps=r["n_steps"])
            for msg in r["msgs"]:
                print(f"        [{tag}/lr{lr:g}/lrw{lw}] {msg}", flush=True)

    # ---- 2) ROUTER: polish (after) / joint vs frozen ----------------------
    print("\n== 2) ROUTER: anchored polish + joint fit (normal block, r=16) ==",
          flush=True)
    Xf, Xe, Yf, Ye = pools["normal"]
    base = fresh_module(blk_n, 16)
    r0 = toy_fit(base, Xf, Yf, 300, 512, 2e-3, DEV, method="adam-cosine",
                 guard_mode="warmup", seed=5, prefix="frozen-router")
    ho0 = eval_mse(base, Xe, Ye)
    rows_add(rows, "router", "frozen-router", variant="frozen", anchor=0.0,
             ho_mse=ho0, drift=0.0, set_chg=0.0)
    print(f"  frozen router: fit mse {r0['last']:.5f}, held-out {ho0:.5f}",
          flush=True)

    global gw0
    for anch in (0.0, 0.01, 0.03, 0.1, 0.3):
        m = fresh_module(blk_n, 16)
        with torch.no_grad():                       # start from phase-1 state
            for n in base.field_names:
                getattr(m, n).copy_(getattr(base, n))
        gw0 = m.gw.detach().clone()
        pr = polish_router(m, Xf, Yf, steps=120, bs=1024, lr=1e-3, anchor=anch,
                           prefix=f"a{anch}")
        set_chg, js = routing_stats(m, Xe)
        ho = eval_mse(m, Xe, Ye)
        rows_add(rows, "router", f"polish-a{anch}", variant="polish",
                 anchor=anch, ho_mse=ho, drift=pr["drift"], set_chg=set_chg,
                 train_before=pr["before"], train_after=pr["after"])
    # joint
    mj = fresh_module(blk_n, 16)
    gw0 = mj.gw.detach().clone()
    rj = toy_fit(mj, Xf, Yf, 300, 512, 2e-3, DEV, method="adam-cosine",
                 guard_mode="warmup", train_router=True, router_anchor=0.03,
                 seed=5, prefix="joint")
    set_chg, js = routing_stats(mj, Xe)
    hoj = eval_mse(mj, Xe, Ye)
    rows_add(rows, "router", "joint-a0.03", variant="joint", anchor=0.03,
             ho_mse=hoj, drift=float((mj.gw.detach() - gw0).norm()
                                     / gw0.norm()), set_chg=set_chg)

    # robustness: SKEWED router (confident routing, expert specialization
    # concentrated) - does router polish help there?
    blk_s = make_block(seed=300, hard=False)
    blk_s["gw"] = blk_s["gw"] * 4.0                      # sharper softmax
    Xs, Ys = make_pool(blk_s, n=16384, seed=8, d=d)
    Xsf, Xse, Ysf, Yse = Xs[:12288], Xs[12288:], Ys[:12288], Ys[12288:]
    ms0 = fresh_module(blk_s, 16)
    toy_fit(ms0, Xsf, Ysf, 300, 512, 2e-3, DEV, method="adam-cosine",
            guard_mode="warmup", seed=5, prefix="skew/frozen")
    ho_s0 = eval_mse(ms0, Xse, Yse)
    gw0 = ms0.gw.detach().clone()
    pr_s = polish_router(ms0, Xsf, Ysf, steps=120, bs=1024, lr=1e-3,
                         anchor=0.03, prefix="skew/a0.03")
    set_chg_s, _ = routing_stats(ms0, Xse)
    rows_add(rows, "router", "skew-polish-a0.03", variant="polish-skew",
             anchor=0.03, ho_mse=eval_mse(ms0, Xse, Yse), drift=pr_s["drift"],
             set_chg=set_chg_s, train_before=pr_s["before"],
             train_after=pr_s["after"])
    rows_add(rows, "router", "skew-frozen", variant="frozen-skew", anchor=0.0,
             ho_mse=ho_s0, drift=0.0, set_chg=0.0)

    # temperature on the router softmax (a DIFFERENT routing lever than
    # gw tuning: rescales score ratios continuously)
    mt = fresh_module(blk_n, 16)
    with torch.no_grad():
        for n in base.field_names:
            getattr(mt, n).copy_(getattr(base, n))
    mt.temb = torch.nn.Parameter(torch.ones(1, device=DEV))
    mt._extra = ["temb"]

    def fz_t(x, z_unused, _m=mt):
        logits = x @ _m.gw.t() * _m.temb
        probs = F.softmax(logits, dim=-1)
        scores, idx = torch.topk(probs, _m.k, dim=-1)
        if _m.norm:
            scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores)
        cgu, cdn = z @ _m.Cgu, z @ _m.Cdn
        gu = x @ _m.wgud.t() + (x @ _m.Vgu * cgu) @ _m.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        h = _m.act_fn(g) * u
        return h @ _m.wdnd.t() + (h @ _m.Vdn * cdn) @ _m.Udn.t()
    mt.forward_from_z = fz_t
    rt = toy_fit(mt, Xf, Yf, 120, 1024, 3e-3, DEV, method="adam",
                 guard_mode="warmup", seed=9, train_names=["temb"],
                 prefix="routertemp")
    rows_add(rows, "router", "routertemp", variant="routertemp",
             anchor=0.0, ho_mse=eval_mse_temp(mt, Xe, Ye),
             drift=float(mt.temb.detach().item() - 1.0), set_chg=0.0)

    # ---- 3) ACTIVATION: rank split + learnable scales ---------------------
    print("\n== 3) ACT: rank split + learnable activation scales ==", flush=True)
    Xf, Xe, Yf, Ye = pools["normal"]

    # gamma: per-dim learnable scale BEFORE the activation (silu(g*gamma))
    def with_gamma(blk, rank):
        m = fresh_module(blk, rank)
        dff = m.wgud.shape[0] // 2
        m.gamma = torch.nn.Parameter(torch.ones(dff, device=DEV))
        m._extra = ["gamma"]

        def fz(x, z, _m=m):
            cgu, cdn = z @ _m.Cgu, z @ _m.Cdn
            gu = x @ _m.wgud.t() + (x @ _m.Vgu * cgu) @ _m.Ugu.t()
            g, u = gu.chunk(2, dim=-1)
            h = _m.act_fn(g * _m.gamma) * u
            return h @ _m.wdnd.t() + (h @ _m.Vdn * cdn) @ _m.Udn.t()
        m.forward_from_z = fz
        return m

    def with_tau(blk, rank):
        m = fresh_module(blk, rank)
        m.tau = torch.nn.Parameter(torch.zeros(1, device=DEV))
        m._extra = ["tau"]

        def fz(x, z, _m=m):
            cgu, cdn = z @ _m.Cgu, z @ _m.Cdn
            gu = x @ _m.wgud.t() + (x @ _m.Vgu * cgu) @ _m.Ugu.t()
            g, u = gu.chunk(2, dim=-1)
            h = _m.act_fn(g * (1.0 + _m.tau)) * u
            return h @ _m.wdnd.t() + (h @ _m.Vdn * cdn) @ _m.Udn.t()
        m.forward_from_z = fz
        return m

    variants = [
        ("base-silu", lambda: fresh_module(blk_n, 16), None),
        ("gu-only", lambda: fresh_module(blk_n, 16),
         ["wgud", "Ugu", "Vgu", "Cgu"]),
        ("dn-only", lambda: fresh_module(blk_n, 16),
         ["wdnd", "Udn", "Vdn", "Cdn"]),
        ("gamma", lambda: with_gamma(blk_n, 16), "EXTRA"),
        ("tau", lambda: with_tau(blk_n, 16), "EXTRA"),
        ("gelu", lambda: fresh_module(blk_n, 16, act_fn=F.gelu), None),
    ]
    for name, mk, names in variants:
        m = mk()
        tn = (list(m.field_names) + getattr(m, "_extra", [])) \
            if names == "EXTRA" else names
        r = toy_fit(m, Xf, Yf, 300, 512, 2e-3, DEV, method="adam-cosine",
                    guard_mode="warmup", seed=5, train_names=tn,
                    prefix=f"act/{name}")
        ho = eval_mse(m, Xe, Ye)
        rows_add(rows, "act", name, ho_mse=ho, fit_last=r["last"])

    # ---- save --------------------------------------------------------------
    import csv
    csv_p = os.path.join(OUT, "toy_router_guard_results.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {csv_p}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        # guard
        ax = axs[0]
        gr = [r for r in rows if r["group"] == "guard"]
        labels = [f"{r['name']}" for r in gr]
        vals = [r["ho_mse"] for r in gr]
        colors = ["#c44" if "-old" in l else "#4a8" for l in labels]
        ax.bar(range(len(gr)), vals, color=colors)
        ax.set_xticks(range(len(gr)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title("Гвард: mse на held-out (зелёный=warmup)")
        ax.set_ylabel("held-out MSE")
        # router
        ax = axs[1]
        pr = [r for r in rows if r["group"] == "router"
              and r["name"].startswith("polish")]
        xs = [r["anchor"] for r in pr]
        ax.plot(xs, [r["ho_mse"] for r in pr], "o-", label="polish (after)")
        fr = [r for r in rows if r["name"] == "frozen-router"][0]
        ax.axhline(fr["ho_mse"], color="gray", ls="--", label="frozen router")
        jt = [r for r in rows if r["name"].startswith("joint")][0]
        ax.plot([jt["anchor"]], [jt["ho_mse"]], "s", color="#e83",
                label="joint fit")
        ax.set_xlabel("anchor")
        ax.set_ylabel("held-out MSE")
        ax.set_title("Роутер в пересборке: mse vs anchor")
        ax.legend(fontsize=8)
        # act
        ax = axs[2]
        ar = [r for r in rows if r["group"] == "act"]
        ax.bar([r["name"] for r in ar], [r["ho_mse"] for r in ar],
               color="#46a")
        ax.set_title("Активатор: mse на held-out (r=8)")
        ax.set_ylabel("held-out MSE")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        png_p = os.path.join(OUT, "toy_router_guard_chart.png")
        fig.savefig(png_p, dpi=130)
        print(f"PNG -> {png_p}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"chart skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
