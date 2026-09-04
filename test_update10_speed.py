#!/usr/bin/env python3
"""update-10 tests: muon (BY-NAME split: C*/router never in Newton-Schulz),
honest autocast probe (bit-exact restore, cache, on/off/auto), bf16-autocast
fit, clean-anchor jitter routing (hole 3), SVD init of U,V,C from the real
expert deltas (hole 1), guard interplay with the new init."""
import os
import subprocess
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
BASE = os.path.dirname(os.path.abspath(__file__))
_CAND = [BASE, os.path.join(BASE, "..", "ep7_work", "expert-press-update"),
         os.path.join(BASE, "..", "scripts")]
PKG = next((q for q in _CAND
            if os.path.isfile(os.path.join(q, "hf_field_transform.py"))),
           _CAND[-1])
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(BASE))

import hf_field_transform as HFT  # noqa: E402
from hf_field_transform import (FieldSparseMoe, expert_basis_init,  # noqa: E402
                                fit_field_module, _muon_split, _ns_orth,
                                _Muon, _resolve_fit_autocast, _time_fit_arms)
from bench_toy_speed import make_block, make_pool          # noqa: E402

DEV = "cpu"
ok, fail = [], []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}", flush=True)
    except Exception:
        fail.append(name)
        print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)


blk = make_block(d=256, dff=192, n_exp=16, top_k=4, seed=100, r_true=12,
                 shared_frac=0.7)
X, Y = make_pool(blk, 8192, seed=7, d=256)
Xf, Yf = X[:6144], Y[:6144]


def new_mod(rank=8, init=None):
    m = FieldSparseMoe(blk["geom"], rank, gate_w=blk["gw"], act_fn=F.silu,
                       dtype=torch.float32, init=(init or {}))
    with torch.no_grad():
        m.wgud.copy_(blk["m_gu"])
        m.wdnd.copy_(blk["m_dn"])
    return m


def t_ns_orth_columns():
    G = torch.randn(64, 32)
    U = _ns_orth(G, steps=5)
    s = torch.linalg.svdvals(U)
    # NS's contract: the spectrum is FLATTENED toward 1 (Muon needs equal
    # singular values, not exact orthogonality)
    assert float(s.min()) > 0.5 and float(s.max()) < 2.0, \
        (float(s.min()), float(s.max()))


def t_muon_split_by_name():
    m = new_mod(8)
    names = list(m.field_names)
    p = m.fit_params()
    mu, ad = _muon_split(names, p, 512)
    mu_names = {n for n, t in zip(names, p) if any(t is q for q in mu)}
    ad_names = {n for n, t in zip(names, p) if any(t is q for q in ad)}
    assert mu_names == {"Ugu", "Vgu", "Udn", "Vdn"}, mu_names
    assert {"Cgu", "Cdn", "wgud", "wdnd"} <= ad_names, ad_names   # hole 2
    # dim gate: tiny cap keeps everything on Adam
    mu2, ad2 = _muon_split(names, p, 4)
    assert not mu2 and len(ad2) == len(p)
    # nothing lost, nothing duplicated
    assert len(mu) + len(ad) == len(p)


def t_muon_optim_converges():
    T = torch.randn(32, 8)
    p = torch.nn.Parameter(torch.zeros(32, 8))
    opt = _Muon([p], lr=0.05)
    l0 = None
    for _ in range(120):
        p.grad = (p.detach() - T) * 0.1
        opt.step()
        loss = float(((p.detach() - T) ** 2).mean())
        l0 = loss if l0 is None else min(l0, loss)
    assert float(((p.detach() - T) ** 2).mean()) < 0.2, \
        float(((p.detach() - T) ** 2).mean())


def t_probe_restores_bitexact():
    m = new_mod(8)
    with torch.no_grad():
        for n in m.field_names:
            getattr(m, n).normal_(0, 0.1)
    before = [getattr(m, n).detach().clone() for n in m.field_names]
    Xfl = X.float()
    with torch.no_grad():
        z_all = m._z(Xfl)
        tgt = Y.float()
    out = _time_fit_arms(m, Xfl, tgt, z_all, 512, 2e-3, DEV, "adam", 0.0,
                         None, False, 512, 5)
    assert len(out) == 4 and all(v > 0 for v in out[:2]), out
    for n, b in zip(m.field_names, before):
        assert torch.equal(getattr(m, n).detach(), b), n


def t_probe_modes_and_cache():
    m = new_mod(8)
    Xfl = X.float()
    with torch.no_grad():
        z_all = m._z(Xfl)
        tgt = Y.float()
    assert _resolve_fit_autocast(m, Xfl, tgt, z_all, 512, 2e-3, DEV, "adam",
                                 0.0, None, False, "on", 512, 5) is True
    assert _resolve_fit_autocast(m, Xfl, tgt, z_all, 512, 2e-3, DEV, "adam",
                                 0.0, None, False, "off", 512, 5) is False
    n0 = len(HFT._HONEST_PROBE_CACHE)
    d1 = _resolve_fit_autocast(m, Xfl, tgt, z_all, 512, 2e-3, DEV, "adam",
                               0.0, None, False, "auto", 512, 5)
    assert isinstance(d1, bool)
    assert len(HFT._HONEST_PROBE_CACHE) == n0 + 1
    d2 = _resolve_fit_autocast(m, Xfl, tgt, z_all, 512, 2e-3, DEV, "adam",
                               0.0, None, False, "auto", 512, 5)
    assert d1 == d2 and len(HFT._HONEST_PROBE_CACHE) == n0 + 1   # cache hit


def t_fit_muon_cosine_deterministic():
    outs = []
    for _ in range(2):
        m = new_mod(8)
        mse = fit_field_module(m, Xf, Yf, 120, 512, 2e-3, DEV,
                               method="muon-cosine", seed=5, log_every=1000)
        outs.append((mse, [getattr(m, n).detach().clone()
                           for n in m.field_names]))
        assert all(getattr(m, n).dtype == torch.float32 for n in m.field_names)
    assert outs[0][0] == outs[1][0]
    for a, b in zip(outs[0][1], outs[1][1]):
        assert torch.equal(a, b)
    assert outs[0][0] < 0.9 * _baseline(), (outs[0][0], _baseline())


def t_fit_autocast_on_fp32_params():
    m = new_mod(8)
    mse = fit_field_module(m, Xf, Yf, 60, 512, 2e-3, DEV, method="adam-cosine",
                           seed=5, log_every=1000, autocast="on")
    assert mse == mse and mse > 0                     # finite
    assert all(getattr(m, n).dtype == torch.float32 for n in m.field_names)


def _baseline():
    m0 = new_mod(8)
    with torch.no_grad():
        m0.Cgu.zero_()
        m0.Cdn.zero_()
        tot, n = 0.0, 0
        for i in range(0, Xf.shape[0], 4096):
            xb = Xf[i:i + 4096]
            out = m0.forward_from_z(xb, m0._z(xb))
            tot += float(F.mse_loss(out, Yf[i:i + 4096])) * xb.shape[0]
            n += xb.shape[0]
    return tot / n


def t_clean_anchor_z_called_once():
    m = new_mod(8)
    calls = {"n": 0}
    orig = m._z

    def spy(x):
        calls["n"] += 1
        return orig(x)
    m._z = spy
    fit_field_module(m, Xf, Yf, 30, 512, 2e-3, DEV, method="adam", seed=5,
                     jitter=0.6, log_every=1000)
    # frozen router: z_all is precomputed ONCE; neither the fit, nor _eval8,
    # nor the probe may recompute _z per step (clean-anchor routing, hole 3)
    assert calls["n"] == 1, calls["n"]


def t_svd_init_beats_random_at_step0():
    class _Exp(nn.Module):
        def __init__(self, wgu, wdn):
            super().__init__()
            self.register_parameter("gate_up_proj", nn.Parameter(wgu))
            self.register_parameter("down_proj", nn.Parameter(wdn))

    class _Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _Exp(blk["Wgu"], blk["Wdn"])

    basis = expert_basis_init(_Blk(), blk["m_gu"], blk["m_dn"], 16,
                              log_prefix="t")
    # diag capture: what step 0 actually reconstructs; proj capture: the U/V
    # subspace pair carries (U,V are trainable - the fit recovers the rest)
    assert 0.25 < basis["capture"]["gu"] < 1.0, basis["capture"]
    assert 0.25 < basis["capture"]["dn"] < 1.0, basis["capture"]
    assert basis["capture_proj_gu"] > 0.8, basis["capture_proj_gu"]
    assert basis["capture_proj_dn"] > 0.8, basis["capture_proj_dn"]
    U, V, C = basis["Ugu"], basis["Vgu"], basis["Cgu"]
    assert U.shape == (2 * blk["geom"]["d_ff"], 16)
    assert V.shape == (blk["geom"]["d_model"], 16)
    assert C.shape == (blk["geom"]["n_exp"], 16)
    # reconstruction at step 0 must beat the random init clearly
    msvd = new_mod(16, {k: basis[k] for k in
                        ("Ugu", "Vgu", "Cgu", "Udn", "Vdn", "Cdn")})
    mrnd = new_mod(16)
    Xho, Yho = make_pool(blk, 2048, seed=8, d=256)

    def emse(m):
        with torch.no_grad():
            return float(F.mse_loss(m.forward_from_z(Xho, m._z(Xho)), Yho))
    assert emse(msvd) < 0.7 * emse(mrnd), (emse(msvd), emse(mrnd))


def t_muon_joint_router_stays_anchored():
    m = new_mod(8)
    gw0 = m.gw.detach().clone()
    mse = fit_field_module(m, Xf, Yf, 60, 512, 2e-3, DEV,
                           method="muon-cosine", seed=5, log_every=1000,
                           train_router=True, router_anchor=0.03)
    drift = float((m.gw.detach() - gw0).norm() / gw0.norm())
    assert mse > 0 and drift < 0.5, (mse, drift)


def t_cli_still_parses():
    r = subprocess.run([sys.executable, os.path.join(PKG, "hf_pipeline.py"),
                        "--list-stages"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert "fit" in r.stdout


for name, fn in list(globals().items()):
    if name.startswith("t_") and callable(fn):
        check(name, fn)

print(f"\n{len(ok)} passed, {len(fail)} failed", flush=True)
if fail:
    print("FAILED:", ", ".join(fail), flush=True)
    sys.exit(1)
