#!/usr/bin/env python3
"""update-9 tests: fit-guard warmup / soft end-guard / lr warmup / router
tuning (joint + polish) / artifact gw_tuned override / GateMaster."""
import json
import os
import sys
import tempfile
import traceback

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
BASE = os.path.dirname(os.path.abspath(__file__))
_CAND = [BASE, os.path.join(BASE, "..", "ep7_work", "expert-press-update")]
PKG = next((q for q in _CAND
            if os.path.isfile(os.path.join(q, "hf_field_transform.py"))),
           _CAND[-1])
sys.path.insert(0, os.path.abspath(PKG))
sys.path.insert(0, os.path.abspath(BASE))

from hf_field_transform import (FieldSparseMoe, fit_field_module,  # noqa: E402
                                polish_router_module, write_field_artifact)
from bench_toy_router_guard import make_block, make_pool, fresh_module  # noqa: E402

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


blk = make_block(seed=100, hard=False)
X, Y = make_pool(blk, n=8192, seed=7, d=256)
Xf, Ye = X[:6144], Y[6144:]
Yf = Y[:6144]


def flat_fit(mod, **kw):
    """the healthy working point (pipeline default preset): adam-cosine,
    mse ends ~38% below the centroid baseline."""
    return fit_field_module(mod, Xf, Yf, 300, 512, 2e-3, DEV,
                            method="adam-cosine", seed=5, log_every=1000, **kw)


def t_default_guard_no_raise():
    m = fresh_module(blk, 8)
    mse = flat_fit(m)                       # default: auto warmup + soft guard
    assert mse < 0.0021, mse                # clearly below the baseline
def t_strict_old_behavior_raises():
    # constant adam @2e-3 drifts UP on this block; v7 (warmup=0) raised on it
    # via the 0.98x end-guard - strict mode must keep raising
    m = fresh_module(blk, 8)
    try:
        fit_field_module(m, Xf, Yf, 300, 512, 2e-3, DEV, method="adam",
                         seed=5, log_every=1000, guard_warmup=0,
                         strict_guard=True)
    except RuntimeError as e:
        assert "FIT GUARD" in str(e)
        return
    raise AssertionError("old semantics did not raise")
def t_worse_still_raises():
    # lr 8e-3 diverges hard; even the soft guard must refuse a WORSE mse
    m = fresh_module(blk, 8)
    try:
        fit_field_module(m, Xf, Yf, 60, 256, 8e-3, DEV, method="adam",
                         seed=5, log_every=1000)
    except RuntimeError as e:
        assert "WORSE" in str(e) or "FIT GUARD" in str(e)
        return
    raise AssertionError("a worse-than-baseline fit did not raise")
def t_lr_warmup_rescues():
    # constant adam @2e-3 drifts and RAISES without warmup; the same fit
    # with a 40-step lr ramp lands BELOW the baseline instead
    import torch.nn.functional as Fn
    m = fresh_module(blk, 8)
    with torch.no_grad():                   # the pristine baseline (8 batches)
        tot = 0.0
        for _ in range(8):
            ix = torch.randint(0, Xf.shape[0], (512,))
            tot += float(Fn.mse_loss(m.forward_from_z(Xf[ix], m._z(Xf[ix])),
                                     Yf[ix]))
        base = tot / 8
    try:
        fit_field_module(fresh_module(blk, 8), Xf, Yf, 300, 512, 2e-3, DEV,
                         method="adam", seed=5, log_every=1000)
        control_ok = True
    except RuntimeError:
        control_ok = False
    assert not control_ok, "expected the no-warmup control to raise"
    r = fit_field_module(fresh_module(blk, 8), Xf, Yf, 300, 512, 2e-3, DEV,
                         method="adam", seed=5, log_every=1000, lr_warmup=40)
    assert r < base, f"warmup fit {r:.5f} not below baseline {base:.5f}"
def t_train_router_joint():
    m = fresh_module(blk, 8)
    gw0 = m.gw.detach().clone()
    mse = fit_field_module(m, Xf, Yf, 150, 512, 2e-3, DEV,
                           method="adam-cosine", seed=5, log_every=1000,
                           train_router=True, router_anchor=0.03)
    drift = float((m.gw.detach() - gw0).norm() / gw0.norm())
    assert 0.0 < drift < 0.5, f"drift {drift}"
    assert mse < 0.0021, mse
def t_polish_router():
    m = fresh_module(blk, 8)
    fit_field_module(m, Xf, Yf, 120, 512, 2e-3, DEV, method="adam-cosine",
                     seed=5, log_every=1000)
    gw0 = m.gw.detach().clone()
    st = polish_router_module(m, Xf, Yf, 40, 512, 1e-3, DEV, anchor=0.03,
                              seed=11, log_prefix="t")
    drift = float((m.gw.detach() - gw0).norm() / gw0.norm())
    assert st["mse_after"] <= st["mse_before"] * 1.001, st
    assert drift < 0.2, drift
    assert st["drift"] < 0.2
def t_init_filter():
    m = fresh_module(blk, 8)
    init = {n: getattr(m, n).detach().clone() for n in m.field_names}
    init["gw_tuned"] = m.gw.detach().clone()          # aux key must be ignored
    m2 = FieldSparseMoe(blk["geom"], 8, gate_w=blk["gw"], init=init)
    assert torch.equal(m2.wgud, init["wgud"])
def t_gatemaster():
    from router_ft import GateMaster
    lin = torch.nn.Linear(16, 4, dtype=torch.bfloat16)
    gm = GateMaster(lin)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    y0 = lin(x)
    y1 = gm(x)
    assert torch.equal(y0, y1), "GateMaster must be bit-identical at start"
    lin.weight.requires_grad_(False)
    y = gm(x).float().sum()
    y.backward()
    assert gm.w.grad is not None and float(gm.w.grad.abs().sum()) > 0
def t_artifact_gw_override():
    d = tempfile.mkdtemp()
    src, pool, fitd, out = (os.path.join(d, n) for n in
                            ("src", "pool", "fit", "out"))
    for p in (src, pool, fitd):
        os.makedirs(p)
    g = blk["geom"]
    d_model, n_exp = g["d_model"], g["n_exp"]
    from safetensors.torch import save_file
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump(dict(model_type="olmoe", hidden_size=d_model), f)
    big = torch.randn(64, d_model)
    save_file({"model.layers.0.input_layernorm.weight": big,
               "model.layers.0.mlp.gate.weight": blk["gw"].clone(),
               "model.layers.0.mlp.experts.gate_up_proj":
                   torch.randn(n_exp, 2 * g["d_ff"], d_model)},
              os.path.join(src, "model.safetensors"))
    with open(os.path.join(pool, "art_meta.json"), "w") as f:
        json.dump(dict(n_layers=1, base_cls="OlmoeForCausalLM",
                       router_cls="OlmoeTopkRouter",
                       router_mod="transformers.models.olmoe.modeling_olmoe",
                       block_names=["model.layers.0.mlp"]), f)
    m = fresh_module(blk, 8)
    torch.save(dict(geom=g, gw=blk["gw"], mgu=blk["m_gu"], mdn=blk["m_dn"]),
               os.path.join(pool, "init_blk0.pt"))
    fit_out = {n: getattr(m, n).detach().clone() for n in m.field_names}
    fit_out["gw_tuned"] = blk["gw"] + 0.01
    torch.save(fit_out, os.path.join(fitd, "fit_blk0.pt"))
    write_field_artifact(src, out, pool, fitd, 8, torch.float32)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "model.safetensors"))
    assert "model.layers.0.mlp.experts.gate_up_proj" not in sd
    assert "model.layers.0.input_layernorm.weight" in sd
    gk = "model.layers.0.mlp.gate.weight"
    assert torch.allclose(sd[gk], fit_out["gw_tuned"]), "gw_tuned not in artifact"
    with open(os.path.join(out, "field_meta.json")) as f:
        meta = json.load(f)
    assert meta.get("router_polish", {}).get("n_layers_tuned") == 1
def t_artifact_no_tune_keeps_gate():
    d = tempfile.mkdtemp()
    src, pool, fitd, out = (os.path.join(d, n) for n in
                            ("src", "pool", "fit", "out"))
    for p in (src, pool, fitd):
        os.makedirs(p)
    g = blk["geom"]
    from safetensors.torch import save_file
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump(dict(model_type="olmoe", hidden_size=g["d_model"]), f)
    save_file({"model.layers.0.mlp.gate.weight": blk["gw"].clone()},
              os.path.join(src, "model.safetensors"))
    with open(os.path.join(pool, "art_meta.json"), "w") as f:
        json.dump(dict(n_layers=1, base_cls="OlmoeForCausalLM",
                       router_cls="OlmoeTopkRouter",
                       router_mod="transformers.models.olmoe.modeling_olmoe",
                       block_names=["model.layers.0.mlp"]), f)
    m = fresh_module(blk, 8)
    torch.save(dict(geom=g, gw=blk["gw"], mgu=blk["m_gu"], mdn=blk["m_dn"]),
               os.path.join(pool, "init_blk0.pt"))
    torch.save({n: getattr(m, n).detach().clone() for n in m.field_names},
               os.path.join(fitd, "fit_blk0.pt"))
    write_field_artifact(src, out, pool, fitd, 8, torch.float32)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "model.safetensors"))
    assert torch.allclose(sd["model.layers.0.mlp.gate.weight"], blk["gw"])
def t_cli_help():
    import subprocess
    for script in ("hf_pipeline.py", "router_ft.py"):
        r = subprocess.run([sys.executable, os.path.join(BASE, script),
                            "--help"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]
        assert "--fit-router" in r.stdout or "router" in r.stdout.lower()


for name, fn in [(n, f) for n, f in list(globals().items())
                 if n.startswith("t_") and callable(f)]:
    check(name, fn)

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", ", ".join(fail))
    sys.exit(1)
