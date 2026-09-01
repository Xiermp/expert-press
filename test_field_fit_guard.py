#!/usr/bin/env python3
"""Field fit guard test (B1): the field MUST learn; a zero init is an error.

1) synthetic data with a known low-rank structure: the fit must end far below
   the centroid baseline;
2) emulating the old bug (U,V,C = zeros): the guard must fire with an error.
Run: python3 test_field_fit_guard.py   (offline, seconds)
"""
import torch
import torch.nn.functional as F

from hf_field_transform import FieldSparseMoe, fit_field_module

D, DFF, N_EXP, K, R = 32, 24, 8, 2, 8
GEOM = dict(n_exp=N_EXP, d_model=D, d_ff=DFF, top_k=K, norm_topk=False,
            hidden_act="silu", router_kind="softmax")

torch.manual_seed(0)
gw = torch.randn(N_EXP, D) * 0.5
X = torch.randn(4096, D)
z = torch.zeros(X.shape[0], N_EXP)
probs = F.softmax(X @ gw.t(), dim=-1)
sc, ix = torch.topk(probs, K, dim=-1)
z = z.scatter(-1, ix, sc)

wgud = torch.randn(2 * DFF, D) * 0.05          # centroid ground truth
wdnd = torch.randn(D, DFF) * 0.05
Ugu, Vgu = torch.randn(2 * DFF, R) * 0.3, torch.randn(D, R) * 0.3
Udn, Vdn = torch.randn(D, R) * 0.3, torch.randn(DFF, R) * 0.3
Cgu, Cdn = torch.randn(N_EXP, R), torch.randn(N_EXP, R)

def truth(x):
    zz = torch.zeros(x.shape[0], N_EXP).scatter(
        -1, torch.topk(F.softmax(x @ gw.t(), -1), K, -1).indices,
        torch.topk(F.softmax(x @ gw.t(), -1), K, -1).values)
    gu = x @ wgud.t() + (x @ Vgu * (zz @ Cgu)) @ Ugu.t()
    h = F.silu(gu.chunk(2, -1)[0]) * gu.chunk(2, -1)[1]
    return h @ wdnd.t() + (h @ Vdn * (zz @ Cdn)) @ Udn.t()

Y = truth(X)

def build(zero_init):
    m = FieldSparseMoe(GEOM, R, gate_w=gw)
    if zero_init:  # emulate the B1 bug
        for p in ("Ugu", "Vgu", "Udn", "Vdn"):
            with torch.no_grad():
                getattr(m, p).zero_()
    with torch.no_grad():
        m.wgud.copy_(wgud)
        m.wdnd.copy_(wdn := wdnd)
    return m

def run(zero_init):
    m = build(zero_init)
    with torch.no_grad():                       # pre-fit loss = centroid baseline
        base = F.mse_loss(m(X.unsqueeze(0)).squeeze(0), Y).item()
    fit_field_module(m, X, Y, steps=300, bs=1024, lr=2e-3, device="cpu",
                     log_prefix="test", log_every=250, guard=True, seed=5)
    with torch.no_grad():
        fin = F.mse_loss(m(X.unsqueeze(0)).squeeze(0), Y).item()
    return base, fin

base, fin = run(zero_init=False)
assert fin < 0.5 * base, f"weak fit: {base:.6f} -> {fin:.6f}"
print(f"OK fit: mse {base:.6f} -> {fin:.6f} ({100*(1-fin/base):.1f}% below the baseline)")

try:
    run(zero_init=True)
except RuntimeError as e:
    assert "FIT GUARD" in str(e)
    print("OK guard: degraded fit caught (old B1 bug reproduced)")
else:
    raise SystemExit("GUARD DID NOT FIRE on the zero initialization!")
print("ALL OK")
