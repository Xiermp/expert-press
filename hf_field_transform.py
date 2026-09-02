# version: 2026-09-02.4 - fix "8.2 got SLOWER than 8.1" (user report): the
#   2026-09-02.3 auto decision timed ONE big 1024x1024 matmul and could
#   enable bf16 where the REAL step (many small GEMMs + per-step weight
#   casts + backward) is net slower - now "auto" times ~8 REAL fit steps
#   (fp32 vs bf16) once per geometry and keeps bf16 only if the full step
#   is >=1.2x faster; the measured seconds go into the muon-split line.
#   Also fixes an 8.2 UnboundLocalError: `ac` was created only in the muon
#   branch, so ANY adam/adamw/adam-cosine/rmsprop fit crashed at the
#   `with ac:` line; autocast resolution now happens for ALL methods.
#   The probe restores the initial params and clears optimizer state and
#   uses a LOCAL RNG, so fitted trajectories are bit-identical to a no-probe
#   run (auto->bf16 == forced bf16, asserted in tests). The fit cache
#   signature is UNCHANGED (still the --fit-autocast string), so 8.2 fit
#   artifacts stay valid - no re-fit is forced by this patch.
#   2026-09-02.3 - fit STEP SPEED (user report: 6.86 s/step with muon
#   on the real model): profiling the exact fit loop shows ~85% of the step
#   is the fp32 forward/backward matmuls (Newton-Schulz is only ~9%, the
#   paired Adam ~2%) - so the step now runs its forward under
#   torch.autocast(bfloat16) with params/optimizer state kept fp32 and the
#   loss on out.float(); --fit-autocast auto (default) probes the machine
#   once and stays fp32 where bf16 matmuls are NOT faster (old AVX2 CPUs).
#   Measured on the synthetic block (2 cores, bs 4096): 0.18 -> 0.10 s/step
#   (AMX box, 1.8x; the fwd+bwd phase alone 2.7x), best-ema IDENTICAL
#   (0.06008 fp32 vs 0.06005 bf16, seed 0; 0.06051 vs 0.06050, seed 1).
#   Also exposes --muon-ns-steps (Newton-Schulz iterations, default 5).
#   2026-09-02.2: hotfix for the real-model fit divergences of the first
#   update-8 run (flat x1.0 Muon scale, "biggest moves" diagnostics,
#   lr/2 retry, s/step + ETA logs). Base: expert-press-update (7).zip
#   (sha256 8a471b7d...).
"""Transforming a real HF MoE model into the "field engine"
(experts are not stored: centroids + low-rank factors + router coordinates).

Works with transformers>=5 (SparseMoeBlock contract: forward -> Tensor,
experts = a container with fused gate_up_proj (N,2dff,d) and down_proj
(N,d,dff)). Verified on OLMoE / Mixtral (v5); by the same contract Qwen3-MoE
and others fit too.

Two phases per block:
  FIT    - FieldSparseMoe with a manual router (fp32 clone of the router
           weight), Adam on pairs
  DEPLOY - the same module with the ORIGINAL base router, params in bf16
           (artifact)
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "modeling_field_template.py")

# available fit optimizer kinds (see fit_field_module / --fit-method)
# "muon" is the DEFAULT since 2026-09-02.1; the adam family stays for A/B
# and rollbacks (muon-cosine = Muon + cosine lr decay to ~0).
FIT_METHODS = ("muon", "muon-cosine", "adam", "adamw", "adam-cosine", "rmsprop")

# In Muon mode, params with min(shape) above this cap go to a paired Adam:
# the Newton-Schulz iteration costs O(min(m,n)^2 * max(m,n)) per step, and a
# full-size centroid (e.g. 2048x2048 wgud on OLMoE) would dominate the step
# time on CPU. The low-rank machinery (U,V,C and the ng heads) always fits
# under the default; raise via --muon-max-dim on GPU to cover everything.
MUON_NS_MAX_DIM = 512


# ---- fit step speed: bf16 autocast with an HONEST real-step probe ---------

_BF16_PROBE_CACHE = {}


def probe_bf16_matmul(device="cpu", n=1024, iters=3):
    """DIAGNOSTICS ONLY since 2026-09-02.4 (the auto decision no longer
    uses it): fp32/bf16 time ratio of ONE big n x n matmul (>1 = bf16
    faster). Kept as a cheap ISA sanity check - but see _resolve_fit_autocast
    for why a big-matmul ratio alone must NOT decide the fit dtype: the real
    step is many small GEMMs + per-step weight casts + backward."""
    key = (str(device), n)
    if key in _BF16_PROBE_CACHE:
        return _BF16_PROBE_CACHE[key]
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def _run(dt):
        x, y = a.to(dt), b.to(dt)
        for _ in range(iters):
            x @ y
        t0 = time.monotonic()
        for _ in range(iters):
            x @ y
        return (time.monotonic() - t0) / iters

    _run(torch.float32)
    _run(torch.bfloat16)                       # warmup / oneDNN JIT
    t32, t16 = _run(torch.float32), _run(torch.bfloat16)
    ratio = t32 / max(t16, 1e-9)
    _BF16_PROBE_CACHE[key] = ratio
    return ratio


_HONEST_PROBE_CACHE = {}
_HONEST_PROBE_SEED = 0xC0FFEE          # local probe RNG - the fit's own RNG
                                       # stream must stay bit-identical


def _time_fit_arms(mod, p, Xf, tgt, noise_pool, x_std, z_all, n_tok, bs,
                   device, opts, jitter):
    """The honest dtype probe: time 1 warmup + 3 REAL fit steps per arm
    (fp32 vs bf16-autocast) with the actual optimizers on the actual pool,
    so weight casts, small GEMMs and the backward pass are all counted -
    unlike a bare big-matmul probe. Returns (t32, t16): the MIN per-arm
    step time (min = the least noise-contaminated measurement). Afterwards
    the initial parameters are restored and the optimizer state cleared,
    and all index draws come from a LOCAL generator, so the fit proceeds
    bit-for-bit as if the probe had never run."""
    saved = [t.detach().clone() for t in p]
    pgen = torch.Generator().manual_seed(_HONEST_PROBE_SEED)
    dev = torch.device(device).type
    ts = {}
    for enabled in (False, True):
        ac = torch.autocast(device_type=dev, dtype=torch.bfloat16,
                            enabled=enabled)
        arm = []
        for k in range(4):                     # 1 warmup + 3 timed
            t0 = time.monotonic() if k else None
            ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=pgen)
            xb = Xf[ix].to(device, non_blocking=True)
            yb = tgt[ix].to(device, non_blocking=True)
            if x_std is not None:
                jx = torch.randint(0, n_tok, (min(bs, n_tok),),
                                   generator=pgen)
                xb = xb + noise_pool[jx].to(device)
            zb = None
            if mod.mode == "router":
                if x_std is not None:
                    with ac:
                        zb = mod._z(xb).detach()
                else:
                    zb = z_all[ix].to(device, non_blocking=True)
            with ac:
                out = (mod.forward_from_z(xb, zb) if zb is not None
                       else mod.forward_from_c(xb, *mod._coords(xb)))
            loss = F.mse_loss(out.float(), yb)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()
            float(loss.item())
            if k:
                arm.append(time.monotonic() - t0)
        ts[enabled] = min(arm)
    with torch.no_grad():                      # undo the probe's updates
        for t, q in zip(p, saved):
            t.copy_(q)
    for o in opts:
        o.state.clear()                        # optimizer starts fresh
    return ts[False], ts[True]


def _resolve_fit_autocast(autocast, mod, p, method, opts, device, bs, steps,
                          jitter, prep, log_prefix=""):
    """--fit-autocast -> (enabled, note). "bf16" forces, "fp32" disables.
    "auto" (2026-09-02.4) measures the REAL fit step in both dtypes and
    picks bf16 only if the full step is >=1.2x faster. The 2026-09-02.3
    probe timed ONE big 1024x1024 matmul and could switch bf16 on where the
    actual step (many small GEMMs + per-step weight casts + backward) is
    net SLOWER - exactly the user-reported "8.2 became slower than 8.1"
    regression. The probe costs ~8 real steps ONCE per (geometry, bs,
    threads) and its measured seconds land in the muon-split line, so the
    decision is auditable in the log."""
    if autocast == "bf16":
        return True, "bf16 (forced)"
    if autocast == "fp32":
        return False, "fp32 (forced)"
    key = (str(device), method, int(bs), bool(jitter and jitter > 0),
           int(torch.get_num_threads()), mod.mode, hasattr(mod, "sh_gu"),
           tuple((nm, tuple(getattr(mod, nm).shape))
                 for nm in mod.field_names))
    hit = _HONEST_PROBE_CACHE.get(key)
    if hit is not None:
        return hit
    if steps < 16:
        res = (False, "fp32 (auto: fit too short for a meaningful probe)")
        _HONEST_PROBE_CACHE[key] = res
        return res
    print(f"    {log_prefix} fit dtype probe: timing 8 real fit steps "
          f"fp32 vs bf16 (one-time per geometry)...", flush=True)
    Xf, tgt, noise_pool, x_std, z_all, n_tok = prep
    t32, t16 = _time_fit_arms(mod, p, Xf, tgt, noise_pool, x_std, z_all,
                              n_tok, bs, device, opts, jitter)
    ratio = t32 / max(t16, 1e-9)
    meas = f"fp32 {t32:.2f}s vs bf16 {t16:.2f}s"
    if ratio >= 1.2:
        res = (True, f"bf16 (auto: real-step probe x{ratio:.2f} faster - "
                     f"{meas})")
    else:
        res = (False, f"fp32 (auto: real-step probe x{ratio:.2f} - {meas})")
    _HONEST_PROBE_CACHE[key] = res
    return res


def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz orthogonalization of G (~ UV^T of its SVD). Runs in
    bf16 on the portrait orientation; cost ~ min(m,n)^2 * max(m,n) per step."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalized update for 2-D params (Keller Jordan's
    Muon, 2024/2025 - the nanoGPT speedrun optimizer). Each step: nesterov
    momentum buffer -> Newton-Schulz -> p -= lr * U (FLAT scale since
    2026-09-02.2: the earlier max(1, rows/cols)**0.5 boost multiplied the
    nominal lr by 3.5-8x on the tall rank-r factors and coincided with most
    blocks diverging on the real model; flat keeps the effective step inside
    the user-proven lr 0.003-0.005 range). 1-D params are not accepted -
    pair with Adam at the call site if any appear."""

    def __init__(self, params, lr=4e-3, momentum=0.95, nesterov=True,
                 ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, mom, nes, ns = (group["lr"], group["momentum"],
                                group["nesterov"], group["ns_steps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(p)
                buf = st["mom"]
                buf.lerp_(p.grad, 1.0 - mom)
                u = p.grad.lerp(buf, mom) if nes else buf
                v = _zeropower_via_newtonschulz5(u, steps=ns)
                p.add_(v.to(p.dtype), alpha=-lr)   # flat scale (see docstring)
        return loss


def _mdev(model):
    """Device of the model's weights. Works for a plain HF model, a bnb
    (device_map) model and BlockStreamRunner (it exposes .device and also
    delegates attribute lookups to the wrapped model)."""
    d = getattr(model, "device", None)
    if isinstance(d, torch.device):
        return d
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------- quantization

def dequant_weight(w):
    """Weight -> fp32 tensor. bitsandbytes 4-bit weights (Params4bit) are
    dequantized."""
    if not torch.is_tensor(w):
        raise TypeError(f"expected a tensor, got {type(w).__name__}")
    qs = getattr(w, "quant_state", None)
    if qs is None:
        return w.detach().float()
    import bitsandbytes.functional as BF
    return BF.dequantize_4bit(w.data, quant_state=qs).float()


def linear_tensor(v):
    """A tensor or Linear-like module -> weight tensor (with 4-bit dequant)."""
    if isinstance(v, nn.Module):
        v = getattr(v, "weight", None)
        if v is None:
            raise RuntimeError("the module has no .weight")
    return dequant_weight(v)


def router_weight(gate):
    """Router weight (dequantized, fp32) - for the fit."""
    w = getattr(gate, "weight", None)
    if w is None:
        raise RuntimeError("the router has no .weight")
    return dequant_weight(w)


# ---------------------------------------------------------------- discovery

def find_moe_blocks(model):
    """All MoE blocks (having both experts and gate) with full names."""
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, "experts") and hasattr(m, "gate")]


def block_router_bias(block):
    """e_score_correction_bias (hy_v3): expert selection score shift, fp32."""
    b = getattr(block, "e_score_correction_bias", None)
    return None if b is None else b.detach().clone().float()


def block_shared_weights(block):
    """Shared-expert weights (hy_v3): fused gate_up (2dffs,d) + down (d,dffs)."""
    se = getattr(block, "shared_experts", None)
    if se is None:
        return None
    g = linear_tensor(se.gate_proj)
    u = linear_tensor(se.up_proj)
    d2 = linear_tensor(se.down_proj)
    return dict(sh_gu=torch.cat([g, u], dim=0), sh_dn=d2)


def block_geometry(block, config):
    gate, experts = block.gate, block.experts
    n = int(getattr(gate, "num_experts", 0) or getattr(experts, "num_experts", 0)
            or (len(experts) if hasattr(experts, "__len__") else 0))
    meta = dict(top_k=int(getattr(gate, "top_k",
                                  getattr(config, "num_experts_per_tok", 2))),
                norm_topk=bool(getattr(config, "norm_topk_prob", False)),
                hidden_act=str(getattr(config, "hidden_act", "silu")))
    if str(getattr(config, "model_type", "")) == "hy_v3":      # NanoColibri and kin
        meta.update(router_kind="sigmoid_bias",
                    router_scale=float(getattr(config, "router_scaling_factor", 1.0)),
                    fp32_combine=bool(getattr(config, "enable_moe_fp32_combine", True)),
                    dff_shexp=int(getattr(getattr(block, "shared_experts", None),
                                          "intermediate_size", 0) or 0))
    gu = getattr(experts, "gate_up_proj", None)          # v5 fused (may be 4-bit)
    if gu is not None and not isinstance(gu, nn.ModuleList):
        w = linear_tensor(gu)                             # (N, 2dff, d)
        return dict(n_exp=n or w.shape[0], d_model=w.shape[2],
                    d_ff=w.shape[1] // 2, **meta)
    if isinstance(experts, nn.ModuleList):                # v4: w1/w3/w2
        e0 = experts[0]
        for nm in ("w1", "gate_proj"):
            if hasattr(e0, nm):
                w = dequant_weight(getattr(e0, nm).weight)
                return dict(n_exp=n or len(experts), d_model=w.shape[1],
                            d_ff=w.shape[0], **meta)
        raise RuntimeError(f"cannot extract geometry from {type(e0).__name__}")
    dff = int(getattr(experts, "intermediate_dim", 0))
    if not dff:  # fallback by weight shape
        for p in experts.parameters():
            if p.ndim == 3 and p.shape[1] == 2 * p.shape[2]:
                dff = p.shape[1] // 2
                break
    d = int(getattr(gate, "hidden_dim", 0) or router_weight(gate).shape[1])
    return dict(n_exp=n, d_model=d, d_ff=dff, **meta)


def expert_stack(block):
    """Fused expert weights: gate_up (N,2dff,d), down (N,d,dff)."""
    exp = block.experts
    gu = getattr(exp, "gate_up_proj", None)
    dn = getattr(exp, "down_proj", None)
    if gu is not None and not isinstance(gu, nn.ModuleList):
        # v5 fused (incl. 4-bit bitsandbytes) -> dequant to fp32
        return linear_tensor(gu), linear_tensor(dn)
    if isinstance(exp, nn.ModuleList):                     # v4 layout w1/w3/w2

        def pick(e, names):
            for nm in names:
                if hasattr(e, nm):
                    v = getattr(e, nm)
                    return v.weight if hasattr(v, "weight") else v
            raise AttributeError(f"none of {names} found in {type(e).__name__}")

        w1 = torch.stack([linear_tensor(pick(e, ("w1", "gate_proj"))) for e in exp])
        w3 = torch.stack([linear_tensor(pick(e, ("w3", "up_proj"))) for e in exp])
        w2 = torch.stack([linear_tensor(pick(e, ("w2", "down_proj"))) for e in exp])
        return torch.cat([w1, w3], dim=1), w2
    raise RuntimeError(f"cannot extract experts from {type(exp).__name__}")


def field_accounting(geoms, rank, norouter_hidden=None):
    """Bytes (fp16) of full experts vs the field over all MoE blocks."""
    full = field = 0
    for g in geoms:
        d, dff, n = g["d_model"], g["d_ff"], g["n_exp"]
        full += n * (2 * dff * d + d * dff) * 2
        field += ((2 * dff * d + d * dff)                    # centroids
                  + rank * (2 * dff + d) + rank * (d + dff)  # U,V
                  + 2 * n * rank) * 2                        # coordinates C
        if norouter_hidden is not None:                      # ng heads (both)
            h = int(norouter_hidden)
            field += (2 * (h * d + rank * h) if h > 0 else 2 * rank * d) * 2
    return full, field


# ------------------------------------------------------------- field module

class FieldSparseMoe(nn.Module):
    """MoE block "field engine". gate=None -> fit (manual router from the
    weight); gate=<base router> -> deploy (contract identical to the base
    block)."""

    def __init__(self, geom, rank, gate=None, gate_w=None, act_fn=F.silu,
                 dtype=torch.float32, init=None, gate_bias=None, shared=None,
                 mode="router", norouter_hidden=0):
        super().__init__()
        d, dff, r = geom["d_model"], geom["d_ff"], rank
        self.d, self.k = d, geom["top_k"]
        self.norm = geom["norm_topk"]
        self.act_fn = act_fn
        self.mode = mode                    # "router" | "norouter"
        self.norouter_hidden = int(norouter_hidden)
        self.n_exp = int(geom["n_exp"])
        self.router_kind = str(geom.get("router_kind", "softmax"))
        self.router_scale = float(geom.get("router_scale", 1.0))
        if mode == "norouter":
            # FieldNoRouter (2026-09-02.1, --replace-router): NO router at all -
            # coordinates come from a smooth direct map g(x): linear
            # (norouter_hidden=0, the toy-proven config) or a one-GELU-hidden
            # MLP. Init mirrors the B1-safe scheme: the "C-like" head output
            # (ng*_w2) starts at ZERO (the field output is the pure centroid,
            # so the FIT GUARD baseline holds), the feature part (ng*_w1) is
            # small random noise for symmetry breaking.
            h = self.norouter_hidden
            nrng = torch.Generator().manual_seed(7654321)   # separate RNG: the
            for pref, inp in (("ngu", "gu"), ("ngd", "dn")):  # U,V stream stays
                if h > 0:                                   # bit-identical to
                    self.register_parameter(                # the router build
                        f"{pref}_w1", nn.Parameter(
                            (torch.randn(h, d, generator=nrng) * 0.02).to(dtype)))
                    self.register_parameter(
                        f"{pref}_w2", nn.Parameter(torch.zeros(r, h, dtype=dtype)))
                else:
                    self.register_parameter(
                        f"{pref}_w2", nn.Parameter(torch.zeros(r, d, dtype=dtype)))
        elif gate is not None:
            self.gate = gate                                 # original router
        else:
            self.register_parameter("gw", nn.Parameter(gate_w.clone().float()))
            self.gw.requires_grad_(False)
        if self.router_kind == "sigmoid_bias":               # hy_v3: selection bias
            eb = torch.zeros(geom["n_exp"]) if gate_bias is None \
                else gate_bias.clone().float()
            self.register_buffer("eb", eb)
        if int(geom.get("dff_shexp", 0)):                    # hy_v3: shared experts
            sh = shared or {}
            dffs = int(geom["dff_shexp"])
            self.register_buffer("sh_gu", sh.get("sh_gu", torch.zeros(2 * dffs, d)).float())
            self.register_buffer("sh_dn", sh.get("sh_dn", torch.zeros(d, dffs)).float())
        self.field_names = []
        # GUARD FIX (B1): U,V = randn*0.02 (as in the PoC), NOT zeros. With a
        # zero init the U,V,C gradients are identically zero (products of
        # parameters), so the fit stays on the centroid line forever. C=0 is
        # fine: the field's initial output stays purely centroidal, no noise.
        grng = torch.Generator().manual_seed(1234567)
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(
                f"w{nm}d", nn.Parameter(torch.zeros(out, inp, dtype=dtype)))
            self.register_parameter(
                f"U{nm}", nn.Parameter((torch.randn(out, r, generator=grng) * 0.02).to(dtype)))
            self.register_parameter(
                f"V{nm}", nn.Parameter((torch.randn(inp, r, generator=grng) * 0.02).to(dtype)))
            self.field_names += [f"w{nm}d", f"U{nm}", f"V{nm}"]
        if mode == "norouter":
            self.field_names += ["ngu_w2", "ngd_w2"]
            if self.norouter_hidden > 0:
                self.field_names += ["ngu_w1", "ngd_w1"]
        else:
            for nm in ("Cgu", "Cdn"):
                self.register_parameter(
                    nm, nn.Parameter(torch.zeros(geom["n_exp"], r, dtype=dtype)))
                self.field_names.append(nm)
        if init is not None:                                 # transfer from the fit
            with torch.no_grad():
                for k, v in init.items():
                    getattr(self, k).copy_(v.to(dtype))

    def _z(self, x):
        if hasattr(self, "gate"):
            if self.router_kind == "sigmoid_bias":           # hy_v3: gate(x, bias)
                out = self.gate(x, self.eb)
                logits, scores, idx = out[0], out[1], out[2]
            else:
                out = self.gate(x)
                if isinstance(out, (tuple, list)):           # v5 router
                    logits, scores, idx = out[0], out[1], out[2]
                else:  # nn.Linear / Linear4bit -> own top-k
                    logits = out
                    probs = F.softmax(logits.float(), dim=-1)
                    scores, idx = torch.topk(probs, self.k, dim=-1)
                    if self.norm:
                        scores = scores / scores.sum(-1, keepdim=True)
        else:
            logits = x @ self.gw.t()
            if self.router_kind == "sigmoid_bias":           # HYV3TopKRouter semantics
                rw = torch.sigmoid(logits)
                scores, idx = torch.topk(rw + self.eb, self.k, dim=-1)
                scores = rw.gather(-1, idx)
                scores = scores / (scores.sum(-1, keepdim=True) + 1e-20) \
                    * self.router_scale
            else:
                probs = F.softmax(logits, dim=-1)
                scores, idx = torch.topk(probs, self.k, dim=-1)
                if self.norm:
                    scores = scores / scores.sum(-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, scores)

    def _coords(self, x):
        """Smooth coordinate map g(x) (norouter mode): returns (cgu, cdn),
        each (T,r). Linear when norouter_hidden=0, one-GELU MLP otherwise."""
        out = []
        for pref in ("ngu", "ngd"):
            w1 = getattr(self, f"{pref}_w1", None)
            feat = F.gelu(x @ w1.t()) if w1 is not None else x
            out.append(feat @ getattr(self, f"{pref}_w2").t())
        return out

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        if self.mode == "norouter":
            cgu, cdn = self._coords(x)
            y = self.forward_from_c(x, cgu, cdn)
        else:
            z = self._z(x)
            y = self.forward_from_z(x, z)
        if hasattr(self, "sh_gu"):                           # hy_v3: shared experts
            sg, su = (x @ self.sh_gu.t()).chunk(2, dim=-1)
            ys = (self.act_fn(sg) * su) @ self.sh_dn.t()
            y = (y.float() + ys.float()).to(y.dtype)         # fp32 combine as in base
        return y.view(B, T, -1)

    def forward_from_z(self, x, z):
        """Router-mode field branch with a PRECOMPUTED routing z (fit fast
        path: the router is frozen, so z depends only on x - computed once
        per pool, not once per step)."""
        return self.forward_from_c(x, z @ self.Cgu, z @ self.Cdn)

    def forward_from_c(self, x, cgu, cdn):
        """FIELD BRANCH only, from explicit coordinates (T,r). Shared experts
        are NOT computed here: in the fit they are frozen buffers folded into
        the target once (an additive constant does not change the gradients of
        the field parameters), which removes ~3/4 of the per-step FLOPs on
        hy_v3 blocks.
        x: (T,d), cgu/cdn: (T,r)."""
        gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        return h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t()

    def fit_params(self):
        return [getattr(self, n) for n in self.field_names]


def fit_field_module(mod, X, Y, steps, bs, lr, device, log_prefix="",
                     log_every=100, guard=True, method="muon", seed=None,
                     jitter=0.0, early_stop=0, muon_max_dim=MUON_NS_MAX_DIM,
                     autocast="auto", muon_ns_steps=5):
    """Fit the field on (MoE input -> output) pairs with Adam-family
    optimizers. X,Y: (T,d) cpu bf16.

    Fit fast paths (identical gradients, big speedups on hy_v3 blocks):
      - the router is frozen, so the routing z is computed ONCE per pool, not
        once per step (with jitter the routing follows the noisy input);
      - shared experts are frozen buffers entering the output additively:
        their contribution is computed once and SUBTRACTED FROM THE TARGET
        (MSE(field + shared, Y) == MSE(field, Y - shared) for the field
        gradients), removing ~3/4 of the per-step FLOPs on hy_v3.

    jitter: Gaussian noise on the fit INPUTS, scaled per-dimension by the
    activation std (targets stay exact). Cheap augmentation for a data-starved
    pool: at ~1.6 pairs/dim, jitter 0.6 gave +2.6% error vs +16.8% without.
    NOTE on scale: the win is variance-reduction for a SMALL pool; at 16+
    pairs/dim the bias (the field is taught noise-invariance the base model
    does not have) starts to outweigh it - prefer 0.0 or <=0.15 there. The
    SYSTEMATIC deploy-time shift is what refine rounds (--refine-rounds) fix;
    jitter only covers the random part. With jitter > 0 the guard mse is
    measured on noisy inputs, so guard numbers are not comparable across
    different jitter values.

    early_stop: 0 = off; N > 0 - checkpoint the mse every N steps (after a
    warmup of ~steps/4) and stop after 2 consecutive checkpoints with <0.5%
    relative improvement (saves steps on plateauing blocks).

    Best-state tracking (divergence insurance): whenever the EMA of the
    per-step minibatch mse improves meaningfully, the parameters are
    snapshotted (cpu copy). If the fit blows up late (a degenerate batch can
    spike Adam: a real run died with mse 0.23 -> 0.10 by step 500, then 33.9
    at step 599, which tripped the old final-vs-first guard), the loop bails
    out early and the best snapshot is restored; a slow end-of-fit drift
    (>2% above the best EMA) restores it too. The returned mse and the FIT
    GUARD therefore always judge the weights that will actually be shipped,
    not a possibly exploded last step.

    method: "muon" | "muon-cosine" | "adam" | "adamw" | "adam-cosine" | "rmsprop"
      muon        - MANDATORY DEFAULT since 2026-09-02.1: momentum +
                    Newton-Schulz orthogonalized update on the matrix params
                    (same quality in ~half the Adam steps in the user's
                    measurements; FLAT update scale since 2026-09-02.2 -
                    see the Muon docstring). Params with min(shape) >
                    muon_max_dim (the full-size centroids on CPU) take a
                    paired constant-lr Adam - the NS iteration is quadratic
                    in min(m,n).
      muon-cosine - Muon + cosine lr decay to ~0 on all its optimizers
      adam        - plain Adam, constant lr (classic baseline / rollback)
      adamw       - AdamW with a small weight decay (less drift in U,V)
      adam-cosine - Adam + cosine lr decay to ~0 (the pre-Muon best)
      rmsprop     - RMSProp, an alternative for instable blocks
    autocast: "auto" | "bf16" | "fp32" - run the per-step forward under
      torch.autocast(bfloat16); the params and ALL optimizer state stay
      fp32, the loss is computed on out.float(), so only the matmuls switch
      dtype. The step is matmul-bound (~85% fwd+bwd, ~9% Newton-Schulz,
      ~2% Adam - measured), and on AMX/AVX512-BF16 CPUs bf16 matmuls are
      2-6x faster: measured 0.18 -> 0.10 s/step at bs 4096 with IDENTICAL
      fit quality (best-ema 0.06008 fp32 vs 0.06005 bf16, seed 0; 0.06051
      vs 0.06050, seed 1; synthetic block, 150 steps, no divergence).
      "auto" (2026-09-02.4) times REAL fit steps in both dtypes ONCE per
      (geometry, bs, threads) and keeps bf16 only where the full step
      (matmuls + weight casts + backward) is >=1.2x faster - the
      2026-09-02.3 big-matmul-only probe could enable bf16 where the real
      step is net slower, which a user measured as "8.2 slower than 8.1".
      The probe restores the initial parameters, clears optimizer state and
      uses a local RNG, so trajectories are bit-identical to a no-probe run.
      Forced bf16/fp32 skip the probe entirely.
    muon_ns_steps: Newton-Schulz iterations per Muon step (default 5; 3
      saves ~40% of the NS time at <=0.5% best-ema cost - relevant only on
      slow CPUs where NS is a visible share of the step).
    seed: None -> the global RNG (legacy behavior); int -> a LOCAL generator
    with that seed, making the fit deterministic per block and safe for
    parallel workers.

    FIT GUARD (B1): at step 0 C=0 -> the output is the pure centroid, i.e.
    the first loss IS the centroid baseline. The fit MUST improve on it; if
    not, the field degraded (the classic zero-init U,V,C failure) - raise an
    error instead of quietly shipping garbage into the artifact."""
    if method not in FIT_METHODS:
        raise ValueError(f"unknown fit method '{method}' "
                         f"(available: {', '.join(FIT_METHODS)})")
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))  # per-block, parallel-safe
    else:
        torch.manual_seed(5)   # legacy global-RNG mode (single-threaded fits)
    p = mod.fit_params()
    for t in p:
        t.requires_grad_(True)
    # init snapshot for the divergence diagnostics (who moved the most)
    p0 = [t.detach().clone() for t in p]
    wd = 0.01 if method == "adamw" else 0.0
    use_sched = method in ("adam-cosine", "muon-cosine")
    opts = []
    split_info = None
    if method in ("muon", "muon-cosine"):
        mu = [t for t in p if t.ndim >= 2 and min(t.shape) <= muon_max_dim]
        ad = [t for t in p if not (t.ndim >= 2 and min(t.shape) <= muon_max_dim)]
        if mu:
            opts.append(Muon(mu, lr=lr, ns_steps=max(1, int(muon_ns_steps))))
        if ad:
            opts.append(torch.optim.Adam(ad, lr=lr))
        big = [n for n in mod.field_names
               if getattr(mod, n).ndim >= 2
               and min(getattr(mod, n).shape) > muon_max_dim]
        split_info = (len(mu), len(ad), big)
    elif method == "rmsprop":
        opts.append(torch.optim.RMSprop(p, lr=lr))
    elif method == "adamw":
        opts.append(torch.optim.AdamW(p, lr=lr, weight_decay=wd))
    else:
        opts.append(torch.optim.Adam(p, lr=lr))
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(steps, 1))
              for o in opts] if use_sched else []
    n_tok = X.shape[0]
    x_std = X.float().std(dim=0) if jitter and jitter > 0 else None
    noise_pool = None
    if x_std is not None:
        # generate the jitter ONCE per block (4M gaussians per step is ~7-10%
        # of the step time); each step then gathers with an INDEPENDENT index
        # draw, so (row, noise) pairings stay fresh while the values come from
        # a fixed Gaussian sample - same regularization, much cheaper
        with torch.no_grad():
            noise_pool = (torch.randn(X.shape, generator=gen) if gen is not None
                          else torch.randn(X.shape)) * (x_std * jitter)
    with torch.no_grad():
        Xf = X.float()
        # frozen router: z fixed per row (router mode). In norouter mode the
        # coordinates depend on TRAINABLE heads - nothing to precompute.
        z_all = mod._z(Xf) if mod.mode == "router" else None
        if hasattr(mod, "sh_gu"):
            sg, su = (Xf @ mod.sh_gu.t()).chunk(2, dim=-1)
            # fold the frozen shared branch into the target once
            tgt = Y.float() - (mod.act_fn(sg) * su) @ mod.sh_dn.t()
        else:
            tgt = Y.float()

    # autocast resolution for ALL methods (the 2026-09-02.3 code created
    # `ac` only in the muon branch - any adam-family fit crashed with an
    # UnboundLocalError at the `with ac:` line below; 2026-09-02.4 fix).
    # Under "auto" this may run the honest dtype probe (see its docstring):
    # ~8 real steps, once per (geometry, bs, threads), params/state restored
    # afterwards, fit RNG untouched.
    ac_on, ac_note = _resolve_fit_autocast(
        autocast, mod, p, method, opts, device, bs, steps, jitter,
        (Xf, tgt, noise_pool, x_std, z_all, n_tok), log_prefix=log_prefix)
    ac = torch.autocast(device_type=torch.device(device).type,
                        dtype=torch.bfloat16, enabled=ac_on)
    if split_info is not None:
        n_mu, n_ad, big = split_info
        print(f"    {log_prefix} muon split: {n_mu} NS params + "
              f"{n_ad} adam params (min-dim > {muon_max_dim}: "
              f"{', '.join(big) or '-'}); ns_steps={muon_ns_steps}; "
              f"autocast {ac_note}", flush=True)

    # best-state machinery (see the docstring: divergence insurance)
    def _snapshot():
        return [t.detach().to("cpu", copy=True) for t in p]

    def _restore():
        with torch.no_grad():
            for t, snap in zip(p, best_state):
                t.copy_(snap.to(t.device))

    last = first = None
    ckpt_ema = None
    stall = 0
    warm = min(150, max(10, steps // 4)) if early_stop else steps
    ema = None                              # EMA of the minibatch mse
    best_score = None                       # ema at the last snapshot
    best_state = None                       # field params at that point (cpu)
    last_snap = 0
    restored = False
    t0 = time.monotonic()                  # AFTER the probe: s/step stays honest
    for s in range(steps):
        if gen is not None:
            ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
        else:
            ix = torch.randint(0, n_tok, (min(bs, n_tok),))
        xb = Xf[ix].to(device, non_blocking=True)   # fp32 rows of the pool
        yb = tgt[ix].to(device, non_blocking=True)
        if x_std is not None:
            if gen is not None:
                jx = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
            else:
                jx = torch.randint(0, n_tok, (min(bs, n_tok),))
            xb = xb + noise_pool[jx].to(device)      # fresh (row, noise) pairing
            # routing follows the noisy input, as at deploy; detach - the
            # router is frozen, autograd must not build a backward graph
            # through it (nothing trainable lives upstream of z)
            with ac:
                zb = mod._z(xb).detach() if mod.mode == "router" else None
        else:
            zb = (z_all[ix].to(device, non_blocking=True)
                  if mod.mode == "router" else None)
        with ac:                                     # bf16 matmuls, fp32 params
            if zb is not None:
                out = mod.forward_from_z(xb, zb)
            else:
                out = mod.forward_from_c(xb, *mod._coords(xb))
        loss = F.mse_loss(out.float(), yb)           # fp32 loss as before
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        for sc in scheds:
            sc.step()
        last = float(loss.item())
        if first is None:
            first = last
        if s % log_every == 0 or s == steps - 1:
            sps = (time.monotonic() - t0) / (s + 1)
            print(f"    {log_prefix} step {s}: mse {last:.5f} "
                  f"({sps:.2f} s/step, eta {sps * (steps - s - 1) / 60:.1f}m)",
                  flush=True)
        if not math.isfinite(last):
            print(f"    {log_prefix} non-finite loss at step {s} - best "
                  f"state restored, stopping this block early", flush=True)
            if best_state is not None:
                _restore()
                restored = True
            break
        ema = last if ema is None else 0.9 * ema + 0.1 * last
        if best_state is None:
            best_score, best_state, last_snap = ema, _snapshot(), s
        elif ema < best_score * 0.998 and s - last_snap >= 20:
            best_score, best_state, last_snap = ema, _snapshot(), s
        elif best_score > 0 and ema > 2.0 * best_score:
            movers = sorted(
                ((nm, float((getattr(mod, nm).detach() - q).norm()
                            / (q.norm() + 1e-12)))
                 for nm, q in zip(mod.field_names, p0)),
                key=lambda kv: -kv[1])[:3]
            print(f"    {log_prefix} loss diverged at step {s} (mse "
                  f"{last:.5f}, ema {ema:.5f} vs best {best_score:.5f}) - "
                  f"best state restored, stopping this block early; biggest "
                  f"moves: " + ", ".join(f"{nm} x{m:.2f}"
                                        for nm, m in movers), flush=True)
            _restore()
            restored = True
            break
        if early_stop and s >= warm and (s % early_stop == 0 or s == steps - 1):
            # 2 consecutive flat checkpoints (<0.5% relative) -> stop early;
            # the checkpoint is an EMA of the minibatch mse (a raw single-batch
            # value is noisy: one lucky/unlucky batch can hide or fake a plateau)
            cema = last if ckpt_ema is None else 0.5 * ckpt_ema + 0.5 * last
            if ckpt_ema is not None and cema > ckpt_ema * (1.0 - 0.005):
                stall += 1
                if stall >= 2:
                    print(f"    {log_prefix} early stop at step {s} "
                          f"(mse plateaued)", flush=True)
                    break
            else:
                stall = 0
            ckpt_ema = cema
    for t in p:
        t.requires_grad_(False)
    if best_state is not None and not restored and math.isfinite(ema) \
            and ema > best_score * 1.02:
        # a slow late drift the 2x bailout never caught - still ship the
        # best point, not the final one
        print(f"    {log_prefix} mse drifted up by the end (ema {ema:.5f} "
              f"vs best {best_score:.5f}) - best state restored", flush=True)
        _restore()
        restored = True
    if restored:
        # the shipped weights are the restored best ones: re-measure the loss
        # on a few fresh batches (same jitter convention as the fit) so the
        # guard and the report describe what is actually being shipped
        tot, n_ev = 0.0, 8
        with torch.no_grad():
            for _ in range(n_ev):
                ix = (torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
                      if gen is not None
                      else torch.randint(0, n_tok, (min(bs, n_tok),)))
                xb = Xf[ix].to(device, non_blocking=True)
                yb = tgt[ix].to(device, non_blocking=True)
                if x_std is not None:
                    jx = (torch.randint(0, n_tok, (min(bs, n_tok),),
                                        generator=gen)
                          if gen is not None
                          else torch.randint(0, n_tok, (min(bs, n_tok),)))
                    xb = xb + noise_pool[jx].to(device)
                if mod.mode == "router":
                    zb = mod._z(xb).detach()
                    tot += float(F.mse_loss(mod.forward_from_z(xb, zb),
                                            yb).item())
                else:
                    tot += float(F.mse_loss(
                        mod.forward_from_c(xb, *mod._coords(xb)),
                        yb).item())
        last = tot / n_ev
        print(f"    {log_prefix} best-state re-eval: mse {last:.5f}",
              flush=True)
        best_state = None
    if guard and first is not None and (not math.isfinite(last)
                                        or last > 0.98 * first):
        if restored:
            # diverged AND the best state is at/below the baseline: a lower
            # lr has a real chance to fix it -> raise so the pipeline's
            # fit_block_with_retry can retry the block once at lr/2
            raise RuntimeError(
                f"FIT GUARD: mse did not drop ({first:.5f} -> {last:.5f}) - "
                f"the fit diverged and even the best state is not below the "
                f"centroid baseline (auto-retried once at lr/2 by the "
                f"pipeline; to disable the check: --skip-fit-guard)")
        # weak but NOT diverged: the field did learn a little - crashing a
        # multi-hour run over a <2% block would be worse than shipping it
        # (the number is recorded in mse.json / the report either way)
        print(f"    {log_prefix} WARNING: weak fit (mse {first:.5f} -> "
              f"{last:.5f}, <2% below the centroid baseline) - shipping "
              f"anyway (the block is not diverged)", flush=True)
    zc_names = ("Cgu", "Cdn") if mod.mode == "router" else ("ngu_w2", "ngd_w2")
    if guard and all(float(getattr(mod, n).abs().max()) == 0.0
                     for n in zc_names):
        # the exact B1 bug signature: centroids learn (mse drops) but the
        # rank path stays frozen at zeros -> artifact = "one averaged expert"
        raise RuntimeError(
            f"FIT GUARD: coordinates {zc_names[0]}/{zc_names[1]} stayed "
            f"exactly zero - the rank path is not learning (the classic "
            f"zero-init bug). The fit degraded into a centroid; to disable "
            f"the check: --skip-fit-guard")
    if guard and first:
        print(f"    {log_prefix} guard: mse {first:.5f} -> {last:.5f} "
              f"({100 * (first - last) / max(first, 1e-12):.1f}% below the "
              f"centroid baseline)", flush=True)
    return last


def build_deploy_block(orig_gate, geom, rank, act_fn, fit_init, dtype,
                       mode="router", norouter_hidden=0):
    """Deploy block: the original base router + trained field parameters
    (router mode), or the router-free smooth-map field (mode="norouter" -
    orig_gate is ignored/None). fit_init - dict of field tensors (keys =
    field_names); a fit module itself is also accepted (compat with the old
    call)."""
    if hasattr(fit_init, "field_names"):
        fit_init = {n: getattr(fit_init, n).detach() for n in fit_init.field_names}
    return FieldSparseMoe(geom, rank, gate=orig_gate, act_fn=act_fn,
                          dtype=dtype, init=fit_init, mode=mode,
                          norouter_hidden=norouter_hidden)


def swap_block(model, block_name, new_mod):
    parent = (model.get_submodule(block_name.rsplit(".", 1)[0])
              if "." in block_name else model)
    setattr(parent, block_name.rsplit(".", 1)[-1], new_mod)


# ------------------------------------------------------------- calibration

@torch.no_grad()
def collect_pairs(model, blocks, batches, per_layer_cap, flush_dir=None,
                  tag="pairs"):
    """Run the calibration batches; capture (input, output) of every MoE block.

    flush_dir set -> as soon as a block reaches the cap its buffer is flushed
    to disk (RAM does not grow with the block count); returns a list of
    (path, tokens). flush_dir=None -> the old behavior: a list of (X, Y)
    tensors in RAM."""
    store = [{"X": [], "Y": [], "n": 0, "path": None} for _ in blocks]

    def flush(i):
        st = store[i]
        if st["n"] == 0:
            return
        st["path"] = os.path.join(flush_dir, f"{tag}_blk{i}.pt")
        save_pairs_block(st["X"], st["Y"], st["path"])
        st["X"], st["Y"] = [], []

    def make_hook(i):
        def hook(m, args, output):
            st = store[i]
            if st["n"] >= per_layer_cap:
                return
            x = args[0].detach()
            y = output.detach() if torch.is_tensor(output) else output[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu()
            y = y.reshape(-1, y.shape[-1]).to(torch.bfloat16).cpu()
            st["X"].append(x)
            st["Y"].append(y)
            st["n"] += x.shape[0]
            if flush_dir is not None and st["n"] >= per_layer_cap:
                flush(i)
        return hook

    hs = [b.register_forward_hook(make_hook(i)) for i, (_, b) in enumerate(blocks)]
    try:
        for batch in batches:
            model(**batch)
            if all(st["n"] >= per_layer_cap for st in store):
                break
    finally:
        for h in hs:
            h.remove()
        if flush_dir is not None:
            for i in range(len(store)):
                if store[i]["path"] is None:
                    flush(i)
    if flush_dir is not None:
        return [(st["path"], st["n"]) for st in store]
    return [(torch.cat(st["X"])[:per_layer_cap],
             torch.cat(st["Y"])[:per_layer_cap]) for st in store]


def save_pairs_block(X_list, Y_list, path):
    """Merge a block's buffers and save them to disk (bf16)."""
    X = torch.cat(X_list) if X_list else torch.empty(0)
    Y = torch.cat(Y_list) if Y_list else torch.empty(0)
    torch.save({"X": X, "Y": Y}, path)
    return path


def load_pairs_block(path):
    """A block's pairs from disk: (X, Y) bf16 on CPU."""
    d = torch.load(path, map_location="cpu")
    return d["X"], d["Y"]


def make_batches(ids, ctx, bsz, n_windows, device, gen):
    out = []
    for _ in range(n_windows):
        starts = torch.randint(0, len(ids) - ctx - 1, (bsz,), generator=gen)
        x = torch.stack([ids[s:s + ctx] for s in starts])
        out.append(dict(input_ids=x.to(device)))
    return out


# ------------------------------------------------- centroids and disk cache

@torch.no_grad()
def _col_means(w):
    """Mean over experts for a fused stack (N,out,inp) WITHOUT the full fp32
    stack. 4-bit is dequantized entirely into native fp16 (half the RAM of
    fp32), then sums accumulate over chunks in fp32. Peak RAM: dequant
    N*out*inp*2 bytes (OLMoE gate_up: ~0.5 GB) instead of a ~2 GB fp32 stack."""
    if not torch.is_tensor(w):
        w = w.weight                                  # Params4bit / plain weight
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        import bitsandbytes.functional as BF
        full = BF.dequantize_4bit(w.data, quant_state=qs)
    else:
        full = w
    n = full.shape[0]
    ch = max(1, (1 << 26) // (int(full.shape[1]) * int(full.shape[2])))
    acc = None
    for s in range(0, n, ch):
        m = full[s:s + ch].float().sum(0)
        acc = m if acc is None else acc + m
        del m
    return acc / n


@torch.no_grad()
def expert_means(block):
    """Centroids (expert weight means) without the full fp32 stack.
    Returns (m_gu (2dff,d) fp32, m_dn (d,dff) fp32) - the field init.
    Replaces expert_stack(...).mean(0): saves ~2 GB RAM on an OLMoE block."""
    exp = block.experts
    gu = getattr(exp, "gate_up_proj", None)
    dn = getattr(exp, "down_proj", None)
    if gu is not None and not isinstance(gu, nn.ModuleList):
        return _col_means(gu), _col_means(dn)
    if isinstance(exp, nn.ModuleList):                     # v4 layout w1/w3/w2
        def pick(e, names):
            for nm in names:
                if hasattr(e, nm):
                    v = getattr(e, nm)
                    return v.weight if hasattr(v, "weight") else v
            raise AttributeError(f"none of {names} found in {type(e).__name__}")
        mg = md = None
        n = len(exp)
        for e in exp:
            g = dequant_weight(pick(e, ("w1", "gate_proj")))
            u = dequant_weight(pick(e, ("w3", "up_proj")))
            d2 = dequant_weight(pick(e, ("w2", "down_proj")))
            s_gu = torch.cat([g, u], dim=0).sum(0)
            s_dn = d2.sum(0)
            mg = s_gu if mg is None else mg + s_gu
            md = s_dn if md is None else md + s_dn
        return mg / n, md / n
    raise RuntimeError(f"cannot extract experts from {type(exp).__name__}")


@torch.no_grad()
def eval_logits_cache_disk(model, ids, ctx, n_chunks, lp_dir, seed=17):
    """Base-model log-probs on fixed chunks -> lp_XXX.pt files (per chunk).
    RAM: one chunk at a time (not the whole cache). Returns X, Y - the tokens
    (cheap); the log-prob caches themselves live on disk."""
    os.makedirs(lp_dir, exist_ok=True)
    model.eval()
    dev = _mdev(model)
    g = torch.Generator().manual_seed(seed)
    X, Y = [], []
    for i in range(n_chunks):
        s = int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g))
        xc = ids[s:s + ctx]          # stays on CPU: it is cached in eval_tokens.pt
        logits = model(input_ids=xc.unsqueeze(0).to(dev)).logits[0]
        lp = torch.log_softmax(logits, dim=-1).to(torch.bfloat16).cpu()
        torch.save(lp, os.path.join(lp_dir, f"lp_{i:03d}.pt"))
        X.append(xc)
        Y.append(ids[s + 1:s + ctx + 1])
    return X, Y


@torch.no_grad()
def eval_vs_cache_disk(model, X, Y, lp_dir, n_max=None):
    """CE/KL against the on-disk cache of base log-probs (per chunk, flat RAM).
    n_max - verify only the first n chunks (for the base the cache is its own:
    a full check of all chunks = wasted model passes, expensive in streaming)."""
    model.eval()
    dev = _mdev(model)
    ces, kls = [], []
    pairs = list(zip(X, Y))
    if n_max:
        pairs = pairs[:n_max]
    for i, (x, y) in enumerate(pairs):
        lp = torch.load(os.path.join(lp_dir, f"lp_{i:03d}.pt"), map_location="cpu")
        logits = model(input_ids=x.unsqueeze(0).to(dev)).logits[0]
        # ppl computed the SAME way as the base ppl from the cache: gather from
        # bf16 log_softmax. An identical-to-base model then gives a delta ppl
        # of exactly 0.0 (no systematic float32-CE vs bf16-cache shift).
        lp_q = torch.log_softmax(logits, dim=-1).to(torch.bfloat16)
        y = y.to(lp_q.device)
        ces.append(float(-(lp_q.gather(1, y.unsqueeze(1))[:, 0]).float().mean()))
        lq = torch.log_softmax(logits.float(), dim=-1)
        lp = lp.to(lq.device)
        p = lp.float().exp()
        kls.append(float((p * (lp.float() - lq)).sum(-1).mean()))
        del lp
    ce = sum(ces) / len(ces)
    return dict(ce=ce, ppl=math.exp(ce),
                kl_bits=(sum(kls) / len(kls)) / math.log(2))


# ------------------------------------------------------------------ metrics

@torch.no_grad()
def base_metrics_from_cache(lp_dir, X, Y):
    """Base-model CE/ppl FROM THE CACHE of its log-probs (no model needed).
    lp files = log_softmax(logits) of the base on the same chunks as the
    Y tokens."""
    ces = []
    for i, y in enumerate(Y):
        lp = torch.load(os.path.join(lp_dir, f"lp_{i:03d}.pt"), map_location="cpu")
        ces.append(float(-(lp.float().gather(1, y.unsqueeze(1))[:, 0]).mean()))
        del lp
    ce = sum(ces) / len(ces)
    return dict(ce=ce, ppl=math.exp(ce), kl_bits=0.0)


@torch.no_grad()
def eval_logits_cache(model, ids, ctx, n_chunks, seed=17):
    """Base-model log-probs on fixed chunks (cache for KL after the swap)."""
    model.eval()
    dev = _mdev(model)
    g = torch.Generator().manual_seed(seed)
    X, Y, LP = [], [], []
    for _ in range(n_chunks):
        s = int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g))
        xc = ids[s:s + ctx]
        logits = model(input_ids=xc.unsqueeze(0).to(dev)).logits[0]
        LP.append(torch.log_softmax(logits, dim=-1).to(torch.bfloat16).cpu())
        X.append(xc)
        Y.append(ids[s + 1:s + ctx + 1])
    return X, Y, LP


@torch.no_grad()
def eval_vs_cache(model, X, Y, LP):
    """CE/KL of the converted model against the base log-prob cache."""
    model.eval()
    dev = _mdev(model)
    ces, kls = [], []
    for x, y, lp in zip(X, Y, LP):
        logits = model(input_ids=x.unsqueeze(0).to(dev)).logits[0]
        lq = torch.log_softmax(logits.float(), dim=-1)
        ces.append(F.cross_entropy(logits.float(), y.to(logits.device)).item())
        lp = lp.to(logits.device)
        p = lp.float().exp()
        kls.append(float((p * (lp.float() - lq)).sum(-1).mean()))
    ce = sum(ces) / len(ces)
    return dict(ce=ce, ppl=math.exp(ce),
                kl_bits=(sum(kls) / len(kls)) / math.log(2))


@torch.no_grad()
def generate_text(model, ids, tokenizer, n_prompt=12, n_new=48, seed=3,
                  repetition_penalty=1.0):
    g = torch.Generator().manual_seed(seed)
    s = int(torch.randint(0, len(ids) - n_prompt - 1, (1,), generator=g))
    prompt = ids[s:s + n_prompt].unsqueeze(0).to(_mdev(model))
    kwargs = dict(max_new_tokens=n_new, do_sample=False)
    # compressed models have slightly shifted logits - plain greedy decoding
    # is the worst case for degenerate repetition loops; a standard
    # repetition penalty keeps the demo text readable (applied to BOTH the
    # base and the field generations, so the comparison stays fair)
    if repetition_penalty and repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    out = model.generate(prompt, **kwargs)
    return tokenizer.decode(out[0].cpu())


# ------------------------------------------------------------------ saving

def render_modeling_file(base_cls, router_cls, router_mod, norouter=False):
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tpl = f.read()
    src = (tpl.replace("@@BASE@@", base_cls)
              .replace("@@ROUTER@@", router_cls)
              .replace("@@ROUTER_MOD@@", router_mod))
    if norouter:
        # the template stays untouched: the no-router runtime is APPENDED and
        # FieldSparseMoe becomes a config-driven factory at import time
        src += _NOROUTER_APPENDIX
    return src


# Self-contained runtime for --replace-router artifacts (2026-09-02.1). No
# router import is used: coordinates come from the smooth map g(x) learned at
# fit time (linear, or one-GELU-hidden MLP). The factory dispatch keeps base
# artifacts byte-compatible: without router_mode=="smooth" in cfg.field the
# original template class is used as before.
_NOROUTER_APPENDIX = '''

# ---------------------------------------------------------------------------
# FieldNoRouter runtime (auto-appended 2026-09-02.1, artifact built with
# --replace-router): the discrete router is replaced by a smooth coordinate
# map g(x); removes the piecewise-constant c(z) / top-k boundary jump.
import torch.nn.functional as _F
from transformers.activations import ACT2FN as _ACT2FN


class FieldSparseMoeNR(nn.Module):
    """Field block WITHOUT the router: cgu/cdn = g_gu(x), g_dn(x)."""

    def __init__(self, config):
        super().__init__()
        fi = config.field
        d, dff, r = fi["d_model"], fi["d_ff"], fi["rank"]
        h = int(fi.get("norouter_hidden", 0) or 0)
        self.d = d
        self.mode = "norouter"
        self.norouter_hidden = h
        self.n_exp = int(fi.get("n_exp", 0))
        self.field_names = ["wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn",
                            "ngu_w2", "ngd_w2"]
        self.act_fn = _ACT2FN[config.hidden_act]
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(
                f"w{nm}d", nn.Parameter(torch.zeros(out, inp)))
            self.register_parameter(f"U{nm}", nn.Parameter(torch.zeros(out, r)))
            self.register_parameter(f"V{nm}", nn.Parameter(torch.zeros(inp, r)))
        if h > 0:
            self.field_names += ["ngu_w1", "ngd_w1"]
            for pref in ("ngu", "ngd"):
                self.register_parameter(
                    f"{pref}_w1", nn.Parameter(torch.zeros(h, d)))
                self.register_parameter(
                    f"{pref}_w2", nn.Parameter(torch.zeros(r, h)))
        else:
            for pref in ("ngu", "ngd"):
                self.register_parameter(
                    f"{pref}_w2", nn.Parameter(torch.zeros(r, d)))

    def _coord(self, nm, x):
        w1 = getattr(self, f"{nm}_w1", None)
        feat = _F.gelu(x @ w1.t()) if w1 is not None else x
        return feat @ getattr(self, f"{nm}_w2").t()

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        cgu, cdn = self._coord("ngu", x), self._coord("ngd", x)
        gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        hh = self.act_fn(g) * u
        y = hh @ self.wdnd.t() + (hh @ self.Vdn * cdn) @ self.Udn.t()
        return y.view(B, T, -1)


_FieldSparseMoeBase = FieldSparseMoe


def FieldSparseMoe(config):  # noqa: F811 - factory: dispatch by cfg.field
    fi = getattr(config, "field", None) or {}
    if str((fi or {}).get("router_mode", "base")) == "smooth":
        return FieldSparseMoeNR(config)
    return _FieldSparseMoeBase(config)
'''


def find_field_modules(model):
    """Field modules after the swap (router mode: gate + Cgu; norouter mode:
    the ng heads instead of the gate)."""
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, "field_names") and (hasattr(m, "Cgu") or
                                              hasattr(m, "ngu_w2"))]


def field_geometry(mod, config):
    """Geometry from a field module (for cfg.field on save)."""
    n_exp = int(mod.Cgu.shape[0]) if hasattr(mod, "Cgu") else int(mod.n_exp)
    return dict(n_exp=n_exp, d_model=int(mod.d),
                d_ff=int(mod.wgud.shape[0] // 2), top_k=int(mod.k),
                norm_topk=bool(mod.norm),
                hidden_act=str(getattr(config, "hidden_act", "silu")),
                router_mode=("smooth" if getattr(mod, "mode", "router") == "norouter"
                             else "base"),
                norouter_hidden=int(getattr(mod, "norouter_hidden", 0)))


def save_field_model(model, tokenizer, out_dir, rank, accounting, meta):
    """Save a NORMAL HF model with the field: config + weights +
    modeling_field.py."""
    os.makedirs(out_dir, exist_ok=True)
    mods = find_field_modules(model)
    if not mods:
        raise RuntimeError("no field modules found in the model - was the swap done?")
    first = mods[0][1]
    norouter = getattr(first, "mode", "router") == "norouter"
    base_cls = type(model).__name__
    if norouter:
        # no router in the runtime: the placeholder import must still be a
        # valid Python import line (harmless, unused)
        router_cls, router_mod = "object", "builtins"
    else:
        router_cls = type(first.gate).__name__
        router_mod = type(first.gate).__module__

    cfg = model.config
    cfg.architectures = ["FieldForCausalLM"]
    cfg.auto_map = {"AutoModelForCausalLM": "modeling_field.FieldForCausalLM"}
    geom = field_geometry(first, cfg)
    cfg.field = dict(rank=int(rank), n_layers=len(mods),
                     base_class=base_cls, **geom)
    model.save_pretrained(out_dir, safe_serialization=True,
                          max_shard_size=meta.get("max_shard_size", "4GB"))
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)

    # v5 save_pretrained overwrites architectures with the model class - fix
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg_json = json.load(f)
    cfg_json["architectures"] = ["FieldForCausalLM"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    import shutil
    shutil.rmtree(os.path.join(out_dir, "__pycache__"), ignore_errors=True)

    src = render_modeling_file(base_cls, router_cls, router_mod,
                               norouter=norouter)
    path = os.path.join(out_dir, "modeling_field.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    import py_compile
    py_compile.compile(path, doraise=True)

    full_b, field_b = accounting
    with open(os.path.join(out_dir, "field_meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rank=rank, full_experts_mb=full_b / 1e6,
                       field_mb=field_b / 1e6,
                       ratio=full_b / max(field_b, 1), **meta), f,
                  ensure_ascii=False, indent=2)
    return out_dir


# ------------------------------------------------- streaming artifact build

def write_field_artifact(src, out_dir, pool_dir, fit_dir, rank, dtype,
                         max_shard_bytes=2_000_000_000, gguf=None, profile=None,
                         io_workers=1, io_cache="disk", replace_router=False,
                         norouter_hidden=0):
    """Assemble the artifact (a plain HF model with the field) WITHOUT loading
    the model into RAM.

    Backbone tensors are copied one by one (from the dequant checkpoint src
    or, when gguf is given, straight from the GGUF with on-the-fly dequant -
    no checkpoint needed), expert tensors are skipped (the field takes their
    place), field parameters come from fit_dir/fit_blk{i}.pt (the fit of a
    specific rank), calibration metadata (art_meta, geometry) comes from
    pool_dir (the pool shared by all ranks). RAM: one tensor + a shard buffer
    (~2 GB) - the full model never loads.

    Artifact: config.json (auto_map + cfg.field), weights (safetensors +
    index), tokenizer, modeling_field.py, field_meta.json.
    """
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(pool_dir, "art_meta.json"), encoding="utf-8") as f:
        am = json.load(f)
    init0 = torch.load(os.path.join(pool_dir, "init_blk0.pt"), map_location="cpu")
    geom = dict(init0["geom"])

    # ---- config: plain model + auto_map + the field description
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["FieldForCausalLM"]
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_field.FieldForCausalLM"}
    cfg["field"] = dict(rank=int(rank), n_layers=int(am["n_layers"]),
                        base_class=am["base_cls"], **geom)
    if replace_router:
        # FieldNoRouter artifact (2026-09-02.1): the appended runtime block
        # dispatches on router_mode=="smooth" (base artifacts unaffected)
        cfg["field"]["router_mode"] = "smooth"
        cfg["field"]["norouter_hidden"] = int(norouter_hidden)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)

    # ---- tokenizer and gen config: files from the source
    TOK_PARTS = ("tokenizer", "special_tokens_map", "added_tokens", "vocab",
                 "merges", "chat_template", "generation_config")
    for name in sorted(os.listdir(src)):
        low = name.lower()
        if name.endswith(".safetensors") or name == "config.json":
            continue
        if low.endswith((".model", ".jinja", ".txt")) or \
           any(low.startswith(p) for p in TOK_PARTS):
            shutil.copy2(os.path.join(src, name), os.path.join(out_dir, name))

    # ---- weights: backbone from src (experts skipped) + the field from fit files
    # (norouter artifacts also skip the frozen router weight/bias: the runtime
    # has no router, keeping them only triggers an UNEXPECTED load warning)
    skip_router = set()
    if replace_router:
        for ln in (am.get("block_names") or []):
            skip_router.add(f"{ln}.gate.weight")
            skip_router.add(f"{ln}.e_score_correction_bias")
    shard, shards_keys, written = {}, [], 0
    total = 0

    def flush():
        nonlocal shard, written
        if not shard:
            return
        p = os.path.join(out_dir, f"_shard_tmp_{len(shards_keys):05d}.safetensors")
        save_file(shard, p, metadata={"format": "pt"})
        shards_keys.append((p, list(shard.keys())))
        shard, written = {}, 0

    src_files = sorted(fn for fn in os.listdir(src) if fn.endswith(".safetensors"))
    gs = None
    if gguf is not None:                      # backbone straight from the GGUF
        from hf_gguf_to_hf import GgufHfSource
        gs = GgufHfSource(gguf, io_workers=io_workers,
                          cache_ram=(io_cache == "ram"))
    elif not src_files:
        raise RuntimeError(f"no safetensors files in {src}")
    for fn in src_files:
        with safe_open(os.path.join(src, fn), framework="pt") as f:
            for key in f.keys():
                if ".experts." in key or key in skip_router:   # the field replaces them
                    continue
                t = f.get_tensor(key)
                shard[key] = t
                nb = t.numel() * t.element_size()
                total += nb
                written += nb
                if written >= max_shard_bytes:
                    flush()
    if gs is not None:
        for key in gs.keys():
            if ".experts." in key or key in skip_router:   # the field replaces them
                continue
            shard[key] = gs.get(key)
            t = shard[key]
            nb = t.numel() * t.element_size()
            total += nb
            written += nb
            if written >= max_shard_bytes:
                flush()
    for i in range(int(am["n_layers"])):
        fit = torch.load(os.path.join(fit_dir, f"fit_blk{i}.pt"),
                         map_location="cpu")
        # layer names MAY differ from the block ordinal (hy_v3 has MoE layers
        # 1..N with layer 0 dense); new caches keep block_names
        layer = (am.get("block_names") or [f"model.layers.{j}.mlp"
                 for j in range(int(am["n_layers"]))])[i]
        for nm, t in fit.items():
            key = f"{layer}.{nm}"
            t = t.to(dtype)
            shard[key] = t
            nb = t.numel() * t.element_size()
            total += nb
            written += nb
            if written >= max_shard_bytes:
                flush()
    flush()

    # ---- rename shards and write the index
    n = len(shards_keys)
    weight_map = {}
    for idx, (p, keys) in enumerate(shards_keys, 1):
        fname = "model.safetensors" if n == 1 else \
            f"model-{idx:05d}-of-{n:05d}.safetensors"
        os.replace(p, os.path.join(out_dir, fname))
        for key in keys:
            weight_map[key] = fname
    with open(os.path.join(out_dir, "model.safetensors.index.json"),
              "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f, ensure_ascii=False, indent=2)

    # ---- modeling_field.py + field metadata
    with open(os.path.join(out_dir, "modeling_field.py"), "w",
              encoding="utf-8") as f:
        f.write(render_modeling_file(am["base_cls"], am["router_cls"],
                                     am["router_mod"],
                                     norouter=bool(replace_router)))
    import py_compile
    py_compile.compile(os.path.join(out_dir, "modeling_field.py"), doraise=True)
    shutil.rmtree(os.path.join(out_dir, "__pycache__"), ignore_errors=True)
    meta_out = dict(rank=int(rank), n_layers=int(am["n_layers"]),
                    backbone="copy of the source backbone (experts skipped)",
                    field_dtype=str(dtype))
    if profile:
        meta_out["profile"] = profile
    with open(os.path.join(out_dir, "field_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    return out_dir
