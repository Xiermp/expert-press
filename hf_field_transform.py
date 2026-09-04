# version: 2026-09-04.3 - RESUME FIX: save_pairs_block is atomic (tmp +
#   os.replace) - an interrupted flush cannot leave a torn pairs_blk file;
#   eval_logits_cache_disk reuses existing loadable lp_XXX.pt chunks (the
#   chunk starts are seed-fixed, so a resumed base pass skips finished chunks).
# version: 2026-09-04.2 - UPDATE-10 speed rebuild + the user's three holes:
#   (1) SVD init of U,V,C from the STACKED expert deltas (shared-basis,
#       streaming randomized SVD, expert_basis_init; the mean-delta SVD is
#       degenerate: sum dW = 0) - captures ~50-70% of the delta energy at
#       step 0 (toy: 2.8x better heldout at equal steps / same quality in
#       ~3x fewer steps); (2) muon/muon-cosine fit methods with a BY-NAME
#       split: only U*/V* operator factors go through Newton-Schulz, the C
#       coordinate tables never do (they would couple unrelated experts),
#       --muon-max-dim/--muon-ns-steps; (3) jitter routing follows the CLEAN
#       anchor row (the target was produced by the base model on the clean
#       input - routing on the noisy input paired targets with the wrong
#       experts' coordinates); (4) honest real-step autocast probe (_time_fit_arms,
#       _resolve_fit_autocast, 1 warmup + 3 timed steps per arm, min, >=1.2x
#       rule, cache keyed by geometry, RNG 0xC0FFEE) + bf16-autocast fit
#       (params stay fp32, matmuls under torch.autocast) - 1.7-1.8x per step
#       on the toy box, --fit-autocast auto|on|off.
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
import contextlib
import hashlib
import json
import math
import os
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "modeling_field_template.py")

# available fit optimizer kinds (see fit_field_module / --fit-method)
FIT_METHODS = ("adam", "adamw", "adam-cosine", "rmsprop", "muon", "muon-cosine")
MUON_METHODS = ("muon", "muon-cosine")


# ------------------------------------------------------------ muon machinery
def _ns_orth(G, steps=5):
    """Newton-Schulz orthogonalization (Muon's 5-iteration quadratic run,
    coefficients from Keller Jordan's implementation). Returns a matrix with
    (approximately) singular values ~= 1 and the same singular vectors as G.
    fp32 on CPU: bf16 NS can fall back to emulation on machines without a
    bf16 ISA and end up SLOWER than fp32 for these small matrices."""
    X = G.float().clone()
    X = X / (X.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.t()
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return (X.t() if tr else X).to(G.dtype)


class _Muon(torch.optim.Optimizer):
    """Muon update for the slim operator factors (U*/V*): nesterov momentum,
    NS-orthogonalized update, spectral-norm lr scaling sqrt(max(1, m/n)).
    The update magnitude follows the GRADIENT direction, not its scale
    (NS normalizes) - pairs well with Adam on the scale-sensitive params."""

    def __init__(self, params, lr, ns_steps=5, momentum=0.95):
        super().__init__(list(params), dict(lr=lr, ns_steps=ns_steps,
                                            momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for gr in self.param_groups:
            for p in gr["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(p)
                st["mom"].mul_(gr["momentum"]).add_(p.grad)
                upd = p.grad.lerp(st["mom"], gr["momentum"])   # nesterov
                u = _ns_orth(upd, gr["ns_steps"])
                p.add_(u, alpha=-gr["lr"] * math.sqrt(
                    max(1.0, p.size(0) / max(1, p.size(1)))))
        return None


def _muon_split(field_names, params, muon_max_dim):
    """BY-NAME split (hole-2 fix): only U*/V* operator factors are Muon
    candidates (further gated by min(shape) <= muon_max_dim, so on real
    models the big centroid matrices wgud/wdnd stay on Adam unless the user
    raises the dim cap). Cgu/Cdn (coordinate tables: row i = expert i's
    coordinates - NS would orthogonogonalize across UNRELATED experts) and
    the router gw always stay on Adam."""
    mu, ad = [], []
    for name, t in zip(field_names, params):
        if name.startswith(("U", "V")) and t.ndim == 2 \
                and min(t.shape) <= muon_max_dim:
            mu.append(t)
        else:
            ad.append(t)
    return mu, ad


class _OptPair:
    """Adam (+/- Muon) behind one handle: zero_grad/step/param_groups - so
    the existing loop (and lr warmup) works unchanged for muon methods."""

    def __init__(self, opts):
        self.opts = [o for o in opts if o is not None]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for o in self.opts:
            o.step()
        return None

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]


# --------------------------------------------------- honest autocast probe
_HONEST_PROBE_CACHE = {}
_HONEST_PROBE_LOCK = threading.Lock()
_HONEST_PROBE_SEED = 0xC0FFEE


def _time_fit_arms(mod, Xf, tgt, z_all, bs, lr, device, method, jitter,
                   noise_pool, train_router, muon_max_dim, muon_ns_steps,
                   n_steps=4):
    """HONEST real-step probe: run actual fit steps for both dtype arms on
    the real module with the real data and take the min of 3 timed steps
    after 1 warmup (oneDNN JIT / cache effects). Params are snapshotted
    before and restored after (try/finally), the RNG is a PRIVATE generator
    (0xC0FFEE) so the global/fit RNG state is untouched, and both arms see
    the SAME batch indices - a fair race. Returns (t_fp32, t_bf16, mse_fp32,
    mse_bf16)."""
    p = mod.fit_params()
    snap = [t.detach().clone() for t in p]
    rg = [t.requires_grad for t in p]     # restored in finally: the probe may
    gen = torch.Generator().manual_seed(_HONEST_PROBE_SEED)   # run MID-SETUP
    n_tok = Xf.shape[0]
    muon = method in MUON_METHODS
    try:
        out = []
        for arm_amp in (False, True):
            for t in p:
                t.requires_grad_(True)
            if muon:
                mu, ad = _muon_split(mod.field_names, p, muon_max_dim)
                opt = _OptPair([torch.optim.Adam(ad, lr=lr) if ad else None,
                                _Muon(mu, lr=lr, ns_steps=muon_ns_steps)
                                if mu else None])
            else:
                opt = torch.optim.Adam(p, lr=lr)
            ac = (torch.autocast(device_type=str(device), dtype=torch.bfloat16)
                  if arm_amp else contextlib.nullcontext())
            ts, last = [], float("nan")
            for s in range(n_steps):
                ix = torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
                xb = Xf[ix].to(device, non_blocking=True)
                yb = tgt[ix].to(device, non_blocking=True)
                if noise_pool is not None:
                    jx = torch.randint(0, n_tok, (min(bs, n_tok),),
                                       generator=gen)
                    xb = xb + noise_pool[jx].to(device)
                    zb = (mod._z(Xf[ix].to(device)) if train_router
                          else z_all[ix].to(device, non_blocking=True))
                else:
                    zb = (mod._z(xb) if train_router
                          else z_all[ix].to(device, non_blocking=True))
                t0 = time.perf_counter()
                with ac:
                    o = mod.forward_from_z(xb, zb)
                loss = F.mse_loss(o.float(), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if s > 0:
                    ts.append(time.perf_counter() - t0)
                last = float(loss.item())
            out.append((min(ts), last))
            del opt
        return out[0][0], out[1][0], out[0][1], out[1][1]
    finally:
        with torch.no_grad():
            for t, s in zip(p, snap):
                t.copy_(s)
        for t, r in zip(p, rg):
            t.requires_grad_(r)


def _resolve_fit_autocast(mod, Xf, tgt, z_all, bs, lr, device, method,
                          jitter, noise_pool, train_router, mode,
                          muon_max_dim, muon_ns_steps, log_prefix=""):
    """Decide whether the fit runs its matmuls under bf16 autocast.
    mode on/off -> direct; auto -> the honest probe (>=1.2x advantage to
    switch), cached per geometry so identical blocks do not re-probe."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    key = (str(device), method, int(bs), round(float(jitter), 4),
           torch.get_num_threads(), bool(noise_pool is not None),
           bool(train_router), bool(hasattr(mod, "sh_gu")),
           Xf.shape[0], tuple(tuple(t.shape) for t in mod.fit_params()))
    with _HONEST_PROBE_LOCK:
        hit = _HONEST_PROBE_CACHE.get(key)
    if hit is not None:
        print(f"    {log_prefix} probe (cached): autocast "
              f"{'ON' if hit else 'OFF'}", flush=True)
        return hit
    t32, t16, m32, m16 = _time_fit_arms(
        mod, Xf, tgt, z_all, bs, lr, device, method, jitter, noise_pool,
        train_router, muon_max_dim, muon_ns_steps)
    speed = t32 / max(t16, 1e-9)
    decision = speed >= 1.2
    with _HONEST_PROBE_LOCK:
        _HONEST_PROBE_CACHE[key] = decision
    print(f"    {log_prefix} honest probe: fp32 {t32 * 1e3:.0f} ms vs bf16 "
          f"{t16 * 1e3:.0f} ms per step ({speed:.2f}x, mse {m32:.4f}/{m16:.4f}) "
          f"-> autocast {'ON' if decision else 'OFF'}", flush=True)
    return decision


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


def field_accounting(geoms, rank):
    """Bytes (fp16) of full experts vs the field over all MoE blocks."""
    full = field = 0
    for g in geoms:
        d, dff, n = g["d_model"], g["d_ff"], g["n_exp"]
        full += n * (2 * dff * d + d * dff) * 2
        field += ((2 * dff * d + d * dff)                    # centroids
                  + rank * (2 * dff + d) + rank * (d + dff)  # U,V
                  + 2 * n * rank) * 2                        # coordinates C
    return full, field


# ------------------------------------------------------------- field module

class FieldSparseMoe(nn.Module):
    """MoE block "field engine". gate=None -> fit (manual router from the
    weight); gate=<base router> -> deploy (contract identical to the base
    block)."""

    def __init__(self, geom, rank, gate=None, gate_w=None, act_fn=F.silu,
                 dtype=torch.float32, init=None, gate_bias=None, shared=None):
        super().__init__()
        d, dff, r = geom["d_model"], geom["d_ff"], rank
        self.d, self.k = d, geom["top_k"]
        self.norm = geom["norm_topk"]
        self.act_fn = act_fn
        self.router_kind = str(geom.get("router_kind", "softmax"))
        self.router_scale = float(geom.get("router_scale", 1.0))
        if gate is not None:
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
        for nm in ("Cgu", "Cdn"):
            self.register_parameter(
                nm, nn.Parameter(torch.zeros(geom["n_exp"], r, dtype=dtype)))
            self.field_names.append(nm)
        if init is not None:                                 # transfer from the fit
            with torch.no_grad():
                for k, v in init.items():
                    if k in self.field_names:   # ignore aux keys (e.g. gw_tuned)
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

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        z = self._z(x)
        y = self.forward_from_z(x, z)
        if hasattr(self, "sh_gu"):                           # hy_v3: shared experts
            sg, su = (x @ self.sh_gu.t()).chunk(2, dim=-1)
            ys = (self.act_fn(sg) * su) @ self.sh_dn.t()
            y = (y.float() + ys.float()).to(y.dtype)         # fp32 combine as in base
        return y.view(B, T, -1)

    def forward_from_z(self, x, z):
        """FIELD BRANCH only, with a PRECOMPUTED routing z (fit fast path: the
        router is frozen, so z depends only on x - computed once per pool,
        not once per step). Shared experts are NOT computed here: in the fit
        they are frozen buffers folded into the target once (an additive
        constant does not change the gradients of the field parameters),
        which removes ~3/4 of the per-step FLOPs on hy_v3 blocks.
        x: (T,d), z: (T,n_exp)."""
        cgu, cdn = z @ self.Cgu, z @ self.Cdn                # movement seed
        gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        return h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t()

    def fit_params(self):
        return [getattr(self, n) for n in self.field_names]


def fit_field_module(mod, X, Y, steps, bs, lr, device, log_prefix="",
                     log_every=100, guard=True, method="adam", seed=None,
                     jitter=0.0, early_stop=0, guard_warmup=None,
                     strict_guard=False, lr_warmup=0, train_router=False,
                     router_anchor=0.0, autocast="auto", muon_max_dim=512,
                     muon_ns_steps=5):
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
    ROUTING under jitter (hole-3 fix): z follows the CLEAN anchor row - the
    target was produced by the base model on the clean input, so routing on
    the noisy input would pair targets with the wrong experts' coordinates
    (every top-k flip under noise adds an irreducible mse floor).
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

    method: "adam" | "adamw" | "adam-cosine" | "rmsprop" | "muon" | "muon-cosine"
      adam        - plain Adam, constant lr (classic baseline)
      adamw       - AdamW with a small weight decay (less drift in U,V)
      adam-cosine - Adam + cosine lr decay to ~0 (usually the best final mse)
      rmsprop     - RMSProp, an alternative for instable blocks
      muon        - BY-NAME split: U*/V* operator factors get Muon's
                    NS-orthogonalized updates, everything else (w*, C*, the
                    router) stays Adam. Cgu/Cdn NEVER go through Newton-
                    Schulz: their rows are independent per-expert coordinates
                    and NS would couple unrelated experts (user report,
                    hole 2); the toy bench shows no gain from C-in-muon.
      muon-cosine - muon + cosine lr decay (manual, warmup-aware)
      muon_max_dim gates U/V additionally by min(shape) <= dim: 512 (default)
      keeps the big centroid matrices wgud/wdnd on Adam on real models; on
      small geometries they join Muon automatically (the toy's best arm),
      or force it with a higher cap - NS on big matrices costs ~40-50%
      extra step time.

    autocast: "auto" -> the honest real-step probe decides (_time_fit_arms:
    1 warmup + 3 timed real steps per dtype arm, min, >=1.2x rule, cached
    per geometry); "on"/"off" force it. bf16 autocast keeps parameters fp32
    (exact fp32 gradients) and runs the forward matmuls in bf16 via oneDNN
    (toy: 1.7-1.8x per step); on machines without a bf16 ISA the probe
    keeps fp32.

    SVD INIT (hole 1): when the caller passes Ugu/Vgu/Cgu/... via init=
    (see expert_basis_init), the fit STARTS from the shared-basis
    reconstruction of the expert deltas (~50-70% of the delta energy at
    step 0) instead of randn*0.02/zeros. The guard's "first" baseline is
    then the init-state mse (the report also prints the pure-centroid
    reference), and best-state tracking ships the init itself if the fit
    cannot improve on it.
    seed: None -> the global RNG (legacy behavior); int -> a LOCAL generator
    with that seed, making the fit deterministic per block and safe for
    parallel workers.

    lr_warmup: linear lr ramp over the first N steps (Adam adaptation
    window). A toy bench (scripts/bench_toy_router_guard.py) showed a flat
    start turning into a real fit (-11.7% vs the centroid baseline on a
    block that previously stalled), and the mid-fit divergence events moved
    past the warmup. 0 = off (legacy constant lr).

    train_router / router_anchor: ALSO train the manual router gw on the
    same pairs (the "original router" joins the expert rebuild). z is then
    recomputed every step (grads flow into gw) and the loss gains
    anchor * ||gw - gw0||^2 / ||gw0||^2, keeping the router close to the
    original. NOTE from the same toy bench: once the field fit has
    converged, the router is usually NOT the bottleneck - expect small
    gains; --refine-rounds (input-shift refits) is the stronger lever.

    FIT GUARD (B1): at step 0 C=0 -> the output is the pure centroid, i.e.
    the first loss IS the centroid baseline. The fit MUST improve on it; if
    not, the field degraded (the classic zero-init U,V,C failure).

    guard_warmup: the 2x divergence bail ("loss diverged ... stopping this
    block early") is ARMED ONLY AFTER this many steps; None = auto
    (max(30, steps//10)); 0 = the old always-armed behavior. The old guard
    fired on Adam's normal early overshoot BEFORE the optimizer had time to
    adapt, cut the fit at step ~2-12 and shipped a near-centroid block; the
    toy bench reproduced it (bail at step 1-5 on lr 4e-3-8e-3) and showed
    the warmup does not change well-behaved fits at all (the bail only
    reacts, the trajectory is untouched).

    strict_guard: False (default) -> a flat end-of-fit (mse in (0.98, 1.0]x
    of the baseline) WARNS and ships the best state - no gain, no harm, the
    run continues; True -> the old hard error. A fit that ended WORSE than
    the centroid baseline still raises in both modes (that is real
    degradation, never worth shipping)."""
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
    wd = 0.01 if method == "adamw" else 0.0
    gw0 = None
    if train_router and hasattr(mod, "gw"):
        gw0 = mod.gw.detach().clone()          # anchor target (original router)
        p = p + [mod.gw]                       # BEFORE the optimizer: Adam must
        for t in (mod.gw,):                    # see the router as trainable
            t.requires_grad_(True)
    if method in MUON_METHODS:
        mu, ad = _muon_split(mod.field_names, p, muon_max_dim)
        if train_router and gw0 is not None:
            ad = ad + [mod.gw]     # the router never goes through NS (hole 2)
        opt = _OptPair([torch.optim.Adam(ad, lr=lr) if ad else None,
                        _Muon(mu, lr=lr, ns_steps=muon_ns_steps) if mu else None])
        sched = None               # muon-cosine: manual cosine in the loop
    elif method == "rmsprop":
        opt = torch.optim.RMSprop(p, lr=lr)
    elif method == "adamw":
        opt = torch.optim.AdamW(p, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(p, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(steps - max(lr_warmup, 0), 1)) \
        if method == "adam-cosine" else None
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
        z_all = mod._z(Xf)                           # frozen router: z fixed per row
        if hasattr(mod, "sh_gu"):
            sg, su = (Xf @ mod.sh_gu.t()).chunk(2, dim=-1)
            # fold the frozen shared branch into the target once
            tgt = Y.float() - (mod.act_fn(sg) * su) @ mod.sh_dn.t()
        else:
            tgt = Y.float()

    use_amp = _resolve_fit_autocast(mod, Xf, tgt, z_all, bs, lr, device,
                                    method, jitter, noise_pool, train_router,
                                    autocast, muon_max_dim, muon_ns_steps,
                                    log_prefix=log_prefix)
    ac = (torch.autocast(device_type=str(device), dtype=torch.bfloat16)
          if use_amp else contextlib.nullcontext())

    # best-state machinery (see the docstring: divergence insurance)
    # + divergence-bail warmup (see the docstring: guard_warmup)
    warm_g = (max(30, steps // 10) if guard_warmup is None
              else max(int(guard_warmup), 0))

    def _eval8():
        """8 fresh batches, same jitter convention as the fit: a stable mse
        estimate. Used for the pre-fit baseline AND for the post-restore
        re-eval - comparing like with like (single-batch numbers jitter by
        ~10% and made the end-guard raise on healthy fits)."""
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
                    # hole-3: judge with the SAME clean-anchor routing the
                    # fit trains with
                    zb = (mod._z(Xf[ix].to(device)) if train_router
                          else z_all[ix].to(device, non_blocking=True))
                else:
                    zb = (mod._z(xb) if train_router
                          else z_all[ix].to(device, non_blocking=True))
                tot += float(F.mse_loss(mod.forward_from_z(xb, zb), yb).item())
        return tot / n_ev

    if guard:
        first = _eval8()        # stable init-state baseline (params pristine)
        with torch.no_grad():
            # pure-centroid reference for the report: with the SVD init the
            # init state is already far below the centroid, so "first" is no
            # longer the centroid baseline it used to be
            cgu0, cdn0 = mod.Cgu.detach().clone(), mod.Cdn.detach().clone()
            mod.Cgu.zero_()
            mod.Cdn.zero_()
            centroid_first = _eval8()
            mod.Cgu.copy_(cgu0)
            mod.Cdn.copy_(cdn0)

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
    for s in range(steps):
        if lr_warmup and s < lr_warmup:
            for gpt in opt.param_groups:
                gpt["lr"] = lr * (s + 1) / lr_warmup
        elif method == "muon-cosine":
            eff = max(steps - max(lr_warmup, 0), 1)
            cosf = 0.5 * (1.0 + math.cos(
                math.pi * (s - max(lr_warmup, 0)) / eff))
            for gpt in opt.param_groups:
                gpt["lr"] = lr * cosf
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
            # HOLE-3 FIX: routing follows the CLEAN anchor row (the target
            # was produced by the base model on the clean input). The old
            # "routing follows the noisy input" paired targets with the
            # wrong experts' coordinates whenever top-k flipped under noise.
            # Frozen router: z is the precomputed clean z_all[ix] (also
            # cheaper: no _z recompute per step). train_router: recompute on
            # the clean rows so grads reach gw.
            zb = (mod._z(Xf[ix].to(device)) if train_router
                  else z_all[ix].to(device, non_blocking=True))
        else:
            if train_router:
                zb = mod._z(xb)      # grads flow into the router via z
            else:
                zb = z_all[ix].to(device, non_blocking=True)
        with ac:
            out = mod.forward_from_z(xb, zb)
        loss = F.mse_loss(out.float(), yb)
        if train_router and router_anchor > 0:
            loss = loss + router_anchor * (
                ((mod.gw - gw0) ** 2).sum()
                / (gw0 ** 2).sum().clamp_min(1e-12))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None and (not lr_warmup or s >= lr_warmup):
            sched.step()
        last = float(loss.item())
        if first is None:
            first = last
        if s % log_every == 0 or s == steps - 1:
            print(f"    {log_prefix} step {s}: mse {last:.5f}", flush=True)
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
        elif best_score > 0 and ema > 2.0 * best_score and s >= warm_g:
            print(f"    {log_prefix} loss diverged at step {s} (mse "
                  f"{last:.5f}, ema {ema:.5f} vs best {best_score:.5f}) - "
                  f"best state restored, stopping this block early",
                  flush=True)
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
        last = _eval8()
        print(f"    {log_prefix} best-state re-eval: mse {last:.5f}",
              flush=True)
        best_state = None
    if guard and first is not None:
        worse = (not math.isfinite(last)) or last > first
        flat = (not worse) and last > 0.98 * first
        if worse or (flat and strict_guard):
            why = ("mse got WORSE than the centroid baseline" if worse else
                   "mse did not drop (strict mode)")
            raise RuntimeError(
                f"FIT GUARD: {why} ({first:.5f} -> {last:.5f}) - the field is "
                f"not learning (a typical cause is a degenerate gradient). "
                f"Continuing is pointless; to disable the check: "
                f"--skip-fit-guard")
        if flat:
            print(f"    {log_prefix} FIT GUARD: mse barely moved "
                  f"({first:.5f} -> {last:.5f}) - shipping the best state "
                  f"(no gain, no harm; --strict-fit-guard makes this an "
                  f"error)", flush=True)
    if guard and all(float(getattr(mod, n).abs().max()) == 0.0
                     for n in ("Cgu", "Cdn")):
        # the exact B1 bug signature: centroids learn (mse drops) but the
        # rank path stays frozen at zeros -> artifact = "one averaged expert"
        raise RuntimeError(
            "FIT GUARD: coordinates Cgu/Cdn stayed exactly zero - the rank "
            "path is not learning (the classic zero-init U,V,C bug). The fit "
            "degraded into a centroid; to disable the check: --skip-fit-guard")
    if guard and first:
        print(f"    {log_prefix} guard: mse {first:.5f} -> {last:.5f} "
              f"({100 * (first - last) / max(first, 1e-12):.1f}% below the "
              f"init baseline; pure-centroid {centroid_first:.5f})", flush=True)
    return last


def polish_router_module(mod, X, Y, steps, bs, lr, device, anchor=0.03,
                         log_prefix="", log_every=20, seed=None):
    """"After" router polish: the field is FROZEN, only the manual router
    (gw) trains on the same pairs, anchored to the original router
    (anchor * ||gw - gw0||^2 / ||gw0||^2). z is recomputed every step so the
    gradients reach gw. In-place: no extra artifact memory, the tuned gw
    simply replaces the router weight when the artifact is written
    (fit files carry it as gw_tuned).

    Best-EMA tracking keeps the best router seen (falling back to the
    ORIGINAL gw0 if the polish only hurt). Returns stats dict.

    Expectation setting (toy bench, 2026-09-04): after a CONVERGED field fit
    the router is usually not the bottleneck - the z-dependent rank
    correction carries a small share of the output energy, so gw gradients
    are tiny and even a 30% top-k set change moves the block mse by ~0.
    Keep expectations low; use it as a cheap diagnostic (--fit-router after)
    or for the last percent of quality."""
    if not hasattr(mod, "gw"):
        raise RuntimeError("router polish needs a FIT-mode field module "
                           "(manual router gw); deploy modules keep the "
                           "original gate object instead")
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(int(seed))
    else:
        torch.manual_seed(5)
    for t in mod.fit_params():
        t.requires_grad_(False)
    gw0 = mod.gw.detach().clone()
    mod.gw.requires_grad_(True)
    opt = torch.optim.Adam([mod.gw], lr=lr)
    n_tok = X.shape[0]
    n_ev = min(n_tok, 8192)
    with torch.no_grad():
        Xf = X.float()
        tgt = Y.float()
        if hasattr(mod, "sh_gu"):               # hy_v3: fold the shared branch
            sg, su = (Xf @ mod.sh_gu.t()).chunk(2, dim=-1)
            tgt = tgt - (mod.act_fn(sg) * su) @ mod.sh_dn.t()
        mse0 = float(F.mse_loss(mod.forward_from_z(
            Xf[:n_ev], mod._z(Xf[:n_ev])).cpu(), tgt[:n_ev]))
    best_ema, best_w = None, gw0.clone()
    last = mse0
    for s in range(steps):
        ix = (torch.randint(0, n_tok, (min(bs, n_tok),), generator=gen)
              if gen is not None
              else torch.randint(0, n_tok, (min(bs, n_tok),)))
        xb = Xf[ix].to(device, non_blocking=True)
        yb = tgt[ix].to(device, non_blocking=True)
        out = mod.forward_from_z(xb, mod._z(xb))
        loss = F.mse_loss(out, yb)
        if anchor > 0:
            loss = loss + anchor * (((mod.gw - gw0) ** 2).sum()
                                    / (gw0 ** 2).sum().clamp_min(1e-12))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        ema = last if best_ema is None else 0.9 * best_ema + 0.1 * last
        if best_ema is None or ema < best_ema:
            best_ema, best_w = ema, mod.gw.detach().clone()
        if s % log_every == 0 or s == steps - 1:
            print(f"    {log_prefix} step {s}: mse {last:.5f}", flush=True)
    with torch.no_grad():
        mod.gw.copy_(best_w)
        mod.gw.requires_grad_(False)
        mse1 = float(F.mse_loss(mod.forward_from_z(
            Xf[:n_ev], mod._z(Xf[:n_ev])).cpu(), tgt[:n_ev]))
    drift = float((mod.gw.detach() - gw0).norm()
                  / gw0.norm().clamp_min(1e-12))
    print(f"    {log_prefix} polish: fit-pool mse {mse0:.5f} -> {mse1:.5f}, "
          f"router drift {drift:.4f}", flush=True)
    return dict(mse_before=mse0, mse_after=mse1, drift=drift)


def build_deploy_block(orig_gate, geom, rank, act_fn, fit_init, dtype):
    """Deploy block: the original base router + trained field parameters.
    fit_init - dict of field tensors (keys = field_names); a fit module itself
    is also accepted (compat with the old call)."""
    if hasattr(fit_init, "field_names"):
        fit_init = {n: getattr(fit_init, n).detach() for n in fit_init.field_names}
    return FieldSparseMoe(geom, rank, gate=orig_gate, act_fn=act_fn,
                          dtype=dtype, init=fit_init)


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
    """Merge a block's buffers and save them to disk (bf16). Atomic
    (tmp + os.replace): a kill mid-save cannot leave a torn pairs_blk file
    that would poison the whole run cache on the next start."""
    X = torch.cat(X_list) if X_list else torch.empty(0)
    Y = torch.cat(Y_list) if Y_list else torch.empty(0)
    tmp = path + ".tmp"
    torch.save({"X": X, "Y": Y}, tmp)
    os.replace(tmp, path)
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


# ------------------------------------------------- SVD init (hole 1, UPDATE-10)

def _topk_eigh(G, r, oversample=16, seed=917):
    """Top-r eigenvectors (descending eigenvalue) of a symmetric PSD matrix
    via a randomized range finder (no full eigh: for (d,d) Grams a full
    decomposition costs seconds-to-minutes on CPU, the range finder is
    ~2 skinny matmuls)."""
    n = G.shape[0]
    q = min(n, r + oversample)
    g = torch.Generator().manual_seed(seed)
    Y = G @ (torch.randn(n, q, generator=g) / math.sqrt(q))
    Q, _ = torch.linalg.qr(Y)
    S = Q.t() @ (G @ Q)
    S = 0.5 * (S + S.t())
    vals, vecs = torch.linalg.eigh(S)
    order = torch.argsort(vals, descending=True)[:r]
    return Q @ vecs[:, order], vals


def _iter_expert_w(block):
    """Yield (w_gu (2dff,d) fp32 cpu, w_dn (d,dff) fp32 cpu) expert by expert
    without materializing the full fp32 stack (bnb is dequantized in its
    native fp16 once per pass, slices go to fp32 one expert at a time)."""
    exp = block.experts
    gu = getattr(exp, "gate_up_proj", None)
    dn = getattr(exp, "down_proj", None)
    if gu is not None and not isinstance(gu, nn.ModuleList):
        full_gu = dequant_weight(gu)
        full_dn = dequant_weight(dn)
        for e in range(full_gu.shape[0]):
            yield full_gu[e].float().cpu(), full_dn[e].float().cpu()
        return
    if isinstance(exp, nn.ModuleList):                     # v4 layout w1/w3/w2
        def pick(e, names):
            for nm in names:
                if hasattr(e, nm):
                    v = getattr(e, nm)
                    return v.weight if hasattr(v, "weight") else v
            raise AttributeError(f"none of {names} found in {type(e).__name__}")
        for e in exp:
            g = dequant_weight(pick(e, ("w1", "gate_proj")))
            u = dequant_weight(pick(e, ("w3", "up_proj")))
            d2 = dequant_weight(pick(e, ("w2", "down_proj")))
            yield torch.cat([g, u], dim=0).float().cpu(), d2.float().cpu()
        return
    raise RuntimeError(f"cannot extract experts from {type(exp).__name__}")


@torch.no_grad()
def expert_basis_init(block, mgu, mdn, rank, oversample=16, seed=917,
                      log_prefix=""):
    """Shared-basis SVD init for the field parameters from the REAL expert
    deltas (hole 1, UPDATE-10). For each side (gu, dn):
      pass 1: Y = sum_e dW_e @ Omega        (randomized range finder)
              Q = orth(Y)                    - output-side subspace
      pass 2: B_e = Q^T dW_e (stashed)      - projected deltas
      U = Q @ topk_eigh(sum B_e B_e^T)       - output-side basis
      V = topk_eigh(sum B_e^T B_e)           - input-side basis
      C_e = diag(U_B^T B_e V)                - exact coordinates of the
                                               projected deltas, NO 3rd pass
    NOTE: the SVD of the MEAN delta is degenerate (sum_e dW_e = 0
    identically); the stacked-delta shared basis is the meaningful variant.
    Streaming: the full expert stack is never materialized (one expert at a
    time), RAM ~= one expert + the B stash. Returns a dict fit for
    FieldSparseMoe(init=...) plus a "capture" diagnostic (share of the
    per-expert delta energy the rank-r basis reconstructs)."""
    g = torch.Generator().manual_seed(seed)
    mgu, mdn = mgu.float(), mdn.float()
    out_dim_gu, in_dim_gu = mgu.shape
    out_dim_dn, in_dim_dn = mdn.shape
    # pass 1 - output-side ranges for both sides in ONE streaming pass
    q_gu = min(out_dim_gu, rank + oversample)
    q_dn = min(out_dim_dn, rank + oversample)
    Om_gu = torch.randn(in_dim_gu, q_gu, generator=g) / math.sqrt(q_gu)
    Om_dn = torch.randn(in_dim_dn, q_dn, generator=g) / math.sqrt(q_dn)
    Ygu = torch.zeros(out_dim_gu, q_gu)
    Ydn = torch.zeros(out_dim_dn, q_dn)
    den = {"gu": 0.0, "dn": 0.0}       # sum_e ||dW_e||^2 (true capture denum)
    n_exp = 0
    for wgu, wdn in _iter_expert_w(block):
        dgu, ddn = wgu - mgu, wdn - mdn
        Ygu += dgu @ Om_gu
        Ydn += ddn @ Om_dn
        den["gu"] += float(dgu.norm() ** 2)
        den["dn"] += float(ddn.norm() ** 2)
        n_exp += 1
    Qgu, _ = torch.linalg.qr(Ygu)
    Qdn, _ = torch.linalg.qr(Ydn)
    # pass 2 - projected deltas (stash B, the small core)
    Bgu = torch.empty(n_exp, q_gu, in_dim_gu)
    Bdn = torch.empty(n_exp, q_dn, in_dim_dn)
    for e, (wgu, wdn) in enumerate(_iter_expert_w(block)):
        Bgu[e] = Qgu.t() @ (wgu - mgu)
        Bdn[e] = Qdn.t() @ (wdn - mdn)
    init = {}
    capture = {}
    for side, B, Q, od, idi in (("gu", Bgu, Qgu, out_dim_gu, in_dim_gu),
                                ("dn", Bdn, Qdn, out_dim_dn, in_dim_dn)):
        if rank > min(od, idi):
            # degenerate geometry (e.g. a toy model with dff < rank): the SVD
            # basis cannot supply `rank` orthonormal columns - keep the
            # module's random init for this side instead of a rank-mismatched
            # tensor (which would crash the init copy in FieldSparseMoe)
            print(f"    {log_prefix} svd init: {side} side skipped "
                  f"(rank {rank} > min(out,in) {min(od, idi)})", flush=True)
            continue
        G_out_q = sum(B[e] @ B[e].t() for e in range(n_exp))     # (q,q)
        G_in_p = sum(B[e].t() @ B[e] for e in range(n_exp))      # (in,in)
        U_B, _ = _topk_eigh(G_out_q, rank, oversample, seed + 1)
        U = Q @ U_B
        V, _ = _topk_eigh(G_in_p, rank, oversample, seed + 2)
        # coordinates from the stash: U^T dW_e V = U_B^T B_e V (exact within
        # range(Q), which the oversampled range finder makes negligible)
        C = torch.stack([torch.diagonal(U_B.t() @ B[e] @ V)
                         for e in range(n_exp)])
        init[f"U{side}"], init[f"V{side}"], init[f"C{side}"] = U, V, C
        # TRUE diag capture: energy of the actual diag-coordinate
        # reconstruction (num = sum_e ||C_e||^2) over the full delta energy
        # (den from pass 1; Q orthonormal preserves norms). NOTE: the U/V
        # subspace PAIR carries ~all of the delta energy, but the field uses
        # DIAGONAL per-expert coordinates, so step-0 capture is lower; U and
        # V are trainable - the fit recovers the rest by rotating the bases
        # (toy: diag ~0.3-0.7 at step 0, still 2.8x better heldout after the
        # same step budget than the random init).
        K = torch.stack([U_B.t() @ B[e] @ V for e in range(n_exp)])
        num = float((C ** 2).sum())
        capture[side] = num / max(den[side], 1e-12)
        init[f"capture_proj_{side}"] = float((K ** 2).sum()) \
            / max(den[side], 1e-12)
    init["capture"] = capture
    parts = [f"{s} {capture[s] * 100:.1f}%" for s in ("gu", "dn") if s in capture]
    print(f"    {log_prefix} svd init: delta energy captured at step 0 - "
          + ", ".join(parts), flush=True)
    return init


@torch.no_grad()
def eval_logits_cache_disk(model, ids, ctx, n_chunks, lp_dir, seed=17):
    """Base-model log-probs on fixed chunks -> lp_XXX.pt files (per chunk).
    RESUMABLE (2026-09-04.3): chunk starts come from a fixed-seed RNG, so a
    chunk is identical across runs - existing loadable lp_XXX.pt files are
    reused and only the missing/corrupt ones are computed. An interrupted
    base pass therefore continues where it stopped instead of starting over.
    A fingerprint (sha1 of the eval token ids + ctx/chunks/seed) guards the
    reuse: a changed dataset invalidates the whole lp cache. RAM: one chunk
    at a time. Returns X, Y - the tokens; the log-probs live on disk."""
    os.makedirs(lp_dir, exist_ok=True)
    fp = hashlib.sha1()
    fp.update(ids.numpy().tobytes())
    sig = {"sha1": fp.hexdigest(), "ctx": int(ctx),
           "n_chunks": int(n_chunks), "seed": int(seed)}
    sig_path = os.path.join(lp_dir, "cache_sig.json")
    reuse = False
    if os.path.isfile(sig_path):
        try:
            with open(sig_path, encoding="utf-8") as f:
                reuse = json.load(f) == sig
        except Exception:  # noqa: BLE001
            reuse = False
    if not reuse:
        for i in range(n_chunks):   # drop stale/foreign chunks
            p = os.path.join(lp_dir, f"lp_{i:03d}.pt")
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    model.eval()
    dev = _mdev(model)
    g = torch.Generator().manual_seed(seed)
    X, Y = [], []
    for i in range(n_chunks):
        s = int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g))
        xc = ids[s:s + ctx]          # stays on CPU: it is cached in eval_tokens.pt
        X.append(xc)
        Y.append(ids[s + 1:s + ctx + 1])
        lp_path = os.path.join(lp_dir, f"lp_{i:03d}.pt")
        if reuse and os.path.isfile(lp_path):
            try:
                torch.load(lp_path, map_location="cpu")   # integrity check
                continue              # cached in a previous run - skip the pass
            except Exception:         # noqa: BLE001 - torn file: recompute
                pass
        logits = model(input_ids=xc.unsqueeze(0).to(dev)).logits[0]
        lp = torch.log_softmax(logits, dim=-1).to(torch.bfloat16).cpu()
        torch.save(lp, lp_path)
    tmp = sig_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sig, f)
    os.replace(tmp, sig_path)
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

def render_modeling_file(base_cls, router_cls, router_mod):
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        tpl = f.read()
    return (tpl.replace("@@BASE@@", base_cls)
               .replace("@@ROUTER@@", router_cls)
               .replace("@@ROUTER_MOD@@", router_mod))


def find_field_modules(model):
    """Field modules after the swap (gate + Cgu coordinates)."""
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, "gate") and hasattr(m, "Cgu")]


def field_geometry(mod, config):
    """Geometry from a field module (for cfg.field on save)."""
    return dict(n_exp=int(mod.Cgu.shape[0]), d_model=int(mod.d),
                d_ff=int(mod.wgud.shape[0] // 2), top_k=int(mod.k),
                norm_topk=bool(mod.norm),
                hidden_act=str(getattr(config, "hidden_act", "silu")))


def save_field_model(model, tokenizer, out_dir, rank, accounting, meta):
    """Save a NORMAL HF model with the field: config + weights +
    modeling_field.py."""
    os.makedirs(out_dir, exist_ok=True)
    mods = find_field_modules(model)
    if not mods:
        raise RuntimeError("no field modules found in the model - was the swap done?")
    first = mods[0][1]
    base_cls = type(model).__name__
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

    src = render_modeling_file(base_cls, router_cls, router_mod)
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
                         io_workers=1, io_cache="disk"):
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
    # gate weights are DEFERRED: they are tiny (n_exp x d) and may be replaced
    # by the TUNED router (fit files carry gw_tuned when --fit-router ran)
    shard, shards_keys, written = {}, [], 0
    total = 0
    gate_stash, gate_dtype = {}, {}
    n_routed = 0

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
                if ".experts." in key:       # the field replaces the experts
                    continue
                t = f.get_tensor(key)
                if key.endswith(".gate.weight"):
                    gate_stash[key] = t      # written after the field loop
                    gate_dtype[key] = t.dtype
                    continue
                shard[key] = t
                nb = t.numel() * t.element_size()
                total += nb
                written += nb
                if written >= max_shard_bytes:
                    flush()
    if gs is not None:
        for key in gs.keys():
            if ".experts." in key:           # the field replaces the experts
                continue
            t = gs.get(key)
            if key.endswith(".gate.weight"):
                gate_stash[key] = t          # written after the field loop
                gate_dtype[key] = t.dtype
                continue
            shard[key] = t
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
        gk = f"{layer}.gate.weight"
        if fit.get("gw_tuned") is not None and gk in gate_dtype:
            # the tuned router REPLACES the backbone copy (same shape)
            shard[gk] = fit["gw_tuned"].to(gate_dtype[gk])
            gate_stash.pop(gk, None)
            n_routed += 1
            nb = shard[gk].numel() * shard[gk].element_size()
            total += nb
            written += nb
            if written >= max_shard_bytes:
                flush()
        for nm, t in fit.items():
            if nm == "gw_tuned":             # handled above (gate key)
                continue
            key = f"{layer}.{nm}"
            t = t.to(dtype)
            shard[key] = t
            nb = t.numel() * t.element_size()
            total += nb
            written += nb
            if written >= max_shard_bytes:
                flush()
    for gk, t in list(gate_stash.items()):    # gates of untuned blocks
        shard[gk] = t
        gate_stash.pop(gk, None)
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
                                     am["router_mod"]))
    import py_compile
    py_compile.compile(os.path.join(out_dir, "modeling_field.py"), doraise=True)
    shutil.rmtree(os.path.join(out_dir, "__pycache__"), ignore_errors=True)
    meta_out = dict(rank=int(rank), n_layers=int(am["n_layers"]),
                    backbone="copy of the source backbone (experts skipped)",
                    field_dtype=str(dtype))
    if n_routed:
        meta_out["router_polish"] = dict(
            n_layers_tuned=int(n_routed),
            note="gate weights were tuned on the calibration pairs "
                 "(--fit-router); see fit_dir/router_meta.json")
    if profile:
        meta_out["profile"] = profile
    with open(os.path.join(out_dir, "field_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    return out_dir
