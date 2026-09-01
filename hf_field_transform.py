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

import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "modeling_field_template.py")

# available fit optimizer kinds (see fit_field_module / --fit-method)
FIT_METHODS = ("adam", "adamw", "adam-cosine", "rmsprop")


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
                     jitter=0.0, early_stop=0):
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

    method: "adam" | "adamw" | "adam-cosine" | "rmsprop"
      adam        - plain Adam, constant lr (classic baseline)
      adamw       - AdamW with a small weight decay (less drift in U,V)
      adam-cosine - Adam + cosine lr decay to ~0 (usually the best final mse)
      rmsprop     - RMSProp, an alternative for instable blocks
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
    wd = 0.01 if method == "adamw" else 0.0
    if method == "rmsprop":
        opt = torch.optim.RMSprop(p, lr=lr)
    elif method == "adamw":
        opt = torch.optim.AdamW(p, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(p, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1)) \
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
    last = first = None
    ckpt_ema = None
    stall = 0
    warm = min(150, max(10, steps // 4)) if early_stop else steps
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
            zb = mod._z(xb).detach()
        else:
            zb = z_all[ix].to(device, non_blocking=True)
        out = mod.forward_from_z(xb, zb)
        loss = F.mse_loss(out, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        last = float(loss.item())
        if first is None:
            first = last
        if s % log_every == 0 or s == steps - 1:
            print(f"    {log_prefix} step {s}: mse {last:.5f}", flush=True)
        if early_stop and s >= warm and (s % early_stop == 0 or s == steps - 1):
            # 2 consecutive flat checkpoints (<0.5% relative) -> stop early;
            # the checkpoint is an EMA of the minibatch mse (a raw single-batch
            # value is noisy: one lucky/unlucky batch can hide or fake a plateau)
            ema = last if ckpt_ema is None else 0.5 * ckpt_ema + 0.5 * last
            if ckpt_ema is not None and ema > ckpt_ema * (1.0 - 0.005):
                stall += 1
                if stall >= 2:
                    print(f"    {log_prefix} early stop at step {s} "
                          f"(mse plateaued)", flush=True)
                    break
            else:
                stall = 0
            ckpt_ema = ema
    for t in p:
        t.requires_grad_(False)
    if guard and first is not None and last > 0.98 * first:
        raise RuntimeError(
            f"FIT GUARD: mse did not drop ({first:.5f} -> {last:.5f}) - the "
            f"field is not learning (a typical cause is a degenerate gradient). "
            f"Continuing is pointless; to disable the check: --skip-fit-guard")
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
              f"centroid baseline)", flush=True)
    return last


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
    g = torch.Generator().manual_seed(seed)
    X, Y = [], []
    for i in range(n_chunks):
        s = int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g))
        x = ids[s:s + ctx].unsqueeze(0)
        logits = model(input_ids=x).logits[0]
        lp = torch.log_softmax(logits, dim=-1).to(torch.bfloat16).cpu()
        torch.save(lp, os.path.join(lp_dir, f"lp_{i:03d}.pt"))
        X.append(x[0])
        Y.append(ids[s + 1:s + ctx + 1])
    return X, Y


@torch.no_grad()
def eval_vs_cache_disk(model, X, Y, lp_dir, n_max=None):
    """CE/KL against the on-disk cache of base log-probs (per chunk, flat RAM).
    n_max - verify only the first n chunks (for the base the cache is its own:
    a full check of all chunks = wasted model passes, expensive in streaming)."""
    model.eval()
    ces, kls = [], []
    pairs = list(zip(X, Y))
    if n_max:
        pairs = pairs[:n_max]
    for i, (x, y) in enumerate(pairs):
        lp = torch.load(os.path.join(lp_dir, f"lp_{i:03d}.pt"), map_location="cpu")
        logits = model(input_ids=x.unsqueeze(0)).logits[0]
        # ppl computed the SAME way as the base ppl from the cache: gather from
        # bf16 log_softmax. An identical-to-base model then gives a delta ppl
        # of exactly 0.0 (no systematic float32-CE vs bf16-cache shift).
        lp_q = torch.log_softmax(logits, dim=-1).to(torch.bfloat16)
        ces.append(float(-(lp_q.gather(1, y.unsqueeze(1))[:, 0]).float().mean()))
        lq = torch.log_softmax(logits.float(), dim=-1)
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
    g = torch.Generator().manual_seed(seed)
    X, Y, LP = [], [], []
    for _ in range(n_chunks):
        s = int(torch.randint(0, len(ids) - ctx - 1, (1,), generator=g))
        x = ids[s:s + ctx].unsqueeze(0)
        logits = model(input_ids=x).logits[0]
        LP.append(torch.log_softmax(logits, dim=-1).to(torch.bfloat16).cpu())
        X.append(x[0])
        Y.append(ids[s + 1:s + ctx + 1])
    return X, Y, LP


@torch.no_grad()
def eval_vs_cache(model, X, Y, LP):
    """CE/KL of the converted model against the base log-prob cache."""
    model.eval()
    ces, kls = [], []
    for x, y, lp in zip(X, Y, LP):
        logits = model(input_ids=x.unsqueeze(0)).logits[0]
        lq = torch.log_softmax(logits.float(), dim=-1)
        ces.append(F.cross_entropy(logits.float(), y).item())
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
    prompt = ids[s:s + n_prompt].unsqueeze(0)
    kwargs = dict(max_new_tokens=n_new, do_sample=False)
    # compressed models have slightly shifted logits - plain greedy decoding
    # is the worst case for degenerate repetition loops; a standard
    # repetition penalty keeps the demo text readable (applied to BOTH the
    # base and the field generations, so the comparison stays fair)
    if repetition_penalty and repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    out = model.generate(prompt, **kwargs)
    return tokenizer.decode(out[0])


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
                if ".experts." in key:       # the field replaces the experts
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
            if ".experts." in key:           # the field replaces the experts
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
                                     am["router_mod"]))
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
