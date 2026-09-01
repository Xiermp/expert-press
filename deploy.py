"""Field-engine runtime (deploy): the working model is assembled from a
compact artifact WITHOUT explicit expert weights.

Artifact format (torch.save):
  format   - "field-engine-v1"
  cfg, itos, r
  backbone - the shared part of the model (embeddings, attention, norms,
             routers), fp32; in a real scenario this is the pre-existing base
             model (not counted as compression)
  layers   - per layer the "field": w1d/w2d (centroids), U1/V1/U2/V2
             (factors), C1/C2 (coordinates) - fp16
  meta     - arbitrary metadata (base_ppl, fit_steps, ...)

An expert is assembled on the fly: W(z) = w1d + U-diag(c(z))-V^T, where the
"movement seed" c(z) = z @ C is computed from the router's soft weights.
Storage fp16, compute fp32.

Runtime dependencies: torch + common.py (TinyMoE). Experimental code not needed.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import TinyMoE

FIELD_KEYS = ("w1d", "w2d", "U1", "V1", "U2", "V2", "C1", "C2")
FORMAT = "field-engine-v1"


def soft_topk(probs, k):
    topw, topi = torch.topk(probs, k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, topi, topw)      # (B,T,N)


class FieldMoE(nn.Module):
    """Engine: centroid + U-diag(c(z))-V^T; c(z) = z @ C - coordinates in the
    field."""

    def __init__(self, router, w1d, w2d, U1, V1, U2, V2, C1, C2, top_k):
        super().__init__()
        self.router = router
        self.w1d, self.w2d = w1d, w2d                          # (d_ff,d), (d,d_ff)
        self.U1, self.V1, self.U2, self.V2 = U1, V1, U2, V2    # (d_ff,r),(d,r),(d,r),(d_ff,r)
        self.C1, self.C2 = C1, C2                              # (N,r)
        self.k = top_k

    def delta(self, acts, V, U, c):
        return (acts @ V * c) @ U.t()

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)
        z = soft_topk(probs, self.k)                           # (B,T,N)
        c1 = z @ self.C1                                       # (B,T,r) - movement seed
        c2 = z @ self.C2
        h = F.gelu(x @ self.w1d.t() + self.delta(x, self.V1, self.U1, c1))
        y = h @ self.w2d.t() + self.delta(h, self.V2, self.U2, c2)
        return y, probs


def field_bytes_claimed(cfg, r):
    """Claimed field size by the formula from field_eval (fp16 accounting, the
    MoE part only)."""
    d, dff, N, L = cfg["d_model"], cfg["d_ff"], cfg["n_exp"], cfg["n_layer"]
    per_matrix = dff * d + r * (dff + d) + N * r               # dense + U,V + coordinates
    return L * 2 * per_matrix * 2                              # fp16, w1+w2 per layer


def artifact_field_bytes(art):
    """Actual size of the artifact's MoE part (sum numel x element_size over
    the field)."""
    total = 0
    for layer in art["layers"]:
        for k in FIELD_KEYS:
            t = layer[k]
            total += t.numel() * t.element_size()
    return total


def save_deployed(path, cfg, itos, base_sd, field_modules, r, meta=None):
    """Assemble the artifact: the backbone as-is + the field in fp16 (that IS
    the "compressed experts")."""
    layers = []
    for m in field_modules:
        layers.append({k: getattr(m, k).detach().to(torch.float16).contiguous()
                       for k in FIELD_KEYS})
    backbone = {k: v.detach().contiguous() for k, v in base_sd.items()
                if ".moe.w1" not in k and ".moe.w2" not in k}   # explicit experts not saved
    art = dict(format=FORMAT, cfg=dict(cfg), itos=list(itos), r=int(r),
               backbone=backbone, layers=layers,
               meta=dict(meta or {}), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(art, path)
    return path


def load_deployed(path):
    """Load the deploy model from the artifact only. Returns (model, art)."""
    art = torch.load(path, map_location="cpu", weights_only=False)
    if art.get("format") != FORMAT:
        raise ValueError(f"unknown artifact format: {art.get('format')!r}")
    cfg, itos = art["cfg"], art["itos"]
    model = TinyMoE(cfg, len(itos))
    missing, unexpected = model.load_state_dict(art["backbone"], strict=False)
    bad_missing = [k for k in missing if ".moe.w1" not in k and ".moe.w2" not in k]
    if bad_missing or unexpected:
        raise RuntimeError(f"artifact does not match the model: missing={bad_missing} "
                           f"unexpected={unexpected}")
    for li, b in enumerate(model.blocks):                      # replace MoE with the field
        L = art["layers"][li]
        d, N = cfg["d_model"], cfg["n_exp"]
        router = nn.Linear(d, N, bias=False)
        with torch.no_grad():
            router.weight.copy_(art["backbone"][f"blocks.{li}.moe.router.weight"])
        params = {k: nn.Parameter(L[k].float()) for k in FIELD_KEYS}   # dequant fp16->fp32
        for p in params.values():
            p.requires_grad_(False)
        router.weight.requires_grad_(False)
        b.moe = FieldMoE(router, top_k=cfg["top_k"], **params)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, art


def list_artifacts(directory):
    import glob
    return sorted(glob.glob(os.path.join(directory, "moe_transformed_field_r*.pt")))
