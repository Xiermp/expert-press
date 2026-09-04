"""Task 28 (build 10.5, 2026-09-05.3): проверка шаблона артефакта на hy_v3-пути.

Сценарий юзера: Stage 7 упал с "HYV3TopKRouter.forward() missing 1 required
positional argument: 'e_score_correction_bias'", в load-отчёте висели
UNEXPECTED shared_experts.* и e_score_correction_bias.

Проверяем:
  1. hy_v3: FieldSparseMoe строится с реальным HYV3TopKRouter; буфер
     e_score_correction_bias и shared_experts на месте; форвард конечен;
     bias реально влияет на выбор экспертов.
  2. Shared-ветка шаблона == HYV3MLP базовой модели (те же веса -> тот же выход).
  3. Математика шаблона == фит-сторона (hf_field_transform.FieldSparseMoe)
     при одинаковых параметрах поля/бias/shared.
  4. Мини-e2e: FieldForCausalLM(HYV3ForCausalLM, 2 слоя) -> save_pretrained
     + modeling_field.py -> from_pretrained(trust_remote_code) ->
     loading_info: missing/unexpected ПУСТЫЕ; логиты совпадают с исходником.
  5. OLMoE-регресс: softmax-путь без bias/shared; tuple-роутер; Linear-гейт
     (не-tuple) больше не падает с NameError (старый баг шаблона).
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hf_field_transform import render_modeling_file, FieldSparseMoe as FitMoe

torch.manual_seed(0)

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" ({extra})" if extra else ""))
    if not cond:
        FAILED.append(name)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_to(tmp, base_cls, router_cls, router_mod, tag):
    src = render_modeling_file(base_cls, router_cls, router_mod)
    p = os.path.join(tmp, f"modeling_field_{tag}.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    import py_compile
    py_compile.compile(p, doraise=True)
    return load_module(p, f"mf_{tag}"), p


# ---------------------------------------------------------------- 1-3: hy_v3
print("== 1-3. hy_v3: real HYV3TopKRouter + parity ==")
from transformers.models.hy_v3.modeling_hy_v3 import (
    HYV3TopKRouter, HYV3MLP)

D, DFF, DFFS, NEXP, R, TOPK = 32, 48, 16, 8, 8, 2
fi = dict(rank=R, n_layers=2, base_class="HYV3ForCausalLM", n_exp=NEXP,
          d_model=D, d_ff=DFF, top_k=TOPK, norm_topk=False,
          hidden_act="silu", router_kind="sigmoid_bias",
          router_scale=2.826, fp32_combine=True, dff_shexp=DFFS)

tmp = tempfile.mkdtemp(prefix="mf_hyv3_")
art_mod, art_path = render_to(tmp, "HYV3ForCausalLM", "HYV3TopKRouter",
                              "transformers.models.hy_v3.modeling_hy_v3",
                              "hyv3")

routed_cfg = types.SimpleNamespace(
    field=fi, hidden_act="silu", num_experts_per_tok=TOPK,
    num_local_experts=NEXP, hidden_size=D, router_scaling_factor=2.826,
    mlp_bias=False)
router = HYV3TopKRouter(routed_cfg)
with torch.no_grad():
    router.weight.normal_(std=0.1)   # torch.empty м.б. мусором (NaN) - в
    # реальном артефакте вес всегда приходит из чекпойнта, тут фиксируем
am = art_mod.FieldSparseMoe(routed_cfg)
with torch.no_grad():
    am.gate.weight.copy_(router.weight)
check("e_score_correction_bias buffer registered",
      hasattr(am, "e_score_correction_bias")
      and am.e_score_correction_bias.shape == (NEXP,))
check("shared_experts submodule registered",
      hasattr(am, "shared_experts")
      and am.shared_experts.gate_proj.weight.shape == (DFFS, D)
      and am.shared_experts.up_proj.weight.shape == (DFFS, D)
      and am.shared_experts.down_proj.weight.shape == (D, DFFS))
with torch.no_grad():
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn",
              "e_score_correction_bias"):
        getattr(am, n).normal_(std=0.1)
x = torch.randn(2, 5, D)
y = am(x)
check("forward runs, shape/finite",
      y.shape == (2, 5, D) and torch.isfinite(y).all())

with torch.no_grad():
    b2 = am.e_score_correction_bias.clone()
    b2[0] += 5.0
    out1 = am.gate(x.reshape(-1, D), am.e_score_correction_bias)
    out2 = am.gate(x.reshape(-1, D), b2)
check("bias actually drives selection",
      not torch.equal(out1[2], out2[2]))

# shared parity vs base HYV3MLP
base_mlp = HYV3MLP(routed_cfg, intermediate_size=DFFS)
with torch.no_grad():
    base_mlp.gate_proj.weight.copy_(am.shared_experts.gate_proj.weight)
    base_mlp.up_proj.weight.copy_(am.shared_experts.up_proj.weight)
    base_mlp.down_proj.weight.copy_(am.shared_experts.down_proj.weight)
ys_ref = base_mlp(x.reshape(-1, D))
se = am.shared_experts
ys_art = se.down_proj(F.silu(se.gate_proj(x.reshape(-1, D)))
                      * se.up_proj(x.reshape(-1, D)))
check("shared branch == base HYV3MLP",
      torch.allclose(ys_ref, ys_art, atol=1e-6))

# math parity vs fit-side module
geom = dict(n_exp=NEXP, d_model=D, d_ff=DFF, top_k=TOPK, norm_topk=False,
            hidden_act="silu", router_kind="sigmoid_bias",
            router_scale=2.826, fp32_combine=True, dff_shexp=DFFS)
sh_gu = torch.cat([am.shared_experts.gate_proj.weight,
                   am.shared_experts.up_proj.weight], dim=0)
fm = FitMoe(geom, R, gate=router, act_fn=F.silu, dtype=torch.float32,
            gate_bias=am.e_score_correction_bias.clone(),
            shared=dict(sh_gu=sh_gu, sh_dn=am.shared_experts.down_proj.weight.clone()))
with torch.no_grad():
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn"):
        getattr(fm, n).copy_(getattr(am, n).float())
y_ref = fm(x)
check("artifact math == fit-side module",
      torch.allclose(y_ref, y.float(), atol=1e-5),
      f"max|d|={ (y_ref - y).abs().max().item():.2e}")

# state-dict keys of the artifact block
keys = set(am.state_dict().keys())
need = {"gate.weight", "e_score_correction_bias", "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight", "shared_experts.down_proj.weight",
        "wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn"}
check("state_dict covers exactly the artifact keys", need <= keys,
      str(sorted(need - keys)) if not need <= keys else "")

# ---------------------------------------------------------------- 4: mini-e2e
print("== 4. mini-e2e: save -> from_pretrained(trust_remote_code) ==")
from transformers import HYV3ForCausalLM
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config

cfg = HYV3Config(vocab_size=64, hidden_size=D, intermediate_size=64,
                 num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, hidden_act="silu",
                 moe_intermediate_size=DFFS, num_experts=NEXP,
                 num_experts_per_tok=TOPK, num_shared_experts=1,
                 router_scaling_factor=2.826, enable_moe_fp32_combine=True)
if not hasattr(cfg, "num_local_experts"):
    cfg.num_local_experts = NEXP
base0 = HYV3ForCausalLM(cfg)
n_moe = sum(1 for _, m in base0.named_modules()
            if hasattr(m, "experts") and hasattr(m, "gate"))
del base0
check("mini hy_v3 has MoE blocks", n_moe >= 1, f"n_moe={n_moe}")

cfg.field = dict(fi, n_layers=n_moe)
cfg.architectures = ["FieldForCausalLM"]
cfg.auto_map = {"AutoModelForCausalLM": "modeling_field.FieldForCausalLM"}
full = art_mod.FieldForCausalLM(cfg)
with torch.no_grad():
    for p in full.parameters():          # всё случайно: torch.empty в
        p.normal_(std=0.05)              # роутере гарантированно конечен
    for name, buf in full.named_buffers():
        if name.endswith("e_score_correction_bias"):
            buf.normal_(std=0.05)

art_dir = os.path.join(tmp, "field_mini_hyv3")
full.save_pretrained(art_dir)
with open(os.path.join(art_dir, "modeling_field.py"), "w", encoding="utf-8") as f:
    f.write(open(art_path, encoding="utf-8").read())

from transformers import AutoModelForCausalLM
loaded, info = AutoModelForCausalLM.from_pretrained(
    art_dir, trust_remote_code=True, output_loading_info=True)
check("no MISSING keys on reload", not info.get("missing_keys"),
      str(info.get("missing_keys"))[:200])
check("no UNEXPECTED keys on reload", not info.get("unexpected_keys"),
      str(info.get("unexpected_keys"))[:200])
ids = torch.randint(0, 64, (1, 8))
with torch.no_grad():
    l1 = full(ids).logits
    l2 = loaded(ids).logits
check("logits identical after reload", torch.allclose(l1, l2, atol=1e-5))

# ---------------------------------------------------------------- 5: OLMoE
print("== 5. OLMoE regression: softmax path + Linear-gate NameError fix ==")
ol_mod, _ = render_to(tmp, "OlmoeForCausalLM", "OlmoeTopKRouter",
                      "transformers.models.olmoe.modeling_olmoe", "olmoe")
fi_ol = dict(rank=4, n_layers=1, base_class="OlmoeForCausalLM", n_exp=NEXP,
             d_model=16, d_ff=32, top_k=TOPK, norm_topk=False,
             hidden_act="silu")
ol_cfg = types.SimpleNamespace(field=fi_ol, hidden_act="silu",
                               num_experts_per_tok=TOPK,
                               num_local_experts=NEXP, num_experts=NEXP,
                               norm_topk_prob=False, hidden_size=16)
om = ol_mod.FieldSparseMoe(ol_cfg)
check("no bias buffer on softmax path", not hasattr(om, "e_score_correction_bias"))
check("no shared_experts on softmax path", not hasattr(om, "shared_experts"))
with torch.no_grad():
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn"):
        getattr(om, n).normal_(std=0.1)

class TupleRouter(torch.nn.Module):
    def forward(self, x):
        lg = x @ torch.randn(16, NEXP)
        pr = torch.softmax(lg.float(), -1)
        s, ix = torch.topk(pr, TOPK, -1)
        return lg, s, ix

om.gate = TupleRouter()
y1 = om(torch.randn(2, 3, 16))
check("tuple-router forward ok", y1.shape == (2, 3, 16) and torch.isfinite(y1).all())
om.gate = torch.nn.Linear(16, NEXP)            # не-tuple ветка
y2 = om(torch.randn(2, 3, 16))
check("Linear-gate forward ok (NameError fixed)",
      y2.shape == (2, 3, 16) and torch.isfinite(y2).all())
om.norm_topk = True
y3 = om(torch.randn(2, 3, 16))
check("norm_topk branch ok", torch.isfinite(y3).all())

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
