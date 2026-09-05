"""Task 29 (build 10.6, 2026-09-05.4): dtype-mismatch краш юзера в Stage 7.

Сценарий: `RuntimeError: expected m1 and m2 to have the same dtype, but got:
float != struct c10::BFloat16` на `cgu, cdn = z @ self.Cgu`.

Причины, проверяемые здесь:
  1. v5-роутер hy_v3 возвращает fp32-логиты/веса даже в bf16-модели (upcast
     внутри роутера) -> z fp32, поле bf16, matmul падает. Проверяем загрузку
     мини-модели через from_pretrained(dtype=bfloat16) - путь Stage 7.
  2. Смешанный чекпойнт (бэкбон fp32 + поле bf16) при загрузке "как сохранён"
     (dtype=None): поле должно выровняться вверх в fp32 без потери точности
     (сравнение с fp32-эталоном до каста).
  3. bf16-хост с fp32-гейтом в модуле - всё приводится вниз без краша.
"""
import importlib.util
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_field_transform import render_modeling_file

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


D, DFF, DFFS, NEXP, R, TOPK = 32, 48, 16, 8, 8, 2
fi = dict(rank=R, n_layers=2, base_class="HYV3ForCausalLM", n_exp=NEXP,
          d_model=D, d_ff=DFF, top_k=TOPK, norm_topk=False,
          hidden_act="silu", router_kind="sigmoid_bias",
          router_scale=2.826, fp32_combine=True, dff_shexp=DFFS)

tmp = tempfile.mkdtemp(prefix="mf_dtype_")
src = render_modeling_file("HYV3ForCausalLM", "HYV3TopKRouter",
                           "transformers.models.hy_v3.modeling_hy_v3")
art_path = os.path.join(tmp, "modeling_field.py")
with open(art_path, "w", encoding="utf-8") as f:
    f.write(src)
art_mod = load_module(art_path, "mf_dtype")

# ------------------------------------------------ 1: bf16 model, fp32 router
print("== 1. Stage-7 path: from_pretrained(dtype=bfloat16), fp32 router out ==")
from transformers import HYV3ForCausalLM, AutoModelForCausalLM
from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config

routed_cfg = types.SimpleNamespace(
    field=fi, hidden_act="silu", num_experts_per_tok=TOPK,
    num_local_experts=NEXP, hidden_size=D, router_scaling_factor=2.826,
    mlp_bias=False)
probe = HYV3TopKRouter(routed_cfg)
px = torch.randn(4, D)
plg, psc, pidx = probe(px, torch.zeros(NEXP))
check("host HYV3TopKRouter returns fp32 logits/weights",
      plg.dtype == torch.float32 and psc.dtype == torch.float32,
      f"logits {plg.dtype}, scores {psc.dtype}")

cfg = HYV3Config(vocab_size=64, hidden_size=D, intermediate_size=64,
                 num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, hidden_act="silu",
                 moe_intermediate_size=DFFS, num_experts=NEXP,
                 num_experts_per_tok=TOPK, num_shared_experts=1,
                 router_scaling_factor=2.826, enable_moe_fp32_combine=True)
if not hasattr(cfg, "num_local_experts"):
    cfg.num_local_experts = NEXP
n_moe = sum(1 for _, m in HYV3ForCausalLM(cfg).named_modules()
            if hasattr(m, "experts") and hasattr(m, "gate"))
cfg.field = dict(fi, n_layers=n_moe)
print(f"  (mini hy_v3: {n_moe} MoE block(s))")
cfg.architectures = ["FieldForCausalLM"]
cfg.auto_map = {"AutoModelForCausalLM": "modeling_field.FieldForCausalLM"}
full = art_mod.FieldForCausalLM(cfg)
with torch.no_grad():
    for p in full.parameters():
        p.normal_(std=0.05)
    for name, buf in full.named_buffers():
        if name.endswith("e_score_correction_bias"):
            buf.normal_(std=0.05)
ids = torch.randint(0, 64, (1, 8))
with torch.no_grad():
    ref = full(ids).logits.float()          # fp32-эталон до каста

art_dir = os.path.join(tmp, "field_mini")
full.save_pretrained(art_dir)
with open(os.path.join(art_dir, "modeling_field.py"), "w", encoding="utf-8") as f:
    f.write(src)
del full

bf16_model = AutoModelForCausalLM.from_pretrained(
    art_dir, trust_remote_code=True, dtype=torch.bfloat16).eval()
with torch.no_grad():
    out = bf16_model(ids).logits
check("bf16 forward survives fp32 router outputs",
      out.dtype == torch.bfloat16 and torch.isfinite(out).all())
check("bf16 logits near fp32 reference",
      torch.allclose(out.float(), ref, atol=0.35),
      f"max|d|={(out.float() - ref).abs().max().item():.3f}")

# ------------------------------------- 2: mixed checkpoint (backbone fp32)
print("== 2. mixed checkpoint: backbone fp32 + field bf16, load as stored ==")
routed_cfg2 = types.SimpleNamespace(
    field=fi, hidden_act="silu", num_experts_per_tok=TOPK,
    num_local_experts=NEXP, hidden_size=D, router_scaling_factor=2.826,
    mlp_bias=False)
am = art_mod.FieldSparseMoe(routed_cfg2)
with torch.no_grad():
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn",
              "e_score_correction_bias"):
        getattr(am, n).normal_(std=0.1)
x = torch.randn(2, 5, D)
with torch.no_grad():
    y_ref = am(x).clone()                   # fp32-эталон
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn"):
        p = getattr(am, n)
        p.data = p.data.to(torch.bfloat16)  # поле в bf16, как в артефакте
    am.e_score_correction_bias = am.e_score_correction_bias.to(torch.bfloat16)
    am._field_dtype = None                  # забыть выравнивание
with torch.no_grad():
    bf16_vals = {n: getattr(am, n).detach().clone().float() for n in
                 ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn")}
    y_mix = am(x)                           # крашиться здесь нечему
check("mixed-dtype forward runs", y_mix.dtype == torch.float32
      and torch.isfinite(y_mix).all())
check("align == pure cast up (params kept bit-exact)",
      all(torch.equal(getattr(am, n).detach().float(), bf16_vals[n])
          for n in bf16_vals))
check("output matches bf16-rounded reference (atol=bf16 eps)",
      torch.allclose(y_mix, y_ref, atol=5e-3),
      f"max|d|={(y_mix - y_ref).abs().max().item():.2e}")
with torch.no_grad():
    y_mix2 = am(x)
check("second forward deterministic", torch.equal(y_mix, y_mix2))
field_dtypes = {n: getattr(am, n).dtype for n in
                ("wgud", "Ugu", "Cgu", "Cdn")}
check("field params aligned to host dtype (fp32)",
      all(dt == torch.float32 for dt in field_dtypes.values()))

# --------------------------------------- 3: bf16 host with fp32 gate weight
print("== 3. bf16 host + fp32 gate in module ==")
am2 = art_mod.FieldSparseMoe(routed_cfg2)
with torch.no_grad():
    for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn",
              "e_score_correction_bias"):
        getattr(am2, n).normal_(std=0.1)
    am2.gate.weight.data = am2.gate.weight.data.to(torch.float32)
    xb = x.to(torch.bfloat16)
    yb = am2(xb)
check("everything aligns down to bf16 host dtype",
      yb.dtype == torch.bfloat16 and torch.isfinite(yb).all())

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
