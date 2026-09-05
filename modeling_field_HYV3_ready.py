# ГОТОВЫЙ ФАЙЛ - см. шапку ниже.
"""Рантайм "поле-движка" — ОБЫЧНАЯ HF-модель, загружается стандартно:
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    (рекомендуется dtype="bfloat16", low_cpu_mem_usage=True)
Сгенерирован hf_pipeline.py. Требует transformers>=5 (SparseMoeBlock.forward
-> Tensor). # version: 2026-09-05.4 - dtype-mismatch fix: v5-роутеры возвращают
fp32-логиты/веса (upcast внутри), а чекпойнт бывает смешанным (бэкбон fp32 из
GGUF-декванта + поле bf16): matmul'ы (@) в torch не смешивают dtype и падали
(float != BFloat16). Поле теперь один раз выравнивает свои тензоры под dtype
вычислений хоста и приводит z к нему же (см. UPDATE-10, 10.6).

Идея: явных весов экспертов нет. Есть "поле": центроиды (w*bd) + низкоранговые
факторы U,V и координаты C. "Сид движения" c(z) = z @ C вычисляется из роутера:
    z = topk(gate(x))
    gu(x) = x@wGud^T + (x@Vgu * cgu)@Ugu^T      # fused gate+up
    h = silu(gate_part) * up_part
    y  = h@wDnd^T + (h@Vdn * cdn)@Udn^T         # down
Один проход FFN вместо top-k проходов (FLOPs/2 при top-2, ~x8 при top-8).

hy_v3 (NanoColibri и родня, DeepSeek-стиль роутинга), 2026-09-05.3:
  - роутер берёт порог выбора эксперт-скор аргументом forward(x, bias)
    (transformers 5.16+; на старых билдах bias живёт внутри роутера - вызов
    подстраивается по сигнатуре автоматически); bias приезжает из чекпойнта
    буфером <layer>.mlp.e_score_correction_bias;
  - блок несёт always-on shared_experts: их веса сохранены в артефакте
    (<layer>.mlp.shared_experts.*) и прибавляются к выходу поля в fp32 -
    ровно как в базовой модели и как при фите (таргет фита = блок минус
    shared-ветка, поле учит только routed-часть);
  - у v5-роутеров топк-веса УЖЕ нормализованы и умножены на scaling-фактор
    внутри роутера - поле берёт их как есть (то же делает фит-сторона);
  - dtype живёт по правилу "как у входа": при первом forward все плавающие
    тензоры модуля приводятся к dtype скрытых состояний хоста (смешанный
    чекпойнт или fp32-логиты v5-роутера больше не роняют matmul'ы).
"""
import inspect

import torch
import torch.nn as nn
from transformers import HYV3ForCausalLM
from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter
from transformers.activations import ACT2FN


class FieldSparseMoe(nn.Module):
    """Замена SparseMoeBlock: тот же роутер, эксперты собираются из поля."""

    def __init__(self, config):
        super().__init__()
        fi = config.field
        d, dff, r = fi["d_model"], fi["d_ff"], fi["rank"]
        self._field_dtype = None                     # выравнивание ещё не делали
        self.top_k = int(fi["top_k"])
        self.norm_topk = bool(fi.get("norm_topk"))
        self.gate = HYV3TopKRouter(config)                 # роутер не трогаем (вес из базы)
        self.act_fn = ACT2FN[config.hidden_act]
        # hy_v3: bias выбора в 5.16+ - аргумент forward(x, bias), на старых
        # билдах он внутри роутера. Определяем по фактической сигнатуре хоста
        # (+ подсказка router_kind="sigmoid_bias" из геометрии фита); сам
        # буфер заполняется из чекпойнта при загрузке.
        try:
            takes_bias = "e_score_correction_bias" in inspect.signature(
                type(self.gate).forward).parameters
        except (TypeError, ValueError):   # экзотический хост - смотрим флаг
            takes_bias = False
        self._gate_takes_bias = takes_bias or \
            str(fi.get("router_kind", "") or "") == "sigmoid_bias"
        if self._gate_takes_bias:
            self.register_buffer("e_score_correction_bias",
                                 torch.zeros(fi["n_exp"]))
        dffs = int(fi.get("dff_shexp", 0) or 0)    # hy_v3: always-on ветка
        if dffs:
            se = nn.Module()                       # ключи как у базовой модели:
            se.gate_proj = nn.Linear(d, dffs, bias=False)   # shared_experts.*
            se.up_proj = nn.Linear(d, dffs, bias=False)     # gate/up/down_proj
            se.down_proj = nn.Linear(dffs, d, bias=False)
            self.shared_experts = se
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(f"w{nm}d", nn.Parameter(torch.zeros(out, inp)))
            self.register_parameter(f"U{nm}", nn.Parameter(torch.zeros(out, r)))
            self.register_parameter(f"V{nm}", nn.Parameter(torch.zeros(inp, r)))
        self.Cgu = nn.Parameter(torch.zeros(fi["n_exp"], r))
        self.Cdn = nn.Parameter(torch.zeros(fi["n_exp"], r))

    def _align_field_dtype(self, dtype):
        """Один раз приводим все плавающие тензоры модуля к dtype вычислений
        хоста. Зачем: (1) v5-роутеры отдают fp32-логиты/веса независимо от
        dtype модели; (2) чекпойнт, загруженный "как сохранён" (dtype=None),
        бывает смешанным - бэкбон fp32 (GGUF-деквант) + поле bf16 (ради
        размера). Сложения в torch приводят типы сами, а matmul'ы (@) - нет,
        отсюда float != BFloat16. Вверх (bf16->fp32) приводим без потерь,
        вниз - только если весь хост уже в этом dtype."""
        for _, p in self.named_parameters(recurse=True):
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype)
        if getattr(self, "_gate_takes_bias", False):
            b = self.e_score_correction_bias
            if b.is_floating_point() and b.dtype != dtype:
                self.e_score_correction_bias = b.to(dtype)
        self._field_dtype = dtype

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        if self._field_dtype != x.dtype:           # один раз на запуск
            self._align_field_dtype(x.dtype)
        if self._gate_takes_bias:                  # hy_v3: sigmoid + bias-порог
            try:
                out = self.gate(x, self.e_score_correction_bias)
            except TypeError:                      # старый хост: bias внутри
                out = self.gate(x)                 # роутера - зовём по-старому
            logits, scores, idx = out[0], out[1], out[2]
        else:
            gout = self.gate(x)
            if isinstance(gout, (tuple, list)):    # v5-роутер: (logits, scores, idx)
                logits, scores, idx = gout[0], gout[1], gout[2]
            else:                                  # обычный Linear-роутер
                logits = gout
                probs = torch.softmax(logits.float(), dim=-1)
                scores, idx = torch.topk(probs, self.top_k, dim=-1)
                if self.norm_topk:
                    scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores).to(x.dtype)
        cgu, cdn = z @ self.Cgu, z @ self.Cdn      # сид движения (T,r)
        gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        y = h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t()
        if hasattr(self, "shared_experts"):        # hy_v3: shared-ветка, fp32
            se = self.shared_experts               # combine - как в базе
            ys = se.down_proj(self.act_fn(se.gate_proj(x)) * se.up_proj(x))
            y = (y.float() + ys.float()).to(y.dtype)
        return y.view(B, T, -1)


class FieldForCausalLM(HYV3ForCausalLM):
    """Базовая архитектура, где каждый MoE-блок заменён на поле."""

    def __init__(self, config):
        super().__init__(config)
        n = 0
        for name, mod in list(self.named_modules()):
            if hasattr(mod, "experts") and hasattr(mod, "gate"):
                parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
                setattr(parent, name.rsplit(".", 1)[-1], FieldSparseMoe(config))
                n += 1
        expected = config.field["n_layers"]
        if n != expected:
            raise RuntimeError(f"заменили {n} MoE-блоков, ожидалось {expected}")
