"""Рантайм "поле-движка" — ОБЫЧНАЯ HF-модель, загружается стандартно:
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    (рекомендуется dtype="bfloat16", low_cpu_mem_usage=True)
Сгенерирован hf_pipeline.py. Требует transformers>=5 (SparseMoeBlock.forward -> Tensor).

Идея: явных весов экспертов нет. Есть "поле": центроиды (w*bd) + низкоранговые
факторы U,V и координаты C. "Сид движения" c(z) = z @ C вычисляется из роутера:
    z = topk(softmax(gate(x)))
    gu(x) = x@wGud^T + (x@Vgu * cgu)@Ugu^T      # fused gate+up
    h = silu(gate_part) * up_part
    y  = h@wDnd^T + (h@Vdn * cdn)@Udn^T         # down
Один проход FFN вместо top-k проходов (FLOPs/2 при top-2, ~x8 при top-8).
"""
import torch
import torch.nn as nn
from transformers import @@BASE@@
from @@ROUTER_MOD@@ import @@ROUTER@@
from transformers.activations import ACT2FN


class FieldSparseMoe(nn.Module):
    """Замена SparseMoeBlock: тот же роутер, эксперты собираются из поля."""

    def __init__(self, config):
        super().__init__()
        fi = config.field
        d, dff, r = fi["d_model"], fi["d_ff"], fi["rank"]
        self.gate = @@ROUTER@@(config)                 # роутер не трогаем (вес из базы)
        self.act_fn = ACT2FN[config.hidden_act]
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(f"w{nm}d", nn.Parameter(torch.zeros(out, inp)))
            self.register_parameter(f"U{nm}", nn.Parameter(torch.zeros(out, r)))
            self.register_parameter(f"V{nm}", nn.Parameter(torch.zeros(inp, r)))
        self.Cgu = nn.Parameter(torch.zeros(fi["n_exp"], r))
        self.Cdn = nn.Parameter(torch.zeros(fi["n_exp"], r))

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        gout = self.gate(x)
        if isinstance(gout, (tuple, list)):            # v5-роутер: (logits, scores, idx)
            logits, scores, idx = gout[0], gout[1], gout[2]
        else:                                          # обычный Linear-роутер
            logits = gout
            probs = torch.softmax(logits.float(), dim=-1)
            scores, idx = torch.topk(probs, fi["top_k"], dim=-1)
            if fi.get("norm_topk"):
                scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores)
        cgu, cdn = z @ self.Cgu, z @ self.Cdn          # сид движения (T,r)
        gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        y = h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t()
        return y.view(B, T, -1)


class FieldForCausalLM(@@BASE@@):
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
