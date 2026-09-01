#!/usr/bin/env python3
"""A tiny FAKE hy_v3 GGUF (NanoColibri-like) for smoke tests.

Structure 1:1 like mradermacher/NanoColibri-Instruct-GGUF: layer 0 dense,
the rest MoE (64->8 experts simplified), shared experts, exp_probs_b, a tied
head (no output.weight). F32 + Q8_0 like the real file. Also writes a local
"base repo" (config.json) next to it for --gguf-base-repo.
"""
import json
import os
import sys

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_hyv3.gguf"
D, DFF, DFFD, DFFS = 32, 16, 24, 32          # d, expert dff, dense ffn, shexp
N_EXP, TOP_K, LAYERS, HEADS, KV, HD, CTX = 8, 2, 2, 4, 2, 8, 256

def bytes_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return [chr(c) for c in cs]

BYTE_TOKS = bytes_unicode()
VOCAB = 2 + len(BYTE_TOKS)

rng = np.random.default_rng(11)
w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)
norm = lambda d: (np.ones(d, np.float32) + rng.standard_normal(d) * 0.01).astype(np.float32)

def q8(x):
    """Q8_0: blocks of 32, scale = max|x|/127 (as in gguf-py)."""
    x = np.asarray(x, np.float32).reshape(-1, 32)
    sc = np.abs(x).max(1) / 127.0
    sc = np.where(sc == 0, 1e-30, sc).astype(np.float16)
    q = np.rint(x / sc[:, None].astype(np.float32)).astype(np.int8)
    return q.reshape(-1), sc

try:  # gguf-py can quantize by itself - not used here (the writer does not write Q8_0)
    from gguf.quants import quantize  # noqa: F401
except Exception:
    pass

wr = GGUFWriter(OUT, "hy_v3")
wr.add_architecture()
wr.add_block_count(LAYERS)
wr.add_context_length(CTX)
wr.add_embedding_length(D)
wr.add_uint32("hy_v3.feed_forward_length", DFFD)                     # dense layer
wr.add_uint32("hy_v3.attention.head_count", HEADS)
wr.add_uint32("hy_v3.attention.head_count_kv", KV)
wr.add_uint32("hy_v3.attention.key_length", HD)
wr.add_uint32("hy_v3.attention.value_length", HD)
wr.add_float32("hy_v3.attention.layer_norm_rms_epsilon", 1e-5)
wr.add_float32("hy_v3.rope.freq_base", 10000.0)
wr.add_uint32("hy_v3.expert_count", N_EXP)
wr.add_uint32("hy_v3.expert_used_count", TOP_K)
wr.add_uint32("hy_v3.expert_feed_forward_length", DFF)
wr.add_uint32("hy_v3.expert_shared_feed_forward_length", DFFS)
wr.add_uint32("hy_v3.expert_gating_func", 2)
wr.add_bool("hy_v3.expert_weights_norm", True)
wr.add_float32("hy_v3.expert_weights_scale", 2.826)

tokens = ["<|padding|>", "|||IP_ADDRESS|||"] + BYTE_TOKS
wr.add_tokenizer_model("gpt2")
wr.add_token_list(tokens)
wr.add_token_types([3, 3] + [1] * len(BYTE_TOKS))
wr.add_token_scores([0.0] * VOCAB)
wr.add_token_merges([f"{a} {b}" for a, b in zip(BYTE_TOKS[:4], BYTE_TOKS[1:5])])
wr.add_eos_token_id(1)
wr.add_pad_token_id(0)
wr.add_add_bos_token(False)

def T(name, arr, quant=False):
    # the gguf-py writer only writes F types: tiny F32 tensors; the Q8_0
    # dequant code is gguf-py's, already battle-tested on real Q4_K_M/Q8_0 GGUFs
    wr.add_tensor(name, np.ascontiguousarray(arr, np.float32))

T("token_embd.weight", w(VOCAB, D))                                   # tied: output.weight ABSENT
T("output_norm.weight", norm(D))
for i in range(LAYERS):
    p = f"blk.{i}"
    T(f"{p}.attn_norm.weight", norm(D))
    T(f"{p}.attn_q_norm.weight", norm(HD))
    T(f"{p}.attn_k_norm.weight", norm(HD))
    T(f"{p}.attn_q.weight", w(D, D), quant=True)
    T(f"{p}.attn_k.weight", w(KV * HD, D), quant=True)
    T(f"{p}.attn_v.weight", w(KV * HD, D), quant=True)
    T(f"{p}.attn_output.weight", w(D, D), quant=True)
    T(f"{p}.ffn_norm.weight", norm(D))
    if i == 0:                                                        # dense layer 0
        T(f"{p}.ffn_gate.weight", w(DFFD, D), quant=True)
        T(f"{p}.ffn_up.weight", w(DFFD, D), quant=True)
        T(f"{p}.ffn_down.weight", w(D, DFFD), quant=True)
    else:                                                             # MoE layer
        T(f"{p}.ffn_gate_inp.weight", (w(N_EXP, D) * 3).astype(np.float32))
        T(f"{p}.exp_probs_b", (rng.standard_normal(N_EXP) * 0.1).astype(np.float32))
        T(f"{p}.ffn_gate_exps.weight", w(N_EXP, DFF, D), quant=True)
        T(f"{p}.ffn_up_exps.weight", w(N_EXP, DFF, D), quant=True)
        T(f"{p}.ffn_down_exps.weight", w(N_EXP, D, DFF), quant=True)
        T(f"{p}.ffn_gate_shexp.weight", w(DFFS, D), quant=True)
        T(f"{p}.ffn_up_shexp.weight", w(DFFS, D), quant=True)
        T(f"{p}.ffn_down_shexp.weight", w(D, DFFS), quant=True)

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()

base = os.path.splitext(OUT)[0] + "-base"
os.makedirs(base, exist_ok=True)
with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
    json.dump(dict(model_type="hy_v3", architectures=["HYV3ForCausalLM"],
                   vocab_size=VOCAB, hidden_size=D, intermediate_size=DFFD,
                   num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                   num_key_value_heads=KV, head_dim=HD, hidden_act="silu",
                   max_position_embeddings=CTX, rms_norm_eps=1e-5,
                   moe_intermediate_size=DFF, num_experts=N_EXP,
                   num_experts_per_tok=TOP_K, num_shared_experts=DFFS // DFF,
                   router_scaling_factor=2.826, enable_moe_fp32_combine=True,
                   tie_word_embeddings=True, mlp_bias=False, attention_bias=False,
                   rope_parameters={"rope_type": "default", "rope_theta": 10000.0}),
              f, indent=2)
print(f"OK: {OUT} ({os.path.getsize(OUT)} bytes) + base repo: {base}/")
