#!/usr/bin/env python3
"""A tiny FAKE olmoe GGUF for converter smoke tests (F32, random weights).

Tensor and metadata structure 1:1 like mradermacher/OLMoE-1B-7B-0924-GGUF
(only minimal sizes: 2 layers, d=32, dff=16, 8 experts, top-2).
"""
import os
import sys

import numpy as np
from gguf import GGUFWriter

OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_olmoe.gguf"
# vocab: 2 special tokens + 256 byte-level chars (so the fake tokenizer can
# encode any text and ids never exceed the vocab)
D, DFF, N_EXP, TOP_K, LAYERS, HEADS, KV, CTX = 32, 16, 8, 2, 2, 4, 4, 256


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

rng = np.random.default_rng(7)
w = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
norm = lambda d: (np.ones(d, np.float32) + rng.standard_normal(d) * 0.01).astype(np.float32)

wr = GGUFWriter(OUT, "olmoe")
wr.add_architecture()
wr.add_block_count(LAYERS)
wr.add_context_length(CTX)
wr.add_embedding_length(D)
wr.add_feed_forward_length(DFF)
wr.add_uint32("olmoe.attention.head_count", HEADS)
wr.add_attention_head_count_kv = lambda n: wr.add_uint32("olmoe.attention.head_count_kv", n)
wr.add_attention_head_count_kv(KV)
wr.add_float32("olmoe.attention.layer_norm_rms_epsilon", 1e-5)
wr.add_rope_freq_base(10000.0)
wr.add_expert_count(N_EXP)
wr.add_expert_used_count(TOP_K)

tokens = ["<|padding|>", "|||IP_ADDRESS|||"] + BYTE_TOKS
types = [3, 3] + [1] * len(BYTE_TOKS)
wr.add_tokenizer_model("gpt2")
wr.add_tokenizer_pre("olmo")
wr.add_token_list(tokens)
wr.add_token_types(types)
wr.add_token_scores([0.0] * VOCAB)
wr.add_token_merges([f"{a} {b}" for a, b in zip(BYTE_TOKS[:4], BYTE_TOKS[1:5])])
wr.add_eos_token_id(1)
wr.add_pad_token_id(0)
wr.add_add_bos_token(False)
wr.add_add_eos_token(False)

wr.add_tensor("token_embd.weight", w(VOCAB, D))
wr.add_tensor("output_norm.weight", norm(D))
wr.add_tensor("output.weight", w(VOCAB, D))
for i in range(LAYERS):
    p = f"blk.{i}"
    wr.add_tensor(f"{p}.attn_norm.weight", norm(D))
    wr.add_tensor(f"{p}.attn_q_norm.weight", norm(D))
    wr.add_tensor(f"{p}.attn_k_norm.weight", norm(D))
    wr.add_tensor(f"{p}.attn_q.weight", w(D, D))
    wr.add_tensor(f"{p}.attn_k.weight", w(D, D))
    wr.add_tensor(f"{p}.attn_v.weight", w(D, D))
    wr.add_tensor(f"{p}.attn_output.weight", w(D, D))
    wr.add_tensor(f"{p}.ffn_norm.weight", norm(D))
    wr.add_tensor(f"{p}.ffn_gate_inp.weight", (w(N_EXP, D) * 3).astype(np.float32))
    # layout as in the real file: [N, dff, d] -> ne=[d, dff, N]
    wr.add_tensor(f"{p}.ffn_gate_exps.weight", w(N_EXP, DFF, D))
    wr.add_tensor(f"{p}.ffn_up_exps.weight", w(N_EXP, DFF, D))
    wr.add_tensor(f"{p}.ffn_down_exps.weight", w(N_EXP, D, DFF))

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print(f"OK: {OUT} ({os.path.getsize(OUT)} bytes)")
