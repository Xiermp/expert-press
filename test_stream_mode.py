#!/usr/bin/env python3
"""A/B test of BlockStreamRunner vs the full model on mini-OLMoE:
logits, calibration pairs, centroids, generation. Bit-exact equality expected.
"""
import os
import resource
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_env  # noqa: F401
from hf_field_transform import (collect_pairs, expert_means, find_moe_blocks,
                                generate_text, make_batches, router_weight)
from hf_stream import BlockStreamRunner

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "gguf_hf", "tiny-hf")
DTYPE = torch.bfloat16


def gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(SRC)
    text = "hello world this is a streaming test " * 6
    ids = torch.tensor(tok(text)["input_ids"])

    print("== full model")
    full = AutoModelForCausalLM.from_pretrained(SRC, dtype=DTYPE).eval()
    x = ids[:64].unsqueeze(0)
    with torch.no_grad():
        A = full(input_ids=x).logits.clone()
    blocks_f = find_moe_blocks(full)
    gen = torch.Generator().manual_seed(11)
    batches = make_batches(ids, 48, 2, 2, "cpu", gen)
    with torch.no_grad():
        pairs_f = collect_pairs(full, blocks_f, batches, 4096)
    means_f = [expert_means(b) for _, b in blocks_f]
    gw_f = router_weight(blocks_f[0][1].gate).clone()
    gen_f = generate_text(full, ids, tok, n_new=8)
    del full

    print("== streaming")
    r = BlockStreamRunner(SRC, dtype=DTYPE)
    # regression guard: experts must NOT materialize in the backbone
    # (RAM would hit ~14 GB instead of ~1 GB - Windows would kill the process)
    exp_meta = all(p.device.type == "meta"
                   for n, p in r.model.named_parameters(remove_duplicate=False)
                   if ".experts." in n)
    print(f"experts not materialized in the backbone (meta): {exp_meta}")
    with torch.no_grad():
        B = r(input_ids=x).logits.clone()
    blocks_s = find_moe_blocks(r)
    with torch.no_grad():
        pairs_s = collect_pairs(r, blocks_s, batches, 4096)
    means_s = []
    for i, (_, b) in enumerate(blocks_s):
        with r.with_block(i):
            means_s.append(expert_means(b))
    gw_s = router_weight(blocks_s[0][1].gate).clone()
    gen_s = generate_text(r, ids, tok, n_new=8)
    r.close()

    d_logits = (A - B).abs().max().item()
    ok = [d_logits == 0.0, exp_meta]
    print(f"\nlogits: max|A-B| = {d_logits}")
    for i, ((Xf, Yf), (Xs, Ys)) in enumerate(zip(pairs_f, pairs_s)):
        dx = (Xf - Xs).abs().max().item()
        dy = (Yf - Ys).abs().max().item()
        print(f"block {i} pairs: max|dX|={dx} max|dY|={dy}")
        ok += [dx == 0.0, dy == 0.0]
    for i, (mf, ms) in enumerate(zip(means_f, means_s)):
        d = max((a - b).abs().max().item() for a, b in zip(mf, ms))
        print(f"block {i} centroids: max|d|={d}")
        ok.append(d == 0.0)
    dgw = (gw_f - gw_s).abs().max().item()
    print(f"router weight: max|d|={dgw}")
    ok.append(dgw == 0.0)
    same_gen = gen_f == gen_s
    print(f"generation matched: {same_gen}")
    print(f"generation: {gen_f[:60]!r}")
    ok.append(same_gen)
    print(f"\nRAM peak: {gb():.2f} GB")
    print("RESULT:", "PASS" if all(ok) else "FAIL")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
