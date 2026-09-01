#!/usr/bin/env python3
"""A/B test: streaming straight from the GGUF (on-the-fly dequant) vs a full
dequant checkpoint. BIT-EXACT equality expected:
  1. every GgufHfSource tensor == the tensor from the dequant checkpoint;
  2. logits/pairs/centroids/generation of BlockStreamRunner(gguf=...) == (dir);
  3. artifacts write_field_artifact(gguf=...) and (dir) - identical weights.
"""
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_env  # noqa: F401
from hf_field_transform import (collect_pairs, expert_means, find_moe_blocks,
                                generate_text, make_batches, router_weight,
                                write_field_artifact)
from hf_stream import BlockStreamRunner

BASE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(BASE, "tiny.gguf")
DIR = os.path.join(BASE, "results", "gguf_hf", "tiny-hf")
DTYPE = torch.bfloat16


def main():
    ok = []
    from safetensors import safe_open
    from hf_gguf_to_hf import GgufHfSource

    # ---- 1. tensors: GGUF on-the-fly dequant == dequant checkpoint
    gs = GgufHfSource(GGUF)
    n_cmp = 0
    for key in gs.keys():
        t_gguf = gs.get(key)
        found = None
        for fn in sorted(f for f in os.listdir(DIR) if f.endswith(".safetensors")):
            with safe_open(os.path.join(DIR, fn), framework="pt") as f:
                if key in f.keys():
                    found = f.get_tensor(key)
                    break
        if found is None:
            print(f"  !! {key}: missing from the dequant checkpoint")
            ok.append(False)
            continue
        d = (t_gguf.float() - found.float()).abs().max().item()
        same = d == 0.0 and t_gguf.shape == found.shape
        if not same:
            print(f"  !! {key}: max|d|={d} shapes {tuple(t_gguf.shape)} vs "
                  f"{tuple(found.shape)}")
        ok.append(same)
        n_cmp += 1
    print(f"1) tensors GGUF vs dequant checkpoint: {n_cmp} compared, "
          f"{'all matched' if all(ok) else 'MISMATCH FOUND'}")

    # ---- 2. end-to-end run: logits/pairs/centroids/generation
    if not all(ok):
        print("RESULT: FAIL (tensors diverged)")
        sys.exit(1)
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DIR)
    ids = torch.tensor(tok("hello world this is a gguf direct test " * 6)
                       ["input_ids"])
    x = ids[:64].unsqueeze(0)
    gen = torch.Generator().manual_seed(11)
    batches = make_batches(ids, 48, 2, 2, "cpu", gen)

    print("== streaming from the dequant checkpoint")
    r1 = BlockStreamRunner(DIR, dtype=DTYPE)
    ok.append(all(p.device.type == "meta"
                  for n, p in r1.model.named_parameters(remove_duplicate=False)
                  if ".experts." in n))
    print(f"experts on meta in the backbone: {ok[-1]}")
    with torch.no_grad():
        A = r1(input_ids=x).logits.clone()
    b1 = find_moe_blocks(r1)
    with torch.no_grad():
        p1 = collect_pairs(r1, b1, batches, 4096)
    m1 = []
    for i, (_, b) in enumerate(b1):
        with r1.with_block(i):
            m1.append(expert_means(b))
    gw1 = router_weight(b1[0][1].gate).clone()
    g1 = generate_text(r1, ids, tok, n_new=8)
    r1.close()

    print("== streaming from the GGUF (on-the-fly dequant)")
    r2 = BlockStreamRunner(DIR, dtype=DTYPE, gguf=GGUF)
    ok.append(all(p.device.type == "meta"
                  for n, p in r2.model.named_parameters(remove_duplicate=False)
                  if ".experts." in n))
    print(f"experts on meta in the backbone (GGUF mode): {ok[-1]}")
    with torch.no_grad():
        B = r2(input_ids=x).logits.clone()
    b2 = find_moe_blocks(r2)
    with torch.no_grad():
        p2 = collect_pairs(r2, b2, batches, 4096)
    m2 = []
    for i, (_, b) in enumerate(b2):
        with r2.with_block(i):
            m2.append(expert_means(b))
    gw2 = router_weight(b2[0][1].gate).clone()
    g2 = generate_text(r2, ids, tok, n_new=8)
    r2.close()

    dl = (A - B).abs().max().item()
    print(f"2) logits: max|d| = {dl}")
    ok.append(dl == 0.0)
    for i, ((Xa, Ya), (Xb, Yb)) in enumerate(zip(p1, p2)):
        dx = (Xa - Xb).abs().max().item()
        dy = (Ya - Yb).abs().max().item()
        print(f"   block {i} pairs: max|dX|={dx} max|dY|={dy}")
        ok += [dx == 0.0, dy == 0.0]
    for i, (ma, mb) in enumerate(zip(m1, m2)):
        d = max((a - b).abs().max().item() for a, b in zip(ma, mb))
        print(f"   block {i} centroids: max|d|={d}")
        ok.append(d == 0.0)
    dgw = (gw1 - gw2).abs().max().item()
    print(f"   router weight: max|d|={dgw}; generation matched: {g1 == g2}")
    ok += [dgw == 0.0, g1 == g2]

    # ---- 3. artifact: gguf mode vs dir mode (backbone from the same fit cache)
    pool = os.path.join(BASE, "results", "cache_tiny.gguf")
    fit = os.path.join(pool, "fit_r32")
    if os.path.isdir(fit) and os.path.isfile(os.path.join(pool, "art_meta.json")):
        out_a = os.path.join(BASE, "results", "_art_gguf")
        out_b = os.path.join(BASE, "results", "_art_dir")
        for d in (out_a, out_b):
            shutil.rmtree(d, ignore_errors=True)
        write_field_artifact(DIR, out_b, pool, fit, 32, DTYPE)
        write_field_artifact(DIR, out_a, pool, fit, 32, DTYPE, gguf=GGUF)
        def load_sd(d):
            sd = {}
            for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
                with safe_open(os.path.join(d, fn), framework="pt") as f:
                    for k in f.keys():
                        sd[k] = f.get_tensor(k)
            return sd
        sa, sb = load_sd(out_a), load_sd(out_b)
        keys_a = {k for k in sa if ".experts." not in k}
        keys_b = {k for k in sb if ".experts." not in k}
        same_keys = keys_a == keys_b
        dmax = max((sa[k].float() - sb[k].float()).abs().max().item()
                   for k in keys_a & keys_b) if same_keys and keys_a else -1
        print(f"3) artifact: keys matched: {same_keys} "
              f"({len(keys_a)} tensors), max|d| = {dmax}")
        ok += [same_keys, dmax == 0.0]
        shutil.rmtree(out_a, ignore_errors=True)
        shutil.rmtree(out_b, ignore_errors=True)
    else:
        print("3) artifact: fit cache not found - skipping")

    print(f"\nRESULT:", "PASS" if all(ok) else "FAIL")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
