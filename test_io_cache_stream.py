#!/usr/bin/env python3
"""Integration test: BlockStreamRunner(io_cache="ram") on mini-OLMoE.

Pass 1 over the layers fills the process-wide raw-GGUF cache; pass 2 must be
served from RAM (hits grow, block loads get faster on slow storage). Logits of
both passes must be identical to the disk-mode runner (bit-exact).
Run: python3 test_io_cache_stream.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_env  # noqa: F401
from hf_stream import BlockStreamRunner

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "gguf_hf", "tiny-hf")
GGUF = os.path.join(HERE, "tiny.gguf")
DTYPE = torch.bfloat16


def run_passes(io_cache, n_passes=2):
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(SRC)
    ids = torch.tensor(tok("hello world this is a streaming test " * 6)
                       ["input_ids"][:64]).unsqueeze(0)
    r = BlockStreamRunner(SRC, dtype=DTYPE, device="cpu", gguf=GGUF,
                          prefetch=0, io_workers=1, io_cache=io_cache)
    outs, times = [], []
    for _ in range(n_passes):
        t0 = time.time()
        with torch.no_grad():
            outs.append(r(input_ids=ids).logits.clone())
        times.append(time.time() - t0)
    stats = dict(n=r._stream_n, gb=r._stream_bytes / 1e9,
                 t=r._load_time, hits=(r._gguf.cache_stats()[2]
                                       if r._gguf else 0))
    r.close()
    return outs, times, stats


def main():
    if not os.path.isfile(GGUF):
        sys.exit("tiny.gguf not found - generate it first: "
                 "python3 make_tiny_olmoe_gguf.py")
    print("== disk mode (reference)")
    out_d, t_d, st_d = run_passes("disk")
    print(f"   passes: {[f'{t:.2f}s' for t in t_d]} | "
          f"block loads {st_d['n']}, {st_d['gb']:.3f} GB")

    print("== ram mode")
    out_r, t_r, st_r = run_passes("ram")
    print(f"   passes: {[f'{t:.2f}s' for t in t_r]} | "
          f"block loads {st_r['n']}, {st_r['gb']:.3f} GB | "
          f"cache hits {st_r['hits']}")

    for i, (a, b) in enumerate(zip(out_d, out_r)):
        assert torch.equal(a, b), f"logits mismatch on pass {i}"
    assert st_r["hits"] > 0, "ram cache recorded no hits"
    print("IO-CACHE STREAM OK (logits bit-identical, cache hits recorded)")


if __name__ == "__main__":
    main()
