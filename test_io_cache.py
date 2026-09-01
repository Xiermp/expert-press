"""Smoke test for --io-cache ram: GgufHfSource with cache_ram=True.

Checks:
  1. tensors fetched with cache_ram=True are bit-identical to cache_ram=False;
  2. the second fetch of the same tensor is served from the process-wide RAM
     cache (hit counter grows) and still bit-identical;
  3. two separate GgufHfSource instances share the cache (hits are counted
     across instances) - this is what saves the refit/artifact passes;
  4. cache_stats() reports non-zero GB.
Run: python3 test_io_cache.py [path/to/model.gguf]
"""
import os
import sys
import numpy as np

import hf_env  # noqa: F401

from hf_gguf_to_hf import GgufHfSource, raw_cache_stats, raw_cache_clear

GGUF = sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf"
if not os.path.isfile(GGUF):
    sys.exit(f"no GGUF: {GGUF} - pass a path or generate one via "
             f"make_tiny_olmoe_gguf.py")

raw_cache_clear()

# ---- disk mode (reference) -------------------------------------------------
src_disk = GgufHfSource(GGUF, io_workers=1, cache_ram=False)
keys = src_disk.keys()
probe = [k for k in keys if ".experts." in k][:2] + \
        [k for k in keys if ".experts." not in k][:2]
ref = {}
for hf in probe:
    t = src_disk.get(hf)
    ref[hf] = t.numpy().copy()
    print(f"  disk  get {hf}: {tuple(t.shape)} {t.dtype}")
src_disk = None

# ---- ram mode, first instance ----------------------------------------------
src_ram = GgufHfSource(GGUF, io_workers=1, cache_ram=True)
for hf in probe:
    t1 = src_ram.get(hf)                      # fills the cache (miss)
    t2 = src_ram.get(hf)                      # served from RAM (hit)
    assert np.array_equal(t1.numpy(), ref[hf]), f"mismatch (miss): {hf}"
    assert np.array_equal(t2.numpy(), ref[hf]), f"mismatch (hit): {hf}"
    print(f"  ram   get {hf}: identical to disk (miss+hit)")
n, gb, hits = src_ram.cache_stats()
print(f"  cache after inst#1: {n} tensors, {gb:.3f} GB, {hits} hits")
assert n > 0 and gb > 0, "cache did not fill"
assert hits >= len(probe), f"expected >= {len(probe)} hits, got {hits}"

# ---- ram mode, second instance (the refit/artifact-writer scenario) --------
src_ram2 = GgufHfSource(GGUF, io_workers=1, cache_ram=True)
before = src_ram2.cache_stats()[2]
for hf in probe:
    t = src_ram2.get(hf)
    assert np.array_equal(t.numpy(), ref[hf]), f"mismatch (inst#2): {hf}"
after = src_ram2.cache_stats()[2]
print(f"  inst#2 hits: {after - before} (all served from RAM, no disk)")
assert after - before >= len(probe), "second instance did not hit the cache"

print("IO-CACHE SMOKE OK")
