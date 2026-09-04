# UPDATE-10 - 2026-09-04.2 - speed rebuild + the three holes

Files changed: `hf_field_transform.py` (version stamp 2026-09-04.2),
`hf_pipeline.py` (2026-09-04.2), new `bench_toy_speed.py`,
new `test_update10_speed.py` (11 checks), README sections.

## Context
User report (morning 2026-09-04), three holes found while reviewing the fit:
1. U,V initialized as white noise, C as zeros - while the real expert deltas
   are on disk (SVD init proposal).
2. The muon split `min(shape) <= muon_max_dim` swallows Cgu/Cdn
   (coordinate tables) - Newton-Schulz would couple unrelated experts.
3. Under jitter the routing was recomputed on the NOISY input while the
   targets were produced on the clean rows (irreducible mse floor).
Plus the standing complaint: the fit is too slow (the 8.x speed stack was
lost with the old environment; the 9.x base was v7-speed).

## What was done (toy bench FIRST: scripts/bench_toy_speed.py, then the repo)
- Bench A (hole 1): SVD of the mean delta is degenerate (sum dW = 0); the
  stacked-delta shared basis is the correct variant. Streaming randomized
  SVD (2 passes over experts, one expert in RAM) == exact Grams to 3
  decimals. Step-0 diag capture 49/68% (gu/dn), subspace-pair capture ~100%;
  held-out mse after 120 identical steps: 0.00198 (svd) vs 0.00548 (random).
  Steps-to-quality: random-120 quality is reached by svd at ~30-60 steps.
- Bench B (hole 2): BY-NAME split (U/V -> Muon; C, w, router -> Adam) loses
  nothing vs the hole (0.002312 vs 0.002319); Muon on the BIG matrices is
  the toy's best arm (0.001706) - exposed as --muon-max-dim, default 512
  keeps real-model centroid matrices on Adam (NS there costs ~40-50%/step).
- Bench C (hole 3): clean-anchor routing >= noisy-routing at jitter 0.15 and
  0.6 (and one _z recompute per step cheaper); adopted as THE behavior.
- Bench D (speed): bf16-autocast (params fp32) 1.71-1.79x per step on a
  2-core CPU box; honest probe rule >=1.2x protects non-AMX machines.
- Bench E (wall clock): autocast x1.7-1.8 combined with svd-init x2-4 fewer
  steps = ~3-6x faster to the same quality.

## Implementation
- hf_field_transform.py: `_ns_orth`, `_Muon`, `_muon_split` (BY NAME),
  `_OptPair`; `_time_fit_arms` / `_resolve_fit_autocast` (honest probe,
  cache keyed by (device, method, bs, jitter, threads, jitter?, train_router,
  sh_gu, n_tok, shapes), RNG 0xC0FFEE, bit-exact param restore incl.
  requires_grad state - caught by tests); `fit_field_module(..., autocast,
  muon_max_dim, muon_ns_steps)` with muon/muon-cosine (manual warmup-aware
  cosine), clean-anchor z in the fit loop AND _eval8, guard reports the
  init-state baseline plus the pure-centroid reference (C zeroed/restored);
  `expert_basis_init` + `_iter_expert_w` + `_topk_eigh` (streaming 2-pass
  randomized SVD, per-side skip when rank > min(out,in), true diag capture +
  projection capture diagnostics).
- hf_pipeline.py: flags `--fit-init {svd,random}` (default svd),
  `--fit-autocast {auto,on,off}` (default auto), `--muon-max-dim`,
  `--muon-ns-steps`, `--refresh-init`; stage 4 saves init_svd_blk*.pt
  (per rank, inside the streaming with_block); refresh-only mode opens the
  model, rebuilds the inits for an existing pool and exits; stage 5 merges
  init_svd into the module init, fit_sig carries init/autocast/muon keys
  (old fits invalidate automatically); stage 5b refine passes the same
  kwargs; print_profile shows init/autocast/muon; _compat_check requires
  expert_basis_init + _resolve_fit_autocast.

## Tests / smoke
- test_update10_speed.py: 11/11 (NS spectrum flattening, split by name +
  dim gate, Muon convergence, probe bit-exact restore incl. requires_grad,
  probe modes + cache, muon-cosine determinism + fp32 params, autocast-on
  fit, clean-anchor z called exactly once, svd-init beats random at step 0 +
  proj capture > 0.8, muon+joint-router anchored, CLI parses).
- test_update9_router_guard.py: 11/11 (no regressions).
- Smoke 1 (fresh, tiny.gguf, --smoke --fit-method muon-cosine --fit-autocast
  auto): PIPELINE FINISHED 12.3 s, KL -0.002 bits, artifact + report written.
- Smoke 2 (--refresh-init on an update-9 pool cache, then --stages
  fit,save,verify): refresh wrote 2/2 init files (with the per-side skip for
  the degenerate tiny geometry), the refit ran (fit_sig init=svd), 8.4 s.

## Upgrade path for the user
1. Unpack the ZIP over the old folder (or fresh).
2. `python3 hf_pipeline.py <usual args> --refresh-init` (one-time, ~minutes).
3. Re-run the fit: same quality in ~3x fewer steps
   (e.g. `--fit-steps 150` where 300+ was needed), or the same steps for
   better quality; `--fit-method muon-cosine` is available and safe
   (C/router never touch Newton-Schulz).

## 2026-09-04.3 - RESUME FIX ("after a restart it starts from scratch")

Report: an interruption during the silent stage-4 init loop made the next run
recalibrate from zero ("as if nothing was saved"). Root causes found and fixed:

1. `art_meta.json` was written LAST in stage 4, and every cache check requires
   it -> a kill between "pairs flushed" and "meta written" invalidated the
   whole pair pool. Now art_meta is written at the START of stage 4 (it only
   describes the block inventory, known before any heavy work).
2. The stage-4 init loop (centroids + SVD init, the silent minutes) is now
   RESUMABLE per block: existing `init_blk*.pt` are reused (inits depend only
   on the model), only missing ones are rebuilt; per-block progress prints +
   timings ("block i/N: centroids + SVD init (rank r)...", "done in Xs",
   "N rebuilt, M reused").
3. The stage-5 fit is RESUMABLE per block: `fit_blk*.pt` were already written
   per block, but nothing recorded them -> a restart re-fitted from block 0.
   Now `fit_partial.json` ({sig, mse}) is updated atomically after every
   finished block; a restart with the same fit_sig reuses finished blocks
   ("fit resume: K/N blocks already fitted...") and fits only the rest.
   On success mse.json + fit_meta.json are written and the partial file is
   removed.
4. ATOMIC saves everywhere in the cache: pairs_blk (tmp + os.replace),
   init_blk, fit_blk, art_meta, mse.json, fit_meta.json, router_meta.json -
   a kill mid-save can no longer leave a torn file that poisons the next run
   (fit_blocks_ok already tolerated 0-byte leftovers; now they also cannot
   appear).
5. Base log-prob cache is RESUMABLE per chunk: chunk starts are seed-fixed,
   so existing loadable `lp_XXX.pt` chunks are skipped on a re-run; a sha1
   fingerprint (eval tokens + ctx/chunks/seed) in `lp_base/cache_sig.json`
   invalidates the cache when the eval dataset changes.
6. Adopting orphaned pools: a pre-.3 interrupted run left a COMPLETE pair
   pool without art_meta.json. Stage 4 now adopts a full consecutive set of
   `pairs_blk*.pt` (cap verified on block 0) and writes the missing meta
   instead of re-collecting.
7. `ensure_prereqs` distinguishes pool-complete from init-complete: an
   interrupted cache with missing init files auto-adds the `calibrate` stage
   ("the pair pool is reused, only the missing init files are rebuilt")
   instead of failing the fit or dying inside it.
8. Fixed two pre-existing crashes on resumed states: verify used X/Y that
   were only loaded when the WHOLE cache was complete (UnboundLocalError);
   the fit stage had no `pairs` list when the base pass re-ran while the pool
   was cached (UnboundLocalError).

Tests: test_update10_speed.py 11/11, test_update9_router_guard.py 11/11
(no regressions). Kill-sim matrix on tiny.gguf (all PIPELINE FINISHED, no
re-collection/re-fit of finished work): fresh run; kill in the init loop
(art_meta missing -> orphan adoption); partial inits (per-block rebuild);
kill mid-fit (per-block resume, sequential and --fit-workers 2); partial lp
chunks (only missing recomputed); complete cache (everything skipped).

Upgrade: unpack over the old folder; no flags change. A cache from an
interrupted pre-.3 run is repaired automatically on the next run (orphaned
pairs are adopted, missing inits are rebuilt, missing fit blocks are fitted).

## 2026-09-05.1 - LOW-RAM FIX (the "stops after caching / restarts from zero" bug)

Report: on an 8 GB box (3.2 GB free) with `--io-cache ram` and a 2.75 GB Q8_0
GGUF, stage 4 died with `numpy._core._exceptions._ArrayMemoryError: Unable to
allocate 128. MiB for an array with shape (64, 512, 1024)`; every restart
printed "cached pool holds 49152 pairs/block < requested cap 65536 -
recalibrating with the bigger pool" and died again - from the outside it
looked like "nothing is saved, it starts from zero every time".

Root causes (three, stacked):
1. `--io-cache ram` copies the whole packed GGUF into RAM (2.75 GB) while the
   backbone (0.54 GB), activations and dequant scratch also live there - on
   3.2 GB free that cannot fit; the old code then hit a hard MemoryError.
2. The dequant scratch was wasteful: `dequantize(full tensor) + astype()`
   allocated 2-3 full-size fp32 intermediates (2-3 x 128 MiB per expert
   tensor); the old threaded split also refused to fire below 4M elems/slab.
3. The recalibration loop: a cached pool SMALLER than the requested cap
   forced a full re-collection (the most expensive stage), which OOM-crashed
   before writing the bigger pool -> the same crash on every restart.

Fixes:
- `hf_gguf_to_hf.py`: `convert_tensor` fills a preallocated fp32 output
  slab-by-slab along the slowest GGUF axis (peak = output + ~4 MB slab,
  bit-identical output - verified against the old implementation on the exact
  crash shape, on 2D/3D Q8_0 and F16, serial and threaded); threaded mode now
  writes slab results into the output directly (no concatenate+astype copies).
- `hf_stream.py`: a MemoryError during a block load is no longer fatal -
  `BlockStreamRunner._degrade_ram_cache()` drops the packed-GGUF RAM copy
  (+2-3 GB), disables prefetch, gc-collects and retries the block from
  disk/mmap with a clear notice. `GgufHfSource.drop_ram_cache()` added.
- `hf_pipeline.py`: a cached pool smaller than the requested cap is KEPT by
  default when still usable (>= max(4096, 16*rank, cap/2) pairs/block) with a
  one-line notice; `--pool-recalibrate` forces the old re-collection. The
  "cached pool holds ..." notice prints once per run instead of 3-6x.
  Orphan adoption accepts usable smaller pools too.

Tests: new `test_lowram_fix.py` 34/34 (bit-exactness + RSS peak 133 vs 259
MiB on the crash shape; keep-smaller semantics incl. the exact 49152-vs-65536
case; fake-runner OOM degrade; drop_ram_cache; ensure_prereqs end-to-end).
Regression: test_update10_speed.py 11/11, test_update9_router_guard.py 11/11.
Live checks on tiny.gguf: full fresh run PIPELINE FINISHED; re-run with a
bigger cap keeps the pool and finishes in 11 s (no re-collection);
`--pool-recalibrate` re-collects; a MemoryError injected into the REAL
streaming runner is absorbed (notice + continue from disk).

Upgrade: unpack over the old folder; no flags change. Recommended on boxes
with <6-8 GB free RAM: pass `--io-cache disk` explicitly (the auto/profile
ram cache will now degrade gracefully anyway, but disk is predictable).

## 2026-09-05.2 - SWAP-STORM GUARD (the refine round freezes at "13 block loads")

Report: on the same 8 GB box the whole fit (stage 5) completed - all 23 blocks
fitted, guards passed - but the refine round (5b.1) froze right after
"experts from disk: 13 block loads, 1 GB read" with no error and no further
output. Killing and restarting reproduced it at the same place.

Root cause: TWO memory consumers stack up during the refine capture pass, and
Windows responds to exhaustion with a swap-storm instead of a MemoryError -
allocations keep succeeding through the pagefile, so the run does not crash,
it just stops making visible progress (wall-clock per operation explodes):
1. the capture hooks keep per-block pair chunks of up to 8192 pairs in RAM
   for ALL blocks at once (~1.1 GB resident for 23x(8192+4096)x1024 bf16
   x2 sides) - they were flushed to disk only after crossing the threshold;
2. the io-cache ram packed copy (~2.7 GB) grows block by block in the same
   pass (0.54 GB backbone + capture chunks + torch runtime on top) - the
   ceiling hits around block 13 of the first window, exactly the freeze
   point. Stage 4's hard crash could not happen here: the biggest single
   allocation in the capture pass is small, so nothing ever gets REFUSED.

Fixes:
- `hf_pipeline.py`: the capture flush threshold adapts to free RAM -
  `refine_flush_at()` - 1024 pairs below 8 GB free (resident ~= one batch,
  ~0.4 GB) vs the old fixed 8192 (~1.1 GB); a notice prints when the
  low-RAM threshold is active. The capture pass now also prints a
  per-window progress line (`... refine capture: window 3/12, pairs
  12288..16384 of 65536`) so "working" and "stuck" are distinguishable.
- `hf_stream.py`: `BlockStreamRunner` re-checks `io-cache ram` at EVERY
  streaming stage (`ram_cache_fits()`: free RAM must cover 1.5x the packed
  GGUF + 1 GB headroom) and downgrades to disk with a clear notice when it
  does not fit - the free RAM measured at stage 0 says nothing about stage
  5b. `MOE_FORCE_IO_RAM=1` restores the old "explicit means explicit".
- `hf_stream.py`: a RAM watchdog daemon thread (2 s sampling, stdlib-only:
  psutil -> /proc/meminfo -> GlobalMemoryStatusEx) drops the io-cache ram
  copy the moment free RAM crosses 0.6 GB - BEFORE the swap-storm starts -
  and continues from disk/mmap with a notice. `MOE_NO_RAM_WATCHDOG=1` off.

Tests: `test_lowram_fix.py` extended to 50/50 (ram_cache_fits boundaries incl.
the user's exact 3.2-vs-2.75 case; runner-level downgrade + override;
watchdog fires and clears the process-wide raw cache; refine_flush_at).
Regression: test_update10_speed.py 11/11, test_update9_router_guard.py
11/11. Live e2e on tiny.gguf with a refine round: PIPELINE FINISHED, the
low-RAM flush branch and the capture progress line both exercised.
