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

----

# 2026-09-05.3 - FIELD RUNTIME FIX (hy_v3): build 10.5

Files changed: `modeling_field_template.py` (version stamp 2026-09-05.3),
new `test_template_hy_v3.py` (18 checks), new
`modeling_field_HYV3_ready.py` (pre-rendered drop-in), README bullet.

## Context
User report (evening 2026-09-05): the first full NanoColibri (hy_v3) run
finished Stage 5 refine + Stage 6 save, then Stage 7 crashed:
`TypeError: HYV3TopKRouter.forward() missing 1 required positional argument:
'e_score_correction_bias'` inside the artifact's `modeling_field.py`, with
`shared_experts.*` and `e_score_correction_bias` reported UNEXPECTED in the
load report. Root cause: the artifact template was written and A/B-tested on
the OLMoE toy only. It never grew the hy_v3 branches that the fit side
(`hf_field_transform.FieldSparseMoe`) has had all along:

1. `self.gate(x)` - since transformers 5.16 the hy_v3 router takes the
   selection bias as a forward argument (`forward(x, e_score_correction_bias)`,
   the router no longer owns it); the template call crashed before any return
   handling. The fit side already called `gate(x, eb)` - Stage 5 numbers were
   computed with the bias, the artifact could never reproduce them.
2. No `e_score_correction_bias` buffer and no `shared_experts` module in the
   artifact class - Stage 6 (correctly) copies both from the backbone into
   the artifact, so they loaded as UNEXPECTED dead weight, and the artifact
   silently LOST the always-on shared-expert branch (the fit target is
   "block output minus shared", so the field must be summed WITH shared) and
   routed without the bias.
3. Latent `NameError`: the template's `forward` referenced the `fi` local
   from `__init__` in the non-tuple router branch (any Linear-gate base
   model would crash there; never triggered on the toy).

## What was done (toy/unit stand FIRST, then the package)
- `modeling_field_template.py`: for `router_kind == "sigmoid_bias"` (or when
  the host router's signature takes the bias - sniffed at runtime, with a
  `TypeError` fallback call for older hosts) the class registers the
  `e_score_correction_bias` buffer (filled from the checkpoint) and passes it
  to the router; for `dff_shexp > 0` it builds a `shared_experts` module with
  the SAME key layout as the base model (`gate_proj/up_proj/down_proj`,
  bias-free) and adds its output to the field output in fp32 - exactly the
  base model's `enable_moe_fp32_combine` semantics and exactly what the fit
  measured. Top-k weights from v5 routers are taken as returned (already
  normalized and scaled inside the router - same contract as the fit side).
  The `fi` NameError is fixed (top_k/norm_topk stored as attributes).
- `modeling_field_HYV3_ready.py`: the template pre-rendered for
  HYV3ForCausalLM/HYV3TopKRouter - a drop-in replacement for the modeling
  file of an EXISTING artifact, so the finished r128 run does NOT need the
  hours-long refine re-run; `--stages verify` then re-checks the artifact.

## Tests (new `test_template_hy_v3.py`, 18/18)
- real HYV3TopKRouter: buffer + shared_experts registered; forward finite;
  perturbing the bias changes expert selection (buffer is actually used).
- shared branch == base HYV3MLP (same weights -> same output).
- artifact math == fit-side `FieldSparseMoe` on identical params
  (max|d| = 3e-08).
- mini-e2e on a REAL HYV3ForCausalLM (2 layers, 8 experts, 1 shared):
  save_pretrained + modeling_field.py -> from_pretrained(trust_remote_code)
  -> loading_info missing/unexpected EMPTY, logits identical after reload.
- OLMoE regression: no bias/shared on the softmax path; tuple-router ok;
  Linear-gate (the old NameError branch) ok; norm_topk ok.
- Live e2e tiny.gguf --smoke --refine-rounds 1: PIPELINE FINISHED 11.6 s,
  artifact reload verified (KL -0.001 bits, +0.0%). Regression suites:
  test_lowram_fix.py 50/50, test_update9_router_guard.py 11/11,
  test_update10_speed.py 11/11.

## How to apply (finished artifacts, no refit needed)
- A (30 s): overwrite the artifact's modeling file with the pre-rendered
  `modeling_field_HYV3_ready.py`, then `--stages verify`.
- B (minutes): re-run the same command with `--refine-rounds 0` - the fits
  are cached (fit_sig unchanged), Stage 6 rebuilds the artifact with the new
  template, Stage 7 verifies. Without `--refine-rounds 0` the refine round
  repeats (it has no skip marker - its pairs are deliberately rebuilt).

----

# 2026-09-05.4 - DTYPE-MISMATCH FIX + REFINE RESUME/PARALLEL REFIT: build 10.6

Files changed: `modeling_field_template.py` + regenerated
`modeling_field_HYV3_ready.py` (version stamp 2026-09-05.4),
`hf_pipeline.py` (2026-09-05.4: refine resume + parallel refit),
new `test_dtype_mismatch.py` (10 checks), README bullets.

## Context
User report: after installing 10.5, Stage 7 got past the router fix and
crashed deeper: `RuntimeError: expected m1 and m2 to have the same dtype,
but got: float != struct c10::BFloat16` at `cgu, cdn = z @ self.Cgu` in the
artifact's `modeling_field.py`. Two independent causes:

1. v5 routers upcast INSIDE: `HYV3TopKRouter` returns fp32 logits/weights
   even in a bf16 model (probed on transformers 5.16.1: logits fp32, scores
   fp32). The field built `z = zeros_like(logits)` -> fp32, while the field
   params are bf16 in the artifact (94 MB = exactly bf16 for 47M params).
   torch ADDITIONS type-promote, matmuls do not -> crash.
2. A checkpoint loaded "as stored" (`dtype=None`) is MIXED: the backbone
   comes from the GGUF dequant (fp32/fp16 tensors) while the field was saved
   bf16 for size. Any external app loading without an explicit dtype would
   hit the same wall - the runtime must be dtype-robust on its own.

Same visit, two long-standing refine complaints:
- a repeated run re-streamed the WHOLE model for the refine capture even
  when the round had already completed (hours wasted after a Stage-7 crash);
- the refine refit loop ignored `--fit-workers` (always one worker, unlike
  the stage-5a fit).

## What was done
- `modeling_field_template.py`: new `_align_field_dtype` - on the first
  forward every floating tensor of the field module (field params, router
  bias buffer, shared_experts weights, gate) is aligned ONCE to the host's
  compute dtype (`hidden_states.dtype`), memoized in `_field_dtype`;
  `z` is cast to `x.dtype` after the scatter. Upcasts (bf16 field -> fp32
  host) are bit-exact pure casts; downcasts only happen when the whole host
  already runs in that dtype. No per-call temporaries, O(1) per forward.
- `hf_pipeline.py` (refine stage 5b):
  * each round is identified by a sha256 signature of its settings (rank,
    caps, calib, fit params, router params, version stamp);
  * a COMPLETED round is skipped on re-runs via `done_r{N}.json` markers
    (mse list restored from the marker); a HALF-DONE round reuses the
    captured pairs via `pairs_sig.json` (signature + chunk-file list) -
    the capture pass (the expensive half) no longer re-runs;
  * `--refresh-refine` forces a full redo (old behavior);
  * the refit loop is now a worker pool with the same pattern as stage 5a:
    `w = min(--fit-workers, n_blocks)`, thread split of CPU cores, errors
    collected and re-raised after join (the per-round failure semantics
    from 2026-09-05.2 are preserved).

## Verification (toy stand FIRST, then package)
- NEW `test_dtype_mismatch.py` (10 checks): probes the host router fp32
  behavior; `from_pretrained(dtype=bfloat16)` on a mini hy_v3 artifact
  (the exact Stage-7 path) survives; mixed checkpoint (backbone fp32 +
  field bf16) runs in fp32 with bit-exact param casts and deterministic
  repeats; bf16 host with an fp32 gate aligns down without a crash.
- `test_template_hy_v3.py` 18/18, `test_lowram_fix.py` 42/42,
  `test_update9_router_guard.py` 11/11, `test_update10_speed.py` 11/11.
- Toy e2e on tiny.gguf (`--smoke --refine-rounds 1`): run 1 PIPELINE
  FINISHED (KL -0.002 bits, +0.0%); run 2 skips the fit AND the refine
  round ("cached (resume)") with identical metrics; run 3 with
  `--refresh-refine --fit-workers 2` re-captures and refits through
  "refine refit: 2 workers x 1 cpu threads"; `--stages verify` alone
  prints BASE ppl (from the log-prob cache) + FIELD KL/ppl/% with no
  model, fit or refine.

----

# 2026-09-05.5 - SVD-INIT SELF-HEAL + STALE-REFINE GUARD: build 10.7

Files changed: `hf_pipeline.py` (2026-09-05.5). No template/runtime changes.

## Context
User report: "при прошлом запуске было falling back to the random init -
ану SVD для 128 ранга не подтянулся. Оптимизатор начинал вслепую." Reproduced
on the toy stand: the fit's SVD init (`init_svd_blk*.pt`, hole 1 from
UPDATE-10) is RANK-SPECIFIC and lives in `fit_r{R}/`; when they are missing,
stage 5 printed a small NOTE and fitted from the random init - which costs
~3x more steps for the same quality (Bench E) and, at 23 MoE blocks x r128,
is exactly the "optimizer starts blind" case. Two ways to get there:

1. the pair pool was built by a PRE-SVD build (init_blk*.pt cached, no SVD
   files): stage 4's per-block resume skipped everything forever;
2. an earlier run of the CURRENT build hit the fallback once (e.g. after a
   rank change done the wrong way): the blind fit was then CACHED in
   fit_meta.json with `init: "random"` and silently reused by every re-run.

Extra finding while reproducing: the refine-round cache (done_r*.json +
pairs_sig.json, added in 10.6) did NOT depend on the fit state - after any
re-fit the stale refine results were reused on top of the new fits (reproduced:
"refine round 1: cached (resume) - skipped" right above a brand-new fit).

## What was done (hf_pipeline.py)
- SVD-INIT SELF-HEAL: before phase A, if a fit is planned with
  `--fit-init svd` (the default) and `init_svd_blk*.pt` are missing for this
  rank while the pool cache exists, the pipeline says so ("SVD init check:
  ... missing for rank R (N blocks) ...") and forces the streaming pass that
  rebuilds them from the REAL expert deltas. Cost: one expert-weights read
  per block, NO model forward (centroids mgu/mdn come from the cached
  init_blk*.pt - no expert_means re-run). The stale random-init fit is
  re-fitted automatically (its signature changes from `random` to `svd`).
- stage-4 loop: the per-block resume now distinguishes "init cached, SVD
  missing" (heal: reuse cached centroids + build the SVD part) from the
  fully-cached skip; `init_svd_blk*.pt` are written ATOMICALLY (.tmp +
  os.replace - no torn files after a kill); blocks whose SVD file already
  exists are skipped -> an interrupted heal/refresh resumes per block.
- `--refresh-init` keeps the same skip-existing semantics (idempotent,
  resumable; delete fit_dir/init_svd_blk*.pt to force a full rebuild) and
  its help now says the heal is automatic.
- stage-5 fallback: the NOTE became a loud WARNING with the consequence
  spelled out (~3x more steps; UPDATE-10), the report JSON gains
  `fit_init_effective` ("svd"/"random"), and the STAGE-5 banner shows the
  effective init.
- STALE-REFINE GUARD: each refine round's signature now includes
  `fit_state` - the sha1 of `fit_meta.json`, which is rewritten exactly
  when the fit (re)runs and is never touched by refine itself. A re-fit
  therefore invalidates the refine rounds (the captured pairs are the
  FIELD's own forward outputs - they MUST be re-captured); a fully-cached
  re-run recomputes the same hash and keeps skipping as before.
- Base-metrics preservation: a heal pass runs with the base stage cached
  (no re-run of stage 3); the base ppl + eval tokens are loaded from the
  log-prob cache so the verify stage keeps its ppl delta.

## Verification (toy stand)
- Reproduced the user's failure on 10.6 first: deleted fit_r32, re-ran ->
  "NOTE ... falling back to the random init", fit_meta.json `init: "random"`,
  stale refine reused. On 10.7: "SVD init check" -> "cached init + SVD init"
  per block -> fit re-runs with `(adam, svd init, ...)` -> refine round
  RE-DONE (stale guard works) -> `fit_meta.json init: "svd"` -> verify green.
- Re-run after the heal: no SVD-init check, no streaming pass, fit and
  refine both "cached - skipped", identical metrics (idempotent).
- Fresh rank on an existing pool (`--rank 16`): heal builds fit_r16/
  init_svd_blk*.pt from the cached centroids, fit uses svd, verify green.
- `--refresh-init` with all files present: per-block "already exists -
  skipped", 0/N written, no wasteful re-streaming.
- `--stages verify`: NO heal, no streaming pass (fit not planned).
- Regressions: test_dtype_mismatch, test_template_hy_v3 (ALL CHECKS
  PASSED), test_lowram_fix 50/50, test_update9_router_guard 11/11,
  test_update10_speed 11/11, py_compile OK.

## For the user's r128 run
The existing r128 artifact was fitted blind (random init) - the fit AND the
refine rounds are tainted. One command fixes the state on 10.7:

    python hf_pipeline.py <your usual args>

The pipeline will: rebuild init_svd_blk*.pt for r128 (~minutes, one expert
read per block), re-fit all blocks from the SVD init (the expensive part,
unavoidable - the blind fit must be re-taught), redo the refine rounds on
top, rebuild the artifact and verify. If you only want to check losses
without re-fitting: `--stages verify` still works off the existing caches.
