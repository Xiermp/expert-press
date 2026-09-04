Update 9 (2026-09-04.1) - guard rework + the router joins the rebuild
Response to the report from 2026-09-04: (1) the guard on Adam was too aggressive—it was bailing out right at the start before the optimizer had time to adapt; (2) router replacement yielded mediocre results and caused a memory leak, as the old and new routers existed simultaneously and the mechanism was unrefined; (3) hypothesis regarding the original activation function; (4) idea to include the original router in the expert rebuild to reduce discontinuities post-conversion.

Everything was first validated on a toy benchmark setup (a synthetic MoE block with a realistic delta spectrum, a ACTUAL FieldSparseMoe, and a replica of the fit cycle): scripts/bench_toy_router_guard.py. Results are available at download/toy_router_guard_results.csv and download/toy_router_guard_chart.png.

Changes Made
Fit Guard (hf_field_transform.py, hf_pipeline.py):

--fit-guard-warmup (default: auto = max(30, steps//10)): the 2x bail-out condition ("loss diverged ... stopping this block early") is armed only after reaching this step. Previously, it triggered during Adam's normal early overshoot at steps ~1–12, cutting the fit before adaptation took place. Setting this to 0 restores the old behavior.

Soft end-guard: fits finishing in the (0.98..1.0]x range of the centroid baseline now issue a WARNING and emit the best state (the run continues). The old hard abort behavior remains optional via --strict-fit-guard. Anything worse than the baseline raises an error in both modes.

Baseline and post-restore reviews now use an 8-batch average (single-batch figures were fluctuating by ~10% and causing false alarms).

--fit-lr-warmup N — linear learning rate warmup: turned failing fits into functional ones on the benchmark setup (-3.9%..-11.7% below baseline). Recommended setting: 30..50.

Original Router in the Rebuild (--fit-router {off,after,joint}):

In-place processing on disk-based pairs; does not require the model in RAM, and the artifact size remains unchanged (the tuned gw replaces gate.weight with the exact same shape).

after = short anchored polish post-fit (steps/lr/anchor configurable via --router-steps 80 --router-lr --router-anchor 0.03).

joint = the router optimizes alongside the field starting from step 0.

The tuned router is written to fit files (gw_tuned) and into the artifact, field_meta.json -> router_polish, and stats -> router_meta.json.

HONEST EXPECTATIONS (benchmarks): once the fit converges, the router is usually NOT the bottleneck—the z-dependent rank correction carries a small fraction of the output energy, making gw gradients tiny (even shifting 30% of the top-4 gates did not move MSE, including on a "wrinkled" router or via softmax temperature). While this tool is safe (anchor + best-state + rollback to original), it serves primarily as a diagnostic. The main lever against discontinuities remains --refine-rounds.

router_ft.py Memory Optimizations:

The artifact is now loaded in bf16 (previously fp32, incurring 2x RAM overhead).

ONLY the fp32 gate master weights are trained (GateMaster via torch.func.functional_call — router math remains bit-exact to the original).

The remainder of the model is frozen (previously, requires_grad was set across all weights, causing backward passes to construct gradients for the full model). Result: ~2.8x reduction in RAM usage.

Added --inplace flag to modify gate weights directly within the artifact (by default, a separate _rft copy is saved, matching previous behavior).

Activation Function Hypothesis — Unconfirmed:

Introducing learnable gamma/temperature parameters prior to SiLU provided no measurable benefit.

Replacing SiLU with GELU degraded performance (+48% MSE): the field activator MUST MATCH the base model's activator. Errors stem from rank limitations and input shifts (addressed by refine rounds), not from the activation function.

Verification
scripts/test_update9_router_guard.py: 11/11 tests passing (covers old/new guard semantics, rescue lr-warmup, joint-router, polishing, bit-exact GateMaster, gw_tuned within artifacts and un-tuned paths, CLI behavior).

End-to-end smoke test on a tiny GGUF using --fit-router after --fit-lr-warmup 10: PIPELINE FINISHED, KL ~0, reload confirmed identical, router_meta.json successfully written (14 seconds on 2 CPU cores).

Pre/post SHA256 checksums: see worklog Task 21.

Quick Start with Update 9

python3 hf_pipeline.py                 # Standard execution; soft guard active by default
python3 hf_pipeline.py --fit-lr-warmup 40  # Use if blocks stall or diverge
python3 hf_pipeline.py --fit-router after  # Includes router polishing (diagnostic mode)
python3 router_ft.py --artifact ... --base ... --inplace   # Low-RAM LM polishing
