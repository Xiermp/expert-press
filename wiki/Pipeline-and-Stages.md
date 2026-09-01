# The auto-pipeline and its stage toggles

`hf_pipeline.py` runs the whole compression as one command, but internally it
is a chain of **9 independent stages**. Since the pools/fits/metrics are all
cached on disk, any stage can be switched off — either because it is already
done, or because you deliberately want only part of the flow.

## The stages

| # | stage | what it does | leaves on disk |
|---|---|---|---|
| 1 | `download` | resolve/download the source GGUF + build the light catalog (config+tokenizer); `--full-dequant` builds the old-style 14 GB checkpoint instead | GGUF (~4.4 GB) + catalog |
| 2 | `texts` | calibration/eval text, leak-free 90/10 split, tokenization | — (in memory) |
| 3 | `base` | base ppl / log-prob cache / demo generation — STREAMING | `cache_*/lp_base/`, `eval_tokens.pt` |
| 4 | `calibrate` | (input → output) pair pool + per-block centroids/geometry — STREAMING | `cache_*/pairs_blk*.pt`, `init_blk*.pt`, `art_meta.json` |
| 5 | `fit` | the field fit per block, model NOT in RAM; per-rank cached | `cache_*/fit_r32/` |
| 5b | `refine` | self-distillation refit rounds (opt-in, `--refine-rounds`) | updates the fit dir |
| 6 | `save` | assemble the artifact streamingly | `field_*_r32/` |
| 7 | `verify` | reload the artifact as a normal model: KL/Δppl + demo generation | — |
| 8 | `report` | write reports (artifact `README.md` + `results/moe_hf_pipeline_report.md`) | reports |

Two cache facts drive everything:

- the **pair pool** (stage 4) depends on neither rank nor fit settings — it
  serves ANY rank without recalibration;
- a specific **fit** (stage 5) is cached per `fit_meta.json` signature
  (steps/bs/lr/method/jitter/early-stop/preset) — same settings → skip.

## The toggles

```bash
python3 hf_pipeline.py --list-stages              # the stage table + toggle help
python3 hf_pipeline.py --stages fit,save,verify   # run ONLY these stages
python3 hf_pipeline.py --skip base,verify,report  # run all EXCEPT these
python3 hf_pipeline.py --skip download            # reuse-only: never touch the network
python3 hf_pipeline.py --stages fit --rank 64     # a new rank from the cached pool
python3 hf_pipeline.py --stages refine --refine-rounds 2
python3 hf_pipeline.py --gen-tokens 0             # no demo generations at all
python3 hf_pipeline.py --no-cache-verify          # skip the 2-chunk cache self-check
```

Rules the planner follows (printed as `PLAN: ...` before the run):

- `--stages` and `--skip` are mutually exclusive; names are validated against
  the table above (a typo fails fast, not mid-run).
- **Cheap missing stages are auto-added** with a notice: `download` (it is
  nearly instant when the catalog exists) and `texts` (pure tokenization)
  when the plan's later stages need them.
- **Expensive missing prerequisites fail fast with a hint.** Examples:
  - `fit` without a pair pool → *"run the full pipeline once ... or lower
    --per-layer-cap (the cached pool holds N pairs/block)"*;
  - `verify` without the log-prob cache → *"keep 'base' in the plan (one
    streaming pass builds the cache), or --skip verify"*;
  - `save`/`refine` without fits → *"keep 'fit' in the plan, or finish a full
    run first"*.
- `refine` is opt-in: it appears in the plan only when you list it in
  `--stages` (which implies ≥1 round) or set `--refine-rounds > 0`.
- `--skip-reload-check` keeps working — it is exactly `--skip verify`.
- A re-run auto-skips whatever is cached (pool → stages 3-4, fits → stage 5);
  the toggles are for going *further* than the cache shortcuts.

`--skip download` (reuse-only) never hits the network: a hub GGUF must
already be in the local HF cache, the light catalog must exist, and a
`--full-dequant` checkpoint must be built — otherwise you get a precise
error telling you which stage to re-enable once.

## Typical scenarios

**A new rank from an existing calibration** (the pool is rank-independent):

```bash
python3 hf_pipeline.py --stages fit,save,verify --rank 16
```

**Re-verify an artifact after external tinkering** (e.g. after
`router_ft.py` produced `*_rft`):

```bash
python3 hf_pipeline.py --stages verify --rank 32 --out results/field_xxx_r32_rft
```

**Night fit, morning verify, no report spam:**

```bash
python3 hf_pipeline.py --skip verify,report          # evening: base+calib+fit+save
python3 hf_pipeline.py --stages verify               # morning: metrics only
```

**Experiment with fit settings, touching nothing else:**

```bash
python3 hf_pipeline.py --stages fit --fit-preset quality --rank 32
```

**Report only** (rebuild `results/moe_hf_pipeline_report.md` from the run
cache without loading anything heavy):

```bash
python3 hf_pipeline.py --stages report --rank 32
```

## What the plan looks like

```
PLAN: download -> texts -> base -> calibrate -> fit -> save -> verify -> report
skipped: refine
...
== STAGE 2 - texts: skipped (--skip texts; token ids come from the run cache)
== PHASE A - no streaming pass needed: everything from the run cache
  block 0: 512 pairs (from cache)
...
```

The run's plan (and what was skipped) is recorded in the report header
(`Stages run: ...`), so a report is never ambiguous about how it was
produced. Fit metrics loaded from disk (`--skip fit`) show up as
`fit mse` in the worst-block line only if `mse.json` was present; reports
clearly mark sections that were skipped (`--skip verify` →
"metrics skipped in this run").
