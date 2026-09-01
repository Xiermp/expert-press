# Router diagnostics and gate calibration

The artifact keeps the ORIGINAL base router — that is a feature (zero router
quality loss by construction), but it also means the router now reads
**slightly shifted inputs**: as layers accumulate field error, the hidden
states drift, and the same gate maps them to somewhat different experts.
Two tools answer "is that happening, where, and how bad" with data instead
of guesswork, and a third fixes it surgically when it is.

## `router_audit.py` — the audit (read-only)

```bash
python3 router_audit.py --artifact results/field_xxx_r32 --base model.Q4_K_M.gguf
python3 router_audit.py --artifact results/field_xxx_r32          # artifact-only mode
python3 router_audit.py --artifact ... --base ... --windows 16    # tighter stats
```

`--base` accepts an HF dir or a `.gguf` (converted once into
`results/gguf_hf/`, reused afterwards). Three phases:

**Phase 1 — base vs field, per MoE layer (needs `--base`).** Both models run
on the same seeded token windows; per layer you get:

| column | meaning | healthy value |
|---|---|---|
| `topk agree` | Jaccard overlap of the chosen top-k experts | closer to 1 = routing intact |
| `z-cos` | cosine of the full routing-weight vectors | ~1 |
| `in-drift` | relative input drift into the block | grows with depth — that is expected; the question is how fast |
| `out relMSE` | block output error | the error profile by depth |
| `out cos` | output cosine | ~1 |

The script prints the mean/min agreement and the **depth trend**
(first-half vs second-half relMSE) — the "does error grow with depth?"
question answered numerically. Deep-layer outliers are the compounding
suspects.

**Phase 2 — artifact routing stats (always).** Expert load balance (1.0 =
uniform usage), top-1 score, logit entropy, top-1 margin per layer. A
collapsed `load-balance` layer means the field there effectively uses fewer
experts than the base did.

**Phase 3 — counterfactual z-scramble (always; `--no-scramble` to skip).**
Each layer's routing vector is shuffled among tokens one layer at a time;
the CE jump says whether the field actually *uses* that layer's routing. A
big jump = routing matters there; a ~0 jump = routing is ignored (or
already broken) in that layer. Also reported: the all-layers-scrambled CE
and the top-5 routing-sensitive layers.

Everything lands in `results/router_audit_<artifact>.json` (all tables,
machine-readable — the phase-1 arrays feed plots directly).

## Reading the results into action

- Phase-1 `out relMSE` concentrated in the last layers + phase-3 big jumps
  there → try `--refine-rounds 1-2` first (it specifically targets
  compounding error), re-run the audit, compare tables.
- Agreement high everywhere, drift small, but generation still drifts → the
  problem is sampling, not routing →
  [calibrate temperature](Quality-and-Calibration.md).
- Phase-2 load-balance collapse in a layer → that block's fit is suspect;
  consider a higher rank there (bump `--rank` and refit — the pool is
  reusable) or `router_ft.py`.

## `router_ft.py` — surgical gate calibration

```bash
python3 router_ft.py --artifact results/field_xxx_r32 --base model.Q4_K_M.gguf
python3 router_ft.py --artifact ... --base ... --dry-run      # measure only
python3 router_ft.py --artifact ... --base ... --steps 60 --lr 5e-5
```

What it does: loads the artifact in fp32 (gradients) and the base in bf16
(targets), then takes `--steps` gentle Adam steps on **the gate weights
only** — everything else is frozen. The loss is `KL(base ‖ field)` plus an
`--anchor` · ‖W−W₀‖² penalty that keeps the router near the original
(anchor 1.0 default; 0 = free router). Before/after CE and KL are printed
on held-out windows, the gate drift per layer is reported, and:

- if quality improved → a NEW artifact dir `<artifact>_rft` is written (the
  original is never touched; the `router_ft` provenance is recorded in the
  artifact's `field_meta.json`);
- if not → **nothing is saved** (`--force-save` overrides for experiments).

Verify the result with the pipeline's verify-only plan:

```bash
python3 hf_pipeline.py --stages verify --rank 32 --out results/field_xxx_r32_rft
```

## `field_dims.py` — the accounting one-liner

```bash
python3 field_dims.py --artifact results/field_xxx_r32
```

Prints "how much onto how much" for any artifact: d_model × d_ff × N →
rank r, per-block and total params/bytes, the field mix (centroid % / U,V %
/ coordinates %), on-disk size, and a ready report line. Reads only
`config.json` + `field_meta.json` + safetensors sizes — instant, no model
load.

## The three tools together

A practical quality loop for a compressed artifact:

```bash
python3 field_dims.py   --artifact art            # what did we build
python3 router_audit.py --artifact art --base gguf # where is the error
python3 temp_calibrate.py --model art --gguf gguf && python3 hf_chat.py      # sampling fix
python3 router_ft.py    --artifact art --base gguf --dry-run   # is there gate headroom
python3 hf_pipeline.py --stages verify --gguf gguf --out art  # final numbers
```
