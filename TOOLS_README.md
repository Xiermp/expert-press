# Router tools — diagnostics & calibration for field artifacts

Three focused tools that operate on a **field artifact** (the output of
`hf_pipeline.py`) and its base GGUF. They answer the two questions every
compressed MoE eventually raises — *where did the quality go?* and *can we
get some of it back without re-fitting?* — with data instead of guesswork.

```
                 ┌────────────────────┐
  base .gguf ───►│  router_audit.py   │  per-layer drift / load balance /
                 └─────────┬──────────┘  z-scramble CE  ->  JSON report
                           │
             where is the error, does the router still work?
                           │
        ┌──────────────────┼─────────────────────┐
        ▼                  ▼                     ▼
 sampling problem    structural, depth-     accounting only
        │            concentrated           │
        ▼                  ▼                     ▼
 temp_calibrate.py   router_ft.py          field_dims.py
 (ships with the     (gate-only KL fit,    (dims/bytes one-liner)
  main repo)         writes <art>_rft)
```

## The tools

| tool | one line | needs base? |
|---|---|---|
| **`router_audit.py`** | per-MoE-layer audit: top-k agreement, routing cosine, input drift, output relMSE; artifact load-balance/entropy stats; counterfactual z-scramble per layer | optional (`--base` enables the drift phase) |
| **`router_ft.py`** | nudges ONLY the gate weights to minimize KL(base‖field), anchored to the original router; saves `<artifact>_rft`, never overwrites, saves nothing if quality did not improve | required |
| **`field_dims.py`** | instant "how much onto how much": dims, per-block params, field mix, bytes; a ready report line | no |

All three are dependency-light: `transformers`, `torch`, `safetensors`
(the same environment as the main pipeline) and `hf_env`/`hf_gguf_to_hf`
from the repo root next to them.

## Quick start

```bash
# 0) what did we build (instant, no model load)
python3 field_dims.py --artifact results/field_xxx_r32

# 1) where is the error? (base .gguf is auto-converted once into results/gguf_hf/)
python3 router_audit.py --artifact results/field_xxx_r32 --base model.Q4_K_M.gguf

# 2) is there gate-calibration headroom? (measure without saving)
python3 router_ft.py --artifact results/field_xxx_r32 --base model.Q4_K_M.gguf --dry-run

# 3) apply it (only writes <artifact>_rft if KL/CE actually improved)
python3 router_ft.py --artifact results/field_xxx_r32 --base model.Q4_K_M.gguf

# 4) verify the result with the pipeline's verify-only plan
python3 hf_pipeline.py --stages verify --gguf model.Q4_K_M.gguf \
    --out results/field_xxx_r32_rft --rank 32
```

Reading the audit (full guide: [wiki/Router-Diagnostics](../wiki/Router-Diagnostics.md)):

- phase 1 table — `topk agree`/`z-cos` near 1 = routing intact; the
  `out relMSE` column is the error-by-depth profile (last-layer outliers =
  compounding suspects);
- phase 2 — `load-balance` collapse in a layer = the field there uses fewer
  experts than the base did;
- phase 3 — a layer with a tiny CE jump under z-scramble has its routing
  effectively ignored (or already broken).

## How this fits the quality workflow

1. generation drifts into an archaic register → **greedy test** first
   (`/temp 0` in `hf_chat.py`): clean greedy = a sampling problem →
   `temp_calibrate.py` + `--min-p` (both in the main repo), no router
   surgery needed;
2. greedy still drifts → **audit**: depth-concentrated relMSE + big
   scramble jumps = structural; try `--refine-rounds 1-2` in the pipeline
   (self-distillation) and/or `router_ft.py`, re-audit, compare JSONs;
3. `field_dims.py` before/after any rank change to keep the report line
   honest.

## Files

```
router_tools/
├── router_audit.py     # audit (3 phases) -> results/router_audit_<tag>.json
├── router_ft.py        # gate-only calibration -> <artifact>_rft
├── field_dims.py       # accounting one-liner
└── TOOLS_README.md     # this file
```

Both auditing tools write JSON with every table they print — diff two audits
(before/after `router_ft`, before/after `--refine-rounds`) to see exactly
what moved.
