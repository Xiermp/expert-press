# Field Engine — project wiki

**What this is:** a compression method for MoE language models in which the
explicit expert weights are **not stored at all**. An expert is assembled on
the fly from a low-rank "field" whose coordinates are computed from the
router's own soft weights: `W(z) = W1d + U · diag(c(z)) · Vᵀ`, `c(z) = z @ C`.
The result is a **normal HF model** (`AutoModelForCausalLM` just works) that
is several times smaller than even the Q4 GGUF it was built from.

**Status:** works end-to-end on real models (OLMoE-1B-7B, HunYuan/NanoColibri
hy_v3) fully streaming — the full model never loads into RAM. Compression on
OLMoE: **~12.9 GB of experts → ~0.21 GB of field** (x~60), artifact ~1.2 GB —
smaller than the source Q4 GGUF (4.4 GB).

## Pages

| page | what is in it |
|---|---|
| [Field-Engine](Field-Engine.md) | the method: formula, what is stored, why the router seed works, accounting |
| [Pipeline-and-Stages](Pipeline-and-Stages.md) | the 9-stage auto pipeline, **stage toggles** (`--stages` / `--skip`), caches |
| [Memory-and-Speed](Memory-and-Speed.md) | streaming, `--io-cache ram`, threads/prefetch/workers, disk layout |
| [Quality-and-Calibration](Quality-and-Calibration.md) | the KL protocol, style-drift findings, temperature calibration, min-p, refine |
| [Router-Diagnostics](Router-Diagnostics.md) | `router_audit.py` (3 phases), `router_ft.py` gate calibration, `field_dims.py` |
| [Research-History](Research-History.md) | the experiment ladder that led to the field, **with all charts** |
| [CLI-Reference](CLI-Reference.md) | every tool, every flag, every default — one page |

## Ten-minute tour

```bash
pip install -r requirements.txt
python3 hf_pipeline.py                # download Q4 GGUF -> compress -> verify
python3 hf_chat.py                    # chat with the artifact
python3 hf_pipeline.py --list-stages  # what the auto-pipeline can do
```

Then, depending on what you want:

- **a smaller/larger artifact** — `--rank 16|64` (the calibration pool is
  reused, only the fit re-runs);
- **only part of the pipeline** — `--stages fit,save,verify` or
  `--skip base,report` (see [Pipeline-and-Stages](Pipeline-and-Stages.md));
- **generation drifts into archaic style** —
  [Quality-and-Calibration](Quality-and-Calibration.md);
- **is the router still picking the right experts?** —
  [Router-Diagnostics](Router-Diagnostics.md);
- **how did we get here** — [Research-History](Research-History.md).

## The one-paragraph summary

A MoE layer stores N expert MLPs but uses only top-k of them per token. The
experts are heavily correlated — the router is effectively choosing a
*direction* in a continuous space of specialists. The field engine exploits
that: store one centroid per role (gate/up/down), a low-rank basis U, V and a
per-expert coordinate table C; the router's soft weights z pick the
coordinates, and the expert is reconstructed as a rank-r matrix. Nothing else
changes: same router, same top-k, same backbone — the 64 expert MLPs are gone.
The fit is a per-block MSE regression against (input → output) pairs captured
from the real model, all streaming, all cached on disk.
