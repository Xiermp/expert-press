# Router & Quality Tools (add-on for the field pipeline)

Three standalone scripts. They do NOT modify any pipeline file — drop them into
the same folder as `hf_pipeline.py` and run with your existing Python.

```
router_audit.py   diagnose: is the router confusing experts? where does the error live?
router_ft.py      (optional) re-align gate weights to the field model, saves a NEW artifact
field_dims.py     print the compression dims note: d_model/d_ff/N -> rank r
```

---

## 1. router_audit.py — diagnosis (run this first)

**Q: "Is the router confusing experts?"** The artifact keeps the original base
router verbatim, so the router itself cannot "break". What CAN drift: the
hidden states the router consumes. As layers accumulate approximation error,
deeper routers see different inputs than in the base model -> different expert
selection -> compounding error. Phase 1 measures exactly that.

**Q: "Do errors grow with depth?"** The same table (out relMSE per layer).

```bat
:: full diagnosis (base gguf + artifact)
python router_audit.py --artifact results\field_NanoColibri-Instruct-GGUF_r128 --base hf_cache\hub\...\NanoColibri-Instruct.Q8_0.gguf

:: artifact only (stats + counterfactual scramble)
python router_audit.py --artifact results\field_NanoColibri-Instruct-GGUF_r128
```

`--base` accepts an HF checkpoint dir or a .gguf file (converted once to
`results/gguf_hf/`, reused afterwards). Use `--ctx 256 --windows 8` defaults;
add `--threads 4` to keep the machine responsive.

### How to read the output

Phase 1 (needs --base), per MoE layer:

| column | meaning | worry when |
|---|---|---|
| topk agree | share of tokens where base and field pick the same expert set (1.0 = identical routing) | < ~0.95 in mid/late layers = routing drift confirmed |
| z-cos | cosine of the routing score vectors | drops together with agreement |
| in-drift | rel. L2 between block inputs (base vs field) | grows with depth = compounding error |
| out relMSE | block output error vs base | THE depth profile: if late layers >> early layers, spend capacity there |
| out cos | output cosine (1.0 = same direction) | — |

Phase 2 (artifact only): `load-balance` (1.0 = all experts used equally,
low value = collapse onto few experts), `top1 score`, `logit entropy`,
`top1 margin` (small margin = router is unsure = "confused" router in the
classical sense).

Phase 3 (artifact only): shuffles the routing vector `z` inside one layer at a
time and measures the CE jump. Big jump = the field really uses that layer's
routing; ~0.0 = routing there is ignored (the layer behaves as a static layer).
`all layers scrambled` shows the total routing dependence of the artifact.

Results are also written to `results/router_audit_<artifact>.json`.

---

## 2. router_ft.py — the "подшаманить роутер" option (only if drift is real)

Fine-tunes ONLY the gate weights of the field model (everything else frozen)
against the base model: loss = KL(base||field) + anchor keeping gates close to
the original. Verdict is printed; **nothing is saved unless metrics improve**
(original artifact is never touched; output is `<artifact>_rft`).

```bat
:: measure the potential first (nothing saved)
python router_ft.py --artifact results\field_..._r128 --base path\to\base.gguf --dry-run

:: real run
python router_ft.py --artifact results\field_..._r128 --base path\to\base.gguf --steps 60 --lr 3e-5
```

Notes:
- a linear remap of router outputs is already expressible through the field's
  own C/w1d parameters, so a gain here comes only from RE-ALIGNING routing
  decisions under drifted hidden states — if phase 1 showed agreement ~1.0,
  expect little (and that is a valid result: the router is NOT the problem);
- the artifact is loaded in fp32 for clean gradients; the base model runs in
  bf16 as the KL target;
- RAM: base + artifact + one window of activations (a few GB on nano models).

---

## 3. field_dims.py — the compression note ("сколько на сколько")

```bat
python field_dims.py --artifact results\field_NanoColibri-Instruct-GGUF_r128
```

Prints, e.g.:

```
dims     : d_model=768, d_ff=3072, experts N=64 (top-8) -> field rank r=128, layers=24
1 block  : experts 56.62M params -> field 1.85M params (30.6x)
all 24 layers: experts 4631 MB -> field 94 MB (x49.2, fp16 accounting)
```

plus a ready-to-paste `report line:`. To bake it into future pipeline runs,
add this right after the "Сжатие экспертов" table in `hf_pipeline.py`'s
`write_report()`:

```python
g = geoms[0]
d, dff, N, r = g["d_model"], g["d_ff"], g["n_exp"], args.rank
exp_p, field_p = N * 3 * dff * d, 3 * dff * d + r * (3 * dff + 2 * d) + 2 * N * r
md.append(f"dims: d_model={d} x d_ff={dff}, {N} experts -> rank r={r} "
          f"(per MoE block {exp_p / 1e6:.1f}M -> {field_p / 1e6:.2f}M params, "
          f"x{exp_p / field_p:.1f}; experts {T['full_experts_mb']:.0f} MB -> "
          f"field {T['field_mb']:.0f} MB, x{T['ratio']:.1f})")
```
