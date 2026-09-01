# Quality and calibration: what we learned about drift

## The metric protocol

All quality numbers are measured the same way everywhere (pipeline, evals,
audits):

- **KL(base ‖ variant)** in bits/token on held-out windows — the
  distribution-shift measure; 0 = identical histograms;
- **Δppl** — perplexity change vs the same base;
- the base is **the quantized model inside the GGUF** (per-block dequant,
  bit-identical values) — never a "similar" model;
- calibration and eval text **do not overlap** (the leak fix: earlier
  activation-aware methods were fitted on the very segment they were scored
  on, flattering KL by ~10-20%; weight-space methods reproduced byte for
  byte — see [Research-History](Research-History.md)).

Reference numbers (mini-PoC, same procedure): field r=32 → KL 0.029
bits/token, Δppl +2.5% at x5.6 expert compression; fp16 deploy is free
(+0.000 KL). The guard: a fit must beat the centroid baseline or the run
fails (`--skip-fit-guard` to disable; `test_field_fit_guard.py`).

## The style-drift finding (real models)

Symptom on a compressed OLMoE artifact: the first chat turns are fine, then
the model drifts into a "neighbour mode" — archaic / poetry register, verse
line breaks ("I pray thee..."). Diagnosis from the user's chat sample:

- turn 1 OK, turn 2+ drifts — the error **compounds over the dialog**, it is
  not a one-shot artifact;
- compression flattens the distribution (measured KL ~0.76 bits/token on
  that artifact); a flattened histogram **samples long-tail tokens too
  often**, and archaic/poetic tokens live exactly in the long tail;
- per-token KL looked acceptable, so the problem is not "the model is
  broken" but "the histogram is wider".

## The decision tree (cheapest first)

1. **Greedy test** — `/temp 0` in `hf_chat.py`. Clean greedy output → the
   weights are fine, it is a *sampling* problem → go to 2. Greedy still
   drifts → structural → go to 4.
2. **Temperature calibration** — `temp_calibrate.py` fits ONE scalar T that
   minimizes `KL(base ‖ softmax(field/T))` on seeded eval windows (the base
   is streamed from the GGUF, never fully loaded). It writes
   `sampling.json` next to the artifact and `hf_chat.py` picks it up
   automatically. Expect T < 1 for a visibly compressed artifact: sharpening
   undoes the flattening. The tool also prints top-1 agreement — a
   *structural* number (temperature-invariant): ~99% on a near-identity
   artifact; much lower on a heavy one.
3. **min-p sampling** — `--min-p 0.1` (or `/minp 0.1`): cuts the tail
   probability mass outright; composes well with the fitted T.
4. **Structural fixes**: audit the router
   ([Router-Diagnostics](Router-Diagnostics.md)) to find WHERE the error
   lives (depth profile, routing agreement); then `--refine-rounds 1-2`
   (self-distillation: the field feeds its own outputs forward while the
   original GGUF experts provide targets — fixes the compounding error the
   first fit cannot see), a bigger calibration pool (`--per-layer-cap`),
   higher rank, or gate calibration (`router_ft.py`).

The pipeline's `worst blocks by final mse` line points at the suspects:
outlier blocks shift the residual stream for every layer after them —
exactly the compounding pattern.

## Why temperature is the right first knob

Sampling temperature scales logits BEFORE softmax; the drift is an
over-dispersed softmax. A single scalar cannot fix routing errors, but it
directly reverses the flatten-compress-sampler loop, costs one short
calibration pass, is saved into the artifact folder, and — importantly —
the greedy test tells you *before* any tuning whether sampling is even the
problem. `temp_calibrate.py` on a near-identity artifact correctly reports
T = 1.0 (KL 0.000, top-1 99.2%), so it never makes a good artifact worse.

## Anti-loop defaults in the chat

Compressed models loop under plain greedy, so `hf_chat.py` ships with
repetition penalty 1.15 (`--repetition-penalty`, `/rep`; 1.0 = off) applied
to both temperature modes. If loops persist at `/rep 1.0`, that is a
quality signal — follow the decision tree above, don't crank the penalty
past ~1.3 (it starts eating repeated words the model actually means).
