Notice: This project is an active experimental sandbox. Due to rapid iterations and heavy architectural experimentation, code was developed with the assistance of AI collaborators.

# Field Engine: MoE model compression without storing expert weights

An expert is not stored - it is **assembled on the fly** as a matrix field:
`W(z) = W1d + U · diag(c(z)) · Vᵀ`, where the coordinates `c(z) = z @ C`
are computed from the router's soft weights. The explicit weights of the 64
experts are discarded; what is stored instead: a centroid + low-rank factors
U, V + coordinates C.

The goal: a **normal model** (opens via `AutoModelForCausalLM`) with experts
compressed into the field and quality measured by our protocol (KL bits/token,
Δppl, compression ratio). No fine-tuning of the base model is required or
included.

---

## Quick start (locally)

```bash
pip install -r requirements.txt          # or it auto-installs on first run
python3 hf_pipeline.py                   # the WHOLE pipeline in one command
python3 hf_chat.py                       # chat with the result
```

Zero-config mode:

```bash
python3 hf_pipeline.py --auto            # auto-quant + balanced fit preset
python3 hf_pipeline.py --auto --model mradermacher/NanoColibri-Instruct-GGUF
```

Windows: the same two commands via double click - `step1_compress.bat` and
`step2_chat.bat` (Python must be installed and on PATH).

What happens:
1. **A ready Q4 checkpoint is downloaded** - `OLMoE-1B-7B-0924.Q4_K_M.gguf`
   from `mradermacher/OLMoE-1B-7B-0924-GGUF` (~4.4 GB, resumes on
   interruption).
2. Weights: by default **no dequant checkpoint is created** - a light catalog
   (config + tokenizer) is built and weights are read STRAIGHT FROM THE GGUF:
   each block's experts are dequantized on the fly (bit-identical result: the
   same dequant functions as the full converter `hf_gguf_to_hf.py`).
   Saves ~14 GB of disk and SSD writes. Full dequant - the `--full-dequant`
   option (faster on a fast SSD, but +14 GB).
3. Base: perplexity, log-probs (cache **on disk**, per chunk), generation.
   **The full model never loads**: everything runs STREAMING
   (`hf_stream.py`) - only the backbone (~1.5 GB) lives in RAM, each block's
   experts are read from disk exactly for their layer's pass (~2-3 GB peak
   instead of ~15 GB), with the NEXT block being dequantized by a background
   prefetch thread while the current layer computes.
4. Calibration: (MoE block input -> MoE block output) pairs captured by
   hooks - also streamed to disk as the cap is reached. The pair pool is the
   complete calibration artifact: the fit samples vectors independently, so
   text/order do not matter, and the cache works for ANY rank.
5. The field fit **r=32** runs on pairs from disk - the model is NOT in RAM,
   stage peak ~1-2 GB. A specific rank's fit is cached: a re-run with the
   same settings skips it. Blocks are independent - they can be fitted in
   parallel with `--fit-workers`.
6. The artifact is assembled **STREAMINGLY from disk**: backbone tensors are
   copied one by one, experts skipped, the field added from fit files. The
   full model never loads during the whole run.
7. Verify: the artifact loads as a normal HF model (~1-2 GB RAM), KL/Δppl
   metrics against the on-disk base cache + demo generation from it.

The result lands in `results/field_OLMoE-1B-7B-0924-GGUF_r32/`:
`config.json`, weights (safetensors), `modeling_field.py`, a `README.md` with
metrics, plus the combined report `results/moe_hf_pipeline_report.md`.

Using the result:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("results/field_OLMoE-1B-7B-0924-GGUF_r32",
                                         trust_remote_code=True)
```

### Chat after compression

You can talk to the result immediately - `hf_chat.py` is included:

```bash
python3 hf_chat.py                # finds the newest artifact in results/
python3 hf_chat.py --model results/field_OLMoE-1B-7B-0924-GGUF_r32
python3 hf_chat.py --prompt "Hi!"         # single question, no dialog
python3 hf_chat.py --temperature 0.8 --max-new 400
python3 hf_chat.py --repetition-penalty 1.2   # stronger anti-loop
```

In-dialog commands: `/help`, `/reset`, `/system <text>`, `/temp <x>`,
`/rep <x>`, `/max <n>`, `/exit`. Replies print streamingly with speed
(tokens/s).

**About repetition loops.** A compressed model has slightly shifted logits;
under plain (greedy/low-temperature) decoding it is prone to degenerate
repetition loops. The chat therefore applies a repetition penalty by default
(1.15; tune with `--repetition-penalty` or `/rep <x>`, 1.0 = off). If loops
persist at `/rep 1.0`, that is a quality signal: raise the calibration pool
(`--per-layer-cap`), add `--refine-rounds 1-2`, or raise the rank - and check
the `worst blocks by final mse` line in the pipeline report for outliers.
OLMoE-0924 is a base model: if the artifact has no chat template, the chat
falls back to a Question/Answer format - coherent, but in the base model's
style. The artifact also plugs into any app of yours as a normal HF model
(code above). The artifact is light (~1.2 GB): chat takes ~2 GB of RAM -
works on a weak machine.

---

## Memory: how the pipeline protects the machine

The key decision is **STREAMING**: the full model (~14 GB) never loads. The
base/calibration stages work through `hf_stream.py`: RAM holds only the
backbone (~1.5 GB) + ONE block's experts for the duration of its layer's pass
(~0.8 GB); layers execute through the stock transformers code - the result is
bit-identical to the full model (A/B test on mini-OLMoE: logits/pairs/
centroids = 0 difference). Everything after that runs without the model: the
field fit uses pairs from disk (~1-2 GB RAM), the artifact is assembled
streamingly (~2 GB), the verification uses the finished artifact (~1-2 GB).
Peak RAM for the whole run ~3 GB - an 8 GB machine is enough.

The price of streaming is speed: experts are read from disk on every pass
(an SSD is required; an HDD is slow but gets there). The **prefetch thread**
(default on) dequantizes the next block in parallel with computing the current
layer, hiding most of that read cost; `--prefetch 0` disables it and saves
~1 block of RAM. Demo generation in streaming is shortened to 12 tokens - it
is a demonstration, not a result.

### Disk: what takes space and why (numbers for OLMoE Q4_K_M)

| what | why | size |
|---|---|---|
| GGUF source (`hf_cache/`) | the only weight carrier, resumes on interruption | ~4.4 GB |
| dequant checkpoint (`results/gguf_hf/*-hf/`) | by default **not created** (weights are read from the GGUF on the fly); the `--full-dequant` option | ~13.8 GB |
| pool cache (`results/cache_*/`) | block pairs bf16 + centroids + base log-probs: the fit restarts without the model, a new rank - without recalibration | ~2.3 GB |
| rank fit (`cache_*/fit_r32/`) | field r=32 parameters (fp32) - artifact assembly/re-run without a fit | ~0.7 GB |
| artifact (`results/field_*/`) | the finished model (backbone + field) | ~1.2 GB |

A fresh run totals **~8.6 GB** (it used to be ~22 GB with full dequant);
after `--cleanup` **~4.2 GB** remain (pool + fit + artifact - everything
needed for new ranks and chat). The dequant checkpoint is never needed: an
old one (from previous pipeline versions) is deleted at start, one created
via `--full-dequant` is deleted after success; when needed it rebuilds from
the GGUF without re-downloading. Experts never leak into the backbone
(an invariant check in the code: "expert tensors skipped: 32") - streaming
RAM is ~2-3 GB, not ~14.

About the "experts from disk: N block loads, X GB read" log line: that is
**reading**, not storage - streaming re-reads each layer's experts on every
model pass. Reads in GGUF mode are ~0.23 GB per block (Q4 packing), in
`--full-dequant` mode ~0.81 GB (fp16) - the whole stage 3 fits into ~110-120
GB of traffic instead of ~580 GB before (the redundant 16-chunk check of the
base against its own cache was removed: now a 2-chunk integrity check + base
ppl from the cache without the model).

- The calibration pool is cached in `results/cache_<model>/` and **does not
  depend on rank/steps**: a re-run with a different `--rank`/`--fit-*` skips
  stages 1-4 entirely (calibration = a pool of activation vectors, order and
  text do not matter). A specific rank's fit is cached separately
  (`cache_<model>/fit_r<rank>/`).
- Centroids accumulate over chunks (no full fp32 expert stack: -2 GB peak per
  block).
- `--low-mem` - a memory-frugal metrics mode: pair/chunk caps halved (lower
  RAM, nearly the same metric quality).
- `--threads 4` - limit torch threads to keep some cores free.
- `--cleanup` - after success erase the GGUF too (~4.4 GB); the calibration
  pool and artifact stay - new ranks build without recalibration (the GGUF
  re-downloads when needed, nothing already downloaded is lost).
- The HF cache (`hf_cache/`) lives inside the project folder; the system
  drive does not grow.

---

## Where the model downloads (not the system drive)

All downloads - the 4.4 GB GGUF from mradermacher, the allenai config/tokenizer,
the wikitext dataset - go to `hf_cache/` **inside the project folder**: the
scripts import `hf_env.py` first thing, and it redirects HF_HOME/HF_HUB_CACHE
before any network access. `C:\Users\<you>\.cache\huggingface` is not used
and does not grow; the cache moves and deletes with the project. You can
relocate the cache with your own `HF_HOME` - an externally set value wins.

---

## Useful run variants

```bash
python3 hf_pipeline.py --gguf-quant Q4_K_S        # smaller file (slightly coarser)
python3 hf_pipeline.py --gguf-quant Q5_K_M        # better base
python3 hf_pipeline.py --gguf-quant auto          # pick the best available quant
python3 hf_pipeline.py --gguf /path/model.Q4_K_M.gguf   # the GGUF is already downloaded
python3 hf_pipeline.py --rank 16                  # more aggressive (r=32 by default)
python3 hf_pipeline.py --calib-dataset wikitext-2-raw-v1   # better calibration (needs datasets)
python3 hf_pipeline.py --fit-steps 800            # longer fit - a more precise field
python3 hf_pipeline.py --low-mem --threads 4      # gentle mode
python3 hf_pipeline.py --skip-reload-check        # skip the final reload check
python3 hf_pipeline.py --smoke                    # quick wiring check
```

### Fit methods (types of refinement)

The fit accepts several optimizer kinds - `--fit-method`:

| method | what it is |
|---|---|
| `adam` | plain Adam, constant lr - the classic default |
| `adamw` | AdamW with a small weight decay (less drift in U,V) |
| `adam-cosine` | Adam + cosine lr decay to ~0 - usually the best final mse |
| `rmsprop` | RMSProp - an alternative for unstable blocks |

And presets - `--fit-preset` (explicit flags always win):

| preset | steps | batch | lr | method |
|---|---|---|---|---|
| `fast` | 120 | 4096 | 3e-3 | adam-cosine |
| `balanced` | 300 | 4096 | 2e-3 | adam-cosine |
| `quality` | 600 | 8192 | 2e-3 | adam-cosine |

No preset = the legacy defaults (300/4096/2e-3/adam). The fit cache signature
includes method+preset, so changing them triggers a re-fit automatically
(the calibration pool is NOT re-run - stages 1-4 stay cached).

`--fit-jitter 0.3` adds Gaussian noise to the fit inputs (in per-dim std
units, targets stay exact) - a cheap augmentation for a small calibration
pool; at low points-per-dimension it beats extra rank/steps.

### Tuning on a real model (first-run findings)

The first real run (NanoColibri Q8_0, defaults: 8192 pairs/layer, 300 steps,
r=64) passed the per-block fit guard but degraded end-to-end quality
(Δppl +67%) - the field was UNDERFIT. Checks done: the fit-time sigmoid
router matches the real HYV3TopKRouter exactly, and the routed+shared fp32
combine matches HYV3MoE - no structural mismatch. The dials that matter, in
order of cost:

1. **More fit steps** (no recalibration - the pool cache is reused):
   `--fit-steps 1200 --fit-method adam-cosine --fit-workers 3`.
   For hy_v3 the first share of steps goes into rescaling the centroid
   (the top-2 weights sum to 2.826, not 1 as in OLMoE), so the default 300
   is not enough for specialization.
2. **A bigger calibration pool** (recalibration): `--per-layer-cap 32768
   --calib-windows 12`. 8192 pairs at d=1024 is only 8 points/dimension -
   deep in the saturation-curve error zone (+30..150% on the toy). The
   pipeline now detects that the cached pool is smaller than the requested
   cap and recalibrates by itself - no manual cache surgery.
3. **Rank vs data**: under a data-limited pool a smaller rank (32) can beat
   a bigger one (64) - higher rank needs more points.
4. **Jitter**: `--fit-jitter 0.3` helps exactly in the data-starved regime.

Diagnosis hint: in the stage 5 log, "guard: mse ... (X% below the centroid
baseline)" - single-digit X per block = underfit confirmed.

### Speed knobs

Where the time actually goes (measured): the fit stage is GEMM-bound (native
MKL - Python overhead is negligible), and on hy_v3 blocks ~3/4 of the fit
FLOPs used to be spent recomputing the FROZEN shared experts every step.
Implemented optimizations:

- **Shared-expert folding** (automatic): the frozen shared branch is computed
  once per block and subtracted from the target
  (`MSE(field + shared, Y) == MSE(field, Y - shared)` - identical gradients,
  verified to 0.00e+00), plus the frozen router's z is computed once per pool.
  ~1.7x faster fit steps on hy_v3 blocks, no quality change.
- `--fit-workers 2-4` (default 1) - fit independent blocks in parallel; the
  longest stage scales with cores.
- `--fit-early-stop 50` (default 0 = off) - stop a block's fit after 2
  consecutive flat mse checkpoints (every N steps). Saves time on
  plateauing blocks.
- `--io-threads 2-4` (default 1) - GGUF expert tensors are split into
  per-expert slabs and dequantized in a thread pool (numpy releases the GIL;
  bit-identical output). Speeds up stage 3-4/6 block reads on a multi-core
  CPU; combines well with prefetch.
- `--prefetch 1` (default) - a background thread dequantizes the NEXT expert
  block while the current layer computes. `--prefetch 0` - synchronous
  (+~1 block of RAM saved).
- `--io-cache ram` (default `disk`) - copies the RAW PACKED GGUF tensors into
  RAM on first touch (lazily, tensor by tensor; total ~= the packed file
  size, e.g. ~2.7 GB for a 1B Q8_0 MoE). The cache is process-wide and shared
  by every streaming pass, so the calibration-pool collection, a refit pass
  and the artifact write each read the experts from disk only ONCE - after
  that all block loads are served from RAM. This is the main lever for slow
  storage (Google Colab disks, Google Drive mounts, HDDs): one sequential
  file read replaces thousands of small random ones. On a machine where RAM
  is too tight even for the packed file, stay on `disk` (prefetch +
  io-threads still help).
- The pool cache makes re-runs (new rank, new fit settings) skip stages 1-4
  entirely.

Colab note: put the GGUF on the LOCAL disk (`/content`), not a Drive mount -
a Drive FUSE mount turns every small read into a network round-trip. Then
`--io-cache ram --io-threads 2-4` on a standard Colab VM (12.7 GB RAM) fits
comfortably: packed GGUF (in cache) + backbone + one expert block.

Rough per-step cost on a NanoColibri-sized block (d=1024, r=64, bs 4096,
2 CPU cores): ~0.30 s/step after folding (was ~0.51 s) -> a 23-block fit at
300 steps ~0.6 h single-worker; `--fit-workers 2` halves it again.

Why not Rust: the fit is dominated by BLAS GEMMs (already native code) -
a Rust rewrite (e.g. candle) would call the same BLAS and gain only the
glue, which is not the bottleneck. The measured levers are parallelism
(workers/io-threads/prefetch), fewer wasted FLOPs (folding, early stop) and,
if ever available, a GPU (`--device auto` already uses it for the fit).

Measured dead ends (do not bother): `torch.compile` (239 vs 227 ms/step -
slightly slower, the step is GEMM-bound), fused Adam on CPU (zero gain -
the optimizer elementwise ops are noise vs GEMMs). ~90% of a fit step is
MKL GEMM; the remaining CPU levers are fewer steps (early stop, better
data) or a GPU.

**Is the plateau optimizer-made? No** (controlled benchmark, `opt_bench.py`:
NanoColibri-like geometry, rank-512 truth vs fit rank 64, 32 pairs/dim, all
variants on the same pool/seed). Final quality (% below the centroid
baseline): adam const 2e-3 88.9 / adamw 88.9 / adam-cosine 2e-3..1e-2
86.4-87.6 / ALS-hybrid (exact ridge solve of the down layer) 85.8 /
rmsprop 80.0 / LBFGS full-batch 68.7. The whole optimizer family spans
~9 pp, while the data lever spans ~20 pp (pool 2048 -> 8192 -> 32768:
66.5 -> 84.0 -> 86.4, same diminishing-returns curve as the bootstrap
calibration experiment). Constant lr slightly beats cosine within a fixed
step budget (~+2.5 pp; cosine spends the tail at a tiny lr) - `--fit-method
adam` is a free small win to A/B; rank 64/128/256 did NOT move the floor on
this synthetic problem (shared U,V bases bind), while on a real model rank
does help - the residual structure differs. Steps still buy a little:
const adam 1000 steps -> 90.6.

- Field quality is governed by **fit steps** (it adds no bytes to the
  artifact). By the budget experiment (toy, r=32): 400 steps -> Δppl +2.5%,
  800 -> +1.4%, 1600 -> +0.9%, 3200 -> +0.6%. The default 300 - "an hour or
  two on CPU"; if a night is fine - `--fit-steps 1600` or `--fit-preset quality`.
- Imatrix quants from mradermacher (slightly better quality):
  `--model mradermacher/OLMoE-1B-7B-0924-i1-GGUF`.
- A bf16 source instead of Q4 (~15 GB): `--model allenai/OLMoE-1B-7B-0924`.
- Q4 bitsandbytes (CUDA required): `--model RichardErkhov/allenai_-_OLMoE-1B-7B-0924-4bits`.

At startup the pipeline prints its **resolved profile** - the model, quant,
rank, fit method/steps/batch/lr, workers, prefetch and caps in one line - and
stores it into the artifact's `field_meta.json` for reproducibility.

---

## Resources on OLMoE-1B-7B (from mini-runs, scaling is linear)

| stage | disk | RAM | time CPU / GPU |
|---|---|---|---|
| Q4_K_M download | +4.4 GB | - | minutes |
| light catalog (config+tokenizer) | ~KB | ~1 GB | seconds |
| base+calibration (STREAMING from GGUF) | cache ~2.3 GB | **~2-3 GB** | tens of minutes (block dequant ~5 s/load + Q4 disk reads) |
| field fit (no model) | +0.7 GB fit | **~1-2 GB** | hours / tens of minutes |
| artifact assembly (streamingly from GGUF) | +1.2 GB artifact | **~2 GB** | minutes |
| artifact verification / chat | - | **~2 GB** | ongoing |

(with `--full-dequant` a "dequant ~14 GB, RAM ~5-6 GB, ~10-20 min" stage is
added before stage 3, but blocks then read without dequant - faster on a fast
SSD; after success the checkpoint deletes itself.)

- Free disk: **~8.6 GB** (GGUF 4.4 + pool 2.3 + fit 0.7 + artifact 1.2);
  with `--cleanup` after success ~4.2 GB remain (pool + fit + artifact).
- 8 GB RAM is enough for the whole pipeline: the model never loads in full,
  peak ~3 GB at stages 3-4 (backbone + one expert block); on an HDD stage 3
  is slow (experts read from disk) - SSD strongly recommended.
- CPU works without a GPU (bf16 inference); the fit is the longest part - a
  night will do; on a GPU - tens of times faster.
- The artifact is light: **~1.2 GB** for OLMoE (backbone ~1 GB fp16 + field
  0.21 GB) - smaller than even the original Q4 GGUF (4.4 GB). The chat opens
  it in bf16 by itself.

### Should the artifact be compressed again - no

The fat is already discarded at the swap: experts 12.9 GB -> field 0.21 GB.
Further options give little or break CPU inference:

- **field in Q4** (U,V,C, centroid in 4-bit): saves ~0.15 GB on a 1.2 GB
  artifact (~12%) - a hand-rolled unpacker in `modeling_field.py` costs more
  than it is worth;
- **Q4 backbone in the artifact** (bitsandbytes): the artifact would be
  ~0.6 GB, but bnb only works on CUDA - CPU inference dies; not recommended.

About "fitting straight in the quant without dequant": quality-wise the field
is ALREADY fitted on Q4 experts - dequant merely reads the packed values, no
loss there. The only removable thing was the intermediate 14-GB checkpoint on
disk (the streaming dequant of the GGUF straight into RAM): -14 GB of disk,
+10-20 minutes per run.

---

## Expected numbers (mini-PoC, same procedure)

On the toy model (4 layers, d=128, 8 experts top-2, fully trained) - numbers
AFTER the protocol fix (calibration and eval are non-overlapping text
segments; activation-aware methods are no longer fitted on the eval segment):

| variant | expert compression | KL, bits/token | Δppl |
|---|---|---|---|
| wh-SVD r=16 | x3.0 | 0.012 | +1.2% |
| field r=32 | x5.6 | 0.029 | +2.5% |
| field r=16 | x6.6 | 0.036 | +2.9% |
| field r=8  | x7.2 | 0.045 | +3.5% |
| dense-MLP | x8.0 | 0.137 | +10.3% |

Before the fix the activation-aware methods showed ~10-20% better KL (e.g.
field r=32: 0.024) by fitting on the same 20K eval segment; the qualitative
conclusions did not change. Weight-space methods (SVD/blocks/PQ/masks) did
not depend on the leak - their numbers reproduced byte for byte. fp16 deploy
is free (KL +0.000). Baseline comparison - in `examples/toy_report/`.

On OLMoE the expert compression will be **substantially higher** (~x60 at
r=32: it has 64 experts, and centroid+U,V amortize over the whole bank; the
full expert bank fp16 is ~12.9 GB, field r=32 ~0.21 GB). Final KL/Δppl on the
real model appear in `results/moe_hf_pipeline_report.md` after your run.

---

## How it works (briefly)

- The base router computes as usual: `probs = softmax(router(x))`; z - the
  top-8 soft weights (OLMoE: no renormalization).
- Movement coordinates: `c_gu = z @ C_gu`, `c_dn = z @ C_dn` (C is (64 x r)).
- The field-expert output:
  `gu = x @ Wgu1dᵀ + (x @ Vgu · c_gu) @ Uguᵀ`, `h = silu(gate) ⊙ up`,
  `y = h @ Wdn1dᵀ + (h @ Vdn · c_dn) @ Udnᵀ`.
- Fit: Adam on (MoE input -> MoE output) pairs; the centroid initializes from
  the expert mean; U,V start from `randn*0.02` (a zero init would freeze the
  rank path - the fit guard checks this); the artifact is then saved with the
  original router.

The artifact format is a plain HF model with `trust_remote_code=True`
(`modeling_field.py` is generated automatically). For vLLM the base model
works directly; the field version needs a small forward adapter (~10 lines,
see `modeling_field.py`).

---

## Package files

| file | purpose |
|---|---|
| `hf_pipeline.py` | the WHOLE compression pipeline in one command (GGUF -> field r=32 -> verify); phased for memory |
| `hf_gguf_to_hf.py` | Q4 GGUF download + dequant into a plain HF checkpoint (olmoe, hy_v3) |
| `hf_stream.py` | streaming runner: backbone in RAM, experts per block, background prefetch, optional raw-GGUF RAM cache (`--io-cache ram`) |
| `hf_field_transform.py` | the core: calibration (disk cache), field fit (several methods), deploy, metrics |
| `hf_chat.py` | chat with the transformed model in a terminal (streaming output, commands) |
| `hf_env.py` | redirects the HF cache into the project (imported first) |
| `modeling_field_template.py` | the template of the artifact's modeling code |
| `step1_compress.bat` / `step2_chat.bat` | double-click launchers on Windows |
| `make_tiny_olmoe_gguf.py` / `make_tiny_hyv3_gguf.py` | mini-GGUF generators (environment check without a 4 GB download) |
| `test_stream_mode.py`, `test_gguf_direct.py`, `test_field_fit_guard.py`, `test_io_cache.py`, `test_io_cache_stream.py` | A/B tests (bit-exact streaming vs full model; GGUF vs checkpoint; fit guard; `--io-cache ram` correctness) |
| `run_pipeline.sh` + `pipeline.py`, `common.py`, `train.py`, `transform_eval.py`, `variants_eval.py`, `upgrade_eval.py`, `bank_eval.py`, `masks_eval.py`, `field_eval.py`, `deploy.py`, `verify_transformed.py` | toy pipeline: trains a mini-MoE, compresses it, compares against baselines (SVD/PQ/BitDelta/dense) |
| `examples/toy_report/` | mini-PoC reports and numbers |

## Environment check without big downloads

```bash
python3 make_tiny_olmoe_gguf.py tiny.gguf
python3 hf_pipeline.py --gguf tiny.gguf --gguf-base-repo "" --smoke --out smoke_out
```

Runs the whole path (GGUF -> dequant -> compression -> verify -> reload+demo)
on a mini model in ~1 min.

## hy_v3 support (NanoColibri and relatives)

The converter understands two architectures: `olmoe` and `hy_v3` (the
HunYuan-v3 family, e.g. `mradermacher/NanoColibri-Instruct-GGUF`). The base
repo for config+tokenizer is detected AUTOMATICALLY from the GGUF metadata
(`general.base_model.0.repo_url`). hy_v3 specifics handled by the pipeline:
layer 0 is dense (the field is not placed there), shared experts (kept
exactly as-is; the field replaces only the routed experts), a sigmoid router
with `e_score_correction_bias` and scale 2.826, a tied lm_head.

Launch for NanoColibri Q8_0 (2.75 GB):

```
python3 hf_pipeline.py --model mradermacher/NanoColibri-Instruct-GGUF --gguf-quant Q8_0
```

or with an already downloaded file: `--gguf path\NanoColibri-Instruct.Q8_0.gguf`.
The mradermacher repo also has a `.f16.gguf` (5.17 GB) - if you want to
exclude source quantization entirely.

## The fit guard (anti-B1)

Field initialization: U,V ~ randn*0.02 (as in the PoC), C = 0. After the fit
a guard checks: mse must drop below the centroid baseline, and the Cgu/Cdn
coordinates cannot remain exactly zero. Otherwise the run FAILS with a clear
error (previously such a fit silently degraded into "one averaged expert").
Disable: `--skip-fit-guard`. A quick check on your machine:
`python3 test_field_fit_guard.py`.
