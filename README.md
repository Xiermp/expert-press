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

## Documentation

Full docs live in the [wiki/](wiki/) folder:

| page | about |
|---|---|
| [wiki/Home.md](wiki/Home.md) | the project card and the ten-minute tour |
| [wiki/Field-Engine.md](wiki/Field-Engine.md) | the method: formula, what is stored, why it works |
| [wiki/Pipeline-and-Stages.md](wiki/Pipeline-and-Stages.md) | the 9-stage pipeline and **every stage toggle** |
| [wiki/Memory-and-Speed.md](wiki/Memory-and-Speed.md) | streaming, `--io-cache ram`, threads, disk layout |
| [wiki/Quality-and-Calibration.md](wiki/Quality-and-Calibration.md) | the KL protocol, style-drift findings, temperature/min-p fixes |
| [wiki/Router-Diagnostics.md](wiki/Router-Diagnostics.md) | `router_audit.py` / `router_ft.py` / `field_dims.py` |
| [wiki/Research-History.md](wiki/Research-History.md) | the experiment ladder with all charts |
| [wiki/CLI-Reference.md](wiki/CLI-Reference.md) | **every tool, every flag, every default** |

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

### Stage toggles: run only what you need

The auto-pipeline is 9 stages (`download → texts → base → calibrate → fit →
refine → save → verify → report`), and every one of them can be switched
on/off. The plan is printed before the run and recorded in the report:

```bash
python3 hf_pipeline.py --list-stages              # the stage table
python3 hf_pipeline.py --stages fit,save,verify   # run ONLY these (from cache)
python3 hf_pipeline.py --skip base,verify,report  # run all EXCEPT these
python3 hf_pipeline.py --skip download            # reuse-only: never touch the network
python3 hf_pipeline.py --stages fit --rank 64     # a new rank from the cached pool
python3 hf_pipeline.py --stages refine --refine-rounds 1   # refine standalone
python3 hf_pipeline.py --stages verify            # ONLY the loss check: KL/ppl/% from the caches
python3 hf_pipeline.py --gen-tokens 0             # no demo generations
python3 hf_pipeline.py --no-cache-verify          # skip the 2-chunk cache self-check
```

Completed refine rounds are cached too (`done_r*.json` + `pairs_sig.json`
in the run cache): a repeated run skips the hours-long capture pass and the
refit entirely; `--refresh-refine` forces a full redo. The refine refit
honors `--fit-workers` (2-4 on a multi-core CPU). The runtime artifact is
dtype-robust: mixed-dtype checkpoints (fp32 backbone + bf16 field) and
fp32-returning v5 routers no longer crash matmuls (10.6).

Cheap missing stages (`download`, `texts`) are auto-added with a notice;
expensive missing prerequisites (`base`, `calibrate`, `fit`, `verify`)
fail fast with a hint instead of surprising you with a multi-hour pass.
A new rank reuses the calibration pool (it is rank-independent), so
`--stages fit,save,verify --rank 16` costs only the fit. Full semantics,
scenarios and the plan behavior: [wiki/Pipeline-and-Stages.md](wiki/Pipeline-and-Stages.md).

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

### Quality: taming the style drift (temperature calibration)

Symptom: short replies are fine, but after a couple of turns the model drifts
into a "neighbour mode" - for OLMoE an archaic / poetry register ("I pray
thee..."), often with verse-like line breaks. Cause: compression flattens the
distribution (the measured KL, e.g. 0.757 bits/token); a flattened histogram
samples long-tail tokens too often, and the drift compounds over the dialog.

First-line fixes, cheapest first:

1. **Temperature calibration** (one scalar, fitted, automatic):

```bash
python3 temp_calibrate.py --model results/field_xxx \
    --gguf /content/OLMoE-1B-7B-0924-Q4_K_M.gguf \
    --calib-file corpus.txt          # the same corpus as during compression
```

The base model is streamed from the GGUF (never fully loaded); the tool
minimizes `KL(base || softmax(field/T))` and saves `sampling.json` next to
the artifact. `hf_chat.py` picks the fitted temperature up automatically
(`--temperature` / `/temp` still override it). Expect T < 1 for a visibly
compressed artifact (e.g. 0.6-0.9).

2. **min-p sampling** - cuts the long tail that carries the archaic tokens:
   `python3 hf_chat.py --min-p 0.1` (or `/minp 0.1` in-dialog; 0 = off).

3. **Greedy test** - `/temp 0`: if greedy output is clean, the drift is a
   sampling problem (fixes 1-2 are enough); if greedy still drifts, the
   error is structural -> audit the router (`router_audit.py`), then
   consider per-depth rank reallocation or gate calibration (`router_ft.py`).

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
- **Interrupted runs resume** (2026-09-04.3): a kill at ANY point - during
  the base log-prob pass, the stage-4 init loop, or mid-fit - loses nothing
  that was already written. The next run reuses the pair pool (never
  re-collected), rebuilds only the missing `init_blk*.pt` (per block, with
  progress prints), re-fits only the blocks without a valid `fit_blk*.pt`
  (`fit resume: K/N blocks already fitted`), and recomputes only the missing
  `lp_XXX.pt` chunks. All cache files are written atomically. A pool left
  orphaned by a pre-.3 interrupted run (no art_meta.json) is adopted
  automatically. To force a clean recalibration instead, delete
  `results/cache_<model>/`.
- **A smaller cached pool is kept by default** (2026-09-05.1): if the pool on
  disk holds fewer pairs/block than the requested `--per-layer-cap` (e.g. you
  raised cap 49152 -> 65536), the run no longer forces the most expensive
  re-collection - it keeps the cached pool when it is still usable
  (>= max(4096, 16*rank, cap/2) pairs/block) and prints a one-line notice
  (49k pairs at rank 128 = 384 pairs/dim is already ample). Pass
  `--pool-recalibrate` to restore the old forced re-collection. This also
  removes the OOM loop on low-RAM boxes, where the re-collection was the
  thing that died (and made every restart look like "nothing was saved").
- **OOM with `--io-cache ram` degrades instead of dying** (2026-09-05.1):
  when a block load fails to allocate (ram cache + backbone + dequant
  scratch > free RAM), the packed-GGUF RAM copy (~= the file size) is
  dropped, prefetch is disabled, and the run continues from disk/mmap with a
  notice - instead of the previous hard `Unable to allocate ... MiB` crash.
  The dequant itself now also peaks at (output + ~4 MB slab) instead of
  2-3 full-size intermediates (bit-identical output). On boxes with <6-8 GB
  free RAM, `--io-cache disk` is still the recommended explicit choice.
- **Swap-storm guard** (2026-09-05.2): low RAM does not always produce a
  clean MemoryError - Windows may slide into a swap-storm where allocations
  succeed but everything crawls (seen in the wild: a refine round frozen at
  "13 block loads" with no error). Now every streaming stage re-checks the
  io-cache ram request against CURRENT free RAM (1.5x the packed GGUF + 1 GB
  headroom) and falls back to disk with a notice (`MOE_FORCE_IO_RAM=1`
  overrides); a watchdog thread drops the ram cache as soon as free RAM
  crosses 0.6 GB, before the thrash starts (`MOE_NO_RAM_WATCHDOG=1` off);
  the refine capture pass flushes pair chunks at a RAM-adaptive threshold
  and prints a per-window progress line so a stall is visible.
- **hy_v3 artifacts now load and run** (2026-09-05.3, build 10.5): the first
  full NanoColibri run hit `HYV3TopKRouter.forward() missing 1 required
  positional argument: 'e_score_correction_bias'` in Stage 7, and the load
  report flagged `shared_experts.*` / `e_score_correction_bias` as UNEXPECTED.
  The artifact runtime template was OLMoE-only; it now (a) passes the routing
  bias to the router (sniffed from the host's signature, with a fallback call
  for older transformers), (b) ships the always-on `shared_experts` branch and
  sums it with the field output in fp32 - the same math the fit measured - and
  (c) fixes a latent NameError in the Linear-gate branch. Existing artifacts
  need NO refit: overwrite their `modeling_field.py` with the pre-rendered
  `modeling_field_HYV3_ready.py` and re-verify (`--stages verify`), or re-run
  with `--refine-rounds 0` to rebuild the artifact from the cached fits.
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
python3 hf_pipeline.py --refresh-init             # manually rebuild the SVD init files (normally automatic since 2026-09-05.5)
python3 hf_pipeline.py --fit-method muon-cosine   # Muon on the U/V factors
python3 hf_pipeline.py --fit-autocast off         # force fp32 matmuls (the probe keeps fp32 automatically when bf16 is slower)
```

### Fit methods (types of refinement)

The fit accepts several optimizer kinds - `--fit-method`:

| method | what it is |
|---|---|
| `adam` | plain Adam, constant lr - the classic default |
| `adamw` | AdamW with a small weight decay (less drift in U,V) |
| `adam-cosine` | Adam + cosine lr decay to ~0 - usually the best final mse |
| `rmsprop` | RMSProp - an alternative for unstable blocks |
| `muon` | BY-NAME split: U*/V* factors get Muon's NS-orthogonalized updates, everything else (centroids, coordinates C, router) stays Adam |
| `muon-cosine` | muon + cosine lr decay |

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
- `--io-cache ram` (default: auto - `ram` when enough free RAM, else `disk`) -
  copies the RAW PACKED GGUF tensors into
  RAM on first touch (lazily, tensor by tensor; total ~= the packed file
  size, e.g. ~2.7 GB for a 1B Q8_0 MoE). The cache is process-wide and shared
  by every streaming pass, so the calibration-pool collection, a refit pass
  and the artifact write each read the experts from disk only ONCE - after
  that all block loads are served from RAM. This is the main lever for slow
  storage (Google Colab disks, Google Drive mounts, HDDs): one sequential
  file read replaces thousands of small random ones. On a machine where RAM
  is too tight even for the packed file, stay on `disk` (prefetch +
  io-threads still help). An explicit `ram` that does not fit is downgraded
  to disk per stage with a notice (see the swap-storm guard above);
  `MOE_FORCE_IO_RAM=1` forces it anyway.
- `--profile auto|low|high` (default `auto`) - the weak-PC limits are lifted
  automatically on stronger hardware: CUDA present (or >= 32 GB RAM + >= 8
  cores) -> `high`: io-cache ram, io-threads 4, calib-bsz 16 on GPU, and
  fp16 on pre-Ampere GPUs (a T4 has no native bf16). `low` (or `--low-mem`)
  keeps the historical behavior, so 8 GB machines are unaffected. The
  resolved profile is printed in the `hardware:` line at startup.
- GPU: `--device auto` runs the whole streaming path on cuda:0 when a GPU is
  present (backbone + one expert block in VRAM at a time); the fit uses it
  as before.
- The pool cache makes re-runs (new rank, new fit settings) skip stages 1-4
  entirely.

Google Colab: the ready notebook `moe_router_colab.ipynb` (in this folder)
does the full loop - T4 GPU runtime, GGUF downloaded ONCE into an HF cache
on Drive (`HF_HOME` -> Drive), pool cache and artifact synced to Drive, so a
reconnect skips the download and stages 1-4 entirely. Same plain
`hf_pipeline.py` command inside, no adaptation. Details: `wiki/Colab.md`.

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
| `hf_chat.py` | chat with the transformed model in a terminal (streaming output, commands, auto temperature from `sampling.json`, min-p) |
| `temp_calibrate.py` | fits the sampling temperature to the base model's confidence (streamed base vs artifact; writes `sampling.json`) |
| `router_audit.py` | per-layer router audit of an artifact (drift, load balance, z-scramble) -> JSON |
| `router_ft.py` | surgical gate calibration: gate-only KL fit, anchored; saves `<artifact>_rft` on improvement |
| `field_dims.py` | artifact accounting one-liner (dims, params, field mix, bytes) |
| `hf_env.py` | redirects the HF cache into the project (imported first) |
| `modeling_field_template.py` | the template of the artifact's modeling code |
| `step1_compress.bat` / `step2_chat.bat` | double-click launchers on Windows |
| `make_tiny_olmoe_gguf.py` / `make_tiny_hyv3_gguf.py` | mini-GGUF generators (environment check without a 4 GB download) |
| `test_stream_mode.py`, `test_gguf_direct.py`, `test_field_fit_guard.py`, `test_io_cache.py`, `test_io_cache_stream.py`, `test_lowram_fix.py` | A/B tests (bit-exact streaming vs full model; GGUF vs checkpoint; fit guard; `--io-cache ram` correctness; LOW-RAM FIX + swap-storm guard: memory-safe dequant, keep-smaller pool, OOM degrade, runner ram-fit re-check, watchdog) |
| `run_pipeline.sh` + `pipeline.py`, `common.py`, `train.py`, `transform_eval.py`, `variants_eval.py`, `upgrade_eval.py`, `bank_eval.py`, `masks_eval.py`, `field_eval.py`, `deploy.py`, `verify_transformed.py` | toy pipeline: trains a mini-MoE, compresses it, compares against baselines (SVD/PQ/BitDelta/dense) |
| `examples/toy_report/` | mini-PoC reports and numbers |
| `wiki/` | the project wiki (method, pipeline, quality findings, research history with charts, CLI reference) |

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

### Guard rework (update 2026-09-04.1) - "the guard is too aggressive"

Three changes, validated on a toy bench (`scripts/bench_toy_router_guard.py`,
reproduced the premature aborts and the rescues):

1. **Divergence-bail warmup** `--fit-guard-warmup` (default `-1` = auto,
   `max(30, steps//10)`): the mid-fit 2x bail ("loss diverged ... stopping
   this block early") is armed only AFTER this many steps. Before: the bail
   fired on Adam's normal early overshoot at step ~1-12, cut the fit before
   the optimizer had time to adapt and shipped a near-centroid block.
   `0` = the old always-armed behavior.
2. **Soft end-guard** (default): a fit that ends within (0.98..1.0]x of the
   centroid baseline now WARNS and ships the best state (no gain, no harm,
   the run continues). The old hard abort is opt-in again:
   `--strict-fit-guard`. A fit that ended WORSE than the baseline still
   aborts in both modes. The baseline and the post-restore re-eval are now
   both 8-batch averages (single-batch numbers jitter ~10% and produced
   false alarms).
3. **Linear lr warmup** `--fit-lr-warmup N` (default 0 = off): the lr ramps
   0 -> fit_lr over the first N steps, giving Adam its adaptation window.
   On the toy bench this turned an aborting fit into a real one
   (-3.9%..-11.7% vs the baseline). Recommended when blocks stall or blow up
   at the start: `--fit-lr-warmup 30..50`.

## The original router joins the rebuild (`--fit-router`)

Your idea from 2026-09-04: while rebuilding the experts, tune the ORIGINAL
router too ("за компанию"), to shrink the post-conversion gaps. Implemented
IN PLACE on the calibration pairs (the model is not in RAM, the tuned router
is a same-shape replacement of the gate weight - zero artifact memory cost):

- `--fit-router after`  - short anchored polish once the field fit is done
  (`--router-steps 80`, `--router-lr` = fit lr, `--router-anchor 0.03`).
- `--fit-router joint`  - the router trains alongside the field from step 0
  (z recomputed every step; costs a few % of step time).
- The tuned gate weight lands in the fit files (`gw_tuned`) and REPLACES the
  backbone gate weight in the artifact; `field_meta.json` records
  `router_polish.n_layers_tuned`, stats in `fit_dir/router_meta.json`.

**Honest expectations (toy bench, 2026-09-04):** after a CONVERGED field fit
the router is usually NOT the bottleneck. The z-dependent rank correction
carries a small share of the output energy, so gw gradients are tiny: even a
30% top-k set change moved the block mse by ~0, and the result was flat on a
confident ("skewed") router too. The polish is SAFE (anchored, best-state
tracked, falls back to the original gw) but treat it as a cheap diagnostic;
`--refine-rounds` (input-shift self-distillation) remains the stronger lever
for the post-conversion gaps.

`router_ft.py` was reworked for memory at the same time: the artifact loads
in bf16 (was fp32 = 2x RAM), ONLY fp32 gate-master copies train
(`torch.func.functional_call` keeps the router math bit-identical on any
architecture), the rest of the model is properly frozen (before: every
backbone parameter still had requires_grad=True and the backward graph
materialized grads for the whole model). Net: ~2.8x less RAM on the tuning
pass. `--inplace` rewrites the gate weights inside the artifact itself
(default stays: a separate `<artifact>_rft` copy).

## The "original activator hinders" hypothesis - checked, not confirmed

Tested on the toy bench at r=16 (same fit budget for all variants, held-out
mse on fresh inputs): a learnable per-dim scale before SiLU (`silu(g*gamma)`)
and a scalar temperature gave NO improvement over the plain SiLU field;
replacing SiLU with GELU made things clearly WORSE (+48% mse) - the field's
act_fn must MATCH the base model's activation, it is load-bearing, not a
hindrance. A rank-split probe (gu-side-only vs dn-side-only corrections vs
both) showed both halves contribute and together they are best. Conclusion:
the residual error lives in the rank budget and the deploy-time input shift
(use `--refine-rounds`), not in the activation's properties.

---

## UPDATE-10: speed rebuild + the three holes (`--fit-init svd`, `--fit-autocast auto`, `--fit-method muon-cosine`)

Three problems reported on 2026-09-04 turned out to be real, and the fix for
the first one is also the biggest speed win to date.

**Hole 1 - the init was throwing the experts' structure away.** U, V started
as white noise (`randn*0.02`) and C as zeros, while the REAL expert deltas
`dW_e = W_e - centroid` sit on disk. One caveat: the SVD of the MEAN delta is
the SVD of zero (`sum_e dW_e = 0` identically) - the meaningful variant is
the shared-basis SVD of the STACKED deltas. The pipeline now computes it in
one streaming pass pair (`expert_basis_init`: randomized range finder + a
small core SVD, one expert in RAM at a time), saves it per rank as
`init_svd_blk*.pt`, and the fit starts from that basis. Toy bench (d=512,
r=32, 70% shared delta structure): the subspace pair carries ~100% of the
delta energy, the diagonal coordinates capture 49-68% at step 0, and after
the SAME 120-step budget the SVD-init fit reaches held-out mse 0.00198 vs
0.00548 for the random init - same quality in roughly 3x fewer steps.

**Hole 2 - C must never go through Newton-Schulz.** `--fit-method muon` /
`muon-cosine` split BY NAME: only the U*/V* operator factors are Muon
candidates; Cgu/Cdn (independent per-expert coordinates - NS would couple
unrelated experts and equalize their singular values) and the router stay on
Adam. The toy bench confirmed C-in-muon brings no gain (0.002312 vs 0.002319
held-out), so excluding it is free. `--muon-max-dim` (default 512) gates U/V
by min(shape): on real models the big centroid matrices stay on Adam; on
small geometries they join Muon automatically - the toy's best arm
(0.001706, ~14% better than adam-cosine), but on real models NS on the big
matrices costs ~40-50% extra step time, hence the cap.

**Hole 3 - jitter routing follows the clean anchor now.** The targets were
produced by the base model on the CLEAN rows; routing on the NOISY input
paired targets with whatever experts survived the noise (top-k flips =
irreducible mse floor). The fit and the guard now both take z of the clean
row (`z_all[ix]`), which is also one `_z` recompute per step cheaper.

**Speed: honest autocast probe.** `--fit-autocast auto` (default) runs 8
REAL fit steps per block geometry (1 warmup + 3 timed per dtype arm, min,
private RNG 0xC0FFEE, params restored bit-exact after) and keeps bf16
autocast only when it wins >=1.2x - on machines without a bf16 ISA it
quietly stays fp32. Parameters remain fp32 either way (exact gradients);
only the forward matmuls run in bf16 via oneDNN. Toy box: 1.7-1.8x per step.
The decision is cached per geometry, so identical blocks do not re-probe.

**SVD-init self-heal (2026-09-05.5).** The per-rank SVD init files
(`fit_r{R}/init_svd_blk*.pt`) can go missing when the pool cache was built
by an older package or a previous run silently fell back to the random init
- the fit then started "blind" (~3x more steps for the same quality, and the
blind fit was cached and reused forever). The pipeline now detects the gap
before the fit and rebuilds the files in a short streaming pass (one expert
read per block, no model forward - minutes on a 4.4 GB GGUF); an existing
random-init fit is re-fitted automatically, and refine rounds are re-done on
top of the new fit (their cached pairs depend on the field's own outputs).
The report JSON carries `fit_init_effective` so a "random" fallback can no
longer slip through unnoticed. `--refresh-init` still exists for a manual
rebuild (idempotent, resumable); `--fit-init random` restores the old
init on purpose.
