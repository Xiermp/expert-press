# CLI reference — every tool, every flag

One page with **all capabilities and all flags** of the project's
command-line tools. Defaults are the values you get when the flag is
omitted. "Preset" means an explicit flag always wins over a preset bundle.

Contents: [hf_pipeline.py](#hf_pipelinepy) ·
[hf_chat.py](#hf_chatpy) · [temp_calibrate.py](#temp_calibratepy) ·
[router_audit.py](#router_auditpy) · [router_ft.py](#router_ftpy) ·
[field_dims.py](#field_dimspy) · [hf_gguf_to_hf.py](#hf_gguf_to_hfpy) ·
[service & eval scripts](#service--eval-scripts)

---

## hf_pipeline.py

The whole compression: source GGUF → field artifact → metrics → reports.
Stages and toggles are described in
[Pipeline-and-Stages](Pipeline-and-Stages.md); `--list-stages` prints the
same table from the tool.

### Source & artifact

| flag | default | meaning |
|---|---|---|
| `--model` | `mradermacher/OLMoE-1B-7B-0924-GGUF` | HF id; a repo ending in `-gguf` is treated as a GGUF repo; bf16 source: `allenai/OLMoE-1B-7B-0924` |
| `--gguf-quant` | `Q4_K_M` | which quant to take from a GGUF repo: `Q4_K_M`, `Q4_K_S`, `Q3_K_M`, `Q8_0`, ... or `auto` (best available) |
| `--gguf-file` | — | exact `.gguf` name in the repo |
| `--gguf` | — | local `.gguf` file (skips downloading) |
| `--gguf-out` | `results/gguf_hf/<name>-hf` | folder for the light catalog / dequant checkpoint |
| `--gguf-base-repo` | auto from GGUF metadata | where config/tokenizer come from (`""` = build from GGUF) |
| `--local-path` | — | path to an already-downloaded HF model |
| `--out` | `results/field_<tag>_r<rank>` | artifact folder |
| `--rank` | `32` | field rank |
| `--auto` | off | zero-config: `--gguf-quant auto` + `balanced` fit preset |

### Compute

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--dtype` | `auto` → bfloat16 (float16 on pre-Ampere GPUs) | `bfloat16` / `float16` / `float32` |
| `--threads` | all cores | limit torch CPU threads (e.g. `4` keeps the machine responsive) |
| `--low-mem` | off | halve pair/chunk caps for the metric passes (lower RAM, ~same metrics) |
| `--profile` | `auto` | hardware profile: `low` = weak-PC defaults, `high` = lift limits (io-cache ram, io-threads 4, calib-bsz 16 on GPU); `auto` = `high` when CUDA present or ≥32 GB RAM + ≥8 cores, else `low`. Explicit `--io-cache`/`--io-threads` win. See [Colab](Colab.md) |

### Texts & calibration

| flag | default | meaning |
|---|---|---|
| `--calib-file` | bundled corpus (auto-download) | calibration text file |
| `--eval-file` | tail 10% of the calibration text | eval text (leak-free split) |
| `--calib-dataset` | — | e.g. `wikitext-2-raw-v1` (needs `datasets`) |
| `--text-cap` | `3000000` | text characters cap |
| `--calib-windows` | `3` | calibration passes over the text |
| `--calib-bsz` | `8` | calibration batch size |
| `--calib-ctx` | `512` | calibration window length |
| `--per-layer-cap` | `8192` | pairs kept per block (the pool is shared by all ranks) |

### Fit

| flag | default | meaning |
|---|---|---|
| `--fit-preset` | — | `fast` (120 steps) / `balanced` (300) / `quality` (600); `--auto` implies `balanced` |
| `--fit-steps` | `300` (or preset) | fit steps per block |
| `--fit-bs` | `4096` (or preset) | fit batch size (pairs per step) |
| `--fit-lr` | `2e-3` (or preset) | learning rate |
| `--fit-method` | `adam` (or preset) | `adam` / `adamw` / `adam-cosine` / `rmsprop` |
| `--fit-jitter` | `0.0` | Gaussian noise on fit inputs; `0.2-0.3` helps only at <8 pairs/dim |
| `--fit-workers` | `1` | parallel fit workers for independent blocks (2-4 on multi-core) |
| `--fit-early-stop` | `0` (off) | stop a block after 2 flat mse checkpoints every N steps |
| `--skip-fit-guard` | off | do not abort when the fit fails to beat the centroid baseline |
| `--refine-rounds` | `0` | self-distillation rounds after the first fit (try 1-2) |

### I/O & streaming

| flag | default | meaning |
|---|---|---|
| `--io-threads` | auto: `4` on the high profile, else `1` | GGUF dequant threads for expert tensors (2-4 on multi-core) |
| `--prefetch` | `1` | background dequant of the next expert block (0 = off, saves ~1 block of RAM) |
| `--io-cache` | auto: `ram` when enough free RAM, else `disk` | `ram` = keep packed GGUF tensors in RAM (huge win on Colab/Drive/HDD); auto downgrades to `disk` if the packed file + ~5 GB margin does not fit |
| `--full-dequant` | off | build the ~14 GB HF checkpoint (old path; default reads the GGUF directly) |
| `--keep-dequant` | off | keep that checkpoint after success (default: deletes itself) |
| `--max-shard` | `4GB` | artifact safetensors shard size |
| `--save-backbone` | `keep` | `keep` (source quant) / `bf16` (dequant backbone, CPU-friendly) |
| `--cleanup` | off | erase the GGUF after success (pool + fit + artifact stay) |

### Metrics & reports

| flag | default | meaning |
|---|---|---|
| `--eval-chunks` | `50` | eval windows for ppl/KL |
| `--kl-chunks` | `16` | chunks for the base log-prob cache |
| `--eval-ctx` | `512` | eval window length |
| `--gen-tokens` | `48` | demo generation length; `0` = no demo generation at all |
| `--gen-rep-pen` | `1.15` | repetition penalty for demo generations (both models) |
| `--skip-reload-check` | off | == `--skip verify` |

### Stage toggles

| flag | default | meaning |
|---|---|---|
| `--stages` | all stages | run ONLY these: `download,texts,base,calibrate,fit,refine,save,verify,report` |
| `--skip` | — | run all EXCEPT these (mutually exclusive with `--stages`) |
| `--list-stages` | — | print the stage table and exit |
| `--no-cache-verify` | off | skip the 2-chunk log-prob cache self-check (stage 3) |
| `--skip download` | — | reuse-only: no network; local GGUF/HF cache + existing catalog required |
| `--smoke` | off | mini wiring run (short fit/eval caps) |

---

## hf_chat.py

Terminal chat with an artifact (finds the newest `results/field_*` itself).
Auto-loads `sampling.json` (fitted temperature) when present.

| flag | default | meaning |
|---|---|---|
| `--model` | newest artifact | artifact folder (or any HF model) |
| `--prompt` | — | single question, no dialog |
| `--system` | — | system prompt |
| `--temperature` | from `sampling.json`, else 1.0 | sampling temperature (`0` = greedy) |
| `--top-p` | `0.9` | nucleus sampling |
| `--min-p` | `0.0` (off) | long-tail cut; `0.05-0.15` tames style drift |
| `--repetition-penalty` | `1.15` | anti-loop; `1.0` = off |
| `--max-new` | `256` | reply length cap |
| `--device` / `--dtype` | `auto` / `auto` | as usual |
| `--no-stream` | off | print the whole reply at once |

In-dialog commands: `/help`, `/reset` (clear history), `/system <text>`,
`/temp <x>` (0 = greedy test), `/rep <x>`, `/minp <x>`, `/max <n>`,
`/exit` (`/quit`).

---

## temp_calibrate.py

Fits one sampling temperature minimizing `KL(base ‖ softmax(field/T))` on
seeded eval windows; base is STREAMED from the GGUF (never fully loaded).
Writes `sampling.json` into the artifact folder (auto-used by `hf_chat.py`).

| flag | default | meaning |
|---|---|---|
| `--model` | newest `field_*` | artifact folder |
| `--gguf` | *(required)* | the SAME GGUF the artifact was built from |
| `--src` | `results/gguf_hf/<gguf>-hf` | config dir for the streamed base |
| `--calib-file` | bundled corpus | the same corpus as during compression (eval tail 10% is used) |
| `--calib-dataset` | — | e.g. `wikitext-2-raw-v1` |
| `--chunks` | `8` | eval windows |
| `--ctx` | `128` | window length |
| `--tmin` / `--tmax` / `--step` | `0.30` / `1.50` / `0.02` | the T grid |
| `--device` / `--dtype` | `auto` / `auto` | as usual |
| `--io-threads` | `1` | GGUF dequant threads |
| `--io-cache` | `ram` | raw packed tensors in RAM (recommended) |
| `--prefetch` | `1` | background block reader |

Output: best T, KL at T=1 vs best, entropies (base / field / field@T),
top-1 agreement (structural, T-invariant), the full grid.

---

## router_audit.py

Read-only router audit of an artifact — three phases: base-vs-field
per-layer drift (needs `--base`), artifact routing stats, counterfactual
z-scramble. JSON goes to `results/router_audit_<tag>.json`. Details:
[Router-Diagnostics](Router-Diagnostics.md).

| flag | default | meaning |
|---|---|---|
| `--artifact` | *(required)* | field artifact dir |
| `--base` | — | base model: HF dir or `.gguf` (auto-converted once; enables phase 1) |
| `--text` | `corpus.txt` | text file for windows |
| `--ctx` | `256` | window length |
| `--windows` | `8` | windows for drift stats |
| `--scramble-windows` | `4` | windows for the counterfactual pass |
| `--no-scramble` | off | skip phase 3 |
| `--dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--threads` | all cores | torch threads |
| `--out` | `results/router_audit_<tag>.json` | JSON output path |

---

## router_ft.py

Surgical gate calibration: Adam steps on the gate weights ONLY (everything
frozen), loss = `KL(base ‖ field)` + anchor ‖W−W₀‖². Saves a NEW artifact
`<artifact>_rft` only if quality improved.

| flag | default | meaning |
|---|---|---|
| `--artifact` | *(required)* | field artifact dir |
| `--base` | *(required)* | HF dir or `.gguf` (targets) |
| `--text` | `corpus.txt` | calibration text |
| `--out` | `<artifact>_rft` | output artifact dir (refuses to overwrite) |
| `--steps` | `40` | Adam steps (one window each) |
| `--lr` | `3e-5` | learning rate |
| `--anchor` | `1.0` | weight of the ‖W−W₀‖² penalty (0 = free router) |
| `--ctx` | `256` | window length |
| `--eval-windows` | `6` | held-out windows for before/after |
| `--dry-run` | off | run the steps in RAM, measure, save nothing |
| `--force-save` | off | save even if metrics did not improve |
| `--device` / `--threads` | `auto` / all | as usual |

---

## field_dims.py

Instant accounting "how much onto how much" for an artifact (no model load).

| flag | default | meaning |
|---|---|---|
| `--artifact` | *(required)* | field artifact dir |

Prints: dims (d_model × d_ff × N → rank r), per-block params
(experts vs field, the x-factor), the field mix (centroid / U,V /
coordinates %), all-layers MB from `field_meta.json`, on-disk safetensors
size, and a ready one-line report.

---

## hf_gguf_to_hf.py

Standalone GGUF → HF converter (olmoe + hy_v3 architectures). The pipeline
uses it for the light catalog / `--full-dequant`; usable directly:

| flag | default | meaning |
|---|---|---|
| `--repo` | `mradermacher/OLMoE-1B-7B-0924-GGUF` | HF repo with GGUF |
| `--quant` | `Q4_K_M` | quant to pick, or `auto` (best available) |
| `--gguf-file` | — | exact file name in the repo |
| `--gguf` | — | local `.gguf` (skip downloading) |
| `--out` | `results/gguf_hf/<name>-hf` | folder for the HF checkpoint |
| `--dtype` | `float16` | `float16` / `float32` |
| `--base-repo` | `allenai/OLMoE-1B-7B-0924` | source of config/tokenizer (auto-detected from GGUF metadata when the pipeline calls it) |

---

## Service & eval scripts

Not user-facing flags — constants at the top of the file:

| script | role |
|---|---|
| `hf_env.py` | redirects the HF cache into the project; imported first everywhere |
| `hf_stream.py` | the streaming runner library (backbone in RAM, per-block experts, prefetch, `io_cache`) |
| `hf_field_transform.py` | the core library: pool collection, fit, artifact write, metrics |
| `modeling_field_template.py` | template of the artifact's `modeling_field.py` |
| `make_tiny_olmoe_gguf.py` / `make_tiny_hyv3_gguf.py` | generate a mini GGUF for a ~1-min end-to-end check |
| `test_stream_mode.py`, `test_gguf_direct.py`, `test_field_fit_guard.py`, `test_io_cache.py`, `test_io_cache_stream.py` | A-tests: streaming bit-exactness, GGUF-direct equivalence, fit guard, io-cache correctness |
| `pipeline.py` + `common.py`, `train.py`, `transform_eval.py`, `variants_eval.py`, `upgrade_eval.py`, `bank_eval.py`, `masks_eval.py`, `field_eval.py` (`--ranks`, `--fit-steps`, `--save-dir`), `deploy.py`, `verify_transformed.py` | the toy-PoC ladder that produced the [Research-History](Research-History.md) charts |
| `step1_compress.bat` / `step2_chat.bat` | Windows double-click launchers (compress / chat) |
