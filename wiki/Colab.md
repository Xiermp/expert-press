# Running in Google Colab (and stopping the re-downloads)

Colab resets its disk every session, so without caching you re-download the
4.4 GB GGUF (and rebuild the calibration pool) on every reconnect. The
project ships a ready notebook — **`moe_router_colab.ipynb`** (in
`expert-press-update.zip`) — that wires three cache layers once and then
never touches the network again.

## One-time setup

1. `Runtime -> Change runtime type -> T4 GPU` (the free tier is enough; the
   pipeline detects the GPU itself, no code changes needed).
2. Upload `expert-press-update.zip` to the **root of Google Drive**.
3. Open the notebook in Colab and run the cells top to bottom.

The notebook is plain Python cells around the usual command —
`python hf_pipeline.py --gguf ... --rank 32 --device auto`. No adaptation.

## What is cached, where

| thing | size | without caching | with the notebook |
|---|---|---|---|
| GGUF source | ~4.4 GB | re-downloaded every session | once into `Drive/moe_router/hf_cache` (`HF_HOME`); then copied Drive -> `/content` per session (~1-2 min) |
| light catalog (config/tokenizer) | MBs | re-fetched | stays in the same HF cache |
| calibration pool (stages 3-4) | ~2.3 GB | re-streamed (~tens of min) | synced to `Drive/moe_router/pool_cache`; stages 1-4 are skipped entirely on re-runs |
| artifact (`field_*_r*`) | ~1.2 GB | **lost** on session reset | synced to `Drive/moe_router/artifacts` |
| corpus texts | ~6 MB | re-fetched | negligible, local |

The run directory itself (`MOE_OUT_DIR=/content/run`) stays on the **local**
disk: a Drive FUSE mount turns every small read into a network round-trip
(see [Memory-and-Speed](Memory-and-Speed.md)). Only finished results are
synced back.

## The GPU: what turns on automatically

`--device auto` + CUDA present -> the `high` hardware profile (`--profile
auto`):

- **GPU streaming**: the backbone and every expert block live on cuda:0
  during their layer's pass (stages 3-4, 7);
- **dtype fp16 on T4** (sm_75): bf16 has no native T4 support — the profile
  flips `auto -> float16` and prints it in the `[profile]` line. Ampere+ GPUs
  keep bf16;
- **io-cache ram** (if free RAM fits the packed GGUF x2, min 4 GB headroom)
  and **io-threads 4**;
- **calib-bsz 16** on GPU — fewer forward calls, fewer block loads in the
  streaming pass (pairs are position-independent vectors, batch size does
  not change the pool).

On a weak PC (no CUDA, little RAM) `auto` resolves to `low` — exactly the
historical conservative behavior. `--profile low` forces it, `--profile
high` forces the boosts, explicit `--io-cache/--io-threads` always win.

## Typical session (second run onwards)

```
cell 1: mount Drive, unzip (cached), GPU check
cell 3: "GGUF готов: ... (4.42 GB)"          <- copied from Drive, no network
cell 4: "восстанавливаю пул калибровки"      <- stages 1-4 will be skipped
cell 5: python hf_pipeline.py --gguf ...     <- only fit/save/verify actually run
cell 6: pool + artifact -> Drive
```

A new rank (`RANK = 64`) from the same pool skips stages 1-4 as well — only
the fit re-runs.

## Troubleshooting

- **OOM (12.7 GB RAM)** — pass `--io-cache disk` explicitly; the auto
  choice downgrades itself when the packed file does not fit, but an
  explicit flag is a hard guarantee.
- **`--io-cache ram` warning "is tight"** — you forced it explicitly; the
  run continues but may hit the commit limit.
- **Chat with the artifact** — `!python hf_chat.py --model {ART} --device auto`.
- The per-page CLI details live in [CLI-Reference](CLI-Reference.md).
