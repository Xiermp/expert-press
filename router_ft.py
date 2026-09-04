# version: 2026-09-04.1 - MEMORY rework: the artifact loads in bf16 (was
#   fp32: 2x RAM) and the whole model is frozen except fp32 MASTER copies of
#   the gate weights (GateMaster via torch.func.functional_call - the router
#   math stays bit-identical on any architecture, gradients reach the master
#   through the bf16 cast). Old code also left every backbone param with
#   requires_grad=True -> the backward graph materialized grads for the
#   WHOLE model. Net RAM: ~2.8x less on the tuning pass. Added --inplace
#   (update the artifact's gate weights in place; the default still writes a
#   separate <artifact>_rft dir).
#!/usr/bin/env python3
"""Post-hoc router (gate) calibration for field-compressed MoE artifacts.

Why: the artifact keeps the ORIGINAL base router, but the field model's hidden
states drift with depth, so the same router now maps slightly different inputs
to routing decisions. This script nudges ONLY the gate weights of the field
model (everything else frozen) to minimize KL(base || field) on calibration
text -- a cheap, surgical "re-align the router" pass.

What it does:
  1. loads the artifact in bf16 with fp32 MASTER gate weights (trainable;
     the router math runs through the original gate class unchanged) and the
     base model (bf16, no grad);
  2. measures CE/KL before tuning on held-out windows;
  3. takes `--steps` gentle Adam steps on the gate masters only, loss =
     KL(base||field) + `--anchor` * mean ||W-W0||^2/||W0||^2 (stays close to
     the original router);
  4. measures CE/KL after; aborts the save if quality got worse;
  5. writes the tuned gate weights into a NEW artifact dir (<artifact>_rft
     by default) or into the artifact itself (--inplace).

Expectation setting (toy bench, 2026-09-04): at the BLOCK level the router
is usually NOT the post-fit bottleneck - the field's z-dependent correction
carries a small share of the output energy, so even a 30% top-k set change
moved the block mse by ~0. This script works at the LM level (hidden-state
drift across layers), where the effect can be larger - but treat it as a
final polish, not a quality lever; --refine-rounds in hf_pipeline.py is the
stronger tool for the post-conversion gaps.

Memory: the artifact loads in bf16 (fp32 was 2x) and ONLY the gate masters
(n_exp x d fp32 per layer, ~0.5 MB each) train; the base stays bf16. On
OLMoE-1B-7B that is ~14 GB (artifact) + ~14 GB (base) instead of ~28 + 14.

Usage:
  python3 router_ft.py --artifact results/field_xxx_r128 --base base.gguf
  python3 router_ft.py --artifact ... --base ... --steps 60 --lr 5e-5 --dry-run
"""
import argparse
import json
import math
import os
import shutil
import sys

import hf_env  # noqa: F401  -- HF cache inside the project; BEFORE transformers

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


class GateMaster(nn.Module):
    """fp32 MASTER copy of a gate weight inside a bf16 model.

    forward re-runs the ORIGINAL gate module with the bf16 cast of the
    master via torch.func.functional_call: the router math stays exactly
    the original's (any architecture / contract), and gradients flow back
    into the fp32 master through the cast."""

    def __init__(self, orig_gate):
        super().__init__()
        self.orig = orig_gate
        for q in self.orig.parameters():
            q.requires_grad_(False)
        w = self.orig.weight
        self.out_dtype = w.dtype
        self.register_parameter("w", nn.Parameter(w.detach().float().clone()))

    def forward(self, x):
        w_cast = self.w.to(self.out_dtype)
        return torch.func.functional_call(self.orig, {"weight": w_cast}, (x,))


def load_base(path, dtype, device):
    from transformers import AutoModelForCausalLM
    if path.endswith(".gguf"):
        out = os.path.join(BASE, "results", "gguf_hf",
                           os.path.basename(path).removesuffix(".gguf") + "-hf")
        if not os.path.isfile(os.path.join(out, "config.json")):
            import hf_gguf_to_hf as g2h
            eprint(f"converting GGUF -> {out} (one-time)...")
            try:
                g2h.convert(path, out, dtype="float16", base_repo=None)
            except TypeError:
                g2h.convert(path, out, dtype="float16")
        path = out
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype,
                                             low_cpu_mem_usage=True)
    return m.to(device).eval()


def load_artifact_train(path, device):
    """Artifact in bf16 (memory!), gate weights replaced by fp32 masters,
    everything except the masters frozen. trust_remote_code."""
    from transformers import AutoModelForCausalLM
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if (cfg.get("quantization_config") or {}).get("quant_method"):
        sys.exit("Q4-quantized artifact is not supported here; "
                 "rebuild with --save-backbone bf16")
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             low_cpu_mem_usage=True)
    m.to(device).train()
    # freeze EVERYTHING: the old fp32 path left the whole backbone with
    # requires_grad=True, so backward built a grad buffer per parameter
    # (RAM + time for nothing)
    for q in m.parameters():
        q.requires_grad_(False)
    n_gates = 0
    for name, mod in list(m.named_modules()):
        if hasattr(mod, "gate") and hasattr(mod, "Cgu"):
            mod.gate = GateMaster(mod.gate)
            n_gates += 1
    if not n_gates:
        sys.exit("no field blocks found")
    eprint(f"artifact bf16 + {n_gates} fp32 gate masters (trainable)")
    return m


def field_blocks(model):
    return [(n, m) for n, m in model.named_modules()
            if hasattr(m, "gate") and hasattr(m, "Cgu")
            and isinstance(m.gate, GateMaster)]


def windows(ids_all, ctx, n, seed, device):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        s = int(torch.randint(0, len(ids_all) - ctx - 1, (1,), generator=g))
        out.append(ids_all[s:s + ctx].unsqueeze(0).to(device))
    return out


@torch.no_grad()
def eval_ce_kl(art, bmodel, wins):
    """CE bits/token of the artifact + KL(base||field) over windows."""
    ces, kls = [], []
    for ids in wins:
        lf = art(input_ids=ids).logits[0, :-1].float()
        lb = bmodel(input_ids=ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        ces.append(float(F.cross_entropy(lf, tgt)) / math.log(2))
        pb = F.log_softmax(lb, -1)
        pf = F.log_softmax(lf, -1)
        kls.append(float((pb.exp() * (pb - pf)).sum(-1).mean()) / math.log(2))
    return sum(ces) / len(ces), sum(kls) / len(kls)


def write_gates_into_dir(out_dir, gate_new):
    """Rewrite the gate tensors inside an EXISTING artifact dir (safetensors
    only, shard-by-shard through a .tmp file + atomic replace)."""
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            wmap = json.load(f)["weight_map"]
        shards = sorted(set(wmap.values()))
    else:
        shards = sorted(f for f in os.listdir(out_dir)
                        if f.endswith(".safetensors"))
    from safetensors.torch import load_file, save_file
    done, written_keys = 0, set()
    for shard in shards:
        sp = os.path.join(out_dir, shard)
        sd = load_file(sp)
        touched = [k for k in sd if k in gate_new]
        if not touched:
            continue
        for k in touched:
            sd[k] = gate_new[k]
            written_keys.add(k)
        tmp = sp + ".rft_tmp"
        save_file(sd, tmp, metadata={"format": "pt"})
        os.replace(tmp, sp)
        done += len(touched)
        eprint(f"  {shard}: updated {len(touched)} gate tensor(s)")
    if done < len(gate_new):
        missing = sorted(set(gate_new) - written_keys)
        eprint(f"warning: only {done}/{len(gate_new)} gate tensors rewritten; "
               f"missing keys like {missing[:2]}")
    return done


def save_gates(artifact_dir, out_dir, blocks, model, inplace=False):
    """Write the tuned gate weights either into a copied dir (default) or
    into the artifact itself (--inplace)."""
    gate_new = {}
    for (name, m), p in zip(blocks, model.gate_params):
        key = f"{name}.gate.weight"
        gate_new[key] = p.detach().to(m.gate.out_dtype).cpu()
    if inplace:
        eprint(f"updating gate weights IN PLACE in {artifact_dir}")
        write_gates_into_dir(artifact_dir, gate_new)
        out_dir = artifact_dir
    else:
        if os.path.exists(out_dir):
            sys.exit(f"refusing to overwrite existing dir: {out_dir}")
        eprint(f"copying artifact -> {out_dir}")
        shutil.copytree(artifact_dir, out_dir)
        write_gates_into_dir(out_dir, gate_new)
    # note in field_meta.json
    mp = os.path.join(out_dir, "field_meta.json")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        meta["router_ft"] = model.ft_info
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--base", required=True, help="HF dir or .gguf")
    ap.add_argument("--text", default=None)
    ap.add_argument("--out", default=None, help="output artifact dir")
    ap.add_argument("--steps", type=int, default=40, help="Adam steps (1 window each)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="weight of ||W-W0||^2 penalty (0 = free router)")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--eval-windows", type=int, default=6)
    ap.add_argument("--inplace", action="store_true",
                    help="rewrite the gate weights inside the artifact itself "
                         "instead of copying it to <artifact>_rft (saves disk; "
                         "the original gate weights are NOT backed up - keep "
                         "the pipeline cache if you may want to revert)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure before/after potential without saving: "
                         "runs --steps steps in RAM and discards them")
    ap.add_argument("--force-save", action="store_true",
                    help="save even if metrics did not improve (experiments)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" \
        else a.device

    from transformers import AutoTokenizer
    text = a.text or os.path.join(BASE, "corpus.txt")
    ids_all = torch.tensor(AutoTokenizer.from_pretrained(a.artifact)(
        open(text, encoding="utf-8", errors="ignore").read())["input_ids"])
    eprint(f"tokens: {len(ids_all)} | device {dev}")

    eprint("loading artifact (bf16, fp32 gate masters)...")
    art = load_artifact_train(a.artifact, dev)
    blocks = field_blocks(art)
    L = len(blocks)
    if not L:
        sys.exit("no field blocks found")
    eprint("loading base (targets, no grad)...")
    bmodel = load_base(a.base, torch.bfloat16, dev)

    # gate masters -> trainable
    params, W0 = [], []
    for _, m in blocks:
        params.append(m.gate.w)
        W0.append(m.gate.w.detach().float().clone())
    art.gate_params = params
    opt = torch.optim.Adam(params, lr=a.lr)

    train_wins = windows(ids_all, a.ctx, a.steps, seed=31, device=dev)
    eval_wins = windows(ids_all, a.ctx, a.eval_windows, seed=77, device=dev)

    ce0, kl0 = eval_ce_kl(art, bmodel, eval_wins)
    print(f"\nBEFORE: CE {ce0:.4f} bits/token | KL(base||field) {kl0:.4f} bits",
          flush=True)

    for step, ids in enumerate(train_wins):
        opt.zero_grad(set_to_none=True)
        lf = art(input_ids=ids).logits[0, :-1].float()
        with torch.no_grad():
            lb = bmodel(input_ids=ids).logits[0, :-1].float()
        pb = F.log_softmax(lb, -1)
        pf = F.log_softmax(lf, -1)
        kl = (pb.exp() * (pb - pf)).sum(-1).mean()
        anch = 0.0
        if a.anchor > 0:
            anch = sum(((w - w0) ** 2).sum() / (w0 ** 2).sum().clamp_min(1e-12)
                       for w, w0 in zip(params, W0)) / L
        loss = kl + a.anchor * anch
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == a.steps - 1:
            print(f"  step {step}: KL {float(kl.detach()) / math.log(2):.4f} bits, "
                  f"anchor {float(anch.detach() if torch.is_tensor(anch) else anch):.2e}",
                  flush=True)

    ce1, kl1 = eval_ce_kl(art, bmodel, eval_wins)
    drift = [float((w.detach().float() - w0).norm() / w0.norm().clamp_min(1e-12))
             for w, w0 in zip(params, W0)]
    print(f"\nAFTER:  CE {ce1:.4f} bits/token | KL(base||field) {kl1:.4f} bits")
    print(f"gate drift (rel L2): mean {sum(drift) / L:.4f}, "
          f"max {max(drift):.4f} (layer {drift.index(max(drift))})")
    improved = (kl1 < kl0) and (ce1 <= ce0 + 1e-6)
    print(f"verdict: {'IMPROVED' if improved else 'NO GAIN'} "
          f"(dKL {(kl1 - kl0) * 100:+.2f}%, dCE {(ce1 - ce0) * 100:+.2f}%)")

    ft_info = dict(steps=a.steps, lr=a.lr, anchor=a.anchor, ctx=a.ctx,
                   mode="bf16+fp32-masters",
                   ce_before=ce0, ce_after=ce1, kl_before=kl0, kl_after=kl1,
                   gate_drift=drift,
                   date=__import__("time").strftime("%Y-%m-%d %H:%M"))
    art.ft_info = ft_info

    if a.dry_run:
        print("\n--dry-run: nothing saved.")
        return
    if not improved and not a.force_save:
        print("quality did not improve -> NOT saving. "
              "Try more --steps, other --lr, or lower --anchor.")
        return
    out = a.out or (a.artifact.rstrip("/") if a.inplace
                    else a.artifact.rstrip("/") + "_rft")
    save_gates(a.artifact, out, blocks, art, inplace=a.inplace)
    print(f"\nsaved -> {out}\nchat: python3 hf_chat.py --model {os.path.basename(out)}")


if __name__ == "__main__":
    main()
