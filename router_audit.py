#!/usr/bin/env python3
"""Router audit for field-compressed MoE artifacts (diagnostic only, no changes).

Answers two questions with data instead of guesswork:

  A) "Is the router confusing experts?" -- the artifact keeps the ORIGINAL base
     router, but its input (hidden states) drifts as layers accumulate error.
     We run the BASE model and the FIELD artifact on the same token windows and
     compare, per MoE layer: top-k expert agreement (Jaccard), routing-vector
     cosine, input drift, and block output error (rel-MSE, cosine).

  B) "Do errors grow with depth?" -- the same per-layer table shows exactly
     where the error lives (Task: depth vs error).

Artifact-only mode (no --base): routing statistics of the artifact itself
(expert load balance, score entropy, dead c(z) coordinates) plus a
counterfactual test: shuffle the routing vector z of one layer at a time and
measure the CE jump. A large jump = the field really uses that layer's
routing; a tiny jump = routing there is ignored (or already broken).

Usage:
  python3 router_audit.py --artifact results/field_xxx_r32 --base path/to/base
  python3 router_audit.py --artifact results/field_xxx_r32            # stats+scramble
  python3 router_audit.py --artifact ... --base model.Q4_K_M.gguf     # gguf auto-convert

--base accepts: HF checkpoint dir, or a .gguf file (converted to results/gguf_hf/,
reused if it already exists).
"""
import argparse
import json
import math
import os
import sys

import hf_env  # noqa: F401  -- HF cache inside the project; BEFORE transformers

import torch  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_base(path, dtype, device):
    """Base model from an HF dir or a .gguf file (converted on first use)."""
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


def load_artifact(path, dtype, device):
    from transformers import AutoModelForCausalLM
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if (cfg.get("quantization_config") or {}).get("quant_method"):
        sys.exit("Q4-quantized artifact needs CUDA; rebuild with --save-backbone bf16")
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype,
                                             trust_remote_code=True,
                                             low_cpu_mem_usage=True)
    return m.to(device).eval()


def moe_blocks(model, field=False):
    """[(name, block)] MoE blocks; field blocks carry Cgu, base blocks experts."""
    out = []
    for n, m in model.named_modules():
        if field and hasattr(m, "gate") and hasattr(m, "Cgu"):
            out.append((n, m))
        elif not field and hasattr(m, "experts") and hasattr(m, "gate"):
            out.append((n, m))
    return out


def gate_cfg(model, field):
    c = model.config
    if field:
        fi = c.field
        return int(fi["top_k"]), bool(fi.get("norm_topk", False))
    return int(getattr(c, "num_experts_per_tok", 2)), bool(getattr(c, "norm_topk_prob", False))


def route_from_gate_out(gout, k, norm):
    """(logits, scores, idx) from a router output (v5 tuple or plain logits)."""
    import torch.nn.functional as F
    if isinstance(gout, (tuple, list)):
        return gout[0], gout[1], gout[2]
    logits = gout
    probs = F.softmax(logits.float(), dim=-1)
    scores, idx = torch.topk(probs, k, dim=-1)
    if norm:
        scores = scores / scores.sum(-1, keepdim=True)
    return logits, scores, idx


class Rec:
    """Forward hooks: block input/output + router output, per layer."""
    def __init__(self, blocks, store):
        self.hs = []
        for i, (_, b) in enumerate(blocks):
            self.hs.append(b.register_forward_hook(self._mk(i, store)))
            self.hs.append(b.gate.register_forward_hook(self._gate(i, store)))

    def _mk(self, i, store):
        def hook(m, args, output):
            y = output if torch.is_tensor(output) else output[0]
            store[i]["y"] = y.detach().reshape(-1, y.shape[-1])
            x = args[0]
            store[i]["x"] = x.detach().reshape(-1, x.shape[-1])
        return hook

    def _gate(self, i, store):
        def hook(m, args, output):
            store[i]["g"] = output
        return hook

    def remove(self):
        for h in self.hs:
            h.remove()


def metrics_pairs(sb, sf, k, norm, acc, L):
    """Accumulate base-vs-field per-layer metrics from one window."""
    import torch.nn.functional as F
    for i in range(L):
        zb_l, zs_l, zi_l = route_from_gate_out(sb[i]["g"], k, norm)
        zf_l, zs_f, zi_f = route_from_gate_out(sf[i]["g"], k, norm)
        xb, yb = sb[i]["x"].float(), sb[i]["y"].float()
        xf, yf = sf[i]["x"].float(), sf[i]["y"].float()
        T = min(xb.shape[0], xf.shape[0])
        zb_l, zf_l = zb_l[:T], zf_l[:T]
        zi_l, zi_f = zi_l[:T], zi_f[:T]
        zs_l, zs_f = zs_l[:T], zs_f[:T]
        xb, xf, yb, yf = xb[:T], xf[:T], yb[:T], yf[:T]
        # top-k agreement (Jaccard, per token -> mean)
        j, cos_z = [], []
        for t in range(T):
            a, b = set(zi_l[t].tolist()), set(zi_f[t].tolist())
            j.append(len(a & b) / max(1, len(a | b)))
        acc["agree"][i] += sum(j) / len(j)
        zt_l = torch.zeros_like(zb_l).scatter_(-1, zi_l, zs_l)
        zt_f = torch.zeros_like(zf_l).scatter_(-1, zi_f, zs_f)
        cos_z = F.cosine_similarity(zt_l, zt_f, dim=-1, eps=1e-6).mean()
        acc["zcos"][i] += float(cos_z)
        acc["indr"][i] += float((xf - xb).norm() / xb.norm().clamp_min(1e-9))
        d = yf - yb
        acc["rmse"][i] += float((d ** 2).sum() / (yb ** 2).sum().clamp_min(1e-12))
        cc = F.cosine_similarity(yf, yb, dim=-1, eps=1e-6).mean()
        acc["ocos"][i] += float(cc)
        acc["yvar"][i] += float(yb.var())
    acc["n"] += 1


def stats_field(sf, k, norm, acc, L, N):
    """Artifact-only routing statistics per layer."""
    for i in range(L):
        gout = sf[i]["g"]
        logits = gout[0] if isinstance(gout, (tuple, list)) else gout
        _, scores, idx = route_from_gate_out(gout, k, norm)
        cnt = torch.bincount(idx.reshape(-1), minlength=N).float()
        p = cnt / cnt.sum().clamp_min(1)
        p = p[p > 0]
        acc["loadent"][i] += float(-(p * p.log()).sum() / math.log(N))
        acc["top1"][i] += float(scores[:, 0].mean())
        lp = torch.log_softmax(logits.float(), dim=-1)
        ent = -(lp.exp() * lp).sum(-1)
        top2 = lp.sort(-1, descending=True).values
        acc["ent"][i] += float(ent.mean())
        acc["t1m"][i] += float((top2[:, 0] - top2[:, 1]).mean())
        acc["n"] += 1


class Scrambler:
    """Temporarily override block.gate.forward so its token rows are permuted
    -> each token receives another token's routing vector z (counterfactual)."""
    def __init__(self, block, gen):
        self.target = block.gate
        self.saved_fwd = self.target.forward
        self.gen = gen

        def scrambled(x, _fwd=self.saved_fwd, _gen=gen):
            out = _fwd(x)
            perm = torch.randint(0, x.shape[0], (x.shape[0],), generator=_gen)
            perm = perm.to(x.device)
            if isinstance(out, (tuple, list)):
                return [o.index_select(0, perm) if torch.is_tensor(o) else o
                        for o in out]
            return out.index_select(0, perm)
        self.target.forward = scrambled

    def restore(self):
        self.target.forward = self.saved_fwd


def ce_of(model, ids, device):
    import torch.nn.functional as F
    with torch.no_grad():
        lg = model(input_ids=ids.to(device)).logits[0, :-1].float()
        tgt = ids[0, 1:].to(device)
        return float(F.cross_entropy(lg, tgt)) / math.log(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True, help="field artifact dir")
    ap.add_argument("--base", default=None,
                    help="base model: HF dir or .gguf (enables drift/depth section)")
    ap.add_argument("--text", default=None, help="text file (default corpus.txt)")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--windows", type=int, default=8, help="windows for drift stats")
    ap.add_argument("--scramble-windows", type=int, default=4)
    ap.add_argument("--no-scramble", action="store_true")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON output path")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    dt = getattr(torch, a.dtype)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" \
        else a.device

    text = a.text or os.path.join(BASE, "corpus.txt")
    if not os.path.isfile(text):
        sys.exit(f"no text file: {text} (pass --text)")
    ids_all = torch.tensor(
        __import__("transformers").AutoTokenizer.from_pretrained(a.artifact)(
            open(text, encoding="utf-8", errors="ignore").read())["input_ids"])
    eprint(f"tokens: {len(ids_all)} | device {dev} | dtype {a.dtype}")

    art = load_artifact(a.artifact, dt, dev)
    fblocks = moe_blocks(art, field=True)
    L = len(fblocks)
    if not L:
        sys.exit("no field blocks found in artifact")
    k, norm = gate_cfg(art, field=True)
    N = int(art.config.field["n_exp"])
    tag = os.path.basename(a.artifact.rstrip("/"))
    out_path = a.out or os.path.join(BASE, "results", f"router_audit_{tag}.json")
    res = {"artifact": a.artifact, "base": a.base, "k": k, "n_exp": N,
           "layers": L, "ctx": a.ctx}

    # ---------- phase 1: base vs field per-layer drift / error ----------
    if a.base:
        eprint("phase 1: base vs field per-layer routing drift + block error")
        bmodel = load_base(a.base, dt, dev)
        bblocks = moe_blocks(bmodel, field=False)
        if len(bblocks) != L:
            sys.exit(f"base has {len(bblocks)} MoE blocks, artifact has {L}")
        acc = {key: [0.0] * L for key in ("agree", "zcos", "indr", "rmse",
                                          "ocos", "yvar")}
        acc["n"] = 0
        g = torch.Generator().manual_seed(17)
        for w in range(a.windows):
            s = int(torch.randint(0, len(ids_all) - a.ctx - 1, (1,), generator=g))
            ids = ids_all[s:s + a.ctx].unsqueeze(0)
            sb = [dict() for _ in range(L)]
            sf = [dict() for _ in range(L)]
            rb, rf = Rec(bblocks, sb), Rec(fblocks, sf)
            with torch.no_grad():
                bmodel(input_ids=ids.to(dev))
                art(input_ids=ids.to(dev))
            rb.remove(); rf.remove()
            metrics_pairs(sb, sf, k, norm, acc, L)
            eprint(f"  window {w + 1}/{a.windows}")
        n = max(1, acc.pop("n"))
        layers = []
        for i in range(L):
            layers.append(dict(
                layer=i, agree=acc["agree"][i] / n, z_cosine=acc["zcos"][i] / n,
                input_drift=acc["indr"][i] / n, out_rel_mse=acc["rmse"][i] / n,
                out_cosine=acc["ocos"][i] / n))
        res["drift"] = layers
        print("\n== base vs field, per MoE layer (same tokens) ==")
        print("layer | topk agree | z-cos | in-drift | out relMSE | out cos")
        for d in layers:
            print(f"{d['layer']:5d} | {d['agree']:10.3f} | {d['z_cosine']:5.3f} | "
                  f"{d['input_drift']:8.4f} | {d['out_rel_mse']:10.4f} | "
                  f"{d['out_cosine']:7.4f}")
        agree = [d["agree"] for d in layers]
        rmse = [d["out_rel_mse"] for d in layers]
        print(f"mean topk agreement: {sum(agree) / L:.3f} | "
              f"min: {min(agree):.3f} (layer {agree.index(min(agree))})")
        half = L // 2 or 1
        print(f"depth trend out_relMSE: first half {sum(rmse[:half]) / half:.4f} "
              f"vs second half {sum(rmse[half:]) / (L - half):.4f}")
        del bmodel
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---------- phase 2: artifact-only routing stats ----------
    eprint("phase 2: artifact routing stats (load balance, entropy)")
    acc2 = {key: [0.0] * L for key in ("loadent", "top1", "ent", "t1m")}
    acc2["n"] = 0
    g = torch.Generator().manual_seed(19)
    for w in range(min(a.windows, 4)):
        s = int(torch.randint(0, len(ids_all) - a.ctx - 1, (1,), generator=g))
        ids = ids_all[s:s + a.ctx].unsqueeze(0)
        sf = [dict() for _ in range(L)]
        rf = Rec(fblocks, sf)
        with torch.no_grad():
            art(input_ids=ids.to(dev))
        rf.remove()
        stats_field(sf, k, norm, acc2, L, N)
    n2 = max(1, acc2.pop("n"))
    res["stats"] = [dict(layer=i, load_balance=acc2["loadent"][i] / n2,
                         top1_score=acc2["top1"][i] / n2,
                         logits_entropy=acc2["ent"][i] / n2,
                         top1_margin=acc2["t1m"][i] / n2) for i in range(L)]
    print("\n== artifact routing stats ==")
    print("layer | load-balance | top1 score | logit entropy | top1 margin")
    for d in res["stats"]:
        print(f"{d['layer']:5d} | {d['load_balance']:12.3f} | "
              f"{d['top1_score']:10.4f} | {d['logits_entropy']:13.3f} | "
              f"{d['top1_margin']:11.3f}")
    lb = [d["load_balance"] for d in res["stats"]]
    print(f"load-balance: 1.0 = uniform usage; worst layer "
          f"{min(lb):.3f} (layer {lb.index(min(lb))})")

    # ---------- phase 3: counterfactual z-scramble per layer ----------
    if not a.no_scramble:
        eprint("phase 3: counterfactual z-scramble per layer")
        g = torch.Generator().manual_seed(23)
        wins = []
        for w in range(a.scramble_windows):
            s = int(torch.randint(0, len(ids_all) - a.ctx - 1, (1,), generator=g))
            wins.append(ids_all[s:s + a.ctx].unsqueeze(0))
        base_ce = sum(ce_of(art, ids, dev) for ids in wins) / len(wins)
        dces = []
        for i in range(L):
            sc = Scrambler(fblocks[i][1], g)
            ce = sum(ce_of(art, ids, dev) for ids in wins) / len(wins)
            sc.restore()
            dces.append(ce - base_ce)
            eprint(f"  layer {i}: dCE {ce - base_ce:+.4f} bits/token")
        allsc = [Scrambler(b, g) for _, b in fblocks]
        ce_all = sum(ce_of(art, ids, dev) for ids in wins) / len(wins)
        for sc in allsc:
            sc.restore()
        res["scramble"] = dict(base_ce_bits=base_ce, dce_bits=dces,
                               all_layers_dce=ce_all - base_ce)
        print(f"\n== z-scramble counterfactual (CE bits/token vs normal) ==")
        print(f"normal CE {base_ce:.4f} | all layers scrambled "
              f"{res['scramble']['all_layers_dce']:+.4f}")
        worst = sorted(range(L), key=lambda i: -dces[i])[:5]
        print("top-5 routing-sensitive layers:",
              ", ".join(f"L{i} {dces[i]:+.3f}" for i in worst))
        quiet = sum(1 for d in dces if d < 0.01)
        print(f"layers with <0.01 bits impact (routing ignored): {quiet}/{L}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
