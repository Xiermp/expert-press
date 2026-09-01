"""Training TinyMoE (the base, uncompressed model).
Run: python3 train.py --steps 1200   |   python3 train.py --smoke (40-step timing)
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from common import CFG, CKPT, TinyMoE, get_batch, prepare_data


@torch.no_grad()
def eval_ppl(model, val_ids, batches=150):
    model.eval()
    g = torch.Generator().manual_seed(7)
    total, n = 0.0, 0
    for _ in range(batches):
        x, y = get_batch(val_ids, 16, CFG["ctx"], gen=g)
        logits, _ = model(x, None)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        n += 1
    model.train()
    return total / n


@torch.no_grad()
def expert_usage(model, val_ids):
    g = torch.Generator().manual_seed(9)
    x, _ = get_batch(val_ids, 32, CFG["ctx"], gen=g)
    probs_out = []
    emb = model.emb(x)
    h = emb
    for b in model.blocks:
        h = b(h, probs_out)
    rows = []
    for li, p in enumerate(probs_out):
        f = F.one_hot(p.argmax(-1), p.size(-1)).float().mean((0, 1))
        rows.append([round(float(v), 3) for v in f])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1337)
    torch.set_num_threads(os.cpu_count() or 2)

    train_ids, val_ids, itos, _ = prepare_data()
    model = TinyMoE(CFG, len(itos))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.2f}M vocab={len(itos)} "
          f"train={len(train_ids)} val={len(val_ids)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                            betas=(0.9, 0.95), weight_decay=0.01)
    warmup, total = 100, args.steps

    def lr_at(s):
        if s < warmup:
            return 3e-3 * (s + 1) / warmup
        t = (s - warmup) / max(1, total - warmup)
        return 3e-4 + 0.5 * (3e-3 - 3e-4) * (1 + math.cos(math.pi * t))

    t0 = time.time()
    model.train()
    for s in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = get_batch(train_ids, 16, CFG["ctx"])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 50 == 0 or s == total - 1:
            el = time.time() - t0
            eta = el / (s + 1) * (total - s - 1)
            print(f"step {s:4d} loss {loss.item():.3f} "
                  f"{el / (s + 1):.2f}s/step eta {eta:.0f}s", flush=True)
        if args.smoke and s >= 40:
            break

    val_ce = eval_ppl(model, val_ids)
    val_ppl = math.exp(val_ce)
    usage = expert_usage(model, val_ids)
    print(f"VAL ce={val_ce:.3f} ppl={val_ppl:.2f}", flush=True)
    for li, row in enumerate(usage):
        print(f"layer {li} expert usage: {row}", flush=True)

    if not args.smoke:
        torch.save({"sd": model.state_dict(), "cfg": CFG, "itos": itos,
                    "val_ppl": val_ppl}, CKPT)
        print(f"saved -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
