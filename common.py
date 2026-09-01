"""TinyMoE PoC: config, data, model. Experiment: base + SVD deltas instead of
experts."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "ckpt.pt")

CFG = dict(
    n_layer=4, n_head=4, d_model=128, d_ff=192,
    n_exp=8, top_k=2, ctx=64, aux_coef=0.05,
)

FALLBACK_TEXT = (
    "The town woke early. Fog hung over the river, and the ferry had not yet "
    "started running. Maria opened her shop, set the bread out on the shelves "
    "and looked out of the window. The rain of the previous night had washed "
    "the dust off the pavement, and the stones shone like copper coins. "
    "Old Simon sat by the fountain cutting an apple into thin slices. "
    "Children with schoolbags ran past, shouted something cheerful and "
    "vanished around the corner. The clock on the tower struck eight. "
    "The town sighed and went to work. "
) * 60


def _strip_gutenberg(raw: str) -> str:
    up = raw.upper()
    marker_start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    marker_end = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if marker_start in up:
        a = raw.index("\n", up.index(marker_start)) + 1
        b = raw.rindex("\n", 0, up.index(marker_end)) if marker_end in up else len(raw)
        return raw[a:b]
    return raw


def _read_corpus() -> str:
    cache = os.path.join(BASE, "corpus.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return f.read()
    text = ""
    ru_path = os.path.join(BASE, "corpus_ru.txt")
    raw_path = os.path.join(BASE, "corpus_raw.txt")
    for path in (ru_path, raw_path):
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            body = _strip_gutenberg(raw)
            sample = body[10000:30000]
            cyr = sum(1 for ch in sample if "\u0400" <= ch <= "\u04FF")
            if len(body) > 300_000 and (cyr / max(len(sample), 1) > 0.15 or path == raw_path):
                text = body
                break
    if len(text) < 100_000:
        text = FALLBACK_TEXT
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def build_vocab(text: str, min_freq: int = 20):
    from collections import Counter
    cnt = Counter(text)
    itos = sorted(ch for ch, c in cnt.items() if c >= min_freq)
    stoi = {ch: i for i, ch in enumerate(itos)}
    unk = stoi.get(" ", 0)
    encode = lambda s: [stoi.get(ch, unk) for ch in s]
    return itos, stoi, encode


def prepare_data():
    text = _read_corpus()
    itos, stoi, encode = build_vocab(text)
    ids = encode(text)
    n_val = min(20_000, len(ids) // 10)
    return ids[:-n_val], ids[-n_val:], itos, stoi


def prepare_data3(val_tokens: int = 20_000, calib_tokens: int = 128_000):
    """Three NON-overlapping segments: train | calib | val.

    val (eval) - the last val_tokens, exactly the same eval segment as in all
    earlier runs (numbers are directly comparable). calib - a separate segment
    immediately BEFORE val: activation-aware methods (field, dense-MLP,
    whitening, cascade) take their calibration activations from here and only
    here. train - everything before calib (base model training).

    Leak fix: collect_moe_pairs/collect_routed_inputs used to sample straight
    from val_ids - the very segment KL/ppl was later measured on - and the
    field was tuned by Adam on the statistics of its own eval set. calib lies
    inside the base train distribution (like GPTQ calibration) - acceptable,
    since both models (base and compressed) share one backbone; the critical
    hygiene is calib ∩ val = ∅, which the split geometry guarantees."""
    text = _read_corpus()
    itos, stoi, encode = build_vocab(text)
    ids = encode(text)
    n_val = min(val_tokens, len(ids) // 10)
    n_cal = min(calib_tokens, max(len(ids) - n_val - 1000, len(ids) // 20))
    val = ids[len(ids) - n_val:]
    calib = ids[len(ids) - n_val - n_cal: len(ids) - n_val]
    train = ids[: len(ids) - n_val - n_cal]
    return train, calib, val, itos, stoi


class MoEFFN(nn.Module):
    """Mixtral-style MoE FFN: top-k experts, mixing weights normalized."""

    def __init__(self, d: int, d_ff: int, n_exp: int, top_k: int):
        super().__init__()
        self.n_exp, self.top_k = n_exp, top_k
        self.router = nn.Linear(d, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_exp, d_ff, d))
        self.w2 = nn.Parameter(torch.empty(n_exp, d, d_ff))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        probs = F.softmax(self.router(x), dim=-1)              # (B,T,N)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)     # (B,T,k)
        topw = topw / topw.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.n_exp):
            sel = (topi == e).float()
            w = (sel * topw).sum(-1)                            # (B,T)
            if w.detach().abs().sum() == 0.0:
                continue
            h = F.gelu(x @ self.w1[e].t())
            y = h @ self.w2[e].t()
            out = out + y * w.unsqueeze(-1)
        return out, probs


class Block(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        d, h = c["d_model"], c["n_head"]
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.moe = MoEFFN(d, c["d_ff"], c["n_exp"], c["top_k"])

    def forward(self, x, probs_out):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.view(B, T, self.h, d // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, d // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, d // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(y)
        moe_y, probs = self.moe(self.ln2(x))
        probs_out.append(probs)
        return x + moe_y


class TinyMoE(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.emb = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        probs_out = []
        for b in self.blocks:
            x = b(x, probs_out)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux = 0.0
            for p in probs_out:                                   # (B,T,N)
                f = F.one_hot(p.argmax(-1), p.size(-1)).float().mean((0, 1))
                pe = p.mean((0, 1))
                aux = aux + float(self.cfg["n_exp"]) * float((f * pe).sum())
            loss = loss + self.cfg["aux_coef"] * aux / len(probs_out)
        return logits, loss


def get_batch(ids, batch: int, ctx: int, gen=None):
    ix = torch.randint(0, len(ids) - ctx - 1, (batch,), generator=gen)
    x = torch.stack([torch.tensor(ids[i:i + ctx]) for i in ix])
    y = torch.stack([torch.tensor(ids[i + 1:i + ctx + 1]) for i in ix])
    return x, y
