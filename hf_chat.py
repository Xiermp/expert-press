#!/usr/bin/env python3
"""Chat with the converted model (the field-engine artifact) in a terminal.

After hf_pipeline.py the artifact is a normal HF model (experts replaced by
the field). This script loads it and lets you talk to it:

  python3 hf_chat.py                                   # finds the artifact in results/
  python3 hf_chat.py --model results/field_OLMoE-1B-7B-0924-GGUF_r32
  python3 hf_chat.py --prompt "Hi!"                    # single question, no dialog
  python3 hf_chat.py --temperature 0.8 --max-new 400   # generation settings
  python3 hf_chat.py --repetition-penalty 1.2          # stronger anti-loop

In-dialog commands:
  /help            - list commands
  /reset           - clear dialog history
  /system <text>   - set/replace the system prompt
  /temp <x>        - temperature (0 = greedy; default 0.7)
  /rep <x>         - repetition penalty (default 1.15; 1.0 = off). Compressed
                     models have slightly shifted logits and tend to fall
                     into repetition loops under plain decoding - the penalty
                     is the standard first-line fix
  /max <n>         - max new tokens per reply
  /exit            - quit

Note: OLMoE-1B-7B-0924 is a base (non-instruct) model. If the artifact has no
chat_template, the chat falls back to a "Question/Answer" format - coherent,
but replies match the base model style rather than an assistant.
"""
import argparse
import os
import sys
import time

import hf_env  # noqa: F401  - HF cache inside the project; BEFORE transformers

BASE = os.path.dirname(os.path.abspath(__file__))


def find_artifacts():
    """field_* artifact folders with config.json, newest first."""
    out = []
    for root in (os.path.join(BASE, "results"), BASE):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if name.startswith("field_") and os.path.isfile(os.path.join(p, "config.json")):
                out.append(p)
    return sorted(out, key=os.path.getmtime, reverse=True)


def pick_model(path):
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            sys.exit(f"no config.json in {path} - pass an artifact folder "
                     f"(the output of hf_pipeline.py)")
        return path
    cands = find_artifacts()
    if not cands:
        sys.exit("artifact not found: run hf_pipeline.py (compression) first, "
                 "then hf_chat.py --model <artifact folder>")
    print(f"artifact: {cands[0]}  (own path: --model <path>)", flush=True)
    return cands[0]


def load_model(path, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    tok = AutoTokenizer.from_pretrained(path)
    dt = {"auto": None, "bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}[dtype]
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        q = (json.load(f).get("quantization_config") or {}).get("quant_method")
    kw = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if q:  # Q4 artifact (bitsandbytes): GPU only
        if not torch.cuda.is_available():
            sys.exit("an artifact with a Q4 backbone requires CUDA; rebuild it "
                     "with --save-backbone bf16 or use a GGUF source")
        m = AutoModelForCausalLM.from_pretrained(
            path, device_map={"": 0}, **kw).eval()
        return m, tok, "cuda"
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if dev == "cpu" and dt is None:
        # bf16, same as in the artifact: half the RAM of fp32 (14 GB instead of
        # 28 for OLMoE-7B); switch back with --dtype float32
        dt = torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=dt or torch.bfloat16, **kw).to(dev).eval()
    return m, tok, dev


def build_prompt(tok, history, system):
    """History -> prompt string: chat template, else Question/Answer fallback."""
    if getattr(tok, "chat_template", None):
        msgs = ([{"role": "system", "content": system}] if system else []) + history
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    parts = [system] if system else []
    for m in history:
        who = "Question" if m["role"] == "user" else "Answer"
        parts.append(f"{who}: {m['content']}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def reply(model, tok, history, system, device, gen_cfg, stream=True):
    import torch
    prompt = build_prompt(tok, history, system)
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    n_in = enc["input_ids"].shape[1]

    budget = getattr(model.config, "max_position_embeddings", 4096) - gen_cfg["max_new"]
    while n_in > budget and len(history) > 1:      # drop older turns
        history.pop(0)
        prompt = build_prompt(tok, history, system)
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]

    do_sample = gen_cfg["temp"] > 0
    kwargs = dict(max_new_tokens=gen_cfg["max_new"], do_sample=do_sample,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if do_sample:
        kwargs.update(temperature=gen_cfg["temp"], top_p=gen_cfg["top_p"])
    # compressed models have slightly shifted logits -> plain decoding loops;
    # the standard repetition penalty is the first-line fix (default 1.15)
    if gen_cfg.get("rep", 1.0) != 1.0:
        kwargs["repetition_penalty"] = gen_cfg["rep"]
    streamer = None
    if stream:
        try:
            from transformers import TextStreamer
            streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            kwargs["streamer"] = streamer
        except ImportError:
            streamer = None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    dt = time.time() - t0
    text = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
    n_new = out.shape[1] - n_in
    return text, n_new, dt, streamer is not None


HELP = """commands: /help  /reset  /system <text>  /temp <x>  /rep <x>  /max <n>  /exit"""


def chat_loop(model, tok, device, gen_cfg, system):
    history = []
    print(HELP, flush=True)
    while True:
        try:
            user = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/reset":
                history = []
                print("(history cleared)", flush=True)
                continue
            if cmd == "/system":
                system = arg.strip()
                print(f"(system prompt: {system or '-'})", flush=True)
                continue
            if cmd == "/temp":
                gen_cfg["temp"] = float(arg or 0)
                print(f"(temperature={gen_cfg['temp']})", flush=True)
                continue
            if cmd == "/rep":
                gen_cfg["rep"] = float(arg or 1.0)
                print(f"(repetition penalty={gen_cfg['rep']})", flush=True)
                continue
            if cmd == "/max":
                gen_cfg["max_new"] = int(arg or gen_cfg["max_new"])
                print(f"(max_new_tokens={gen_cfg['max_new']})", flush=True)
                continue
            print(HELP, flush=True)
            continue
        history.append({"role": "user", "content": user})
        print("Model > ", end="", flush=True)
        text, n_new, dt, was_stream = reply(model, tok, history, system, device,
                                            gen_cfg)
        if not text:
            text = "(empty reply - try /temp 0 or keep going)"
        if not was_stream:
            print(text, flush=True)
        else:
            print(flush=True)
        print(f"[{n_new} tokens, {n_new / max(dt, 1e-9):.1f} tok/s]", flush=True)
        history.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="artifact folder after hf_pipeline.py (default: newest field_* in results/)")
    ap.add_argument("--prompt", default=None,
                    help="single question without dialog (print the reply and exit)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--system", default=None, help="system prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15,
                    help="repetition penalty (compressed models loop under "
                         "plain decoding; 1.0 = off)")
    ap.add_argument("--max-new", type=int, default=256,
                    help="max new tokens per reply")
    ap.add_argument("--no-stream", action="store_true", help="print the whole reply at once")
    a = ap.parse_args()

    path = pick_model(a.model)
    print("loading model...", flush=True)
    model, tok, device = load_model(path, a.device, a.dtype)
    gen_cfg = dict(temp=max(0.0, a.temperature), top_p=a.top_p, max_new=a.max_new,
                   rep=a.repetition_penalty)
    system = a.system

    if a.prompt is not None:                      # single-shot mode
        text, n_new, dt, was_stream = reply(
            model, tok, [{"role": "user", "content": a.prompt}],
            system, device, gen_cfg, stream=not a.no_stream)
        if not text:
            text = "(empty reply)"
        if not was_stream:
            print(text, flush=True)
        else:
            print(flush=True)
        print(f"[{n_new} tokens, {n_new / max(dt, 1e-9):.1f} tok/s]", flush=True)
        return

    print(f"ready: device {device}. Type a message ({HELP})", flush=True)
    chat_loop(model, tok, device, gen_cfg, system)


if __name__ == "__main__":
    main()
