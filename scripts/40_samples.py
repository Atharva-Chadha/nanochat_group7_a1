"""
Sampling evidence for Task 2 (Q4: raw completions of the pre-trained model) and Task 4A (Q1: temperature 0.1 / 0.7 / 1.5).

For a checkpoint (base | sft, model tag) it generates
  * N unconditioned samples (just <|bos|>, temperature 1.0)             -> "what does raw next-token prediction do"
  * completions for 5 fixed prompts at each temperature                  -> quality/diversity vs temperature
    base models get the raw text prefix; chat models get the chat template (<|user_start|> ... <|assistant_start|>)
and writes assignment/results/samples_<label>.md and .json.

Run from nanochat-master with the venv active:
    python ../assignment/scripts/40_samples.py --source base --model-tag d8 --label d8-base
    python ../assignment/scripts/40_samples.py --source sft  --model-tag d8_sft --label d8-sft --temperatures 0.1 0.7 1.5
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.getcwd())
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

PROMPTS = [
    "Why is the sky blue?",
    "Write a short poem about the sea.",
    "What is the capital of France?",
    "Explain in two sentences what a transformer neural network is.",
    "If I have 3 apples and buy 4 more, how many apples do I have?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="base", choices=["base", "sft"])
ap.add_argument("--model-tag", default=None)
ap.add_argument("--step", type=int, default=None)
ap.add_argument("--label", default=None)
ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.7, 1.5])
ap.add_argument("--num-unconditioned", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=160)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
args = ap.parse_args()
label = args.label or f"{args.model_tag or 'model'}-{args.source}"

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
model, tok, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tok)
bos = tok.get_bos_token_id()
chat = args.source != "base"
if chat:
    user_start, user_end = tok.encode_special("<|user_start|>"), tok.encode_special("<|user_end|>")
    assistant_start = tok.encode_special("<|assistant_start|>")

def prompt_tokens(text):
    if chat:
        return [bos, user_start, *tok.encode(text), user_end, assistant_start]
    return [bos, *tok.encode(text)]

def generate(tokens, temperature, num_samples=1, seed=None):
    t0 = time.time()
    results, _ = engine.generate_batch(tokens, num_samples=num_samples, max_tokens=args.max_tokens,
                                       temperature=temperature, top_k=args.top_k, seed=args.seed if seed is None else seed)
    dt = time.time() - t0
    outs = [tok.decode(r[len(tokens):]) for r in results]
    ntoks = sum(len(r) - len(tokens) for r in results)
    return outs, ntoks, dt

record = {"label": label, "source": args.source, "model_tag": args.model_tag, "step": meta.get("step"),
          "model_config": meta.get("model_config"), "max_tokens": args.max_tokens, "top_k": args.top_k, "unconditioned": [], "prompted": []}
md = [f"# Samples: {label} ({args.source}/{args.model_tag}, step {meta.get('step')})\n",
      f"max_tokens={args.max_tokens}, top_k={args.top_k}, seed={args.seed}\n"]

# 1) unconditioned: raw next-token prediction from <|bos|> only
md.append(f"## {args.num_unconditioned} unconditioned samples (temperature 1.0)\n")
outs, ntoks, dt = generate([bos], 1.0, num_samples=args.num_unconditioned)
for i, o in enumerate(outs):
    record["unconditioned"].append(o)
    md.append(f"**sample {i+1}**\n\n```text\n{o}\n```\n")
print(f"[unconditioned] {ntoks} tokens in {dt:.1f}s ({ntoks/dt:.0f} tok/s)")

# 2) prompted at each temperature
for T in args.temperatures:
    md.append(f"## Prompted completions at temperature {T}\n")
    for p in PROMPTS:
        outs, ntoks, dt = generate(prompt_tokens(p), T)
        record["prompted"].append({"temperature": T, "prompt": p, "completion": outs[0], "tokens": ntoks, "seconds": dt})
        md.append(f"**{p}**\n\n```text\n{outs[0]}\n```\n")
        print(f"[T={T}] {p[:40]:40s} -> {ntoks:4d} tokens, {ntoks/max(dt,1e-6):.0f} tok/s")

os.makedirs(args.out_dir, exist_ok=True)
base = os.path.join(args.out_dir, f"samples_{label}")
open(base + ".md", "w", encoding="utf-8").write("\n".join(md))
json.dump(record, open(base + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", base + ".md/.json")
