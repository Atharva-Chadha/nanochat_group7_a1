"""
Side-by-side comparison of several nanochat checkpoints on identical prompts.

Built for two jobs:
  * the depth series (base checkpoints d6 / d8 / d10) -- does a bigger model know more?
  * the training stages (d8 base -> d8_mid -> d8_sft) -- what did each stage change?

Loads one checkpoint at a time and frees it before the next, so a 24 GB card
handles the whole series. Writes a markdown table per prompt to
assignment/results/compare_<name>.md plus a json dump of the raw generations.

Run from nanochat-master with the venv active:

    # depth series, greedy, factual probes
    python ../assignment/scripts/50_compare_models.py --models base:d6 base:d8 base:d10

    # training stages, chat template, sampled
    python ../assignment/scripts/50_compare_models.py \
        --models base:d8 sft:d8_mid sft:d8_sft --prompts chat --temperature 0.7 --name stages

    # one model, interactive
    python ../assignment/scripts/50_compare_models.py --models sft:d8_sft --interactive

On the shared 3090 node (upstream nanochat, no assignment/ tree) pass --out-dir .
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

# Probes chosen because they separate the depths: each one is a single fact or a
# single arithmetic step, so a wrong answer is unambiguous. The gold and planets
# probes are the two that changed between depth 8 and depth 10.
PROBE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "The planets of the solar system are",
    "The opposite of hot is",
    "If yesterday was Friday, then tomorrow will be",
    "If 5*x + 3 = 13, then x is",
    "Water boils at a temperature of",
    "The largest ocean on Earth is",
]

CHAT_PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the sea.",
    "Why is the sky blue?",
    "If I have 3 apples and buy 4 more, how many apples do I have?",
    "Explain in two sentences what a transformer neural network is.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True,
                help="checkpoints as source:tag, e.g. base:d6 base:d8 sft:d8_sft")
ap.add_argument("--prompts", default="probes", choices=["probes", "chat", "file"])
ap.add_argument("--prompt-file", default=None, help="one prompt per line, with --prompts file")
ap.add_argument("--temperature", type=float, default=0.0,
                help="0.0 = greedy, which is what you want for comparing factual recall")
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--name", default="depths")
ap.add_argument("--interactive", action="store_true", help="chat with the first --models entry")
ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "results"))
args = ap.parse_args()

if args.prompts == "file":
    if not args.prompt_file:
        ap.error("--prompts file needs --prompt-file")
    with open(args.prompt_file, encoding="utf-8") as fh:
        prompts = [ln.strip() for ln in fh if ln.strip()]
else:
    prompts = PROBE_PROMPTS if args.prompts == "probes" else CHAT_PROMPTS

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)


def wrap(tok, text, chat):
    """Base models continue raw text; chat models need the turn markers."""
    bos = tok.get_bos_token_id()
    if not chat:
        return [bos, *tok.encode(text)]
    return [bos,
            tok.encode_special("<|user_start|>"), *tok.encode(text),
            tok.encode_special("<|user_end|>"), tok.encode_special("<|assistant_start|>")]


def run_one(spec):
    source, _, tag = spec.partition(":")
    assert source in ("base", "sft"), "model spec must be base:<tag> or sft:<tag>"
    model, tok, meta = load_model(source, device, phase="eval",
                                  model_tag=tag or None, step=None)
    engine = Engine(model, tok)
    chat = source != "base"
    cfg = meta.get("model_config", {}) or {}
    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "spec": spec, "source": source, "tag": tag,
        "params": n_params,
        "depth": cfg.get("n_layer"), "d_model": cfg.get("n_embd"),
        "heads": cfg.get("n_head"), "window": cfg.get("window_pattern"),
        "step": meta.get("step"), "val_bpb": meta.get("val_bpb"),
    }

    if args.interactive:
        print("\ninteractive: %s (%.1fM params). blank line or ctrl-c to quit.\n"
              % (spec, n_params / 1e6))
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            out, _ = engine.generate_batch(
                wrap(tok, text, chat), num_samples=1, max_tokens=256,
                temperature=max(args.temperature, 0.7), top_k=args.top_k, seed=args.seed)
            print(tok.decode(out[0]).strip(), "\n")
        return info, {}

    gens = {}
    for p in prompts:
        t0 = time.time()
        out, _ = engine.generate_batch(
            wrap(tok, p, chat), num_samples=1, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed)
        text = tok.decode(out[0])
        # decode renders the special tokens, so drop the scaffolding and the
        # echoed prompt and keep only what the model added
        for marker in ("<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>"):
            text = text.replace(marker, "")
        cut = text.find(p)
        if cut != -1:
            text = text[cut + len(p):]
        text = text.split("<|assistant_end|>")[0]
        gens[p] = {"text": " ".join(text.split()), "seconds": round(time.time() - t0, 2)}
        print("  %-42s %s" % (p[:42], gens[p]["text"][:70]))

    del engine, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info, gens


infos, all_gens = [], {}
for spec in args.models:
    print("\n=== %s ===" % spec)
    info, gens = run_one(spec)
    infos.append(info)
    all_gens[spec] = gens

if args.interactive:
    sys.exit(0)

os.makedirs(args.out_dir, exist_ok=True)
stem = os.path.join(args.out_dir, "compare_%s" % args.name)

with open(stem + ".md", "w", encoding="utf-8") as fh:
    fh.write("# Model comparison: %s\n\n" % ", ".join(args.models))
    fh.write("temperature %.2f, top-k %d, max %d tokens, seed %d. "
             % (args.temperature, args.top_k, args.max_tokens, args.seed))
    fh.write("Base models continue the prompt as raw text; SFT models receive it "
             "through the chat template.\n\n")

    fh.write("| checkpoint | depth | d_model | heads | params | step | val bpb |\n")
    fh.write("|---|---|---|---|---|---|---|\n")
    for i in infos:
        fh.write("| `%s` | %s | %s | %s | %s | %s | %s |\n" % (
            i["spec"], i["depth"], i["d_model"], i["heads"],
            "{:,}".format(i["params"]), i["step"],
            "%.4f" % i["val_bpb"] if isinstance(i["val_bpb"], float) else "-"))
    fh.write("\n")

    for p in prompts:
        fh.write("### %s\n\n" % p)
        fh.write("| checkpoint | continuation |\n|---|---|\n")
        for spec in args.models:
            txt = all_gens[spec].get(p, {}).get("text", "")
            fh.write("| `%s` | %s |\n" % (spec, txt.replace("|", "\\|")))
        fh.write("\n")

with open(stem + ".json", "w", encoding="utf-8") as fh:
    json.dump({"args": vars(args), "models": infos, "generations": all_gens},
              fh, indent=2, ensure_ascii=False)

print("\nwrote %s.md and %s.json" % (stem, stem))
