"""
Task 1 Q3: how often each vocabulary entry is actually used, on held-out ClimbMix text.

A token type that almost never occurs gets almost no gradient on its embedding and lm_head rows, so the
tail of this distribution is the part of the vocabulary the model cannot learn well. For each tokenizer we
count occurrences over the first --chars characters of the validation shard and report the unused types,
the rare types, and how many training occurrences the frequencies imply over the depth-8 horizon.

Run from the nanochat checkout with its venv active:
    python /path/to/repo/scripts/14_token_usage.py
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--chars", type=int, default=20_000_000)
ap.add_argument("--train-tokens", type=int, default=503_316_480, help="depth-8 horizon")
args = ap.parse_args()

docs, n = [], 0
for batch in parquets_iter_batched(split="val"):
    for doc in batch:
        docs.append(doc)
        n += len(doc)
        if n >= args.chars:
            break
    if n >= args.chars:
        break
print(f"held-out sample: {len(docs):,} docs, {n:,} chars")

for vocab in (8192, 32768):
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), f"tokenizer_v{vocab}"))
    counts = Counter()
    for ids in tok.encode(docs):
        counts.update(ids)
    total = sum(counts.values())
    n_regular = vocab - len(tok.get_special_tokens())
    freq = sorted((counts.get(i, 0) for i in range(n_regular)), reverse=True)
    unused = sum(1 for c in freq if c == 0)
    rare = sum(1 for c in freq if 0 < c < 10)
    covered, k = 0, 0
    while covered < 0.99 * total:
        covered += freq[k]
        k += 1
    median = freq[len(freq) // 2]
    print(f"vocab {vocab:>6,}: {total:,} tokens | unused {unused:,} | seen 1-9 times {rare:,} | "
          f"99% of occurrences from {k:,} types | median type seen {median:,} times, "
          f"~{median / total * args.train_tokens:,.0f} times in {args.train_tokens / 1e6:.0f}M training tokens")
