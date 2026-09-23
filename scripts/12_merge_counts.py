"""
Task 1 Q1 support: recover the exact pair frequency at the moment each BPE merge was learned.

rustbpe only logs the frequency of every 1% of merges, so the count behind an arbitrary token is not in
the training log. This script recovers all of them:

  1. Pre-tokenise the same sample tok_train.py trains on (first --max-chars characters of the train split,
     each document cropped to --doc-cap) with nanochat's SPLIT_PATTERN, giving a Counter over chunks.
  2. Read the merge order out of the saved tokenizer. A token of rank r is re-encoded from its raw bytes
     with merges of rank < r only; that stops at exactly two pieces, which are its parents.
  3. Replay the merges in rank order with incrementally maintained pair counts and record the count of
     each pair just before it is merged.

Step 2 means rustbpe's tie-breaking never has to be guessed. With --check-log the recovered counts are
compared against every frequency rustbpe printed during training (results/tokenizer/task1_tok_train_*.log).

Run from the nanochat checkout with its venv active:
    python /path/to/repo/scripts/12_merge_counts.py --vocab-size 32768 \
        --check-log /path/to/repo/results/tokenizer/task1_tok_train_v32768_20260921_125851.log

Writes $NANOCHAT_BASE_DIR/tokenizer_v<V>/merge_counts.json (~10 MB at 32k, so it stays out of the repo).
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
import regex
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import SPLIT_PATTERN

parser = argparse.ArgumentParser(description="Recover exact BPE merge counts by replay")
parser.add_argument("--vocab-size", type=int, default=32768)
parser.add_argument("--max-chars", type=int, default=500_000_000, help="must match 10_train_tokenizers.sh")
parser.add_argument("--doc-cap", type=int, default=10_000, help="must match tok_train.py")
parser.add_argument("--check-log", type=str, default=None, help="rustbpe training log to verify against")
args = parser.parse_args()

tok_dir = os.path.join(get_base_dir(), f"tokenizer_v{args.vocab_size}")
with open(os.path.join(tok_dir, "tokenizer.pkl"), "rb") as f:
    ranks = pickle.load(f)._mergeable_ranks
print(f"tokenizer: {tok_dir} ({len(ranks):,} mergeable ranks)")


def pretoken_counts():
    """Same stopping rule as tok_train.text_iterator: the document that crosses max_chars is included."""
    pat = regex.compile(SPLIT_PATTERN)
    counts = Counter()
    nchars = ndocs = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc = doc[:args.doc_cap]
            for piece in pat.findall(doc):
                counts[piece.encode("utf-8")] += 1
            nchars += len(doc)
            ndocs += 1
            if nchars > args.max_chars:
                return counts, {"num_docs": ndocs, "num_chars": nchars}
    raise RuntimeError("ran out of data before max_chars")


def bpe_merge(token, max_rank):
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < max_rank and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts


def merge_order():
    merges = []
    for token, rank in ranks.items():
        if len(token) > 1:
            parts = bpe_merge(token, rank)
            assert len(parts) == 2, f"expected 2 parents for {token!r}, got {parts!r}"
            merges.append((rank, parts[0], parts[1], token))
    return sorted(merges)


def replay(counts, merges):
    """{rank: pair count just before that merge}. Ids 0..255 are bytes, merge r creates id r."""
    tok_to_id = {bytes([i]): i for i in range(256)}
    words = [list(tok) for tok in counts]
    freq = list(counts.values())
    pair_counts, pair_words = Counter(), defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freq[wi]
            pair_words[p].add(wi)

    recorded = {}
    for n, (rank, left, right, result) in enumerate(merges):
        a, b = tok_to_id[left], tok_to_id[right]
        tok_to_id[result] = rank
        recorded[rank] = pair_counts.get((a, b), 0)
        for wi in list(pair_words.get((a, b), ())):
            w, f = words[wi], freq[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                pair_words[p].discard(wi)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(rank)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_counts[p] += f
                pair_words[p].add(wi)
        pair_words.pop((a, b), None)
        if (n + 1) % 4000 == 0:
            print(f"  replayed {n + 1:,}/{len(merges):,}", flush=True)
    return recorded


t0 = time.time()
counts, sample = pretoken_counts()
print(f"sample: {sample['num_docs']:,} docs, {sample['num_chars']:,} chars, "
      f"{len(counts):,} unique chunks ({time.time() - t0:.0f}s)")
merges = merge_order()
recorded = replay(counts, merges)
print(f"replayed {len(merges):,} merges ({time.time() - t0:.0f}s total)")

dec = lambda b: b.decode("utf-8", errors="replace")
out = [{"rank": r, "left": dec(l), "right": dec(rt), "result": dec(res), "count_at_merge": recorded[r]}
       for r, l, rt, res in merges]
out_path = os.path.join(tok_dir, "merge_counts.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"tokenizer_dir": tok_dir, "sample": sample, "merges": out}, f, ensure_ascii=False)
print(f"wrote {out_path}")

if args.check_log:
    text = open(args.check_log, encoding="utf-8", errors="replace").read()
    logged = [(int(r), int(c)) for r, c in re.findall(r"-> (\d+) \(frequency: (\d+)\)", text)]
    bad = [(r, c, recorded.get(r)) for r, c in logged if recorded.get(r) != c]
    print(f"rustbpe log check: {len(logged) - len(bad)}/{len(logged)} logged frequencies reproduced")
    for r, c, got in bad[:10]:
        print(f"  rank {r}: rustbpe {c:,}, replay {got}")
