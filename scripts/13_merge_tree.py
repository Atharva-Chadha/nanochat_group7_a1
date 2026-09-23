"""
Task 1 Q1: the BPE merge tree of one word, with the exact count of every merge (from 12_merge_counts.py).

The parents of a token of rank r are found by re-encoding its bytes with merges of rank < r, which stops
at exactly two pieces. Merges with rank >= --cut are drawn dashed: on the same sample the 8k tokenizer
holds exactly the ranks 0..8182 of the 32k one (256 bytes + 7,927 merges; the 9 special tokens use up the
rest), so the dashed part of the tree is what the smaller vocabulary never builds.

Run from the nanochat checkout with its venv active:
    python /path/to/repo/scripts/13_merge_tree.py --word " photosynthesis"

Writes results/tokenizer/merge_tree_v<V>_<word>.json and report/figures/01_bpe_merge_tree.png.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="Draw a BPE merge tree with exact merge counts")
parser.add_argument("--word", type=str, default=" photosynthesis", help="the leading space matters")
parser.add_argument("--vocab-size", type=int, default=32768)
parser.add_argument("--cut", type=int, default=8183, help="ranks >= cut are drawn dashed (0 = off)")
parser.add_argument("--fig", type=str, default=os.path.join(REPO, "report", "figures", "01_bpe_merge_tree.png"))
args = parser.parse_args()

tok_dir = os.path.join(get_base_dir(), f"tokenizer_v{args.vocab_size}")
with open(os.path.join(tok_dir, "tokenizer.pkl"), "rb") as f:
    ranks = pickle.load(f)._mergeable_ranks
merge_data = json.load(open(os.path.join(tok_dir, "merge_counts.json"), encoding="utf-8"))
count_of = {m["rank"]: m["count_at_merge"] for m in merge_data["merges"]}
tokenizer = RustBPETokenizer.from_directory(tok_dir)


def parents(token):
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < ranks[token] and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts


def build(token):
    node = {"token": token.decode("utf-8", errors="replace"), "rank": ranks[token]}
    if len(token) > 1:
        node["count_at_merge"] = count_of[ranks[token]]
        node["children"] = [build(p) for p in parents(token)]
    return node


ids = tokenizer.encode(args.word)
pieces = [tokenizer.decode_single_token_bytes(i) for i in ids]
trees = [build(p) for p in pieces]


def show(node, depth=0):
    what = "byte" if "children" not in node else f"rank {node['rank']:>6}  count {node['count_at_merge']:>10,}"
    print(f"{'    ' * depth}{node['token']!r:<18} {what}")
    for k in node.get("children", []):
        show(k, depth + 1)


print(f"{args.word!r} -> {len(ids)} token(s) at vocab {args.vocab_size:,}")
for t in trees:
    show(t)

safe = "".join(c for c in args.word if c.isalnum())
out = os.path.join(REPO, "results", "tokenizer", f"merge_tree_v{args.vocab_size}_{safe}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"word": args.word, "vocab_size": args.vocab_size, "sample": merge_data["sample"],
               "token_ids": ids, "trees": trees}, f, indent=1, ensure_ascii=False)
print("wrote", out)

# ---------------------------------------------------------------- figure, sized for one ICML column
leaves = []


def place(node):
    """Leaves get consecutive x; parents sit over the mean of their children, at 1 + max child height."""
    if "children" not in node:
        node["x"], node["y"] = len(leaves), 0
        leaves.append(node)
    else:
        for k in node["children"]:
            place(k)
        node["x"] = sum(k["x"] for k in node["children"]) / 2
        node["y"] = 1 + max(k["y"] for k in node["children"])


for t in trees:
    place(t)
top = max(t["y"] for t in trees)
fig, ax = plt.subplots(figsize=(3.4, 2.35))
blue, grey = "#2a6ebb", "#999999"


def draw(node):
    late = args.cut and node.get("rank", 0) >= args.cut
    for k in node.get("children", []):
        k_late = args.cut and k.get("rank", 0) >= args.cut
        ax.plot([node["x"], k["x"]], [node["y"], k["y"]], lw=0.8, color=grey, zorder=1,
                ls=(0, (2.5, 1.5)) if late else "-")
        draw(k)
    label = node["token"].replace(" ", "␣")
    leaf = "children" not in node
    ax.text(node["x"], node["y"], label, ha="center", va="center", fontsize=6.5 if leaf else 6.8,
            family="monospace", zorder=3,
            bbox=dict(boxstyle="round,pad=0.18", lw=0.6 if leaf else 0.9,
                      fc="white", ec=grey if leaf else blue, ls="--" if late else "-"))
    if not leaf:
        ax.text(node["x"], node["y"] + 0.33, f"{node['count_at_merge']:,}", ha="center", va="bottom",
                fontsize=5.6, color="#c0392b", zorder=2,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))


for t in trees:
    draw(t)
ax.set_xlim(-0.7, len(leaves) - 0.3)
ax.set_ylim(-0.5, top + 0.75)
ax.axis("off")
fig.tight_layout(pad=0.1)
fig.savefig(args.fig, dpi=300)
print("wrote", args.fig)
