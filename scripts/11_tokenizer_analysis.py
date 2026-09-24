"""
Task 1 evidence: compare our two BPE tokenizers (vocab 8,192 and 32,768, trained on 500 MB of ClimbMix) with the
GPT-2 (50,257) and GPT-4 cl100k (100,277) tokenizers.

  * compression: bytes/token and tokens/character, and the resulting sequence length, per text type
  * vocabulary composition: token length distribution, digit-only tokens, non-ASCII tokens
  * failure cases: how numbers, source code, whitespace, non-English text and emoji are split

Writes results/tokenizer/tokenizer_analysis.{md,json} and report/figures/02_tokenizer_compression.png.
Run from the nanochat checkout with its venv active:
    python /path/to/repo/scripts/11_tokenizer_analysis.py
"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from nanochat.common import get_base_dir

base = get_base_dir()
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

toks = {
    "ours-8k": RustBPETokenizer.from_directory(os.path.join(base, "tokenizer_v8192")),
    "ours-32k": RustBPETokenizer.from_directory(os.path.join(base, "tokenizer_v32768")),
    "gpt2-50k": RustBPETokenizer.from_pretrained("gpt2"),
    "gpt4-100k": RustBPETokenizer.from_pretrained("cl100k_base"),
}

# ---------------------------------------------------------------- texts (our own, plus ClimbMix samples)
english = (
    "The village council met on Tuesday evening to discuss the new footpath along the river. Several residents "
    "argued that the gravel surface would wash away in the first heavy rain, while others pointed out that the "
    "paved alternative would cost nearly three times as much. After two hours the council agreed to run a small "
    "trial section of two hundred metres before deciding on the rest of the route."
)
code = '''def moving_average(values, window=5):
    """Return the centred moving average of a list of floats."""
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out

if __name__ == "__main__":
    print(moving_average([1.0, 2.0, 4.0, 8.0, 16.0]))
'''
numbers = (
    "Invoice 2026-09-21: 3 items at $1,234.56 each, subtotal 3703.68, VAT 21% = 777.77, total 4481.45. "
    "Reference 0044-8812-9931, serial 1234567890, pi = 3.14159265, 1e-9, 0x7FFF, 12:45:07."
)
korean = "오늘 아침에는 비가 조금 내렸지만 오후에는 맑아졌다. 시장에서 사과와 배를 사고 집으로 돌아왔다."
german = "Die Straße war nass, aber der Bäcker öffnete trotzdem pünktlich um sechs Uhr; die Brötchen waren noch warm."
hindi = "आज सुबह हल्की बारिश हुई, लेकिन दोपहर तक आसमान साफ हो गया।"
emoji_mixed = "Launch day 🚀🌍 — team says: \"ça marche!\" naïve café résumé 😀"

val_docs = next(parquets_iter_batched(split="val"))
climbmix_val = "\n".join(val_docs)[:2_000_000]

texts = [("English (own)", english), ("Python code", code), ("Numbers", numbers), ("Korean", korean),
         ("German", german), ("Hindi", hindi), ("Emoji/accents", emoji_mixed), ("ClimbMix val (2M chars)", climbmix_val)]

# ---------------------------------------------------------------- compression table
rows = []
for name, text in texts:
    nbytes, nchars = len(text.encode("utf-8")), len(text)
    entry = {"text": name, "bytes": nbytes, "chars": nchars}
    for tname, tk in toks.items():
        ids = tk.encode(text)
        assert tk.decode(ids) == text
        entry[tname] = {"tokens": len(ids), "bytes_per_token": nbytes / len(ids), "tokens_per_char": len(ids) / nchars}
    rows.append(entry)

# ---------------------------------------------------------------- vocabulary composition
vocab_stats = {}
for tname in ("ours-8k", "ours-32k"):
    tk = toks[tname]
    special = set(tk.encode_special(s) for s in tk.get_special_tokens())
    lens, digit_only, non_ascii, longest = [], 0, 0, []
    for tid in range(tk.get_vocab_size()):
        if tid in special:
            continue
        b = tk.decode_single_token_bytes(tid)
        lens.append(len(b))
        s = b.decode("utf-8", errors="replace")
        if s.strip() and s.strip().isdigit():
            digit_only += 1
        if any(ch >= 128 for ch in b):
            non_ascii += 1
        longest.append((len(b), s))
    longest.sort(reverse=True)
    vocab_stats[tname] = {
        "vocab_size": tk.get_vocab_size(), "mean_token_bytes": sum(lens) / len(lens),
        "tokens_len_1": sum(1 for l in lens if l == 1), "tokens_len_ge_8": sum(1 for l in lens if l >= 8),
        "digit_only_tokens": digit_only, "non_ascii_tokens": non_ascii,
        "longest_tokens": [s for _, s in longest[:12]],
    }

# ---------------------------------------------------------------- failure cases: token-by-token views
probes = [
    "1234567890", "3.14159265", "2026-09-21", "$1,234.56", "0x7FFF",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "    x = 1\n        y = 2\n",
    "안녕하세요 세계", "Straße", "naïve café", "🚀🌍😀", "supercalifragilisticexpialidocious", "unbelievably",
]
probe_rows = []
for p in probes:
    r = {"probe": p}
    for tname in ("ours-8k", "ours-32k", "gpt4-100k"):
        tk = toks[tname]
        ids = tk.encode(p)
        pieces = [tk.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        r[tname] = {"n": len(ids), "pieces": pieces}
    probe_rows.append(r)

# ---------------------------------------------------------------- figure: bytes per token by text type
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e2", "#ffffff"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
names = [r["text"].replace(" (2M chars)", "").replace(" (own)", "") for r in rows]
tnames = list(toks)
w = 0.2
for i, tname in enumerate(tnames):
    xs = [j + (i - 1.5) * w for j in range(len(rows))]
    ys = [r[tname]["bytes_per_token"] for r in rows]
    ax.bar(xs, ys, width=w - 0.02, color=SERIES[i], label=tname, linewidth=0)
ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("bytes per token (higher = better compression)")
ax.grid(True, axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Tokenizer compression by text type: our 8k / 32k BPE vs GPT-2 / GPT-4", color=TEXT, loc="left", fontsize=10)
ax.set_ylim(0, max(r[t]["bytes_per_token"] for r in rows for t in tnames) * 1.28)  # headroom for the legend
ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tokenizer_compression.png")); plt.close(fig)

# ---------------------------------------------------------------- markdown report
md = ["# Tokenizer analysis (Task 1)\n",
      "Tokenizers: ours-8k / ours-32k = nanochat rustbpe BPE trained on 500M chars of ClimbMix (`10_train_tokenizers.sh`), "
      "GPT-2 (50,257) and GPT-4 cl100k_base (100,277) via tiktoken for reference.\n",
      "## Compression and sequence length\n",
      "| text | bytes | " + " | ".join(f"{t} tokens" for t in tnames) + " | " + " | ".join(f"{t} bytes/tok" for t in tnames) + " | 8k/32k length ratio |",
      "|---|---|" + "---|" * (2 * len(tnames)) + "---|"]
for r in rows:
    md.append(f"| {r['text']} | {r['bytes']:,} | " + " | ".join(f"{r[t]['tokens']:,}" for t in tnames) + " | "
              + " | ".join(f"{r[t]['bytes_per_token']:.2f}" for t in tnames)
              + f" | {r['ours-8k']['tokens'] / r['ours-32k']['tokens']:.2f}x |")
md.append("\ntokens per character (the spec's 'compression ratio'): " + "; ".join(
    f"{r['text']}: 8k {r['ours-8k']['tokens_per_char']:.3f}, 32k {r['ours-32k']['tokens_per_char']:.3f}" for r in rows) + "\n")
md.append("## Vocabulary composition\n")
md.append("| tokenizer | vocab | mean token bytes | 1-byte tokens | tokens >= 8 bytes | digit-only tokens | tokens with non-ASCII bytes |\n|---|---|---|---|---|---|---|")
for t, v in vocab_stats.items():
    md.append(f"| {t} | {v['vocab_size']:,} | {v['mean_token_bytes']:.2f} | {v['tokens_len_1']} | {v['tokens_len_ge_8']:,} | {v['digit_only_tokens']} | {v['non_ascii_tokens']:,} |")
for t, v in vocab_stats.items():
    md.append(f"\nlongest tokens ({t}): " + ", ".join(f"`{s}`" for s in v["longest_tokens"]))
md.append("\n## Probes: how numbers, code, whitespace, non-English text and emoji are split\n")
md.append("| probe | ours-8k | ours-32k | gpt4-100k |\n|---|---|---|---|")
def show(piece):
    return "`" + piece.replace("|", "\\|").replace("\n", "\\n") + "`"
for r in probe_rows:
    cell = lambda t: str(r[t]["n"]) + ": " + " ".join(show(p) for p in r[t]["pieces"])
    probe_disp = show(r["probe"])
    md.append("| " + probe_disp + " | " + cell("ours-8k") + " | " + cell("ours-32k") + " | " + cell("gpt4-100k") + " |")
md.append("\n![compression](../report/figures/tokenizer_compression.png)\n")
open(os.path.join(OUT_DIR, "tokenizer_analysis.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump({"compression": rows, "vocab": vocab_stats, "probes": probe_rows}, open(os.path.join(OUT_DIR, "tokenizer_analysis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n".join(md[:5 + len(rows)]))
print("\nwrote tokenizer_analysis.md/.json and figures/tokenizer_compression.png")
