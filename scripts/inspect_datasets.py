"""
Task 3 sanity check + loss-masking evidence.

Prints, for the SFT/mid-training datasets (SmolTalk, MMLU auxiliary_train, GSM8K):
  * how many training examples each contributes (and per stage mixture),
  * the first few raw rows,
  * one example rendered to tokens with the loss mask (1 = assistant token trained on, 0 = ignored),
  * token statistics over a sample (tokens per conversation, share of supervised tokens),
and writes everything to assignment/results/dataset_inspection.md (plus a JSON with the numbers).

Run from nanochat-master with the venv active:
    python ../assignment/scripts/inspect_datasets.py [--sample 2000]
"""
import os, sys, json, argparse, statistics
sys.path.insert(0, os.getcwd())  # nanochat-master

from nanochat.tokenizer import get_tokenizer
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.common import TaskMixture

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=2000, help="examples per dataset for token statistics")
parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "dataset_inspection"))
args = parser.parse_args()

tok = get_tokenizer()
out_md, out_json = [], {}
def emit(line=""):
    print(line); out_md.append(line)

datasets = {
    "SmolTalk train": SmolTalk(split="train"),
    "MMLU auxiliary_train": MMLU(subset="all", split="auxiliary_train"),
    "GSM8K train": GSM8K(subset="main", split="train"),
}
emit("# Dataset inspection (Task 3 sanity check)\n")
emit("## Number of training examples\n")
emit("| dataset | rows |\n|---|---|")
for name, ds in datasets.items():
    emit(f"| {name} | {len(ds):,} |")
    out_json[name] = {"rows": len(ds)}
mid_rows = 3 * len(datasets["MMLU auxiliary_train"]) + 4 * len(datasets["GSM8K train"])
full_rows = len(datasets["SmolTalk train"]) + mid_rows
emit(f"| **Stage 1 mixture (MMLU x3 + GSM8K x4)** | {mid_rows:,} |")
emit(f"| **Stage 2 mixture (SmolTalk x1)** | {len(datasets['SmolTalk train']):,} |")
emit(f"| upstream nanochat mixture (all three) | {full_rows:,} |")
out_json["stage1_rows"], out_json["stage2_rows"], out_json["upstream_rows"] = mid_rows, len(datasets["SmolTalk train"]), full_rows

emit("\n## First rows (as the conversation dicts the trainer sees)\n")
for name, ds in datasets.items():
    emit(f"### {name}\n")
    for i in range(3):
        conv = ds[i]
        emit(f"**example {i}** " + "```json\n" + json.dumps(conv["messages"], ensure_ascii=False, indent=1)[:1500] + "\n```")

emit("\n## Loss mask on one rendered example per dataset\n")
emit("`render_conversation` returns token ids and a mask; the SFT loss is computed only where mask = 1 "
     "(assistant text and `<|assistant_end|>`). BOS, user turns, `<|user_start|>`/`<|user_end|>`, `<|assistant_start|>` and "
     "python tool *outputs* are mask = 0 (targets set to -1 = ignore_index).\n")
for name, ds in datasets.items():
    conv = ds[0] if name != "SmolTalk train" else ds[1]
    ids, mask = tok.render_conversation(conv)
    emit(f"### {name} - {len(ids)} tokens, {sum(mask)} supervised ({100*sum(mask)/len(ids):.1f}%)\n")
    rows = ["| pos | token | mask |", "|---|---|---|"]
    for pos, (t, m) in enumerate(zip(ids, mask)):
        s = tok.decode([t]).replace("\n", "\\n").replace("|", "\\|")
        rows.append(f"| {pos} | `{s}` | {m} |")
        if pos >= 60:
            rows.append(f"| ... | ({len(ids)-61} more tokens) | |"); break
    emit("\n".join(rows) + "\n")

emit("\n## Token statistics (sample of %d examples per dataset)\n" % args.sample)
emit("| dataset | mean tokens/conv | median | max | mean supervised share | est. tokens per epoch |\n|---|---|---|---|---|---|")
for name, ds in datasets.items():
    n = min(args.sample, len(ds))
    lens, shares = [], []
    for i in range(n):
        ids, mask = tok.render_conversation(ds[i])
        lens.append(len(ids)); shares.append(sum(mask) / len(ids))
    mean_len = statistics.mean(lens)
    est = mean_len * len(ds)
    emit(f"| {name} | {mean_len:.0f} | {statistics.median(lens):.0f} | {max(lens)} | {100*statistics.mean(shares):.1f}% | {est/1e6:.1f}M |")
    out_json[name].update({"mean_tokens": mean_len, "median_tokens": statistics.median(lens), "max_tokens": max(lens),
                           "mean_supervised_share": statistics.mean(shares), "est_tokens_per_epoch": est})
s1 = 3 * out_json["MMLU auxiliary_train"]["est_tokens_per_epoch"] + 4 * out_json["GSM8K train"]["est_tokens_per_epoch"]
s2 = out_json["SmolTalk train"]["est_tokens_per_epoch"]
emit(f"\nEstimated tokens seen: stage 1 (mid) ~{s1/1e6:.0f}M, stage 2 (sft) ~{s2/1e6:.0f}M (before packing/padding).")
out_json["stage1_est_tokens"], out_json["stage2_est_tokens"] = s1, s2

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
open(args.out + ".md", "w", encoding="utf-8").write("\n".join(out_md) + "\n")
json.dump(out_json, open(args.out + ".json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {args.out}.md and {args.out}.json")
