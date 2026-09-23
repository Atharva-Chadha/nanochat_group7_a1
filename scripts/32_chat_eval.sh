#!/usr/bin/env bash
# Task 3 evaluation: ARC-Easy, ARC-Challenge (categorical, fast) and GSM8K (generative, slow) with scripts/chat_eval.py,
# for any checkpoint: the pre-trained base model, the mid-training checkpoint or the SFT checkpoint.
# Appends one row per task to assignment/results/eval_scores.csv (the table for the report).
#
# Usage: bash 32_chat_eval.sh <source: base|sft> <model-tag> [label]
#   bash 32_chat_eval.sh base d8      pretrained         # base model through the chat harness (expected ~random)
#   bash 32_chat_eval.sh sft  d8_mid  stage1-mid
#   bash 32_chat_eval.sh sft  d8_sft  stage2-sft
# Env: MAX_PROBLEMS (default: all)  GSM8K_MAX_PROBLEMS (default: MAX_PROBLEMS)  MAX_NEW_TOKENS=512  TASKS="ARC-Easy ARC-Challenge GSM8K"
# GSM8K runs one problem at a time (1,319 test problems x up to MAX_NEW_TOKENS tokens): ~20-40 min for a depth-8
# chat model, longer for the base model which never emits <|assistant_end|> - use GSM8K_MAX_PROBLEMS for it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SOURCE="${1:?usage: 32_chat_eval.sh <base|sft> <model-tag> [label]}"
TAG="${2:?usage: 32_chat_eval.sh <base|sft> <model-tag> [label]}"
LABEL="${3:-$TAG}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
GSM8K_MAX_PROBLEMS="${GSM8K_MAX_PROBLEMS:-$MAX_PROBLEMS}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TASKS="${TASKS:-ARC-Easy ARC-Challenge GSM8K}"
CSV="$RESULTS_DIR/eval_scores.csv"
[ -f "$CSV" ] || echo "timestamp,label,source,model_tag,task,correct,total,accuracy,max_problems,max_new_tokens,log" > "$CSV"

for task in $TASKS; do
    limit="$MAX_PROBLEMS"; [ "$task" = "GSM8K" ] && limit="$GSM8K_MAX_PROBLEMS"
    extra=(); [ -n "$limit" ] && extra+=(-x "$limit")
    name="eval_${LABEL}_${task}"
    log="$RESULTS_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "=== $LABEL ($SOURCE/$TAG) on $task ${limit:+(first $limit problems)} ==="
    python -m scripts.chat_eval -i "$SOURCE" -g "$TAG" -a "$task" -b 8 -m "$MAX_NEW_TOKENS" "${extra[@]}" 2>&1 | tee "$log"
    # chat_eval prints e.g. "Final: 1234/2376 (51.94%)"; take the last one
    final=$(grep -Eo 'Final: [0-9]+/[0-9]+ \([0-9.]+%\)' "$log" | tail -n 1)
    correct=$(echo "$final" | sed -E 's/Final: ([0-9]+)\/([0-9]+).*/\1/')
    total=$(echo "$final" | sed -E 's/Final: ([0-9]+)\/([0-9]+).*/\2/')
    acc=$(echo "$final" | sed -E 's/.*\(([0-9.]+)%\)/\1/')
    echo "$(date -Iseconds),$LABEL,$SOURCE,$TAG,$task,$correct,$total,$acc,${limit:-all},$MAX_NEW_TOKENS,$(basename "$log")" >> "$CSV"
    echo ">>> $LABEL $task: $correct/$total = $acc%"
done

echo
echo "eval_scores.csv:"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
