#!/usr/bin/env bash
# Unattended chain after pretraining: mid-training -> SFT -> evaluation of all three stages -> samples -> figures.
# Everything lands in results/ and report/figures/. ~2-2.5 h on the RTX 5070 Ti for depth 8
# (dominated by the sequential GSM8K generation in chat_eval).
#
# Usage: bash 35_run_task3_task4.sh            Env: BASE_TAG=d8  BASE_GSM8K_MAX=300 (the base model never emits
#        <|assistant_end|>, so its GSM8K eval is capped)  SKIP_TRAIN=1 (only evals/samples/figures)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SCRIPTS="$ASS_DIR/scripts"
BASE_TAG="${BASE_TAG:-d8}"
MID_TAG="${BASE_TAG}_mid"
SFT_TAG="${BASE_TAG}_sft"
BASE_GSM8K_MAX="${BASE_GSM8K_MAX:-300}"
T0=$(date +%s)
stamp() { echo; echo "######## [$(date +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 )) min] $*"; }

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    stamp "Stage 1: mid-training (MMLU x3 + GSM8K x4) from base/$BASE_TAG"
    BASE_TAG="$BASE_TAG" OUT_TAG="$MID_TAG" bash "$SCRIPTS/30_midtrain.sh"
    stamp "Stage 2: SFT (SmolTalk) from chatsft/$MID_TAG"
    MID_TAG="$MID_TAG" OUT_TAG="$SFT_TAG" bash "$SCRIPTS/31_sft.sh"
fi

stamp "Evaluation: pre-trained base model through the chat harness (GSM8K capped at $BASE_GSM8K_MAX problems, 256 new tokens)"
GSM8K_MAX_PROBLEMS="$BASE_GSM8K_MAX" MAX_NEW_TOKENS=256 bash "$SCRIPTS/32_chat_eval.sh" base "$BASE_TAG" "${BASE_TAG}-pretrained"
stamp "Evaluation: stage 1 (mid-training)"
bash "$SCRIPTS/32_chat_eval.sh" sft "$MID_TAG" "${BASE_TAG}-stage1-mid"
stamp "Evaluation: stage 2 (SFT)"
bash "$SCRIPTS/32_chat_eval.sh" sft "$SFT_TAG" "${BASE_TAG}-stage2-sft"

stamp "Samples: base (raw completions), mid, sft (temperature 0.1 / 0.7 / 1.5)"
python "$SCRIPTS/40_samples.py" --source base --model-tag "$BASE_TAG" --label "${BASE_TAG}-base" 2>&1 | tee "$RESULTS_DIR/samples_${BASE_TAG}-base.log"
python "$SCRIPTS/40_samples.py" --source sft --model-tag "$MID_TAG" --label "${MID_TAG}" 2>&1 | tee "$RESULTS_DIR/samples_${MID_TAG}.log"
python "$SCRIPTS/40_samples.py" --source sft --model-tag "$SFT_TAG" --label "${SFT_TAG}" 2>&1 | tee "$RESULTS_DIR/samples_${SFT_TAG}.log"

stamp "Figures"
FIG="$ASS_DIR/report/figures"
python "$SCRIPTS/plot_curves.py" "$RESULTS_DIR/${BASE_TAG}_pretrain_metrics.jsonl" --out "$FIG" --out-name "${BASE_TAG}_pretrain" | tee "$RESULTS_DIR/${BASE_TAG}_curve_summary.txt"
python "$SCRIPTS/plot_curves.py" "$RESULTS_DIR/${MID_TAG}_metrics.jsonl" --out "$FIG" --out-name "$MID_TAG" | tee -a "$RESULTS_DIR/${BASE_TAG}_curve_summary.txt"
python "$SCRIPTS/plot_curves.py" "$RESULTS_DIR/${SFT_TAG}_metrics.jsonl" --out "$FIG" --out-name "$SFT_TAG" | tee -a "$RESULTS_DIR/${BASE_TAG}_curve_summary.txt"

stamp "ALL DONE"
column -t -s, "$RESULTS_DIR/eval_scores.csv" 2>/dev/null || cat "$RESULTS_DIR/eval_scores.csv"
