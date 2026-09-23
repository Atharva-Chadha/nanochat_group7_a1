#!/usr/bin/env bash
# Task 3, Stage 1 - Mid-training: reasoning + knowledge data only (MMLU auxiliary_train x3, GSM8K train x4),
# starting from the pre-trained base checkpoint. Implemented via our --stage/--source/--output-tag extension of
# scripts/chat_sft.py (see the report's "Implementation" section and `git log -p nanochat-master/scripts/chat_sft.py`).
#
# Usage: bash 30_midtrain.sh [extra chat_sft args]      Env: BASE_TAG=d8  OUT_TAG=${BASE_TAG}_mid  EVAL_EVERY=100
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BASE_TAG="${BASE_TAG:-d8}"
OUT_TAG="${OUT_TAG:-${BASE_TAG}_mid}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
METRICS="$RESULTS_DIR/${OUT_TAG}_metrics.jsonl"
[ -f "$METRICS" ] && mv "$METRICS" "$METRICS.bak.$(date +%Y%m%d_%H%M%S)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu --format=csv -l 10 \
        > "$RESULTS_DIR/${OUT_TAG}_power.csv" 2>/dev/null &
    trap 'kill "$!" 2>/dev/null || true' EXIT
fi

echo "=== Stage 1 (mid-training): base/$BASE_TAG -> chatsft/$OUT_TAG ==="
run_logged "${OUT_TAG}_train" python -m scripts.chat_sft \
    --stage=mid --source=base --model-tag="$BASE_TAG" --output-tag="$OUT_TAG" \
    --device-batch-size=8 \
    --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" --chatcore-every=-1 \
    --log-file="$METRICS" "$@"

echo "DONE  checkpoint: $NANOCHAT_BASE_DIR/chatsft_checkpoints/$OUT_TAG   metrics: $METRICS"
ls -la "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$OUT_TAG"
