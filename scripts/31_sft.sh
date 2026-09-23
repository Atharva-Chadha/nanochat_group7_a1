#!/usr/bin/env bash
# Task 3, Stage 2 - SFT: SmolTalk conversations only, starting from the Stage-1 (mid-training) checkpoint.
# --source=sft makes chat_sft.py load from chatsft_checkpoints/<MID_TAG> (model + optimizer momentum) instead of
# base_checkpoints/, and inherit the batch/LR settings that stage 1 saved in its checkpoint metadata.
#
# Usage: bash 31_sft.sh [extra chat_sft args]      Env: MID_TAG=d8_mid  OUT_TAG=d8_sft  EVAL_EVERY=100
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MID_TAG="${MID_TAG:-d8_mid}"
OUT_TAG="${OUT_TAG:-${MID_TAG%_mid}_sft}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
METRICS="$RESULTS_DIR/${OUT_TAG}_metrics.jsonl"
[ -f "$METRICS" ] && mv "$METRICS" "$METRICS.bak.$(date +%Y%m%d_%H%M%S)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu --format=csv -l 10 \
        > "$RESULTS_DIR/${OUT_TAG}_power.csv" 2>/dev/null &
    trap 'kill "$!" 2>/dev/null || true' EXIT
fi

echo "=== Stage 2 (SFT on SmolTalk): chatsft/$MID_TAG -> chatsft/$OUT_TAG ==="
run_logged "${OUT_TAG}_train" python -m scripts.chat_sft \
    --stage=sft --source=sft --model-tag="$MID_TAG" --output-tag="$OUT_TAG" \
    --device-batch-size=8 \
    --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" --chatcore-every=-1 \
    --log-file="$METRICS" "$@"

echo "DONE  checkpoint: $NANOCHAT_BASE_DIR/chatsft_checkpoints/$OUT_TAG   metrics: $METRICS"
ls -la "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$OUT_TAG"
