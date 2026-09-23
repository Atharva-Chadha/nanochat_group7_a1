#!/usr/bin/env bash
# Task 2 - Pre-training on ClimbMix with nanochat's single dial (--depth), then base-model evaluation.
#
# Usage:  bash 20_pretrain.sh [extra scripts.base_train args]
# Env:    DEPTH=8  RATIO=12 (tokens per scaling-param; nanochat default)  TAG=d<DEPTH>[_r<RATIO>]
#         EVAL_EVERY=50  EVAL_TOKENS=4194304  SAVE_EVERY=500
#
# Depth 8 on the RTX 5070 Ti: ~126M params, 503M tokens, 1,920 steps, ~11.5 GiB, ~1.6 h eager (less with compile).
# Note: the spec suggests --depth 2 (or 4); depth 8 was chosen deliberately because compute allows it.
#
# What this run leaves behind (all needed for the report):
#   $NANOCHAT_BASE_DIR/base_checkpoints/<TAG>/model_*.pt + optim_*.pt + meta_*.json   checkpoints (every SAVE_EVERY + final)
#   assignment/results/<TAG>_pretrain_metrics.jsonl   config + every step (loss, train bpb) + every val bpb + samples + final
#   assignment/results/<TAG>_pretrain_*.log           full stdout
#   assignment/results/<TAG>_power.csv                GPU power/memory/utilization every 10 s (Task 4B carbon estimate)
#   assignment/results/<TAG>_base_eval_*.log          train/val bpb on 8M tokens each + conditioned/unconditioned samples
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DEPTH="${DEPTH:-8}"
RATIO="${RATIO:-12}"
if [ "$RATIO" = "12" ]; then TAG="${TAG:-d${DEPTH}}"; else TAG="${TAG:-d${DEPTH}_r${RATIO}}"; fi
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"   # 4M tokens per val eval (~15 s at depth 8). Upstream default (42M) takes longer than a small run.
SAVE_EVERY="${SAVE_EVERY:-500}"
METRICS="$RESULTS_DIR/${TAG}_pretrain_metrics.jsonl"

if [ -f "$METRICS" ]; then
    mv "$METRICS" "$METRICS.bak.$(date +%Y%m%d_%H%M%S)"   # never mix two runs in one metrics file
fi

# GPU power/memory log for the environmental-cost question (Task 4B); stops when this script exits
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu,temperature.gpu --format=csv -l 10 \
        > "$RESULTS_DIR/${TAG}_power.csv" 2>/dev/null &
    POWER_PID=$!
    trap 'kill "$POWER_PID" 2>/dev/null || true' EXIT
fi

echo "=== pretraining: depth=$DEPTH ratio=$RATIO tag=$TAG ==="
run_logged "${TAG}_pretrain" python -m scripts.base_train \
    --depth="$DEPTH" --target-param-data-ratio="$RATIO" --model-tag="$TAG" \
    "${GPU_FLAGS[@]}" \
    --eval-every="$EVAL_EVERY" --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=-1 --sample-every=250 --save-every="$SAVE_EVERY" \
    --log-file="$METRICS" "$@"

echo "=== base-model evaluation: bpb on train/val (8M tokens each) + samples ==="
run_logged "${TAG}_base_eval" python -m scripts.base_eval \
    --model-tag="$TAG" --eval=bpb,sample --device-batch-size=8 --split-tokens=8388608

echo
echo "DONE  tag=$TAG"
echo "  metrics:     $METRICS  ($(wc -l < "$METRICS") lines)"
echo "  checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/$TAG"
ls -la "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"
