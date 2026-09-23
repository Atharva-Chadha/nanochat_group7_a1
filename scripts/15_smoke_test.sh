#!/usr/bin/env bash
# End-to-end smoke test (~5-10 min, mostly dataset downloads and torch.compile warm-up).
# Runs every pipeline stage on a throwaway depth-2 model tagged "smoke" so that the real runs
# never fail on a missing dependency, a broken kernel, or a missing dataset:
#   base_train -> base_eval (bpb + samples) -> chat_sft -> chat_eval (ARC-E, ARC-C, GSM8K) -> chat_cli
# It also pre-downloads SmolTalk, MMLU, GSM8K and ARC into $NANOCHAT_BASE_DIR/task_data.
# Requires: bootstrap done (venv + shards) and 10_train_tokenizers.sh (tokenizer at $NANOCHAT_BASE_DIR/tokenizer).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TAG=smoke

echo "=== [1/5] base_train: depth 2, 10 iterations ==="
run_logged "smoke_base_train" python -m scripts.base_train \
    --depth=2 --num-iterations=10 --model-tag="$TAG" \
    "${GPU_FLAGS[@]}" --total-batch-size=32768 \
    --eval-every=5 --eval-tokens=32768 --core-metric-every=-1 --sample-every=10

echo "=== [2/5] base_eval: bpb on train/val + samples ==="
run_logged "smoke_base_eval" python -m scripts.base_eval \
    --model-tag="$TAG" --eval=bpb,sample --device-batch-size=8 --split-tokens=32768

echo "=== [3/5] chat_sft: 5 iterations from the smoke base checkpoint (downloads SmolTalk/MMLU/GSM8K) ==="
run_logged "smoke_chat_sft" python -m scripts.chat_sft \
    --model-tag="$TAG" --num-iterations=5 \
    --eval-every=-1 --eval-tokens=32768 --chatcore-every=-1

echo "=== [4/5] chat_eval: ARC-Easy, ARC-Challenge, GSM8K on 8 problems each (downloads ARC) ==="
run_logged "smoke_chat_eval" python -m scripts.chat_eval \
    -i sft -g "$TAG" -a "ARC-Easy|ARC-Challenge|GSM8K" -x 8 -b 8 -m 64

echo "=== [5/5] chat_cli: one prompt ==="
run_logged "smoke_chat_cli" python -m scripts.chat_cli -i sft -g "$TAG" -t 0.7 -p "What is the capital of France?"

echo
echo "SMOKE TEST PASSED. Artifacts (throwaway):"
ls -la "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG" "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$TAG"
echo "Datasets cached under: $NANOCHAT_BASE_DIR/task_data"
du -sh "$NANOCHAT_BASE_DIR"/task_data/* 2>/dev/null || true
