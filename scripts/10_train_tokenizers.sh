#!/usr/bin/env bash
# Task 1 - Tokenization.
# Train two BPE tokenizers with nanochat's Rust trainer (rustbpe) on a 500 MB sample of ClimbMix:
# vocab 8,192 and vocab 32,768. This is the "short script that reproduces your tokenizer training run".
#
# Details that matter for the report:
#   * --max-chars=500000000  -> tok_train.py streams train-split documents until 500M characters (~500 MB of
#                               mostly-ASCII text; the first 2 of the ~250M-char shards) and crops every
#                               document to --doc-cap=10000 chars (upstream default).
#   * vocab_size includes the 9 special tokens, so 8192 -> 8183 BPE merges + 256 byte tokens, etc.
#   * tok_train.py always writes to $NANOCHAT_BASE_DIR/tokenizer/, so each result is copied to
#     tokenizer_v<vocab>/ and the 32,768 one is left in place: it is the tokenizer pretraining uses.
#   * tok_train.py also writes token_bytes.pt (bytes per token id), needed for bits-per-byte evaluation.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MAX_CHARS="${MAX_CHARS:-500000000}"

for V in 8192 32768; do
    echo "=================== vocab_size=$V ==================="
    run_logged "task1_tok_train_v$V" python -m scripts.tok_train --max-chars="$MAX_CHARS" --vocab-size="$V"
    rm -rf "$NANOCHAT_BASE_DIR/tokenizer_v$V"
    cp -r "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer_v$V"
    echo "saved -> $NANOCHAT_BASE_DIR/tokenizer_v$V"
done

# Upstream compression-ratio report for the default (32,768) tokenizer vs GPT-2 / GPT-4 tokenizers.
run_logged "task1_tok_eval_v32768" python -m scripts.tok_eval

echo
echo "Tokenizers ready:"
ls -la "$NANOCHAT_BASE_DIR"/tokenizer "$NANOCHAT_BASE_DIR"/tokenizer_v8192 "$NANOCHAT_BASE_DIR"/tokenizer_v32768
