#!/usr/bin/env bash
# Common environment for all run scripts. Source it, don't execute it:
#   source scripts/env.sh
#
# Layout (relative to this repository's root):
#   scripts/              run scripts (this folder), executed on the GPU machine
#   analysis/             post-hoc analysis, executed on a laptop against results/
#   results/              logs, jsonl metrics, eval outputs -> committed (small text files)
#   report/               LaTeX source and figures
#   $NANOCHAT_BASE_DIR    data shards, tokenizers, checkpoints -> NOT in git (default ~/.cache/nanochat)
#   $NANOCHAT_DIR         the nanochat checkout itself -> NOT in git (default ../nanochat)

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1                 # same as runs/speedrun.sh; avoids CPU oversubscription in the tokenizer threads
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1                # logs reach the tee'd file line by line instead of in 8 KB chunks

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this repository
ASS_DIR="$REPO_DIR"                                            # kept for the older scripts that reference it
NANO_DIR="${NANOCHAT_DIR:-$(cd "$REPO_DIR/.." && pwd)/nanochat}"
RESULTS_DIR="$REPO_DIR/results"
mkdir -p "$RESULTS_DIR" "$NANOCHAT_BASE_DIR"

if [ ! -d "$NANO_DIR" ]; then
    echo "ERROR: no nanochat checkout at $NANO_DIR" >&2
    echo "       clone it there, or set NANOCHAT_DIR=/path/to/nanochat" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "$NANO_DIR/.venv/bin/activate" ]; then
    echo "ERROR: no virtualenv at $NANO_DIR/.venv - run scripts/bootstrap_wsl.sh first" >&2
    return 1 2>/dev/null || exit 1
fi
cd "$NANO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

# Log helper: run a command, tee everything to a timestamped log in results/, keep the exit status.
# usage: run_logged <name> <command...>
run_logged() {
    local name="$1"; shift
    local log="$RESULTS_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> $* " | tee "$log"
    echo ">>> log: $log"
    "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
}

# GPU flags that make sense on a single RTX 5070 Ti (16 GB, Blackwell sm_120 -> no FA3, SDPA fallback):
#   --window-pattern L      SDPA has no sliding-window kernel; with S windows the mask path is very slow
#   --device-batch-size 8   measured peak <= 13.4 GiB up to depth 10 (depth >= 12 needs 4)
GPU_FLAGS=(--window-pattern=L --device-batch-size=8)   # bash array; available because scripts *source* this file
