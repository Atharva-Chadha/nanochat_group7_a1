#!/usr/bin/env bash
# One-shot setup for a fresh WSL2 Ubuntu (or any Ubuntu with an NVIDIA GPU).
# Idempotent: safe to re-run, every step skips what is already done.
#
# This repository holds only our scripts, results and report. nanochat itself is a separate
# checkout; this script creates it, builds its virtualenv and downloads the corpus shards.
#
#   bash scripts/bootstrap_wsl.sh
#
# Steps:
#   1. apt packages (a C compiler is required by triton / torch.compile)
#   2. locate or clone nanochat            -> $NANOCHAT_DIR
#   3. uv + the locked virtualenv          -> $NANOCHAT_DIR/.venv
#   4. verify GPU, bf16 and torch.compile with nanochat's own tests
#   5. download NSHARDS ClimbMix shards    -> $NANOCHAT_BASE_DIR/base_data_climbmix
#
# Overrides:
#   NANOCHAT_DIR=../nanochat            where nanochat lives (cloned here if absent)
#   NANOCHAT_BASE_DIR=~/.cache/nanochat  data shards, tokenizers and checkpoints
#   NSHARDS=24                           train shards to fetch (+1 pinned validation shard)
#   SKIP_APT=1                           packages already installed
#   SKIP_TESTS=1                         skip the verification test run
set -euo pipefail

log() { printf '\n\033[1;32m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NANOCHAT_DIR="${NANOCHAT_DIR:-$(cd "$REPO_DIR/.." && pwd)/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NSHARDS="${NSHARDS:-24}"
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# ---------------------------------------------------------------- 0. sanity
[ -d "$REPO_DIR/scripts" ] || die "run this from the repository: bash scripts/bootstrap_wsl.sh"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "WARNING: nvidia-smi not found. Inside WSL2, CUDA comes from the *Windows* NVIDIA driver;"
    echo "         never install a Linux NVIDIA driver in WSL. Update the Windows driver instead."
fi

# ---------------------------------------------------------------- 1. apt packages
if [ "${SKIP_APT:-0}" = "1" ]; then
    log "apt packages: skipped (SKIP_APT=1)"
elif [ "$(id -u)" -eq 0 ]; then
    log "apt packages (running as root)"
    apt-get update -qq && apt-get install -y -qq build-essential git curl ca-certificates screen unzip
else
    log "apt packages (sudo may ask for your password)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential git curl ca-certificates screen unzip
fi
command -v gcc >/dev/null 2>&1 || die "gcc missing: torch.compile and triton need a C compiler (build-essential)"

# ---------------------------------------------------------------- 2. nanochat checkout
if [ -d "$NANOCHAT_DIR/.git" ] || [ -f "$NANOCHAT_DIR/pyproject.toml" ]; then
    log "nanochat found at $NANOCHAT_DIR"
else
    log "cloning nanochat into $NANOCHAT_DIR"
    git clone --quiet https://github.com/karpathy/nanochat "$NANOCHAT_DIR"
fi
cd "$NANOCHAT_DIR"

# ---------------------------------------------------------------- 3. uv + virtualenv
log "uv + virtualenv (first run downloads CPython 3.10 and torch 2.9.1+cu128, about 3 GB)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra gpu --group dev
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------- 4. verify the stack
log "verifying python / torch / GPU / attention / compile"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available: check nvidia-smi and the (Windows) NVIDIA driver"
prop = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0),
      "| VRAM %.1f GiB" % (prop.total_memory / 2**30))
import rustbpe, tiktoken, pyarrow, wandb, matplotlib, kernels  # noqa: F401  (all must import)
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanochat.flash_attention import HAS_FA3, USE_FA3
print("COMPUTE_DTYPE:", COMPUTE_DTYPE, "(", COMPUTE_DTYPE_REASON, ")")
print("FlashAttention-3:", "available" if HAS_FA3 else "not available", "| in use:", USE_FA3)
print("  (the SDPA fallback is expected on Blackwell, sm_120; use --window-pattern L there)")
PY

if [ "${SKIP_TESTS:-0}" = "1" ]; then
    log "nanochat tests: skipped (SKIP_TESTS=1)"
else
    python -m pytest -q -x tests/test_tokenizer.py tests/test_tasks.py tests/test_engine.py \
        tests/test_attention_fallback.py
    log "compiling the fused Muon/AdamW kernels (this is the real torch.compile + triton check, 1-3 min)"
    python -m pytest -q -x tests/test_optim.py
fi

# ---------------------------------------------------------------- 5. corpus shards
log "downloading $NSHARDS ClimbMix train shards + the pinned validation shard (~100 MB each)"
python -m nanochat.dataset -n "$NSHARDS"
du -sh "$NANOCHAT_BASE_DIR/base_data_climbmix"

log "BOOTSTRAP COMPLETE"
cat <<EOF

  repository     $REPO_DIR
  nanochat       $NANOCHAT_DIR
  data + models  $NANOCHAT_BASE_DIR   (not in git)

  Next:
    export NANOCHAT_DIR="$NANOCHAT_DIR"
    source scripts/env.sh
    bash scripts/10_train_tokenizers.sh
    bash scripts/15_smoke_test.sh
    DEPTH=8 bash scripts/20_pretrain.sh

EOF
