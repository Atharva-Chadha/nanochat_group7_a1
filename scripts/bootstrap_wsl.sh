#!/usr/bin/env bash
# One-shot bootstrap for WSL2 Ubuntu. Idempotent: safe to re-run, every step skips what is already done.
#
# Run it from an Ubuntu (WSL2) terminal:
#   bash "/mnt/c/X/Masters/SEM 3/Agentic LLM/Ass 1/agentic-llm-assignment1.Building-LLMs-from-Scratch-main/assignment/bootstrap_wsl.sh"
#
# What it does (details in assignment/RUNBOOK.md):
#   1. apt packages (compiler for triton/torch.compile, git, curl, screen)
#   2. git identity + baseline commits in the Windows folder: "pristine nanochat" then "scaffolding"
#   3. clone into the WSL ext4 filesystem (~/agentic-llm-ass1) - much faster than working on /mnt/c
#   4. uv + locked virtualenv (CPython 3.10, torch 2.9.1+cu128, rustbpe, tiktoken, kernels, wandb, matplotlib, pytest)
#   5. verify GPU / bf16 / attention fallback / torch.compile with nanochat's own tests
#   6. download NSHARDS ClimbMix shards (+ the validation shard)
#   7. shell alias 'ass1'
#   8. unless SKIP_RUNS=1: train the two Task-1 tokenizers and run the end-to-end smoke test
#
# Overrides: WSL_REPO=~/agentic-llm-ass1  NSHARDS=24  NANOCHAT_BASE_DIR=~/.cache/nanochat  SKIP_RUNS=1
#            GIT_NAME="..." GIT_EMAIL="..."  (skip the identity prompts)   SKIP_APT=1 (packages already installed as root)
# Unattended from Windows (after the Linux user exists), e.g.:
#   wsl -d Ubuntu-24.04 -u root -- apt-get install -y build-essential git curl ca-certificates screen unzip
#   wsl -d Ubuntu-24.04 -- bash -lc 'GIT_NAME="A B" GIT_EMAIL=a@b.c SKIP_APT=1 bash "/mnt/c/.../assignment/bootstrap_wsl.sh"'
set -euo pipefail

log(){ printf '\n\033[1;32m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die(){ printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

WIN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WSL_REPO="${WSL_REPO:-$HOME/agentic-llm-ass1}"
NSHARDS="${NSHARDS:-24}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# ---------------------------------------------------------------- 0. sanity
grep -qi microsoft /proc/version 2>/dev/null || die "this script is meant for WSL2 Ubuntu (see assignment/RUNBOOK.md)"
[ -d "$WIN_REPO/nanochat-master" ] || die "cannot find nanochat-master next to assignment/ (resolved repo root: $WIN_REPO)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "WARNING: nvidia-smi not found. CUDA inside WSL2 comes from the *Windows* NVIDIA driver;"
    echo "         never install a Linux NVIDIA driver inside WSL. Update the Windows driver if CUDA is missing."
fi

# ---------------------------------------------------------------- 1. apt
if [ "${SKIP_APT:-0}" = "1" ]; then
    log "apt packages: skipped (SKIP_APT=1)"
elif [ "$(id -u)" -eq 0 ]; then
    log "apt packages (running as root)"
    apt-get update -qq && apt-get install -y -qq build-essential git curl ca-certificates screen unzip
else
    log "apt packages (your sudo password may be requested)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential git curl ca-certificates screen unzip
fi
command -v gcc >/dev/null 2>&1 || die "gcc missing: torch.compile/triton need a C compiler (install build-essential)"

# ---------------------------------------------------------------- 2. git identity
log "git identity (used for the graded commits)"
if [ -n "${GIT_NAME:-}" ]; then git config --global user.name "$GIT_NAME"; fi
if [ -n "${GIT_EMAIL:-}" ]; then git config --global user.email "$GIT_EMAIL"; fi
if [ -z "$(git config --global user.name 2>/dev/null || true)" ]; then
    read -rp "  git user.name (your full name): " gname; git config --global user.name "$gname"
fi
if [ -z "$(git config --global user.email 2>/dev/null || true)" ]; then
    read -rp "  git user.email: " gemail; git config --global user.email "$gemail"
fi
git config --global init.defaultBranch main
# /mnt/c paths trip git's "dubious ownership" check; a local clone also opens <repo>/.git directly
for d in "$WIN_REPO" "$WIN_REPO/.git"; do
    git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$d" || git config --global --add safe.directory "$d"
done
echo "  committing as: $(git config --global user.name) <$(git config --global user.email)>"

# ---------------------------------------------------------------- 3. baseline commits (Windows folder)
log "git repository at $WIN_REPO"
cd "$WIN_REPO"
[ -d .git ] || git init -q
if ! git rev-parse --verify -q HEAD >/dev/null; then
    git add .gitattributes README.md nanochat-master
    git commit -q -m "Pristine nanochat (course-provided version, line endings normalized to LF)"
    git add -A
    git update-index --chmod=+x assignment/bootstrap_wsl.sh assignment/scripts/*.sh
    git commit -q -m "Assignment scaffolding: WSL2 bootstrap, run scripts, runbook"
fi
git --no-pager log --oneline | head -5

# ---------------------------------------------------------------- 4. clone into WSL ext4
log "working copy in the WSL filesystem: $WSL_REPO"
if [ ! -d "$WSL_REPO/.git" ]; then
    git clone -q "$WIN_REPO" "$WSL_REPO"
else
    (cd "$WSL_REPO" && git pull -q --ff-only) || echo "  (pull skipped: local changes or diverged history, fine)"
fi
chmod +x "$WSL_REPO"/assignment/*.sh "$WSL_REPO"/assignment/scripts/*.sh
cd "$WSL_REPO/nanochat-master"

# ---------------------------------------------------------------- 5. uv + venv
log "uv + virtualenv (first time: downloads CPython 3.10 and torch 2.9.1+cu128, ~3 GB)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra gpu --group dev
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------- 6. verify the stack
log "verifying python / torch / GPU / attention / compile"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available inside WSL2: check 'nvidia-smi' and the Windows NVIDIA driver"
p = torch.cuda.get_device_properties(0)
print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0), "| VRAM %.1f GiB" % (p.total_memory / 2**30))
import rustbpe, tiktoken, pyarrow, wandb, matplotlib, kernels  # noqa: F401  (all must import)
from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanochat.flash_attention import HAS_FA3, USE_FA3
print("COMPUTE_DTYPE:", COMPUTE_DTYPE, "(", COMPUTE_DTYPE_REASON, ")")
print("Flash Attention 3:", "available" if HAS_FA3 else "not available", "| in use:", USE_FA3, "  (SDPA fallback is expected on Blackwell sm_120)")
PY
python -m pytest -q -x tests/test_tokenizer.py tests/test_tasks.py tests/test_engine.py tests/test_attention_fallback.py
log "compiling the fused Muon/AdamW kernels (validates torch.compile + triton on this GPU, 1-3 min)"
python -m pytest -q -x tests/test_optim.py

# ---------------------------------------------------------------- 7. data
log "downloading $NSHARDS ClimbMix train shards + the validation shard (~100 MB each) -> $NANOCHAT_BASE_DIR/base_data_climbmix"
python -m nanochat.dataset -n "$NSHARDS"
du -sh "$NANOCHAT_BASE_DIR/base_data_climbmix"

# ---------------------------------------------------------------- 8. shell convenience
log "shell setup (~/.bashrc): alias 'ass1'"
if ! grep -q 'agentic-llm-ass1 environment' ~/.bashrc; then
cat >> ~/.bashrc <<EOF

# --- agentic-llm-ass1 environment ---
export NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
export OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PATH="\$HOME/.local/bin:\$PATH"
alias ass1='cd "$WSL_REPO/nanochat-master" && source .venv/bin/activate'
EOF
fi

# ---------------------------------------------------------------- 9. first runs
if [ "${SKIP_RUNS:-0}" != "1" ]; then
    log "Task 1: training the two tokenizers (vocab 8,192 and 32,768 on 500 MB)"
    bash "$WSL_REPO/assignment/scripts/10_train_tokenizers.sh"
    log "end-to-end smoke test of every pipeline stage"
    bash "$WSL_REPO/assignment/scripts/15_smoke_test.sh"
fi

log "BOOTSTRAP COMPLETE"
cat <<EOF

  code (WSL, fast):    $WSL_REPO            <- work here. VS Code: 'code .' (Remote-WSL) or \\\\wsl.localhost\\Ubuntu\\home\\$USER
  code (Windows):      $WIN_REPO
                       ^ this is 'origin' for now. Once you create the GitHub repo:
                         cd "$WSL_REPO" && git remote set-url origin <github-url> && git push -u origin main
  data + checkpoints:  $NANOCHAT_BASE_DIR   (not in git)
  logs / results:      $WSL_REPO/assignment/results   (in git)
  new terminal:        type 'ass1' to cd into the code with the venv active

EOF
