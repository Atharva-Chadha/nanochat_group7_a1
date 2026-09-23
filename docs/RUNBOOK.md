# Assignment 1 runbook — Building LLMs from scratch with nanochat

Everything needed to go from a fresh Windows PC to the four graded tasks. Spec: `Ass1Build.pdf`
(chapter 5 of the course book). Code: `nanochat-master/` (Karpathy's nanochat, course-provided
March-2026 version). Our additions live in `assignment/` so the diff against pristine nanochat stays clean.

## 0. Machine and the one decision already taken

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB, Blackwell sm_120 → bf16 yes, **Flash-Attention-3 no** (PyTorch SDPA fallback), fp8 not used |
| CPU / RAM / disk | Ryzen 7 9700X, 31 GB, ~700 GB free |
| OS | Windows 11 Pro. nanochat is Linux-first (`torch.compile`/triton, `signal.SIGALRM`, `resource`, `uv`) |
| **Decision** | **Run everything inside WSL2 Ubuntu.** Native Windows would need MSVC + triton-windows + source patches and still be off the tested path. |

Measured on this GPU (eager mode, i.e. before `torch.compile`, so pessimistic), default nanochat horizon:

| depth | params | tokens / iters | device batch | peak VRAM | pretrain time | shards |
|---|---|---|---|---|---|---|
| 2 | 13M | 55M / 420 | 8 | 8.3 GiB | ~5 min | 2 |
| 4 | 37M | 138M / 528 | 8 | 9.0 GiB | ~14 min | 5 |
| 6 | 74M | 278M / 1,062 | 8 | 10.0 GiB | ~40 min | 9 |
| 8 | 126M | 503M / 1,920 | 8 | 11.5 GiB | ~1.6 h | 15 |
| 10 | 196M | 841M / 1,605 | 8 | 13.4 GiB | ~3.8 h | 25 |
| 12 | 286M | 1.32B / 2,520 | 4 | 8.9 GiB | ~9.6 h | 39 |

The spec asks for **depth 2 (or 4)** and forbids depth ≥ 12. Decision (2026-09-21): **depth 8** for the graded
deliverables — it fits the 8-hour-per-run budget with margin and compute allows it; say so explicitly in the report.
Extra depth sweep (2/4/6) and a ratio sweep on depth 4 remain cheap evidence for the scaling-law questions.

## 1. One-time Windows steps (need Administrator, ~10 min + reboot)

1. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Reboot when asked. After the reboot Ubuntu opens (or start "Ubuntu 24.04 LTS" from the Start menu) and asks for a
   UNIX username + password — this password is what `sudo` asks for later.
   (Done on this machine: WSL2 platform installed by the user, distro `Ubuntu-24.04` installed with `--no-launch`;
   the first interactive launch creates the user. Fallback if it drops you into a root shell instead:
   `adduser <name> && usermod -aG sudo <name>`, then write `[user]` / `default=<name>` into `/etc/wsl.conf`
   and run `wsl --shutdown` from PowerShell.)
2. GPU: nothing to install. WSL2 uses the Windows NVIDIA driver (616.xx is fine). Inside Ubuntu `nvidia-smi`
   must list the 5070 Ti. **Never install a Linux NVIDIA driver inside WSL.**
3. Optional but recommended, `C:\Users\<you>\.wslconfig` (WSL defaults to 50 % of RAM):
   ```ini
   [wsl2]
   memory=24GB
   ```
   then `wsl --shutdown` from PowerShell once.

## 2. Bootstrap (inside Ubuntu, ~20–40 min, mostly downloads)

```bash
bash "/mnt/c/X/Masters/SEM 3/Agentic LLM/Ass 1/agentic-llm-assignment1.Building-LLMs-from-Scratch-main/assignment/bootstrap_wsl.sh"
```

Unattended variant, driven from a Windows terminal once the Linux user exists (root steps via `-u root`, no password):
```powershell
wsl -d Ubuntu-24.04 -u root -- bash -c "apt-get update -qq && apt-get install -y -qq build-essential git curl ca-certificates screen unzip"
wsl -d Ubuntu-24.04 -- bash -lc 'GIT_NAME="Atharva Chadha" GIT_EMAIL="chadhaatharva2002@gmail.com" SKIP_APT=1 bash "/mnt/c/X/Masters/SEM 3/Agentic LLM/Ass 1/agentic-llm-assignment1.Building-LLMs-from-Scratch-main/assignment/bootstrap_wsl.sh"'
```

It is idempotent; if it stops, fix the reported problem and re-run it. It will:

1. `apt install build-essential git curl screen` (compiler needed by triton / `torch.compile`).
2. Ask once for your git name/email, then create the baseline history **in the Windows folder**:
   `Pristine nanochat` → `Assignment scaffolding`. Every later change is a clean diff against upstream.
3. Clone into `~/agentic-llm-ass1` (WSL ext4; `/mnt/c` is 10–50× slower for Python imports and parquet reads).
4. Install `uv` and the locked environment: `uv sync --extra gpu --group dev` (CPython 3.10, torch 2.9.1+cu128).
5. Verify: CUDA visible, bf16 auto-detected, FA3→SDPA fallback, and `tests/test_optim.py` which compiles the
   fused Muon/AdamW kernels — if `torch.compile` is broken it fails here, not 20 minutes into a run.
6. `python -m nanochat.dataset -n 24` → **only** 24 train shards + the val shard (~2.5 GB of the 6,543-shard /
   ~650 GB `karpathy/climbmix-400b-shuffle` dataset), exactly the mechanism `runs/speedrun.sh` uses (`-n 8`, `-n 170`).
   Depth 8 at the default horizon needs ~15 shards; 24 leaves margin for a ratio sweep on depth 4.
7. Add the `ass1` alias (cd into the code + activate venv) to `~/.bashrc`.
8. Run `assignment/scripts/10_train_tokenizers.sh` (Task 1, ~1 min) and `15_smoke_test.sh` (every stage on a
   10-step depth-2 model; also pre-downloads SmolTalk, MMLU, GSM8K, ARC).

When it prints `BOOTSTRAP COMPLETE`, create the shared GitHub repo (private, add your partner) and:
```bash
cd ~/agentic-llm-ass1 && git remote set-url origin git@github.com:<you>/<repo>.git && git push -u origin main
```

## 3. Daily workflow

```bash
ass1                                  # cd ~/agentic-llm-ass1/nanochat-master + venv
screen -S run                         # long runs survive a closed terminal; Ctrl-A D detaches, screen -r run reattaches
bash ../assignment/scripts/<script>.sh
```

Two repos, one history: code is edited and committed in the Windows folder (`origin` of the WSL clone), runs and
their results are committed in the WSL clone. Keep it linear: `git pull --ff-only origin main` in WSL before a
run, commit results in WSL, then on Windows `git pull --no-rebase wsl main` (remote `wsl` =
`//wsl.localhost/Ubuntu-24.04/home/chico/agentic-llm-ass1`). Once the GitHub remote exists both sides push there.

Driving WSL from Windows without an interactive terminal: a detached
`Start-Process wsl.exe -ArgumentList '-d Ubuntu-24.04 -- bash -lc "..."'` survives the tool call, whereas
`nohup ... &` inside a `wsl.exe` session dies with the session. `wsl.exe -- bash -c '<complex command>'` is
re-parsed by the login shell (quotes lost), so anything non-trivial goes into a script file run as
`MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/.../script.sh`.

- Every script tees its output to `assignment/results/<name>_<timestamp>.log`. Commit results with the code
  that produced them, one commit per experiment (`git add -A && git commit -m "d4 pretrain, ratio 12"`).
- Checkpoints live in `~/.cache/nanochat/{base,chatsft}_checkpoints/<tag>/` (not in git). For the submission,
  upload the final ones to a Hugging Face model repo or a Drive folder and link them in the report.
- Watch the GPU in a second terminal: `nvidia-smi -l 5`. Peak must stay < 14 GiB (the desktop uses ~1 GiB).
  If a run is suddenly 10× slower, VRAM spilled to system RAM — lower `--device-batch-size`.
- Log power for the environmental-cost prompt (Task 4B):
  `nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu --format=csv -l 10 > ../assignment/results/power_<run>.csv`

## 4. Task plan and the code changes each one needs

| # | Task | Script | Code changes in `nanochat-master/` (to do next, then test on the smoke model) |
|---|---|---|---|
| 1 | Tokenizers 8k / 32k on 500 MB | `10_train_tokenizers.sh` ✅ | none for training. Analysis script (TBD): compression + sequence length on English/code/numbers/Korean for both vocabs, plus a pure-Python BPE with merge counts for the merge-tree figure |
| 2 | Pretrain depth 8 + bpb curves | `20_pretrain.sh` ✅ (`DEPTH`, `RATIO`, `TAG` env overrides) | ✅ `scripts/base_train.py`: exact **train bpb** per step (same definition as `evaluate_bpb`, via `loss_reduction='none'` + `.mean()`, identical gradient) and `--log-file` **`.jsonl`** with config / every step / every val / samples / final; ✅ `nanochat/common.py`: RTX 5070 Ti & 5080 peak FLOPS (dense bf16, fp32-accumulate) and bandwidth so MFU/MBU are real. Plot script for the curves: TBD |
| 3 | Mid-training → SFT | `30_midtrain.sh` ✅ (`BASE_TAG=d8` → `d8_mid`), `31_sft.sh` ✅ (`MID_TAG=d8_mid` → `d8_sft`), `32_chat_eval.sh <base\|sft> <tag> [label]` ✅ → `results/eval_scores.csv`; `inspect_datasets.py` ✅ (sanity check + loss-mask rendering → `results/dataset_inspection.md`) | ✅ `scripts/chat_sft.py`: `--stage {full,mid,sft}` selects `train_tasks`/`val_tasks`; `--source {base,sft}` for `load_model`/`load_optimizer_state`; `--output-tag` (stage 2 no longer overwrites stage 1); stage-1 checkpoints store `resolved_config` + batch keys so stage 2 inherits them; `--log-file` jsonl; exact loss (`sum/#unmasked` ≡ upstream mean with ignore_index) and train bpb on assistant tokens |
| 4 | Inference, temperature, benchmarks | `40_samples.py --source base\|sft --model-tag <tag>` ✅ → `results/samples_<label>.md` (5 unconditioned + 5 prompts × T=0.1/0.7/1.5) | none required (`chat_cli.py -t`, `engine.py`). Note: this nanochat has **no web UI** (`chat_web.py`/`ui.html` were removed upstream) — say so in the report or port them from an older commit |

Recommended run order: `20_pretrain.sh` (d8, includes base_eval) → chat_eval of the **base** model (`-i base -g d8`) →
mid-training → eval → SFT → eval → samples at T=0.1/0.7/1.5 → depth sweep (`DEPTH=2/4/6 bash 20_pretrain.sh`) →
ratio sweep (`DEPTH=4 RATIO=6|24|48 bash 20_pretrain.sh`).

Flags that stay fixed on this GPU (already in `scripts/env.sh` as `GPU_FLAGS`): `--window-pattern=L` (SDPA has
no sliding-window kernel) and `--device-batch-size=8`. Also always pass `--core-metric-every=-1` (CORE is not
required and slow) and a small `--eval-tokens` (the default 42M-token val eval is longer than a depth-4 run).

## 5. Things the spec says that differ from this code version (state them in the report)

- No web UI / inference server in this version, only `scripts/chat_cli.py`.
- The spec's SFT mixture mentions identity + spelling tasks (~1.1M examples); here it is SmolTalk + MMLU×3 + GSM8K×4 (~790K).
- Spec and `dev/LEADERBOARD.md` quote a compute-optimal ratio of 10.5; `base_train.py` defaults to `--target-param-data-ratio=12`.
- Spec says nanochat has "pretraining and SFT" only — true here (older versions also had `mid_train.py`; we add the stage ourselves).
- `dev/LOG.md` explains *why* the single-dial hyperparameters are what they are — cite it for Task 2 Q1/Q2.

## 6. Gotchas we already hit

- The course zip had CRLF line endings on every file; converted to LF and enforced via `.gitattributes`
  (CRLF shell scripts fail under Linux with `$'\r': command not found`).
- Windows NVIDIA driver silently spills VRAM into system RAM instead of raising OOM (10× slowdown). Under WSL2
  keep an eye on `nvidia-smi`; on native Windows set *CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback*.
- `tok_train.py` always writes to `$NANOCHAT_BASE_DIR/tokenizer/`; the Task 1 script copies each result to
  `tokenizer_v<vocab>/` and leaves the 32,768 one in place for pretraining.
- Depth 2 has a single attention head (n_embd 128 / head_dim 128); depth 4 has two. Both are ~90 % embeddings.
- `chat_eval.py` GSM8K is generative and sequential (1,319 problems × up to 512 tokens): use `-x <n>` for quick
  checks and the full set only for the final table. ARC is categorical (fast).
- The calculator tool in `engine.py` uses `signal.SIGALRM` — works on Linux only (another reason for WSL2).
- The Hub `kernels-community/flash-attn3` build *loads* on Blackwell (sm_120) but the first launch aborts the whole
  process with `no kernel image is available for execution on the device` (no Python traceback). Fixed in
  `nanochat/flash_attention.py`: FA3 is only loaded on capability 8.x/9.x; everything else uses SDPA.
- Under `torch.compile` the real numbers are much better than the eager benchmark: depth 8 runs at ~200K tok/s,
  MFU ~80 % (of 87.9 TFLOPS dense bf16), 7 GB VRAM, ~35 min for the default horizon.
- Dataset facts (`results/dataset_inspection.md`): SmolTalk 460,341 rows ≈ 364M tokens/epoch (73 % supervised);
  MMLU auxiliary_train 99,842 rows ≈ 38M tokens/epoch but only **2 supervised tokens per example** (answer letter +
  `<|assistant_end|>`); GSM8K 7,473 rows ≈ 1.3M tokens (56 % supervised). Stage 1 ≈ 120M tokens, stage 2 ≈ 364M.
