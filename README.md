# Building an LLM from scratch with nanochat — Assignment 1

Agentic LLMs, LIACS, Leiden University. Chapter 5 of the course book, Tasks 0–4.

This repository holds everything we wrote and everything our runs produced. It does **not** contain
nanochat itself, the training corpus, or the model checkpoints — those are large, external, and
reproducible from what is here.

---

## 1. What we actually did

We trained the same architecture at **four depths** (2, 6, 8, 10), fitted a compute-scaling law to
the four points, took the depth-8 model through a **two-stage** mid-training and SFT pipeline with two
controls, and then fine-tuned depth 10 with upstream's unmodified single-stage script so we had a
chattable model for Task 4.

| Run | Params | Tokens | Compute (FLOPs) | Val bits-per-byte | Wall clock |
|---|---:|---:|---:|---:|---|
| `d2` | 12,976,170 | 55,050,240 | 1.862e15 | 1.3314 | 13.2 min, 1 GPU (fp32) |
| `d6` | 73,531,538 | 278,396,928 | 5.452e16 | 1.0129 | 9.1 min, 2 GPUs |
| `d8` | 125,829,354 | 503,316,480 | 1.773e17 | 0.9393 | 41.6 min, 1 GPU |
| `d10` | 195,952,986 | 841,482,240 | 4.864e17 | 0.8878 | 161.8 min, 1 GPU |

Least squares on `L(C) = L∞ + a·C^(−α)` over those four points gives

```
L(C) = 0.623 + 356 · C^(−0.177)
```

with a maximum residual of 8×10⁻⁵ bpb across a 261× span in compute, and one degree of freedom left
over. Checked out of sample against the course text's depth-24 reference (C = 3.2e19, published
0.717 bpb) the fit predicts 0.750 — an error of +0.033 across a 65× extrapolation.

Two machines were involved. A laptop (RTX 5070 Ti, 16 GB, Blackwell sm_120) under WSL2 ran depth 8
and all of Tasks 3–4 against the course copy of nanochat. The LIACS node `vibranium` (2× RTX 3090,
24 GB, sm_86) ran depths 6 and 10 against upstream nanochat (`master`, September 2026). Before
mixing results from two code versions we verified they agree on `has_ve()`, the smear and backout
components, `num_scaling_params()`, the `--target-param-data-ratio` default of 12, the corpus, and
both learning-rate transfer rules.

---

## 2. Repository layout

```
.
├── scripts/        run on the GPU machine; these produce results/
├── analysis/       run on a laptop against results/; these produce report/figures/ and the numbers
├── results/        every log, metric and generation the runs emitted
├── report/         LaTeX source, bibliography, figures
└── docs/           setup runbook and the interim notes from 21 September
```

Nothing is duplicated: each artefact appears exactly once, and `analysis/` reads from `results/`
rather than keeping its own copies.

### What is deliberately absent

| Not here | Why | Where it is |
|---|---|---|
| `nanochat/` | upstream code, ~large, and our diff is what matters | clone separately, set `NANOCHAT_DIR` |
| `*.pt` checkpoints | hundreds of MB each | linked from the report |
| ClimbMix shards | 2.2 GB+ | fetched by `scripts/bootstrap_wsl.sh` |
| tokenizer binaries | rebuilt in 8 seconds | `scripts/10_train_tokenizers.sh` |

---

## 3. What we changed inside nanochat

Our edits sit on top of an unmodified baseline, one commit each, so `git log -p` in the nanochat
checkout shows exactly the diff. Four changes:

**`scripts/chat_sft.py` — staged fine-tuning.** Upstream runs one mixed stage. We split it:

- `--stage {full,mid,sft}` selects the `train_tasks` / `val_tasks` lists. `mid` is MMLU×3 plus
  GSM-8K×4; `sft` is SmolTalk; `full` is upstream's combined mixture.
- `--source {base,sft}` chooses which checkpoint family `load_model` and `load_optimizer_state`
  read, so stage 2 can resume from stage 1.
- `--output-tag` stops stage 2 overwriting stage 1's checkpoint.

**Checkpoint metadata.** Upstream inherits batch size and learning rates from the base checkpoint's
metadata. With two stages that breaks, because stage 2's source is no longer a base model. Stage-1
checkpoints now also store their `resolved_config`, and stage 2 inherits those settings instead of
re-deriving them from a depth that is not its source.

**Metrics logging.** Both `base_train.py` and `chat_sft.py` take `--log-file` and emit newline
-delimited JSON: one `config` record, one record per step with loss and train bpb, one per validation
pass, plus samples and a final summary. Every figure and every number in the report is computed from
these files.

**Exact assistant-token bpb.** `chat_sft.py` now reports bits-per-byte over supervised tokens only,
which makes the stage curves comparable with the pre-training curves.

---

## 4. `scripts/` — the run layer

Executed on the GPU machine, in this order. All of them `source scripts/env.sh`, which resolves
paths, activates the venv, and provides `run_logged` (tees stdout to a timestamped file in
`results/`) and `GPU_FLAGS`.

| Script | What it does |
|---|---|
| `bootstrap_wsl.sh` | Fresh WSL2 Ubuntu → build tools, `uv`, nanochat venv, corpus shards. Run once. |
| `env.sh` | Shared environment. **Source it, don't execute it.** Honours `NANOCHAT_DIR` and `NANOCHAT_BASE_DIR`. |
| `10_train_tokenizers.sh` | Trains two BPE tokenizers with `rustbpe` on 500M characters of ClimbMix: vocab 8,192 and 32,768. Copies each to `tokenizer_v<n>/` and leaves 32,768 in place as the one pre-training uses. Also runs upstream's `tok_eval`. |
| `11_tokenizer_analysis.py` | Task 1 evidence: bytes/token against GPT-2 and GPT-4 `cl100k` on a 2 MB held-out sample and on short probes, vocabulary composition, and the failure cases (digits, code, non-English, emoji). |
| `15_smoke_test.sh` | Runs all five pipeline stages on a throwaway depth-2 model so the real runs never die on a missing dataset or a broken kernel. ~5–10 minutes. |
| `20_pretrain.sh` | Task 2. `DEPTH=n bash scripts/20_pretrain.sh`. Starts an `nvidia-smi` power logger in the background (that CSV is what Task 4B's energy figure comes from), trains, then runs `base_eval` for bpb and samples. |
| `30_midtrain.sh` | Task 3 stage 1 — MMLU×3 + GSM-8K×4 from the base checkpoint. |
| `31_sft.sh` | Task 3 stage 2 — SmolTalk, resuming from stage 1. |
| `32_chat_eval.sh` | ARC-Easy, ARC-Challenge (categorical) and GSM-8K (generative, with the calculator tool) for any checkpoint. Appends one row per task to `eval_scores.csv`. |
| `35_run_task3_task4.sh` | Unattended chain: stage 1 → stage 2 → evaluate all three checkpoints → samples → figures. ~2–2.5 h at depth 8, dominated by sequential GSM-8K generation. |
| `inspect_datasets.py` | Prints token-level renderings of MMLU, GSM-8K and SmolTalk rows with the loss mask applied, which is how we found that only 0.8% of the MMLU example is supervised. |
| `40_samples.py` | Base-model completions and chat-template generations at a given temperature. |
| `50_compare_models.py` | Loads several checkpoints one at a time, runs identical prompts through each, frees VRAM between them, and writes a markdown table plus a JSON dump. Used for the depth probes, the base-vs-SFT comparison and the temperature sweep. |
| `plot_curves.py` | Training and validation bpb curves from the jsonl metrics, full-run and zoomed. |

### Typical invocations

```bash
export NANOCHAT_DIR=/path/to/nanochat
source scripts/env.sh

bash scripts/10_train_tokenizers.sh
bash scripts/15_smoke_test.sh
DEPTH=8 bash scripts/20_pretrain.sh
bash scripts/35_run_task3_task4.sh
```

Comparison and sweeps (run from the nanochat checkout, with its venv active):

```bash
python /path/to/this/repo/scripts/50_compare_models.py --models base:d6 base:d8 base:d10
python /path/to/this/repo/scripts/50_compare_models.py --models base:d10 sft:d10 --prompts chat --temperature 0.7 --name d10_stages
for T in 0.1 0.7 1.5; do
  python /path/to/this/repo/scripts/50_compare_models.py --models sft:d10 --prompts chat --temperature "$T" --max-tokens 128 --name "T$T"
done
```

---

## 5. `analysis/` — the post-hoc layer

Pure Python, no GPU, run from anywhere. `_paths.py` resolves everything relative to the repository
root, so `python analysis/fit_scaling_law.py` and `cd analysis && python fit_scaling_law.py` both
work.

| Script | What it does |
|---|---|
| `_paths.py` | Path resolution. Honours `NANOCHAT_BASE_DIR` for the depth-2 log, which predates this repo and stayed in the nanochat cache. |
| `fit_scaling_law.py` | Parses all four pre-training runs from **three different log schemas**, fits `L∞ + a·C^(−α)` by least squares over α with `a` and `L∞` solved linearly, reports residuals, checks the fit out of sample against the published depth-24 point, does the compute-matched comparison, and draws `report/figures/15_depth_series.png`. |
| `gpu_energy.py` | Trapezoidal integration of the `nvidia-smi` power trace, split at the pre-training / evaluation handover. |

Expected output of `fit_scaling_law.py`:

```
4-point fit: L(C) = 0.6232 + 356.4 C^-0.1769   SSE=1.07e-08  dof=1
   d2   measured 1.3314  fit 1.3314  resid +0.00000
   d6   measured 1.0129  fit 1.0128  resid -0.00004
   d8   measured 0.9393  fit 0.9394  resid +0.00008
   d10  measured 0.8878  fit 0.8877  resid -0.00005
   compute span 261x
   published d24 0.7170 -> fit 0.7495 (err +0.0325)
```

---

## 6. `results/` — what the runs produced

### `results/tokenizer/`
Training logs for both vocabularies (8.1 s at 8,192; 8.4 s at 32,768), upstream's `tok_eval`
output, and `tokenizer_analysis.{md,json}`. Headline: quadrupling the vocabulary removes 14.7% of
the tokens on in-domain text, 4.00 → 4.70 bytes per token, which is sub-linear in vocabulary size.

### `results/pretrain/`
- `d8_pretrain_metrics.jsonl` — the structured record for depth 8: config, every step, every
  validation pass.
- `d8_pretrain_*.log`, `d6_pretrain.log`, `d10_pretrain.log` — full stdout. Note these use three
  different formats, which is why `fit_scaling_law.py` has three parsers.
- `d8_power.csv` — 296 samples at 10-second intervals. Integrates to 0.2029 kWh at 247.5 W mean,
  262 W peak.
- `d{6,10}_meta_*.json` — checkpoint metadata, renamed by depth because the raw filenames collide.

### `results/finetune/`
`d10_sft.log`. Upstream's unmodified single-stage `chat_sft.py` on 789,759 rows (MMLU×3, GSM8K×4):
934 steps, 84.1 minutes on one 3090 at up to 81.8% MFU, validation bpb 0.6558 → **0.3851**. For
comparison the depth-8 `d8_full` single-stage control reached 0.421. The curve is still falling at
the last step, so the run is data-limited rather than converged.

### `results/eval/`
`d8_base_eval_*.log` — train bpb 0.947346, val bpb 0.941478, plus the unconditioned base-model
samples quoted in Task 2 Q4. `dataset_inspection.{md,json}` — the masked token renderings.

### `results/samples/`
- `compare_depths.{md,json}` — identical greedy probes on d6, d8, d10. Depth 8 answers the gold
  prompt with "the chemical symbol of gold is gold"; depth 10 answers "Au".
- `compare_d10_stages.{md,json}` — the same five prompts through `base:d10` and `sft:d10`. The base
  model continues text instead of answering it; the fine-tuned model answers in register and still
  cannot explain why the sky is blue.
- `compare_T{0.1,0.7,1.5}.{md,json}` and `task4_temperature_sweep.log` — Task 4A. Five prompts, three
  temperatures, top-k 50, seed 42, 128 tokens.

### `results/smoke/`
The five smoke-test logs, kept because they are the evidence that the pipeline was verified
end-to-end before the graded runs.

### Benchmark scores (depth-8 chain)

| Checkpoint | ARC-Easy | ARC-Challenge | GSM-8K |
|---|---|---|---|
| base `d8` | 25.1% | 23.0% | 0.0% |
| stage 1 `d8_mid` | **32.6%** | **31.3%** | **3.3%** |
| stage 2 `d8_sft` | 24.5% | 23.2% | 0.0% |
| (B) `d8_sft_replay` | 32.3% | 27.5% | 0.0% |
| (A) `d8_full` | 31.2% | 29.2% | 0.2% |

The same weights score 32.6% or 24.5% on ARC-Easy depending only on what the final stage trained on.

---

## 7. `report/`

`main.tex` plus `main.bib` (24 entries) and 13 figures, of which the report uses four. Build:

```bash
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## 8. Known gaps

Recorded here rather than papered over.

1. **The depth-8 chain's raw artefacts are incomplete.** `eval_scores.csv`,
   `samples_d8-base.md`, `samples_d8_sft.md` and the mid/SFT/control `*_metrics.jsonl` files were
   produced on a machine we lost access to before the deadline. The numbers in Section 6 trace to
   `docs/interim_results.md`, and the figures were recovered from the plotted PDF. The base-model
   samples survive inside `results/eval/d8_base_eval_20260921_134929.log`.
2. **Checkpoints are not in the repository** because of their size. They are published separately
   at <https://huggingface.co/chicoballer/nanochat-d8-assignment1>: the base model, both
   fine-tuning stages and both controls, with the tokenizer. Optimizer state is not included, so
   they load for inference and evaluation but cannot be resumed for further training.
3. **Depth 12 was never run.** The fit predicts 0.850 bpb against 0.821 from the floorless power
   law, which would have separated the two functional forms. We lost the shared node first. The
   prediction is recorded in the report; the test was not performed.
4. **A temperature-sweep table was cut for length.** The brief allows 8 pages including figures
   and references. The five verbatim excerpts it held are quoted in the prose of Section 7, which
   is where the argument was made anyway.
5. **Benchmark-sweep energy was never logged**, so Task 4B reports the training energy and leaves
   that term open rather than estimating it.

---

## 9. Reproducing from nothing

```bash
git clone <this repo> && cd nanochat-assignment1
git clone https://github.com/karpathy/nanochat ../nanochat
export NANOCHAT_DIR=$(cd ../nanochat && pwd)

bash scripts/bootstrap_wsl.sh          # venv + corpus shards
source scripts/env.sh
bash scripts/10_train_tokenizers.sh    # ~20 s
bash scripts/15_smoke_test.sh          # ~10 min, catches environment problems early
DEPTH=8 bash scripts/20_pretrain.sh    # ~42 min on a 5070 Ti
bash scripts/35_run_task3_task4.sh     # ~2.5 h

python analysis/fit_scaling_law.py     # needs all four runs present
python analysis/gpu_energy.py
```

Depths 6 and 10 are the same command with `DEPTH=6` and `DEPTH=10`. On a 24 GB card with another
user on it, drop `--device-batch-size` to 2 — depth 10 failed at 8 while allocating a
16,384 × 32,768 bf16 logits tensor, 1.07 GB.
