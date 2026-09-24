# How this project works, step by step

Plain-language tour of the whole pipeline: what each file does, why it exists, and what it produced.
Written to be read top to bottom. Technical terms are defined the first time they appear.

---

## 0. The one-paragraph version

We built a small ChatGPT-style language model from scratch, four times over at different sizes, using
Karpathy's **nanochat** codebase. First we built a **tokenizer** (the thing that turns text into
numbers). Then we **pre-trained** models to predict the next token on half a billion words of web
text. Then we **fine-tuned** one of them in two stages to make it answer questions instead of just
continuing text — and discovered that the second stage erased what the first stage taught. Then we
measured how fast it generates text and what it costs in electricity.

---

## 1. The vocabulary you need

| Term | Plain meaning |
|---|---|
| **Token** | A chunk of text the model treats as one unit. Usually a word or word-piece: `photosynthesis` is 1 token, `photo`+`synthesis` is 2. |
| **Tokenizer** | The dictionary that converts text ↔ token numbers. Ours has 32,768 entries. |
| **BPE** | Byte-Pair Encoding: how the tokenizer is built. Start with single bytes, repeatedly glue the most common adjacent pair into a new token. |
| **Pre-training** | Showing the model raw text and training it to predict the next token. This is where it learns grammar. |
| **Fine-tuning (SFT)** | Further training on question→answer examples, so it answers rather than continues. |
| **bpb** (bits per byte) | Our score for "how well does the model predict text". Lower is better. Unlike raw loss, it is comparable between different tokenizers. |
| **FLOPs** | Floating-point operations — a count of arithmetic done. Our unit of "how much compute did this cost". |
| **Checkpoint** | The saved weights of a model at one moment. |
| **Depth** | Number of transformer layers. nanochat derives *everything else* from this one number. |

---

## 2. The shape of the repository

```
scripts/     run on a GPU machine. These produce results/.
analysis/    run afterwards on any laptop. These read results/ and produce report/figures/ + numbers.
results/     every log, metric and generation our runs emitted.
report/      the LaTeX source, bibliography and figures.
docs/        this file, the runbook, the interim notes, the cleanup log.
```

Two rules that keep it honest:

1. **`scripts/` writes, `analysis/` reads.** No analysis script touches the GPU; no run script
   computes a reported number. So any number in the report can be recomputed from committed files.
2. **Nothing large is in git.** nanochat itself, the 2.2 GB corpus and the checkpoints live outside,
   under `$NANOCHAT_DIR` and `$NANOCHAT_BASE_DIR`. The repository is text.

---

## 3. Setup — `scripts/bootstrap_wsl.sh`, `scripts/env.sh`

### `bootstrap_wsl.sh` — run once on a fresh machine

Does five things in order: installs a C compiler and git; clones nanochat next door; builds its
virtual environment with `uv`; **verifies** the machine actually works; downloads 24 chunks of the
training corpus.

The verification step is the interesting one. It runs nanochat's own test suite, and in particular
`tests/test_optim.py`, because that test compiles the fused optimizer kernels. If `torch.compile` or
Triton is broken on this machine, the failure appears here in two minutes instead of twenty minutes
into a training run.

### `env.sh` — **sourced**, not executed, by every run script

```bash
source scripts/env.sh
```

It must be *sourced* because it changes the current shell: it activates the Python environment and
defines a bash array. If you `bash scripts/env.sh` instead, everything happens in a child shell and
then vanishes.

It sets:
- `NANOCHAT_BASE_DIR` — where data, tokenizers and checkpoints live (default `~/.cache/nanochat`)
- `NANOCHAT_DIR` — the nanochat checkout (default `../nanochat`)
- `OMP_NUM_THREADS=1` — stops the tokenizer's threads fighting each other for CPU
- `PYTHONUNBUFFERED=1` — so a log file fills line by line rather than in 8 KB blocks; important when
  you are watching a run over SSH
- `run_logged` — a helper that runs a command, shows it on screen **and** writes it to a timestamped
  file in `results/`. This is why every run has a log.
- `GPU_FLAGS` — `--window-pattern=L --device-batch-size=8`, explained in section 5.

---

## 4. Task 1 — building the tokenizer

### Step 1: train two tokenizers — `10_train_tokenizers.sh`

Runs nanochat's Rust BPE trainer twice on the same 500 million characters of text: once producing a
**8,192**-word vocabulary, once **32,768**. Takes 8 seconds each. We trained two so we could measure
what vocabulary size actually buys.

A detail worth knowing: nanochat always saves to the same folder, so the script copies each result to
`tokenizer_v8192/` and `tokenizer_v32768/` and leaves the 32k one in place as the one pre-training
will use.

### Step 2: measure them — `11_tokenizer_analysis.py`

Encodes the same texts with our two tokenizers and with GPT-2's and GPT-4's, and reports
**bytes per token** — how much text fits in one token. Higher is better.

The headline: 4× the vocabulary buys **14.7% shorter sequences** on our own kind of text. Not 4×
better — the returns are strongly sub-linear, which is the whole point of the question.

It also probes failure cases: numbers, code, German, Korean, Hindi, emoji.

### Step 3: recover the merge counts — `12_merge_counts.py`

*This is the cleverest script in the repository and a likely viva topic.*

**The problem.** Task 1 asks for a merge tree "with the counts for your merges". But the trainer only
prints a progress line every 1% — 100 lines for 32,503 merges. The count behind any *particular*
merge is not in the log.

**The solution — replay the training.**

1. Re-create the exact text sample the tokenizer saw, split into chunks the same way, and count how
   often each chunk occurs.
2. Recover the **order** the merges were learned in. Trick: a token of rank *r* was built from two
   parents. Re-encode its raw bytes allowing only merges of rank *below r*. That necessarily stops
   at exactly two pieces — its parents. This reads the answer out of the saved tokenizer instead of
   guessing how the trainer broke ties.
3. Replay the merges in that order over the counted chunks, recording the frequency of each pair
   *just before* it is merged.

**How we know it is right.** `--check-log` compares the replayed counts against every frequency the
trainer did print. All 100 match, for both vocabularies. That is the verification, and it is why the
numbers in Figure 1 can be trusted.

### Step 4: draw the tree — `13_merge_tree.py`

Takes one word (` photosynthesis`), walks the parent relationship recursively, and draws the tree with
each merge's rank and count. Merges numbered ≥ 8,183 are drawn **dashed**: that is exactly where the
8k tokenizer's vocabulary ends (256 bytes + 7,927 merges), so the dashed part is *what the smaller
tokenizer never learns to build*. One figure answers "what does vocabulary size do?" visually.

### Step 5: how often is each token actually used? — `14_token_usage.py`

Counts token occurrences over 20M held-out characters. Finding: **512 of the 32k tokens never appear
at all**, against 65 of the 8k ones, and the median 32k token appears 22 times versus 189. Since a
token's embedding row only gets a gradient when that token appears, a bigger vocabulary means a longer
tail of rows the model barely trains. That is the evidence behind Task 1 Q3.

---

## 5. Task 2 — pre-training

### The run — `20_pretrain.sh`

```bash
DEPTH=8 bash scripts/20_pretrain.sh
```

What happens:

1. Starts `nvidia-smi` in the background logging power every 10 seconds → this CSV is where the
   Task 4B energy number comes from. A `trap` kills it when the script exits.
2. Runs nanochat's `base_train.py`.
3. Runs `base_eval.py` afterwards for final train/validation bpb and sample text.

**The two GPU flags, and why:**

- `--window-pattern L` — nanochat normally has most layers attend to only a quarter of the context
  (cheaper). That needs FlashAttention-3, which has no kernel for the RTX 5070 Ti's Blackwell
  architecture. The fallback implementation has no sliding-window support and emulates it with a
  slow mask, so we turned windows off entirely: every layer sees the full context.
- `--device-batch-size 8` — how many sequences go through the GPU at once. It does **not** change the
  maths: nanochat accumulates gradients until it reaches the real batch of 262,144 tokens regardless.
  It only trades memory for speed.

**What "depth 8" derives.** You give one number. The code computes: width = 64 × depth rounded to a
multiple of 128 (→ 512); heads = width / 128 (→ 4); training tokens = 12 × scaling parameters
(→ 503M); batch size from a power law fitted at depth 12; learning rates ∝ √batch; weight decay from
batch and horizon. One integer → an entire configuration.

**Result:** 125.8M parameters, 1,920 steps, 41.6 minutes, validation bpb from 3.166 down to **0.9393**.

### The curves — `plot_curves.py`

Reads the `.jsonl` metrics file (one JSON object per line: one for the config, one per step, one per
validation pass) and draws training and validation bpb. It smooths the training curve with a
**centred** moving average rather than a running one, because a running average lags behind and the
first phase of training falls very steeply.

### Four depths, one law — `analysis/fit_scaling_law.py`

We ran depths 2, 6, 8 and 10 — 13M to 196M parameters, a 261× span of compute. This script:

1. Parses all four runs. They are in **three different log formats**, so it has three parsers.
2. Fits `L(C) = L∞ + a·C^(−α)`: loss falls as a power of compute but flattens towards a floor `L∞`,
   because text has irreducible entropy — no model reaches zero.
3. Checks the fit against a run it never saw: the course text's depth-24 model.

**Result:** `L(C) = 0.623 + 356·C^(−0.177)`, predicting the depth-24 model to within **0.033 bpb**
across a 65× extrapolation.

**The honest caveat, which you should volunteer.** Three parameters fitted to four points leaves one
degree of freedom. The tiny residuals (8×10⁻⁵) are *not* evidence the model is right — with four
points and three parameters almost any smooth decreasing curve fits that well. The out-of-sample
depth-24 check is the only real test, and that is why it is in the report.

### Energy — `analysis/gpu_energy.py`

Integrates the power log by the trapezoid rule (average two neighbouring readings, multiply by the
gap, sum) and splits at the moment training handed over to evaluation. **0.199 kWh** for the training,
0.004 for the evaluation.

---

## 6. Task 3 — fine-tuning in two stages

### What we changed inside nanochat

Upstream runs **one** fine-tuning stage on a mixture of everything. The assignment asks us to split
it. Four changes to `chat_sft.py`:

1. `--stage {full,mid,sft}` — picks the data. `mid` = MMLU ×3 + GSM-8K ×4 (reasoning and knowledge);
   `sft` = SmolTalk (conversation); `full` = upstream's mixture.
2. `--source {base,sft}` — which family of checkpoints to resume from, so stage 2 can start from
   stage 1 instead of from the base model.
3. `--output-tag` — without it, stage 2 **silently overwrites stage 1's checkpoint**.
4. **Checkpoint metadata.** Upstream copies batch size and learning rates from the base checkpoint.
   That breaks when the source is itself a fine-tuned model, so stage 1 now saves its resolved
   settings and stage 2 inherits those.

### Loss masking — the key concept

In a conversation the model should only be graded on the **assistant's** words, not the user's
question. `render_conversation` returns a mask that is 1 on assistant tokens. Everything else gets
target `−1`, which cross-entropy ignores.

The numbers make it vivid: in one MMLU example, **2 of 237 tokens are supervised** (the answer letter
and the end marker). Without masking, 99% of the training signal would teach the model to write exam
questions.

Tool output is also masked to 0. When the model writes `<|python_start|>3+4<|python_end|>` the engine
runs it and inserts `7`. The model must learn to *call* the calculator but never to *predict its
answer* — at test time that comes from the interpreter.

`inspect_datasets.py` prints these token-by-token renderings with the mask, which is how we found the
MMLU figure.

### The runs — `30_midtrain.sh`, `31_sft.sh`, `35_run_task3_task4.sh`

`35_run_task3_task4.sh` chains everything unattended: stage 1 → stage 2 → evaluate all three
checkpoints → sample → plot. About 2.5 hours, most of it GSM-8K, which must generate an answer for
each of 1,319 problems one at a time.

### The result — and the surprise

| Checkpoint | ARC-Easy | ARC-Challenge | GSM-8K |
|---|---|---|---|
| base | 25.1 | 23.0 | 0.0 |
| **stage 1** (reasoning data) | **32.6** | **31.3** | **3.3** |
| stage 2 (conversation only) | 24.5 | 23.2 | 0.0 |
| control B: stage 2 + replay | 32.3 | 27.5 | 0.0 |
| control A: one mixed stage | 31.2 | 29.2 | 0.2 |
| chance | 25 | 25 | 0 |

Stage 1 lifts the model 7–8 points above chance. **Stage 2 puts it straight back to chance.** This is
*catastrophic forgetting*: one epoch of conversation data re-fits the output distribution and the
multiple-choice skill disappears.

Two controls tell us why:

- **Control B** mixes one epoch of the stage-1 data back into stage 2 → 32.3%, as good as stage 1. So
  the cause was the stage-2 *data*, not anything permanent in the weights.
- **Control A** runs upstream's single mixed stage → 31.2%. Statistically the same as the two-stage
  route, in **one** run instead of two. At this size, separating the stages buys nothing.

### Comparing checkpoints — `50_compare_models.py`

Loads several checkpoints **one at a time**, runs identical prompts through each, and frees the GPU
between them, so a whole series fits on one card. Used for the depth comparison (where "Au" as the
symbol for gold appears only at depth 10) and the temperature sweep.

---

## 7. Task 4 — generation, speed and cost

### How generation works

The model predicts one token, you append it, and repeat. Done naively, every step re-reads the entire
text so far, costing O(T²).

The **KV cache** fixes this. Attention needs a "key" and "value" for every previous token; those never
change once computed. So nanochat computes them once for the prompt (**prefill**), stores them, and
each later step only computes the new token's key and value and appends. Cost per step becomes flat
instead of growing.

The consequence is that generating is **memory-bound**: each step re-reads all the weights to produce
one token, so the GPU spends most of its time waiting on memory, not calculating. That is why
generating for many users at once is nearly free — the weights are read once for the whole batch.

### Temperature

Before picking a token, divide the scores by T. T < 1 sharpens toward the most likely token; T > 1
flattens toward the unlikely ones.

- **T = 0.1** — confident and repetitive. Our poem repeats "a symphony of sound" seven times: a weak
  model's most-likely path loops.
- **T = 0.7** — fluent and varied, and wrong in ways you could miss ("the sky is blue because of the
  tilt of the Earth's axis").
- **T = 1.5** — grammar survives, meaning does not: the capital of France becomes Annecy.

### Energy

Every run, totalled: **1.86 kWh** of GPU energy, ~2.6 kWh at the wall, ≈0.7 kg CO₂e on the Dutch grid.
The argument in the report is that "cheap" and "low-impact" are different claims — the table omits the
failed attempts, the controls and the benchmark sweeps, and a per-run calculator makes those easy to
forget.

---

## 8. The analysis scripts that check the report itself

| Script | Question it answers |
|---|---|
| `check_report.py` | Does the LaTeX have broken references, missing citations, unbalanced braces, missing figures, leftover TODOs? |
| `page_budget.py` | How long is the report really — prose words outside floats, plus float count? |
| `section_words.py` | Which section is heaviest, so a cut can be aimed? |

These exist because "is the report finished" should be a command, not an opinion.

---

## 9. The order everything runs in

```bash
bash scripts/bootstrap_wsl.sh              # once per machine
export NANOCHAT_DIR=/path/to/nanochat
source scripts/env.sh

bash scripts/10_train_tokenizers.sh        # ~20 s      Task 1
python scripts/11_tokenizer_analysis.py    #            Task 1
python scripts/12_merge_counts.py --vocab-size 32768 --check-log results/tokenizer/task1_tok_train_v32768_*.log
python scripts/13_merge_tree.py --word " photosynthesis"
python scripts/14_token_usage.py

bash scripts/15_smoke_test.sh              # ~10 min    catches problems before the real run
DEPTH=8 bash scripts/20_pretrain.sh        # ~42 min    Task 2
bash scripts/35_run_task3_task4.sh         # ~2.5 h     Task 3 + 4

python analysis/fit_scaling_law.py         # needs all four depths
python analysis/gpu_energy.py
python analysis/check_report.py
```

`15_smoke_test.sh` deserves its place: it runs all five pipeline stages on a throwaway depth-2 model
in ten minutes. Every kind of failure that would otherwise appear an hour into a real run — a missing
dataset, a broken kernel, a wrong path — appears here instead.
