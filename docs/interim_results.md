# Assignment 1 — interim results (21 Sep 2026, 14:10)

Status (17:45): Task 1 complete (incl. merge tree), Task 2 complete, Task 3 two-stage run + controls A/B complete, Task 4 samples done. All numbers below come
from `assignment/results/*` (logs, `*.jsonl`, `*.csv`, `*.md`) produced by the scripts in `assignment/scripts/`.

## 0. Setup

| | |
|---|---|
| Hardware | RTX 5070 Ti 16 GB (Blackwell sm_120, no FlashAttention-3 → PyTorch SDPA), Ryzen 7 9700X, 31 GB RAM, WSL2 Ubuntu 24.04 |
| Software | nanochat (course copy, Mar 2026), torch 2.9.1+cu128, `torch.compile` + Triton, bf16 compute |
| Data | `karpathy/climbmix-400b-shuffle`, **24 train shards + 1 val shard (2.2 GB)** out of 6,543 — same mechanism as `runs/speedrun.sh` |
| Model | `--depth 8` (spec: 2 or 4; 8 chosen because the run fits in 42 min) — 8 layers, 4 heads × 128, d_model 512, 125.8M params |
| Code changes vs upstream | exact train bits-per-byte + JSONL logging in `base_train.py`; staged `chat_sft.py` (`--stage mid/sft`, `--source base/sft`, `--output-tag`); FA3 disabled on Blackwell; RTX 5070 Ti peak FLOPS for MFU. Every change is a separate commit on top of the pristine baseline. |

## 1. Task 1 — Tokenization (two BPE tokenizers on 500 MB of ClimbMix)

Training: `rustbpe`, 500,000,000 characters (first two shards, documents capped at 10,000 chars), 8.1 s (8,192) and
8.4 s (32,768). Both share nanochat's GPT-4-style split pattern with `\p{N}{1,2}` (numbers are pre-split into ≤2-digit
chunks) and 9 special tokens.

**Compression and sequence length** (`results/tokenizer_analysis.md`; GPT-2/GPT-4 via tiktoken for reference):

| text | bytes | ours-8k tokens | ours-32k tokens | bytes/token 8k → 32k (GPT-2 / GPT-4) | 8k/32k length |
|---|---|---|---|---|---|
| ClimbMix val, 2M chars | 2,002,641 | 500,045 | 426,451 | **4.00 → 4.70** (4.65 / 4.80) | 1.17× |
| English paragraph (own) | 396 | 92 | 74 | 4.30 → 5.35 (5.35 / 5.35) | 1.24× |
| Python code | 377 | 175 | 151 | 2.15 → 2.50 (2.33 / 3.07) | 1.16× |
| Numbers / invoice | 183 | 105 | 100 | 1.74 → 1.83 (2.23 / 2.01) | 1.05× |
| German | 112 | 58 | 47 | 1.93 → 2.38 (2.55 / 3.29) | 1.23× |
| Korean | 134 | 134 | 122 | **1.00 → 1.10** (1.14 / 2.63) | 1.10× |
| Hindi | 147 | 147 | 91 | **1.00 → 1.62** (1.69 / 2.53) | 1.62× |
| Emoji / accents | 75 | 41 | 31 | 1.83 → 2.42 (3.12 / 3.12) | 1.32× |

![tokenizer compression](figures/tokenizer_compression.png)

Reading: on in-distribution English web text the 32k tokenizer is as good as GPT-2's 50k vocabulary and within 2 % of
GPT-4's 100k, whereas the 8k tokenizer produces 17 % longer sequences (0.250 vs 0.213 tokens/char). The same 2,048-token
context therefore holds ~15 % less text with the 8k vocabulary, and every training step processes fewer bytes.

**Vocabulary composition:** mean token length 5.5 bytes (8k) vs 6.6 bytes (32k); 1,800 vs 11,290 tokens of ≥ 8 bytes;
only 108 vs 112 digit-only tokens (the `{1,2}` split caps them at 100 two-digit + 10 one-digit + variants); 165 vs 410
tokens containing non-ASCII bytes. Longest tokens are English words (` recommendations`, ` telecommunications`) and
formatting runs (`----------------`, `……………………`).

**Failure cases / artifacts (Q4):**
- Numbers: `1234567890` → `12 34 56 78 90` in both of our tokenizers (GPT-4: `123 456 789 0`); `$1,234.56` → 7 tokens.
  The pattern makes digit tokenization *deterministic and length-invariant* but never shorter than ⌈digits/2⌉ tokens —
  a deliberate trade-off Karpathy validated for 32k vocabularies (`dev/LOG.md`, 2026-01-13).
- Code: `    x = 1` — indentation becomes whitespace tokens; the 8k tokenizer splits `def` into `d`+`ef`, the 32k one has
  `def` and `(n` as tokens. GPT-4 is 20 % better on code because its training mix contains far more code.
- Non-English: Korean and Hindi fall back to **one token per UTF-8 byte** with the 8k vocabulary (1.00 bytes/token; a
  Korean syllable costs 3 tokens), 32k is barely better for Korean (1.10). `Straße` is split *through* the two bytes of
  `ß`; emoji are 3 bytes → 3 tokens each. The 500 MB training sample is overwhelmingly English, so merges for other
  scripts never reach the merge priority list.
- Long rare words: `supercalifragilisticexpialidocious` → 15 tokens (8k) / 12 (32k) / 11 (GPT-4).

**Merge tree (Q1)** — `12_bpe_merge_tree.py` trains a byte-level BPE in pure Python on a 316-character original
corpus (same regex pre-split as rustbpe) and records the count of every merge: `' '+'t'` ×33 first, `k+e` ×14,
`' t'+'o'` ×12, … `' tokenization'` ends up as `' tokeniz' a t i on` (5 tokens) because "tokenization" occurs once
while "token(s|izer|izes)" occur 12×. Table and figure: `results/bpe_merge_tree.md`, `figures/bpe_merge_tree.png`.

![merge tree](figures/bpe_merge_tree.png)

## 2. Task 2 — Pre-training (depth 8)

**What `--depth 8` derives** (`scripts/base_train.py`, all logged in `results/d8_pretrain_metrics.jsonl`):

| quantity | value | rule in the code |
|---|---|---|
| layers / d_model / heads | 8 / 512 / 4 (head_dim 128) | `d_model = ceil(64·depth/128)·128`, `heads = d_model/128` |
| parameters | 125,829,354 total: wte 16.8M, value-embeddings 67.1M, lm_head 16.8M, transformer matrices 25.2M, 42 scalars | `num_scaling_params()` |
| "scaling" parameters | 41,943,232 (transformer matrices + lm_head) | Kaplan-style count used for the horizon |
| training tokens | **503,316,480** = 12 × 41.9M | `--target-param-data-ratio 12` (nanochat's measured optimum, `dev/LOG.md` 2026-01-27) |
| batch / iterations | 262,144 tokens (auto: B ∝ D^0.383 from the d12 reference) / 1,920 | Power-Lines rule (`dev/LOG.md` 2026-02-05) |
| learning-rate scale, weight decay | 0.707 (√(B/B_ref)), 0.52 (0.28·√(B/B_ref)·D_ref/D) | muP-style transfer from d12 |
| schedule | warmup 40 steps, constant, linear warmdown over the last 65 %, Muon momentum 0.85→0.97→0.90 | |
| FLOPs | 3.52e8 per token, **1.77e17 total** | `estimate_flops()` (6·N_matmul + attention) |

**Scaling-law check (Chinchilla):** with N = 125.8M *total* parameters the Chinchilla-optimal horizon is ≈ 20·N ≈
**2.5B tokens**; nanochat trained on 503M, i.e. 4.0 tokens per total parameter — *under-trained* in Chinchilla terms.
Counting only the 41.9M matmul parameters (the count nanochat's own scaling fits found cleanest, because the 67M value
embeddings and the embedding tables are cheap look-ups that cost no FLOPs), the ratio is 12.0, which nanochat's own
sweeps found compute-optimal for this architecture. The two conventions disagree by 6× purely because of *which
parameters count* — a point to make explicitly in the report.

**Run statistics:** 1,920 steps in **41.6 min** of training (2,495 s), mean 200,789 tok/s, **mean MFU 80.5 %** of
87.9 TFLOPS (dense bf16, fp32-accumulate), peak 5.8 GB VRAM. GPU power log (`results/d8_power.csv`, 10 s samples over
49 min including evals): mean 247 W, max 262 W, **0.203 kWh** for the whole pretraining incl. evaluation.

**Loss / bpb curves** (`figures/d8_pretrain_bpb.png`, `figures/d8_pretrain_bpb_zoom.png`):

![d8 pretraining](figures/d8_pretrain_bpb.png)
![d8 pretraining zoom](figures/d8_pretrain_bpb_zoom.png)

- Validation bpb **3.166 → 0.939** (train 3.2 → 0.951 smoothed). At step 0 the model is at log₂(32768)/~4.7 ≈ 3.1 bpb,
  i.e. uniform over the vocabulary; by step 50 (13M tokens) it is at 1.76, by step 250 (66M tokens) at 1.19.
- Shape: a very steep first 5 % (unigram/bigram statistics, then local syntax), then a slowly decaying power-law-like
  tail. Per-100-step improvement drops below 1 % of the total drop at ≈ step 550 (0.019 bpb/100 steps), but the curve is
  **still decreasing at the end** (0.947 → 0.939 over the last 100 steps, helped by the LR warmdown that starts at step
  672). Nothing has "converged": longer training would keep paying, as the Chinchilla estimate predicts.
- Train–val gap: +0.012 bpb (train higher). With 503M tokens from 24 shards the model sees every token once, so there
  is no memorisation; the residual gap is the per-batch noise of the train estimate plus a slightly easier val shard.
  bpb (not loss) is used because it is invariant to the tokenizer's vocabulary — the same loss in nats/token would not be
  comparable between the 8k and 32k tokenizers.

**Raw completions of the base model** (`results/d8_base_eval_*.log`, greedy on 7 prompts; 8 unconditioned samples at
T = 1.0): the model produces fluent web-register text — product descriptions, listicles, a zoo article — with correct
local grammar and topic persistence over ~100 tokens, but no factual grounding: *"The capital of France is the capital
of France."*, *"The chemical symbol of gold is gold. It is a colorless, odorless, and tasteless metal."*, *"If 5·x + 3 = 13,
then x is the number of times the number of times…"*. Greedy decoding also exposes repetition loops. This is what a
126M-parameter model trained on 0.5B tokens can do: it has learned the *form* of English web text (bpb 0.94 ≈ 0.94 bits
per byte, a compression rate a bit better than gzip's ~2.5 bits/byte on similar text) but stores few facts and no
instruction-following behaviour — that needs the SFT stages.

## 3. Task 3 — Mid-training → SFT (two-stage run + two control variants complete)

**Data sanity check** (`results/dataset_inspection.md`): SmolTalk train 460,341 conversations (≈ 364M tokens/epoch,
73 % of tokens supervised); MMLU auxiliary_train 99,842 (≈ 38M tokens/epoch, but only **2 supervised tokens per
example**: the answer letter and `<|assistant_end|>` — 0.8 %); GSM8K train 7,473 (≈ 1.3M tokens, 56 % supervised).
Stage 1 mixture (MMLU×3 + GSM8K×4) = 329,418 examples ≈ 120M tokens; stage 2 (SmolTalk) = 460,341 ≈ 364M tokens.
A token-level rendering of one example per dataset with its loss mask is in the same file (evidence for the
loss-masking question).

**Implementation:** `scripts/chat_sft.py` gained `--stage {full,mid,sft}` (selects `train_tasks`/`val_tasks`),
`--source {base,sft}` (which checkpoint family `load_model`/`load_optimizer_state` read), `--output-tag` (so stage 2
does not overwrite stage 1), and it now stores its resolved batch/LR settings in the checkpoint metadata so the next
stage inherits them exactly as upstream inherits from a base checkpoint. Hyperparameters otherwise as upstream:
batch 262,144 tokens (inherited from pretraining), LRs 0.8× the pretraining LRs with a 50 % linear warmdown, optimizer
momentum warm-started from the previous checkpoint, one epoch over the stage mixture, MMLU×3 / GSM8K×4 epochs.

**Runs:** stage 1 `d8_mid`: 464 steps, 10.1 min. Stage 2 `d8_sft`: 1,435 steps, 31.2 min.

![mid-training](figures/d8_mid_bpb.png)
![SFT](figures/d8_sft_bpb.png)

**Benchmark scores** (`results/eval_scores.csv`; ARC = full test sets, categorical over the answer letters; GSM8K =
greedy generation with the calculator tool, full test set except the base model which was capped at 300 problems /
256 tokens because it never emits `<|assistant_end|>`):

| checkpoint | ARC-Easy (2,376) | ARC-Challenge (1,172) | GSM8K (1,319) |
|---|---|---|---|
| pre-trained base `d8` | 25.1 % (596) | 23.0 % (270) | 0.0 % (0/300) |
| stage 1 mid-training `d8_mid` | **32.6 %** (775) | **31.3 %** (367) | **3.3 %** (44) |
| stage 2 SFT on SmolTalk `d8_sft` | 24.5 % (583) | 23.2 % (272) | 0.0 % (0) |
| *(B)* stage 2 **with replay** `d8_sft_replay` (SmolTalk + MMLU×1 + GSM8K×1 from `d8_mid`) | **32.3 %** (768) | 27.5 % (322) | 0.0 % (0) |
| *(A)* upstream single-stage mixture from base `d8_full` (SmolTalk + MMLU×3 + GSM8K×4) | 31.2 % (741) | 29.2 % (342) | 0.2 % (2) |
| random baseline | 25 % | 25 % | 0 % |

![eval scores](figures/eval_scores.png)

**What happened and why.**
1. *Base → stage 1:* the base model is exactly at chance (it has never seen `<|user_start|>…<|assistant_start|>` or the
   "Respond only with the letter" format). 300K multiple-choice examples teach the format and the letter-binding
   trick (`render_mc` puts the letter *after* the choice), and 4 epochs of GSM8K teach the `#### answer` format plus
   calculator calls (`<|python_start|>3+4<|python_end|>` → the engine injects `7`). ARC rises 7–8 points above chance
   and 44 GSM8K problems are solved with correct tool use.
2. *Stage 1 overfits:* val bpb on the MMLU/GSM8K test mix reaches its minimum (0.667) after 100 steps and then
   climbs to 0.865 while train bpb falls to 0.14 — the model memorises the 7.5K GSM8K solutions it sees four times.
   Upstream nanochat uses the same epoch counts, but there they are diluted by 460K SmolTalk conversations.
3. *Stage 1 → stage 2 = catastrophic forgetting:* after one epoch of SmolTalk **only**, both ARC scores fall back to
   chance and GSM8K to 0/1,319. The samples show why: the SFT model answers "What is the capital of France?" with
   *"The capital of France is Paris. It is a city rich in history…"* (correct, fluent, assistant-like), writes verse
   for the poem prompt, but reasons about the apples in prose without a single arithmetic step and, for a
   multiple-choice prompt, no longer produces a bare letter — so the letter logits the categorical evaluation reads are
   uncalibrated. The mid-training model, conversely, turned *every* prompt into a GSM8K-style calculation and answered
   the poem request with "D". Each stage's data distribution overwrote the previous behaviour.
4. *Controls (A, B) confirm the mechanism.* Mixing one epoch of MMLU/GSM8K back into stage 2 (B) keeps ARC-Easy at
   32.3 % — the whole loss was caused by the data distribution of stage 2, not by anything irreversible in the
   weights. Upstream's single mixed stage (A) reaches 31.2 / 29.2 % from the base model directly, i.e. mid-training
   followed by mixed SFT (B) ends at the same multiple-choice level as the standard recipe: at this scale a separate
   reasoning stage brings **no measurable benefit** over mixing, while costing an extra run. GSM8K is different: the
   only checkpoint that solves any problems is the GSM8K-saturated mid-training model (3.3 %, with a memorising
   val curve); once SmolTalk dominates the mixture (A: 2/1319, B: 0/1319) the model answers word problems in prose
   and never emits the calculator format — a 126M model has capacity for the *format* only when it is a large share
   of what it sees. The val-bpb curves of (A) and (B) are clean (0.421 and 0.429, train ≈ val), so the difference is
   not under-training.
5. Hyperparameters that would be worth changing next: fewer GSM8K epochs in stage 1 (its val minimum is at 100/464
   steps), and a GSM8K-heavier replay in stage 2 if math matters — both are one-line changes in `chat_sft.py`.

**Loss masking** (Q2): `render_conversation` marks only assistant-content tokens and `<|assistant_end|>` with mask 1;
`chat_sft.py` sets the corresponding targets to −1 (`ignore_index`) so the cross-entropy is
Σ_{mask=1} −log p / #(mask=1). Without masking the loss would also reward predicting the user's words and the special
tokens — 27 % of SmolTalk tokens and **99.2 %** of an MMLU example (237 tokens of question/choices vs 2 supervised
tokens), so an unmasked stage 1 would mostly learn to copy MMLU questions. The rendered example tables in
`results/dataset_inspection.md` give the concrete token-level view.

## 4. Task 4 — inference (samples done), benchmarks and reflection pending

Temperature sweep on the SFT model (`results/samples_d8_sft.md`, prompts through the chat template, top-k 50):
- **T = 0.1** (near-greedy): confident but degenerate — the poem loops *"A place where the sea's calm waters"* five
  times; explanations repeat the same clause.
- **T = 0.7**: fluent, varied and mostly sensible; Paris is correct, the poem has structure; factual errors are of the
  plausible-nonsense kind ("Paris … a unique blend of French and Italian influences").
- **T = 1.5**: syntax survives but meaning dissolves ("The sky seems to blue if not for centuries…"), a corrupted
  byte sequence appears (`Le�`) — high temperature samples low-probability byte-level tokens that do not form valid
  UTF-8.
The unconditioned base-model samples (`results/samples_d8-base.md`) and the KV-cache/engine description are ready for
the report. Energy: pretraining **0.203 kWh**, mid-training + SFT ≈ 0.17 kWh (41 min at ~245 W), evaluations extra;
at ~0.3 kg CO₂e/kWh (to be replaced by the ML CO₂ Impact calculator's figure) the whole pipeline so far is ≈ 0.15 kg CO₂e.

## 5. Things to declare in the final report

- Depth 8 instead of the suggested 2/4; the code version has no web UI (`chat_cli.py` only); the SFT mixture differs
  from the PDF's description (no identity/spelling tasks); `--target-param-data-ratio` default 12 vs 10.5 in the text.
- FlashAttention-3 unavailable on Blackwell → SDPA, `--window-pattern L` (full context) instead of upstream's `SSSL`.
