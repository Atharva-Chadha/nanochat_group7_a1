# Code cleanup — every change, file by file

Cleanup pass over the submission repository, 24 September 2026. 16 files touched,
+174 / −171 lines.

**Nothing in this pass changes a single number in the report.** Every analysis script was re-run
afterwards and produces byte-identical results: energy 0.1988 / 0.0039 / 0.2029 kWh, the scaling fit
`L(C) = 0.6232 + 356.5·C^-0.1769` with SSE 1.07e-08, the depth-24 prediction 0.7495 (+0.0325). The
changes are of three kinds: **broken things fixed**, **stale text corrected**, and **robustness**.

---

## Summary table

| File | Lines | Kind | What changed |
|---|---|---|---|
| `scripts/bootstrap_wsl.sh` | +66 −107 | **broken** | Rewritten: it could not run in this repository at all |
| `analysis/fit_scaling_law.py` | +53 −27 | **broken** | Crashed on a clean clone; now runs and still reproduces the fit |
| `analysis/_paths.py` | +2 −0 | broken | Added the fallback path used by the above |
| `results/pretrain/d2_summary.json` | new | broken | The depth-2 numbers, so the fit no longer needs a lost file |
| `scripts/plot_curves.py` | +13 −5 | broken + stale | Fine-tuning figures were titled "depth None, 0M params" |
| `scripts/50_compare_models.py` | +10 −6 | robustness + stale | Interactive mode never freed the GPU |
| `scripts/14_token_usage.py` | +1 −1 | robustness | Possible `IndexError` |
| `analysis/check_report.py` | +4 −2 | robustness | Files were opened and never closed |
| `analysis/page_budget.py` | +2 −1 | accuracy | Bibliography size was hard-coded at 21; it is 25 |
| `report/main.tex` | +5 −5 | accuracy | Author name, and a caption describing a column that does not exist |
| `scripts/11_tokenizer_analysis.py` | +3 −2 | stale | Paths from the old repository layout |
| `scripts/40_samples.py` | +4 −4 | stale | Same |
| `scripts/inspect_datasets.py` | +4 −4 | stale | Same |
| `scripts/20_pretrain.sh` | +4 −4 | stale | Same |
| `scripts/30_midtrain.sh` | +1 −1 | stale | Same |
| `scripts/32_chat_eval.sh` | +1 −1 | stale | Same |
| `scripts/35_run_task3_task4.sh` | +1 −1 | stale | Same |
| `analysis/__pycache__/`, `scripts/__pycache__/` | deleted | hygiene | Compiled bytecode, 9 files |

---

## 1. `scripts/bootstrap_wsl.sh` — rewritten (+66 −107)

### The problem

This script is the first thing README section 9 tells a reader to run. **It could not work in this
repository.** It was written for the earlier layout, where our scripts lived in an `assignment/`
subfolder *next to* a `nanochat-master/` checkout inside one repository. This repository deliberately
does not contain nanochat. Five separate lines therefore failed:

| Old line | Why it failed |
|---|---|
| `[ -d "$WIN_REPO/nanochat-master" ] \|\| die ...` | There is no `nanochat-master/` here → **died immediately on line 36** |
| `git add .gitattributes README.md nanochat-master` | Adds a directory that does not exist |
| `git update-index --chmod=+x assignment/bootstrap_wsl.sh assignment/scripts/*.sh` | Wrong paths |
| `cd "$WSL_REPO/nanochat-master"` | Directory does not exist |
| `bash "/mnt/c/X/Masters/SEM 3/Agentic LLM/Ass 1/agentic-llm-.../assignment/bootstrap_wsl.sh"` | A personal Windows path hard-coded in the usage comment |

It also did work that no longer belongs to it: creating the git repository and making the first two
commits. That was one-off scaffolding for a repository that already exists.

### What it does now

Five steps, each idempotent:

1. **apt packages** — `build-essential git curl ca-certificates screen unzip`. The C compiler is not
   optional: `torch.compile` calls Triton, which shells out to a compiler.
2. **Locate or clone nanochat** — uses `$NANOCHAT_DIR`, defaults to `../nanochat`, and
   `git clone`s it if absent. This is the step that replaces the whole broken section.
3. **`uv sync --extra gpu --group dev`** — builds the locked virtualenv inside the nanochat checkout.
4. **Verify the stack** — prints torch/CUDA/GPU, asserts CUDA is visible, imports every package the
   scripts need, prints the compute dtype and whether FlashAttention-3 loaded, then runs nanochat's
   own tests. `tests/test_optim.py` is the real check, because compiling the fused Muon/AdamW kernels
   is what actually exercises `torch.compile` + Triton. `SKIP_TESTS=1` skips it.
5. **Download the corpus** — `python -m nanochat.dataset -n $NSHARDS` (default 24 train shards plus
   the pinned validation shard).

It finishes by printing the three paths and the exact next four commands.

### Why this matters for the viva

"Show me how you set this up" now has a runnable answer. Before, the answer was a script that exits
with an error on its first check.

---

## 2. `analysis/fit_scaling_law.py` — made reproducible (+53 −27)

### The problem

This script produces the report's headline result — the scaling law in the abstract. On a clean clone
it **crashed**:

```
FileNotFoundError: ~/.cache/nanochat/base_checkpoints/d2/train_log.jsonl
```

The depth-2 run predates the repository and its log stayed in the nanochat cache on the machine we
lost (README section 8, gap 1). So the one script whose expected output the README prints, and whose
result the report quotes, could not be run by anyone reading the submission.

### The fix, in three parts

**(a) `results/pretrain/d2_summary.json`** — a new 14-line file holding the depth-2 run's published
numbers: parameters, scaling parameters, FLOPs per token, batch, steps, tokens, compute, and final
validation bpb 1.3314. Its `note` field says plainly why it exists. These are the same values the
README table and the report already publish; nothing new is asserted.

**(b) A fallback in the script.** It still prefers the raw log when present:

```python
if os.path.exists(_paths.D2_LOG):
    ...parse the full curve as before...
else:
    ...load d2_summary.json, print "depth-2 raw log not found; using d2_summary.json"...
```

The fallback is announced on stdout, so nobody can mistake it for a complete run.

**(c) A guard so the figure cannot be silently degraded.** In fallback mode only the final depth-2
point is known, so panel (a) of `15_depth_series.png` would lose its depth-2 curve. A new flag
`D2_FULL_CURVE` makes the script *skip writing the figure* and say so, instead of overwriting a good
committed figure with a worse one. The fit itself is unaffected and still printed.

A related one-line fix: the frontier plot dropped the first validation point of every run with
`[1:]` (correct — bpb at step 0 is the uniform prior, not a measurement). With a single-point run that
left an empty list and raised `IndexError`. It now only drops the first point when there is more than
one.

### Readability, same behaviour

- `import matplotlib.ticker as mticker` added. The script called `matplotlib.ticker.ScalarFormatter()`
  while only importing `matplotlib`; that happens to work because `pyplot` imports `ticker` as a side
  effect. Relying on another module's import is fragile.
- The compute-matched table used an assignment expression inside a list comprehension inside a
  format string:

  ```python
  row = [("%s %s" % (o, ("%.3f" % v) if (v := (runs[o]["min_bpb"] if o == fin
                                               else bpb_at(o, Cf))) else "  -  ")) for o in order]
  ```

  Rewritten as a plain four-line loop with a comment saying what it answers ("where was every other
  run when it had spent the same compute?"). Identical output.

### Verified

```
4-point fit: L(C) = 0.6232 + 356.5 C^-0.1769   SSE=1.07e-08  dof=1
   published d24 0.7170 -> fit 0.7495 (err +0.0325)
```

Identical to the README's published output and to the report.

---

## 3. `scripts/plot_curves.py` — fine-tuning titles (+13 −5)

The title line assumed every metrics file was a **pre-training** run:

```python
title = f"{tag}: depth {mc.get('n_layer')}, {cfg.get('param_counts', {}).get('total', 0)/1e6:.0f}M params, ..."
```

A `chat_sft.py` config record has no `param_counts` and no `total_tokens` — it records `stage`,
`source`, `train_rows` and `train_mixture` instead. So every fine-tuning figure was titled
**"depth None, 0M params, 0M tokens"**. It now branches:

- pre-training → `d8: depth 8, 126M params, 503M tokens`
- a stage → `d8_mid: stage 'mid' from base/d8, 329,418 examples (GSM8K+MMLU)`

The figures already in `report/figures/` were produced with the corrected version, so they are right;
this brings the script in the repository into line with them.

---

## 4. Robustness fixes

**`scripts/50_compare_models.py` — interactive mode leaked the model.** The batch path ends with
`del engine, model; gc.collect(); torch.cuda.empty_cache()`. The interactive path `return`ed before
reaching it. The whole point of the script is loading checkpoints one at a time on a 24 GB shared
card, so the free is now on both paths.

**`scripts/14_token_usage.py` — possible `IndexError`.**

```python
while covered < 0.99 * total:      # total includes special tokens
    covered += freq[k]             # freq covers only the regular ones
    k += 1
```

`total` sums every counted id; `freq` only holds the non-special ids. If special tokens ever made up
more than 1% of the sample the loop would run off the end. They do not today, because `encode()` adds
no BOS — so this never fired and no number changes — but it is one `and k < len(freq)` away from being
safe, so it now has it.

**`analysis/check_report.py` — unclosed files.** `s = open(...).read()` twice, relying on the garbage
collector. Replaced with `with` blocks.

---

## 5. Accuracy fixes

**`analysis/page_budget.py`** printed `references block: ~0.5 page for 21 entries` with 21 hard-coded.
The bibliography has **25** entries. It now counts them:

```python
n_bib = len(re.findall(r"@\w+\{", open(os.path.join(ROOT, "main.bib"), encoding="utf-8").read()))
```

(The README also says 23 in section 7 — see the open items below.)

**`report/main.tex`, two mechanical fixes:**

1. `\icmlauthor{AtharvaChadha}` → `\icmlauthor{Atharva Chadha}`. The missing space renders on the
   title page of the PDF.
2. Table 1's caption read *"MFU is model FLOPs utilisation over the run against nanochat's bf16 peak;
   the RTX 2070 has no bf16 and no peak entry."* — but the table has **no MFU column**. The caption
   described a column that was presumably cut. Replaced with a caption that matches the table:
   *"Depth 2 ran in fp32 on the RTX 2070, which has no bf16 tensor cores; every other run used bf16."*

No prose, no number and no argument in the report was touched.

---

## 6. Stale paths (7 files)

The repository was reorganised (commit `cd2dc96`, *"Point env.sh at the new repository layout"*), but
only `env.sh` was updated. Seven files still documented the **old** layout in their headers — the
paths a reader would type, and the ones a TA would try:

| Was | Now |
|---|---|
| `python ../assignment/scripts/40_samples.py` | `python /path/to/repo/scripts/40_samples.py` |
| `Run from nanochat-master with the venv active` | `Run from the nanochat checkout with its venv active` |
| `writes assignment/results/samples_<label>.md` | `writes results/samples_<label>.md` |
| `git log -p nanochat-master/scripts/chat_sft.py` | `git log -p scripts/chat_sft.py` (in the checkout) |
| `Writes assignment/report/figures/tokenizer_compression.png` | `Writes report/figures/02_tokenizer_compression.png` |

Comments only — no executable line changed. Verified afterwards:
`grep -rn "assignment/results\|assignment/scripts\|nanochat-master" scripts/ analysis/` returns nothing.

---

## 7. Hygiene

`analysis/__pycache__/` and `scripts/__pycache__/` deleted — 9 `.pyc` files, including
`cpython-314` bytecode from a Python version the project does not target. They are in `.gitignore`,
so they were never committed, but they were sitting in the folder being submitted.

---

## Verification performed after the cleanup

```
python analysis/gpu_energy.py        0.1988 / 0.0039 / 0.2029 kWh   unchanged
python analysis/fit_scaling_law.py   L = 0.6232 + 356.5 C^-0.1769   unchanged
python analysis/check_report.py      balanced, 0 dangling refs      unchanged
python analysis/page_budget.py       4,949 prose words, 25 entries  bib count corrected
python analysis/section_words.py     per-section table              unchanged
python -c ast.parse   on all 14 .py files                           all parse
bash -n               on all 9 .sh files                            all parse
tr -cd '\r'           on every source file                          all LF
```

---

## Open items this pass did **not** fix

Deliberately left, because each needs information or a content decision that is yours:

1. **Three `\TODO{}` markers render in red in the submitted PDF** — repository URL (page 1),
   checkpoint URL (page 1), fork URL (page 5). The brief requires *"links to model checkpoints"* as a
   submission item, so the checkpoint one is a missing deliverable, not just a cosmetic blemish.
2. **`\icmlcorrespondingauthor{...}{TODO@umail.leidenuniv.nl}`** — placeholder email on page 1.
3. **Page count.** Body plus references is 9 pages; the appendix is page 10 and is exempt.
   `Ass1Build.pdf` says *"maximum 8 pages, including figures and references"*. The header comment in
   `main.tex` says *"Limit: 10 pages, excluding references"* — check that against any course
   announcement, because the two rules disagree and one of them costs marks.
4. **"every hundredth merge"** (Task 1, Q1) is wrong. rustbpe logs every **1%** of merges — exactly
   100 lines for 32,503 merges, which is what the same sentence's "all 100 logged counts" refers to.
   "Every hundredth merge" would be ~325 lines. One word: *hundredth* → *1%*.
5. **README section 7 says the bibliography has 23 entries; it has 25.**
6. **Four unused labels** (`sec:setup`, `sec:pretrain`, `sec:sft`, `sec:infer`). Harmless — standard
   practice to label sections you may cross-reference later — but `check_report.py` lists them.
7. **`icml2026/`** is an untracked copy of the template distribution (336 KB), including
   `example_paper.pdf` and `icml_numpapers.pdf`. `report/` already holds the four style files it
   needs. Safe to delete from the submission folder.
8. **Uncommitted work.** `report/main.pdf`, `results/samples/compare_d2.*` and
   `results/tokenizer/merge_tree_v32768_photosynthesis.json` are untracked; the submitted repository
   should be committed clean so `git log` tells the whole story.
