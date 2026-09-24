# Submission review — correctness, accuracy, efficiency

Full review of `Final submission/` (code + `main.pdf`) on 24 September 2026, before cleanup.
Everything verified by running the code and recomputing the numbers, not by reading alone.

**Verdict.** The science is sound and the reporting is unusually honest. Every headline number in the
report reproduces exactly from the committed files. The defects are engineering and presentation,
and the ones that cost marks are cheap to fix.

---

## 1. What was verified, and passed

I recomputed the report's main claims independently.

| Claim in the report | Recomputed | Result |
|---|---|---|
| `L(C) = 0.623 + 356·C^(−0.177)` | 0.6232 + 356.5·C^(−0.1769) | ✅ exact |
| largest residual 8×10⁻⁵ bpb | 8.4×10⁻⁵ | ✅ |
| compute span 261× | 261× | ✅ |
| depth-24 prediction 0.750, error +0.033 | 0.7495, +0.0325 | ✅ |
| floorless exponent 0.074, d12 → 0.821 | 0.0740, 0.8206 | ✅ |
| depth-12 prediction 0.850 | 0.8496 | ✅ |
| pre-training energy 0.199 kWh at 249 W | 0.1988 kWh, 249.4 W | ✅ |
| evaluation energy 0.004 kWh | 0.0039 | ✅ |
| total 1.86 kWh over six runs | table sums to 1.863 | ✅ |
| 2.6 kWh at the wall (+40%) | 1.86 × 1.4 = 2.604 | ✅ |
| 1.5 kg / 0.7 kg CO₂e | 2.6 × 569 g, 2.6 × 269 g | ✅ |
| nanochat reference ≈22 kWh, 12× ours | 8 × 700 W × 4 h = 22.4 | ✅ |
| `D = 12·N_s` for all four depths | matches after batch-size flooring | ✅ |
| depth-2 vocabulary share 97.0% | 12,582,912 / 12,976,170 = 96.97% | ✅ |
| Chinchilla optimum N ≈ 38M on 770M tokens | √(1.77e17/120) = 38.4M, 769M | ✅ |
| "differ by a factor of 3" | 125.8/41.9 = 3.00 | ✅ |
| replay within one standard error of stage 1 | Δ0.3 pp vs 1.36 pp | ✅ |
| "eight standard errors" on ARC-Easy | 7.6/0.96 = 7.9 vs chance | ✅ |
| merge replay reproduces "all 100 logged counts" | log has exactly 100 lines | ✅ |
| 32,503 merges at 32k, 7,927 at 8k | vocab − 256 − 9 | ✅ |

I found **no incorrect quantitative claim** in the report. That is worth saying plainly, because it is
the hardest part to get right and the part the viva will probe.

---

## 2. Blocking issues — fix before submitting

### 2.1 The report is one page over the limit

- PDF is 10 pages. Body + references = pages 1–9; the AI disclosure is page 10 and the brief exempts it.
- `Ass1Build.pdf`: *"a written report as a PDF (maximum 8 pages, including figures and references)"*.
- `main.tex` line 3 says *"Limit: 10 pages, excluding references and the AI disclosure appendix."*
  **These two rules disagree.** If the course did raise the limit, fine — but the source comment
  should not silently contradict the brief.

Where the page would come from (`analysis/section_words.py`, prose words excluding floats):

| Section | Words |
|---|---|
| The inference loop | **582** |
| Q2: scaling laws | 374 |
| Q4: benchmark scores | 370 |
| Part B: environmental cost | 352 |
| Setup and deviations | 278 |

Those five are 1,956 of 4,949. The report also carries **11 floats (8 tables, 3 figures)** in the
body, and floats cost 0.25–0.45 page each. Dropping or merging one table is worth more than a
paragraph of cutting.

### 2.2 Three red `[TODO: ...]` markers print in the PDF

| Where | What |
|---|---|
| page 1 | `\TODO{repository URL}` |
| page 1 | `\TODO{checkpoint URL}` |
| page 5 | `\TODO{fork URL}` |
| page 1 | `\icmlcorrespondingauthor{...}{TODO@umail.leidenuniv.nl}` |

The checkpoint link is a **listed submission deliverable** — the brief asks for "(3) links to model
checkpoints". This is not cosmetic.

### 2.3 `scripts/bootstrap_wsl.sh` could not run *(fixed)*

README section 9 opens with it. It was written for the previous repository layout and died on its
own sanity check, because it required a `nanochat-master/` directory that this repository
deliberately does not contain. Four further lines referenced the same missing path, and the usage
comment hard-coded a personal Windows path. **Rewritten** — see `CHANGES_cleanup.md` §1.

### 2.4 `analysis/fit_scaling_law.py` crashed on a clean clone *(fixed)*

The script that produces the abstract's headline result raised `FileNotFoundError` on the depth-2 log,
which lives outside the repository on a machine that was lost. The README publishes its expected
output; a TA running it would have seen a traceback. **Fixed** with a committed summary and an
announced fallback — see `CHANGES_cleanup.md` §2.

---

## 3. Accuracy issues in the report

### 3.1 "every hundredth merge" is wrong

Task 1 Q1: *"rustbpe logs the pair count for every hundredth merge only"*. It logs every **1%** —
exactly 100 lines for 32,503 merges, which is what the same sentence's "all 100 logged counts" means.
Every hundredth merge would be ~325 lines. One word: *hundredth* → *1%*.

### 3.2 Table 1's caption described a column that isn't there *(fixed)*

The caption explained MFU and noted "the RTX 2070 has no bf16 and no peak entry", but the table has
three columns: machine, GPU, runs. The MFU column was presumably cut during the rewrite and the
caption was not. Replaced with a caption matching the table.

### 3.3 Author name ran together *(fixed)*

`\icmlauthor{AtharvaChadha}` rendered as one word on the title page. Now `Atharva Chadha`.

### 3.4 Bibliography size quoted three different ways

Actual: **25** entries. README section 7 says 23. `page_budget.py` hard-coded 21. *(script fixed to
count them; README still says 23.)*

---

## 4. Methodological points worth owning in the viva

Not errors — the report is honest about most of these — but a TA will push, so have the answer.

### 4.1 The scaling fit's residuals prove almost nothing

Three free parameters (`L∞`, `a`, `α`) against four points leaves one degree of freedom. The 8×10⁻⁵
residuals are what you would expect from *any* smooth decreasing function, not evidence the functional
form is right. The report presents them prominently and the out-of-sample depth-24 check second; the
honest ordering is the reverse, and that is how to present it verbally. The +0.033 error across a 65×
extrapolation is the defensible claim.

### 4.2 The four runs are not a controlled series

Two GPU architectures, two nanochat versions, two precisions (depth 2 in fp32), one run on 2 GPUs.
Setup deviation (b) declares it and lists the agreement checks performed, which is the right
treatment — but the confound is real and should be conceded before it is raised.

### 4.3 Single seed throughout

Every run is n = 1. The ± values are binomial standard errors from test-set size and do **not**
include run-to-run training variance. The report does state this. Differences of 1–2 points (control A
vs stage 1) are ties; the conclusions drawn rest on 8σ effects, which is the right place to stand.

### 4.4 `min_bpb` is used as each run's loss

Reasonable — for monotone pre-training curves the minimum is the final value — but it is a choice, and
for a run that overfits it would silently pick a different point than "final". Worth one sentence if
asked how the four `L` values were selected.

---

## 5. Efficiency review

The code is efficient where it matters and I found no wasteful hot path.

**Good:**
- `12_merge_counts.py` maintains pair counts **incrementally** during the replay, touching only the
  words affected by each merge, rather than recounting 32,503 times. This is the difference between
  minutes and hours.
- `50_compare_models.py` loads one checkpoint at a time and frees it, so a depth series fits on a
  shared 24 GB card.
- `20_pretrain.sh` backgrounds the power logger and kills it with a `trap` on exit.
- `plot_curves.py` uses a **centred** moving average rather than a running one — no lag on the steep
  early phase, which matters because that is where the curve moves most.
- `gpu_energy.py` integrates by trapezoid rather than assuming constant power.
- Training itself ran at 80.5% MFU at depth 8, which is high.

**Minor, now fixed:**
- `50_compare_models.py` never freed the model on the `--interactive` path.
- `14_token_usage.py` had a reachable-in-principle `IndexError`.
- `check_report.py` left two file handles open.
- `fit_scaling_law.py` relied on `matplotlib.ticker` being imported as a side effect of `pyplot`.

**Not worth changing:** the grid search over α in `fit_ls` evaluates 8,001 closed-form linear fits.
It looks brute-force, but each step is exact and the whole thing runs in well under a second; a
non-linear solver would be less robust for no gain.

---

## 6. Repository hygiene

| Item | Status |
|---|---|
| `analysis/__pycache__`, `scripts/__pycache__` (9 `.pyc`, incl. cpython-314) | **deleted** |
| Stale `assignment/…`, `nanochat-master` paths in 8 files | **fixed** |
| `icml2026/` — untracked 336 KB template copy incl. `example_paper.pdf` | flagged; `report/` already has the 4 style files it needs |
| Uncommitted: `report/main.pdf`, `compare_d2.*`, `merge_tree_*.json`, `main.tex` | flagged — commit before submitting so `git log` tells the whole story |
| Four unused LaTeX labels | flagged; harmless |
| `.gitignore`, `.gitattributes` | good — `!results/**/*.log` correctly rescues logs from the `*.log` rule |

---

## 7. What is genuinely strong

Worth knowing, because the viva is also an opportunity:

1. **Four depths and a scaling law** goes well beyond "train at depth 2 or 4", and the out-of-sample
   check against the course text's depth-24 number is a real test rather than a self-consistency one.
2. **Two controls in Task 3.** Most submissions will report that mid-training helped. Having run both
   the replay control and the upstream single-stage control means you can say *why* it stopped
   helping, and that mid-training buys nothing at this scale — a negative result, properly supported.
3. **The merge-count replay** with verification against all 100 logged frequencies is a genuinely
   clever piece of work, and it answers a question the trainer's output does not.
4. **Known gaps are documented** (README section 8) instead of hidden. The brief explicitly asks for
   honesty about what did not work.
5. **The AI disclosure is specific and self-critical** — it names concrete wrong answers the tool gave
   and how they were caught. That is exactly what the policy asks for.

---

## 8. Priority order

| # | Action | Effort |
|---|---|---|
| 1 | Fill the three `\TODO{}` URLs and the corresponding-author email | minutes |
| 2 | Decide the page limit; if 8, cut ~1 page (start with "The inference loop", 582 words, and one table) | ~1 hour |
| 3 | `hundredth` → `1%` in Task 1 Q1 | seconds |
| 4 | Commit the working tree; delete `icml2026/` | minutes |
| 5 | Rebuild the PDF and re-run `check_report.py` (TODO markers should read 0) | minutes |
| 6 | Fix "23 entries" → 25 in README section 7 | seconds |
| — | *(done)* bootstrap, scaling-fit reproducibility, stale paths, robustness, hygiene | — |
