"""Repo-relative paths, so every analysis script runs from anywhere.

    python analysis/fit_scaling_law.py
    cd analysis && python fit_scaling_law.py
both work.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PRETRAIN = os.path.join(RESULTS, "pretrain")
FINETUNE = os.path.join(RESULTS, "finetune")
EVAL = os.path.join(RESULTS, "eval")
SAMPLES = os.path.join(RESULTS, "samples")
REPORT = os.path.join(ROOT, "report")
FIGURES = os.path.join(REPORT, "figures")

# The depth-2 run predates this repo and its metrics stayed in the nanochat cache.
# Override with NANOCHAT_BASE_DIR if yours lives elsewhere.
BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "nanochat"))
D2_LOG = os.path.join(BASE_DIR, "base_checkpoints", "d2", "train_log.jsonl")
# Fallback when that log is not on this machine: the run summary committed with the results.
D2_SUMMARY = os.path.join(PRETRAIN, "d2_summary.json")
