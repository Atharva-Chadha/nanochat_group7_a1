"""
Plot training / validation bits-per-byte curves from the JSONL metrics written by our patched
scripts/base_train.py and scripts/chat_sft.py (--log-file), for the report.

Single run  -> <out>/<tag>_bpb.png (full run) and <out>/<tag>_bpb_zoom.png (last 90 % of steps),
               plus <tag>_val_bpb.csv and a printed "flattening" summary.
Several runs -> one overlay of val bpb vs training tokens (e.g. the depth sweep or the ratio sweep).

Usage (any python with matplotlib; the venv's is fine):
    python assignment/scripts/plot_curves.py assignment/results/d8_pretrain_metrics.jsonl
    python assignment/scripts/plot_curves.py d2=results/d2_pretrain_metrics.jsonl d4=... d8=... --title "Depth sweep" --out-name depth_sweep
"""
import argparse, json, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed order, colorblind-safe for adjacent pairs) + text/grid tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e2", "#ffffff"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cfg = next((r for r in recs if r["type"] == "config"), {})
    train = [r for r in recs if r["type"] == "train"]
    val = [r for r in recs if r["type"] == "val"]
    final = next((r for r in recs if r["type"] == "final"), None)
    return cfg, train, val, final

def centered_mean(values, window=21):
    # symmetric moving average: no lag (an EMA trails the true value during the steep early phase)
    h = window // 2
    return [sum(values[max(0, i - h):i + h + 1]) / len(values[max(0, i - h):i + h + 1]) for i in range(len(values))]

def style_axes(ax, xlabel, ylabel):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.tick_params(length=0)

def plot_single(path, label, out_dir, out_name=None):
    cfg, train, val, final = load(path)
    tag = out_name or label or cfg.get("user_config", {}).get("model_tag") or "run"
    bs = cfg.get("total_batch_size", 1)
    steps_t = [r["step"] for r in train]
    for suffix, start_frac in (("", 0.0), ("_zoom", 0.10)):
        n = steps_t[-1] + 1 if steps_t else 1
        s0 = int(n * start_frac)
        tr = [r for r in train if r["step"] >= s0]
        va = [r for r in val if r["step"] >= s0]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
        tr_smooth = centered_mean([r["bpb"] for r in tr])
        ax.plot([r["step"] for r in tr], [r["bpb"] for r in tr], color=SERIES[0], linewidth=0.6, alpha=0.18)
        ax.plot([r["step"] for r in tr], tr_smooth, color=SERIES[0], linewidth=2, label="train bpb (21-step centered mean; raw faint)")
        if va:
            ax.plot([r["step"] for r in va], [r["val_bpb"] for r in va], color=SERIES[1], linewidth=2, marker="o", markersize=3.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label="validation bpb")
            # direct labels at the right end
            close = tr and abs(va[-1]["val_bpb"] - tr_smooth[-1]) < 0.05
            ax.annotate(f"val {va[-1]['val_bpb']:.3f}", (va[-1]["step"], va[-1]["val_bpb"]), xytext=(4, 7 if close else 0), textcoords="offset points",
                        color=TEXT2, fontsize=8, va="center")
        if tr:
            close = va and abs(va[-1]["val_bpb"] - tr_smooth[-1]) < 0.05
            ax.annotate(f"train {tr_smooth[-1]:.3f}", (tr[-1]["step"], tr_smooth[-1]), xytext=(4, -7 if close else 0),
                        textcoords="offset points", color=TEXT2, fontsize=8, va="center")
        style_axes(ax, "optimizer step", "bits per byte")
        sec = ax.secondary_xaxis("top", functions=(lambda s: s * bs / 1e6, lambda t: t * 1e6 / bs))
        sec.set_xlabel("training tokens (M)", color=TEXT2); sec.tick_params(length=0, colors=TEXT2)
        mc = cfg.get("model_config", {})
        title = f"{tag}: depth {mc.get('n_layer')}, {cfg.get('param_counts', {}).get('total', 0)/1e6:.0f}M params, " \
                f"{cfg.get('total_tokens', 0)/1e6:.0f}M tokens" + (" (last 90 % of training)" if suffix else "")
        ax.set_title(title, color=TEXT, loc="left")
        ax.legend(frameon=False, loc="upper right")
        ax.margins(x=0.08)
        fig.tight_layout()
        out = os.path.join(out_dir, f"{tag}_bpb{suffix}.png")
        fig.savefig(out); plt.close(fig)
        print("wrote", out)
    # val table + flattening summary
    if val:
        with open(os.path.join(out_dir, f"{tag}_val_bpb.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step", "tokens", "val_bpb", "train_bpb_smoothed_at_step"])
            smooth = dict(zip([r["step"] for r in train], centered_mean([r["bpb"] for r in train])))
            for r in val:
                w.writerow([r["step"], r["step"] * bs, f"{r['val_bpb']:.5f}", f"{smooth.get(r['step'] - 1, float('nan')):.5f}"])
        print(f"{tag}: val bpb {val[0]['val_bpb']:.4f} @0 -> {val[-1]['val_bpb']:.4f} @{val[-1]['step']}  (min {min(r['val_bpb'] for r in val):.4f})")
        # improvement per 100 steps between consecutive evals: where does it drop below 1 % of the total drop?
        total_drop = val[0]["val_bpb"] - min(r["val_bpb"] for r in val)
        for a, b in zip(val, val[1:]):
            rate = (a["val_bpb"] - b["val_bpb"]) / max(b["step"] - a["step"], 1) * 100
            if rate < 0.01 * total_drop:
                print(f"{tag}: val improvement falls below 1 % of the total drop per 100 steps at step ~{b['step']} "
                      f"(rate {rate:.4f} bpb/100 steps); final train-val gap {centered_mean([r['bpb'] for r in train])[-1] - val[-1]['val_bpb']:+.4f}")
                break
    if final:
        print(f"{tag}: final -> {json.dumps({k: final.get(k) for k in ('step', 'min_val_bpb', 'total_training_time', 'peak_memory_mib')})}")

def plot_overlay(items, out_dir, out_name, title, x="tokens"):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for i, (label, path) in enumerate(items):
        cfg, train, val, final = load(path)
        bs = cfg.get("total_batch_size", 1)
        xs = [r["step"] * bs / 1e6 if x == "tokens" else r["step"] for r in val]
        ys = [r["val_bpb"] for r in val]
        c = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=2.5, markeredgecolor=SURFACE, markeredgewidth=0.6, label=label)
        if xs:
            ax.annotate(f"{label} {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", color=TEXT2, fontsize=8, va="center")
    style_axes(ax, "training tokens (M)" if x == "tokens" else "optimizer step", "validation bits per byte")
    ax.set_title(title, color=TEXT, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.margins(x=0.12)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="metrics.jsonl paths, optionally as label=path")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "figures"))
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--title", default="Validation bpb")
    ap.add_argument("--x", default="tokens", choices=["tokens", "step"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = [(r.split("=", 1)[0], r.split("=", 1)[1]) if "=" in r else (None, r) for r in a.runs]
    if len(items) == 1:
        plot_single(items[0][1], items[0][0], a.out, a.out_name)
    else:
        items = [(lab or os.path.basename(p).split("_")[0], p) for lab, p in items]
        plot_overlay(items, a.out, a.out_name or "overlay", a.title, a.x)
