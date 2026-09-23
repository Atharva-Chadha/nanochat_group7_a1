"""Parse all four pretraining runs, fit L = Linf + a*C^-alpha by least squares,
and draw (a) the per-run curves and (b) the compute frontier."""
import json, re, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

FIG = _paths.FIGURES
runs = {}

# --- d2: oldest run, jsonl with val/bpb keys and logged FLOPs --------------
d2 = {"val": [], "flops": {}, "machine": "older course code, eager"}
with open(_paths.D2_LOG,
          encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if "val/bpb" in r:
            d2["val"].append((r["step"], r["val/bpb"]))
            d2["flops"][r["step"]] = r["total_training_flops"]
d2.update(params=12_976_170, vocab_par=4_194_304 * 3, scaling=4_587_532,
          fpt=33_816_648, batch=131_072, tokens=55_050_240, d_model=128, heads=1)
d2["C"] = d2["fpt"] * d2["tokens"]
runs["d2"] = d2

# --- d8: structured jsonl from the course-code run -------------------------
d8 = {"val": [], "machine": "course code, 5070 Ti"}
with open(os.path.join(_paths.PRETRAIN, "d8_pretrain_metrics.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r["type"] == "config":
            pc = r["param_counts"]
            d8.update(params=pc["total"], scaling=r["num_scaling_params"],
                      fpt=r["num_flops_per_token"], C=r["total_training_flops_estimate"],
                      tokens=r["total_tokens"], batch=r["total_batch_size"],
                      d_model=r["model_config"]["n_embd"], heads=r["model_config"]["n_head"],
                      vocab_par=pc["wte"] + pc["value_embeds"] + pc["lm_head"])
        elif r["type"] == "val":
            d8["val"].append((r["step"], r["val_bpb"]))
runs["d8"] = d8

# --- d6, d10: upstream text logs -------------------------------------------
PAT_VAL = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
PAT_NUM = re.compile(r"^(wte|value_embeds|lm_head|transformer_matrices|scalars|total)\s*:\s*([\d,]+)")
FIELDS = (("fpt", r"Estimated FLOPs per token: ([\d.e+]+)"),
          ("C", r"Total training FLOPs estimate: ([\d.e+]+)"),
          ("tokens", r"Total number of training tokens: ([\d,]+)"),
          ("batch", r"Auto-computed optimal batch size: ([\d,]+)"),
          ("d_model", r'"n_embd": (\d+)'), ("heads", r'"n_head": (\d+)'))
for tag in ("d6", "d10"):
    r = {"val": [], "machine": "upstream, 3090"}
    counts = {}
    with open(os.path.join(_paths.PRETRAIN, "%s_pretrain.log" % tag),
              encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT_VAL.search(line)
            if m:
                r["val"].append((int(m.group(1)), float(m.group(2))))
                continue
            m = PAT_NUM.match(line.strip())
            if m:
                counts[m.group(1)] = int(m.group(2).replace(",", ""))
            for key, pat in FIELDS:
                mm = re.search(pat, line)
                if mm and key not in r:
                    v = mm.group(1).replace(",", "")
                    r[key] = float(v) if ("." in v or "e" in v) else int(v)
    r["params"] = counts["total"]
    r["scaling"] = counts["transformer_matrices"] + counts["lm_head"]
    r["vocab_par"] = counts["wte"] + counts["value_embeds"] + counts["lm_head"]
    runs[tag] = r

order = ["d2", "d6", "d8", "d10"]
for t in order:
    r = runs[t]
    r["min_bpb"] = min(b for _, b in r["val"])
    r["vocab_share"] = r["vocab_par"] / r["params"]

print("%-4s %-11s %-13s %-11s %-9s %-7s %s" %
      ("run", "params", "tokens", "C", "min bpb", "vocab%", "val pts"))
for t in order:
    r = runs[t]
    print("%-4s %-11s %-13s %.4e  %.6f  %.1f%%  %d" %
          (t, "{:,}".format(r["params"]), "{:,}".format(int(r["tokens"])),
           r["C"], r["min_bpb"], 100 * r["vocab_share"], len(r["val"])))

C = [runs[t]["C"] for t in order]
L = [runs[t]["min_bpb"] for t in order]


def fit_ls(Cs, Ls, alo=0.01, ahi=1.0, n=8000):
    best = None
    for i in range(n + 1):
        al = alo + (ahi - alo) * i / n
        x = [c ** (-al) for c in Cs]
        m = len(Cs)
        sx, sy, sxx = sum(x), sum(Ls), sum(v * v for v in x)
        sxy = sum(v * l for v, l in zip(x, Ls))
        den = m * sxx - sx * sx
        if abs(den) < 1e-30:
            continue
        aa = (m * sxy - sx * sy) / den
        bb = (sy - aa * sx) / m
        ss = sum((bb + aa * v - l) ** 2 for v, l in zip(x, Ls))
        if best is None or ss < best[0]:
            best = (ss, al, aa, bb)
    return best


sse, alpha, a, Linf = fit_ls(C, L)
fit = lambda c: Linf + a * c ** (-alpha)
print("\n4-point fit: L(C) = %.4f + %.4g C^-%.4f   SSE=%.2e  dof=%d"
      % (Linf, a, alpha, sse, len(C) - 3))
for t, c, l in zip(order, C, L):
    print("   %-4s measured %.4f  fit %.4f  resid %+.5f" % (t, l, fit(c), fit(c) - l))
print("   compute span %.0fx" % (C[-1] / C[0]))

C24 = 5.2847e9 * 6.0e9
C12 = 8.870953e8 * 1_321_205_760
print("   published d24 0.7170 -> fit %.4f (err %+.4f)" % (fit(C24), fit(C24) - 0.717))
print("   d12 extrapolation %.4f" % fit(C12))

x, y = [math.log(c) for c in C], [math.log(l) for l in L]
mx, my = sum(x) / len(C), sum(y) / len(C)
sl = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sum((xi - mx) ** 2 for xi in x)
k = math.exp(my - sl * mx)
print("   floorless alpha %.4f -> d12 %.4f" % (-sl, k * C12 ** sl))

# ---------------------------------------------------------------- figure
col = {"d2": "#8a6bbf", "d6": "#6aa84f", "d8": "#2a6ebb", "d10": "#d1495b"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
for t in order:
    r = runs[t]
    ax1.plot([s for s, _ in r["val"]], [b for _, b in r["val"]], "-o", ms=2.4, lw=1.3,
             color=col[t], label="%s (%.0fM, %.3f)" % (t, r["params"] / 1e6, r["min_bpb"]))
ax1.set_xlabel("training step")
ax1.set_ylabel("validation bits-per-byte")
ax1.set_yscale("log")
ax1.set_yticks([0.9, 1.0, 1.35, 2.0, 3.0])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_title("(a) per-run validation curves", fontsize=10)
ax1.grid(alpha=0.25, lw=0.5)
ax1.legend(fontsize=7.5)

for t in order:
    r = runs[t]
    if t == "d2":
        xs = [r["flops"][s] for s, _ in r["val"]][1:]
    else:
        xs = [s * r["batch"] * r["fpt"] for s, _ in r["val"]][1:]
    ys = [b for _, b in r["val"]][1:]
    ax2.plot(xs, ys, "-", lw=1.4, color=col[t], label=t)
    ax2.plot([xs[-1]], [ys[-1]], "o", ms=6, color=col[t])
cs = [10 ** (13.8 + 0.02 * i) for i in range(int((18.4 - 13.8) / 0.02) + 1)]
ax2.plot(cs, [fit(c) for c in cs], "--", lw=1.3, color="#444",
         label=r"fit $%.3f+%.0f\,C^{-%.3f}$" % (Linf, a, alpha))
ax2.axhline(Linf, ls=":", lw=1.1, color="#888")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_yticks([0.9, 1.0, 1.35, 2.0, 3.0])
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_xlabel("cumulative training compute (FLOPs)")
ax2.set_title("(b) the compute frontier", fontsize=10)
ax2.grid(alpha=0.25, which="both", lw=0.5)
ax2.legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
out = os.path.join(FIG, "15_depth_series.png")
fig.savefig(out, dpi=200)
print("\nwrote", out)

print("\ncompute-matched:")


def bpb_at(tag, Ct):
    r = runs[tag]
    pts = [((r["flops"][s] if tag == "d2" else s * r["batch"] * r["fpt"]), b)
           for s, b in r["val"] if s > 0]
    if Ct < pts[0][0] or Ct > pts[-1][0]:
        return None
    for (c0, b0), (c1, b1) in zip(pts, pts[1:]):
        if c0 <= Ct <= c1:
            w = (math.log(Ct) - math.log(c0)) / (math.log(c1) - math.log(c0))
            return b0 + w * (b1 - b0)


for fin in order:
    Cf = runs[fin]["C"]
    row = [("%s %s" % (o, ("%.3f" % v) if (v := (runs[o]["min_bpb"] if o == fin
                                                 else bpb_at(o, Cf))) else "  -  "))
           for o in order]
    print("  at C=%.3e (%s done, %.4f): %s" % (Cf, fin, runs[fin]["min_bpb"], " | ".join(row)))
