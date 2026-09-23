"""Trapezoidal energy from the nvidia-smi power log, split at the eval handover."""
import csv
import datetime as dt
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

P = os.path.join(_paths.PRETRAIN, "d8_power.csv")

rows = []
with open(P, encoding="utf-8") as fh:
    for r in csv.reader(fh):
        if not r or r[0].strip().startswith("timestamp"):
            continue
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, float(r[1].split()[0]), float(r[3].split()[0])))

t0, t1 = rows[0][0], rows[-1][0]
SPLIT = dt.datetime(2026, 9, 21, 13, 49, 29)


def energy(seg):
    """kWh by trapezoid, plus mean/peak watts and minutes."""
    j = 0.0
    for (ta, pa, _), (tb, pb, _) in zip(seg, seg[1:]):
        j += 0.5 * (pa + pb) * (tb - ta).total_seconds()
    mins = (seg[-1][0] - seg[0][0]).total_seconds() / 60
    kwh = j / 3.6e6
    mean_w = j / (mins * 60)          # joules / seconds = watts
    return kwh, mean_w, max(p for _, p, _ in seg), mins


for name, seg in (("whole log    ", rows),
                  ("pre-training ", [r for r in rows if r[0] <= SPLIT]),
                  ("base eval    ", [r for r in rows if r[0] >= SPLIT])):
    if len(seg) < 2:
        print(name, "too few samples")
        continue
    kwh, mean, peak, mins = energy(seg)
    print("%s %6.2f min  mean %5.1f W  peak %5.1f W  %.4f kWh" % (name, mins, mean, peak, kwh))

idle = [p for _, p, u in rows if u < 5]
print("\nsamples %d  idle-ish points %d (mean %.1f W)"
      % (len(rows), len(idle), sum(idle) / len(idle) if idle else 0))
print("window %s -> %s" % (t0, t1))
