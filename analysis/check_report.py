"""Static sanity check for main.tex: refs, cites, environments, braces, length."""
import re
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

ROOT = _paths.REPORT
with open(os.path.join(ROOT, "main.tex"), encoding="utf-8") as fh:
    s = fh.read()
with open(os.path.join(ROOT, "main.bib"), encoding="utf-8") as fh:
    bibsrc = fh.read()

labels = set(re.findall(r"\\label\{([^}]*)\}", s))
refs = {r.strip() for g in re.findall(r"\\(?:[cC]ref|ref)\{([^}]*)\}", s) for r in g.split(",")}
cites = {c.strip() for g in re.findall(r"\\cite[tp]?\{([^}]*)\}", s) for c in g.split(",")}
bib = set(re.findall(r"@\w+\{([^,]+),", bibsrc))

print("dangling refs :", sorted(refs - labels) or "none")
print("unused labels :", sorted(labels - refs) or "none")
print("missing cites :", sorted(cites - bib) or "none")
print("unused bib    :", len(bib - cites), "of", len(bib), sorted(bib - cites))

st = []
for kind, env in re.findall(r"\\(begin|end)\{([^}]*)\}", s):
    if kind == "begin":
        st.append(env)
    elif not st or st[-1] != env:
        print("ENV MISMATCH at", env, "| open:", st[-3:])
        break
    else:
        st.pop()
print("env stack     :", st or "balanced")
print("braces        :", s.count("{") - s.count("}"))
print("inline $      :", "even" if s.count("$") % 2 == 0 else "ODD -- unbalanced")

for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", s):
    p = os.path.join(ROOT, g)
    if not os.path.exists(p):
        print("MISSING FIGURE:", g)

print("TODO markers  :", len(re.findall(r"\\TODO\{", s)))
body = re.sub(r"\\[a-zA-Z]+|[{}$&\\]", " ", s)
print("approx words  :", len(body.split()))
