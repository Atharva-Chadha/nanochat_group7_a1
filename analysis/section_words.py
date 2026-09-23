"""Per-section prose word counts, floats excluded, so cuts can be targeted."""
import re
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

ROOT = _paths.REPORT
s = open(os.path.join(ROOT, "main.tex"), encoding="utf-8").read()
body = s.split(r"\appendix")[0].split(r"\begin{document}")[1]


def words(t):
    t = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$&_^~\\]", " ", t)
    return len(t.split())


heads = [(m.start(), m.group(1), m.group(2))
         for m in re.finditer(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}", body)]
heads.append((len(body), "end", ""))

print("%-9s %-52s %5s" % ("level", "heading", "words"))
print("-" * 70)
for (a, lvl, name), (b, _, _) in zip(heads, heads[1:]):
    w = words(body[a:b])
    mark = " <<<" if w > 450 else ""
    print("%-9s %-52s %5d%s" % (lvl, name[:52], w, mark))
print("-" * 70)
print("%-62s %5d" % ("TOTAL body prose", words(body)))
