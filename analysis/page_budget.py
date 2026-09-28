"""Rough page budget: prose words outside floats, plus float count."""
import re
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

ROOT = _paths.REPORT
s = open(os.path.join(ROOT, "main.tex"), encoding="utf-8").read()
body = s.split(r"\appendix")[0]
body = body.split(r"\begin{document}")[1]

# the star has to be in the pattern: a table* is the most expensive kind of float,
# and without it the cell text was being counted as prose
FLOAT = r"\\begin\{(table|figure)(\*?)\}.*?\\end\{\1\2\}"
floats = [m[0] for m in re.findall(FLOAT, body, re.S)]
prose = re.sub(FLOAT, " ", body, flags=re.S)
prose = re.sub(r"\\(bibliography|bibliographystyle)\{[^}]*\}", " ", prose)


def words(t):
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$&_^~\\]", " ", t)
    return len(t.split())


pw = words(prose)
print("prose words (body, floats excluded):", pw)
print("floats in body: %d (%d tables, %d figures)"
      % (len(floats), sum(f == "table" for f in floats), sum(f == "figure" for f in floats)))

# 10pt a4 2cm margins single column ~= 620-700 words of prose per full page
for wpp in (620, 700):
    text_pages = pw / wpp
    print("  at %d w/page: %.1f pages of prose" % (wpp, text_pages))
print("floats typically consume 0.25-0.45 page each ->",
      "%.1f to %.1f pages" % (len(floats) * 0.25, len(floats) * 0.45))
n_bib = len(re.findall(r"@\w+\{", open(os.path.join(ROOT, "main.bib"), encoding="utf-8").read()))
print("references block: ~0.5 page for %d entries" % n_bib)
