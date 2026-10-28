"""Generate notebooks/01_walkthrough.ipynb. Run: python scripts/build_notebook.py"""
import nbformat as nbf

C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): C.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Task 4 Track A — a walkthrough of every module

This notebook opens each part of `src/task4/`, says what it does, gives the
mathematics it implements, and runs it on real data from the release.

It is meant to be read top to bottom once, then used as a bench: change a number
in a cell, re-run that cell, see what moves. When you are done playing here, the
same code runs from the command line through `task4.cli`, and the last section
shows those commands.

**The task.** You are given three story summaries: an anchor $a$ and two
candidates $b_1, b_2$. Decide which candidate is narratively closer to the
anchor. Every layer in this project answers that question the same way: it
defines a similarity function $s(\cdot,\cdot)$ and predicts

$$\hat{y} = \mathbb{1}\left[\, s(a, b_1) \ge s(a, b_2) \,\right]$$

So a layer is nothing but a choice of $s$. That is the whole design.
""")

code("""
import json, sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / "raw" / "narrative-similarity-dataset"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
print("project root:", ROOT)
print("dataset present:", DATA.exists())
""")

md(r"""
---
## 1. `dataset.py` — turning files into objects

The release ships JSON Lines: one JSON object per line. This module reads those
lines into a `Triple`, so nothing downstream ever has to know the file format.

A `Triple` carries the three texts, the gold label (`a_is_closer`), and three
lists of annotator flags — one list per similarity aspect. Each list has one
entry per annotator, `True` if that annotator said this aspect drove the
decision.

Two derived properties matter later:

- `n_annotators` — 2 for most items, 3 where a third was called in to break a tie
- `unanimous` — whether the annotators agreed on the binary label
""")

code("""
from task4.dataset import load_triples, Triple

dev = load_triples(DATA / "dev" / "dev_track_a.jsonl")
print(f"{len(dev)} dev triples")

t = dev[0]
print("\\nanchor :", t.anchor[:110], "...")
print("cand A :", t.text_a[:110], "...")
print("cand B :", t.text_b[:110], "...")
print("\\ngold: A is closer =", t.a_is_closer)
print("annotators:", t.n_annotators, "| unanimous:", t.unanimous)
print("aspect flags -> outcomes:", t.outcomes,
      "| course_of_actions:", t.course_of_actions,
      "| abstract_theme:", t.abstract_theme)
""")

code("""
from collections import Counter

print("annotators per item:", dict(Counter(x.n_annotators for x in dev)))
print("unanimous:", sum(x.unanimous for x in dev), "of", len(dev))
print("label balance (A closer):", sum(bool(x.a_is_closer) for x in dev), "of", len(dev))
lens = sorted(len(x.anchor.split()) for x in dev)
print("anchor length in words -> median", lens[len(lens)//2], "range", lens[0], "-", lens[-1])
""")

md(r"""
The third annotator appears **only** where the first two disagreed. That is why
"unanimous" and "two annotators" turn out to be the same set — a fact that
matters in section 7.

---
## 2. `scorers/base.py` — the one interface

Every layer subclasses `Scorer` and implements a single method:

```python
def score(self, anchor: str, candidate: str) -> float
```

Higher means more similar. The base class supplies `predict`, which applies the
decision rule from the top of this notebook to a whole split, and `fit`, which
does nothing unless a layer needs a pass over the corpus first.

Because the interface is this narrow, a new idea is a new `score` and nothing
else. The harness, the statistics and the plots never change.
""")

code("""
from task4.scorers.base import Scorer

class LengthScorer(Scorer):
    \"\"\"A deliberately silly layer: prefer the candidate closest in length.

    Written here only to show how little it takes to add one.\"\"\"
    name = "demo_length"
    def score(self, anchor, candidate):
        return -abs(len(anchor.split()) - len(candidate.split()))

preds = LengthScorer().predict(dev)
hits = sum(p == bool(t.a_is_closer) for p, t in zip(preds, dev))
print(f"length heuristic: {100*hits/len(dev):.2f}% on dev  (chance is 50)")
""")

md(r"""
---
## 3. L0 and L1 — the lexical layers

### Jaccard overlap

Treat each story as a **set** of word types. For two sets $A$ and $B$:

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

It is 1 when the vocabularies are identical and 0 when they share nothing. It
ignores word order, repetition and meaning entirely — which is the point. It
measures how much of the task is solvable by surface overlap alone.

### TF-IDF cosine

Each story becomes a vector over terms. Term $t$ in document $d$ gets

$$w_{t,d} = \left(1 + \log \mathrm{tf}_{t,d}\right) \cdot \log \frac{N}{\mathrm{df}_t}$$

The first factor dampens repetition (sublinear term frequency); the second
discounts terms that appear in many documents. Similarity is the cosine of the
angle between the two vectors:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}$$

**Where the document frequencies come from matters.** $\mathrm{df}_t$ is counted
over whatever split `fit()` is given. Count it over dev *and* test and you have
leaked test information into a dev number. This project fits on dev alone, which
is why the dev TF-IDF figure here is **51.50**, not the 49.0 that appeared in
the proposal under the leaky protocol.
""")

code("""
from task4.scorers.lexical import JaccardScorer, TfidfScorer, RandomScorer, tokenize

print("tokenizer:", tokenize(\"The Dog's, bark! 3 times\"))

jac = JaccardScorer()
a, b = "a knight rescues a princess from a dragon", "a princess is rescued from a dragon"
c = "a chemist synthesises a compound in a laboratory"
print(f"\\nJ(anchor, related)   = {jac.score(a,b):.3f}")
print(f"J(anchor, unrelated) = {jac.score(a,c):.3f}")
""")

code("""
def run(scorer, triples, label):
    scorer.fit(triples)
    preds = scorer.predict(triples)
    correct = [p == bool(t.a_is_closer) for p, t in zip(preds, triples)]
    print(f"{label:<14} {100*sum(correct)/len(correct):6.2f}%")
    return correct

print("accuracy on dev (n=200)")
c_rand  = run(RandomScorer(seed=0), dev, "L0 random")
c_jac   = run(JaccardScorer(),      dev, "L0 jaccard")
c_tfidf = run(TfidfScorer(),        dev, "L1 tfidf")
""")

md(r"""
Jaccard lands **below** chance on dev. That is not a bug. The organisers kept
only triples where two commercial language models disagreed, so the easy signals
were filtered out on purpose; on dev, word overlap is mildly anti-correlated
with the right answer.
""")
nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
import pathlib, pickle
pathlib.Path("scripts/_cells_part1.pkl").write_bytes(pickle.dumps(C))
print(f"part 1: {len(C)} cells staged")
