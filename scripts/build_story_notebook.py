"""Build notebooks/02_building_up.ipynb - the brick-by-brick narrative."""
import pickle, pathlib
import nbformat as nbf

C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): C.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Building it up, one brick at a time

This notebook is the argument of the project in the order it was discovered.
Each section adds **one** idea, writes it in plain code, measures it, and asks a
single question: *did that buy anything?*

Read it top to bottom and you have the report. Present it top to bottom and you
have the talk.

## The plan

| brick | idea | question it answers |
|---|---|---|
| 0 | the decision rule | what are we even predicting? |
| 1 | shared words | is this solvable by word overlap? |
| 2 | weighted words | does weighting rare words rescue it? |
| 3 | meaning | how far does a sentence encoder get? |
| 4 | who is in the story | do the casts line up? |
| 5 | who does what | do the roles line up? |
| 6 | structure as alignment | align structurally, compare by meaning |
| 7 | all of it together | does combining help? |

Then three diagnostics that decide how to read the numbers, and the plan for
what remains.

**A note on how this runs.** Each brick shows the real implementation in plain
code and checks it on a handful of items so the notebook stays fast. The
headline numbers are read from `reports/results/`, where the full runs were
written by `task4.cli`. Nothing here recomputes a three-minute run.
""")

code("""
import json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / "raw" / "narrative-similarity-dataset"

def saved(**must_match):
    \"\"\"Pull a finished run out of reports/results by its parameters.\"\"\"
    for p in sorted((ROOT / "reports" / "results").glob("dev__*.json")):
        d = json.loads(p.read_text())
        params = (d["config"]["scorer"] or {}).get("params") or {}
        if all(params.get(k) == v for k, v in must_match.items() if k != "scorer"):
            if d["scorer"] == must_match["scorer"]:
                return d
    raise KeyError(f"no saved run matching {must_match}")

STORY = []          # the running record every chart is drawn from

def brick(label, acc, note=""):
    STORY.append({"label": label, "acc": acc, "note": note})
    bar = "#" * int(round(acc - 40))
    print(f"{label:<34} {acc:5.2f}  {bar}")
    return acc

print("project root:", ROOT.name)
""")

md(r"""
---
## Brick 0 — what are we predicting?

Three summaries: an anchor $a$, and two candidates $b_1, b_2$. Say which
candidate is narratively closer.

Every method in this project is a similarity function $s$, and they all make the
decision the same way:

$$\hat{y} = \mathbb{1}\left[\, s(a, b_1) \ge s(a, b_2) \,\right]$$

So a "method" is only ever a choice of $s$. That is the whole design, and it is
why every brick below is a handful of lines.
""")

code("""
from task4.dataset import load_triples

dev = load_triples(DATA / "dev" / "dev_track_a.jsonl")
t = dev[0]
print(f"{len(dev)} triples in dev\\n")
print("ANCHOR   :", t.anchor[:150], "...")
print("\\nCAND A   :", t.text_a[:150], "...")
print("\\nCAND B   :", t.text_b[:150], "...")
print("\\ngold answer: A is closer =", t.a_is_closer)
""")

code("""
def decide(score_fn, triples):
    \"\"\"The decision rule, once, for every method below.\"\"\"
    return [score_fn(x.anchor, x.text_a) >= score_fn(x.anchor, x.text_b) for x in triples]

def measure(score_fn, triples):
    hits = [p == bool(x.a_is_closer) for p, x in zip(decide(score_fn, triples), triples)]
    return 100 * sum(hits) / len(hits)

import random
rng = random.Random(0)
brick("0. random guessing", saved(scorer="l0_random", seed=0)["accuracy"],
      "the floor: 50% by construction")
""")

md(r"""
---
## Brick 1 — shared words

The simplest possible idea. Treat each story as a *set of words* and count the
overlap:

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$

Five lines.
""")

code("""
import re
TOKEN = re.compile(r"[a-z0-9']+")

def words(text):
    return set(TOKEN.findall(text.lower()))

def jaccard(a, b):
    A, B = words(a), words(b)
    return len(A & B) / len(A | B) if A | B else 0.0

# sanity check on something obvious
assert jaccard("a knight rescues a princess", "a princess is rescued") > \\
       jaccard("a knight rescues a princess", "a chemist runs an experiment")
print("sanity check passed\\n")

print("on the first 40 dev triples:", round(measure(jaccard, dev[:40]), 2), "%")
brick("1. shared words (Jaccard)", saved(scorer="l0_jaccard")["accuracy"],
      "BELOW chance")
""")

md(r"""
**45.00%. Below the 50% chance line.** Word overlap is not merely useless here,
it is mildly *anti*-correlated with the right answer.

That is not a bug, it is the benchmark working as designed. The organisers kept
only triples where two commercial language models **disagreed** about the
answer, and drew the wrong candidate from the 98th–99th percentile of
similarity. Everything easy was filtered out on purpose.

So brick 1 fails, and the failure is informative: whatever solves this task, it
will not be word matching.
""")

md(r"""
---
## Brick 2 — weight the words

Maybe raw overlap is too crude. Rare words should count more than "the". That is
TF-IDF: weight each term by how often it appears here against how rare it is
overall, then take the cosine between the two vectors.

$$w_{t,d} = (1 + \log \mathrm{tf}_{t,d}) \cdot \log \frac{N}{\mathrm{df}_t}
\qquad
s(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$$
""")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [s for x in dev for s in (x.anchor, x.text_a, x.text_b)]
vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                      ngram_range=(1, 2), min_df=2).fit(corpus)

def tfidf(a, b):
    M = vec.transform([a, b])
    return float(cosine_similarity(M[0], M[1])[0][0])

brick("2. weighted words (TF-IDF)", saved(scorer="l1_tfidf")["accuracy"],
      "cross-validated")
""")

md(r"""
**50.50%. Chance.**

This brick also taught the project a lesson that changed how everything after it
was measured.

The first time TF-IDF ran it scored **51.50**, because the document frequencies
were counted over dev *and* test — the model had seen the test vocabulary.
Counting them on dev alone drops it to 51.50 → and cross-validating (fit the
weights on four fifths, predict the fifth) drops it again to **50.50**.

Two full points of the original number were the measurement flattering itself.
Every layer that learns anything is cross-validated from here on.
""")

code("""
# The honesty rule, demonstrated. Fit on everything vs fit on four fifths.
import statistics

def cv_accuracy(fit_and_score, triples, k=5, seed=0):
    idx = list(range(len(triples)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    hits = []
    for f in range(k):
        held = set(folds[f])
        train = [triples[i] for i in idx if i not in held]
        test  = [triples[i] for i in folds[f]]
        score = fit_and_score(train)
        hits += [p == bool(x.a_is_closer) for p, x in zip(decide(score, test), test)]
    return 100 * sum(hits) / len(hits)

def fit_tfidf(train):
    c = [s for x in train for s in (x.anchor, x.text_a, x.text_b)]
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                        ngram_range=(1, 2), min_df=2).fit(c)
    def score(a, b):
        M = v.transform([a, b])
        return float(cosine_similarity(M[0], M[1])[0][0])
    return score

print(f"fit on all of dev, scored on dev : {measure(tfidf, dev):5.2f}%   <- flattering")
print(f"5-fold cross-validated           : {cv_accuracy(fit_tfidf, dev):5.2f}%   <- honest")
""")

pathlib.Path("scripts/_story_p1.pkl").write_bytes(pickle.dumps(C))
print(f"part 1 staged: {len(C)} cells")
