"""Part 2 of the walkthrough notebook: L2-L5, strata, stats, harness, figures."""
import pickle, pathlib
import nbformat as nbf

C = pickle.loads(pathlib.Path("scripts/_cells_part1.pkl").read_bytes())
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): C.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
---
## 4. L2 — dense sentence embeddings

An encoder maps a whole story to a vector $\mathbf{e} \in \mathbb{R}^d$ that is
meant to place similar meanings near each other. Vectors are L2-normalised, so
$\|\mathbf{e}\| = 1$ and cosine similarity reduces to a dot product:

$$s(a,b) = \mathbf{e}_a \cdot \mathbf{e}_b$$

The model is `intfloat/e5-base-v2`. e5 was trained with a `"query: "` /
`"passage: "` prefix convention; here both stories are documents of the same
kind, so both receive the same prefix and the comparison stays symmetric.

This is the **backbone**: the layers that follow are judged on whether they add
anything to it, not on whether they beat word overlap.

The cell below needs `uv pip install -e ".[embeddings]"` and one model download.
It is written to skip cleanly if that is not installed yet.
""")

code("""
try:
    from task4.scorers.embeddings import E5Scorer
    e5 = E5Scorer()
    e5._load()
    c_e5 = run(E5Scorer(), dev, "L2 e5")
except Exception as exc:
    print("L2 not available yet:", type(exc).__name__, str(exc)[:90])
    print('install with: uv pip install -e ".[embeddings]"')
    c_e5 = None
""")

md(r"""
---
## 5. L3 — characters, and the alignment problem

This is where the project's hypothesis starts.

**Step one: who is in the story.** A coreference resolver groups every mention
that refers to the same entity into one chain. "The knight", "he", "Sir Gawain"
collapse into a single `Character`. The project uses the LitBank Maverick model
because it keeps singletons — characters mentioned once still count.

**Step two: matching two casts.** Given casts $X$ and $Y$, which character in
$X$ corresponds to which in $Y$? Score every possible pair:

$$w(x,y) = \alpha \, J(\text{mentions}_x, \text{mentions}_y) + (1-\alpha)\, J(\text{context}_x, \text{context}_y)$$

Then choose the one-to-one matching $M$ that maximises the total:

$$\max_{M} \sum_{(x,y) \in M} w(x,y)$$

This is the **assignment problem**. A greedy pass — take the best pair, then the
best remaining — can be arbitrarily bad. The Hungarian algorithm solves it
exactly in $O(n^3)$; `scipy.optimize.linear_sum_assignment` is that algorithm.
Story similarity is the mean weight of the optimal matching.

The cell below builds casts **by hand**, so the matching mathematics runs
without the coreference model installed.
""")

code("""
import numpy as np
from scipy.optimize import linear_sum_assignment
from task4.scorers.characters import Character, CharacterAlignmentScorer

L3 = CharacterAlignmentScorer.__new__(CharacterAlignmentScorer)
L3.alpha = 0.5

story1 = [Character(mentions=["the knight", "he", "gawain"], context=["rescue","sword","ride","castle"]),
          Character(mentions=["the princess", "she"],         context=["tower","wait","rescue"]),
          Character(mentions=["the dragon", "it"],            context=["fire","guard","tower"])]
story2 = [Character(mentions=["the dragon", "the beast"],     context=["fire","guard","cave"]),
          Character(mentions=["the knight", "sir lancelot"],  context=["rescue","sword","ride"]),
          Character(mentions=["the maiden"],                  context=["cave","wait","rescue"])]

W = np.array([[L3.pair_weight(x, y) for y in story2] for x in story1])
print("pairwise weights (rows = story 1, cols = story 2)")
print(np.round(W, 3))

rows, cols = linear_sum_assignment(-W)
print("\\noptimal matching:")
for r, c in zip(rows, cols):
    print(f"  {story1[r].head:<14} <-> {story2[c].head:<16} w={W[r,c]:.3f}")
print(f"\\nstory similarity = mean of matched weights = {W[rows, cols].mean():.3f}")
""")

code("""
# Why the matching has to be optimal rather than greedy.
def greedy(W):
    W = W.copy(); total = 0.0
    for _ in range(min(W.shape)):
        r, c = np.unravel_index(np.argmax(W), W.shape)
        total += W[r, c]; W[r, :] = -np.inf; W[:, c] = -np.inf
    return total / min(W.shape[0], W.shape[1])

# Greedy grabs 0.90 first, which strands the second row on 0.02.
trap = np.array([[0.90, 0.89],
                 [0.88, 0.02]])
r, c = linear_sum_assignment(-trap)
print("weights:"); print(trap)
print(f"\\ngreedy    : {greedy(trap):.3f}   (takes 0.90, then is left with 0.02)")
print(f"hungarian : {trap[r, c].mean():.3f}   (gives up 0.90 to win 0.89 + 0.88)")
print("\\nThe gap is the whole reason this uses an exact solver rather than a loop.")
""")

md(r"""
---
## 6. L4 — roles, or *who does what to whom*

L3 asks whether two stories contain similar **people**. L4 asks whether those
people **do similar things**. It inherits everything from L3 and changes only
what makes a pairing good.

No maintained PropBank semantic role labeller exists — AllenNLP was archived in
2022, and prompting a language model for roles reaches 48.6% argument match
against 88.3% for human annotators. So roles come from syntax instead:

- verbs where the character is the **subject** $\rightarrow$ agent proxy, $A_x$
- verbs where the character is the **object** $\rightarrow$ patient proxy, $P_x$

Character similarity becomes a blend of the two role sets, and then the whole
pairing weight blends that with L3's mention similarity:

$$r(x,y) = \gamma\, J(A_x, A_y) + (1-\gamma)\, J(P_x, P_y)$$
$$w_{L4}(x,y) = \rho\, r(x,y) + (1-\rho)\, w_{L3}(x,y)$$

$\gamma$ is `agent_weight`, $\rho$ is `role_weight`. Both live in
`configs/scorer/l4_roles.yaml`, so sweeping them is a config change.

This is the mechanism Chaturvedi et al. (2018) used to lift precision-at-1 from
0.558 to 0.637 on movie remakes — except they never resolved coreference, and
named that as a source of error. L3 supplies the piece they lacked.
""")

code("""
from task4.scorers.roles import RoleProfile

knight = RoleProfile(agent={"rescue","ride","fight"}, patient={"wound"})
lancelot = RoleProfile(agent={"rescue","ride","charge"}, patient={"wound"})
princess = RoleProfile(agent={"wait"}, patient={"rescue","capture"})

print(f"knight vs lancelot (both act) : {knight.similarity(lancelot):.3f}")
print(f"knight vs princess (opposite) : {knight.similarity(princess):.3f}")

print("\\nthe same pair under different agent weights:")
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    print(f"  gamma={g:.2f} -> {knight.similarity(lancelot, agent_weight=g):.3f}")
""")

md(r"""
Notice what the last cell shows. The knight and Lancelot overlap strongly as
**agents**; the knight and the princess overlap only in that one *rescues* and
the other *is rescued*. A bag-of-words model cannot tell those apart, because
the word "rescue" appears in both. That asymmetry is the signal this project is
testing for.

---
## 7. L5 — combining the layers

Component scores live on different scales: a Jaccard value sits in $[0,1]$, a
cosine sits near 0.8, an alignment mean sits wherever it lands. Adding them
directly would let whichever has the largest variance dominate. So each is
standardised using statistics from the **fitting split only**:

$$z_i = \frac{s_i - \mu_i}{\sigma_i}, \qquad s_{L5} = \sum_i w_i z_i,\quad \sum_i w_i = 1$$

Weights are found by grid search over the simplex in steps of `grid_step`. With
three components and a step of 0.1 that is 66 candidate weight vectors — small
enough to enumerate, and the count is capped in advance and recorded in the run
config so the search space is part of the record.
""")

code("""
import itertools
step = 0.1
grid = [c for c in itertools.product([round(i*step,4) for i in range(int(1/step)+1)], repeat=3)
        if abs(sum(c) - 1.0) < 1e-9]
print(f"{len(grid)} weight vectors on the 3-component simplex at step {step}")
print("first five:", grid[:5])
print("\\nz-normalisation keeps one component from swamping the others:")
raw = {"e5 cosine": [0.80, 0.82, 0.79], "jaccard": [0.05, 0.31, 0.12]}
for k, v in raw.items():
    m = sum(v)/len(v); sd = (sum((x-m)**2 for x in v)/len(v))**0.5
    print(f"  {k:<10} spread before {max(v)-min(v):.3f}  ->  after {(max(v)-m)/sd - (min(v)-m)/sd:.3f}")
""")

md(r"""
---
## 8. `strata.py` — the rule that came out of a mistake

The diagnostic asks: *where* does a layer help? Each annotator marked which
aspects drove their decision, so items can be split by aspect.

The obvious rule is a majority vote. With **two** annotators that rule is
broken. One of two marking an aspect is not a majority, so the item lands in
"not marked" — pooled together with items where *neither* annotator marked it.
Those are not the same thing at all.

This is not hypothetical. Under the majority rule, TF-IDF scored 62.4% on
"not marked" and 54.7% on "marked" on test, a 7.7-point gap. The same split on
dev **reversed** to 44.3 against 53.4. Neither gap was significant. The cause
was the pooling.

So this module sorts every item into three buckets and the diagnostic uses only
the first two:

| bucket | meaning |
|---|---|
| `all_marked` | every annotator flagged the aspect |
| `none_marked` | no annotator flagged it |
| `split` | they disagreed — excluded |
""")

code("""
from task4.strata import Stratum, aspect_stratum, partition, usable

test = load_triples(DATA / "test" / "labels" / "test_track_a_labels.jsonl")

for aspect in ("outcomes", "course_of_actions", "abstract_theme"):
    p = partition(test, aspect)
    print(f"{aspect:<20} all={len(p[Stratum.ALL_MARKED]):>3}  "
          f"none={len(p[Stratum.NONE_MARKED]):>3}  split={len(p[Stratum.SPLIT]):>3}  "
          f"usable={usable(test, aspect)}")
""")

md(r"""
That table decides the diagnostic:

- **`abstract_theme` is useless** — every annotator marked it on 392 of 400 items,
  so there is no contrast to measure.
- **`course_of_actions` is lopsided** — 188 against 32. A 32-item stratum carries
  a confidence interval too wide to conclude anything.
- **`outcomes` is the one that works** — 79 against 133, both large enough.

So the final diagnostic reports `outcomes`. That decision is in
`configs/eval/diagnostic.yaml`, derived from the data rather than from taste.
""")

md(r"""
---
## 9. `stats.py` — honest arithmetic in one place

Four tools, each for a different question.

**Wilson interval** — where does the true accuracy plausibly sit? Unlike the
normal approximation it never runs past 0 or 100 and stays sane at small $n$:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p (1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

**Bootstrap** — resample the items with replacement $B$ times, recompute
accuracy each time, and take the 2.5th and 97.5th percentiles. Assumes nothing
about the shape of the distribution.

**McNemar's test** — the right way to compare two systems, because they are
scored on the *same items*. Only the disagreements carry information. Let $b_{01}$
be items where only B is right and $b_{10}$ where only A is right:

$$\chi^2 = \frac{(|b_{01} - b_{10}| - 1)^2}{b_{01} + b_{10}}$$

Below 25 discordant pairs the module uses the exact binomial test instead.

**Two-proportion z** — for disjoint groups, such as two strata. It is *not* a
system comparison, and the result carries a note saying so.
""")

code("""
from task4.stats import wilson, bootstrap, mcnemar, two_proportion_z, describe

print("Wilson   ", wilson(sum(c_jac), len(c_jac)))
print("Bootstrap", bootstrap(c_jac, n_boot=2000, seed=0))

print("\\nTF-IDF vs Jaccard on dev, paired:")
print(" ", mcnemar(c_jac, c_tfidf))

print("\\nthe proposal's stratified gap, recomputed (116/186 vs 117/214):")
print(" ", two_proportion_z(116, 186, 117, 214))
""")

code("""
# The instructor's note: compute the spread of the field, not just the winner.
track_a = {"best": 78.00, "strongest_baseline": 67.00, "worst_ranked": 53.50}
print("Track A field, 44 ranked systems:", describe(list(track_a.values())))
print("\\nmean 66.51 and SD 6.23 in the proposal came from this routine over the")
print("overview's per-system table. Re-run it there for the final report.")
""")

md(r"""
---
## 10. `evaluate.py` — the harness everything goes through

One function takes a layer and a split and returns everything needed to report
it: accuracy, two kinds of interval, and the per-stratum breakdown. It writes
one JSON file carrying the resolved config, the git commit, the seed, and the
**per-item correctness vector**.

That last field is what makes McNemar possible weeks later without rerunning
anything, and it is why results are committed to git while Hydra's run folders
are not.
""")

code("""
from task4.evaluate import evaluate, compare, save

res_jac = evaluate(JaccardScorer(), dev, "dev", n_boot=1000)
print(f"{res_jac.scorer}: {res_jac.accuracy:.2f}%  "
      f"bootstrap [{res_jac.ci_low:.2f}, {res_jac.ci_high:.2f}]  "
      f"wilson [{res_jac.wilson_low:.2f}, {res_jac.wilson_high:.2f}]")
print("\\nstrata on outcomes:")
for k, v in res_jac.strata["outcomes"].items():
    print(" ", k, v)
""")

code("""
tf = TfidfScorer(); tf.fit(dev)
res_tf = evaluate(tf, dev, "dev", n_boot=1000)
verdict = compare(res_jac, res_tf)
print("McNemar, TF-IDF against Jaccard on the same 200 items:")
for k, v in verdict.items():
    print(f"  {k:<10} {v}")
""")

md(r"""
---
## 11. Figures for the report and the talk

`plots.py` reads the saved result files, so a chart can never disagree with the
numbers in the write-up — both come from the same JSON. Colours are a validated
categorical palette: checked for colour-vision separation, and supplied in light
and dark steps so a slide deck in either theme works.

Single-series charts carry no legend and label their bars directly, which is
also what keeps them readable when a projector washes out the colour.
""")

code("""
from task4.plots import load_results, ablation_chart, strata_chart
from IPython.display import Image, SVG, display

saved = load_results(RESULTS, "dev")
print("layers with saved dev results:", sorted(saved))

p = ablation_chart(saved, FIGURES / "ablation_dev.png",
                   title="Accuracy by layer (dev, n=200)", human_ceiling=89)
display(Image(filename=str(p)))
""")

code("""
# The same figure in dark steps, for a dark slide template.
p = ablation_chart(saved, FIGURES / "ablation_dev_dark.png",
                   title="Accuracy by layer (dev, n=200)", human_ceiling=89, dark=True)
display(Image(filename=str(p)))
""")

code("""
try:
    p = strata_chart(saved, FIGURES / "strata_outcomes_dev.png", aspect="outcomes")
    display(Image(filename=str(p)))
except ValueError as exc:
    print("strata chart needs runs carrying strata:", exc)
""")

md(r"""
---
## 12. How the pieces connect

""")

code("""
display(SVG(filename=str(FIGURES / "system_diagram.svg")))
""")

md(r"""
---
## 13. Leaving the notebook

Everything above runs from the command line, with the same code and the same
statistics. The notebook is for understanding; the CLI is for the record,
because every CLI run writes a result file stamped with its config and commit.

```bash
# one layer, one split
python -m task4.cli scorer=l1_tfidf data=dev

# change a parameter without touching code
python -m task4.cli scorer=l4_roles scorer.params.agent_weight=0.7

# every layer, swept
make ablation

# the diagnostic, paired against the L2 backbone
make diagnostic

# the single held-out run, which refuses to start without the explicit unlock
python -m task4.cli --multirun experiment=ablation data=test eval.allow_test=true
```

**Next steps** are dated in `docs/PLAN.md`: L2 running by 2 November, L3 plus the
50-item coreference audit by 9 November, L4 and L5 by 16 November, configuration
frozen 19 November, the one test run on 20 November, writing and the video
through 25 November.
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
out = pathlib.Path("notebooks/01_walkthrough.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
pathlib.Path("scripts/_cells_part1.pkl").unlink(missing_ok=True)
print(f"wrote {out} with {len(C)} cells")
