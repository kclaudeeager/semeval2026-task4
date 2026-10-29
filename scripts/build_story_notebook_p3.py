"""Part 3: the three diagnostics, the figure, and the plan."""
import pickle, pathlib
import nbformat as nbf

C = pickle.loads(pathlib.Path("scripts/_story_p2.pkl").read_bytes())
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): C.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
---
# Three diagnostics

Seven bricks, and the tallest is still the plain encoder. Before concluding
anything, three checks decide how much any of these differences mean.
""")

md(r"""
## Diagnostic 1 — is any of this significant?

The layers are scored on the **same items**, so the right test is paired.
McNemar looks only at the items where two layers disagree:

$$\chi^2 = \frac{(|b_{01} - b_{10}| - 1)^2}{b_{01} + b_{10}}$$
""")

code("""
from task4.stats import mcnemar

def pair(a_spec, b_spec, label):
    a, b = saved(**a_spec), saved(**b_spec)
    m = mcnemar(a["correct"], b["correct"])
    d = b["accuracy"] - a["accuracy"]
    verdict = "SIGNIFICANT" if m["p_value"] < 0.05 else "not significant"
    print(f"  {label:<44} {d:+6.2f}  p={m['p_value']:.3f}  {verdict}")

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
print("every comparison that matters, paired on the same 200 items:\\n")
pair(dict(scorer="l0_jaccard"), dict(scorer="l1_tfidf"), "words -> weighted words")
pair(dict(scorer="l1_tfidf"), dict(scorer="l2_e5"), "weighted words -> meaning")
pair(dict(scorer="l3n_characters_nocoref", actors_only=True),
     dict(scorer="l3x_characters_embed", actors_only=True, embed_model=MINILM),
     "casts: Jaccard inside -> meaning inside")
pair(dict(scorer="l2_e5"),
     dict(scorer="l3x_characters_embed", actors_only=True, embed_model=MINILM),
     "meaning -> structure + meaning")
""")

md(r"""
**Only one comparison in the whole project clears significance: plain words to
meaning.** Everything structural is noise at this sample size.
""")

md(r"""
## Diagnostic 2 — are the structural layers still lexical?

This is the sharpest test in the project, and it does not depend on accuracy at
all. If a layer captures narrative structure, its *mistakes* should look
different from TF-IDF's mistakes. If it is lexical in disguise, it should fail
on the same items.

$\phi$ is the correlation between two layers' **error** vectors. Positive means
shared failures.
""")

code("""
import math

def phi(a, b):
    A, B = [not x for x in a], [not x for x in b]
    n11 = sum(x and y for x, y in zip(A, B)); n10 = sum(x and not y for x, y in zip(A, B))
    n01 = sum((not x) and y for x, y in zip(A, B)); n00 = sum((not x) and (not y) for x, y in zip(A, B))
    den = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return (n11*n00 - n10*n01)/den if den else 0.0

ref = saved(scorer="l1_tfidf")["correct"]
print("error correlation with TF-IDF (lower = less lexical):\\n")
for label, spec in [
    ("brick 4  casts, Jaccard inside", dict(scorer="l3n_characters_nocoref", actors_only=True)),
    ("brick 5  roles, Jaccard inside", dict(scorer="l4n_roles_nocoref", actors_only=True)),
    ("brick 6  casts, meaning inside", dict(scorer="l3x_characters_embed", actors_only=True, embed_model=MINILM)),
    ("brick 3  the encoder alone",     dict(scorer="l2_e5")),
]:
    print(f"  {label:<36} phi = {phi(ref, saved(**spec)['correct']):+.3f}")
""")

md(r"""
The structural layers sit at **+0.22 to +0.30**. The encoder alone sits far
lower. Swapping Jaccard for embeddings inside the alignment moved $\phi$ from
+0.300 to +0.222 — a nudge, not a break.

**The structural layers fail where TF-IDF fails.** That is the finding, and it
holds whatever the accuracy column says.
""")

md(r"""
## Diagnostic 3 — could any combination have worked?

The layers disagree a lot, and the union of all of them covers most of dev. That
looks like headroom. It is not.

On a **binary** task, systems near chance cover a large union automatically. The
honest comparison is against what *independent* systems at the same accuracies
would cover: $1 - \prod_i (1 - p_i)$.
""")

code("""
specs = [dict(scorer="l2_e5"),
         dict(scorer="l3x_characters_embed", actors_only=True, embed_model=MINILM),
         dict(scorer="l4n_roles_nocoref", actors_only=True),
         dict(scorer="l1_tfidf")]
runs = [saved(**s) for s in specs]
n = len(runs[0]["correct"])

observed = sum(any(r["correct"][i] for r in runs) for i in range(n)) / n
independent = 1 - math.prod(1 - r["accuracy"]/100 for r in runs)
majority = sum(sum(r["correct"][i] for r in runs) > len(runs)/2 for i in range(n)) / n

print(f"  union of the four layers            {100*observed:5.1f}%")
print(f"  expected if their errors were independent {100*independent:5.1f}%")
print(f"  shortfall                           {100*(observed-independent):+5.1f} pts")
print(f"\\n  majority vote of the four           {100*majority:5.1f}%   <- below chance")
""")

md(r"""
The union is **below** the independence baseline, and a majority vote lands
*below chance*. Correlated errors do not cancel; they compound. There was never
a combination to find.
""")

md(r"""
---
# The figure

One chart, the whole story. This is the slide.
""")

code("""
import matplotlib.pyplot as plt

LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e",
         "grid": "#e4e3df", "bar": "#2a78d6", "mark": "#eb6834"}

fig, ax = plt.subplots(figsize=(9.2, 0.52 * len(STORY) + 1.8))
labels = [s["label"] for s in STORY]
vals = [s["acc"] for s in STORY]
y = range(len(STORY))

ax.barh(list(y), vals, height=0.6, color=LIGHT["bar"], zorder=3)
ax.axvline(50, color=LIGHT["muted"], lw=2, ls=(0, (4, 3)), zorder=2)
ax.axvline(E5, color=LIGHT["mark"], lw=2, ls=(0, (4, 3)), zorder=2)

for i, s in enumerate(STORY):
    ax.text(s["acc"] + 0.6, i, f"{s['acc']:.1f}", va="center",
            fontsize=10, color=LIGHT["ink"])
    if s["note"]:
        ax.text(s["acc"] + 4.6, i, s["note"], va="center",
                fontsize=8.5, color=LIGHT["muted"])

ax.set_yticks(list(y), labels, color=LIGHT["ink"], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(40, 78)
top = ax.get_ylim()[1]
ax.text(50, top - 0.1, "chance", ha="center", va="bottom",
        fontsize=9, color=LIGHT["muted"])
ax.text(E5, top - 0.1, "the encoder", ha="center", va="bottom",
        fontsize=9, color=LIGHT["mark"])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(LIGHT["grid"])
ax.tick_params(colors=LIGHT["muted"], length=0)
ax.xaxis.grid(True, color=LIGHT["grid"], lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("accuracy on dev (%),  n = 200", color=LIGHT["muted"], fontsize=10)
ax.set_title("Seven bricks. The tallest is still the plain encoder.",
             loc="left", fontsize=13, color=LIGHT["ink"], pad=16)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
fig.tight_layout()
out = ROOT / "reports" / "figures" / "the_story.png"
fig.savefig(out, dpi=200, facecolor=LIGHT["surface"])
print("saved:", out.name)
plt.show()
""")

md(r"""
---
# What this adds up to

Six measurements, pointing the same way:

| # | finding | reading |
|---|---|---|
| 1 | Word overlap scores **below chance** | the benchmark is adversarial by construction |
| 2 | A *better* definition of character costs 4.5 pts (with Jaccard inside) | the layer was not measuring characters |
| 3 | No structural layer beats the encoder | structure adds nothing on top of meaning |
| 4 | A *stronger* encoder inside the alignment makes it worse | the alignment is not feeding real signal |
| 5 | Structural errors correlate with TF-IDF at **+0.22 to +0.30** | the layers are lexical in disguise |
| 6 | Union is **below** the independence baseline; majority vote below chance | there was no combination to find |

And three facts from outside this project that say the same thing:

- **StoryNet** built character/event/theme graphs for this exact task and scored
  **57.75%, rank 39 of 44** — barely above the Jaccard baseline.
- **Fisseni & Löwe (2012)**: untrained readers judge a *temporally reversed*
  story to be the same story, and cite details and motifs, not structure.
- **Elsner (EACL 2012)**: a sentiment-arc-only baseline scored exactly **50** —
  chance — and his character kernel could not distinguish reversed chapters.

The last point is the one worth building the report around. These labels come
from annotators like Fisseni & Löwe's, at $\alpha = 0.33$. A method optimising
for structural correspondence is partly optimising against what the labels
encode. That explains why every symbolic entrant in the task clustered at the
bottom, rather than merely reporting that they did.

## What is honestly still open

**Coreference has never been tested**, and it is the component the hypothesis
actually rests on. Chaturvedi et al. named missed coreference as a source of
error in exactly this setting. The result above is the *pre-coreference*
condition.

## To do

1. Re-run `l5_combined` over the brick 4+5 components **under cross-validation**,
   so the two combination numbers are comparable.
2. Run coreference (`l3_characters`, `l4_roles`) and report L3 minus L3n — that
   difference *is* what coreference contributes.
3. Freeze the configuration, then the single held-out test run.
4. Write, using this notebook's order as the outline.
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
out = pathlib.Path("notebooks/02_building_up.ipynb")
nbf.write(nb, str(out))
for f in ("scripts/_story_p1.pkl", "scripts/_story_p2.pkl"):
    pathlib.Path(f).unlink(missing_ok=True)
print(f"wrote {out} with {len(C)} cells")
