"""Part 2: bricks 3-7, the diagnostics, and the plan."""
import pickle, pathlib
import nbformat as nbf

C = pickle.loads(pathlib.Path("scripts/_story_p1.pkl").read_bytes())
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip()))
def code(t): C.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
---
## Brick 3 — meaning

Words failed. Try meaning: push each story through a sentence encoder trained so
that texts with similar meaning land near each other, then take the cosine.

$$s(a,b) = \mathbf{e}_a \cdot \mathbf{e}_b \qquad \text{with } \|\mathbf{e}\| = 1$$

This is `intfloat/e5-base-v2`, which the organisers also report as a baseline.
""")

code("""
# The real run took about two minutes on CPU; the number comes from that run.
# The implementation is four lines:
#
#     model = SentenceTransformer("intfloat/e5-base-v2")
#     ea = model.encode("query: " + anchor,    normalize_embeddings=True)
#     eb = model.encode("query: " + candidate, normalize_embeddings=True)
#     return float(np.dot(ea, eb))

E5 = brick("3. meaning (e5 embeddings)", saved(scorer="l2_e5")["accuracy"],
           "<- the bar everything else must clear")
""")

md(r"""
**58.50%.** A real jump: +13.5 over word overlap, +8 over chance.

From here on, this is **the bar**. The project's question was never "can we beat
word matching" — it is *can narrative structure add anything to a dense
embedding*. Every brick below is judged against 58.50, not against 45.
""")

md(r"""
---
## Brick 4 — who is in the story

Now the actual hypothesis. Two retellings of one plot share their **cast** even
when they share few words. So: find the characters in each story, work out which
character in A corresponds to which in B, and score how well the two casts line
up.

Two steps, and the second is the interesting one.

**Finding characters.** Without a coreference resolver, group noun phrases by
their head noun. Keep a group if it is a proper noun or ever acts as the subject
of a verb — a cheap behavioural test for "this is a character, not scenery".

**Matching two casts.** Score every possible character pair, then choose the
**one-to-one** matching that maximises the total:

$$\max_{M} \sum_{(x,y)\in M} w(x,y)$$

That is the assignment problem. Greedy matching can be arbitrarily bad; the
Hungarian algorithm solves it exactly.
""")

code("""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Why the matching must be optimal, not greedy.
W = np.array([[0.90, 0.89],
              [0.88, 0.02]])

def greedy(M):
    M = M.copy(); total = 0.0
    for _ in range(min(M.shape)):
        r, c = np.unravel_index(np.argmax(M), M.shape)
        total += M[r, c]; M[r, :] = -np.inf; M[:, c] = -np.inf
    return total / min(W.shape)

rows, cols = linear_sum_assignment(-W)
print("pairwise weights:\\n", W)
print(f"\\ngreedy    {greedy(W):.3f}   takes 0.90 first, then is stuck with 0.02")
print(f"hungarian {W[rows, cols].mean():.3f}   gives up 0.90 to win 0.89 + 0.88")
assert W[rows, cols].mean() > greedy(W)
""")

code("""
from task4.registry import build

caster = build("l3n_characters_nocoref", actors_only=True)
cast = caster.cast(dev[0].anchor)
print(f"{len(cast)} characters found in the anchor:\\n")
for c in cast[:8]:
    print(f"  {c.head:<32} mentioned {len(c.mentions)}x")

brick("4. casts, Jaccard inside",
      saved(scorer="l3n_characters_nocoref", actors_only=False)["accuracy"],
      "all noun chunks")
brick("4b. casts, filtered to actors",
      saved(scorer="l3n_characters_nocoref", actors_only=True)["accuracy"],
      "a BETTER definition scores WORSE")
""")

md(r"""
**57.00 — and then 52.50 when the cast is filtered to things that actually
behave like characters.**

Read that again. Making the definition of "character" *more correct* cost 4.5
points. If the layer were finding character structure, tightening the definition
should have helped.

The straightforward reading: this is not measuring characters. It is measuring
noun-phrase overlap through an expensive pipeline. Brick 6 tests that directly.
""")

md(r"""
---
## Brick 5 — who does what

Characters alone might be too weak. Add the **roles** they play: for each
character, collect the verbs where it is the grammatical subject (agent) and
where it is the object (patient), then compare those sets.

$$r(x,y) = \gamma\, J(A_x, A_y) + (1-\gamma)\, J(P_x, P_y)$$

No semantic role labeller is used, because no maintained one exists — AllenNLP
was archived in 2022 and prompting an LLM for roles reaches 48.6% argument match
against 88.3% for humans. Syntax is the honest proxy.
""")

code("""
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The knight rescued the princess. The dragon burned the village.")
print("who does what, from the parse:\\n")
for tok in doc:
    if tok.dep_ in ("nsubj", "dobj") and tok.head.pos_ == "VERB":
        role = "AGENT  " if tok.dep_ == "nsubj" else "PATIENT"
        print(f"  {role}  {tok.text:<10} of  {tok.head.lemma_}")

brick("5. roles, Jaccard inside",
      saved(scorer="l4n_roles_nocoref", actors_only=False)["accuracy"], "all noun chunks")
""")

md(r"""
---
## Brick 6 — the same structure, compared by meaning

Here is the flaw in bricks 4 and 5, and it is the one that matters.

The alignment is structural. But look at what gets compared *inside* it:

```python
mention = jaccard(set(mentions_x), set(mentions_y))
context = jaccard(set(context_x),  set(context_y))
```

Jaccard. On word sets. Brick 1 established that word overlap fails on this data
— and then bricks 4 and 5 put word overlap right back in the middle of the
structural machinery.

**Fix: keep the alignment, replace the inner comparison with embeddings.** Each
character becomes a small document (its mentions plus surrounding words), that
document is embedded, and aligned characters are compared by cosine.

Structure decides *what* to compare. Meaning decides *how similar* it is.
""")

code("""
brick("6. alignment + meaning inside",
      saved(scorer="l3x_characters_embed", actors_only=True,
            embed_model="sentence-transformers/all-MiniLM-L6-v2")["accuracy"],
      "actor-filtered")
brick("6b. same, all noun chunks",
      saved(scorer="l3x_characters_embed", actors_only=False,
            embed_model="sentence-transformers/all-MiniLM-L6-v2")["accuracy"])
brick("6c. same, stronger encoder (e5)",
      saved(scorer="l3x_characters_embed", actors_only=False,
            embed_model="intfloat/e5-base-v2")["accuracy"],
      "a better encoder made it WORSE")
""")

md(r"""
**58.50 — exactly level with the embedding backbone, and no higher.**

Two details worth keeping:

- The actor filter **stops being a penalty** once the comparison is semantic
  (57.00 → 58.00 unfiltered, 52.50 → 58.50 filtered). The character definition
  was never the problem; the lexical inner step was punishing it.
- A **stronger** encoder (e5 instead of MiniLM) made it **worse**, 58.00 → 56.00.
  If the alignment were feeding real signal to the encoder, better encoding
  should not hurt.
""")

md(r"""
---
## Brick 7 — put it all together

Standardise each component's scores and learn a weighted blend on held-out
folds:

$$z_i = \frac{s_i - \mu_i}{\sigma_i}, \qquad
s_{L5} = \sum_i w_i z_i, \qquad \sum_i w_i = 1$$
""")

code("""
brick("7. combination over bricks 4+5",
      saved(scorer="l5_combined",
            components=["l2_e5", "l3n_characters_nocoref", "l4n_roles_nocoref"])["accuracy"],
      "NOT cross-validated - optimistic")
brick("7b. combination over brick 6",
      saved(scorer="l5_combined",
            components=["l2_e5", "l3x_characters_embed", "l4x_roles_embed"])["accuracy"],
      "cross-validated")
""")

md(r"""
⚠️ **These two are not comparable and must not be reported side by side.** The
first ran before cross-validation was added, so it is an in-sample number; the
second is cross-validated. Re-running the first under cross-validation is on the
to-do list at the bottom.

What *is* safe to say: the cross-validated combination scores **56.00 against
the backbone's 58.50**. Combining did not help.
""")

pathlib.Path("scripts/_story_p2.pkl").write_bytes(pickle.dumps(C))
print(f"part 2 staged: {len(C)} cells")
