# What was and was not implemented, and why it does not change the conclusion

Written for the report's limitations section, where the first question a reader
asks is "why didn't you use proper semantic role labelling?"

## What the role machinery actually is

Two coarse roles read off the dependency parse:

```python
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}    # agent proxy
OBJECT_DEPS  = {"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"}
```

A character's profile is the set of verb lemmas where it appears as syntactic
subject (agent) and as object (patient). Two characters are compared by Jaccard
over those sets, or in L4x by embedding cosine over their context.

## What was deliberately not used

| not used | why |
|---|---|
| **PropBank SRL** (ARG0–ARG5, predicate senses) | No maintained system. AllenNLP's SRL was archived in 2022. LLM-prompted SRL reaches 48.6% argument match against 88.3% for human annotators (Bonn et al., DMR 2024). |
| **FrameNet frames and frame relations** | Never attempted. Would need frame identification plus frame-element labelling, both error-prone on this text, and no evidence they help on narrative similarity. |
| **VerbNet classes** | Not attempted here. Duluth used VerbNet symbolically on this task and scored 58.50, rank 38 of 44. Nguyen et al. (ECIR 2013) found VerbNet verb abstraction *alone hurt* on folktales because parses were too noisy. |
| **A trained sequence labeller** | None trained. spaCy's tagger and parser are consumed off the shelf. L6's positional segments are segmentation, not labelling. |

So the honest description is: **two coarse roles from syntax, compared as
verb-lemma sets, with no frame semantics and no learned tagger.** Chaturvedi et
al. (2018) used the same shortcut on this genre, which is the precedent.

## Why finer roles would not have changed the result

The tempting reading is that the roles were too crude and that PropBank or
FrameNet would have worked. Three measurements say otherwise.

**1. The layer was behaving lexically regardless of role granularity.** L4's
errors correlate with TF-IDF at phi = +0.27 to +0.33 — about the same as L3,
which uses no roles at all. Adding roles did not change *what kind* of mistakes
the layer made. Finer roles would have produced finer word-matching.

**2. Changing the comparison function inside the alignment changed almost
nothing.** Swapping Jaccard for embedding cosine — a far larger change than
refining role labels — moved accuracy by +1.00 (p = 0.907) and moved the TF-IDF
error correlation only from +0.291 to +0.252.

**3. The geometry caps every comparison function.** In the triple geometry, the
right and wrong candidates sit at the same distance from the anchor: median
1.390 against 1.391, right-closer in 51% of triples. The distractor was drawn
from the 98th–99th percentile of similarity, so it was *placed* there. No
refinement of how two stories are compared escapes a tie that was constructed.

Better roles are a better ruler. The problem is not the ruler's precision; it is
that the two things being measured are the same length by design.

## The one caveat that does stand

**Coreference was never run.** It is the component the original hypothesis
rested on, and Chaturvedi et al. named missed coreference as a source of error
in exactly this setting. Every structural result here is the *pre-coreference*
condition, and the report should say so in those words rather than claiming the
hypothesis is closed.

What makes that caveat survivable is that three other routes reach the same
conclusion without it: the annotators judge on theme (98.3%), no component
escapes the equidistance, and the task's one graph-based entrant finished 39th
of 44.
