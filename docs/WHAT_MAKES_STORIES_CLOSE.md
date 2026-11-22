# What actually makes two stories close

> ## ⚠ Superseded in part — read this first
>
> This file is a snapshot from late October. The component table in it is
> wrong in a specific way, and any summary built from this file will repeat
> it:
>
> - **The figures 56.0 / 54.0 / 52.7 are not component accuracies.** They are
>   per-component *right-closer rates* from the geometry notebook at n=150 —
>   a distance statistic, not an accuracy. They were mislabelled.
> - The real component accuracies, computed by `make components` over
>   n=200, are **theme 55.0%, action 54.5%, outcome 54.0%**, every interval
>   containing 50, with the component-wise combination at 53.00%.
> - **The rank order of the three carries no information.** A one-point
>   spread over 200 items is two items. An earlier reading treated the
>   ordering as evidence the decomposition was real; it is not.
>
> `docs/EXPERIMENT_NARRATIVE.md` is the current account.

Asked of the annotators rather than the models, on 30 October, by reading the
226 free-text comments they left across the 600 labelled triples.

Annotator comments are evidence about the labels. They are quoted here as data.

## The operative criterion is theme, and it is not close

How often each aspect was marked as driving the decision:

| aspect | marked on | share |
|---|---|---|
| **abstract theme** | 590 / 600 | **98.3%** |
| course of action | 317 / 600 | 52.8% |
| outcomes | 144 / 600 | 24.0% |

And what they reach for when they explain themselves in prose:

| what the comment talks about | share of 226 comments |
|---|---|
| **theme, emotional subject** | **80.1%** |
| course of action | 48.2% |
| outcome, ending | 45.1% |
| characters | 33.6% |
| setting | 10.6% |

**Characters appear in a third of the explanations. Theme appears in four
fifths.** This project spent its effort on the aspect annotators cite least and
ignored the one they cite nearly always.

## What the comments look like

> "Text B is closer to the anchor because both stories center around a virus
> changing the situation, but they differ in how the outcomes unfold."

> "Anchor is about a boy's difficult childhood that ends with a shocking
> discovery, taking away his innocence. Text A is about a girl's lonely life in
> a troubled home that leads to a big family secret and a new start."

> "Text A is narratively closer because both stories focus on criminals involved
> in robbery, police chases, and complicated relationships."

The shape is always the same. The annotator **compresses each story to a
sentence of gist, then compares the gists.** Median comment length is 38 words
against a 116-word summary — roughly a 3x compression. They are not matching
characters, events, or sequence. They are abstracting first and comparing
second.

Note also *what survives* the compression: "a boy", "a girl", "a newcomer",
"criminals". Generic roles, not identities. Names appear, but the comparison is
carried by the abstraction.

## Why every layer in this project failed

Each one compared stories at the level of their *content*:

| layer | what it compares | aspect it targets |
|---|---|---|
| Jaccard, TF-IDF | words | none |
| e5 | whole-summary meaning | theme, indirectly |
| L3, L3x | characters | the 33.6% aspect |
| L4, L4x | agent/patient verbs | course of action, 52.8% |
| L6 | positional segments | sequence, barely mentioned |

Nothing in the stack performs the operation the annotators perform: **abstract
first, then compare.** e5 comes closest by accident, because a sentence encoder
partially abstracts, and e5 is the layer that scores highest. That is not a
coincidence, it is the finding.

## This explains the winner

COGNAC took first place at 78% by decomposing each story into *abstract theme,
course of action and outcome* **as text**, then comparing those. Not a graph:
text, at the level of abstraction the annotators work at, in the aspect
vocabulary the annotation scheme itself uses.

The evidence above says why that works and why graphs do not. The winning
representation is the one that matches how the labels were produced.

## The honest conclusion for the report

Structural methods did not fail because they were badly built. They failed
because they model what a story *is made of*, while the labels record what a
story *is about*.

Fisseni & Löwe (2012) predicted this from the other direction: untrained readers
judge a temporally reversed story to be the same story, and cite details and
motifs rather than structure. These annotators, at alpha = 0.33, behave the same
way. A method optimising for structural correspondence is optimising against
the target.

## What follows from this

The one layer worth building next is the one that imitates the annotator:
compress each story to its theme in a sentence or two, then compare those
compressions. That is a layer in this harness like any other, it inherits the
whole evaluation path, and it is the only idea left that is supported both by
the task's own winner and by the annotators' own words.

---

# The component programme

Two phases, in this order. The order is the point.

## Phase 1 — are the components real?

`python scripts/validate_components.py dev`

Each component is run **alone**, then split by whether annotators marked the
matching aspect on that triple. A component that genuinely captures "outcome"
should score higher on the triples where annotators said outcomes drove their
decision than on the triples where they said it did not.

This is a **validity** test, not an accuracy test, and it comes first. A
component that fails it is mislabelled, and no amount of clever alignment
downstream repairs a mislabelled component. Only unanimous strata are used.

Read the *direction* before the p-value. These strata are small — the power
analysis puts the detectable difference at 17 points — so a non-significant
difference in the predicted direction is weak support, while a difference in the
wrong direction is a problem whatever its p-value.

## Phase 2 — does comparing component-wise help?

`make components` runs both phases. Each component alone, then the weighted
combination with weights fitted by grid search.

**The fitted weight vector is the result**, more than the accuracy is. It is a
direct estimate of what each component contributes, measured on this data rather
than asserted. If it puts everything on theme, that is the finding, and it
agrees with the annotators marking abstract theme on 98.3% of triples.

## How the components are extracted, and why so crudely

| component | extraction |
|---|---|
| theme | the whole summary |
| action | the sentences densest in non-copular verbs that have a subject |
| outcome | the closing quarter of the sentences |

No LLM, so it runs on a laptop. The crudeness is deliberate: the question is
whether components extracted this cheaply carry the signal the annotation scheme
says they should. If they do, the decomposition is doing the work rather than
the extractor's sophistication. If they do not, phase 1 says so before any time
goes into phase 2.

Theme is the whole summary on purpose. A sentence encoder already abstracts, and
the whole-summary embedding is the strongest single layer in this project — that
is itself the evidence that it proxies theme.

---

# Phase 1 result: no component escapes the equidistance

Run 30 October, n = 150 dev triples, MiniLM.

| component | right candidate closer | 95% CI | p vs 50% |
|---|---|---|---|
| theme | 56.0% | [48.0, 63.7] | 0.165 |
| action | 54.0% | [46.0, 61.8] | 0.369 |
| outcome | 52.7% | [44.7, 60.5] | 0.568 |

**Every interval contains 50%.** None of the three components separates the
right candidate from the wrong one better than a coin flip.

And the direction is the opposite of what the decomposition hypothesis needs.
The hypothesis was that the whole-story view *averages away* a signal that
survives in a part — so some component should beat theme. Instead **theme is the
strongest and outcome the weakest.** Theme is the whole summary. Slicing the
story loses signal rather than revealing it.

## The ordering is the interesting part

| component | annotators cite it | geometric signal |
|---|---|---|
| theme | 98.3% | 56.0% |
| action | 52.8% | 54.0% |
| outcome | 24.0% | 52.7% |

The rank order is identical. The components that annotators invoke most often
are the ones carrying the most geometric separation, in the same order.

This is three points, so a matching rank order happens by chance one time in
six, and no individual difference is significant. It is corroboration, not
proof. But it is the second independent route to the same conclusion: **this
task is decided by what a story is about, and the whole-summary view is already
the best available estimate of that.**

## What this closes

The component programme is answered before it was built. Phase 2 would fit
weights over three components, and the geometry predicts where those weights
will go: onto theme, which is the whole summary, which is the `l2_e5` layer that
has led the table since the beginning.

Running `make components` once is still worth doing, as a **confirmation** with
a concrete artifact — the fitted weight vector is a single sentence for the
report. It is no longer an exploration.
