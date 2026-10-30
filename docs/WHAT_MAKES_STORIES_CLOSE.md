# What actually makes two stories close

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
