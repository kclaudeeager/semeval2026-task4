# Structure as alignment, not as signal

Written 28 October, after the no-coreference layers came back negative.

## The distinction that matters

Two different things get called "using structure", and the literature treats
them very differently.

**Structure as the signal.** Build a graph, compare the graphs, ignore content.
This has failed everywhere it has been tried, including on this exact task:
IIITH Boys' StoryNet built character/event/theme graphs, encoded them with a
Graph Attention Network and compared the resulting vectors. Track A **57.75%,
rank 39 of 44** — against a Jaccard baseline of 56.25%. The overview calls them
"the only team [that] relies primarily on symbolic representations."

**Structure as the alignment.** Use structure to decide *which pieces of story A
correspond to which pieces of story B*, then compare those corresponding pieces
semantically. Every positive result on this genre has this shape:

| work | alignment | comparison | result |
|---|---|---|---|
| Chaturvedi et al. 2018 | Hungarian over characters | character features | P@1 0.558 → 0.637 |
| Opitz et al. 2020 (S²match) | structural AMR alignment | embedding-soft concepts | small but positive |
| Vayer et al. 2019 (FGW) | transport plan over structure | feature cost in the same objective | beats kernels and GCNs |
| Glavaš & Šnajder 2013 | event-graph kernels | plus vector cosine | 86.7 → 95.0 F |
| Barancová & Sileno 2026 | coreference-derived roles | added to dense embeddings | +14.6% relative H@1 |

The pattern is consistent: **structure never replaces content, it routes it.**

## What this project got wrong

L3n and L4n already implement the alignment half. `CharacterAlignmentScorer`
builds casts, computes a pairwise weight matrix, and solves it optimally with
the Hungarian algorithm. That part is right.

The inner comparison was not:

```python
mention = _jaccard(set(...mentions...), set(...))
context = _jaccard(set(...context...), set(...))
```

Jaccard over word sets. Retellings of one story rarely share literal words, so
the structural alignment is immediately handed to a lexical matcher. That is why
L3n's and L4n's errors correlate with TF-IDF at phi = +0.27 and with each other
at +0.446 — they are lexical features with extra steps.

**The alignment was not the problem. What sat inside it was.**

## The experiment

`l3x_characters_embed` and `l4x_roles_embed` keep the alignment exactly as it is
and replace the inner comparison with embedding cosine. Each character becomes a
short document — its head mention plus the words around every mention — which is
embedded once and compared by cosine. Structure decides what to compare; meaning
decides how similar it is.

```bash
python -m task4.cli scorer=l3x_characters_embed data=dev
python -m task4.cli scorer=l4x_roles_embed data=dev
```

Default encoder is MiniLM for speed (the organisers list all-MiniLM-L6-v2 as a
58.50 Track B baseline). If the idea shows anything, re-run with
`scorer.params.embed_model=intfloat/e5-base-v2 scorer.params.embed_prefix="query: "`.

## How to read the outcome

The comparison that matters is **L3x against L3n**, not against L2. Both use the
same casts and the same alignment; the only difference is lexical versus
semantic comparison inside it. McNemar on that pair isolates exactly one
variable, which is more than any single number in the task's leaderboard does.

Three possible outcomes, all reportable:

1. **L3x beats L3n and the error correlation with TF-IDF drops.** Then the
   structural alignment was carrying signal all along and the lexical inner
   comparison was destroying it. That is a genuine finding and the per-layer
   error correlations are the evidence for it.
2. **L3x ≈ L3n.** Then the alignment itself is not finding correspondences that
   matter, and the negative result stands on firmer ground than before, because
   the obvious objection has been tested.
3. **L3x beats L2.** Unlikely, and it would be the result of the project.

Check the error correlation against TF-IDF either way: if it stays near +0.27,
the layer is still lexical whatever the accuracy says.

## What is still untested and why it is a real gap

Gromov-Wasserstein compares the internal geometry of two spaces without needing
any correspondence between their points — mathematically the cleanest version of
"compare the shape, not the labels." Alvarez-Melis & Jaakkola (EMNLP 2018) made
it work for unsupervised word translation. **No published work applies GW or
Fused-GW to narrative similarity.** Fused-GW is the right variant here because
its feature term keeps soft node labels rather than discarding them.

The caution: Xu et al. (ICML 2019) report pure GW "degrades sharply once noise
exceeds 20%", and extraction noise on these summaries is well above that
(Glavaš & Šnajder measured 80% / 60% / 67% F1 for event, temporal and
coreference extraction). So FGW is a defensible *future work* paragraph, not a
four-week plan.

## The constraint nobody can engineer around

Fisseni & Löwe (2012) asked untrained readers whether manipulated variants were
"the same story". Structure was barely mentioned; readers cited details and
motifs; **most judged a temporally reversed story to be the same story**. Elsner
(EACL 2012) found the same from the other side: his character-kernel could not
distinguish chapter-reversed novels at significance, and a sentiment-curve-only
baseline scored exactly 50 — chance.

This task's labels come from annotators of that kind, at alpha = 0.33. A method
optimising for structural correspondence is partly optimising against the target
the labels encode. That, rather than any engineering failure, is the most likely
explanation for why every symbolic entrant finished near the bottom — and it is
the strongest claim this project can make.
