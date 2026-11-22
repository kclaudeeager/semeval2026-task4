# Seed results, 28 October

> ## ⚠ Superseded in part — read this first
>
> This file is a snapshot from late October. Two figures in it are no longer
> live results, and any summary built from this file will repeat them:
>
> - **59.00% for the L5 fitted combination is an in-sample number.** Weights
>   were selected on dev and the same split was scored. It has been moved to
>   `reports/archive/` and `make audit` fails if it reappears among live
>   results. The cross-validated combination scores **56.00%**, below the L2
>   backbone. Against L2 the in-sample figure is +0.50 points at p = 1.000.
> - **"None of the gaps is significant" understates it.** Nothing in the
>   ablation reaches p < 0.05 at all, including the largest effect
>   (TF-IDF to e5, +8.00 points, p = 0.10).
>
> `docs/EXPERIMENT_NARRATIVE.md` is the current account.

Run four weeks before the deadline, deliberately, to find out what breaks while
there is still time to change course.

## What could and could not be run

HuggingFace is unreachable from this machine's sandbox, so **coreference (L3)
and the e5 encoder (L2) could not run**. GitHub is reachable, so spaCy and its
English model installed, which gives a real dependency parse.

That makes the layers below *the condition Chaturvedi et al. (2018) actually
ran*: characters from surface mentions, roles from the parse, **no coreference
anywhere**. They are labelled `l3n` and `l4n` for "no coreference". Running them
first is useful on its own — they fix the pre-coreference point on this
benchmark, so the eventual L3 minus L3n difference measures what coreference
contributes rather than what the whole pipeline contributes.

## Dev results (n = 200)

| layer | accuracy | 95% bootstrap CI |
|---|---|---|
| L0 random | 46.00 | [39.5, 52.5] |
| L0 Jaccard | 45.00 | [38.0, 52.0] |
| L1 TF-IDF | 51.50 | [44.0, 58.5] |
| L3n characters, *actors only* | 52.50 | [45.5, 59.5] |
| L3n characters, *all noun chunks* | **57.00** | [50.0, 63.5] |
| L4n roles | 53.50 | [47.0, 60.0] |

Paired comparisons (McNemar, same 200 items):

| comparison | delta | p |
|---|---|---|
| Jaccard → TF-IDF | +6.50 | 0.137 |
| TF-IDF → L3n (actors only) | +1.00 | 0.909 |
| TF-IDF → L3n (all noun chunks) | +5.50 | 0.242 |
| L3n → L4n roles | +1.00 | 0.850 |
| **Jaccard → L3n (all noun chunks)** | **+12.00** | **0.006** |

## The finding that matters

**Restricting the cast to things that behave like characters makes accuracy
worse: 57.00 drops to 52.50.**

The filter keeps a mention group if it is a proper noun or ever appears as the
subject of a verb. It is a defensible definition of "character" — it keeps
*the organization*, *citizens*, *mary murphy*, and discards *safety*,
*its mission*, *the book*. Applying it costs 4.5 points.

The straightforward reading is that the gain over Jaccard is **not coming from
character structure**. It is coming from noun-phrase content overlap: a richer
lexical feature, computed through an expensive pipeline. The only comparison
that clears significance, Jaccard → L3n at +12.00, p = 0.006, is also the
comparison where the baseline is weakest, and L3n's advantage over TF-IDF
(+5.50, p = 0.242) does not clear it.

This is evidence against the proposal's hypothesis in its current form. It is
not yet decisive, for three reasons worth stating plainly:

1. **Coreference is exactly what is missing**, and it is the component the
   hypothesis rests on. Chaturvedi et al. named missed coreference as a source
   of error in this same setting. The honest statement today is that the
   pre-coreference condition shows no character-specific signal.
2. **n = 200 on dev.** The power analysis puts the detectable effect at 5.9-10.7
   points. A real 3-point effect would be invisible here.
3. **Dev is the harder split** (Jaccard 45.00 on dev against 58.00 on test), so
   dev understates everything.

## What changes because of this

1. **Add the actors-only contrast to the report.** It is a clean, cheap ablation
   that asks "is it characters, or is it nouns?" and today the answer leans
   towards nouns. That question is more interesting than the original one.
2. **L2 is now the priority**, not L3. Without the embedding backbone there is
   no way to tell whether any of this adds to a dense representation, which is
   the actual claim. Everything above may be noise around a 66.75-level baseline.
3. **Budget compute for L4.** The first L4n implementation rebuilt each
   character's role profile inside the pair loop and did not finish a 200-item
   split. Fixed by computing profiles once per character; a dev run is now
   about 60 seconds.
4. **Expect the same filtering question for real L3.** When coreference lands,
   run it both ways.

## Engineering faults found and fixed

- **Result files silently overwrote each other.** Two runs of one layer under
  different parameters wrote to the same filename, and the second destroyed the
  first. Filenames now carry an 8-character hash of the scorer parameters.
- **L4 was quadratic in redundant work**, as above.
- **Background runs were being reaped** when the shell exited; they need
  `setsid` on this machine.

---

# Update, 28 October: the backbone landed

`l2_e5` ran on dev: **58.50%** [52.0, 65.5]. It is now the best single layer.

| layer | dev | vs L2, paired |
|---|---|---|
| **L2 e5 embeddings** | **58.50** | — |
| L3n characters (all noun chunks) | 57.00 | −1.50, p = 0.828 |
| L4n roles | 53.50 | −5.00, p = 0.332 |
| L3n characters (actors only) | 52.50 | −6.00, p = 0.207 |
| L1 TF-IDF | 51.50 | −7.00, p = 0.151 |

No structural layer beats the backbone, and none of the gaps is significant.

## Does combining them have any headroom?

The per-item vectors say the layers disagree a lot: L2 and L3n agree on only
57.5% of items, and L3n gets 41 items right that L2 gets wrong. The union of
all four layers covers 86.5% of dev, close to the 89% human ceiling. That looks
like a great deal of room for a combiner.

**It is not.** On a binary task, four systems at 51–58% accuracy would cover
**96.0%** of items by chance alone if their errors were independent. The
observed union is 86.5%, which is **9.5 points below** the independence
baseline. The layers are failing on the *same* items.

The pairwise error correlations confirm it, and the structure of them is the
interesting part:

| pair | phi |
|---|---|
| L3n vs L4n | **+0.446** |
| L4n vs TF-IDF | +0.279 |
| L3n vs TF-IDF | +0.269 |
| L2 vs TF-IDF | +0.178 |
| L2 vs L4n | +0.130 |
| L2 vs L3n | +0.129 |

L3n and L4n are nearly redundant with each other (+0.446), and both correlate
with TF-IDF about as much as they do with the embeddings. That is what you would
expect if the structural layers are, as the actors-only result already
suggested, mostly measuring lexical content through a more expensive pipeline.

A naive majority vote over the four layers scores **45.5%** — below chance.
Correlated errors do not average out; they compound.

## What this means for the hypothesis

Three independent lines now point the same way:

1. Filtering the cast to things that behave like characters **costs** 4.5 points.
2. No structural layer beats the embedding backbone.
3. The structural layers' errors correlate with each other and with TF-IDF more
   than they reduce the backbone's errors.

The reading is that these features capture lexical content, not narrative
structure. **Coreference is still the untested component**, and it is the one
the hypothesis actually rests on — so this is not yet the final word. But the
pre-coreference evidence is consistently negative, and the report should say so
in those terms rather than waiting for a result that rescues it.

## Still to run

`python -m task4.cli scorer=l5n_combined_nocoref data=dev` — the fitted
combination of L2 + L3n + L4n. Given the error correlations, expect it to land
at or just above L2 alone. Run it anyway: "the combination does not beat its
strongest component" is a reportable result, and it is the experiment the
proposal promised.

---

# Update 2, 28 October: the combination adds nothing

`l5n_combined_nocoref` — L2 + L3n + L4n, weights fitted by grid search — scored
**59.00%** on dev against L2's 58.50%.

| | dev |
|---|---|
| L5 fitted combination | 59.00 |
| L2 alone | 58.50 |
| delta | **+0.50** |

Paired: it fixes 22 items and breaks 21. **p = 1.000.**

That is the proposal's central experiment, and the answer is no. Worse than the
size of the gap is the condition it was measured under: the weights were fitted
on dev and scored on dev. Given the most favourable setup available — allowed to
tune on the very items it was graded on — the combination could not beat its
strongest component by more than a single item.

## Two faults this exposed

**The fitted weights were never recorded.** `weights: null` went into the config
and nothing came out. The most informative number a combiner run produces — what
it decided L3n and L4n were worth — was discarded. `Scorer.fitted_params()` now
returns what `fit()` learned and it lands in the result JSON.

**Fitting and scoring the same split is circular.** Any layer that learns
parameters is now cross-validated over 5 folds when the fitting split and the
evaluation split coincide, so no item contributes to the parameters used to
score it. The correction is real: TF-IDF drops from **51.50 to 50.50** under
cross-validation. Stateless layers are exempt by a `fits_parameters` flag —
re-fitting a seeded random baseline per fold would move its number without
making it more honest.

Re-run L5 to get its cross-validated figure, which is the one for the report:

```bash
python -m task4.cli scorer=l5n_combined_nocoref data=dev
```

Expect it at or below 58.50, and expect the fitted weights to be worth reading.

## Where the project stands

Every result points the same way, and the proposal anticipated this outcome:

| finding | reading |
|---|---|
| Filtering to real characters costs 4.5 pts | the signal is not character-specific |
| No structural layer beats L2 | structure adds nothing over embeddings |
| Union 86.5% vs 96.0% expected under independence | the layers fail on the same items |
| L3n/L4n error correlation +0.446 | the two structural layers are near-redundant |
| Fitted combination +0.50, p = 1.000, fitted on its own test | combining does not help |

Coreference is still untested and remains the one component that could change
this. It should be run. But the report should be written around the negative
result that five independent measurements now support, with coreference as the
final test of it rather than the hoped-for rescue.

---

# Update 3, 30 October: why it is hard

The obvious objection to everything above is that comparing two short texts
should not be difficult. Measuring the *margin* — how much daylight a method
puts between the right and the wrong candidate — says why it is.

On dev, with the gold answer known:

| method | mean signed margin (right − wrong) | median absolute margin |
|---|---|---|
| Jaccard | **−0.0024** | 0.0166 |
| TF-IDF | +0.0045 | 0.0180 |

**Jaccard's mean margin is negative.** On average the *wrong* candidate shares
more vocabulary with the anchor than the right one does. That is not an absence
of signal, it is inverted signal, and it is exactly what the construction
predicts: the distractor is drawn from the 98th–99th percentile of similarity
under `story-emb`, corrected by subtracting `e5` similarity. The wrong answer
was selected *because* it looks close.

TF-IDF's margin is positive but negligible — a median separation of 0.018 on a
cosine scale. On the 20% of triples where Jaccard puts the candidates within 5%
of each other it is right 48% of the time, which is a coin flip.

So the question is not "why did structure fail". It is "why would anything
succeed", and the answer is that the benchmark was built so that similarity
itself points the wrong way. A method has to find a signal that survives after
surface similarity has been deliberately equalised.

`evaluate()` now records the per-item signed margin alongside the correctness
vector, so this analysis can be run for any layer.
