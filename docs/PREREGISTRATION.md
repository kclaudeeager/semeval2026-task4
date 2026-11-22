# Pre-registration for the single test run

Written before the test split is touched. Nothing in this file may be edited
after the run; corrections go in a dated section at the bottom.

The point of writing it down is that a decision made after seeing results is
not the same decision. One team in this shared task was disqualified for
overfitting the public test labels. The guard in `cli.py` and in
`lexical_direction.py` enforces the mechanics; this file enforces the intent.

---

## Date of freeze

`_____2026-11-21___` (fill in before running, and do not change afterwards)

## Git SHA at freeze

`____ a9b9881____` (`git rev-parse --short HEAD`)

---

## 1. The confirmatory claim

**One claim, decided in advance.**

> No structural layer improves on the `e5` embedding backbone (L2) on the
> Track A test split.

Tested by McNemar on the same items, L2 against the best structural layer as
measured on development data, which is `l3x_characters_embed` with
`actors_only=true` and MiniLM span embeddings.

Pre-specified: significance level 0.05, two-sided. The power analysis
(`docs/POWER.md`) gives a minimum detectable difference of 4.3 to 7.6 points
on 400 test items, so the test is only able to detect a difference of roughly
that size. A null result is therefore consistent with both "no effect" and
"an effect smaller than this design can see", and will be reported as such.

**This is the only confirmatory test.** Everything below is descriptive or
exploratory and will be labelled that way in the report.

## 2. Layers to run on test

Fixed now. Adding a layer after seeing any test number invalidates the run.

- [ ] `l0_random`
- [ ] `l0_jaccard` (already run once, to reconcile against the published baseline)
- [ ] `l1_tfidf`
- [ ] `l2_e5`  — the backbone
- [ ] `l3x_characters_embed`, `actors_only=true`, MiniLM — the best structural layer
- [ ] `l5_combined` over L3x, cross-validated

Layers deliberately NOT run on test, and why:

- `l3_characters` / `l4_roles` (with coreference) — only if the dev runs
  complete before the freeze. If they do not, the report says the
  coreference variant could not be run on the available hardware.
- `l6` / `l6e` (rhythm) — built, never run on development either. Reporting a
  test number for a layer with no development result would be indefensible.
- `l7` components — development result is well below the backbone; no reason
  to spend a test evaluation on it.

## 3. Descriptive measures to report on test

Not tests of a hypothesis. Reported with intervals, no significance claims.

- Accuracy and bootstrap interval for every layer in section 2.
- Per-stratum accuracy on the unanimous `outcomes` strata (79 and 133 items).
  The power analysis says this contrast needs 17.3 points, which no plausible
  effect reaches, so it is reported as description and **no** claim is made
  from it either way.

## 4. Pre-specified exploratory measure: lexical direction

Declared here **before** the run, because the dev result is the kind of number
that invites a decision made afterwards.

On development data, distributional divergence from the anchor is
*anti*-correlated with the gold label: cross-entropy of the candidate under
the anchor's unigram model over content words selects the distractor on
59.5% of triples (the chosen candidate wins 81/200, 40.5%, exact binomial
p = 0.009). The mechanism is the sampling procedure — distractors were drawn
from the 98th–99th percentile of `story-emb` similarity, so they are selected
for surface proximity, while the correct answer is a paraphrasing retelling.

**What will be reported:** the same measure, computed on test by
`scripts/lexical_direction.py test`, with the direction and the binomial test.

**What will not be claimed:** that this constitutes a system. The decision to
read the measure in the inverted direction was made after seeing development
data. If the effect holds on test it is evidence of a construction artifact in
the benchmark, which is a finding about the data. It is not a method, it will
not appear in the results table as a system, and no accuracy figure derived
from inverting it will be presented as a contribution.

**What would falsify it:** a test right-closer rate within the interval
containing 50%, which would indicate the dev effect was noise.

## 5. Protocol

1. Confirm `make audit` is green and `make test` passes.
2. Record the git SHA above.
3. Run the layers in section 2 with `eval.allow_test=true`, once.
4. Run `TASK4_ALLOW_TEST=1 python scripts/lexical_direction.py test`, once.
5. `make numbers`, `make audit`.
6. Change no configuration afterwards. If something is broken, say so in the
   report rather than re-running.

## 6. Corrections after the freeze

Anything discovered after the run goes here, dated, rather than by editing
the text above.

<!-- e.g. 2026-11-__: layer X failed to run because ...; reported as such. -->

**2026-11-21 — tokenizer reconciliation of the L0 baseline on test.**
`scripts/tokenizer_reconciliation.py` recomputed the `l0_jaccard` baseline on
the test split under ten tokenizations, to explain the 1.75-point gap between
the measured 58.00% and the published 56.25%. Section 2 of this document lists
`l0_jaccard` as already run once "to reconcile against the published baseline",
and this is that reconciliation carried through to an explanation.

What it did **not** do: it fit nothing, selected nothing, and changed no
reported result. The L0 baseline is a reference point in the data section, not a
term in any comparison, and the confirmatory test in section 1 does not involve
it. The tokenizer in `scorers/lexical.py` was left unchanged, so every accuracy
reported elsewhere is the same number it was before this run. The cause
(apostrophe handling) and the direction of the bias (it flatters the baseline by
1.75 points) are both stated in the report.

**2026-11-21 — component set for `l5_combined`.** Section 2 specified the
combination over L3x. The first test run used the configuration default
instead, which is the coreference variant set
(`l2_e5`, `l3_characters`, `l4_roles`). The pre-registered variant was run
afterwards. Both are reported:

| components | weights fitted on dev | test |
|---|---|---|
| `l2_e5`, `l3_characters`, `l4_roles` | 1.0, 0.0, 0.0 | 63.50% |
| `l2_e5`, `l3x_characters_embed`, `l4x_roles_embed` (pre-registered) | 0.3, 0.0, 0.7 | 57.75% |

Neither result is withheld. The second run was made because the specification
named it, not because the first was unfavourable.

**2026-11-21 — audit whitelist.** `scripts/audit.py` previously permitted only
`l0_jaccard` on the test split. It was widened to the six layers named in
Section 2 after those runs were made, so that the check reflects the
pre-registration rather than the pre-registration's first line.
