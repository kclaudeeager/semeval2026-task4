# What we did, in plain terms

The story of the SemEval-2026 Task 4 project, written to be read start to
finish. Each method says which part of the course it comes from, what it does,
what it scored, and what it taught.

Every number is in `reports/results/`, sourced in `report/PROVENANCE.md`, and
recomputed a second way by `make audit`.

Parts 1–13 are the story. The appendices hold the working material: how to
write Section 7 (A), the report outline (B), the course mapping (C), and the
citations still to verify (D).

---

## 1. The problem

You get three story summaries: an anchor and two candidates. Pick the candidate
more similar to the anchor. Accuracy is the only score. Chance is 50%.

Three numbers fix the scale:

- The winner, COGNAC, got **78.00%**.
- The best organizer baseline — a zero-shot GPT-4o-mini prompt — got
  **67.00%**, about 22nd of 44 ranked systems.
- The estimated human ceiling is about **89%**.

So the real range is 50 to 89. There are 39 points of headroom, not 50. This
matters for how you talk about every result:

| | accuracy | above chance | share of headroom |
|---|---|---|---|
| Jaccard | 45.0 | −5.0 | below chance |
| TF-IDF | 50.5 | +0.5 | 1% |
| **our best (e5)** | **58.5** | **+8.5** | **22%** |
| StoryNet | 57.75 | +7.75 | 20% |
| COGNAC | 78.0 | +28.0 | 72% |
| human | 89.0 | +39.0 | 100% |

"We got 58.5 and the winner got 78" sounds like a distant failure. "We recovered
a fifth of the available headroom" is the same fact at the right scale.

Agreement between annotators is low: Krippendorff's α is 0.33 overall and 0.05
to 0.11 per dimension. That is why the human ceiling is 89 and not 100.

### Why 89% and α = 0.33 are not a contradiction

They measure different things. Agreement is annotator against annotator: about
67% raw. The 89% is one human against the **gold** label, and the gold was
built by combining annotators — two, plus a third to adjudicate splits.
Combining removes noise, so you are comparing a person against a committee.

The arithmetic fits. If each annotator picks the consensus answer with
probability *p*, two agree when both are right or both wrong:
*p*² + (1−*p*)² = 0.67 gives *p* ≈ 0.79. The majority of three such annotators
is right 3*p*²(1−*p*) + *p*³ ≈ 0.89 of the time. *(Arithmetic done here, not a
published derivation — the organizers do not state how they computed 89%.)*

Two consequences. The gold label is defined **by** the annotators, so the
"human upper bound" is a ceiling on agreeing with a consensus, not on
recovering an external truth. And aggregation rescues the binary label but
cannot rescue the per-dimension flags at α = 0.05–0.11 — an order of magnitude
worse, and no amount of combining repairs a signal that weak. That is why the
validity test in Section 4 comes out inconclusive.

**Course connection.** This is ambiguity, at the level of the judgment rather
than the word. Two careful people read the same three stories and disagree
about which pair is closer. The ceiling is a property of the construct.

## 2. What the data turned out to be

This section explains every result that follows.

Summaries come from *Tell Me Again*, cut to four to eight sentences. Each
anchor is picked at random. Then both candidates are chosen using `story-emb`,
an encoder trained so that different summaries of the same story land close
together:

- the **nearest** story becomes the assumed answer;
- the **distractor** is taken from the 98th–99th similarity percentile.

Then two commercial language models judge each triple, and **only the triples
they disagree on go to human annotators.** Of 1,237 triples, 831 had two
annotators agree and 208 more were settled by a third, leaving 1,039.

Read that again. The items that survive are the ones where a narrative-trained
encoder's best guess is unreliable and two strong language models cannot agree.
They were selected to defeat the kind of method we then built.

You can see it in the data straight away. Jaccard overlap scores **45.00%** on
dev — below chance — with a mean signed margin of −0.0024. The distractor shares
slightly *more* vocabulary with the anchor than the answer does. The sign is
consistent, so this is the sampling procedure, not noise.

Annotators also recorded which dimensions drove each decision. Over the 600
annotated dev and test triples, a majority of annotators marked abstract theme
on **98.3%**, course of action on **52.8%**, outcomes on **24.0%**.

Those do not sum to 100%, and should not. The three are independent
checkboxes, not one choice: an annotator ticks **2.08 of the three** on
average. The rates say how often each dimension was invoked, not how the
decisions divide between them.

The definition matters, because three give three different tables. Majority of
annotators per triple: 98.3 / 52.8 / 24.0. *Every* annotator: 95.0 / 44.0 /
20.0. Per individual judgment: 97.7 / 68.2 / 41.7. The spread between them is
the low per-dimension agreement showing up directly. The stratified analysis
uses the unanimous version, because mixing disagreement with genuine absence
is what made the first attempt at the diagnostic meaningless.

One more thing follows. A dimension marked on 95% of items is
**near-constant**, and a near-constant flag carries almost no discriminative
information — there is nothing to contrast against. That is why theme cannot
be validated, and why outcomes, the least-cited dimension, is the only one
with usable strata. The most-cited dimension is the least analytically
useful.

## 3. The hypothesis

If word overlap is weak by design, the next place to look is narrative
structure: who acts, who is affected, and whether those relationships survive
retelling even when the words change.

This was not a hunch. Two papers support it:

- Chaturvedi et al. (2018) on movie remakes: an event model underperformed
  bag-of-words, but matching **characters** through dependency-derived agent and
  patient features raised precision@1 from 0.558 to 0.637.
- Barancová and Sileno (2026) on *Tell Me Again* itself: coreference-derived
  character roles plus dense embeddings gave a 14.6% relative retrieval gain
  across four encoders.

Two cut against it:

- Narrative-chain embeddings that win at narrative cloze identify similar
  stories poorly.
- In this very task, StoryNet's symbolic narrative graphs got **57.75%**, 39th
  of 44. The winning team abandoned three attempts at Propp's folktale
  functions.

A live hypothesis with real evidence on both sides. That is the right kind of
thing to test.

## 4. The design: one change at a time

Instead of building one system, we built a stack. Each layer adds exactly one
representation, and every layer runs through the same evaluation code. That
way a difference between two layers can be blamed on one thing.

### L0 — random and Jaccard

**Course topic:** tokenization; lexical overlap.

Random choice, and word-set overlap. The floor.

**Result:** 46.00% and 45.00% on dev. Jaccard hit 58.00% on test against a
published 56.25% — a 7-item gap, traced on 21 Nov to apostrophe handling in the
tokenizer and **reported rather than corrected**. `[a-z0-9']+` keeps
possessives and contractions whole (`father's`, `doesn't`), where `\w+` splits
them; 930 token types differ, 74.7% of them apostrophe-bearing. 17 test items
change prediction, 12 for the current tokenizer and 5 against, netting the 7.
Four ordinary alternatives — `\w+`, whitespace, alphabetic-only and
case-sensitive — reproduce the published 56.25% exactly. The tokenizer is kept
for consistency with the rest of the pipeline, and the report states that it
flatters this baseline by 1.75 points.

**The check that mattered more:** the below-chance *dev* result is not an
artifact of that choice. All ten tokenizations fall between 44.00% and 47.50%,
every one below chance. See `scripts/tokenizer_reconciliation.py`.

**What it taught:** reproduce the published baseline first, and when it does not
reproduce, measure the cause instead of asserting one. The discrepancy was small
and boring; finding it early is what made every later number believable, and
the version of this paragraph that claimed it had been *fixed* stood uncorrected
for three weeks.

### L1 — TF-IDF cosine

**Course topic:** vector semantics, sparse representations. tf-idf weighting,
cosine similarity, L2 normalization.

How much word-overlap signal exists at all, weighted properly this time.

**Result:** 50.50%, cross-validated. Essentially nothing.

**What it taught:** the first version scored 51.50% because it was fitted and
evaluated on the same data. Adding cross-validation cost a full point. Any
layer that fits parameters must be cross-validated or its number is fiction.
This is now enforced — `make audit` fails if a fitting layer lacks per-fold
weights.

**A later question that changed the picture.** Jaccard ignores word frequency;
TF-IDF cosine normalizes length away. A distributional divergence keeps both.
Measuring several (`scripts/lexical_direction.py`, dev, n=200):

| measure | chosen candidate wins | p |
|---|---|---|
| Jensen–Shannon (symmetric) | 49.5% | 0.94 |
| KL(anchor ‖ candidate) | **42.5%** | **0.040** |
| KL(candidate ‖ anchor) | 56.0% | 0.10 |
| cross-entropy, content words | **40.5%** | **0.009** |
| length difference | 47.0% | 0.44 |

Below 50% means the measure prefers the *distractor*. The lexical signal is
not merely absent — it is significantly reversed, and **cross-entropy at
p = 0.009 is the most significant single effect measured anywhere in this
project.** Nothing in the ablation reached p < 0.05; a lexical measure
pointing backwards did.

The asymmetry is the informative part. The symmetric measure sees nothing; the
two directional ones disagree. Read together: the distractor covers the
anchor's vocabulary better, while the correct answer introduces more novel
words. Length does not explain it. That is what sampling distractors at the
98th–99th percentile did to the lexicon — they are selected for surface
proximity, while the true answer is a *retelling*, and retellings paraphrase.

**The trap, and why it stays a trap.** A measure at 40.5% inverts to 59.5%,
which would beat the embedding backbone, and the mechanism above is a real
causal story rather than a bare sign flip. But the decision to invert is made
*after* seeing development data, and a system built on it would be measuring
how the triples were sampled rather than narrative similarity. This is a
dataset artifact — the category of hypothesis-only baselines in natural
language inference, which the field treated as evidence a benchmark was broken
rather than as a modelling result. It is reported as a property of the data,
pre-registered in `docs/PREREGISTRATION.md` before the test run, and never
presented as a system.

### L2 — sentence embeddings

**Course topic:** vector semantics, dense representations; meaning
representation. A transformer encoder trained contrastively, pooled over the
whole summary.

`e5-base-v2`, whole summaries, ranked by cosine.

**Result:** **58.50%** — the biggest single jump, +8.00 over TF-IDF. But
McNemar gives **p = 0.10** over 84 discordant pairs. Even our largest effect is
not significant at n=200.

**What it taught:** the gap between a point estimate and a defensible claim.
Eight points looks decisive in a table and dissolves under a paired test.

### L3 — character alignment

**Course topic:** coreference resolution; named entity recognition;
entity-based coherence.

Find each story's characters, then match the two casts one-to-one using the
Hungarian algorithm over mention-string and contextual similarity.

**Result:** without coreference, 57.00% using all nouns and 52.50% restricted
to entities that pass a character filter. With embedded spans, 58.00% and
58.50%. The best variant ties L2 exactly — 32 items won, 32 lost, p = 0.90.

**What it taught:** the Hungarian algorithm properly — greedy and optimal
matching genuinely differ, and we built a case where greedy scores 0.460
against Hungarian's 0.885. Also the filter lesson below, which is the more
uncomfortable one.

### L4 — dependency-derived roles

**Course topic:** dependency structure; semantic role labelling.

Verbs where a character is the subject approximate agent actions; where it is
the object, patient roles. Alignments scored on shared role–lemma pairs rather
than mentions alone.

**Result:** 56.00% to 57.50% depending on variant. Nothing above L2.

**What it taught:** two separate things.

First, why we used a proxy: AllenNLP was archived in 2022, leaving no
maintained PropBank SRL system, and LLM-prompted SRL reaches 48.6% argument
match against 88.3% for humans.

Second, and this is the better point, **why the proxy is lossy**:

> *John broke the window.* → `window` is **dobj**
> *The window broke.* → `window` is **nsubj**

The window is the patient both times. A dependency proxy calls it an object in
one sentence and an agent in the other. Every unaccusative verb, every passive
without a by-phrase, every middle construction in the corpus does this. That is
the standard reason semantic role labelling exists as a task separate from
parsing — and it is a named mechanism for why L4 added nothing.

### L5 — weighted combination

**Course topic:** model combination; cross-validation.

Combine L2, L3 and L4 with weights chosen by cross-validation.

**Result:** **56.00%** — worse than L2 alone.

There is also a 59.00% figure from an earlier run. It was fitted and scored on
the same split. It now sits in `reports/archive/` with a README saying why, and
the audit fails if it reappears among the live results. Against L2 the gap is
+0.50 points at p = 1.000.

**What it taught:** the most attractive number in the project was an artefact.
We only know because cross-validation was added later and the old run was
re-examined instead of quoted.

### L6 — rhythm (built, not run)

**Course topic:** discourse structure; sequence alignment (DTW).

Compare stories by *pacing* rather than content: positional segments, and
elastic matching that allows stretching.

**Result:** none. The code exists and is tested. By the time it was ready the
geometry result below made the outcome predictable.

**What it taught:** when to stop building. This is the one layer where not
running it was the right call, and it should be reported as a decision rather
than hidden as a gap.

### L7 — component decomposition

**Course topic:** discourse structure. Positional segmentation: theme = whole
text, action = verb-dense sentences, outcome = closing quarter.

Cut each summary into theme, action and outcome; compare component by
component.

**Result:** each alone — theme 55.0%, action 54.5%, outcome 54.0%. Every
confidence interval contains 50. Combined with cross-validated weights,
**53.00%** — below each component on its own.

**What it taught:** the validity check matters more than the accuracy. We asked
whether a span labelled "outcome" actually does better on triples where
annotators marked outcomes. Action moved the right way, outcome moved the wrong
way, and **theme could not be tested at all**: annotators marked it unanimously
on 190 of 200 triples and unanimously declined on 0, so there is no comparison
group. The annotation scheme cannot validate its own most-cited dimension.

## 5. The measurement that explains everything

Halfway through we stopped building layers and asked something that needed no
layer: **where do the two candidates actually sit relative to the anchor?**

Put the anchor at the origin. Measure the two arrows — how long they are, and
the angle between them.

| space | to chosen | to distractor | chosen is nearer | p |
|---|---|---|---|---|
| TF-IDF | 1.394 | 1.395 | 104/200 (52.0%) | 0.62 |
| MiniLM | 1.173 | 1.188 | 110/200 (55.0%) | 0.18 |
| **e5** | 0.638 | 0.647 | 117/200 (58.5%) | **0.019** |

In TF-IDF and MiniLM space the candidates are **the same distance** from the
anchor. Only e5 separates them.

One caveat that must be stated: the e5 row is not independent of what it
explains. L2 ranks by e5 similarity and picks the nearer candidate, so 58.5%
right-closer and 58.50% accuracy are the same comparison computed twice. It is
a consistency check, not an explanation.

The independent content is the other two rows, and it is the whole answer.
**The alignment layers build their cost matrices in MiniLM span space — where
distance is a coin flip.** The only space carrying signal is whole-story e5,
and decomposing a story into characters destroys it. That predicts a result
which otherwise looks strange: putting the *stronger* encoder inside the
alignment made it worse, 58.00 → 56.00. The alignment throws away exactly what
e5 is good at.

The angles agree, once compared to the right baseline. Both arrows contain
−anchor, so they are forced to point alike; the reference is not 90°. At the
observed anchor-candidate cosine the baseline is 60.9° for TF-IDF (observed
60.5°: candidates unrelated to each other) and 73.5° for MiniLM (observed
61.3°: partly shared direction). For e5 the baseline is undefined — at cosine
0.794 two mutually orthogonal candidates cannot both lie that close to the
anchor.

**What it taught, and this is the main lesson of the project:** this
measurement took about twenty minutes and it explains every null result we had
spent weeks collecting. It should have been first. *Can the signal exist in
this representation?* comes before *does my method extract it?*

## 6. Seven things that all said the same

None of these relies on an accuracy comparison clearing significance.

1. **Word overlap points the wrong way.** 45.00%, below chance, mean signed
   margin −0.0024.
2. **Filtering the cast to real characters costs accuracy.** 57.00 → 52.50, and
   the roles layer moves the same way. Whatever the alignment was matching on,
   it was not characters — removing the non-character nouns removed signal. The
   cost matrix was partly scoring setting, objects and topic words. Word
   overlap in a structural costume.
3. **No structural layer beats the embedding backbone.** The best ties it.
4. **A stronger encoder inside the alignment makes it worse.** 58.00 → 56.00.
5. **Every layer fails on the same items as TF-IDF.** Error correlation stays
   between +0.16 and +0.33 for all of them. A genuinely different
   representation should fail differently.
6. **The combination ceiling is far below independence.** Across five layers,
   at least one is right on 86.0% of items, against 98.6% expected if their
   errors were independent. Majority vote reaches 60.0%.

7. **The reversal is significant, and directional.** Cross-entropy under the
   anchor's unigram model prefers the distractor on 59.5% of triples
   (p = 0.009); the symmetric divergence sees nothing. The benchmark did not
   merely remove the lexical signal, it inverted it.

Findings 5 and 6 are one fact seen twice: the layers are not representing
different things, so combining them cannot help. Findings 1 and 7 are the same
relationship at two levels of precision — 7 names the mechanism that 1 only
hints at.

## 7. How the course maps onto the stack

Worth stating in the report and in the talk, because it shows the stack was
designed from the material rather than assembled from available libraries.

| Course topic | Where it appears |
|---|---|
| Tokenization | L0; the Jaccard/published-baseline discrepancy |
| Vector semantics, sparse | L1 tf-idf + cosine |
| Vector semantics, dense; meaning representation | L2 `e5`; the geometry measurement |
| Named entity recognition | the `actors_only` cast filter |
| Coreference resolution | L3 vs L3n is the ablation that isolates it |
| Constituency and dependency structure | L4 role extraction (dependency chosen; see below) |
| Semantic role labelling | L4 as a deliberate proxy, and why the proxy is lossy |
| Discourse and coherence | L7 components; L6 rhythm; the entity-grid reading below |
| Affect | considered, not implemented; see below |
| Ambiguity | α = 0.33; the human ceiling at 89 |
| Evaluation methodology | bootstrap, Wilson, McNemar, cross-validation, power analysis |

### The connection worth saying loudly

**L3 and L4 together build an entity grid.** Entity-based coherence models
represent a document as entities × sentences, with each cell holding the
entity's syntactic role: **S** subject, **O** object, **X** other. Now look at
what L4 uses:

```python
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}
OBJECT_DEPS  = {"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"}
```

That is the same S/O/X distinction from the same dependency relations. L3
supplies the entity dimension — which characters exist, matched across two
documents. L4 supplies the role dimension. The only missing piece is the
**sentence** dimension, the grid's columns, which is exactly what the unrun L6
would have added.

So the accurate framing is: this project repurposed an entity-based
**coherence** representation as a **similarity** representation, and it did not
transfer. That is a claim about the representation, not about our code, and it
is a stronger result. (Cite Barzilay and Lapata for the entity grid — verify
whether to use the 2005 ACL paper or the 2008 *Computational Linguistics*
article.)

### Two deliberate choices to state, not hide

**Dependency, not constituency.** Both were covered. Dependency parses give
head–dependent relations directly, so agent and patient proxies are read
straight off the relation labels; getting the same from a constituency parse
needs a head-finding step and then the same inference anyway. A choice, not an
omission — but only if you say so.

**Affect: considered, not implemented.** Elsner found sentiment-trajectory
features alone perform at roughly chance on plot-structure matching, while
Reagan et al. found emotional arcs cluster into a few shapes. Given the
geometry result, an affect layer was not expected to escape the same ceiling.
Listing it as considered-and-declined beats silence.

## 8. The statistics, decided in advance

The power analysis, run before the results came in:

```
Two strata against each other (unpaired):
  test outcomes            n= 79 vs 133   needs  17.3 pts
  dev  outcomes            n= 40 vs  71   needs  22.8 pts

Two layers on the same items (McNemar, paired):
  test, all items               4.3 - 7.6 pts
  dev,  all items               5.9 - 10.7 pts
```

Two commitments followed. The confirmatory claim has to be the overall paired
test, because per-stratum contrasts need 17 to 29 points and no plausible
effect is that big. And paired comparison beats interval overlap, because
overlapping intervals routinely hide real differences on the same items.

This is why the report can say "nothing separates" rather than "we failed to
find anything." We said in advance what would count.

## 9. What we chose

Nothing beat L2, so the system we report is L2, and the contribution is not a
system. It is:

1. A layered ablation where each comparison isolates one variable, verified
   mechanically rather than asserted.
2. A geometric account of **why** the structural hypothesis fails here, which
   generalises beyond this implementation.
3. A validity check on the three annotated dimensions, including the finding
   that the benchmark cannot validate its own most-cited dimension.
4. A characterisation of the benchmark's lexical construction artifact,
   measured and reported rather than exploited.
4. A reproducible harness: one evaluation path, run-identity hashes, a
   test-split guard, 35 tests, every number generated from the result files and
   cross-checked a second way.

A null result reported this way is worth more than a half-point win reported
carelessly.

## 10. Engineering lessons

Several of these nearly changed a conclusion.

**Silent data loss.** Two runs of one layer with different parameters wrote the
same filename; the second destroyed the first. Fixed with a parameter hash in
every result filename. *A result file must be named by everything that produced
it.*

**Circular evaluation.** L5 fitted on dev and scored on dev. Fixed with
`evaluate_cv()` and a `fits_parameters` flag. *The harness has to make the
honest path the default path.*

**A key collision that hid a finding.** `compare_runs.py` keyed on scorer plus
one flag but not the encoder, merging two L3x runs and concealing that e5
scored 56.00 against MiniLM's 58.00 — which later became central evidence. *An
aggregation key that drops a variable deletes the finding that variable
produces.*

**Space conflation.** The geometry numbers were measured in TF-IDF space and
written up as describing the embedding space, on 150 triples rather than 200.
Caught by asking "is this number actually a cosine distance?" *Every
measurement is in a specific space on a specific sample, and the claim must
match both.*

**A baseline that clipped.** The angle null produced "180°" for e5 because
`arccos` saturated an impossible value instead of reporting that the
configuration cannot exist. *A formula that silently saturates will hand you a
number you can publish.*

**A vague provenance note.** The dimension percentages were recorded as
"annotator-flag counts over 600 triples" without saying *which* count. A later
attempt to reproduce them tried three definitions, missed the right one
(majority of annotators), and briefly concluded that correct numbers were
wrong. *A source line must give the definition, not just the location.*

**Environment reality.** `torch` stops at 2.2.2 on Intel macOS and the
ecosystem has moved past it. This surfaced three times: the original install
failure, fixed with PEP 508 environment markers; an `--upgrade` that floated
numpy and transformers off pins that live in a different extra and broke the
working environment; and `transformers` refusing to `torch.load` a `.bin`
checkpoint under CVE-2025-32434 unless torch ≥ 2.6. *Pin for the machine you
have, and install extras together when the pins live in one of them.*

Three practices came out of this and are now enforced: `make audit` recomputes
every claim a second way; `report/PROVENANCE.md` sources every number and fails
on an unsourced one; `make check` forbids typing a figure into prose by hand.

## 11. Conclusion

The structural hypothesis is reasonable, well-precedented, and wrong for this
benchmark — and we can say why rather than only that.

The construction places both candidates at the same distance in every space
except whole-story e5, and the alignment methods work in span space, where
distance is a coin flip. The components annotators report using are
individually at chance, and the one they cite most cannot be validated at all.
Every layer errs on the same items, so combining them cannot rescue it.

What won the task was text-level decomposition with LLM voting — the same
decomposition we tested structurally, scored by a model rather than matched by
a cost matrix. That contrast is the most useful thing the project produced.

## 12. What would be worth doing next

1. **Score the decomposition instead of matching it.** The components are real
   text and rank in the right order; the failure is that cost-matrix alignment
   cannot use them. Asking a model to compare theme spans directly is the
   obvious test, and it is effectively what the winning system did.
2. **Add the sentence dimension.** L3 + L4 is an entity grid missing its
   columns. L6 would supply them, which makes running it a principled next step
   rather than a loose end.
3. **Measure the geometry in `story-emb` space.** It is the construction
   encoder, so the measurement would show the sampling mechanism directly. Not
   done here: it is a 7B model and the hardware was not available.
4. **Ask for a benchmark that can validate its own dimensions.** A dimension
   marked on 95% of items has no contrast group.

The method also transfers past this task. The layered ablation, the paired
testing, and above all the habit of asking whether signal *exists* in a
representation before building an extractor apply to any detection problem
where negatives are sampled for difficulty — AI-content detection and
misinformation detection among them. The specific null does not transfer; the
diagnostic does.

## 13. Status — 7 days out

Done: all development runs; geometry in three spaces; component validity;
lexical direction; power analysis; 35 tests; audit green; every reported number
generated and sourced; Sections 7.1 and 7.2 of the report drafted;
pre-registration written.

Open: **L3 and L4 with coreference.** The configs and scorers exist and
maverick is now installed, but three stacked incompatibilities have blocked the
run. If it does not complete before the freeze, the report says the coreference
variant could not be run on the available hardware, and what is reported is the
pre-coreference condition — which is also the condition Chaturvedi et al.
actually ran.

Remaining, in priority order:

1. Write 7.3–7.6. The evidence is on disk; these are the shortest subsections.
2. Write Sections 6, 4, 5 — results, system, setup. Mostly description.
3. **Fill in the freeze date and git SHA in `docs/PREREGISTRATION.md`, then the
   single held-out test run.** This cannot slip: nothing in Section 8 exists
   before it, and nothing may change after it.
4. Sections 2, 3, 9, 10; introduction and abstract last.
5. Verify the citations in `report/VERIFY.md`, or drop them.
6. Record the video walkthrough.

If something has to give, give up the coreference run, the rhythm layer and the
extra citations. Do not give up the test run or the audit.


---

# Appendices

The four documents below were separate files; they are merged here so there is
one thing to read. Two documents are deliberately **not** merged:

- `docs/PREREGISTRATION.md` — written before the test run and never edited
  afterwards. Folding it into a living document would destroy the only
  property that makes it evidence.
- `report/PROVENANCE.md` — regenerated by `make numbers`; any edit is lost on
  the next run.

---

## Appendix A — How to write Section 7

*The four-move paragraph pattern, 7.1 written out as a model, and 7.2-7.6 pre-filled.*

You are not writing an essay. You are writing six paragraphs that all have the
same four moves. Once you have the shape, each one takes about twenty minutes.

#### The four moves

Every subsection in Section 7 is this, in this order:

1. **Claim** — one sentence, stated flatly. No "we investigated whether".
2. **Why anyone would expect otherwise** — one sentence. This is what makes
   the measurement worth reporting; without it the number is trivia.
3. **How it was measured** — one or two sentences. Enough that a reader could
   redo it. Name the split and the n.
4. **What was found, and what it rules out** — the number, then the sentence
   it licenses. End on the consequence, not the figure.

That's it. 150–180 words. If a paragraph runs past 200 words, move 2 or move 4
has grown an extra idea that belongs somewhere else.

**The discipline:** every number is a macro. If you find yourself typing
`58.50`, stop and look up the macro name in `numbers.tex`.

---

#### 7.1 written out, as the model

Copy this shape five times. The prose below is a real draft — edit it, don't
admire it.

```latex
\subsection{The two candidates are equidistant from the anchor}

If the answer were recoverable from embedding distance, the correct candidate
would sit nearer to the anchor than the distractor. It does not. Across the
\DevN\ development triples the median cosine distance from anchor to correct
candidate is \GeomRightMedian\ and to the distractor \GeomWrongMedian, and the
correct candidate is the nearer of the two in \GeomRightCloser\% of triples --
a coin flip. The two displacement vectors from the anchor meet at a mean angle
of \GeomAngle\ degrees, so the candidates are not merely equidistant but
scattered in unrelated directions.

This follows from how the triples were built. Distractors were drawn from the
98th to 99th percentile of \texttt{story-emb} similarity, which places them
inside the same neighbourhood as the answer, and only triples on which two
language models disagreed survived to annotation. The surviving items are
therefore the ones where distance carries no signal, by construction. Any
method whose decision reduces to a distance in this space -- including every
alignment layer in this report, whose cost matrices are built from the same
encoders -- inherits that ceiling. This is the single measurement that most
constrains what follows.
```

Look at what that paragraph does. Move 1 is the title plus the first two
sentences. Move 2 is "if the answer were recoverable from embedding distance"
— the expectation, stated so it can fail. Move 3 is the medians and the n.
Move 4 is the last three sentences: the construction explains it, and the
consequence is that alignment inherits the ceiling.

---

#### 7.2 – 7.6, pre-filled

For each one below you have the claim and the measurement already. Write
moves 2 and 4 — two or three sentences each. That is the whole remaining job.

##### 7.2 No annotated dimension carries the decision

- **Claim.** Scoring a triple on a single annotated dimension performs at
  chance, and the annotation scheme cannot establish that the extracted
  components are the dimensions they are named after.
- **Measured, overall.** Each component extracted and scored alone with
  MiniLM on `\CompN` items. Theme `\CompThemeAcc`\% `\CompThemeCI`, action
  `\CompActionAcc`\% `\CompActionCI`, outcome `\CompOutcomeAcc`\%
  `\CompOutcomeCI`. All three intervals contain 50. Combining the three
  component-wise under cross-validation gives `\LSevenAcc`\% `\LSevenCI` --
  below every single component.
- **Measured, validity.** A component that really captures "outcome" should
  score higher on triples where annotators marked outcomes. Action is +8.7
  points in the predicted direction (p = 0.481; n = 88 marked, 19 unmarked).
  Outcome is 5.2 points **backwards** (p = 0.593; n = 40, 71). Theme cannot
  be tested at all: annotators marked it on 190 of 200 triples, so no
  contrast group exists.
- **Move 2 to write.** The annotators were asked to decide along these three
  axes and cited them constantly, so a system scoring one axis should beat
  chance at least on the items where that axis was cited.
- **Move 4 to write.** Three sentences, in this order.
  1. The overall result settles it: no dimension beats chance, and combining
     them is worse than any one alone.
  2. The validity test cannot adjudicate. With 19 and 40 items in the smaller
     strata the detectable difference is roughly 20 to 27 points, so +8.7 and
     -5.2 are both inside noise. Report them as inconclusive, not as one
     success and one failure.
  3. The theme case is a finding about the task, not about this system: the
     annotation scheme cannot validate its own most-cited dimension, because
     a dimension marked on 95\% of items has nothing to contrast against.
     That is worth saying plainly -- it is a limitation of the benchmark that
     a reader will not have noticed.

> **Do not claim the rank order means anything.** The three components land at
> 55.0, 54.5 and 54.0 -- a one-point spread, which is two items out of 200.
> An earlier reading of this project treated the rank order as matching the
> annotator citation frequencies and took that as evidence the decomposition
> was real. At this spread it is evidence of nothing, and a reader who checks
> will notice. Drop the argument.

##### 7.3 Lexical overlap points the wrong way by construction

- **Claim.** Vocabulary overlap is not merely uninformative here; it is
  anti-correlated with the answer.
- **Measured.** Jaccard overlap scores `\JaccardAcc`\% on dev, below chance.
  Mean signed margin `\JaccardMargin`: averaged over all triples, the
  distractor shares slightly *more* vocabulary with the anchor than the
  answer does.
- **Move 2 to write.** Lexical overlap is the first thing anyone tries and it
  is the published baseline, so a reader expects it near or just above chance.
  Below chance needs explaining.
- **Move 4 to write.** The signed margin shows this is systematic rather than
  noise — the sign is consistent, not a near-50 result that happened to land
  low. Connect to the sampling: distractors were selected *for* proximity in
  an encoder trained on retellings, which rewards shared surface. Then the
  consequence: a system cannot be rescued by adding lexical features, and any
  layer that quietly reintroduces them will be dragged down.

##### 7.4 Filtering the cast to real characters costs accuracy

- **Claim.** Restricting the cast to entities that look like characters makes
  the alignment worse, not better.
- **Measured.** Same layer, one flag changed. All nouns `\LThreeNAcc`\%,
  actors only `\LThreeNActorsAcc`\%, a drop of
  `\McnLThreeNVsLThreeNActorsDiff` points, p = `\McnLThreeNVsLThreeNActorsP`
  over `\McnLThreeNVsLThreeNActorsDisc` discordant pairs. The roles layer
  moves the same direction: `\LFourNAcc`\% to `\LFourNActorsAcc`\%.
- **Move 2 to write.** Cleaning the cast is the obvious improvement — it is
  what you would do to reduce noise, and Chaturvedi et al. attributed error to
  exactly the entities a filter removes. Both layers moving the same way makes
  it harder to dismiss as one bad run.
- **Move 4 to write.** The honest reading: whatever the alignment was matching
  on, it was not characters specifically. Removing the non-character nouns
  removed signal, so the cost matrix was partly scoring setting, objects and
  topic words — lexical overlap wearing a structural costume. Note the p-value
  is not significant and say so; the direction's consistency across two layers
  is the argument, not the margin.

##### 7.5 The structural layers fail on the same items as the lexical one

- **Claim.** No structural layer errs independently of TF-IDF.
- **Measured.** Correlation between error vectors, every layer against the
  lexical baseline on the same `\DevN` items: `\PhiMin` to `\PhiMax` across
  all of them. Embeddings are the least correlated at `\PhiLTwo`; the
  structural layers sit higher.
- **Move 2 to write.** A genuinely different representation should fail on
  different items. That is the whole premise of adding a layer — and of
  combining layers afterwards.
- **Move 4 to write.** Same items hard for all of them means the layers are
  not representing different things. Then the forward link: this is why the
  combination layer in 7.6 cannot help, and the two subsections should be read
  together.

##### 7.6 The combination ceiling is far below independence

- **Claim.** Even a perfect oracle over all the layers would fall well short
  of what independent errors would allow.
- **Measured.** Over the `\PoolSize` layers in `\PoolNames`: at least one is
  correct on `\OracleUnion`\% of items, against `\OracleIndep`\% expected if
  their errors were independent. Majority vote reaches `\MajorityVote`\%. Mean
  cross-validated weights `\LFiveXWeights` over `\LFiveXFolds` folds.
- **Move 2 to write.** Combining diverse systems is the standard way to buy a
  few points, and the independence figure is what you would get for free if
  the layers were as different as their descriptions suggest.
- **Move 4 to write.** The gap between `\OracleUnion` and `\OracleIndep` is
  the cost of the correlation measured in 7.5, in accuracy. State one caveat
  honestly: the union is a choice of pool, named in `\PoolNames`, and a
  different pool gives a different ceiling. Then close the section — you have
  now shown the plateau is a property of the data, not of this code.

---

#### The argument as one spine

You asked about mindmapping. A map is the wrong shape for this, because the
report is not a tree of topics — it is a single chain, and every link is
measured. Write the chain out and you have your argument:

```
  Surface overlap fails           ->  7.3  below chance, and systematically so
            |
  So structure is the hypothesis  ->  Sections 1 and 3
            |
  Built five layers of it         ->  Section 4
            |
  None separates from any other   ->  Section 6  (nothing significant)
            |
  Is that low power, or absence?  ->  Section 5 said 4.3-6.2 points in advance
            |
            +-- the embedding space has no answer in it      ->  7.1
            +-- no single dimension carries it either         ->  7.2
            +-- the "structural" layers were partly lexical   ->  7.4
            +-- they all fail on the same items               ->  7.5
            +-- so combining them cannot rescue it            ->  7.6
            |
  The plateau is in the data       ->  Section 10: what would actually work
```

Read down that chain. Each arrow is a claim you can defend with a macro. If a
sentence you write does not advance one of those arrows, cut it.

Tape this chain above your desk and write one box per sitting.

---

#### Getting unstuck, concretely

The blank page is a symptom of trying to write and decide at the same time. So
don't decide anything while writing:

1. **Write move 3 first, for all six.** Pure transcription — which macro,
   which split, which n. No judgment required. Half an hour, and Section 7 is
   a third drafted.
2. **Then move 1 for all six.** The claim sentences are already written above;
   rephrase each in your own voice.
3. **Then move 2 for all six.** This is the only part that needs thinking, and
   by now you have momentum and the surrounding sentences to push against.
4. **Then move 4.** Easiest last, because by then you can see what each
   paragraph has earned the right to say.

Six paragraphs × four moves = 24 small writing tasks, none of which requires a
decision you have not already made. You decided all of it in October when you
ran the experiments.

Build after each pass: `make numbers && make report`. Seeing a real number
land in a real sentence is the thing that makes the next paragraph easier.

---

## Appendix B — Report outline

*Section by section: the claim, the evidence, the macros, the length, and the order to write in.*

Due 25 November. This file is the plan; `report.tex` is the thing you type into.
Each section below says what it must establish, what evidence already exists,
and roughly how long it should be. Nothing here asks you to run anything new.

**The thesis, in one sentence.** Narrative structure is the natural hypothesis
once surface overlap is shown to fail, and it does not help on this data — and
six independent measurements say why, which makes the negative result a
finding rather than a failure.

Keep that sentence in view. Any paragraph that does not serve it comes out.

> ### One correction before you start
>
> The generated paired tests do not support "the only significant difference
> is lexical to embeddings". **Nothing is significant on dev.** The
> words-to-meaning gap is `\McnTfidfVsLTwoDiff` points at
> p = `\McnTfidfVsLTwoP`, from `\McnTfidfVsLTwoDisc` discordant pairs. The
> best structural layer is `\McnLTwoVsLThreeXDiff` from the backbone at
> p = `\McnLTwoVsLThreeXP`.
>
> This is better for the report, not worse. The argument stops being "our
> layer lost narrowly" and becomes "accuracy cannot settle this at n = 200,
> as Section 5 predicted in advance, so Section 7 settles it another way."
> Section 7 is where your evidence actually is.
>
> Second correction: the oracle-union figure depends on which layers are
> pooled. `\OracleUnion` is over `\PoolSize` layers, named in `\PoolNames`.
> Any union figure computed earlier over three layers is a different
> quantity — do not cite both.

---

#### Rules that keep the writing honest

1. **No number typed by hand.** Every figure is a macro from `numbers.tex`.
   Run `make check` before you commit; it lists any decimal you typed into a
   sentence.
2. **Two combination numbers are not comparable.** `\LFiveXAcc` is
   cross-validated. `\LFiveNAcc` is fitted and scored on the same split. The
   second one belongs in Limitations as a demonstration of what in-sample
   fitting buys, never in the results table as a best score.
3. **Claim only what a paired test supports.** Overlapping intervals hide real
   differences and separated intervals invent them. The macro for every
   comparison you make is `\Mcn<A>Vs<B>P`.
4. **Write the test section last**, after the single frozen run, and change no
   configuration after it.

---

#### Section by section

##### 1. Introduction — 500 words

Must establish: the task; why structure is the obvious next hypothesis once
lexical overlap is shown to be below chance; what was built; what happened.

Evidence in hand: `\JaccardAcc` (below chance), `\LTwoAcc` (the backbone),
`\LThreeXAcc` (best structural layer), `\McnLTwoVsLThreeXP`.

End with a numbered contributions list. Four items:
a layered ablation with one evaluation path; a geometric account of why the
distance signal is absent; a validity check on the three annotated dimensions;
and a reproducible harness with a frozen test protocol.

##### 2. Task and Data — 450 words

The construction pipeline comes before anything else, because it explains
every later result. Reuse the paragraph from the proposal (Section 1 of
`references/proposal.tex`) — it is already checked against the overview paper.

Then the annotation: three dimensions, how often each was cited
(`\AnnThemePct` / `\AnnActionPct` / `\AnnOutcomePct`), and the agreement
figures (`\AlphaOverall`, `\AlphaPerDimLo`–`\AlphaPerDimHi`) with what they
cap.

##### 3. Related Work — 400 words

Group by implication, not by paper. Three groups:

- **Characters helped elsewhere.** Chaturvedi et al. on movie remakes
  (`\ChaturvediBefore` → `\ChaturvediAfter`); Barancová and Sileno on *Tell Me
  Again* (`\BarancovaGain`% relative).
- **Structure did not help here.** StoryNet's symbolic graphs at
  `\StoryNetAcc`, rank `\StoryNetRank` of `\OfficialRankedA`; COGNAC
  abandoning Propp's functions three times; narrative-chain embeddings that
  win narrative cloze and lose story similarity.
- **What did win.** Text-level decomposition plus LLM voting, at
  `\OfficialBestA`. Note that this is the same decomposition this project
  tested structurally — that parallel is the hinge of the conclusion.

##### 4. System — 700 words

One short paragraph per layer, each saying what it adds **and what would have
to be true for it to help**. That second clause is what makes the ablation
legible.

Then the harness, in its own subsection: one evaluation path, intervals,
paired tests, cross-validation wherever parameters are fitted, a run-identity
hash in every filename, a guard that refuses the test split. Mention the two
bugs this caught — results silently overwriting each other, and fitting on the
evaluation split, which cost TF-IDF a full point once corrected. A harness
that caught real mistakes is worth more to a reader than one that merely
exists.

Figure: `system_diagram.svg` (convert to PDF once, see the note in
`report.tex`).

##### 5. Experimental Setup — 350 words

Splits, encoders, seeds, hardware. Then the power analysis: paired MDE of
`\PowerPairedLo`–`\PowerPairedHi` points on the test split, and
`\PowerStrataMDE` points for the between-strata contrast on
`\PowerStrataSmall` and `\PowerStrataLarge` items.

**Put the power analysis here, before the results.** Stated afterwards it
reads as an excuse; stated here it is a pre-registered constraint.

##### 6. Results — 600 words + table

The table is already wired in `report.tex`. Do not narrate it row by row.

Three paragraphs: the largest gap and its paired test (`\McnTfidfVsLTwoDiff`
points, p = `\McnTfidfVsLTwoP`, `\McnTfidfVsLTwoDisc` discordant pairs — say
"suggestive", not "significant"); the plateau (`\McnLTwoVsLThreeXDiff` at
p = `\McnLTwoVsLThreeXP`, with `\McnLTwoVsLThreeXB` items won and
`\McnLTwoVsLThreeXC` lost — the layers disagree on a third of the items and
finish level, which is more informative than the margin); and baseline
reproduction (`\JaccardTestAcc` against `\PublishedJaccardTest` published,
with the tokenization gap explained).

End the section by handing off to Section 7 explicitly: accuracy cannot
adjudicate at this sample size, so the rest of the report does not ask it to.

Figure: `ablation_dev.png`.

##### 7. Why Structure Did Not Help — 1000 words

**This is the section the report is for.** Six subsections, ~170 words each,
each ending in the single sentence it licenses.

| # | Measurement | Macros |
|---|---|---|
| 1 | The candidates are equidistant from the anchor | `\GeomRightMedian`, `\GeomWrongMedian`, `\GeomRightCloser`, `\GeomAngle` |
| 2 | Each annotated dimension is individually at chance | `\CompThemeAcc`+CI, `\CompActionAcc`+CI, `\CompOutcomeAcc`+CI, `\CompN` |
| 3 | Lexical overlap points the wrong way by construction | `\JaccardAcc`, `\JaccardMargin` |
| 4 | Filtering the cast to real characters costs accuracy | `\LThreeNAcc` → `\LThreeNActorsAcc`, `\McnLThreeNVsLThreeNActorsP` |
| 5 | Structural layers fail on the same items as the lexical one | `\PhiMin`–`\PhiMax` |
| 6 | The combination ceiling is far below independence | `\OracleUnion` vs `\OracleIndep`, `\MajorityVote`, `\LFiveXWeights` |

Lead with (1); it is the strongest and the easiest to picture.

On (2), make the point that matters: all three intervals contain 50, **and**
the rank order of the three components matches the annotator citation order
exactly. That combination says the decomposition is real but too weak to
exploit by alignment — which is a sharper claim than "it didn't work".

Figures: `triple_geometry.png`, and `strata_outcomes_dev.png` if you use the
strata argument.

##### 8. Held-out Test Result — 250 words

Write nothing here until the frozen run. Freeze the configuration, run once,
report accuracy with an interval and the one paired comparison you
pre-committed to.

##### 9. Limitations — 350 words

Short declarative sentences, no hedging ritual. The real ones:
sample size and what it hides; dependency-derived roles as a proxy for
PropBank SRL; story overlap across splits; one encoder family and English
only; and `\LFiveNAcc` as the in-sample figure, labelled as such.

##### 10. Conclusion — 200 words

What the next person should do differently. The honest answer: the winning
system decomposed at the text level and voted with a language model, and the
components measured here rank in the right order but too weakly for alignment
to exploit. Say what would test that — the same decomposition scored by a
model rather than matched by a cost matrix.

---

#### Optional runs, ranked by what they add to the writing

You can write the whole report without these. In order of value:

1. **`l5_combined` over L3n, cross-validated.** Makes the two combination
   numbers comparable and removes the asterisk from the results table.
   Smallest change, biggest gain in cleanliness.
2. **The coreference layers (`l3_characters`, `l4_roles`).** L3 − L3n is the
   contribution of coreference, which the proposal promised and the report
   currently cannot report.
3. **`make rhythm`.** Built, never run. A null result here costs two sentences
   and closes the sequential-structure question a reader will ask.
4. **`make components`.** Turns the fitted weight vector into a concrete
   artifact for Section 7.6.

---

#### Order to write in

Section 7 first — it is the content, and everything else is framing for it.
Then 6, 4, 5, 2, 3, 9, 10, 1, abstract. The introduction and abstract are
written last because they summarise claims that must already exist.

---

## Appendix C — The course, mapped onto the stack

*Full version of the mapping summarised in Part 7, with sentences you can lift.*

Each layer of the ablation instantiates something taught in CS 421. Saying so
explicitly is worth doing in the report: it shows the stack was designed from
the material rather than assembled from whatever libraries were to hand, and
it gives every null result a named technique to attach to.

Chapter numbers in Jurafsky & Martin shift between releases — check them
against your edition before citing. Topic names below are stable.

---

#### Direct correspondences

| Layer | Course topic | What the layer actually uses |
|---|---|---|
| L0 Jaccard | Tokenization; lexical overlap | Set intersection over word types; the whole point is that it is pre-semantic |
| L1 TF-IDF | **Vector semantics**, sparse representations | tf-idf weighting, cosine similarity, L2 normalization |
| L2 `e5` | **Vector semantics**, dense representations; **meaning representation** | Transformer encoder, contrastive training objective, whole-sequence pooling |
| L3n/L3x | **Named entity recognition**; **entity-based coherence** | Noun-phrase extraction, `actors_only` filter as an NER-style entity-type restriction, Hungarian assignment over a similarity matrix |
| L3 (planned) | **Coreference resolution** | `maverick-coref`, singleton-keeping LitBank model |
| L4n/L4x | **Dependency structure**; **semantic role labelling** | `nsubj/nsubjpass/csubj/agent` as agent proxy, `dobj/obj/iobj/pobj/dative/attr/oprd` as patient proxy |
| L7 components | **Discourse structure** | Positional segmentation: theme = whole text, action = verb-dense sentences, outcome = closing quarter |
| L6 rhythm (built, unrun) | **Discourse**; sequence alignment | Positional and elastic (DTW) segment comparison |
| Evaluation throughout | **Ambiguity**; evaluation methodology | Krippendorff's α of 0.33; bootstrap and Wilson intervals; McNemar |

---

#### The connection worth making loudly

**L3 and L4 are entity-grid coherence machinery pointed at a similarity task.**

Entity-based coherence models — the entity grid — represent a document as a
matrix of entities against sentences, where each cell records the entity's
syntactic role in that sentence: **S** for subject, **O** for object, **X** for
other, and absent. Coherent documents have characteristic transition patterns
in that grid.

Now look at what L4 builds. From `src/task4/scorers/roles.py`:

```python
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}
OBJECT_DEPS  = {"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"}
```

That is the S/O/X distinction of the entity grid, built from exactly the same
dependency relations. L3 supplies the entity dimension (which characters exist,
aligned across two documents); L4 supplies the role dimension. The one thing
missing is the **sentence** dimension — the grid's columns — which is precisely
what the unrun L6 rhythm layer would have added.

So the honest and more sophisticated framing is: this project repurposed an
entity-based coherence representation as a similarity representation, and it
did not transfer. That is a stronger claim than "our character features did
not help," because it says something about the representation rather than about
the implementation. Cite Barzilay and Lapata for the entity grid
(**verify the venue** — there is a 2005 ACL paper and a 2008 *Computational
Linguistics* article; pick one deliberately).

---

#### The distinction your L4 result illustrates

This is the best teaching point in the project, and it belongs in Section 7.4.

Your agent/patient features come from **syntactic dependency relations**, not
from semantic roles. Those are not the same thing, and the course covers
exactly why:

> *John broke the window.* → `window` is **dobj**
> *The window broke.* → `window` is **nsubj**

The window is the **patient** in both sentences. A dependency-based proxy calls
it an object in one and an agent in the other. Every unaccusative verb,
passive without a by-phrase, and middle construction in your corpus produces
this error.

That is the standard motivation for semantic role labelling existing as a task
separate from parsing — and it is a concrete, named mechanism for why L4 added
nothing. You already have the practical reason you used a proxy (AllenNLP
archived in 2022; LLM-prompted SRL at 48.6% argument match against 88.3% for
humans). Now you also have the linguistic reason it was a lossy proxy.

Pair it with the measured finding: filtering the cast to real characters *cost*
4.5 points, which says the cost matrix was partly scoring non-character nouns.
Two independent accounts of the same failure — one linguistic, one empirical.

---

#### What you used, and what you deliberately did not

**Dependency, not constituency.** Both were covered; you chose dependency. Say
why in one sentence: constituency parses give phrase structure (NP, VP),
whereas dependency parses give head–dependent relations directly, and
agent/patient proxies are read straight off the relation labels. Extracting the
same information from a constituency parse requires a head-finding step and
then the same relation inference anyway. This is a defensible choice, not an
omission — but it only reads that way if you state it.

**Affect: considered, not implemented.** Worth one honest sentence, because it
has a strong precedent on both sides. Elsner found that sentiment-trajectory
features alone gave essentially chance performance on plot-structure matching,
while Reagan et al. found emotional arcs cluster into a small number of shapes.
Given the geometry result — distance at chance in every space the layers use —
an affect layer was not expected to escape the same ceiling. Listing it as a
considered-and-declined option is better than silence, and it is a legitimate
next step.

**Ambiguity, as the task's premise rather than a layer.** The α of 0.33 is
annotator disagreement about what "similar" means — ambiguity at the level of
the judgment, not the word. Worth naming as such: the human upper bound of
~89% exists *because* the construct is ambiguous, and that is why 78% is a
strong result rather than a mediocre one.

---

#### Sentences you can lift

For the introduction:

> The ablation is organised around representations covered in the course: lexical
> overlap, sparse and dense vector semantics, coreference-derived entity sets,
> dependency-derived role profiles, and positional discourse segmentation. Each
> layer adds one representation, so a null result identifies which
> representation failed to carry signal.

For Section 4, introducing L3 and L4:

> Together these two layers assemble an entity-by-role representation closely
> related to the entity grid used in work on local coherence, with entities
> supplied by cast alignment and roles by dependency relations. The sentence
> dimension of the grid is omitted; the sequential layer that would have
> supplied it was implemented but not run, for reasons given in
> Section~\ref{sec:analysis}.

For Section 7.4, after the 4.5-point drop:

> The role features are derived from syntactic dependency relations rather than
> semantic roles, and the two diverge systematically: an unaccusative or
> passive construction assigns the patient a subject relation, so the proxy
> labels it an agent. This is the standard motivation for semantic role
> labelling as a task distinct from parsing, and it gives a linguistic account
> of the failure that the ablation measures empirically.

For Section 9, Limitations:

> Constituency parses were not used; dependency relations were preferred because
> agent and patient proxies are read directly off relation labels. Affect-based
> representations were considered and not implemented, on the grounds that
> sentiment-trajectory features alone perform at chance in comparable work and
> that the geometric result of Section~\ref{sec:analysis} predicts the same
> ceiling.

---

## Appendix D — Citations and claims to verify

*Nothing here may enter the bibliography or the prose until it is checked.*

The proposal nearly went out with a wrong rank for StoryNet ("43rd of 47"
instead of 39th of 44). That was caught by reading the overview PDF. The same
discipline applies here: an entry goes into `custom.bib` only once its record
has been checked, and until then it lives in this file.

`report/custom_additions.bib` holds the entries that *are* checked.

#### Needs checking

| Key I would use | What I believe it is | What to confirm |
|---|---|---|
| `fisseni2012` | Fisseni and Löwe, on which dimensions of narrative matter to human judgments of story equivalence; Computational Models of Narrative workshop, 2012 | Exact title, venue, year, pages. The finding worth citing is that readers judge temporally reversed stories as the same story — confirm that claim is in this paper and not a companion one. |
| `vayer2019` | Vayer et al., fused Gromov–Wasserstein distance for structured objects | Which publication to cite: the ICML 2019 paper on optimal transport for structured data, or the 2020 *Algorithms* journal version. Pick one and match the page numbers. |
| `opitz` | Opitz, on decomposable metrics for graph-structured meaning representations | Which paper. There are several candidates and I am not confident which one supports the point. Skip it unless the point genuinely needs it. |
| `glavas2014` | Glavaš and Šnajder, event graphs for retrieval and multi-document summarisation | Journal, volume, year. This was dropped from the proposal as uncited; only add it back if Section 3 actually uses it. |
| `cascademind` | CascadeMind, SemEval-2026 Task 4 system paper | ACL Anthology ID, authors, pages, reported accuracy. |
| `ttda704` | ttda704, SemEval-2026 Task 4 system paper | Same. |

#### Figures to re-check against the source

- `ElsnerSentimentOnly` in `facts.json` is marked VERIFY. Confirm the exact
  accuracy for the sentiment-trajectory-only condition and the page it is on
  before using it. If it cannot be pinned down, cite the paper for the
  qualitative point and drop the number.
- `OfficialMeanA` is computed here from the ranked systems table rather than
  quoted. Keep the word "computed" in the sentence that uses it.

#### How to check

The overview and system papers are in the ACL Anthology under
`2026.semeval-1.*`. Open the PDF and read the table, do not trust a search
snippet. When a figure is confirmed, move the entry into
`custom_additions.bib` and delete its row here.

#### Added 17 Nov — entity grid

| Key I would use | What I believe it is | What to confirm |
|---|---|---|
| `barzilay2008` | Barzilay and Lapata, the entity-grid model of local coherence — a document as an entity-by-sentence matrix with S/O/X syntactic roles | There are two candidates: an ACL 2005 paper and a 2008 *Computational Linguistics* article (vol. 34). Pick one deliberately and match volume/pages. This supports the claim in `docs/LECTURE_MAP.md` that L3+L4 assemble an entity-grid-like representation, so it needs to be right before that framing goes in Section 4. |

#### Added 17 Nov — the annotation scheme itself

**What the three dimension flags actually mean is not documented in the data
release.** The README describes `anchor_text`, `text_a`, `text_b`,
`text_a_is_closer` and the Wikidata detail fields, and says nothing about
`abstract_theme`, `course_of_actions` or `outcomes`.

Everything this project says about them is inferred from the field names. That
is fine for computing strata, and not fine for a sentence in the report
characterising what annotators were asked.

To check, in the Hatzel et al. 2026 task overview (you have the PDF):

1. Were annotators asked which dimensions **drove** the decision, or which
   dimensions the stories were **similar along**? These are different claims
   and the report currently implies the first.
2. Were the three presented as independent checkboxes, as the data shape
   suggests, or ranked?
3. Is the per-dimension Krippendorff alpha (0.05-0.11) computed over these
   same flags?

Until confirmed, use: "annotators recorded, for each of three dimensions,
whether that dimension applied to the similarity they judged."

#### Compliance — the canary string

The dataset README requires that the canary GUID accompany any reproduction of
samples **with their labels**. If the report, the notebooks or the slides show
an example triple together with its gold label, the canary belongs in a
footnote or appendix. Check before submitting.
