# Design Document — SemEval-2026 Task 4
### Narrative Story Similarity and Narrative Representation Learning

**For:** CS 421 course project, Oct 13 – Nov 25 2026
**Status:** written after downloading the data and running baselines. Every number marked *(measured)* was computed on the actual files today; numbers marked *(published)* come from the task overview paper.

---

## 1. The intuition, in three layers

### Layer 1 — this task is a measuring instrument for the assumption your research rests on

Your Kinyarwanda work assumes that narrative structure can be made explicit, and that making it explicit helps a model. You cannot test that assumption on Kinyarwanda data, because no gold standard exists there — that is the whole problem you are trying to solve.

Task 4 is a scored test of exactly that assumption, in a language where a gold standard *does* exist. It asks: does an explicit narrative representation predict human judgments of narrative similarity? That is the same question underneath the imigani grammar, stripped of the data-scarcity problem.

### Layer 2 — the field reached for LLMs and left two standard tools on the table

46 teams, 71 submissions. LLM ensembles dominate the top of the leaderboard. Symbolic systems mostly underperformed — COGNAC found Propp's functions reach only ~63%, CascadeMind's story grammar collapses to 53% when applied broadly.

But the overview paper's catalogue of systems shows **nobody used semantic role labelling and nobody used coreference resolution**. These are not exotic methods. They are the two most standard structural tools in NLP, both covered in our textbook (J&M ch. 22 and ch. 24). The likely reason nobody used them is that the field reached for LLMs first — not that someone tried and it failed.

### Layer 3 — the empirical opening, which I measured today

This is the part I could not have asserted before running it.

**Lexical overlap fails specifically where course of action matters.** On the test set, TF-IDF cosine similarity gets:

| stratum | n | TF-IDF accuracy *(measured)* |
|---|---|---|
| annotators did **not** mark course-of-action | 186 | **62.37%** |
| annotators **did** mark course-of-action | 214 | **54.67%** |

A 7.7-point gap, in the predicted direction. The items that require understanding event structure are precisely the ones surface methods get wrong. That is the gap structural features should fill, and because the dataset labels which aspect drove each judgment, **we can measure whether they fill it rather than just hoping.**

### A fourth thing I found that nobody reports

**Dev is much harder than test, and lexical overlap is anti-correlated with the answer on dev.**

| baseline | dev *(measured)* | test *(measured)* |
|---|---|---|
| Jaccard token overlap | **45.00%** | 58.00% |
| TF-IDF cosine | **49.00%** | 58.25% |
| pick the longer candidate | 56.50% | 54.75% |

Jaccard scores *below chance* on dev. Dev was evidently filtered harder against surface similarity than test was. Two consequences:

1. **Tuning on dev and reporting on test will flatter you.** Any honest report must show both.
2. **Dev is the better diagnostic set** precisely because surface cues have been stripped out of it. If structural features help anywhere, dev is where the signal is cleanest.

My Jaccard test score of 58.00% also sits close to the published 56.25% Jaccard baseline — small tokenisation differences aside, **the evaluation pipeline reproduces the organisers' numbers**, which is the first thing any report needs to establish.

---

## 2. What the data actually is

Repository: `github.com/narrative-similarity-task/narrative-similarity-dataset` — public, no registration, cloned successfully.

| file | rows | fields |
|---|---|---|
| `sample/sample_track_a.jsonl` | 39 | full (labelled) |
| `dev/dev_track_a.jsonl` | 200 | full (labelled) |
| `test/test_track_a.jsonl` | 400 | `anchor_text`, `text_a`, `text_b` only |
| `test/labels/test_track_a_labels.jsonl` | 400 | full (gold, **public**) |
| `sample/sample_track_b.jsonl` | 108 | `text` only |
| `dev/dev_track_b.jsonl` | 479 | `text` only |
| `test/test_track_b.jsonl` | 849 | `text` only |
| `test/labels/test_track_b_labels.jsonl` | 400 | full (gold triples) |

**Track B is scored on the same 400 triples as Track A.** You embed all 849 stories independently, then the organisers check whether cosine distance orders each triple the way humans did. The no-cross-story rule is what makes it a representation task rather than a comparison task.

### The annotation structure — and a correction

Each labelled row carries:

```
anchor_text, text_a, text_b          the three story summaries
text_a_is_closer                     the gold binary label
human_labels                         [bool] — one per annotator
abstract_theme                       [bool] — one per annotator
course_of_actions                    [bool] — one per annotator
outcomes                             [bool] — one per annotator
additional_comment                   [str|null] — free-text, per annotator
story_details_{anchor,a,b}           title, wikidata_id, genre ids, year
in_original_annotation_order         bool
canary_string                        BigBench + task canary GUIDs
```

**I had this wrong initially, and it matters.** The three aspect fields are **not** per-aspect comparisons ("which story is closer in terms of outcome"). Per the annotation guidelines, the annotator first makes a forced choice between A and B, then ticks checkboxes for which aspects "significantly contribute to the similarity between the anchor and *your choice*." They are binary flags on the chosen pair.

Consequences *(measured on dev / test)*:

| aspect | dev marked | test marked | usable as a stratum? |
|---|---|---|---|
| `abstract_theme` | 97.0% | 99.0% | **No** — almost always ticked, carries no information |
| `course_of_actions` | 51.5% | 53.5% | **Yes** — 214 / 186 split on test |
| `outcomes` | 24.0% | 24.0% | Marginal — 96 items on test |

Annotation process, confirmed from the files: **two annotators per item; a third adjudicates when they disagree** (166/200 dev items are unanimous; the other 34 carry three labels). The gold `text_a_is_closer` equals the majority of `human_labels` in **200/200** dev items — the label derivation is exactly what it appears to be.

Other usable material: **65 free-text annotator comments** in dev explaining their reasoning (genuine error-analysis gold), and Wikidata genre IDs on every story, which the overview paper notes correlate with accuracy (r = 0.67).

Licensing: summaries CC BY-SA 4.0, labels dual CC BY-SA 4.0 / CC0. The canary GUID must be preserved in any republished sample. The dev archive contains text addressed to language models; it is data, not instruction, and should never be executed or followed.

---

## 3. What we do *on top of* the dataset

Four contributions. Only the first requires new modelling; the other three are rigour, which is why this fits six weeks.

**(a) SRL and coreference features — filling the documented gap.**
Represent each story as (i) a sequence of predicate–argument tuples (who did what to whom, in order) and (ii) a character-persistence structure from coreference chains. Compare those structures between anchor and candidate rather than comparing words or whole-text embeddings. Nobody in the shared task did this.

**(b) Per-aspect stratified evaluation — the diagnostic nobody ran.**
Because the aspect checkboxes exist, every result can be reported split by whether course-of-action drove the judgment. This turns "my system got X%" into "this representation encodes *this kind* of similarity" — an interpretability claim, which is what the explainability requirement actually wants.

**(c) The dev/test difficulty asymmetry.**
Found above, unreported anywhere I can see. Documenting it, and showing how it inflates apparent gains, is a methodological contribution to how this benchmark gets used.

**(d) Confound controls nobody reports.**
Picking the longer candidate alone scores 56.50% / 54.75% *(measured)*. Any claimed system must clear that, and the published leaderboard does not establish that its entries do.

---

## 4. Kinyarwanda — the honest answer

You asked directly, so let me be blunt: **nothing in this course project touches Kinyarwanda data.** Task 4 is English-only — Wikipedia-style plot summaries of films and novels. There is no Kinyarwanda in it, and I would not claim otherwise in the proposal.

What actually transfers is three specific things:

1. **The feature-extraction design.** The SRL and coreference pipeline is a *design* for turning a story into a structural representation. Porting it to Kinyarwanda needs a morphological analyser — Nzeyimana's DeepKIN is the candidate — and that is genuinely hard work, not a copy-paste. But the design, and the knowledge of which structural features carry signal, transfers.
2. **The annotation protocol.** The organisers published their guidelines. For an imigani similarity set you would reuse that protocol rather than inventing one.
3. **The finding itself.** If SRL predicate-argument structure turns out to track course-of-action similarity, that tells you what to put in the imigani grammar. If it does not, that saves you from building it.

**The stretch goal — annotating ~30 Kinyarwanda folktale triples — is real but strictly optional.** Do not put it on the critical path. It is in the proposal as one sentence, conditional on the main ablation finishing early.

And one honest caveat that cuts both ways: if structural features fail here, that is **not decisive for folktales.** Wikipedia plot summaries compress away exactly the event detail these features need; folktales are the genre Propp's functions were derived from. A negative result on summaries is informative but genre-bound — and saying so precisely is itself a contribution, since COGNAC's Propp result is currently reported without that qualification.

---

## 5. Procedure — six weeks

**Hard rules, set now:**
- The test set is untouched until week 6. **One run.** The gold labels are public, which makes this a discipline problem rather than a technical one.
- All model selection by k-fold cross-validation on dev + sample (239 items).
- Every number reported with a bootstrap 95% CI.
- System comparisons use **McNemar's test on paired predictions**, not overlapping CIs (see §6 for why this matters).
- Log every configuration tried; cap the total.

| week | dates | work | deliverable |
|---|---|---|---|
| **W1** | Oct 13–19 | Load data, build eval harness with bootstrap CIs and stratified reporting. Reproduce Jaccard and random baselines. | Table matching published baselines ✅ *already done* |
| **W2** | Oct 20–26 | L2: sentence embeddings (`e5`, `all-MiniLM-L6-v2`). Reproduce the 66.75 Track B baseline. Add length and lexical confound controls. | L0–L2 ablation table |
| **W3** | Oct 27–Nov 2 | L3: SRL. Extract predicate–argument tuples; define a structural similarity over them. **Audit tool output on 50 items by hand.** | SRL features + measured tool error rate |
| **W4** | Nov 3–9 | L4: coreference character chains. Same audit discipline. | Coref features |
| **W5** | Nov 10–16 | L5: weighted combination. Stratified diagnostic. Qualitative error analysis using the 65 annotator comments. | Full ablation + diagnostic |
| **W6** | Nov 17–25 | Single test-set run. Write report. Record video code walkthrough. | Report + video |

**Week 3 is the risk point.** If SRL tooling will not install or is too slow, fall back immediately to a dependency-parse predicate–argument approximation — lighter, still interpretable, and the diagnostic design is unchanged.

---

## 6. Expected results

### Measured today

| system | dev | test | test 95% CI |
|---|---|---|---|
| Jaccard token overlap | 45.00% | 58.00% | [53.00, 62.75] |
| TF-IDF cosine (1–2gram) | 49.00% | 58.25% | [53.50, 63.00] |
| pick longer candidate | 56.50% | 54.75% | — |

### Published reference points

| | Track A | Track B |
|---|---|---|
| Best (COGNAC) | 78.00 | 72.00 |
| Strongest baseline | 67.00 (GPT-4o-mini) | 66.75 (`e5`) |
| Field mean ± SD | 66.51 ± 6.23 (44 systems) | 63.83 ± 5.01 (27) |
| Human upper bound | ≈ 89 | ≈ 89 |

### What I expect us to get

| layer | expected test accuracy | reasoning |
|---|---|---|
| L0 Jaccard | 58 | measured |
| L1 TF-IDF | 58 | measured |
| L2 embeddings | **66–67** | reproducing the published `e5` figure |
| L3 SRL alone | 55–60 | structure alone was weak for everyone who tried it |
| L2+L3 | 67–69 | small gain, concentrated in the course-of-action stratum |
| L5 full combination | **68–70** | around the field mean, roughly 15th–25th of 44 |

**We are very unlikely to beat 78.** Say so in the report. The top systems are LLM ensembles from well-resourced groups, and the instructor has explicitly said that carefully analysed negative results are acceptable.

### The result that actually matters

> On items where annotators marked course-of-action as driving the similarity, adding SRL features improves accuracy by **3–8 points** over embeddings alone; on items where they did not, the gain is **≈0 or negative**.

If that holds, the claim is: *predicate–argument structure encodes course-of-action similarity specifically.* That is an interpretability result about what a representation captures, and it is what transfers to the Kinyarwanda work.

### The honest statistical caveat

This is the weakest part of the design and it should be in the report, not hidden.

| comparison | n | 95% CI half-width at p=0.6 |
|---|---|---|
| test, all items | 400 | ±4.8 pts |
| test, course=True | 214 | ±6.6 pts |
| test, outcome=True | 96 | ±9.8 pts |

Comparing two independent confidence intervals at these sizes would require an enormous effect. **This is why system comparisons must be paired** — McNemar's test on the same items is far more sensitive. Minimum detectable effect, 80% power:

| stratum | if systems disagree on 10% of items | on 20% |
|---|---|---|
| test, all (n=400) | ~4.4 pts | ~6.3 pts |
| test, course=True (n=214) | ~6.1 pts | ~8.6 pts |
| test, outcome=True (n=96) | ~9.0 pts | ~12.8 pts |

So: the **outcomes stratum is probably underpowered** — report it, but do not build a claim on it. The course-of-action stratum can detect a real effect only if it is on the order of 6–9 points. My honest estimate is roughly a coin flip whether the effect is that large.

### If the primary result does not appear

The report still has: reproduced baselines, the dev/test asymmetry finding, the confound controls, a measured SRL/coref tool error rate, and a stratified analysis showing structural features *do not* add signal — with the genre scope condition stated. That is a complete, honest report, and it is the realistic worst case rather than a failure.

---

## 7. Open items before the proposal is final

1. The proposal currently describes the per-aspect diagnostic in general terms. It should now cite the measured TF-IDF stratum gap (54.67 vs 62.37) as preliminary evidence — that converts "planned approach" from a plan into a demonstrated lead.
2. Decide whether to report Track B at all, or focus on Track A. Track B is cheaper (off-the-shelf `e5` gets 66.75) but the structural features fit Track A's pairwise framing more naturally.
3. Confirm the citation format CS 421 wants for the ACL Anthology ID.
