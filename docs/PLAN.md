# Schedule to 25 November 2026

Final report and video walkthrough are due **25 November**. Written 27 October,
so 29 days. One milestone per week, each ending in something committed.

## Week 1 — 27 Oct to 2 Nov: the backbone (L2)

1. **28 Oct** — install the embeddings extra (`uv pip install -e ".[embeddings]"`)
   and confirm `intfloat/e5-base-v2` loads offline after one download.
2. **30 Oct** — run `scorer=l2_e5 data=dev`. Cache embeddings to
   `data/interim/` so later runs do not re-encode 600 summaries.
3. **1 Nov** — commit the first Track A figure for e5. The organisers publish
   only a Track B number (66.75), so this is a result the task does not have.
4. **2 Nov** — `make ablation` over L0–L2; McNemar L1 against L2.

**Exit:** L0–L2 reproduced on dev with intervals, one JSON per layer.

## Week 2 — 3 to 9 Nov: characters (L3)

5. **4 Nov** — install the structure extra; load `maverick-mes-litbank`.
6. **5 Nov** — cast extraction on 20 dev stories, eyeballed for sanity.
7. **7 Nov** — **hand-audit 50 items** and record the coreference error rate in
   `reports/coref_audit.md`. This is promised in failure case F2 and is a
   reportable finding whichever way it comes out.
8. **9 Nov** — `scorer=l3_characters data=dev`, paired against L2.

**Exit:** L3 measured, coreference error rate written down.

## Week 3 — 10 to 16 Nov: roles and the blend (L4, L5)

9. **11 Nov** — `python -m spacy download en_core_web_sm`; role profiles on dev.
10. **13 Nov** — `scorer=l4_roles data=dev`, paired against L3 (isolating roles)
    and against L2 (isolating the whole structural addition).
11. **15 Nov** — fit L5 weights by cross-validation on dev + sample (239 items),
    configuration count capped in advance and recorded in the run config.
12. **16 Nov** — full `make ablation`; assemble the dev results table.

**Exit:** the complete L0–L5 ablation on dev, every pair compared with McNemar.

## Week 4 — 17 to 21 Nov: the diagnostic and the one test run

13. **18 Nov** — `make diagnostic`: accuracy inside the unanimous `outcomes`
    strata (79 and 133 on test), L4 paired against L2.
14. **19 Nov** — freeze the configuration. No further tuning after this point.
15. **20 Nov** — **the single held-out run**:
    `python -m task4.cli --multirun experiment=ablation data=test eval.allow_test=true`.
    Run it once. Whatever it says is the result.
16. **21 Nov** — figures into `reports/figures/`.

**Exit:** test results committed, config frozen before they were seen.

## Week 5 — 22 to 25 Nov: writing and the walkthrough

17. **22 Nov** — report draft: method, ablation table, diagnostic, failure cases.
18. **23 Nov** — add the standard deviations the instructor asked for, computed
    with `task4.stats.describe` over the per-system figures in the overview.
19. **24 Nov** — record the video: repo tour, one `make ablation` run, the
    diagnostic, and the negative results stated plainly.
20. **25 Nov** — submit.

## Carried forward from the proposal

- **The TF-IDF dev figure changes under the clean protocol.** The proposal
  reports 49.0% on dev; that vectorizer was fitted over dev *and* test. Fitted on
  dev alone, as `configs/data/dev.yaml` now requires, it is **51.50%**. Use the
  new number in the report and say why it moved.
- **The seven-item Jaccard gap** (58.00 measured against 56.25 published) is still
  unexplained. Likely tokenization. Resolve it in Week 1 or report it as open.
- **Abstract theme stays out of the diagnostic**: every annotator marked it on
  392 of 400 test items, so it has no contrast to measure.

## If something slips

L3 and L4 are the risk. If coreference will not install or is too slow, fall back
to string-matched character mentions, keep the same alignment and the same
diagnostic, and report the substitution. The ablation design survives it; a
negative result is still a result.
