# SemEval-2026 Task 4, Track A — narrative story similarity

CS 421 course project. Given an anchor story and two candidates, decide which
candidate is narratively closer. The project is a **layered ablation**: each layer
adds one interpretable representation, and the question is whether character and
role structure contributes anything beyond dense embeddings on triples that were
selected because two language models disagreed about them.

## Why the layout looks like this

Comparisons go through configuration, never through edited code. Every layer
implements one interface (`Scorer.score(anchor, candidate) -> float`) and is
registered under a name. A run names a layer, a split and an eval profile:

```bash
make setup                                   # venv outside OneDrive + install
make data                                    # unpack the release into data/raw
make test                                    # 24 tests, no network needed
python -m task4.cli scorer=l1_tfidf data=dev # one layer, one split
make ablation                                # every layer, swept over dev
```

Adding a variant is a new YAML file in `configs/scorer/`, optionally with a new
`Scorer` subclass. Nothing in the harness, the statistics or the reporting
changes, so two runs are always comparable.

## The layers

| | layer | what it adds |
|---|---|---|
| L0 | `l0_random`, `l0_jaccard` | chance and token overlap; reproduces the published baselines |
| L1 | `l1_tfidf` | how much of the task is lexical overlap |
| L2 | `l2_e5` | dense sentence embeddings — the backbone |
| L3 | `l3_characters` | coreference chains, casts aligned one-to-one (Hungarian) |
| L4 | `l4_roles` | agent/patient verb lemmas from the dependency parse |
| L5 | `l5_combined` | weighted blend of L2–L4, weights fitted by cross-validation |

## Guard rails that are code, not good intentions

- **The test split is locked.** `data=test` exits unless `eval.allow_test=true`
  is passed explicitly. The proposal commits to one held-out run.
- **Fitting never sees test.** `configs/data/test.yaml` sets `fit_split: dev`, so
  document frequencies and L5 weights come from dev.
- **Strata are unanimous or excluded.** A majority rule over two annotators folds
  the 1-of-2 case into "not marked", which is why the proposal's first stratified
  result reversed between splits. `task4.strata` keeps `all_marked`,
  `none_marked` and `split` apart, and the diagnostic uses only the first two.
- **Every result is self-describing.** `reports/results/<split>__<scorer>.json`
  carries the resolved config, the git SHA, the seed and the per-item correctness
  vector, so McNemar can be run later without recomputing anything.

## Layout

```
configs/      composition groups: scorer/ data/ eval/ experiment/
src/task4/    scorers/ (the layers), evaluate.py, stats.py, strata.py, cli.py
data/raw/     the release, unpacked by scripts/fetch_data.sh (never committed)
data/interim/ coreference and parse caches
reports/      results/ (one JSON per run) and figures/
references/   proposal, design notes, paper extracts
tests/        24 tests covering the statistics, the strata rule and the harness
docs/PLAN.md  the schedule through 25 November
```

## Reproduced so far

| layer | dev | test | published |
|---|---|---|---|
| `l0_jaccard` | 45.00 | 58.00 | 56.25 |
| `l0_random` | 46.00 | — | 50.00 |
| `l1_tfidf` | 51.50 | — | — |

Jaccard reproduces the published figure to within seven items on test. The dev
Jaccard figure of 45.00 sits below the 50% chance line, which is what the
adversarial construction of the benchmark predicts.
