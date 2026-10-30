"""Does each extracted component carry the signal its name claims?

Before asking whether a component decomposition improves accuracy, ask whether
the decomposition is real. The annotation scheme gives a way to check: each
triple records which aspects drove the annotators' decision, so a component that
genuinely captures "outcome" should do better on triples where annotators marked
outcomes than on triples where they did not.

That is a validity test, and it is independent of overall accuracy. A component
that fails it is mislabelled, and no downstream alignment will repair that.

Only unanimous strata are used: with two annotators a majority rule folds the
1-of-2 case into "not marked", which mixes disagreement with genuine absence.

Usage: python scripts/validate_components.py [dev|test]
"""
from __future__ import annotations
import sys

from task4.dataset import load_triples
from task4.evaluate import load_split
from task4.registry import build
from task4.stats import two_proportion_z, wilson
from task4.strata import Stratum, partition

PAIRS = [("theme", "abstract_theme"), ("action", "course_of_actions"),
         ("outcome", "outcomes")]


def accuracy_on(preds, triples, rows):
    hits = [preds[i] == bool(triples[i].a_is_closer) for i in rows]
    return sum(hits), len(hits)


def main(split: str = "dev") -> None:
    triples = load_split("data/raw/narrative-similarity-dataset", split)
    print(f"{split}: {len(triples)} triples\n")
    print("Each component is run alone, then split by whether annotators marked")
    print("the matching aspect. A real component scores higher where it was marked.\n")

    for component, aspect in PAIRS:
        scorer = build("l7s_single_component", component=component)
        preds = scorer.predict(triples)
        idx = partition(triples, aspect)
        marked, unmarked = idx[Stratum.ALL_MARKED], idx[Stratum.NONE_MARKED]

        k1, n1 = accuracy_on(preds, triples, marked)
        k0, n0 = accuracy_on(preds, triples, unmarked)
        print(f"=== component '{component}'  vs aspect '{aspect}' ===")
        if n1 == 0 or n0 == 0:
            print(f"    cannot test: strata are {n1} and {n0}\n")
            continue
        print(f"    where annotators marked it   {wilson(k1, n1)}   n={n1}")
        print(f"    where they did not           {wilson(k0, n0)}   n={n0}")
        t = two_proportion_z(k1, n1, k0, n0)
        direction = "as predicted" if t["diff"] > 0 else "BACKWARDS"
        print(f"    difference {t['diff']:+.1f} pts   z={t['z']:+.2f}  p={t['p_value']:.3f}   {direction}\n")

    print("Read the direction before the p-value. These strata are small, so a")
    print("non-significant difference in the right direction is weak support, and")
    print("a difference in the wrong direction is a problem whatever its p-value.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
