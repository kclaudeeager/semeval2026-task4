"""Does the token distribution separate the candidates, and in which direction?

Jaccard ignores frequency; TF-IDF cosine normalizes length away. A
distributional divergence keeps both, so it can see things neither of those
can. This script asks whether it does, for several measures at once.

Each measure is a distance: lower means "closer to the anchor". For every
triple we ask whether the chosen candidate scores lower than the distractor,
count how often, and test that count against a coin flip. A rate ABOVE 50%
means the measure agrees with the annotators; BELOW 50% means it
systematically prefers the distractor.

The asymmetric measures matter most. KL(anchor || candidate) is large when
the candidate fails to cover the anchor's vocabulary; KL(candidate || anchor)
is large when the candidate introduces vocabulary the anchor lacks. The two
can disagree, and that disagreement describes how the distractor was built.

    python scripts/lexical_direction.py [dev|test]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from task4.evaluate import load_split

OUT_DIR = Path("reports/results")
DATA_ROOT = "data/raw/narrative-similarity-dataset"
SMOOTH = 0.1

STOP = set("""a an the and or but if while of to in on at by for with from as is are was were
be been being it its this that these those he she they them his her their i you we not no so
then than there here who whom which what when where how all any both each few more most other
some such only own same too very s t can will just don should now into out up down over under
again further once about against between during before after above below""".split())


def tokens(text: str, content_only: bool) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in STOP] if content_only else words


def distribution(toks: list[str], vocab: set[str]) -> dict[str, float]:
    counts = Counter(toks)
    total = sum(counts.values()) + SMOOTH * len(vocab)
    return {w: (counts.get(w, 0) + SMOOTH) / total for w in vocab}


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(p[w] * math.log(p[w] / q[w]) for w in p)


def jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    m = {w: 0.5 * (p[w] + q[w]) for w in p}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cross_entropy(p_cand: dict[str, float], p_anchor: dict[str, float]) -> float:
    """Surprise of the candidate's vocabulary under the anchor's model."""
    return -sum(p_cand[w] * math.log(p_anchor[w]) for w in p_cand)



def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token sequences.

    Jaccard compares sets, so it is blind to order: "John killed Mary" and
    "Mary killed John" score identically. Edit distance is the cheapest
    surface measure that is NOT order-blind, which makes it the right control
    for whether order is the missing ingredient.
    """
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def type_token_ratio(toks: list[str]) -> float:
    return len(set(toks)) / len(toks) if toks else 0.0


def exact_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


MEASURES = ("js", "kl_anchor_given_cand", "kl_cand_given_anchor",
            "cross_entropy", "edit_distance", "edit_distance_norm",
            "length_diff", "type_token_diff")


def score_triple(anchor, cand, content_only):
    a = tokens(anchor, content_only)
    c = tokens(cand, content_only)
    vocab = set(a) | set(c)
    pa, pc = distribution(a, vocab), distribution(c, vocab)
    ed = edit_distance(a, c)
    return {
        "js": jensen_shannon(pa, pc),
        "kl_anchor_given_cand": kl(pa, pc),
        "kl_cand_given_anchor": kl(pc, pa),
        "cross_entropy": cross_entropy(pc, pa),
        "edit_distance": float(ed),
        # normalized, so a candidate is not favoured merely for being short
        "edit_distance_norm": ed / max(len(a), len(c), 1),
        "length_diff": abs(len(c) - len(a)),
        "type_token_diff": abs(type_token_ratio(c) - type_token_ratio(a)),
    }


def main(split: str = "dev") -> None:
    # The CLI refuses the test split unless explicitly unlocked. This script
    # bypasses the CLI, so it carries the same guard -- otherwise the project
    # has a side door onto the data it has promised to touch once.
    if split == "test" and os.environ.get("TASK4_ALLOW_TEST") != "1":
        raise SystemExit(
            "Refusing to touch the test split.\n"
            "The protocol allows one frozen test run. If this measure is part of\n"
            "it, say so in writing first (see docs/PREREGISTRATION.md), then set\n"
            "TASK4_ALLOW_TEST=1 for that run."
        )
    triples = [t for t in load_split(DATA_ROOT, split) if t.a_is_closer is not None]
    record = {"split": split, "n": len(triples), "variants": {}}
    print(f"{split}: {len(triples)} triples\n")
    print("Each measure is a distance. The count is how often the CHOSEN candidate")
    print("scores lower. Above 50% agrees with the annotators; below 50% means the")
    print("measure systematically prefers the distractor.\n")

    for content_only in (False, True):
        key = "content_words" if content_only else "all_tokens"
        wins = {m: 0 for m in MEASURES}
        for t in triples:
            right = t.text_a if t.a_is_closer else t.text_b
            wrong = t.text_b if t.a_is_closer else t.text_a
            sr = score_triple(t.anchor, right, content_only)
            sw = score_triple(t.anchor, wrong, content_only)
            for m in MEASURES:
                if sr[m] < sw[m]:
                    wins[m] += 1

        n = len(triples)
        entry = {}
        print(f"=== {key.replace('_', ' ')} ===")
        for m in MEASURES:
            k = wins[m]
            p = exact_binomial(k, n)
            entry[m] = {"k": k, "n": n, "pct": 100.0 * k / n, "p": p}
            direction = ""
            if p < 0.05:
                direction = ("  <-- AGREES with annotators" if k > n / 2
                             else "  <-- PREFERS THE DISTRACTOR")
            print(f"  {m:<22} {k:>4}/{n}  ({100*k/n:5.1f}%)  p={p:.3f}{direction}")
        record["variants"][key] = entry
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lexical_direction__{split}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")
    print("\nA measure below 50% can always be inverted to score above 50%, but the")
    print("decision to invert is made after seeing these numbers. Any inverted")
    print("variant is a fitted choice and means nothing until it is confirmed on")
    print("data that was not used to pick the direction.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
