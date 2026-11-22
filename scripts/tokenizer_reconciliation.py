"""Why the L0 token-overlap baseline does not match the published figure.

The overview reports 56.25% for a lexical baseline on Track A test; the
implementation in `scorers/lexical.py` measures 58.00%. This script explains
the difference rather than hiding it, by recomputing the same baseline under a
set of ordinary tokenizations and reporting which ones reproduce the published
number.

    TASK4_ALLOW_TEST=1 python scripts/tokenizer_reconciliation.py

It writes reports/results/tokenizer_reconciliation.json, from which
scripts/export_numbers.py generates the macros used in the data section. It
fits nothing, selects nothing, and changes no reported system result: the L0
baseline is a reference point, not a term in any comparison. The
pre-registration anticipates exactly this run for `l0_jaccard`.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task4.evaluate import load_split  # noqa: E402

DATA_ROOT = "data/raw/narrative-similarity-dataset"
OUT = Path("reports/results/tokenizer_reconciliation.json")
PUBLISHED_TEST = 56.25

# An ad-hoc list, used only to show that stopword handling is not the cause.
STOP = set("""a an the and or but if then than that this these those of in on at to for
with from by as is are was were be been being it its he she they them his her their
not no so such own same too very can will just don t s""".split())


def _mk(pattern: str, lower: bool = True, strip_stop: bool = False,
        stem: bool = False):
    rx = re.compile(pattern)

    def tok(text: str) -> list[str]:
        s = str(text).lower() if lower else str(text)
        out = rx.findall(s)
        if strip_stop:
            out = [w for w in out if w not in STOP]
        if stem:
            out = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in out]
        return out

    return tok


# The first entry is the tokenizer actually used by scorers/lexical.py.
VARIANTS: dict[str, object] = {
    "apostrophe_kept": _mk(r"[a-z0-9']+"),
    "word_chars": _mk(r"\w+"),
    "whitespace": lambda x: str(x).lower().split(),
    "alphabetic": _mk(r"[a-z]+"),
    "case_sensitive": _mk(r"\w+", lower=False),
    "apostrophe_dropped": _mk(r"[a-z0-9]+"),
    "whitespace_cased": lambda x: str(x).split(),
    "stopworded": _mk(r"[a-z0-9']+", strip_stop=True),
    "stopworded_word_chars": _mk(r"\w+", strip_stop=True),
    "stopworded_stemmed": _mk(r"\w+", strip_stop=True, stem=True),
}
CURRENT = "apostrophe_kept"
REFERENCE = "word_chars"


def jaccard(a: str, b: str, tok) -> float:
    A, B = set(tok(a)), set(tok(b))
    union = A | B
    return len(A & B) / len(union) if union else 0.0


def predictions(triples, tok) -> list[bool]:
    """Matches evaluate.py: the comparison is `>=`, so ties go to candidate A."""
    return [jaccard(t.anchor, t.text_a, tok) >= jaccard(t.anchor, t.text_b, tok)
            for t in triples]


def sweep(triples) -> dict:
    out = {}
    for name, tok in VARIANTS.items():
        preds = predictions(triples, tok)
        correct = [p == bool(t.a_is_closer) for p, t in zip(preds, triples)]
        pairs = [(jaccard(t.anchor, t.text_a, tok), jaccard(t.anchor, t.text_b, tok))
                 for t in triples]
        out[name] = {
            "k": sum(correct),
            "n": len(triples),
            "accuracy": round(100.0 * sum(correct) / len(triples), 2),
            "ties": sum(1 for a, b in pairs if a == b),
        }
    return out


def vocabulary_difference(triples) -> dict:
    """Which token types the current pattern produces that `\\w+` does not."""
    cur, ref = VARIANTS[CURRENT], VARIANTS[REFERENCE]
    counts: collections.Counter = collections.Counter()
    for t in triples:
        for text in (t.anchor, t.text_a, t.text_b):
            for w in set(cur(text)) - set(ref(text)):
                counts[w] += 1
    total = sum(counts.values())
    apost = sum(n for w, n in counts.items() if "'" in w)
    apost_types = sum(1 for w in counts if "'" in w)
    # Types and occurrences give different shares, so both are recorded and
    # named: quoting one against the other's denominator is the easy mistake.
    return {
        "types": len(counts),
        "occurrences": total,
        "apostrophe_occurrences": apost,
        "apostrophe_types": apost_types,
        "apostrophe_pct_occurrences": round(100.0 * apost / total, 1) if total else 0.0,
        "apostrophe_pct_types": round(100.0 * apost_types / len(counts), 1) if counts else 0.0,
        "apostrophe_pct": round(100.0 * apost / total, 1) if total else 0.0,
        "most_common": counts.most_common(12),
    }


def flips(triples) -> dict:
    cur, ref = VARIANTS[CURRENT], VARIANTS[REFERENCE]
    pc, pr = predictions(triples, cur), predictions(triples, ref)
    gold = [bool(t.a_is_closer) for t in triples]
    changed = [(c, r, g) for c, r, g in zip(pc, pr, gold) if c != r]
    return {
        "changed": len(changed),
        "current_right": sum(1 for c, r, g in changed if c == g),
        "current_wrong": sum(1 for c, r, g in changed if r == g),
    }


def main() -> int:
    if os.environ.get("TASK4_ALLOW_TEST") != "1":
        raise SystemExit(
            "Refusing to touch the test split. This run is permitted by "
            "docs/PREREGISTRATION.md section 2 for l0_jaccard reconciliation "
            "only; set TASK4_ALLOW_TEST=1 to confirm that is what you mean.")

    out: dict = {"published_test": PUBLISHED_TEST, "current": CURRENT,
                 "reference": REFERENCE, "splits": {}}
    for split in ("dev", "test"):
        triples = [t for t in load_split(DATA_ROOT, split)
                   if t.a_is_closer is not None]
        out["splits"][split] = sweep(triples)
        if split == "test":
            out["vocabulary"] = vocabulary_difference(triples)
            out["flips"] = flips(triples)

    test = out["splits"]["test"]
    out["gap_points"] = round(test[CURRENT]["accuracy"] - PUBLISHED_TEST, 2)
    out["gap_items"] = test[CURRENT]["k"] - round(PUBLISHED_TEST * 4)
    out["matching_variants"] = sorted(
        n for n, v in test.items() if abs(v["accuracy"] - PUBLISHED_TEST) < 1e-9)

    dev = [v["accuracy"] for v in out["splits"]["dev"].values()]
    out["dev_variant_min"] = min(dev)
    out["dev_variant_max"] = max(dev)
    out["dev_all_below_chance"] = max(dev) < 50.0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"wrote {OUT}")
    print(f"  test: {CURRENT} {test[CURRENT]['accuracy']:.2f} "
          f"vs published {PUBLISHED_TEST:.2f}  "
          f"({out['gap_points']:+.2f} pts, {out['gap_items']:+d} items)")
    print(f"  reproduce the published figure: "
          f"{', '.join(out['matching_variants'])}")
    print(f"  prediction changes vs {REFERENCE}: {out['flips']['changed']} "
          f"({out['flips']['current_right']} for, "
          f"{out['flips']['current_wrong']} against)")
    print(f"  dev across all {len(VARIANTS)} variants: "
          f"{out['dev_variant_min']:.2f}-{out['dev_variant_max']:.2f}, "
          f"all below chance: {out['dev_all_below_chance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
