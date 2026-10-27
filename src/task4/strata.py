"""Per-aspect strata.

The proposal's one measured negative result lives here: a majority rule over two
annotators collapses the 1-of-2 case into "not marked", which mixes unanimous
non-selection with genuine disagreement and makes the stratum unstable. Only the
unanimous strata are used for the diagnostic.
"""
from __future__ import annotations
from enum import Enum

from .dataset import Triple

ASPECTS = ("course_of_actions", "outcomes", "abstract_theme")


class Stratum(str, Enum):
    ALL_MARKED = "all_marked"     # every annotator flagged this aspect
    NONE_MARKED = "none_marked"   # no annotator flagged it
    SPLIT = "split"               # annotators disagreed: excluded from the diagnostic


def aspect_stratum(triple: Triple, aspect: str) -> Stratum:
    if aspect not in ASPECTS:
        raise ValueError(f"unknown aspect {aspect!r}; expected one of {ASPECTS}")
    flags = getattr(triple, aspect)
    if not flags:
        return Stratum.SPLIT
    if all(flags):
        return Stratum.ALL_MARKED
    if not any(flags):
        return Stratum.NONE_MARKED
    return Stratum.SPLIT


def partition(triples: list[Triple], aspect: str) -> dict[Stratum, list[int]]:
    """Index lists per stratum, so predictions can be sliced without recomputing."""
    out: dict[Stratum, list[int]] = {s: [] for s in Stratum}
    for i, t in enumerate(triples):
        out[aspect_stratum(t, aspect)].append(i)
    return out


def usable(triples: list[Triple], aspect: str, min_per_stratum: int = 30) -> bool:
    """An aspect is usable when both unanimous strata are large enough to compare.

    On the Task 4 test set this is true for `outcomes` (79 / 133) and false for
    `course_of_actions` (188 / 32) and `abstract_theme` (392 / 0).
    """
    p = partition(triples, aspect)
    return min(len(p[Stratum.ALL_MARKED]), len(p[Stratum.NONE_MARKED])) >= min_per_stratum
