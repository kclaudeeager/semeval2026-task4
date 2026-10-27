"""Loading Track A triples. File contents are treated strictly as data."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Triple:
    anchor: str
    text_a: str
    text_b: str
    a_is_closer: bool | None = None
    human_labels: list[bool] = field(default_factory=list)
    course_of_actions: list[bool] = field(default_factory=list)
    outcomes: list[bool] = field(default_factory=list)
    abstract_theme: list[bool] = field(default_factory=list)

    @property
    def n_annotators(self) -> int:
        return len(self.human_labels)

    @property
    def unanimous(self) -> bool:
        return len(set(self.human_labels)) == 1 if self.human_labels else False


def _bools(raw) -> list[bool]:
    return [bool(x) for x in (raw or [])]


def load_triples(path: str | Path) -> list[Triple]:
    """Read a Track A jsonl file into Triple records."""
    rows: list[Triple] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            label = r.get("text_a_is_closer")
            rows.append(
                Triple(
                    anchor=r["anchor_text"],
                    text_a=r["text_a"],
                    text_b=r["text_b"],
                    a_is_closer=None if label is None else bool(label),
                    human_labels=_bools(r.get("human_labels")),
                    course_of_actions=_bools(r.get("course_of_actions")),
                    outcomes=_bools(r.get("outcomes")),
                    abstract_theme=_bools(r.get("abstract_theme")),
                )
            )
    return rows
