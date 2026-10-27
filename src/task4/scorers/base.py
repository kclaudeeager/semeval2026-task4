"""The one interface every layer implements.

A Scorer maps (anchor, candidate) to a real number where higher means more
similar. The triple decision is always the same: score both candidates, pick the
higher. Adding a layer or a variant means adding a Scorer and a config file —
nothing else in the pipeline changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from ..dataset import Triple


class Scorer(ABC):
    name: str = "unnamed"

    def fit(self, triples: list[Triple]) -> "Scorer":
        """Optional corpus pass (document frequencies, weight fitting). Default: none."""
        return self

    @abstractmethod
    def score(self, anchor: str, candidate: str) -> float:
        ...

    def predict(self, triples: list[Triple]) -> list[bool]:
        """True when candidate A is judged closer, matching `text_a_is_closer`."""
        return [self.score(t.anchor, t.text_a) >= self.score(t.anchor, t.text_b) for t in triples]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
