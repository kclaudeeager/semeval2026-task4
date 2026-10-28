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

    #: True when fit() learns parameters from the data, which makes fitting and
    #: scoring the same split circular. Those layers get cross-validated; the
    #: stateless ones must not be, because re-fitting per fold would disturb a
    #: seeded baseline without making it any more honest.
    fits_parameters: bool = False

    def fit(self, triples: list[Triple]) -> "Scorer":
        """Optional corpus pass (document frequencies, weight fitting). Default: none."""
        return self

    @abstractmethod
    def score(self, anchor: str, candidate: str) -> float:
        ...

    def fitted_params(self) -> dict:
        """Whatever fit() learned, for the record.

        A fitted combiner's weights are the most informative number a run
        produces; leaving them out of the result file was a real omission.
        """
        return {}

    def predict(self, triples: list[Triple]) -> list[bool]:
        """True when candidate A is judged closer, matching `text_a_is_closer`."""
        return [self.score(t.anchor, t.text_a) >= self.score(t.anchor, t.text_b) for t in triples]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
