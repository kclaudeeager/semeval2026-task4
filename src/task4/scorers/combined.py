"""L5: weighted combination of L2, L3 and L4, weights fitted by cross-validation."""
from __future__ import annotations

from ..dataset import Triple
from ..registry import build, register
from .base import Scorer


@register("l5_combined")
class CombinedScorer(Scorer):
    """Linear blend of component scores, each z-normalised over the fitting split.

    `weights` may be given directly, or left None to be fitted by grid search on
    the split passed to fit(). Fitting never sees the test split.
    """

    fits_parameters = True

    def __init__(self, components: list[str] | None = None,
                 weights: list[float] | None = None, grid_step: float = 0.1):
        self.component_names = components or ["l2_e5", "l3_characters", "l4_roles"]
        self.weights = weights
        self.grid_step = grid_step
        self._components = [build(n) for n in self.component_names]
        self._stats: list[tuple[float, float]] = []

    def _raw(self, anchor: str, candidate: str) -> list[float]:
        return [c.score(anchor, candidate) for c in self._components]

    def fit(self, triples: list[Triple]) -> "CombinedScorer":
        import itertools
        import statistics

        for c in self._components:
            c.fit(triples)
        rows = [(self._raw(t.anchor, t.text_a), self._raw(t.anchor, t.text_b)) for t in triples]
        flat = [[r[i] for pair in rows for r in pair] for i in range(len(self._components))]
        self._stats = [(statistics.mean(v), statistics.pstdev(v) or 1.0) for v in flat]

        def norm(vals):
            return [(v - m) / s for v, (m, s) in zip(vals, self._stats)]

        if self.weights is None:
            labels = [bool(t.a_is_closer) for t in triples]
            steps = [round(i * self.grid_step, 4)
                     for i in range(int(1 / self.grid_step) + 1)]
            best, best_acc = None, -1.0
            for combo in itertools.product(steps, repeat=len(self._components)):
                total = sum(combo)
                if abs(total - 1.0) > 1e-9:
                    continue
                hits = sum(
                    (sum(w * v for w, v in zip(combo, norm(a)))
                     >= sum(w * v for w, v in zip(combo, norm(b)))) == y
                    for (a, b), y in zip(rows, labels)
                )
                if hits > best_acc:
                    best, best_acc = combo, hits
            self.weights = list(best) if best else [1.0] + [0.0] * (len(self._components) - 1)
        return self

    def fitted_params(self) -> dict:
        return {"components": self.component_names,
                "weights": [round(float(w), 4) for w in (self.weights or [])]}

    def score(self, anchor: str, candidate: str) -> float:
        if not self._stats:
            raise RuntimeError("CombinedScorer.fit() must run before score()")
        vals = self._raw(anchor, candidate)
        return sum(w * (v - m) / s for w, v, (m, s) in zip(self.weights, vals, self._stats))
