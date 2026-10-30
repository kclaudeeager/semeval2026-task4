"""Compare stories component by component, the way the annotators did."""
from __future__ import annotations

from ..components import decompose
from ..dataset import Triple
from ..registry import register
from .base import Scorer

COMPONENTS = ("theme", "action", "outcome")


@register("l7_components")
class ComponentScorer(Scorer):
    """Embed each component separately; score is a weighted sum of cosines.

    Weights are fitted by grid search unless given, so the data decides how much
    each component is worth. That fitted weight vector is the result: it is a
    direct estimate of what the three components contribute.
    """

    fits_parameters = True

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embed_prefix: str = "", spacy_model: str = "en_core_web_sm",
                 weights: list[float] | None = None, grid_step: float = 0.1,
                 batch_size: int = 64, use_parser: bool = True):
        self.embed_model = embed_model
        self.embed_prefix = embed_prefix
        self.spacy_model = spacy_model
        self.weights = weights
        self.grid_step = grid_step
        self.batch_size = batch_size
        self.use_parser = use_parser
        self._enc = None
        self._nlp = None
        self._cache: dict[str, dict[str, object]] = {}

    def _encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer

            self._enc = SentenceTransformer(self.embed_model)
        return self._enc

    def _parser(self):
        if self._nlp is None and self.use_parser:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def parts(self, text: str) -> dict[str, object]:
        """Embedded components for one story, cached."""
        if text in self._cache:
            return self._cache[text]
        comp = decompose(text, self._parser()).as_dict()
        vecs = self._encoder().encode(
            [self.embed_prefix + comp[c] for c in COMPONENTS],
            normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=False)
        self._cache[text] = dict(zip(COMPONENTS, vecs))
        return self._cache[text]

    def similarities(self, anchor: str, candidate: str) -> list[float]:
        """One cosine per component, in COMPONENTS order."""
        import numpy as np

        A, B = self.parts(anchor), self.parts(candidate)
        return [float(np.dot(A[c], B[c])) for c in COMPONENTS]

    def fit(self, triples: list[Triple]) -> "ComponentScorer":
        if self.weights is not None:
            return self
        import itertools

        rows = [(self.similarities(t.anchor, t.text_a),
                 self.similarities(t.anchor, t.text_b)) for t in triples]
        labels = [bool(t.a_is_closer) for t in triples]
        steps = [round(i * self.grid_step, 4) for i in range(int(1 / self.grid_step) + 1)]
        best, best_hits = None, -1
        for combo in itertools.product(steps, repeat=len(COMPONENTS)):
            if abs(sum(combo) - 1.0) > 1e-9:
                continue
            hits = sum(((sum(w * v for w, v in zip(combo, a))
                         >= sum(w * v for w, v in zip(combo, b))) == y)
                       for (a, b), y in zip(rows, labels))
            if hits > best_hits:
                best, best_hits = combo, hits
        self.weights = list(best) if best else [1.0, 0.0, 0.0]
        return self

    def fitted_params(self) -> dict:
        return {"components": list(COMPONENTS),
                "weights": [round(float(w), 3) for w in (self.weights or [])]}

    def score(self, anchor: str, candidate: str) -> float:
        w = self.weights or [1.0, 0.0, 0.0]
        return sum(wi * s for wi, s in zip(w, self.similarities(anchor, candidate)))


@register("l7s_single_component")
class SingleComponent(ComponentScorer):
    """One component on its own, for the validity check against aspect flags."""

    fits_parameters = False

    def __init__(self, component: str = "outcome", **kwargs):
        kwargs.pop("weights", None)
        super().__init__(weights=[1.0 if c == component else 0.0 for c in COMPONENTS],
                         **kwargs)
        self.component = component
