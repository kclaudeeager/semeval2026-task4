"""L2: dense sentence embeddings (e5). The backbone the structural layers add to."""
from __future__ import annotations

from ..registry import register
from .base import Scorer


@register("l2_e5")
class E5Scorer(Scorer):
    """Cosine over e5 embeddings.

    e5 expects a "query: " / "passage: " prefix. Both stories are documents here,
    so both get the same prefix and the comparison stays symmetric.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2", prefix: str = "query: ",
                 batch_size: int = 32, cache: bool = True):
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self._cache: dict[str, object] | None = {} if cache else None
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, text: str):
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        vec = self._load().encode(self.prefix + text, normalize_embeddings=True,
                                 batch_size=self.batch_size, show_progress_bar=False)
        if self._cache is not None:
            self._cache[text] = vec
        return vec

    def score(self, anchor: str, candidate: str) -> float:
        import numpy as np

        return float(np.dot(self._embed(anchor), self._embed(candidate)))
