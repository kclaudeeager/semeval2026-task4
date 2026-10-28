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

    @staticmethod
    def _preflight() -> None:
        """Fail with a sentence that says what to do.

        transformers 5.x disables PyTorch when it finds torch < 2.6 and then
        raises an unrelated NameError deep inside an import chain. Checking
        first turns forty lines of traceback into one line of instruction.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'L2 needs the embeddings extra: uv pip install -e ".[embeddings]"'
            ) from exc
        import transformers

        torch_v = tuple(int(x) for x in torch.__version__.split(".")[:2])
        tf_v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        # transformers 5.x disables PyTorch below 2.6 and then dies with an
        # unrelated NameError. On Intel macOS torch cannot go above 2.2.2, so
        # there the fix is to hold transformers at 4.x instead.
        if tf_v >= (5, 0) and torch_v < (2, 6):
            raise RuntimeError(
                f"transformers {transformers.__version__} needs torch>=2.6 but "
                f"torch {torch.__version__} is installed.\n"
                'Fix: uv pip install -e ".[embeddings]" --upgrade\n'
                "On Intel macOS torch stops at 2.2.2, so that command pins "
                "transformers to 4.x instead."
            )

    def _load(self):
        if self._model is None:
            self._preflight()
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
