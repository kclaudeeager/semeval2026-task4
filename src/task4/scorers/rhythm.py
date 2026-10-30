"""Sequential structure: where in the story things happen.

Two readings of "rhythm", with opposite evidence behind them.

The emotional arc -- does tension rise here and fall there -- has been tested on
this problem twice and failed twice. Elsner (EACL 2012) found a sentiment-curve
baseline scored exactly 50, chance. CascadeMind included a tension-curve signal
in a learned combination on this very task and it received weight 0.02.

Positional structure is different, and is the one structural signal in the task
with a positive result: CascadeMind's five-phase story grammar received weight
0.40, against 0.49 for TF-IDF and 0.08 for embeddings. These layers test it.

`PositionalSegments` cuts both stories into the same number of ordered segments
and compares them position by position: beginnings with beginnings, endings with
endings. It asks whether two stories put comparable material in comparable
places.

`ElasticSegments` relaxes that with dynamic time warping, so a story that dwells
on its opening and rushes its ending can still align with one that does the
reverse. Order is preserved, pacing is not. The difference between the two
layers isolates pacing from order.
"""
from __future__ import annotations

from ..registry import register
from .base import Scorer


def split_segments(text: str, k: int) -> list[str]:
    """Cut text into k ordered segments of roughly equal sentence count."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return [text] * k
    if len(sentences) <= k:
        # Pad by repeating the last sentence rather than emitting empties, so
        # every story yields k comparable slots.
        return sentences + [sentences[-1]] * (k - len(sentences))
    size = len(sentences) / k
    return [" ".join(sentences[int(i * size):int((i + 1) * size)]) or sentences[-1]
            for i in range(k)]


class _SegmentEncoder(Scorer):
    """Shared embedding and caching for the two sequential layers."""

    def __init__(self, segments: int = 5,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embed_prefix: str = "", batch_size: int = 64):
        self.segments = segments
        self.embed_model = embed_model
        self.embed_prefix = embed_prefix
        self.batch_size = batch_size
        self._enc = None
        self._cache: dict[str, object] = {}

    def _encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer

            self._enc = SentenceTransformer(self.embed_model)
        return self._enc

    def vectors(self, text: str):
        """k normalised segment vectors, in order."""
        import numpy as np

        if text in self._cache:
            return self._cache[text]
        segs = split_segments(text, self.segments)
        mat = np.asarray(self._encoder().encode(
            [self.embed_prefix + s for s in segs],
            normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=False))
        self._cache[text] = mat
        return mat


@register("l6_positional_segments")
class PositionalSegments(_SegmentEncoder):
    """Compare segment i of one story with segment i of the other.

    Strictly positional: no re-ordering and no stretching. Two stories score
    highly only if comparable material sits at comparable points.
    """

    def score(self, anchor: str, candidate: str) -> float:
        import numpy as np

        A, B = self.vectors(anchor), self.vectors(candidate)
        return float(np.mean(np.sum(A * B, axis=1)))


@register("l6e_elastic_segments")
class ElasticSegments(_SegmentEncoder):
    """Dynamic time warping over segments: order kept, pacing free.

    The warping path may linger on one segment or skip ahead, so a story that
    spends half its length on setup can still match one that does not. Reading
    this against the positional layer separates *pacing* from *order*: if the
    elastic version wins, the stories share a sequence but not a tempo.
    """

    def score(self, anchor: str, candidate: str) -> float:
        import numpy as np

        A, B = self.vectors(anchor), self.vectors(candidate)
        n, m = len(A), len(B)
        sim = A @ B.T                      # cosine, already normalised
        cost = np.full((n + 1, m + 1), -np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost[i, j] = sim[i - 1, j - 1] + max(
                    cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
        # Normalise by path length so k does not inflate the score.
        return float(cost[n, m] / max(n, m))
