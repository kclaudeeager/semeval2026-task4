"""L0 and L1: random, Jaccard overlap, TF-IDF cosine."""
from __future__ import annotations
import random
import re

from ..dataset import Triple
from ..registry import register
from .base import Scorer

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


@register("l0_random")
class RandomScorer(Scorer):
    """Chance baseline. Seeded, so a run is reproducible."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def score(self, anchor: str, candidate: str) -> float:
        return self._rng.random()


@register("l0_jaccard")
class JaccardScorer(Scorer):
    """Token-set overlap. Published Track A baseline: 56.25 on test."""

    def score(self, anchor: str, candidate: str) -> float:
        a, b = set(tokenize(anchor)), set(tokenize(candidate))
        union = a | b
        return len(a & b) / len(union) if union else 0.0


@register("l1_tfidf")
class TfidfScorer(Scorer):
    """TF-IDF cosine. Document frequencies come from fit(), never from the test split."""

    def __init__(self, ngram_max: int = 2, min_df: int = 2, sublinear_tf: bool = True,
                 stop_words: str | None = "english"):
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words
        self._vec = None

    def fit(self, triples: list[Triple]) -> "TfidfScorer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [t for tr in triples for t in (tr.anchor, tr.text_a, tr.text_b)]
        self._vec = TfidfVectorizer(
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
            ngram_range=(1, self.ngram_max),
            min_df=self.min_df,
        ).fit(corpus)
        return self

    def score(self, anchor: str, candidate: str) -> float:
        if self._vec is None:
            raise RuntimeError("TfidfScorer.fit() must run before score()")
        from sklearn.metrics.pairwise import cosine_similarity

        m = self._vec.transform([anchor, candidate])
        return float(cosine_similarity(m[0], m[1])[0][0])
