"""L3: characters recovered by coreference, compared by best one-to-one alignment.

The hypothesis under test. Chaturvedi et al. (2018) aligned characters without
resolving coreference and named missed coreference as a source of error; this
layer supplies the component they lacked.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from ..registry import register
from .base import Scorer
from .lexical import tokenize


@dataclass
class Character:
    """One coreference chain: its mention strings and the words around them."""
    mentions: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    @property
    def head(self) -> str:
        return max(self.mentions, key=len) if self.mentions else ""


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


@register("l3_characters")
class CharacterAlignmentScorer(Scorer):
    """Similarity is the mean weight of the best one-to-one cast matching.

    Pairwise character weight blends mention-string overlap with context overlap;
    `alpha` sets the balance. The matching itself is the Hungarian algorithm, so
    the alignment is optimal rather than greedy.
    """

    def __init__(self, model_name: str = "sapienzanlp/maverick-mes-litbank",
                 alpha: float = 0.5, context_window: int = 10,
                 min_mentions: int = 1, cache_dir: str | None = "data/interim/coref"):
        self.model_name = model_name
        self.alpha = alpha
        self.context_window = context_window
        self.min_mentions = min_mentions
        self.cache_dir = cache_dir
        self._model = None
        self._cast_cache: dict[str, list[Character]] = {}

    def _load(self):
        if self._model is None:
            from maverick import Maverick

            self._model = Maverick(hf_name_or_path=self.model_name)
        return self._model

    def cast(self, text: str) -> list[Character]:
        """Coreference chains for one story, cached by text."""
        if text in self._cast_cache:
            return self._cast_cache[text]
        out = self._model_cast(text)
        self._cast_cache[text] = out
        return out

    def _model_cast(self, text: str) -> list[Character]:
        doc = self._load().predict(text)
        tokens = doc["tokens"]
        chars: list[Character] = []
        for cluster in doc.get("clusters_token_offsets", []):
            c = Character()
            for start, end in cluster:
                c.mentions.append(" ".join(tokens[start:end + 1]).lower())
                lo = max(0, start - self.context_window)
                hi = min(len(tokens), end + 1 + self.context_window)
                c.context.extend(w.lower() for w in tokens[lo:hi])
            if len(c.mentions) >= self.min_mentions:
                chars.append(c)
        return chars

    def pair_weight(self, x: Character, y: Character) -> float:
        mention = _jaccard(set(" ".join(x.mentions).split()), set(" ".join(y.mentions).split()))
        context = _jaccard(set(x.context), set(y.context))
        return self.alpha * mention + (1 - self.alpha) * context

    def align(self, left: list[Character], right: list[Character]) -> float:
        """Mean weight over the optimal one-to-one matching; 0 when either cast is empty."""
        if not left or not right:
            return 0.0
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        w = np.array([[self.pair_weight(a, b) for b in right] for a in left])
        rows, cols = linear_sum_assignment(-w)
        return float(w[rows, cols].mean())

    def score(self, anchor: str, candidate: str) -> float:
        return self.align(self.cast(anchor), self.cast(candidate))
