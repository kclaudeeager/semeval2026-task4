"""L3: characters recovered by coreference, compared by best one-to-one alignment.

The hypothesis under test. Chaturvedi et al. (2018) aligned characters without
resolving coreference and named missed coreference as a source of error; this
layer supplies the component they lacked.
"""
from __future__ import annotations
import contextlib
from dataclasses import dataclass, field

from ..registry import register
from .base import Scorer


@contextlib.contextmanager
def _prefer_safetensors():
    """Make every nested ``from_pretrained`` load safetensors.

    transformers 4.57 refuses to ``torch.load`` a ``pytorch_model.bin`` unless
    torch >= 2.6 (CVE-2025-32434), and torch is capped at 2.2.2 on Intel macOS.
    The restriction does not apply to safetensors, and the maverick encoder
    publishes both, but the loader reaches for the ``.bin`` first. Default the
    flag for the duration of the load, then put the method back.
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        yield
        return
    original = PreTrainedModel.from_pretrained.__func__

    def patched(cls, *args, **kwargs):
        kwargs.setdefault("use_safetensors", True)
        return original(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = classmethod(patched)
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(original)


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
                 min_mentions: int = 1, cache_dir: str | None = "data/interim/coref",
                 device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.alpha = alpha
        self.context_window = context_window
        self.min_mentions = min_mentions
        self.cache_dir = cache_dir
        self._model = None
        self._cast_cache: dict[str, list[Character]] = {}

    def _load(self):
        """Load the coreference model on a device that exists.

        Maverick defaults to CUDA and passes ``map_location=self.device`` into
        Lightning's ``load_from_checkpoint``, so on a CPU-only machine the
        checkpoint fails to deserialize. Choose the device here rather than
        inheriting that default.
        """
        if self._model is None:
            import torch
            from maverick import Maverick

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            with _prefer_safetensors():
                self._model = self._construct(Maverick, device)
        return self._model

    def _construct(self, Maverick, device):
        try:
            return Maverick(hf_name_or_path=self.model_name, device=device)
        except TypeError:
            # A maverick build whose constructor takes no device argument:
            # force CPU deserialization by defaulting torch.load instead.
            import functools
            import torch
            original = torch.load
            torch.load = functools.partial(original,
                                           map_location=torch.device(device))
            try:
                return Maverick(hf_name_or_path=self.model_name)
            finally:
                torch.load = original

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
