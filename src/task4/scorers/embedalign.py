"""Structure as alignment, meaning as comparison.

L3n and L4n align characters structurally and then compare the aligned pairs
with Jaccard over word sets, which is lexical matching wearing a structural
coat. Retellings of one story rarely share literal words, so that inner step
throws away the thing the alignment just earned.

These layers keep the alignment exactly as it is and swap the inner comparison
for embedding cosine: each character becomes a short document (the sentences it
appears in), that document is embedded, and aligned characters are compared by
cosine. Structure decides *what* to compare; meaning decides *how similar* it is.

This is the shape of the one design in the literature with a positive result on
this genre (Chaturvedi et al. 2018: align characters, then compare them), and
it is not what the task's one graph entrant did -- StoryNet encoded a graph with
a GNN and compared the resulting vectors.
"""
from __future__ import annotations

from ..registry import register
from .characters import Character
from .nocoref import CharactersNoCoref, RolesNoCoref


class _EmbeddingPairs:
    """Character-level similarity from sentence embeddings rather than word sets."""

    def _encoder(self):
        if getattr(self, "_enc", None) is None:
            from sentence_transformers import SentenceTransformer

            self._enc = SentenceTransformer(self.embed_model)
        return self._enc

    def character_doc(self, character: Character) -> str:
        """The character as text: its mentions plus the words around them."""
        return f"{character.head}: {' '.join(character.context)}"

    def warm(self, *casts) -> None:
        """Embed every uncached character in one batched call.

        Encoding one short document at a time is dominated by per-call
        overhead; a cast of ~60 characters over 600 stories is ~35k calls.
        Batching the whole cast turns a forty-minute run into a few minutes.
        """
        cache = self.__dict__.setdefault("_vec_cache", {})
        todo = []
        for cast in casts:
            for c in cast:
                key = self.character_doc(c)
                if key not in cache and key not in todo:
                    todo.append(key)
        if not todo:
            return
        vecs = self._encoder().encode(
            [self.embed_prefix + k for k in todo],
            normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=False)
        cache.update(zip(todo, vecs))

    def _vector(self, character: Character):
        key = self.character_doc(character)
        cache = self.__dict__.setdefault("_vec_cache", {})
        if key not in cache:
            self.warm([character])
        return cache[key]

    def pair_weight(self, x: Character, y: Character) -> float:
        import numpy as np

        return float(np.dot(self._vector(x), self._vector(y)))

    def score(self, anchor: str, candidate: str) -> float:
        self.warm(self.cast(anchor), self.cast(candidate))
        return super().score(anchor, candidate)


@register("l3x_characters_embed")
class CharactersEmbedAligned(_EmbeddingPairs, CharactersNoCoref):
    """Structural alignment, embedding comparison. No coreference."""

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embed_prefix: str = "", batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.embed_model = embed_model
        self.embed_prefix = embed_prefix
        self.batch_size = batch_size
        self._enc = None


@register("l4x_roles_embed")
class RolesEmbedAligned(_EmbeddingPairs, RolesNoCoref):
    """As above, with the role term kept: role overlap plus embedded context."""

    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embed_prefix: str = "", batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.embed_model = embed_model
        self.embed_prefix = embed_prefix
        self.batch_size = batch_size
        self._enc = None
