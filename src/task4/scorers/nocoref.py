"""Character and role layers WITHOUT coreference resolution.

These are not substitutes for L3 and L4. They are the condition Chaturvedi et
al. (2018) actually ran: characters identified from surface mentions, roles from
the dependency parse, and no coreference anywhere. Their system lifted
precision-at-1 from 0.558 to 0.637 in exactly this setting, and they named
missed coreference as a source of error.

Running them first is therefore worth something on its own. They establish the
pre-coreference point on this benchmark, so when the coreference model arrives,
the difference between L3n and L3 measures what coreference contributes rather
than what the whole pipeline contributes.

Neither layer downloads anything beyond the spaCy model.
"""
from __future__ import annotations

from ..registry import register
from .characters import Character, CharacterAlignmentScorer
from .roles import RoleProfile, SUBJECT_DEPS, OBJECT_DEPS

# Mentions that refer to people or animate agents often head a noun chunk.
# Without coreference, pronouns cannot be attached to anyone, so they are
# dropped rather than guessed at.
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "hers",
            "its", "their", "theirs", "i", "you", "we", "us", "me", "my",
            "this", "that", "these", "those", "who", "which", "what"}


class _SpacyCastMixin:
    """Characters as surface mention groups, keyed by the head noun's lemma."""

    def _parser(self):
        if getattr(self, "_nlp", None) is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model, disable=["lemmatizer"]
                                   if False else [])
        return self._nlp

    def cast(self, text: str) -> list[Character]:
        """Mention groups that behave like characters.

        Without coreference there is no entity typing either, so the filter is
        behavioural: keep a group if it is a proper noun, or if it ever appears
        as the subject of a verb. "the organization" and "citizens" survive;
        "safety" and "its mission" do not. Besides being closer to what a
        character is, this cuts a cast of ~59 noun chunks to a handful, and the
        alignment is quadratic in cast size.
        """
        if text in self._cast_cache:
            return self._cast_cache[text]
        doc = self._parser()(text)
        actors = {t.lemma_.lower() for t in doc
                  if t.dep_ in SUBJECT_DEPS and t.pos_ in ("NOUN", "PROPN")}
        groups: dict[str, Character] = {}
        for chunk in doc.noun_chunks:
            head = chunk.root
            key = head.lemma_.lower()
            if key in PRONOUNS or head.pos_ not in ("NOUN", "PROPN"):
                continue
            if self.actors_only and head.pos_ != "PROPN" and key not in actors:
                continue
            c = groups.setdefault(key, Character())
            c.mentions.append(chunk.text.lower())
            lo = max(0, chunk.start - self.context_window)
            hi = min(len(doc), chunk.end + self.context_window)
            c.context.extend(t.lemma_.lower() for t in doc[lo:hi] if not t.is_punct)
        out = [c for c in groups.values() if len(c.mentions) >= self.min_mentions]
        self._cast_cache[text] = out
        return out


@register("l3n_characters_nocoref")
class CharactersNoCoref(_SpacyCastMixin, CharacterAlignmentScorer):
    """L3's alignment over surface mentions. No coreference."""

    def __init__(self, spacy_model: str = "en_core_web_sm", alpha: float = 0.5,
                 context_window: int = 10, min_mentions: int = 1,
                 actors_only: bool = True):
        super().__init__(alpha=alpha, context_window=context_window,
                         min_mentions=min_mentions)
        self.spacy_model = spacy_model
        self.actors_only = actors_only
        self._nlp = None


@register("l4n_roles_nocoref")
class RolesNoCoref(_SpacyCastMixin, CharacterAlignmentScorer):
    """L4's role weighting over the same surface casts. No coreference."""

    def __init__(self, spacy_model: str = "en_core_web_sm", alpha: float = 0.5,
                 context_window: int = 10, min_mentions: int = 1,
                 agent_weight: float = 0.5, role_weight: float = 0.7,
                 actors_only: bool = True):
        super().__init__(alpha=alpha, context_window=context_window,
                         min_mentions=min_mentions)
        self.spacy_model = spacy_model
        self.actors_only = actors_only
        self.agent_weight = agent_weight
        self.role_weight = role_weight
        self._nlp = None
        self._role_cache: dict[str, dict[str, RoleProfile]] = {}

    def roles(self, text: str) -> dict[str, RoleProfile]:
        """Agent and patient verb lemmas per head-noun lemma, from the parse."""
        if text in self._role_cache:
            return self._role_cache[text]
        profiles: dict[str, RoleProfile] = {}
        for tok in self._parser()(text):
            if tok.head.pos_ != "VERB" or tok.pos_ not in ("NOUN", "PROPN"):
                continue
            key = tok.lemma_.lower()
            if key in PRONOUNS:
                continue
            prof = profiles.setdefault(key, RoleProfile())
            if tok.dep_ in SUBJECT_DEPS:
                prof.agent.add(tok.head.lemma_.lower())
            elif tok.dep_ in OBJECT_DEPS:
                prof.patient.add(tok.head.lemma_.lower())
        self._role_cache[text] = profiles
        return profiles

    def _profile_for(self, character: Character, profiles) -> RoleProfile:
        merged = RoleProfile()
        for mention in character.mentions:
            for word in mention.split():
                p = profiles.get(word)
                if p:
                    merged.agent |= p.agent
                    merged.patient |= p.patient
        return merged

    def score(self, anchor: str, candidate: str) -> float:
        left, right = self.cast(anchor), self.cast(candidate)
        if not left or not right:
            return 0.0
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        pa, pb = self.roles(anchor), self.roles(candidate)
        # Each character's profile is built once. Building it inside the pair
        # loop repeats the same work |cast| times and does not finish on a
        # 200-item split.
        prof_l = [self._profile_for(a, pa) for a in left]
        prof_r = [self._profile_for(b, pb) for b in right]
        w = np.array([[
            self.role_weight * pl.similarity(pr, self.agent_weight)
            + (1 - self.role_weight) * self.pair_weight(a, b)
            for b, pr in zip(right, prof_r)] for a, pl in zip(left, prof_l)])
        rows, cols = linear_sum_assignment(-w)
        return float(w[rows, cols].mean())
