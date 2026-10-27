"""L4: agent and patient proxies from the dependency parse.

No maintained PropBank SRL system exists (AllenNLP archived 2022; LLM-prompted
SRL reaches 48.6% argument match against 88.3% for humans), so roles come from
syntax: verbs where a character is the subject proxy for agent actions, verbs
where it is the object proxy for patient roles.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from ..registry import register
from .base import Scorer
from .characters import CharacterAlignmentScorer, _jaccard

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}
OBJECT_DEPS = {"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"}


@dataclass
class RoleProfile:
    """Which verbs a character acts in, split by the role it plays."""
    agent: set[str] = field(default_factory=set)
    patient: set[str] = field(default_factory=set)

    def similarity(self, other: "RoleProfile", agent_weight: float = 0.5) -> float:
        return (agent_weight * _jaccard(self.agent, other.agent)
                + (1 - agent_weight) * _jaccard(self.patient, other.patient))


@register("l4_roles")
class RoleAlignmentScorer(CharacterAlignmentScorer):
    """L3's alignment, re-weighted on shared role-lemma pairs instead of mentions."""

    def __init__(self, spacy_model: str = "en_core_web_sm", agent_weight: float = 0.5,
                 role_weight: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.spacy_model = spacy_model
        self.agent_weight = agent_weight
        self.role_weight = role_weight
        self._nlp = None
        self._role_cache: dict[str, dict[str, RoleProfile]] = {}

    def _parser(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model, disable=["ner"])
        return self._nlp

    def roles(self, text: str) -> dict[str, RoleProfile]:
        """Role profile per mention head, keyed by lowercase surface form."""
        if text in self._role_cache:
            return self._role_cache[text]
        profiles: dict[str, RoleProfile] = {}
        for tok in self._parser()(text):
            if tok.head.pos_ != "VERB":
                continue
            key = tok.lemma_.lower()
            prof = profiles.setdefault(key, RoleProfile())
            if tok.dep_ in SUBJECT_DEPS:
                prof.agent.add(tok.head.lemma_.lower())
            elif tok.dep_ in OBJECT_DEPS:
                prof.patient.add(tok.head.lemma_.lower())
        self._role_cache[text] = profiles
        return profiles

    def _profile_for(self, character, profiles: dict[str, RoleProfile]) -> RoleProfile:
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
        w = np.array([[
            self.role_weight * self._profile_for(a, pa).similarity(
                self._profile_for(b, pb), self.agent_weight)
            + (1 - self.role_weight) * self.pair_weight(a, b)
            for b in right] for a in left])
        rows, cols = linear_sum_assignment(-w)
        return float(w[rows, cols].mean())
