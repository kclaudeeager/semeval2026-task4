"""Decompose a story into the three components the annotation scheme uses.

The annotators judged similarity along abstract theme, course of action and
outcome, and their free-text comments show them abstracting each story to a
sentence of gist before comparing. The task winner decomposed stories the same
way. This module does it without an LLM, so it runs on a laptop.

Each component is a span of the original text, not a generated paraphrase:

  theme    the whole summary. A sentence encoder already abstracts somewhat,
           and the whole-summary embedding is the strongest single layer in
           this project, which is itself evidence that it proxies theme.
  action   the sentences carrying the most narrative action, measured by
           density of non-copular past-tense verbs with a subject.
  outcome  the closing sentences, where a plot summary puts its resolution.

Extraction is deliberately crude. The point is to find out whether components
extracted this cheaply carry the signal the annotation scheme says they should;
`scripts/validate_components.py` answers that against the aspect flags before
any accuracy claim is made.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

COPULAS = {"be", "is", "are", "was", "were", "seem", "appear", "become", "remain"}


@dataclass(frozen=True)
class Components:
    theme: str
    action: str
    outcome: str

    def as_dict(self) -> dict[str, str]:
        return {"theme": self.theme, "action": self.action, "outcome": self.outcome}


def sentences(text: str) -> list[str]:
    out = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return out or [text.strip()]


def action_density(sent, nlp) -> float:
    """Share of tokens that are action verbs with a grammatical subject."""
    doc = nlp(sent)
    if not len(doc):
        return 0.0
    hits = sum(1 for t in doc
               if t.pos_ == "VERB" and t.lemma_.lower() not in COPULAS
               and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children))
    return hits / len(doc)


def decompose(text: str, nlp=None, outcome_frac: float = 0.25,
              action_top: float = 0.5) -> Components:
    """Split one summary into theme / action / outcome spans."""
    sents = sentences(text)
    n = len(sents)

    # Outcome: the closing sentences, at least one.
    k = max(1, round(n * outcome_frac))
    outcome = " ".join(sents[-k:])

    # Action: the most action-dense sentences before the outcome, in order.
    body = sents[:-k] or sents
    if nlp is not None and len(body) > 1:
        scored = sorted(range(len(body)), key=lambda i: -action_density(body[i], nlp))
        keep = sorted(scored[:max(1, round(len(body) * action_top))])
        action = " ".join(body[i] for i in keep)
    else:
        action = " ".join(body)

    return Components(theme=text.strip(), action=action, outcome=outcome)
