"""Name -> Scorer lookup, so configs name a layer and never import one."""
from __future__ import annotations
from typing import Callable

from .scorers.base import Scorer

_REGISTRY: dict[str, Callable[..., Scorer]] = {}


def register(name: str):
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def build(name: str, **kwargs) -> Scorer:
    if name not in _REGISTRY:
        _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def _load_all() -> None:
    from .scorers import (lexical, embeddings, characters, roles,  # noqa: F401
                          combined, nocoref)
