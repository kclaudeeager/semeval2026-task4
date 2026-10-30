"""The evaluation harness. Every layer is judged by this one path."""
from __future__ import annotations
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .dataset import Triple, load_triples
from .scorers.base import Scorer
from .stats import Interval, bootstrap, mcnemar, two_proportion_z, wilson
from .strata import Stratum, partition


@dataclass
class Result:
    scorer: str
    split: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    wilson_low: float
    wilson_high: float
    strata: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    correct: list[bool] = field(default_factory=list)
    fitted: dict = field(default_factory=dict)
    margins: list[float] = field(default_factory=list)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unversioned"


def evaluate(scorer: Scorer, triples: list[Triple], split: str,
             n_boot: int = 2000, seed: int = 0,
             aspects: tuple[str, ...] = ("outcomes", "course_of_actions")) -> Result:
    """Accuracy with intervals, plus accuracy inside each unanimous stratum."""
    labelled = [t for t in triples if t.a_is_closer is not None]
    if not labelled:
        raise ValueError(f"split {split!r} has no gold labels")
    pairs = scorer.score_pairs(labelled)
    preds = [a >= b for a, b in pairs]
    correct = [p == bool(t.a_is_closer) for p, t in zip(preds, labelled)]
    # Signed margin: positive when the gold candidate scored higher.
    margins = [(a - b) if t.a_is_closer else (b - a)
               for (a, b), t in zip(pairs, labelled)]

    boot: Interval = bootstrap(correct, n_boot=n_boot, seed=seed)
    wil: Interval = wilson(sum(correct), len(correct))

    strata: dict = {}
    for aspect in aspects:
        idx = partition(labelled, aspect)
        block: dict = {}
        for stratum in (Stratum.ALL_MARKED, Stratum.NONE_MARKED, Stratum.SPLIT):
            rows = idx[stratum]
            sub = [correct[i] for i in rows]
            block[stratum.value] = {
                "n": len(sub),
                "accuracy": (100.0 * sum(sub) / len(sub)) if sub else None,
            }
        a, b = idx[Stratum.ALL_MARKED], idx[Stratum.NONE_MARKED]
        if a and b:
            block["contrast"] = two_proportion_z(
                sum(correct[i] for i in a), len(a),
                sum(correct[i] for i in b), len(b))
            block["contrast"]["note"] = (
                "unpaired across disjoint strata: suggestive, not a system comparison")
        strata[aspect] = block

    return Result(
        fitted=scorer.fitted_params(),
        scorer=scorer.name, split=split, n=len(correct),
        accuracy=boot.point, ci_low=boot.lo, ci_high=boot.hi,
        wilson_low=wil.lo, wilson_high=wil.hi, strata=strata,
        correct=correct, margins=margins,
        meta={"git_sha": _git_sha(), "python": platform.python_version(),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
              "n_boot": n_boot},
    )


def evaluate_cv(factory, triples: list[Triple], split: str, k: int = 5,
                seed: int = 0, **kwargs) -> Result:
    """K-fold evaluation for layers whose fit() learns parameters.

    Fitting weights on a split and scoring the same split is circular: the
    estimate is optimistic by construction. Here each fold is predicted by a
    scorer fitted on the other folds, so no item contributes to the parameters
    used to score it. `factory` returns a fresh, unfitted scorer.
    """
    import random

    labelled = [t for t in triples if t.a_is_closer is not None]
    idx = list(range(len(labelled)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]

    preds: dict[int, bool] = {}
    per_fold: list[dict] = []
    for f in range(k):
        held = set(folds[f])
        train = [labelled[i] for i in idx if i not in held]
        test_rows = [labelled[i] for i in folds[f]]
        model = factory()
        model.fit(train)
        for i, p in zip(folds[f], model.predict(test_rows)):
            preds[i] = p
        per_fold.append(model.fitted_params())

    ordered = [preds[i] for i in range(len(labelled))]
    frozen = _Frozen(factory().name, ordered)
    out = evaluate(frozen, labelled, split, **kwargs)
    out.meta["cv_folds"] = k
    out.fitted = {"per_fold": per_fold}
    return out


class _Frozen(Scorer):
    """Replays precomputed predictions so cross-validated runs reuse the harness."""

    def __init__(self, name: str, preds: list[bool]):
        self.name = name
        self._preds = preds

    def score(self, anchor: str, candidate: str) -> float:  # pragma: no cover
        raise NotImplementedError("frozen predictions carry no score function")

    def predict(self, triples: list[Triple]) -> list[bool]:
        return list(self._preds)

    def score_pairs(self, triples: list[Triple]) -> list[tuple[float, float]]:
        """Margins are not recoverable once folds are stitched together, so
        emit a unit pseudo-margin in the decided direction."""
        return [(1.0, 0.0) if p else (0.0, 1.0) for p in self._preds]


def compare(a: Result, b: Result) -> dict:
    """Paired comparison of two layers on the same split."""
    if a.split != b.split or a.n != b.n:
        raise ValueError("paired comparison requires the same split and item count")
    out = mcnemar(a.correct, b.correct)
    out.update({"a": a.scorer, "b": b.scorer, "split": a.split,
                "delta": b.accuracy - a.accuracy})
    return out


def params_tag(config: dict | None) -> str:
    """Short, stable tag for a run's scorer parameters.

    Without this, two runs of the same layer under different parameters write to
    the same filename and the second silently destroys the first. That happened
    once; hence the tag.
    """
    import hashlib

    params = ((config or {}).get("scorer") or {}).get("params") or {}
    if not params:
        return "default"
    blob = json.dumps(params, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def save(result: Result, out_dir: str | Path, config: dict | None = None) -> Path:
    """One JSON per run, carrying the resolved config, so results are self-describing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["config"] = config or {}
    payload["params_tag"] = params_tag(config)
    path = out_dir / f"{result.split}__{result.scorer}__{payload['params_tag']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_result(path: str | Path) -> Result:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("config", None)
    return Result(**data)


def split_path(root: str | Path, split: str) -> Path:
    """Where each split's labelled file lives in the released archive."""
    root = Path(root)
    return {
        "sample": root / "sample" / "sample_track_a.jsonl",
        "dev": root / "dev" / "dev_track_a.jsonl",
        "test": root / "test" / "labels" / "test_track_a_labels.jsonl",
    }[split]


def load_split(root: str | Path, split: str) -> list[Triple]:
    return load_triples(split_path(root, split))
