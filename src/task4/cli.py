"""Hydra entry point: `python -m task4.cli scorer=l1_tfidf data=dev`."""
from __future__ import annotations
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .evaluate import compare, evaluate, load_split, save
from .registry import build

log = logging.getLogger(__name__)


def _run_one(cfg: DictConfig, name: str, params: dict):
    root = Path(hydra.utils.get_original_cwd()) / cfg.data.root
    triples = load_split(root, cfg.data.split)
    fit_on = load_split(root, cfg.data.fit_split) if cfg.data.get("fit_split") else triples

    scorer = build(name, **params)
    scorer.fit(fit_on)
    result = evaluate(scorer, triples, cfg.data.split,
                      n_boot=cfg.eval.n_boot, seed=cfg.eval.seed,
                      aspects=tuple(cfg.eval.aspects))
    out = Path(hydra.utils.get_original_cwd()) / cfg.eval.out_dir
    path = save(result, out, OmegaConf.to_container(cfg, resolve=True))
    log.info("%-16s %-5s n=%-4d acc=%.2f%% CI [%.2f, %.2f] -> %s",
             result.scorer, result.split, result.n, result.accuracy,
             result.ci_low, result.ci_high, path.name)
    return result


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.split == "test" and not cfg.eval.allow_test:
        raise SystemExit(
            "Refusing to touch the test split.\n"
            "The proposal commits to one held-out test run. When that run is the "
            "intended one, pass eval.allow_test=true explicitly."
        )
    raw = cfg.scorer.get("params")
    # An empty DictConfig is falsy, so `or {}` would swap in a plain dict here.
    params = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    params = params or {}
    result = _run_one(cfg, cfg.scorer.name, params)

    if cfg.eval.get("compare_against"):
        base = _run_one(cfg, cfg.eval.compare_against, {})
        verdict = compare(base, result)
        log.info("McNemar %s vs %s: delta=%+.2f b01=%d b10=%d p=%.4f (%s)",
                 verdict["a"], verdict["b"], verdict["delta"],
                 verdict["b01"], verdict["b10"], verdict["p_value"], verdict["test"])


if __name__ == "__main__":
    main()
