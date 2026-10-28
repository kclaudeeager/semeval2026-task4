"""Summarise every dev result and run the comparisons that matter.

Usage: python scripts/compare_runs.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

from task4.stats import mcnemar


def load(split: str = "dev") -> dict[str, dict]:
    out = {}
    for p in sorted(Path("reports/results").glob(f"{split}__*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        params = (d.get("config", {}).get("scorer", {}) or {}).get("params", {}) or {}
        key = d["scorer"]
        if "actors_only" in params:
            key += "/all-nouns" if params["actors_only"] is False else "/actors"
        out[key] = d
    return out


def phi(a: list[bool], b: list[bool]) -> float:
    """Correlation between two layers' ERROR vectors; positive = shared failures."""
    A, B = [not x for x in a], [not x for x in b]
    n11 = sum(x and y for x, y in zip(A, B))
    n10 = sum(x and not y for x, y in zip(A, B))
    n01 = sum((not x) and y for x, y in zip(A, B))
    n00 = sum((not x) and (not y) for x, y in zip(A, B))
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else 0.0


if __name__ == "__main__":
    R = load()
    print("DEV RESULTS")
    for k in sorted(R, key=lambda k: -R[k]["accuracy"]):
        d = R[k]
        cv = " (cross-validated)" if d.get("meta", {}).get("cv_folds") else ""
        print(f"  {k:<38} {d['accuracy']:6.2f}  [{d['ci_low']:5.1f},{d['ci_high']:5.1f}]{cv}")

    pairs = [("l3n_characters_nocoref/all-nouns", "l3x_characters_embed/all-nouns"),
             ("l3n_characters_nocoref/actors", "l3x_characters_embed/actors"),
             ("l4n_roles_nocoref/actors", "l4x_roles_embed/all-nouns"),
             ("l2_e5", "l3x_characters_embed/all-nouns")]
    print("\nTHE COMPARISON THAT ISOLATES ONE VARIABLE")
    for a, b in pairs:
        if a in R and b in R:
            m = mcnemar(R[a]["correct"], R[b]["correct"])
            delta = R[b]["accuracy"] - R[a]["accuracy"]
            verdict = "SIGNIFICANT" if m["p_value"] < 0.05 else "not significant"
            print(f"  {a:<34}\n    -> {b:<32} {delta:+6.2f} pts  p={m['p_value']:.3f}  {verdict}")

    if "l1_tfidf" in R:
        print("\nSTILL LEXICAL? error correlation with TF-IDF (was +0.27 for L3n)")
        for k in sorted(R):
            if k.startswith(("l3", "l4")):
                print(f"  {k:<38} phi={phi(R['l1_tfidf']['correct'], R[k]['correct']):+.3f}")
