"""Significance and interval estimates. One place, so every layer is judged alike."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.2f}% [{self.lo:.2f}, {self.hi:.2f}]"


def accuracy(correct: list[bool]) -> float:
    return 100.0 * sum(correct) / len(correct) if correct else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval; stable at small n, unlike the normal approximation."""
    if n == 0:
        return Interval(0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return Interval(100 * p, 100 * (centre - half), 100 * (centre + half))


def bootstrap(correct: list[bool], n_boot: int = 2000, seed: int = 0) -> Interval:
    """Percentile bootstrap over items (Jurafsky & Martin, Section 4.11.1)."""
    rng = random.Random(seed)
    n = len(correct)
    if n == 0:
        return Interval(0.0, 0.0, 0.0)
    draws = sorted(
        100.0 * sum(correct[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    return Interval(accuracy(correct), draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)])


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Paired test on the same items. Exact binomial below 25 discordant pairs."""
    if len(a_correct) != len(b_correct):
        raise ValueError("McNemar requires predictions over the same items")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "statistic": 0.0, "p_value": 1.0, "test": "none"}
    if n < 25:
        k = min(b01, b10)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
        return {"b01": b01, "b10": b10, "statistic": float(k), "p_value": p, "test": "exact"}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b01": b01, "b10": b10, "statistic": chi2, "p_value": p, "test": "chi2_cc"}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Unpaired comparison across disjoint strata. Not a substitute for McNemar."""
    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p_value": 1.0, "diff": 0.0}
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    return {"z": z, "p_value": math.erfc(abs(z) / math.sqrt(2)), "diff": 100 * (p1 - p2)}


def describe(values: list[float]) -> dict:
    """Mean and sample SD — the instructor asked for SD alongside reported figures."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": min(values), "max": max(values)}
