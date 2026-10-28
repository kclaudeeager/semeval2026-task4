"""What effect size can this design actually detect?

Run: python scripts/power_analysis.py
Answers a question that is cheaper to ask now than on the day of the test run:
if character and role structure helps by the 3-8 points the design hoped for,
would any of the planned tests see it?
"""
from __future__ import annotations
import math

ALPHA, POWER = 0.05, 0.80


def _z(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


Z_A, Z_B = _z(1 - ALPHA / 2), _z(POWER)


def mde_two_proportion(n1: int, n2: int, p_base: float = 0.65) -> float:
    """Smallest gap between two independent strata that would reach significance."""
    for step in range(1, 400):
        d = step / 1000
        p1, p2 = p_base, min(0.999, p_base + d)
        pbar = (p1 * n1 + p2 * n2) / (n1 + n2)
        se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
        se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        if abs(p2 - p1) >= Z_A * se0 + Z_B * se1:
            return 100 * d
    return float("nan")


def mde_mcnemar(n: int, discordance: float) -> float:
    """Smallest paired accuracy gain detectable, given how often two layers differ."""
    n_disc = n * discordance
    for step in range(1, 600):
        d = step / 2000
        net = d * n
        if net > n_disc:
            break
        psi = 0.5 + net / (2 * n_disc)
        if psi >= 1:
            break
        need = (Z_A * 0.5 + Z_B * math.sqrt(psi * (1 - psi))) ** 2 / (psi - 0.5) ** 2
        if need <= n_disc:
            return 100 * d
    return float("nan")


STRATA = {                      # measured from the release, see notebook section 8
    "test outcomes":          (79, 133),
    "test course_of_actions": (188, 32),
    "dev  outcomes":          (40, 71),
    "dev  course_of_actions": (88, 19),
}

if __name__ == "__main__":
    print(f"alpha={ALPHA}, power={POWER}\n")
    print("Comparing two strata against each other (unpaired):")
    for name, (a, b) in STRATA.items():
        print(f"  {name:<24} n={a:>3} vs {b:>3}   needs {mde_two_proportion(a, b):5.1f} pts")

    print("\nComparing two layers on the same items (McNemar, paired):")
    print(f"  {'':<24} {'10%':>8} {'20%':>8} {'30%':>8}   <- discordance")
    for name, n in [("test, all items", 400), ("dev, all items", 200),
                    ("test outcomes none_marked", 133), ("test outcomes all_marked", 79)]:
        row = "".join(f"{mde_mcnemar(n, r):8.1f}" for r in (0.10, 0.20, 0.30))
        print(f"  {name:<24} {row}   pts")

    print("\nThe design doc expects 3-8 points in the marked stratum, ~0 elsewhere.")
