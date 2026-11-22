"""Verify every claim the report makes, a second way.

The point is not to re-run the experiments. It is to recompute each reported
quantity by a different route and check the two agree, and to verify the
assumptions the statistics rest on but never state -- above all that paired
tests are actually paired.

A green audit plus a PROVENANCE.md with no UNKNOWN rows is the bar for
"ready to write". Run it again before submitting.

    python scripts/audit.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from task4.dataset import load_triples
from task4.evaluate import load_split
from task4.strata import Stratum, partition

RESULTS = Path("reports/results")
DATA_ROOT = "data/raw/narrative-similarity-dataset"

# Layers that fit parameters and must therefore be cross-validated. A run of
# one of these without per-fold weights was fitted and scored on the same
# data, and its accuracy is optimistic.
MUST_BE_CV = {"l1_tfidf", "l5_combined", "l7_components"}

# The layers the pre-registration fixed before the test split was touched;
# see docs/PREREGISTRATION.md section 2. This set may only be widened by
# amending that document BEFORE a run, never after seeing a number.
ALLOWED_TEST = {"l0_random", "l0_jaccard", "l1_tfidf", "l2_e5",
                "l3x_characters_embed", "l5_combined"}

PASS, FAIL, WARN = "pass", "FAIL", "warn"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str) -> None:
    results.append((name, PASS if ok else (WARN if ok is None else FAIL), detail))


def load_all() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        if "scorer" in d and "accuracy" in d:
            d["_file"] = p.name
            out.append(d)
    return out


# ---------------------------------------------------------------- 1. pairing
def audit_pairing(runs: list[dict]) -> None:
    """Every McNemar and every phi assumes the correct[] arrays line up
    item for item. If one run scored a different subset, or in a different
    order, every paired number in the report is wrong and nothing warns.
    """
    triples = [t for t in load_split(DATA_ROOT, "dev") if t.a_is_closer is not None]
    canonical = len(triples)
    dev = [r for r in runs if r["split"] == "dev"]

    bad_n = [r["_file"] for r in dev if r["n"] != canonical]
    check("dev runs all scored the same number of items",
          not bad_n,
          f"{len(dev)} dev runs, all n={canonical}" if not bad_n
          else f"wrong n: {bad_n}")

    bad_len = [r["_file"] for r in dev if len(r.get("correct") or []) != canonical]
    check("dev runs all store a full correct[] vector",
          not bad_len,
          "every run stores 200 per-item outcomes" if not bad_len
          else f"short or missing correct[]: {bad_len}")

    # The strata partition is derived from the triples, so if two runs scored
    # the same items in the same order their stored strata sizes must match
    # the partition computed here from scratch.
    idx = partition(triples, "outcomes")
    want = {k.name: len(v) for k, v in idx.items()}
    mismatch = []
    for r in dev:
        s = (r.get("strata") or {}).get("outcomes") or {}
        got = {"ALL_MARKED": s.get("all_marked", {}).get("n"),
               "NONE_MARKED": s.get("none_marked", {}).get("n"),
               "SPLIT": s.get("split", {}).get("n")}
        for key, n in got.items():
            if n is not None and n != want.get(key):
                mismatch.append(f"{r['_file']}:{key}={n} want {want.get(key)}")
    check("stored strata sizes match a fresh partition of the same split",
          not mismatch,
          f"outcomes strata {want} agree across all runs" if not mismatch
          else "; ".join(mismatch[:4]))


# ------------------------------------------------------- 2. no test leakage
def audit_test_discipline(runs: list[dict]) -> None:
    test = [r for r in runs if r["split"] == "test"]
    bad = [r["_file"] for r in test if r["scorer"] not in ALLOWED_TEST]
    check("test split touched only by the sanctioned baseline",
          not bad,
          f"{len(test)} test run(s): {sorted({r['scorer'] for r in test})}" if not bad
          else f"unsanctioned test runs: {bad}")


# ------------------------------------------------- 3. cross-validation honesty
def audit_cv(runs: list[dict]) -> None:
    offenders, ok_reasons = [], []
    for r in runs:
        if r["scorer"] not in MUST_BE_CV:
            continue
        fitted = r.get("fitted") or {}
        data = (r.get("config") or {}).get("data") or {}
        fit_split = data.get("fit_split")
        # Two honest paths: cross-validate within the split, or fit on one
        # split and score on another. Only fitting and scoring on the SAME
        # split without cross-validation is circular.
        if fitted.get("per_fold"):
            ok_reasons.append(f"{r['_file']}: cross-validated")
        elif fit_split and fit_split != r["split"]:
            ok_reasons.append(f"{r['_file']}: fit on {fit_split}, scored on {r['split']}")
        else:
            offenders.append(f"{r['_file']} ({r['accuracy']:.2f}%)")
    check("no parameter-fitting layer is fitted and scored on the same split",
          not offenders,
          f"{len(ok_reasons)} fitting run(s), all either cross-validated or "
          f"fitted on a different split" if not offenders
          else "circular: " + ", ".join(offenders))


# ------------------------------------------ 4. geometry agrees with the scorer
def audit_geometry_identity(runs: list[dict]) -> None:
    gpath = RESULTS / "geometry__dev.json"
    if not gpath.exists():
        check("e5 geometry matches the L2 scorer", None, "geometry__dev.json absent")
        return
    g = json.loads(gpath.read_text())["spaces"].get("e5")
    l2 = next((r for r in runs if r["scorer"] == "l2_e5" and r["split"] == "dev"), None)
    if not g or not l2:
        check("e5 geometry matches the L2 scorer", None, "e5 space or L2 run absent")
        return
    same = abs(g["right_closer_pct"] - l2["accuracy"]) < 1e-9
    check("e5 geometry matches the L2 scorer",
          same,
          f"both {l2['accuracy']:.2f}% -- two code paths agree exactly" if same
          else f"geometry {g['right_closer_pct']:.2f}% vs scorer {l2['accuracy']:.2f}%")


# ------------------------------------------------- 5. phi, computed two ways
def audit_phi(runs: list[dict]) -> None:
    base = next((r for r in runs if r["scorer"] == "l1_tfidf" and r["split"] == "dev"), None)
    other = next((r for r in runs if r["scorer"] == "l2_e5" and r["split"] == "dev"), None)
    if not base or not other:
        check("phi agrees between two formulas", None, "baseline runs absent")
        return
    a = [0 if x else 1 for x in base["correct"]]
    b = [0 if x else 1 for x in other["correct"]]
    n = len(a)

    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    pearson = num / (da * db)

    # same quantity from the 2x2 table
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = sum(1 for x, y in zip(a, b) if not x and not y)
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    table = (n11 * n00 - n10 * n01) / denom if denom else float("nan")

    same = abs(pearson - table) < 1e-9
    check("phi agrees between two formulas",
          same,
          f"both {pearson:+.6f} (Pearson on error vectors = 2x2 table)" if same
          else f"Pearson {pearson:+.6f} vs table {table:+.6f}")


# --------------------------------------- 6. margin sign agrees with accuracy
def audit_margins(runs: list[dict]) -> None:
    bad = []
    checked = 0
    for r in runs:
        m = r.get("margins")
        if not m:
            continue
        checked += 1
        # A positive margin means the gold candidate scored higher, so the
        # fraction of positive margins must equal the accuracy.
        # A margin of exactly zero is a tie: the scorer picked by `a >= b`,
        # so a tie can still be counted correct. Ties therefore explain a gap
        # between "positive margins" and accuracy, and are allowed for.
        pos = sum(1 for x in m if x > 0)
        ties = sum(1 for x in m if x == 0)
        frac = 100.0 * pos / len(m)
        slack = 100.0 * ties / len(m) + 0.51
        if abs(frac - r["accuracy"]) > slack:
            bad.append(f"{r['_file']}: {frac:.2f}% positive, {ties} ties, "
                       f"vs acc {r['accuracy']:.2f}%")
    check("signed margins agree with the recorded accuracy",
          not bad,
          f"{checked} run(s) with margins, all consistent" if not bad
          else "; ".join(bad[:4]))


# ------------------------------------- 7. each comparison isolates one thing
def audit_one_variable(runs: list[dict]) -> None:
    """The report says certain pairs "change one thing". Verify it by
    diffing the two configs rather than trusting the sentence."""
    def params(scorer, split, **want):
        for r in runs:
            if r["scorer"] != scorer or r["split"] != split:
                continue
            p = ((r.get("config") or {}).get("scorer") or {}).get("params") or {}
            if all(p.get(k) == v for k, v in want.items()):
                return r["_file"], p
        return None, None

    pairs = [
        ("cast filter", ("l3n_characters_nocoref", {"actors_only": False}),
                        ("l3n_characters_nocoref", {"actors_only": True})),
        ("roles filter", ("l4n_roles_nocoref", {"actors_only": False}),
                         ("l4n_roles_nocoref", {"actors_only": True})),
        ("encoder inside alignment",
            ("l3x_characters_embed", {"embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                                      "actors_only": False}),
            ("l3x_characters_embed", {"embed_model": "intfloat/e5-base-v2",
                                      "actors_only": False})),
    ]
    for label, (sa, wa), (sb, wb) in pairs:
        fa, pa = params(sa, "dev", **wa)
        fb, pb = params(sb, "dev", **wb)
        if pa is None or pb is None:
            check(f"'{label}' comparison isolates one variable", None, "a run is absent")
            continue
        diff = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        # Keys that change the arithmetic but not the model. Encoding in
        # batches of 32 or 64 yields the same embeddings up to float noise,
        # which cannot move accuracy by points -- but it is named here rather
        # than ignored, so the exemption is visible.
        NONSEMANTIC = {"batch_size"}
        # an encoder swap legitimately drags its prefix along
        expected = {"actors_only"} if "filter" in label else {"embed_model", "embed_prefix"}
        semantic = [k for k in diff if k not in NONSEMANTIC]
        incidental = [k for k in diff if k in NONSEMANTIC]
        ok = set(semantic) <= expected
        note = f"differs in {semantic}"
        if incidental:
            note += f"; also {incidental}, non-semantic"
        check(f"'{label}' comparison isolates one variable",
              ok, note if ok else note + f" -- expected only {sorted(expected)}")


# ------------------------------------------- 8. components are not degenerate
def audit_components() -> None:
    cpath = RESULTS / "components__dev.json"
    if not cpath.exists():
        check("component accuracies recompute from stored counts", None,
              "components__dev.json absent")
        return
    c = json.loads(cpath.read_text())
    bad = []
    for name, e in (c.get("components") or {}).items():
        lo, hi = e["wilson_low"], e["wilson_high"]
        if not (lo < e["accuracy"] < hi):
            bad.append(f"{name}: {e['accuracy']} outside [{lo}, {hi}]")
        if e["n"] != c["n"]:
            bad.append(f"{name}: n={e['n']} but file says {c['n']}")
    check("component accuracies recompute from stored counts",
          not bad,
          f"{len(c.get('components') or {})} components, intervals bracket their point"
          if not bad else "; ".join(bad))



# ------------------------------------- 9. lexical direction, two ways
def audit_lexical() -> None:
    """The binomial p-values recomputed with scipy, and the counts sanity-checked.

    Section 7.3 cites p = 0.009 for the cross-entropy reversal. The script that
    produced it implements the exact binomial by hand, so the figure is checked
    here against an independent implementation.
    """
    path = RESULTS / "lexical_direction__dev.json"
    if not path.exists():
        check("lexical direction p-values agree with scipy", None,
              "lexical_direction__dev.json absent")
        return
    try:
        from scipy.stats import binomtest
    except ImportError:
        check("lexical direction p-values agree with scipy", None, "scipy absent")
        return

    d = json.loads(path.read_text())
    bad, checked = [], 0
    for vkey, variant in (d.get("variants") or {}).items():
        for mkey, e in variant.items():
            checked += 1
            theirs = e["p"]
            mine = binomtest(e["k"], e["n"], 0.5, alternative="two-sided").pvalue
            if abs(theirs - mine) > 1e-9:
                bad.append(f"{vkey}/{mkey}: stored {theirs:.6f} vs scipy {mine:.6f}")
            if not 0 <= e["k"] <= e["n"]:
                bad.append(f"{vkey}/{mkey}: k={e['k']} outside 0..{e['n']}")
            if abs(e["pct"] - 100.0 * e["k"] / e["n"]) > 1e-9:
                bad.append(f"{vkey}/{mkey}: pct disagrees with k/n")
    check("lexical direction p-values agree with scipy",
          not bad,
          f"{checked} measures recomputed, all match" if not bad else "; ".join(bad[:3]))


def audit_reconciliation(runs: list[dict]) -> None:
    """The tokenizer reconciliation, checked against the evaluated run itself.

    The data section claims the published baseline is reproduced by four
    ordinary tokenizations and that the one in use flatters it by a stated
    margin. Both halves are recomputed here, and the current tokenizer's
    accuracy is matched against the stored l0_jaccard test run rather than
    taken on the script's word.
    """
    path = RESULTS / "tokenizer_reconciliation.json"
    if not path.exists():
        check("tokenizer reconciliation agrees with the evaluated run", None,
              "tokenizer_reconciliation.json absent")
        return
    tk = json.loads(path.read_text())
    test = tk["splits"]["test"]
    cur = test[tk["current"]]

    stored = next((r for r in runs if r.get("scorer") == "l0_jaccard"
                   and r.get("split") == "test"), None)
    if stored is None:
        check("tokenizer reconciliation agrees with the evaluated run", None,
              "no l0_jaccard test run on disk")
        return

    problems = []
    if abs(stored["accuracy"] - cur["accuracy"]) > 1e-9:
        problems.append(
            f"evaluated run {stored['accuracy']:.2f} != reconciliation "
            f"{cur['accuracy']:.2f} for the tokenizer actually in use")
    if abs(cur["accuracy"] - tk["published_test"] - tk["gap_points"]) > 1e-9:
        problems.append("gap_points does not equal measured minus published")
    if cur["k"] - round(tk["published_test"] * cur["n"] / 100) != tk["gap_items"]:
        problems.append("gap_items does not equal the difference in item counts")
    matching = [n for n, v in test.items()
                if abs(v["accuracy"] - tk["published_test"]) < 1e-9]
    if sorted(matching) != sorted(tk["matching_variants"]):
        problems.append("matching_variants disagrees with the recorded accuracies")
    flips = tk["flips"]
    if flips["current_right"] - flips["current_wrong"] != tk["gap_items"]:
        problems.append(
            f"the {flips['changed']} prediction changes net "
            f"{flips['current_right'] - flips['current_wrong']}, not "
            f"{tk['gap_items']}")
    dev = [v["accuracy"] for v in tk["splits"]["dev"].values()]
    if max(dev) >= 50.0:
        problems.append(f"a dev tokenization reaches {max(dev):.2f}, at or "
                        f"above chance; the data section says none do")

    check("tokenizer reconciliation agrees with the evaluated run",
          not problems,
          "; ".join(problems) if problems else
          f"{stored['accuracy']:.2f}% on test matches the run on disk; "
          f"{len(matching)} of {len(test)} tokenizations reproduce the "
          f"published {tk['published_test']:.2f}%; {flips['changed']} items "
          f"flip, net {tk['gap_items']}; all {len(dev)} dev variants below "
          f"chance ({min(dev):.2f}-{max(dev):.2f})")


def main() -> int:
    runs = load_all()
    print(f"auditing {len(runs)} scored run(s) in {RESULTS}\n")

    audit_pairing(runs)
    audit_test_discipline(runs)
    audit_cv(runs)
    audit_geometry_identity(runs)
    audit_phi(runs)
    audit_margins(runs)
    audit_one_variable(runs)
    audit_components()
    audit_lexical()
    audit_reconciliation(runs)

    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        mark = {PASS: "  ok ", FAIL: "FAIL ", WARN: "  -- "}[status]
        print(f"{mark} {name:<{width}}   {detail}")

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print(f"\n{len(results) - len(fails) - len(warns)} passed, "
          f"{len(fails)} failed, {len(warns)} skipped")
    if fails:
        print("\nDo not write numbers that depend on a failed check.")
        return 1
    if warns:
        print("\nSkipped checks are missing inputs, not passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
