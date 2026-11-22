#!/usr/bin/env python3
"""Turn reports/results/*.json into LaTeX macros.

Why this exists: a number typed into prose by hand is a number that can
go stale the next time a layer is re-run. Every figure in the report is
therefore a macro, and every macro is generated here.

    python scripts/export_numbers.py              # write report/numbers.tex
    python scripts/export_numbers.py --list       # show every run on disk
    python scripts/export_numbers.py --check      # find hardcoded numbers in the .tex

Two sources feed the output:

  1. reports/results/*.json  -- accuracy, n, intervals, SD, paired tests.
     Selected by the RUNS table below, so macro names are chosen by hand
     and a renamed config fails loudly instead of silently shifting.
  2. report/facts.json       -- figures that come from the notebooks rather
     than a scored run (geometry, component validity, annotator counts).
     Each carries a `source` field, so the provenance travels with it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
REPORT = ROOT / "report"

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
EFIVE = "intfloat/e5-base-v2"


# --------------------------------------------------------------------------
# Which run gets which macro name.
#
# stem -> (split, scorer name, params that must match)
# Every stem X emits \XAcc \XN \XLo \XHi \XCI \XSD.
# --------------------------------------------------------------------------
RUNS: dict[str, tuple[str, str, dict]] = {
    "Random":          ("dev",  "l0_random",               {}),
    "Jaccard":         ("dev",  "l0_jaccard",              {}),
    "JaccardTest":     ("test", "l0_jaccard",              {}),
    "Tfidf":           ("dev",  "l1_tfidf",                {}),
    "LTwo":            ("dev",  "l2_e5",                   {}),

    "LThreeN":         ("dev",  "l3n_characters_nocoref",  {"actors_only": False}),
    "LThreeNActors":   ("dev",  "l3n_characters_nocoref",  {"actors_only": True}),
    "LThreeX":         ("dev",  "l3x_characters_embed",     {"actors_only": False, "embed_model": MINILM}),
    "LThreeXActors":   ("dev",  "l3x_characters_embed",     {"actors_only": True,  "embed_model": MINILM}),
    "LThreeXEFive":    ("dev",  "l3x_characters_embed",     {"actors_only": False, "embed_model": EFIVE}),

    "LFourN":          ("dev",  "l4n_roles_nocoref",       {"actors_only": False}),
    "LFourNActors":    ("dev",  "l4n_roles_nocoref",       {"actors_only": True}),
    "LFourX":          ("dev",  "l4x_roles_embed",          {"actors_only": False, "embed_model": MINILM}),

    # With coreference (maverick-mes-litbank). NOTE: these have no
    # actors_only flag, because coreference chains define the cast directly.
    # So LThreeN -> LThree is NOT a clean isolation of coreference: the two
    # paths build the cast by different mechanisms. Say so wherever cited.
    "LThree":          ("dev",  "l3_characters",           {}),

    # ---- the single held-out test run (docs/PREREGISTRATION.md) ----
    "RandomTest":      ("test", "l0_random",               {}),
    "TfidfTest":       ("test", "l1_tfidf",                {}),
    "LTwoTest":        ("test", "l2_e5",                   {}),
    "LThreeXTest":     ("test", "l3x_characters_embed",    {"actors_only": True}),
    "LFiveTest":       ("test", "l5_combined",             {"components": ["l2_e5", "l3x_characters_embed", "l4x_roles_embed"]}),
    "LFiveCorefTest":  ("test", "l5_combined",             {"components": ["l2_e5", "l3_characters", "l4_roles"]}),
    "LFour":           ("dev",  "l4_roles",                {}),
    "LFourXActors":    ("dev",  "l4x_roles_embed",          {"actors_only": True,  "embed_model": MINILM}),

    # Two combination runs that are NOT comparable: one is cross-validated,
    # one is fitted and scored on the same split. Kept separate on purpose.
    "LFiveX":          ("dev",  "l5_combined",             {"components": ["l2_e5", "l3x_characters_embed", "l4x_roles_embed"]}),
    # LFiveN is archived, not live: fitted and scored on the same split.
    # Its figure comes from facts.json as \LFiveNInSample.

    # Component-wise comparison (Section 7.2). The three single-component
    # runs are covered by components__dev.json; this is their combination.
    "LSeven":          ("dev",  "l7_components",           {}),
}

# Paired comparisons on the same items. Each emits
# \McnAVsBP \McnAVsBB \McnAVsBC \McnAVsBDiff \McnAVsBN
PAIRS: list[tuple[str, str]] = [
    ("Jaccard", "Tfidf"),
    ("Tfidf", "LTwo"),          # words -> meaning: the one real jump
    ("LTwo", "LThreeX"),        # does cast alignment add anything?
    ("LTwo", "LThreeXActors"),
    ("LTwo", "LFourX"),         # do roles add anything?
    ("LTwo", "LFiveX"),         # does combining add anything?
    ("LThreeX", "LThreeXEFive"),      # stronger encoder inside alignment
    ("LThreeN", "LThreeNActors"),     # cost of filtering the cast
    ("LFourN", "LFourNActors"),
    # coreference against its two coreference-free counterparts
    ("LThreeNActors", "LThree"),
    ("LThreeN", "LThree"),
    ("LFourNActors", "LFour"),
    ("LFourN", "LFour"),
    # the confirmatory comparison, and the combination against the backbone
    ("LTwoTest", "LThreeXTest"),
    ("LTwoTest", "LFiveTest"),
    ("LTwoTest", "TfidfTest"),
]

# Error-vector correlation against the lexical layer: "is this still lexical?"
PHI_AGAINST = "Tfidf"
PHI_LAYERS = ["LTwo", "LThreeN", "LThreeX", "LThreeXEFive", "LFourN", "LFourX", "LFiveX"]

# Which layers the oracle-union and majority-vote ceiling is computed over.
# This is a CHOICE, and the numbers change with it, so it is named here and
# echoed into numbers.tex. Never quote a union figure without saying which
# layers it pooled.
POOL = ["LTwo", "LThreeX", "LFourX", "LThreeN", "LFourN"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@dataclass
class Run:
    path: Path
    scorer: str
    split: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    correct: list[bool]
    params: dict
    fitted: dict
    cv: bool
    strata: dict

    @property
    def sd(self) -> float:
        """Sample SD of the 0/1 correctness vector, in percentage points.

        The instructor asked for a standard deviation alongside accuracy.
        For a binary outcome this is determined by the accuracy itself --
        it describes the spread of single items, not the uncertainty of the
        mean. The bootstrap interval is the one that answers that.
        """
        if self.n < 2:
            return 0.0
        p = self.accuracy / 100.0
        var = self.n / (self.n - 1) * p * (1 - p)
        return 100.0 * math.sqrt(var)

    @property
    def sem(self) -> float:
        """Standard error of the accuracy, percentage points."""
        return self.sd / math.sqrt(self.n) if self.n else 0.0


def load_runs() -> list[Run]:
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        d = json.loads(path.read_text())
        # Not every JSON in this directory is a scored run: the component
        # validity file lives here too and is read separately below.
        if "scorer" not in d or "accuracy" not in d:
            continue
        cfg = d.get("config") or {}
        scorer_cfg = cfg.get("scorer") or {}
        fitted = d.get("fitted") or {}
        out.append(Run(
            path=path,
            scorer=d["scorer"],
            split=d["split"],
            n=d["n"],
            accuracy=d["accuracy"],
            ci_low=d["ci_low"],
            ci_high=d["ci_high"],
            correct=d.get("correct") or [],
            params=scorer_cfg.get("params") or {},
            fitted=fitted,
            cv="per_fold" in fitted,
            strata=d.get("strata") or {},
        ))
    return out


def select(runs: list[Run], split: str, scorer: str, params: dict) -> Run | None:
    hits = [r for r in runs if r.split == split and r.scorer == scorer
            and all(r.params.get(k) == v for k, v in params.items())]
    if len(hits) > 1:
        # Prefer the newest file; warn, because an ambiguous selector is a bug.
        print(f"  ! {scorer}/{split} matched {len(hits)} runs: "
              + ", ".join(h.path.name for h in hits), file=sys.stderr)
        hits.sort(key=lambda r: r.path.stat().st_mtime)
    return hits[-1] if hits else None


# --------------------------------------------------------------------------
# paired statistics
# --------------------------------------------------------------------------
def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired test on the same items. b = a-only-right, c = b-only-right."""
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    disc = bb + cc
    if disc == 0:
        return {"b": 0, "c": 0, "p": 1.0, "exact": True, "stat": 0.0}
    if disc < 25:
        # Exact binomial, two-sided.
        k = min(bb, cc)
        tail = sum(math.comb(disc, i) for i in range(k + 1)) / (2 ** disc)
        return {"b": bb, "c": cc, "p": min(1.0, 2 * tail), "exact": True, "stat": float(k)}
    chi = (abs(bb - cc) - 1) ** 2 / disc
    p = math.erfc(math.sqrt(chi / 2))
    return {"b": bb, "c": cc, "p": p, "exact": False, "stat": chi}


def phi(a: list[bool], b: list[bool]) -> float:
    """Correlation between two error vectors. High = the same items are hard."""
    n = len(a)
    if n == 0 or len(b) != n:
        return float("nan")
    ea = [0 if x else 1 for x in a]
    eb = [0 if x else 1 for x in b]
    ma, mb = sum(ea) / n, sum(eb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ea, eb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ea))
    db = math.sqrt(sum((y - mb) ** 2 for y in eb))
    return num / (da * db) if da and db else float("nan")


# --------------------------------------------------------------------------
# emitting
# --------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}".rstrip("0").rstrip(".") if p < 0.1 else f"{p:.2f}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build(runs: list[Run]) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    problems: list[str] = []
    resolved: dict[str, Run] = {}

    lines.append("% Generated by scripts/export_numbers.py -- do not edit by hand.")
    lines.append("% Regenerate with: make numbers")
    lines.append("")
    lines.append("% ---- split sizes " + "-" * 56)
    for stem, split in (("DevN", "dev"), ("TestN", "test"), ("SampleN", "sample")):
        hit = next((r for r in runs if r.split == split), None)
        if hit:
            lines.append(f"% {stem}: labelled triples in the {split} split, "
                         f"counted from {hit.path.name}")
            lines.append(macro(stem, str(hit.n)))
    lines.append("")

    lines.append("% ---- per-run figures " + "-" * 52)
    for stem, (split, scorer, params) in RUNS.items():
        run = select(runs, split, scorer, params)
        if run is None:
            problems.append(f"no run on disk for {stem} ({scorer}/{split} {params})")
            lines.append(f"% MISSING: {stem} -- {scorer}/{split} {params}")
            continue
        resolved[stem] = run
        tag = " (cross-validated)" if run.cv else ""
        lines.append(f"% {stem}: {run.path.name}{tag}")
        lines.append(macro(f"{stem}Acc", f"{run.accuracy:.2f}"))
        lines.append(macro(f"{stem}N", str(run.n)))
        lines.append(macro(f"{stem}Lo", f"{run.ci_low:.1f}"))
        lines.append(macro(f"{stem}Hi", f"{run.ci_high:.1f}"))
        lines.append(macro(f"{stem}CI", f"[{run.ci_low:.1f}, {run.ci_high:.1f}]"))
        lines.append(macro(f"{stem}SD", f"{run.sd:.2f}"))
        lines.append(macro(f"{stem}SE", f"{run.sem:.2f}"))
        lines.append(macro(f"{stem}CV", "cross-validated" if run.cv else "single fit"))
    lines.append("")

    lines.append("% ---- paired comparisons (McNemar, same items) " + "-" * 27)
    for left, right in PAIRS:
        a, b = resolved.get(left), resolved.get(right)
        if not a or not b:
            problems.append(f"pair {left}-{right} incomplete")
            continue
        if len(a.correct) != len(b.correct):
            problems.append(f"pair {left}-{right} has unequal n "
                            f"({len(a.correct)} vs {len(b.correct)}) -- not paired")
            continue
        m = mcnemar(a.correct, b.correct)
        stem = f"Mcn{left}Vs{right}"
        kind = "exact binomial" if m["exact"] else "chi-square, continuity corrected"
        lines.append(f"% {left} {a.accuracy:.2f} vs {right} {b.accuracy:.2f}  [{kind}]")
        lines.append(macro(f"{stem}P", fmt_p(m["p"])))
        lines.append(macro(f"{stem}B", str(m["b"])))
        lines.append(macro(f"{stem}C", str(m["c"])))
        lines.append(macro(f"{stem}Disc", str(m["b"] + m["c"])))
        lines.append(macro(f"{stem}Diff", f"{b.accuracy - a.accuracy:+.2f}"))
        lines.append(macro(f"{stem}Test", kind))
    lines.append("")

    lines.append(f"% ---- error correlation against {PHI_AGAINST} " + "-" * 36)
    base = resolved.get(PHI_AGAINST)
    if base:
        vals = []
        for stem in PHI_LAYERS:
            run = resolved.get(stem)
            if not run or len(run.correct) != len(base.correct):
                continue
            v = phi(base.correct, run.correct)
            vals.append(v)
            lines.append(f"% Phi{stem}: error-vector correlation of "
                         f"{run.path.name} with {base.path.name}, "
                         f"same {len(base.correct)} {base.split} items")
            lines.append(macro(f"Phi{stem}", f"{v:+.2f}"))
        if vals:
            lines.append(f"% PhiMin/PhiMax: range of the {len(vals)} "
                         f"correlations listed above")
            lines.append(macro("PhiMin", f"{min(vals):+.2f}"))
            lines.append(macro("PhiMax", f"{max(vals):+.2f}"))
    lines.append("")

    # Oracle union and majority vote over the structural layers.
    lines.append("% ---- combination ceiling " + "-" * 48)
    pool = [resolved[s] for s in POOL if s in resolved]
    pool = [r for r in pool if len(r.correct) == len(pool[0].correct)] if pool else []
    if len(pool) >= 2:
        n = len(pool[0].correct)
        union = sum(1 for i in range(n) if any(r.correct[i] for r in pool)) / n * 100
        indep = (1 - math.prod(1 - r.accuracy / 100 for r in pool)) * 100
        vote = sum(1 for i in range(n)
                   if sum(r.correct[i] for r in pool) * 2 > len(pool)) / n * 100
        lines.append(f"% pool: {', '.join(r.scorer for r in pool)}")
        lines.append(macro("PoolNames", ", ".join(POOL[:len(pool)])))
        lines.append(macro("PoolSize", str(len(pool))))
        lines.append(macro("OracleUnion", f"{union:.1f}"))
        lines.append(macro("OracleIndep", f"{indep:.1f}"))
        lines.append(macro("MajorityVote", f"{vote:.1f}"))
    lines.append("")

    # Fitted weights, so the combination layer has a concrete artifact.
    lines.append("% ---- fitted weights " + "-" * 53)
    for stem in ("LFiveX", "LFiveN"):
        run = resolved.get(stem)
        folds = (run.fitted or {}).get("per_fold") if run else None
        if not folds:
            continue
        keys = folds[0].get("components") or []
        if not keys:
            continue
        mean = [sum(f["weights"][i] for f in folds) / len(folds) for i in range(len(keys))]
        lines.append(f"% {stem} mean weights over {len(folds)} folds: "
                     + ", ".join(f"{k}={w:.2f}" for k, w in zip(keys, mean)))
        lines.append(macro(f"{stem}Folds", str(len(folds))))
        lines.append(macro(f"{stem}Weights",
                           ", ".join(f"{w:.2f}" for w in mean)))
    lines.append("")

    # Strata, for the diagnostic section.
    lines.append("% ---- unanimous strata, outcomes " + "-" * 41)
    for stem in ("Tfidf", "LTwo", "LThreeX"):
        run = resolved.get(stem)
        s = (run.strata or {}).get("outcomes") if run else None
        if not s:
            continue
        for key, short in (("all_marked", "Marked"), ("none_marked", "Unmarked")):
            if key in s:
                where = ("every annotator marked outcomes" if key == "all_marked"
                         else "no annotator marked outcomes")
                lines.append(f"% {stem}{short}: {run.path.name}, "
                             f"unanimous stratum where {where}")
                lines.append(macro(f"{stem}{short}Acc", f"{s[key]['accuracy']:.1f}"))
                lines.append(macro(f"{stem}{short}N", str(s[key]["n"])))
    lines.append("")

    # Component validity, from the generated file rather than by hand.
    # Until `make components` has been run this file does not exist, and the
    # hand-entered values in facts.json are used instead -- which is flagged,
    # because a figure that cannot be re-derived should not be cited.
    generated: set[str] = set()
    comp_path = RESULTS / "components__dev.json"
    lines.append("% ---- component validity (generated) " + "-" * 37)
    if comp_path.exists():
        comp = json.loads(comp_path.read_text())
        short = {"theme": "Theme", "action": "Action", "outcome": "Outcome"}
        lines.append(f"% from {comp_path.name}, split {comp['split']}")
        for key, name in short.items():
            e = (comp.get("components") or {}).get(key)
            if not e:
                continue
            lo, hi = e["wilson_low"], e["wilson_high"]
            for mname, val in (
                (f"Comp{name}Acc", f"{e['accuracy']:.1f}"),
                (f"Comp{name}CI", f"[{lo:.1f}, {hi:.1f}]"),
                (f"Comp{name}Lo", f"{lo:.1f}"),
                (f"Comp{name}Hi", f"{hi:.1f}"),
            ):
                lines.append(macro(mname, val))
                generated.add(mname)
            if lo <= 50.0 <= hi:
                lines.append(f"%   {key}: interval contains 50")
        lines.append(macro("CompN", str(comp["n"])))
        generated.add("CompN")
    else:
        lines.append("% not generated yet -- run: make components")
        problems.append("reports/results/components__dev.json is missing; the "
                        "component figures in Section 7.2 are still hand-entered "
                        "in facts.json and cannot be re-derived. Run: make components")
    lines.append("")

    # Triple geometry, per space, from the generated file.
    geom_path = RESULTS / "geometry__dev.json"
    lines.append("% ---- triple geometry per space (generated) " + "-" * 31)
    if geom_path.exists():
        g = json.loads(geom_path.read_text())
        short = {"tfidf": "Tfidf", "minilm": "MiniLM", "e5": "EFive"}
        lines.append(f"% from {geom_path.name}, split {g['split']}")
        for key, name in short.items():
            e = (g.get("spaces") or {}).get(key)
            if not e:
                lines.append(f"% {key}: not computed")
                continue
            lines.append(f"% {name}: {e['label']}")
            for mn, val in (
                (f"Geom{name}Right", f"{e['right_median']:.3f}"),
                (f"Geom{name}Wrong", f"{e['wrong_median']:.3f}"),
                (f"Geom{name}Diff", f"{e['difference']:+.4f}"),
                (f"Geom{name}CosRight", f"{e['cos_right_median']:+.3f}"),
                (f"Geom{name}CosWrong", f"{e['cos_wrong_median']:+.3f}"),
                (f"Geom{name}CloserK", str(e["right_closer_k"])),
                (f"Geom{name}CloserPct", f"{e['right_closer_pct']:.1f}"),
                (f"Geom{name}CloserP", fmt_p(e["right_closer_p"])),
                (f"Geom{name}Angle", f"{e['angle_median']:.1f}"),
                (f"Geom{name}AngleNull",
                 f"{e['angle_null_orthogonal']:.1f}"
                 if e.get("angle_null_orthogonal") is not None else "undefined"),
                (f"Geom{name}N", str(e["n"])),
            ):
                lines.append(macro(mn, val))
                generated.add(mn)
    else:
        lines.append("% not generated yet -- run: make geometry")
        problems.append("reports/results/geometry__dev.json is missing; the "
                        "Section 7.1 figures are TF-IDF only, measured on 150 "
                        "of 200 triples, and say nothing about the embedding "
                        "space the layers actually use. Run: make geometry")
    lines.append("")

    # Distributional direction: does the token distribution separate the
    # candidates, and which way does it point?
    lex_path = RESULTS / "lexical_direction__dev.json"
    lines.append("% ---- lexical direction (generated) " + "-" * 38)
    if lex_path.exists():
        lx = json.loads(lex_path.read_text())
        short = {"js": "JS", "kl_anchor_given_cand": "KLAnchor",
                 "kl_cand_given_anchor": "KLCand", "cross_entropy": "CrossEnt",
                 "edit_distance": "Edit", "edit_distance_norm": "EditNorm",
                 "length_diff": "Length", "type_token_diff": "TypeToken"}
        for vkey, vtag in (("all_tokens", "All"), ("content_words", "Content")):
            v = (lx.get("variants") or {}).get(vkey) or {}
            lines.append(f"% {vkey}, dev, n={lx['n']}")
            for mkey, mtag in short.items():
                e = v.get(mkey)
                if not e:
                    continue
                stem = f"Lex{vtag}{mtag}"
                for mn, val in ((f"{stem}Pct", f"{e['pct']:.1f}"),
                                (f"{stem}K", str(e["k"])),
                                (f"{stem}P", fmt_p(e["p"]))):
                    lines.append(macro(mn, val))
                    generated.add(mn)
        lines.append(macro("LexN", str(lx["n"])))
        generated.add("LexN")
    else:
        lines.append("% not generated yet -- run: python scripts/lexical_direction.py dev")
    lines.append("")

    # Spread across the configurations actually built. This is the one
    # standard deviation in the report that carries information: for a single
    # binary system SD is a function of the accuracy, but across systems it
    # says how much the choice of representation moves the score at all.
    lines.append("% ---- spread across configurations (generated) " + "-" * 27)
    for tag, split in (("Dev", "dev"), ("Test", "test")):
        accs = sorted({(r.scorer, r.path.stem, r.accuracy) for r in runs
                       if r.split == split})
        vals = [a for _, _, a in accs]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        lines.append(f"% computed here over the {len(vals)} distinct {split} "
                     f"configurations in reports/results")
        for mn, val in ((f"RunCount{tag}", str(len(vals))),
                        (f"SpreadMean{tag}", f"{mean:.2f}"),
                        (f"Spread{tag}SD", f"{sd:.2f}"),
                        (f"SpreadMin{tag}", f"{min(vals):.2f}"),
                        (f"SpreadMax{tag}", f"{max(vals):.2f}")):
            lines.append(macro(mn, val))
            generated.add(mn)
    lines.append("")

    # Why the L0 baseline does not match the published figure. Generated so
    # that the explanation in the data section cannot drift from the measurement.
    tok_path = RESULTS / "tokenizer_reconciliation.json"
    lines.append("% ---- tokenizer reconciliation (generated) " + "-" * 31)
    if tok_path.exists():
        tk = json.loads(tok_path.read_text())
        cur = tk["splits"]["test"][tk["current"]]
        nvar = len(tk["splits"]["test"])
        lines.append(
            f"% scripts/tokenizer_reconciliation.py: the L0 token-overlap "
            f"baseline on test under {nvar} tokenizations, against the "
            f"published {tk['published_test']:.2f}\\%")
        for mn, val in (
            ("JaccardGapPts", f"{tk['gap_points']:.2f}"),
            ("JaccardGapItems", str(tk["gap_items"])),
            ("JaccardFlipN", str(tk["flips"]["changed"])),
            ("JaccardFlipFor", str(tk["flips"]["current_right"])),
            ("JaccardFlipAgainst", str(tk["flips"]["current_wrong"])),
            ("JaccardDiffTypes", str(tk["vocabulary"]["types"])),
            ("JaccardDiffApostPct",
             f"{tk['vocabulary'].get('apostrophe_pct_types', tk['vocabulary']['apostrophe_pct']):.1f}"),
            ("JaccardDiffApostN", str(tk["vocabulary"].get("apostrophe_types", 0))),
            ("JaccardVariantsMatching", str(len(tk["matching_variants"]))),
        ):
            lines.append(macro(mn, val))
            generated.add(mn)
        lines.append(
            f"% scripts/tokenizer_reconciliation.py: the same "
            f"{len(tk['splits']['dev'])} tokenizations on dev, showing the "
            f"below-chance result is not an artifact of the choice")
        for mn, val in (
            ("JaccardMinVariant", f"{tk['dev_variant_min']:.2f}"),
            ("JaccardMaxVariant", f"{tk['dev_variant_max']:.2f}"),
            ("JaccardVariantsN", str(len(tk["splits"]["dev"]))),
        ):
            lines.append(macro(mn, val))
            generated.add(mn)
        if not tk["dev_all_below_chance"]:
            problems.append(
                "tokenizer_reconciliation: a dev variant is at or above "
                "chance; the claim in the data section no longer holds")
        if abs(cur["accuracy"] - tk["published_test"] - tk["gap_points"]) > 1e-9:
            problems.append("tokenizer_reconciliation: gap_points disagrees "
                            "with the measured accuracies")
    else:
        lines.append("% not generated yet -- run: "
                     "TASK4_ALLOW_TEST=1 python "
                     "scripts/tokenizer_reconciliation.py")
    lines.append("")

    # Hand-entered figures with provenance.
    facts_path = REPORT / "facts.json"
    lines.append("% ---- notebook-derived figures (report/facts.json) " + "-" * 23)
    if facts_path.exists():
        facts = json.loads(facts_path.read_text())
        for name, entry in facts.items():
            if name.startswith("_"):
                continue
            if name in generated:
                lines.append(f"% {name}: superseded by the generated value above")
                continue
            lines.append(f"% {name}: {entry.get('source', 'NO SOURCE RECORDED')}")
            lines.append(macro(name, str(entry["value"])))
            if "source" not in entry:
                problems.append(f"fact {name} has no source recorded")
            if "VERIFY" in str(entry.get("note", "")):
                problems.append(f"fact {name} is marked VERIFY -- check it before citing")
    else:
        problems.append("report/facts.json is missing")
    lines.append("")
    # Self-counting: the report states how many figures are generated, so
    # that claim is generated too.
    n_macros = sum(1 for ln in lines if ln.startswith("\\newcommand")) + 1
    lines.append("% MacroCount: counted here, the number of macros in this file")
    lines.append(macro("MacroCount", str(n_macros)))
    lines.append("")
    return lines, problems


# --------------------------------------------------------------------------
# the lint
# --------------------------------------------------------------------------
NUMBERISH = re.compile(r"(?<![\\A-Za-z0-9.])(\d{1,3}\.\d{1,2})(?![0-9])")
# Bare integers were slipping through: "wrong on 99 of the 200 items" is a
# claim about the data exactly as much as "45.00%" is.
INTEGERISH = re.compile(r"(?<![\\A-Za-z0-9.,])(\d{2,4})(?![0-9.,])")
PERCENTISH = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*\\%")
SAFE = re.compile(r"\\(ref|label|cite[a-z]*|includegraphics|input|usepackage|"
                  r"documentclass|geometry|vspace|hspace|setlength|titlespacing|"
                  r"titleformat|setlist|columnsep|textwidth|linewidth|"
                  r"baselinestretch|bibsep|graphicspath)")



def provenance(lines: list[str]) -> list[str]:
    """Every macro, its value, and where it came from.

    Parsed from the comments build() writes, so this cannot drift from
    numbers.tex: if a macro has no comment above it, it shows up here as
    UNKNOWN and --check fails.
    """
    out = ["# Where every number comes from", "",
           "Generated by `scripts/export_numbers.py`. Do not edit.",
           "", "Regenerate with `make numbers`. A row marked **UNKNOWN** is a",
           "number with no recorded source: fix it before citing it.", ""]
    section, detail, rows = "", "", []
    for ln in lines:
        if ln.startswith("% ----"):
            section = ln.strip("% -").strip()
            detail = ""
        elif ln.startswith("%"):
            detail = ln.lstrip("% ").strip()
        elif ln.startswith("\\newcommand"):
            name = ln.split("{\\", 1)[1].split("}", 1)[0]
            value = ln.split("}{", 1)[1].rsplit("}", 1)[0]
            rows.append((name, value, section or "?", detail or "UNKNOWN"))
    out.append("| macro | value | group | measured where |")
    out.append("|---|---|---|---|")
    for name, value, sec, det in rows:
        v = value.replace("|", "\\|")
        out.append(f"| `\\{name}` | {v} | {sec} | {det} |")
    unknown = [r[0] for r in rows if r[3] == "UNKNOWN"]
    out += ["", f"{len(rows)} macros, {len(unknown)} without a source."]
    if unknown:
        out += ["", "Missing a source: " + ", ".join(f"`\\{u}`" for u in unknown)]
    return out

def check(tex: Path) -> int:
    """Flag figures typed into the prose. Only the body is linted, since the
    preamble is full of lengths. Put `% ok` at the end of a line to exempt it.
    """
    if not tex.exists():
        print(f"{tex} does not exist yet -- nothing to check.")
        return 0
    hits = []
    in_body = False
    for i, line in enumerate(tex.read_text().splitlines(), 1):
        if not in_body:
            in_body = "\\begin{document}" in line
            continue
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        if re.search(r"%\s*ok\b", line):
            continue
        body = line.split("  %")[0]
        if SAFE.search(body):
            continue
        for m in (list(NUMBERISH.finditer(body)) + list(PERCENTISH.finditer(body))
                  + list(INTEGERISH.finditer(body))):
            hits.append((i, m.group(1), body.strip()[:88]))
    if not hits:
        print("No hardcoded figures found in the prose. Every number is a macro.")
        return 0
    print(f"{len(hits)} number(s) typed directly into {tex.name}:\n")
    for line_no, val, ctx in hits:
        print(f"  line {line_no:>4}  {val:>8}   {ctx}")
    print("\nEach of these should be a macro from numbers.tex, or moved into")
    print("report/facts.json with its source recorded.")
    return 1


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every run on disk and exit")
    ap.add_argument("--check", action="store_true", help="find hardcoded numbers in report.tex")
    ap.add_argument("--out", default=str(REPORT / "numbers.tex"))
    args = ap.parse_args()

    if args.check:
        return check(REPORT / "report.tex")

    runs = load_runs()
    if args.list:
        print(f"{len(runs)} run(s) in {RESULTS.relative_to(ROOT)}\n")
        claimed = {}
        for stem, (split, scorer, params) in RUNS.items():
            r = select(runs, split, scorer, params)
            if r:
                claimed[r.path.name] = stem
        for r in runs:
            stem = claimed.get(r.path.name, "-- unclaimed --")
            keep = {k: v for k, v in r.params.items()
                    if k in ("actors_only", "embed_model", "components", "model_name")}
            print(f"  {r.split:<7} {r.scorer:<26} {r.accuracy:>6.2f}  "
                  f"{stem:<16} {keep}")
        return 0

    lines, problems = build(runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    n_macros = sum(1 for ln in lines if ln.startswith("\\newcommand"))
    print(f"wrote {out.relative_to(ROOT)}  ({n_macros} macros)")

    prov = REPORT / "PROVENANCE.md"
    prov_lines = provenance(lines)
    prov.write_text("\n".join(prov_lines) + "\n")
    print(f"wrote {prov.relative_to(ROOT)}")
    missing = [ln for ln in prov_lines if ln.startswith("Missing a source")]
    if missing:
        problems.append(missing[0])
    if problems:
        print(f"\n{len(problems)} thing(s) to deal with:")
        for p in problems:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
