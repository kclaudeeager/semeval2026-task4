"""Presentation figures built from saved run results.

Every figure reads `reports/results/*.json`, so a chart can never disagree with
the numbers in the report: both come from the same file. Colors follow a
validated categorical palette (CVD-separated, light and dark steps supplied);
single-series charts carry no legend and label their values directly.
"""
from __future__ import annotations
import json
from pathlib import Path

# Categorical slots 1-3, validated for all-pairs separation in both modes.
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
         "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3a38",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"}

LAYER_ORDER = ["l0_random", "l0_jaccard", "l1_tfidf", "l2_e5",
               "l3_characters", "l4_roles", "l5_combined"]
LAYER_LABEL = {"l0_random": "L0 random", "l0_jaccard": "L0 Jaccard", "l1_tfidf": "L1 TF-IDF",
               "l2_e5": "L2 e5 embeddings", "l3_characters": "L3 characters",
               "l4_roles": "L4 roles", "l5_combined": "L5 combined"}
CHANCE = 50.0


def _theme(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _style(ax, t: dict, xlabel: str) -> None:
    """Recessive axes: no box, grid only on the value axis."""
    ax.set_facecolor(t["surface"])
    ax.figure.patch.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=t["muted"], fontsize=10)


def load_results(results_dir: str | Path, split: str) -> dict[str, dict]:
    """Every saved result for one split, keyed by layer name."""
    out = {}
    for path in sorted(Path(results_dir).glob(f"{split}__*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        out[d["scorer"]] = d
    return out


def ablation_chart(results: dict[str, dict], out_path: str | Path,
                   title: str = "Accuracy by layer", dark: bool = False,
                   human_ceiling: float | None = None):
    """Horizontal bars with bootstrap intervals, one per layer.

    Horizontal because the category labels are words, and a magnitude comparison
    across named categories reads along the x-axis without turning the head.
    """
    import matplotlib.pyplot as plt

    t = _theme(dark)
    layers = [l for l in LAYER_ORDER if l in results]
    if not layers:
        raise ValueError("no results to plot")
    acc = [results[l]["accuracy"] for l in layers]
    lo = [results[l]["accuracy"] - results[l]["ci_low"] for l in layers]
    hi = [results[l]["ci_high"] - results[l]["accuracy"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 0.62 * len(layers) + 1.6))
    y = range(len(layers))
    ax.barh(y, acc, height=0.62, color=t["s1"], zorder=3)
    ax.errorbar(acc, y, xerr=[lo, hi], fmt="none", ecolor=t["muted"],
                elinewidth=2, capsize=4, zorder=4)
    ax.axvline(CHANCE, color=t["muted"], linewidth=2, linestyle=(0, (4, 3)), zorder=2)
    if human_ceiling:
        ax.axvline(human_ceiling, color=t["s2"], linewidth=2, linestyle=(0, (4, 3)), zorder=2)

    # Direct labels: the contrast rule requires visible values, not color alone.
    for yi, (a, h) in enumerate(zip(acc, hi)):
        ax.text(a + h + 1.2, yi, f"{a:.1f}", va="center", color=t["ink"],
                fontsize=10, fontweight="medium")

    ax.set_yticks(list(y), [LAYER_LABEL.get(l, l) for l in layers], color=t["ink"])
    ax.invert_yaxis()
    # Reference-line labels go above the plot, clear of the bars and the tick row.
    top = ax.get_ylim()[1]
    ax.text(CHANCE, top - 0.08, "chance", color=t["muted"], fontsize=9,
            ha="center", va="bottom")
    if human_ceiling:
        ax.text(human_ceiling, top - 0.08, "human", color=t["s2"], fontsize=9,
                ha="center", va="bottom")
    ax.set_xlim(0, max(max(acc) + 12, (human_ceiling or 0) + 6, 70))
    _style(ax, t, "accuracy (%)")
    ax.set_title(title, color=t["ink"], fontsize=13, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out_path


def strata_chart(results: dict[str, dict], out_path: str | Path, aspect: str = "outcomes",
                 title: str | None = None, dark: bool = False):
    """Two bars per layer: accuracy where every annotator marked the aspect, and
    where none did. Split votes are excluded, which is the whole point."""
    import matplotlib.pyplot as plt
    import numpy as np

    t = _theme(dark)
    layers = [l for l in LAYER_ORDER if l in results and aspect in results[l].get("strata", {})]
    if not layers:
        raise ValueError(f"no results carry strata for {aspect!r}")

    def get(layer, key):
        v = results[layer]["strata"][aspect][key]
        return v["accuracy"] if v["accuracy"] is not None else np.nan

    marked = [get(l, "all_marked") for l in layers]
    unmarked = [get(l, "none_marked") for l in layers]
    n_marked = results[layers[0]]["strata"][aspect]["all_marked"]["n"]
    n_unmarked = results[layers[0]]["strata"][aspect]["none_marked"]["n"]

    x = np.arange(len(layers), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.45 * len(layers) + 2.4, 4.6))
    ax.bar(x - w / 2 - 0.01, marked, w, color=t["s1"], zorder=3,
           label=f"all annotators marked it (n={n_marked})")
    ax.bar(x + w / 2 + 0.01, unmarked, w, color=t["s2"], zorder=3,
           label=f"none did (n={n_unmarked})")
    ax.axhline(CHANCE, color=t["muted"], linewidth=2, linestyle=(0, (4, 3)), zorder=2)

    for xi, (m, u) in enumerate(zip(marked, unmarked)):
        for off, val in ((-w / 2 - 0.01, m), (w / 2 + 0.01, u)):
            if not np.isnan(val):
                ax.text(xi + off, val + 1.0, f"{val:.1f}", ha="center",
                        color=t["ink"], fontsize=9)

    ax.set_xticks(x, [LAYER_LABEL.get(l, l).replace(" ", "\n", 1) for l in layers], color=t["ink"])
    ax.set_ylim(0, max([v for v in marked + unmarked if v == v] + [60]) + 12)
    _style(ax, t, "")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_ylabel("accuracy (%)", color=t["muted"], fontsize=10)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9,
                    bbox_to_anchor=(0, 1.14), ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    ax.set_title(title or f"Where the signal sits: {aspect.replace('_', ' ')}",
                 color=t["ink"], fontsize=13, loc="left", pad=34)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out_path


def split_comparison(dev: dict[str, dict], test: dict[str, dict], out_path: str | Path,
                     title: str = "Dev is harder than test", dark: bool = False):
    """Dev against test for the layers measured on both. One axis, two series."""
    import matplotlib.pyplot as plt
    import numpy as np

    t = _theme(dark)
    layers = [l for l in LAYER_ORDER if l in dev and l in test]
    if not layers:
        raise ValueError("no layer has been run on both splits")
    x = np.arange(len(layers), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.5 * len(layers) + 2.4, 4.4))
    ax.bar(x - w / 2 - 0.01, [dev[l]["accuracy"] for l in layers], w,
           color=t["s1"], zorder=3, label="dev")
    ax.bar(x + w / 2 + 0.01, [test[l]["accuracy"] for l in layers], w,
           color=t["s3"], zorder=3, label="test")
    ax.axhline(CHANCE, color=t["muted"], linewidth=2, linestyle=(0, (4, 3)), zorder=2)
    for xi, l in enumerate(layers):
        for off, val in ((-w / 2 - 0.01, dev[l]["accuracy"]), (w / 2 + 0.01, test[l]["accuracy"])):
            ax.text(xi + off, val + 1.0, f"{val:.1f}", ha="center", color=t["ink"], fontsize=9)
    ax.set_xticks(x, [LAYER_LABEL.get(l, l) for l in layers], color=t["ink"])
    ax.set_ylim(0, 80)
    _style(ax, t, "")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_ylabel("accuracy (%)", color=t["muted"], fontsize=10)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.1), ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["muted"])
    ax.set_title(title, color=t["ink"], fontsize=13, loc="left", pad=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out_path
