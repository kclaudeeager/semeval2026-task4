"""Where do the two candidates sit relative to the anchor, in each space?

Notebook 03 answered this for TF-IDF only, and for 150 of the 200 triples.
That is not enough to support the claim Section 7.1 wants to make, because the
layers under test are built on sentence encoders, not on TF-IDF. This script
runs the same measurement in every space that matters, on the whole split,
and writes the result where the report can cite it.

For each space:
  * median distance from anchor to the chosen candidate and to the distractor
  * how often the chosen candidate is the nearer one, with an exact binomial
    test against a coin flip
  * the median angle at the anchor between the two displacement vectors

Distance is the Euclidean norm of the displacement between L2-normalized
vectors, which is sqrt(2 - 2*cos): a monotone function of cosine similarity,
so it ranks identically, but it is NOT a cosine distance. The report should
say "displacement norm between unit-normalized vectors" or give the cosine
figure, which is also recorded here.

    python scripts/triple_geometry.py [dev|test]
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

from task4.dataset import load_triples
from task4.evaluate import load_split

OUT_DIR = Path("reports/results")
DATA_ROOT = "data/raw/narrative-similarity-dataset"


def exact_binomial(k: int, n: int) -> float:
    """Two-sided exact test against p = 0.5."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def shape(triples, embed) -> dict:
    rows = []
    for tr in triples:
        a = embed(tr.anchor)
        right = embed(tr.text_a if tr.a_is_closer else tr.text_b)
        wrong = embed(tr.text_b if tr.a_is_closer else tr.text_a)
        dr, dw = right - a, wrong - a
        nr, nw = float(np.linalg.norm(dr)), float(np.linalg.norm(dw))
        cos = float(np.dot(dr, dw) / (nr * nw + 1e-12))
        rows.append({
            "right": nr,
            "wrong": nw,
            "angle": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
            # cosine similarity between anchor and each candidate, for readers
            # who want the familiar quantity rather than the displacement norm
            "cos_right": 1.0 - nr * nr / 2.0,
            "cos_wrong": 1.0 - nw * nw / 2.0,
        })
    r = [x["right"] for x in rows]
    w = [x["wrong"] for x in rows]

    # The reference angle. Both displacement vectors contain -anchor, which
    # forces positive correlation however unrelated the candidates are. So the
    # null for this angle is NOT 90 degrees. Under mutually orthogonal
    # candidates with the observed anchor-candidate cosine c:
    #     dr . dw = 0 - c - c + 1 = 1 - 2c,   |dr| = |dw| = sqrt(2 - 2c)
    # Comparing the observed angle against this is the only way to read it.
    c = st.median([x["cos_right"] for x in rows] + [x["cos_wrong"] for x in rows])
    # The baseline is only defined while two mutually orthogonal candidates
    # could BOTH sit at cosine c from the anchor. Three unit vectors with
    # pairwise cosines (c, c, 0) need a positive-semidefinite Gram matrix,
    # which requires c <= 1/sqrt(2) ~ 0.707. Past that the configuration is
    # geometrically impossible, arccos clips, and the printed angle is an
    # artefact rather than a baseline -- so report it as undefined instead.
    denom = 2.0 - 2.0 * c
    if c > 0.5 ** 0.5 or denom <= 1e-12:
        angle_null = None
    else:
        angle_null = float(np.degrees(np.arccos(
            np.clip((1.0 - 2.0 * c) / denom, -1, 1))))
    k = sum(1 for x in rows if x["right"] < x["wrong"])
    n = len(rows)
    return {
        "n": n,
        "right_median": st.median(r),
        "wrong_median": st.median(w),
        "difference": st.median(r) - st.median(w),
        "cos_right_median": st.median([x["cos_right"] for x in rows]),
        "cos_wrong_median": st.median([x["cos_wrong"] for x in rows]),
        "right_closer_k": k,
        "right_closer_pct": 100.0 * k / n,
        "right_closer_p": exact_binomial(k, n),
        "angle_median": st.median([x["angle"] for x in rows]),
        "angle_null_orthogonal": angle_null,
    }


def tfidf_embedder(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    W = TfidfVectorizer(min_df=2, stop_words="english").fit(texts)
    cache: dict[str, np.ndarray] = {}

    def embed(t: str) -> np.ndarray:
        if t not in cache:
            v = W.transform([t]).toarray()[0]
            n = np.linalg.norm(v)
            cache[t] = v / n if n else v
        return cache[t]
    return embed


def encoder_embedder(model_name: str, texts, prefix: str = ""):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    V = enc.encode([prefix + t for t in texts], normalize_embeddings=True,
                   batch_size=32, show_progress_bar=False)
    table = dict(zip(texts, V))
    return table.__getitem__


SPACES = [
    ("tfidf", "TF-IDF, unit-normalized", None, ""),
    ("minilm", "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ""),
    ("e5", "e5-base-v2", "intfloat/e5-base-v2", "query: "),
]


def main(split: str = "dev") -> None:
    triples = load_split(DATA_ROOT, split)
    triples = [t for t in triples if t.a_is_closer is not None]
    texts = sorted({t for tr in triples for t in (tr.anchor, tr.text_a, tr.text_b)})
    print(f"{split}: {len(triples)} labelled triples, {len(texts)} distinct stories\n")

    record = {"split": split, "n": len(triples), "spaces": {}}

    for key, label, model, prefix in SPACES:
        try:
            embed = (tfidf_embedder(texts) if model is None
                     else encoder_embedder(model, texts, prefix))
        except ImportError as e:
            print(f"  {label}: skipped ({e})\n")
            continue
        s = shape(triples, embed)
        s["label"] = label
        s["model"] = model
        record["spaces"][key] = s

        flat = "" if s["right_closer_p"] < 0.05 else "   indistinguishable from chance"
        print(f"=== {label} ===")
        print(f"    |anchor -> chosen|      median {s['right_median']:.3f}"
              f"   (cosine {s['cos_right_median']:+.3f})")
        print(f"    |anchor -> distractor|  median {s['wrong_median']:.3f}"
              f"   (cosine {s['cos_wrong_median']:+.3f})")
        print(f"    difference              {s['difference']:+.4f}")
        print(f"    chosen is nearer        {s['right_closer_k']}/{s['n']}"
              f"  ({s['right_closer_pct']:.1f}%)  p={s['right_closer_p']:.3f}{flat}")
        nul = s.get("angle_null_orthogonal")
        nultxt = (f"   (unrelated candidates would give {nul:.1f})" if nul is not None
                  else "   (no baseline: anchor-candidate cosine too high for "
                       "orthogonal candidates to exist)")
        print(f"    median angle at anchor  {s['angle_median']:.1f} degrees{nultxt}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"geometry__{split}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")
    print("Now run: make numbers")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
