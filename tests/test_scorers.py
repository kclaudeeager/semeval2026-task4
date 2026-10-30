from task4.dataset import Triple
from task4.registry import available, build
from task4.scorers.lexical import JaccardScorer, tokenize


def _triple(anchor, a, b, label=True):
    return Triple(anchor=anchor, text_a=a, text_b=b, a_is_closer=label)


def test_every_layer_is_registered():
    names = available()
    for expected in ["l0_random", "l0_jaccard", "l1_tfidf", "l2_e5",
                     "l3_characters", "l4_roles", "l5_combined"]:
        assert expected in names


def test_unknown_scorer_name_fails_loudly():
    try:
        build("l9_imaginary")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unregistered scorer")


def test_jaccard_prefers_the_overlapping_candidate():
    s = JaccardScorer()
    assert s.score("a dog barks", "a dog sleeps") > s.score("a dog barks", "quantum chromodynamics")


def test_jaccard_is_symmetric():
    s = JaccardScorer()
    assert s.score("x y", "y z") == s.score("y z", "x y")


def test_predict_returns_one_decision_per_triple():
    rows = [_triple("a dog", "a dog", "a cat"), _triple("a cat", "a dog", "a cat", False)]
    preds = JaccardScorer().predict(rows)
    assert len(preds) == 2 and preds[0] is True


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("The Dog's, bark!") == ["the", "dog's", "bark"]


def test_embedding_alignment_batches_its_encoder_calls():
    """One batched call per triple, not one call per character."""
    from task4.scorers.embedalign import CharactersEmbedAligned
    from task4.scorers.characters import Character
    import numpy as np

    calls = {"n": 0, "texts": 0}

    class FakeEncoder:
        def encode(self, texts, **kw):
            batch = [texts] if isinstance(texts, str) else list(texts)
            calls["n"] += 1
            calls["texts"] += len(batch)
            return np.array([[float(len(t)), 1.0] for t in batch])

    s = CharactersEmbedAligned.__new__(CharactersEmbedAligned)
    s.embed_prefix = ""
    s.batch_size = 64
    s._enc = FakeEncoder()
    s.__dict__["_vec_cache"] = {}

    cast_a = [Character(mentions=[f"c{i}"], context=["x"]) for i in range(30)]
    cast_b = [Character(mentions=[f"d{i}"], context=["y"]) for i in range(30)]
    s.warm(cast_a, cast_b)

    assert calls["n"] == 1, "60 characters should take one encoder call, not 60"
    assert calls["texts"] == 60
    s.warm(cast_a, cast_b)
    assert calls["n"] == 1, "a second warm on cached characters must not re-encode"


def test_segments_are_ordered_and_cover_the_text():
    from task4.scorers.rhythm import split_segments

    segs = split_segments("A. B. C. D. E. F.", 3)
    assert len(segs) == 3
    assert segs[0].startswith("A") and segs[-1].endswith("F.")


def test_short_stories_still_yield_k_segments():
    from task4.scorers.rhythm import split_segments

    assert len(split_segments("Just one.", 5)) == 5
    assert len(split_segments("", 4)) == 4


def test_elastic_alignment_tolerates_pacing_but_not_reordering():
    """DTW should forgive a stretched story and punish a shuffled one."""
    import numpy as np
    from task4.scorers.rhythm import ElasticSegments

    s = ElasticSegments.__new__(ElasticSegments)
    s.segments = 4

    base = np.eye(4)                      # four orthogonal "phases"
    stretched = np.array([base[0], base[0], base[1], base[2]])
    shuffled = np.array([base[3], base[1], base[2], base[0]])

    s._cache = {"base": base, "stretch": stretched, "shuffle": shuffled}
    s.vectors = lambda t: s._cache[t]

    assert s.score("base", "stretch") > s.score("base", "shuffle")


def test_outcome_component_is_the_tail_of_the_story():
    from task4.components import decompose

    c = decompose("One. Two. Three. Four. Five. Six. Seven. Eight.")
    assert c.outcome.endswith("Eight.")
    assert "One." not in c.outcome
    assert c.theme.startswith("One.")


def test_short_story_still_yields_all_three_components():
    from task4.components import decompose

    c = decompose("A single sentence summary.")
    assert c.theme and c.action and c.outcome


def test_single_component_scorer_zeroes_the_others():
    from task4.registry import build

    s = build("l7s_single_component", component="outcome")
    assert s.weights == [0.0, 0.0, 1.0]
    assert s.fits_parameters is False
