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
