from task4.dataset import Triple
from task4.evaluate import Result, compare, evaluate
from task4.scorers.lexical import JaccardScorer


def _rows(n=40):
    return [Triple(anchor="dog cat", text_a="dog cat" if i % 2 == 0 else "zebra",
                   text_b="zebra" if i % 2 == 0 else "dog cat",
                   a_is_closer=(i % 2 == 0), outcomes=[True, True] if i % 3 else [False, False])
            for i in range(n)]


def test_evaluate_reports_n_accuracy_and_an_interval():
    r = evaluate(JaccardScorer(), _rows(), "dev", n_boot=200)
    assert r.n == 40 and r.accuracy == 100.0
    assert r.ci_low <= r.accuracy <= r.ci_high


def test_evaluate_breaks_results_down_by_unanimous_stratum():
    r = evaluate(JaccardScorer(), _rows(), "dev", n_boot=200, aspects=("outcomes",))
    block = r.strata["outcomes"]
    assert set(block) >= {"all_marked", "none_marked", "split"}
    assert block["all_marked"]["n"] + block["none_marked"]["n"] + block["split"]["n"] == 40


def test_evaluate_refuses_an_unlabelled_split():
    rows = [Triple(anchor="a", text_a="b", text_b="c")]
    try:
        evaluate(JaccardScorer(), rows, "test")
    except ValueError:
        return
    raise AssertionError("expected ValueError when no gold labels are present")


def test_compare_requires_the_same_items():
    a = Result("x", "dev", 3, 50.0, 0, 0, 0, 0, correct=[True, False, True])
    b = Result("y", "dev", 2, 50.0, 0, 0, 0, 0, correct=[True, False])
    try:
        compare(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched item counts")


def test_cross_validation_never_scores_an_item_with_its_own_fold():
    """The point of evaluate_cv: no item influences the parameters used on it."""
    from task4.evaluate import evaluate_cv

    seen: list[set[int]] = []

    class Spy(JaccardScorer):
        fits_parameters = True

        def fit(self, triples):
            seen.append({id(t) for t in triples})
            return self

    rows = _rows(40)
    evaluate_cv(Spy, rows, "dev", k=4, n_boot=50)
    assert len(seen) == 4
    for train in seen:
        assert len(train) == 30          # each fold trains on the other three


def test_stateless_layers_are_not_marked_for_cross_validation():
    from task4.scorers.lexical import JaccardScorer, RandomScorer, TfidfScorer
    from task4.scorers.combined import CombinedScorer

    assert RandomScorer().fits_parameters is False
    assert JaccardScorer().fits_parameters is False
    assert TfidfScorer().fits_parameters is True
    assert CombinedScorer.fits_parameters is True


def test_fitted_parameters_are_recorded_in_the_result():
    r = evaluate(JaccardScorer(), _rows(), "dev", n_boot=50)
    assert r.fitted == {}


def test_margins_are_signed_by_the_gold_answer():
    """Positive margin means the gold candidate scored higher."""
    rows = [Triple(anchor="dog", text_a="dog", text_b="zebra", a_is_closer=True),
            Triple(anchor="dog", text_a="zebra", text_b="dog", a_is_closer=False)]
    r = evaluate(JaccardScorer(), rows, "dev", n_boot=50)
    assert len(r.margins) == 2
    assert all(m > 0 for m in r.margins), "both should favour the gold candidate"
