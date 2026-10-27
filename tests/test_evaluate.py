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
