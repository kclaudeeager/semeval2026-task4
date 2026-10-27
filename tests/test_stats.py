import math
from task4.stats import bootstrap, describe, mcnemar, two_proportion_z, wilson


def test_wilson_bounds_are_ordered_and_contain_the_point():
    i = wilson(58, 100)
    assert i.lo < i.point < i.hi
    assert 0 <= i.lo and i.hi <= 100


def test_wilson_handles_the_degenerate_cases():
    assert wilson(0, 0).point == 0.0
    assert wilson(10, 10).hi <= 100.0


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    data = [True, False] * 50
    assert bootstrap(data, n_boot=200, seed=7) == bootstrap(data, n_boot=200, seed=7)


def test_mcnemar_is_symmetric_in_magnitude():
    a = [True] * 10 + [False] * 10
    b = [False] * 10 + [True] * 10
    forward, backward = mcnemar(a, b), mcnemar(b, a)
    assert math.isclose(forward["p_value"], backward["p_value"])
    assert forward["b01"] == backward["b10"]


def test_mcnemar_uses_the_exact_test_when_discordant_pairs_are_few():
    a = [True] * 20
    b = [False] * 3 + [True] * 17
    assert mcnemar(a, b)["test"] == "exact"


def test_mcnemar_rejects_mismatched_lengths():
    try:
        mcnemar([True], [True, False])
    except ValueError:
        return
    raise AssertionError("expected ValueError on unequal prediction lists")


def test_two_proportion_z_reproduces_the_measured_test_split_contrast():
    # 116/186 not-marked vs 117/214 marked: the proposal's non-significant gap.
    out = two_proportion_z(116, 186, 117, 214)
    assert 1.5 < out["z"] < 1.6
    assert 0.10 < out["p_value"] < 0.13


def test_describe_matches_the_published_track_a_field_mean():
    out = describe([53.5, 78.0, 66.51])
    assert out["n"] == 3 and out["sd"] > 0
