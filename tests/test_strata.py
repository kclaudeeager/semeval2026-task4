from task4.dataset import Triple
from task4.strata import Stratum, aspect_stratum, partition, usable


def _t(outcomes):
    return Triple(anchor="a", text_a="b", text_b="c", a_is_closer=True, outcomes=outcomes)


def test_unanimous_yes_and_no_are_separated():
    assert aspect_stratum(_t([True, True]), "outcomes") is Stratum.ALL_MARKED
    assert aspect_stratum(_t([False, False]), "outcomes") is Stratum.NONE_MARKED


def test_a_one_of_two_split_is_never_counted_as_not_marked():
    # The failure the proposal reports: a majority rule hides these in "not marked".
    assert aspect_stratum(_t([True, False]), "outcomes") is Stratum.SPLIT


def test_missing_annotations_fall_into_split():
    assert aspect_stratum(_t([]), "outcomes") is Stratum.SPLIT


def test_partition_covers_every_item_exactly_once():
    rows = [_t([True, True]), _t([False, False]), _t([True, False])]
    idx = partition(rows, "outcomes")
    assert sorted(i for v in idx.values() for i in v) == [0, 1, 2]


def test_usable_requires_both_unanimous_strata_to_be_large_enough():
    lopsided = [_t([True, True])] * 100 + [_t([False, False])] * 2
    assert not usable(lopsided, "outcomes", min_per_stratum=30)
    balanced = [_t([True, True])] * 50 + [_t([False, False])] * 50
    assert usable(balanced, "outcomes", min_per_stratum=30)


def test_unknown_aspect_is_rejected():
    try:
        aspect_stratum(_t([True]), "plot_twist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown aspect")
