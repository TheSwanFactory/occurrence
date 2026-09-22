from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from decision_model import Distribution, Probability, Test


def _constant(value: object):
    return lambda _state: value


@pytest.mark.parametrize("value", [0, 1, Fraction(1, 3), 0.25])
def test_probability_accepts_closed_unit_interval(value: object) -> None:
    probability = Probability(value)  # type: ignore[arg-type]
    assert probability.value == value
    assert float(probability) == float(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, True, False, "0.5", 1j])
def test_probability_rejects_non_real_values(value: object) -> None:
    with pytest.raises(TypeError):
        Probability(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, -1e-12, 1.000000000001, 2])
def test_probability_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Probability(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_probability_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Probability(value)


def test_probability_is_immutable() -> None:
    probability = Probability(Fraction(1, 2))
    with pytest.raises(FrozenInstanceError):
        probability.value = Fraction(1, 3)  # type: ignore[misc]


def test_test_is_ordered_labeled_nonempty_and_detached_from_input() -> None:
    first = _constant(Probability(1))
    second = _constant(Probability(0))
    source = [("yes", first), ("no", second)]

    test = Test(source)
    source.reverse()

    assert test.effects == (("yes", first), ("no", second))
    assert test.labels == ("yes", "no")
    assert tuple(test) == test.effects
    assert len(test) == 2
    assert test["yes"] is first
    with pytest.raises(KeyError):
        test["missing"]


def test_test_accepts_an_insertion_ordered_mapping() -> None:
    first = _constant(1)
    second = _constant(0)
    test = Test({"first": first, "second": second})
    assert test.effects == (("first", first), ("second", second))


def test_test_is_immutable() -> None:
    test = Test((("only", _constant(1)),))
    with pytest.raises(FrozenInstanceError):
        test.effects = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("entries", "error"),
    [
        ((), ValueError),
        ((("same", _constant(1)), ("same", _constant(0))), ValueError),
        ((([], _constant(1)),), TypeError),
        ((("bad", 42),), TypeError),
        (("not-a-pair",), TypeError),
        ((("too", _constant(1), "many"),), TypeError),
    ],
)
def test_test_rejects_malformed_families(entries: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        Test(entries)  # type: ignore[arg-type]


def test_distribution_preserves_exact_labeled_probabilities() -> None:
    one_third = Probability(Fraction(1, 3))
    distribution = Distribution(
        (("yes", one_third), ("no", Fraction(2, 3)))
    )

    assert distribution.outcomes == (
        ("yes", one_third),
        ("no", Probability(Fraction(2, 3))),
    )
    assert distribution.labels == ("yes", "no")
    assert distribution.probabilities == (
        one_third,
        Probability(Fraction(2, 3)),
    )
    assert distribution.total == 1
    assert distribution.tolerance == 0
    assert distribution["yes"] is one_third
    assert tuple(distribution) == distribution.outcomes
    assert len(distribution) == 2
    with pytest.raises(KeyError):
        distribution["missing"]


def test_distribution_accepts_insertion_ordered_mapping() -> None:
    distribution = Distribution({"zero": 0, "one": 1})
    assert distribution.labels == ("zero", "one")


def test_distribution_accepts_visible_tolerance_without_renormalizing() -> None:
    original = (0.5000001, 0.5)
    distribution = Distribution(
        (("left", original[0]), ("right", original[1])),
        tolerance=2e-7,
    )

    assert tuple(probability.value for probability in distribution.probabilities) == original
    assert distribution.total == sum(original)
    assert distribution.total != 1
    assert distribution.tolerance == 2e-7


def test_distribution_rejects_total_outside_explicit_tolerance() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Distribution((("left", 0.6), ("right", 0.5)), tolerance=0.09)


def test_distribution_is_exact_by_default() -> None:
    with pytest.raises(ValueError, match="tolerance 0"):
        Distribution((("left", 0.5000001), ("right", 0.5)))


@pytest.mark.parametrize(
    ("entries", "error"),
    [
        ((), ValueError),
        ((("same", 1), ("same", 0)), ValueError),
        ((([], 1),), TypeError),
        (("not-a-pair",), TypeError),
        ((("too", 0, "many"),), TypeError),
        ((("invalid-probability", 2),), ValueError),
    ],
)
def test_distribution_rejects_malformed_outcomes(
    entries: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        Distribution(entries)  # type: ignore[arg-type]


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_distribution_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError):
        Distribution((("only", 1),), tolerance=tolerance)


def test_distribution_rejects_boolean_tolerance() -> None:
    with pytest.raises(TypeError):
        Distribution((("only", 1),), tolerance=True)


def test_distribution_is_immutable() -> None:
    distribution = Distribution((("only", 1),))
    with pytest.raises(FrozenInstanceError):
        distribution.outcomes = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        distribution.tolerance = 1  # type: ignore[misc]
