from __future__ import annotations

from fractions import Fraction

import pytest

from decision_model import (
    AnnihilationError,
    BackendResolutionError,
    Distribution,
    Effect,
    InvalidResolutionError,
    Probability,
    Resolver,
    Test,
    resolve_effect,
    resolve_test,
)


def test_function_is_a_runtime_callable_effect() -> None:
    def half(_state: object) -> Fraction:
        return Fraction(1, 2)

    assert isinstance(half, Effect)
    assert not isinstance(object(), Effect)
    assert resolve_effect(object(), half) == Probability(Fraction(1, 2))


def test_callable_object_is_an_effect() -> None:
    class FieldEffect:
        def __init__(self, field: str) -> None:
            self.field = field

        def __call__(self, state: dict[str, Fraction]) -> Fraction:
            return state[self.field]

    effect = FieldEffect("probability")
    assert isinstance(effect, Effect)
    assert resolve_effect({"probability": Fraction(2, 5)}, effect) == Probability(
        Fraction(2, 5)
    )


def test_resolve_effect_preserves_probability_identity() -> None:
    expected = Probability(Fraction(3, 7))
    assert resolve_effect(None, lambda _state: expected) is expected


def test_resolve_effect_rejects_noncallable_effect() -> None:
    with pytest.raises(TypeError, match="callable Effect"):
        resolve_effect(None, 1)  # type: ignore[arg-type]


def test_resolve_effect_wraps_backend_exception_and_preserves_cause() -> None:
    failure = RuntimeError("backend failed")

    def failing_effect(_state: object) -> Probability:
        raise failure

    with pytest.raises(BackendResolutionError) as captured:
        resolve_effect(None, failing_effect)
    assert captured.value.__cause__ is failure


@pytest.mark.parametrize("value", [-1, 2, float("nan"), "not a number"])
def test_resolve_effect_rejects_invalid_endpoint(value: object) -> None:
    with pytest.raises(InvalidResolutionError) as captured:
        resolve_effect(None, lambda _state: value)  # type: ignore[return-value]
    assert isinstance(captured.value.__cause__, (TypeError, ValueError))


def test_typed_resolution_failure_propagates_unchanged() -> None:
    failure = BackendResolutionError("typed backend failure")

    def failing_effect(_state: object) -> Probability:
        raise failure

    with pytest.raises(BackendResolutionError) as captured:
        resolve_effect(None, failing_effect)
    assert captured.value is failure


def test_zero_probability_is_distinct_from_annihilation() -> None:
    zero = resolve_effect(None, lambda _state: 0)
    assert zero == Probability(0)

    annihilation = AnnihilationError("state is in the effect kernel")

    def annihilating_effect(_state: object) -> Probability:
        raise annihilation

    with pytest.raises(AnnihilationError) as captured:
        resolve_effect(None, annihilating_effect)
    assert captured.value is annihilation
    assert captured.value != zero


def test_resolve_test_preserves_order_labels_and_exact_values() -> None:
    test = Test(
        (
            ("yes", lambda state: state),
            ("no", lambda state: 1 - state),
        )
    )

    distribution = resolve_test(Fraction(1, 3), test)

    assert isinstance(distribution, Distribution)
    assert distribution == Distribution(
        (("yes", Fraction(1, 3)), ("no", Fraction(2, 3)))
    )
    assert distribution.labels == test.labels


def test_resolve_test_does_not_normalize_weights() -> None:
    test = Test((("left", lambda _state: 0.2), ("right", lambda _state: 0.3)))

    with pytest.raises(InvalidResolutionError) as captured:
        resolve_test(None, test)
    assert isinstance(captured.value.__cause__, ValueError)


def test_resolve_test_propagates_typed_effect_failure_unchanged() -> None:
    failure = AnnihilationError("annihilated")

    def annihilate(_state: object) -> Probability:
        raise failure

    test = Test((("survives", lambda _state: 1), ("fails", annihilate)))
    with pytest.raises(AnnihilationError) as captured:
        resolve_test(None, test)
    assert captured.value is failure


def test_resolve_test_requires_validated_test() -> None:
    with pytest.raises(TypeError, match="Test instance"):
        resolve_test(None, (("only", lambda _state: 1),))  # type: ignore[arg-type]


class CompleteResolver:
    def resolve_effect(self, state: object, effect: Effect[object], /) -> Probability:
        return resolve_effect(state, effect)

    def resolve_test(
        self,
        state: object,
        test: Test[object, str],
        /,
    ) -> Distribution[str]:
        return resolve_test(state, test)


class IncompleteResolver:
    def resolve_effect(self, state: object, effect: Effect[object], /) -> Probability:
        return resolve_effect(state, effect)


def test_resolver_is_runtime_checkable_and_structural() -> None:
    assert isinstance(CompleteResolver(), Resolver)
    assert not isinstance(IncompleteResolver(), Resolver)
