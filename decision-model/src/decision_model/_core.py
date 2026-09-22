"""Backend-neutral state, effect, test, and resolution contracts."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from ._errors import BackendResolutionError, InvalidResolutionError, ResolutionError

__all__ = (
    "Distribution",
    "Effect",
    "Probability",
    "Resolver",
    "State",
    "Test",
    "resolve_effect",
    "resolve_test",
)

State = TypeVar("State", contravariant=True)
_ResolverState = TypeVar("_ResolverState")
_Label = TypeVar("_Label", bound=Hashable)
ProbabilityValue: TypeAlias = int | float | Fraction


def _finite_real(value: object, *, name: str) -> ProbabilityValue:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return cast(ProbabilityValue, value)


@dataclass(frozen=True, slots=True)
class Probability:
    """An immutable finite real value in the closed interval [0, 1]."""

    value: ProbabilityValue

    def __post_init__(self) -> None:
        value = _finite_real(self.value, name="probability")
        if value < 0 or value > 1:
            raise ValueError("probability must be in the closed interval [0, 1]")

    def __float__(self) -> float:
        return float(self.value)


@runtime_checkable
class Effect(Protocol[State]):
    """A callable effect that evaluates an already-constituted state."""

    def __call__(self, state: State, /) -> Probability | ProbabilityValue:
        """Evaluate *state* and return a probability value."""
        ...


def _labeled_values(
    values: Iterable[tuple[_Label, Any]] | Mapping[_Label, Any],
    *,
    kind: str,
) -> tuple[tuple[_Label, Any], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{kind} entries must be an iterable of (label, value) pairs")

    source: Iterable[object]
    if isinstance(values, Mapping):
        source = values.items()
    else:
        source = values

    try:
        entries = tuple(source)
    except TypeError as exc:
        raise TypeError(
            f"{kind} entries must be an iterable of (label, value) pairs"
        ) from exc

    if not entries:
        raise ValueError(f"{kind} must contain at least one labeled entry")

    result: list[tuple[_Label, Any]] = []
    seen: set[Hashable] = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, (str, bytes)):
            raise TypeError(f"{kind} entry {index} must be a (label, value) pair")
        try:
            pair = tuple(cast(Iterable[Any], entry))
        except TypeError as exc:
            raise TypeError(
                f"{kind} entry {index} must be a (label, value) pair"
            ) from exc
        if len(pair) != 2:
            raise TypeError(f"{kind} entry {index} must be a (label, value) pair")
        label, value = pair
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError(f"{kind} label at index {index} must be hashable") from exc
        if label in seen:
            raise ValueError(f"{kind} labels must be unique; duplicate {label!r}")
        seen.add(label)
        result.append((cast(_Label, label), value))
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class Test(Generic[State, _Label]):
    """An immutable, nonempty, labeled exhaustive family of callable effects."""

    __test__ = False
    effects: tuple[tuple[_Label, Effect[State]], ...]

    def __init__(
        self,
        effects: (
            Iterable[tuple[_Label, Effect[State]]]
            | Mapping[_Label, Effect[State]]
        ),
    ) -> None:
        entries = _labeled_values(effects, kind="test")
        checked: list[tuple[_Label, Effect[State]]] = []
        for index, (label, effect) in enumerate(entries):
            if not callable(effect):
                raise TypeError(f"test effect at index {index} must be callable")
            checked.append((label, cast(Effect[State], effect)))
        object.__setattr__(self, "effects", tuple(checked))

    @property
    def labels(self) -> tuple[_Label, ...]:
        """Labels in their declared resolution order."""
        return tuple(label for label, _ in self.effects)

    def __iter__(self) -> Iterator[tuple[_Label, Effect[State]]]:
        return iter(self.effects)

    def __len__(self) -> int:
        return len(self.effects)

    def __getitem__(self, label: _Label) -> Effect[State]:
        for candidate, effect in self.effects:
            if candidate == label:
                return effect
        raise KeyError(label)


@dataclass(frozen=True, slots=True, init=False)
class Distribution(Generic[_Label]):
    """An immutable labeled probability distribution with visible tolerance."""

    outcomes: tuple[tuple[_Label, Probability], ...]
    tolerance: ProbabilityValue

    def __init__(
        self,
        outcomes: (
            Iterable[tuple[_Label, Probability | ProbabilityValue]]
            | Mapping[_Label, Probability | ProbabilityValue]
        ),
        *,
        tolerance: ProbabilityValue = 0,
    ) -> None:
        entries = _labeled_values(outcomes, kind="distribution")
        checked = tuple(
            (label, _as_probability(value)) for label, value in entries
        )
        checked_tolerance = _finite_real(tolerance, name="tolerance")
        if checked_tolerance < 0:
            raise ValueError("tolerance must be nonnegative")

        numeric_total: Any = sum(
            (probability.value for _, probability in checked),
            0,
        )
        total = cast(ProbabilityValue, numeric_total)
        if abs(numeric_total - 1) > checked_tolerance:
            raise ValueError(
                "distribution probabilities must sum to 1 within the explicit "
                f"tolerance {checked_tolerance!r}; got {total!r}"
            )

        object.__setattr__(self, "outcomes", checked)
        object.__setattr__(self, "tolerance", checked_tolerance)

    @property
    def labels(self) -> tuple[_Label, ...]:
        """Labels in their declared order."""
        return tuple(label for label, _ in self.outcomes)

    @property
    def probabilities(self) -> tuple[Probability, ...]:
        """Probabilities in label order, without normalization or repair."""
        return tuple(probability for _, probability in self.outcomes)

    @property
    def total(self) -> ProbabilityValue:
        """The stored probability total, without normalization or repair."""
        return sum((probability.value for _, probability in self.outcomes), 0)

    def __iter__(self) -> Iterator[tuple[_Label, Probability]]:
        return iter(self.outcomes)

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, label: _Label) -> Probability:
        for candidate, probability in self.outcomes:
            if candidate == label:
                return probability
        raise KeyError(label)


def _as_probability(value: Probability | ProbabilityValue) -> Probability:
    if isinstance(value, Probability):
        return value
    return Probability(value)


@runtime_checkable
class Resolver(Protocol[_ResolverState, _Label]):
    """Structural interface implemented by state/effect resolution backends."""

    def resolve_effect(
        self,
        state: _ResolverState,
        effect: Effect[_ResolverState],
        /,
    ) -> Probability:
        """Resolve one effect against an already-constituted state."""
        ...

    def resolve_test(
        self,
        state: _ResolverState,
        test: Test[_ResolverState, _Label],
        /,
    ) -> Distribution[_Label]:
        """Resolve an exhaustive test against an already-constituted state."""
        ...


def resolve_effect(state: State, effect: Effect[State], /) -> Probability:
    """Resolve a callable effect, validating its probability endpoint."""
    if not callable(effect):
        raise TypeError("effect must satisfy the callable Effect protocol")
    try:
        value = effect(state)
    except ResolutionError:
        raise
    except Exception as exc:
        raise BackendResolutionError("effect evaluation failed") from exc

    try:
        return _as_probability(value)
    except (TypeError, ValueError) as exc:
        raise InvalidResolutionError(
            "effect did not return a valid Probability or real value in [0, 1]"
        ) from exc


def resolve_test(state: State, test: Test[State, _Label], /) -> Distribution[_Label]:
    """Resolve every effect in a test and require an unmodified unit total."""
    if not isinstance(test, Test):
        raise TypeError("test must be a Test instance")

    outcomes = tuple(
        (label, resolve_effect(state, effect)) for label, effect in test
    )
    try:
        return Distribution(outcomes)
    except (TypeError, ValueError) as exc:
        raise InvalidResolutionError(
            "resolved test probabilities do not form a normalized distribution"
        ) from exc
