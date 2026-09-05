"""Exact rational sedenions and explicitly ordered event programs.

Values are immutable tuples of sixteen :class:`fractions.Fraction`
coefficients in the ordered basis ``e0, ..., e15``.  This module implements
the convention pinned by the Occurrence Theory 061.11 programming handoff::

    conj((a, b)) = (conj(a), -b)
    (a, b)(c, d) = (ac - d*conj(b), conj(a)*d + c*b)

The exact machine exposes the coordinate presentation pinned by programming
handoff 061.11. Multiplication is derived from the package's single exact
signed-basis table in core coordinates through the explicit map
``Phi(a, b) = (conj(a), b)``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, TypeAlias

from topographo.core.cayley_dickson import signed_basis_table

DIMENSION = 16
CONVENTION = "(a,b)(c,d) = (ac - d*conj(b), conj(a)*d + c*b)"
Value: TypeAlias = tuple[Fraction, ...]
FractionInput: TypeAlias = Fraction | int | str
JsonObject: TypeAlias = dict[str, Any]

_COORDINATE_SIGNS = (1,) + (-1,) * 7 + (1,) * 8
_CORE_TABLE = signed_basis_table(DIMENSION)


def _fraction(raw: FractionInput, *, where: str) -> Fraction:
    if isinstance(raw, bool) or not isinstance(raw, (Fraction, int, str)):
        raise TypeError(
            f"{where} must be a Fraction, integer, or rational string; "
            f"got {type(raw).__name__}"
        )
    try:
        return Fraction(raw)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{where} is not a valid rational: {raw!r}") from error


def value(coefficients: Sequence[FractionInput]) -> Value:
    """Construct a checked value from sixteen exact rational coefficients."""

    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence):
        raise TypeError("coefficients must be a sequence of 16 rationals")
    if len(coefficients) != DIMENSION:
        raise ValueError(
            f"sedenion values require {DIMENSION} coefficients; got {len(coefficients)}"
        )
    return tuple(
        _fraction(coefficient, where=f"coefficient[{index}]")
        for index, coefficient in enumerate(coefficients)
    )


def _checked(candidate: Value, *, where: str = "value") -> Value:
    if not isinstance(candidate, tuple):
        raise TypeError(f"{where} must be an immutable tuple of 16 Fractions")
    if len(candidate) != DIMENSION:
        raise ValueError(
            f"{where} must have {DIMENSION} coefficients; got {len(candidate)}"
        )
    for index, coefficient in enumerate(candidate):
        if not isinstance(coefficient, Fraction):
            raise TypeError(
                f"{where}[{index}] must be Fraction; got {type(coefficient).__name__}"
            )
    return candidate


def zero() -> Value:
    """Return the additive identity."""

    return (Fraction(0),) * DIMENSION


def one() -> Value:
    """Return the multiplicative identity."""

    return basis(0)


def basis(index: int) -> Value:
    """Return the requested ordered basis vector."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("basis index must be an integer")
    if not 0 <= index < DIMENSION:
        raise ValueError(f"basis index must be in [0, {DIMENSION}); got {index}")
    return tuple(Fraction(int(position == index)) for position in range(DIMENSION))


def add(left: Value, right: Value) -> Value:
    """Add two exact values coefficient-wise."""

    left = _checked(left, where="left operand")
    right = _checked(right, where="right operand")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def sub(left: Value, right: Value) -> Value:
    """Subtract two exact values coefficient-wise."""

    left = _checked(left, where="left operand")
    right = _checked(right, where="right operand")
    return tuple(a - b for a, b in zip(left, right, strict=True))


def scale(scalar: FractionInput, operand: Value) -> Value:
    """Scale an exact value by a rational."""

    factor = _fraction(scalar, where="scale")
    operand = _checked(operand, where="operand")
    return tuple(factor * coefficient for coefficient in operand)


def _conj(operand: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    if len(operand) == 1:
        return operand
    halfway = len(operand) // 2
    first, second = operand[:halfway], operand[halfway:]
    return _conj(first) + tuple(-coefficient for coefficient in second)


def conj(operand: Value) -> Value:
    """Return Cayley--Dickson conjugation under the pinned convention."""

    return _conj(_checked(operand, where="operand"))


def to_core_coordinates(operand: Value) -> Value:
    """Apply ``Phi(a, b) = (conj(a), b)`` to a 061.11 value."""

    operand = _checked(operand, where="operand")
    return tuple(
        sign * coefficient
        for sign, coefficient in zip(_COORDINATE_SIGNS, operand, strict=True)
    )


def from_core_coordinates(operand: Value) -> Value:
    """Convert a core-coordinate value to the public 061.11 presentation."""

    operand = _checked(operand, where="operand")
    return tuple(
        sign * coefficient
        for sign, coefficient in zip(_COORDINATE_SIGNS, operand, strict=True)
    )


def mul(left: Value, right: Value) -> Value:
    """Multiply two 061.11 values through the shared exact core table."""

    core_left = to_core_coordinates(_checked(left, where="left operand"))
    core_right = to_core_coordinates(_checked(right, where="right operand"))
    product = [Fraction(0) for _ in range(DIMENSION)]
    for left_index, left_coefficient in enumerate(core_left):
        if not left_coefficient:
            continue
        for right_index, right_coefficient in enumerate(core_right):
            if not right_coefficient:
                continue
            result, sign = _CORE_TABLE[left_index][right_index]
            product[result] += sign * left_coefficient * right_coefficient
    return from_core_coordinates(tuple(product))


def norm2(operand: Value) -> Fraction:
    """Return the exact Euclidean squared norm."""

    operand = _checked(operand, where="operand")
    return sum((coefficient * coefficient for coefficient in operand), Fraction(0))


def equal(left: Value, right: Value) -> bool:
    """Compare two checked exact values."""

    return _checked(left, where="left operand") == _checked(
        right, where="right operand"
    )


def _fraction_json(number: Fraction) -> str:
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def value_to_json(operand: Value) -> list[str]:
    """Encode a value as canonical rational strings."""

    return [_fraction_json(number) for number in _checked(operand)]


def value_from_json(raw: Any, *, where: str = "value") -> Value:
    """Decode a value from canonical rational strings."""

    if not isinstance(raw, list):
        raise TypeError(f"{where} must be a JSON array of 16 rational strings")
    try:
        return value(raw)
    except (TypeError, ValueError) as error:
        raise type(error)(f"invalid {where}: {error}") from error


def _node(raw: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{where} must be a JSON object")
    return raw


def evaluate(expression: Mapping[str, Any]) -> Value:
    """Evaluate a fully parenthesized JSON expression without reassociation."""

    node = _node(expression, where="expression")
    operation = node.get("op")
    if operation == "literal":
        return value_from_json(node.get("value"), where="literal value")
    if operation == "conj":
        return conj(evaluate(_node(node.get("arg"), where="conj.arg")))
    if operation in {"add", "sub", "mul"}:
        left = evaluate(_node(node.get("left"), where=f"{operation}.left"))
        right = evaluate(_node(node.get("right"), where=f"{operation}.right"))
        return {"add": add, "sub": sub, "mul": mul}[operation](left, right)
    raise ValueError(f"unsupported expression operation: {operation!r}")


def literal(operand: Value) -> JsonObject:
    """Build an exact literal expression node."""

    return {"op": "literal", "value": value_to_json(operand)}


@dataclass(frozen=True)
class TraceStep:
    """One ordered event transition."""

    event: Value
    before: Value
    after: Value
    norm2: Fraction


@dataclass(frozen=True)
class RunResult:
    """Final state and complete ordered trace."""

    state: Value
    trace: tuple[TraceStep, ...]


def run(initial: Value, events: Sequence[Value]) -> RunResult:
    """Left-multiply each event in order and record every transition."""

    state = _checked(initial, where="initial state")
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of sedenion values")
    trace: list[TraceStep] = []
    for index, raw_event in enumerate(events):
        event = _checked(raw_event, where=f"event[{index}]")
        before = state
        state = mul(event, state)
        trace.append(TraceStep(event, before, state, norm2(state)))
    return RunResult(state, tuple(trace))


def run_to_json(
    initial: Value, events: Sequence[Value], result: RunResult
) -> JsonObject:
    """Encode a complete exact run as a JSON-compatible object."""

    initial = _checked(initial, where="initial state")
    checked_events = tuple(
        _checked(event, where=f"event[{index}]") for index, event in enumerate(events)
    )
    return {
        "initial": value_to_json(initial),
        "events": [value_to_json(event) for event in checked_events],
        "state": value_to_json(result.state),
        "trace": [
            {
                "event": value_to_json(step.event),
                "before": value_to_json(step.before),
                "after": value_to_json(step.after),
                "norm2": _fraction_json(step.norm2),
            }
            for step in result.trace
        ],
    }


def serialize_run(initial: Value, events: Sequence[Value]) -> str:
    """Serialize an ordered run, including its exact endpoint and trace."""

    result = run(initial, events)
    return json.dumps(run_to_json(initial, events, result), indent=2, sort_keys=True)


def replay_serialized(document: str) -> RunResult:
    """Replay a serialized run and reject any altered endpoint or trace."""

    try:
        raw = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise TypeError("serialized run must contain a JSON object")
    initial = value_from_json(raw.get("initial"), where="initial")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("events must be a JSON array")
    events = [
        value_from_json(event, where=f"events[{index}]")
        for index, event in enumerate(raw_events)
    ]
    replayed = run(initial, events)
    expected = run_to_json(initial, events, replayed)
    if raw != expected:
        raise ValueError("serialized endpoint or trace does not match exact replay")
    return replayed


def canonicalize(operand: Value) -> Value:
    """Choose the representative whose first nonzero coefficient is one."""

    operand = _checked(operand, where="ray value")
    for coefficient in operand:
        if coefficient:
            return scale(1 / coefficient, operand)
    raise ValueError("cannot canonicalize the zero vector")


@dataclass(frozen=True)
class RayResult:
    """Canonical endpoint and trace for a non-annihilated ray run."""

    state: Value
    trace: tuple[TraceStep, ...]


@dataclass(frozen=True)
class Annihilated:
    """Explicit terminal result when an event product becomes zero."""

    step: int
    trace: tuple[TraceStep, ...]


def run_ray(initial: Value, events: Sequence[Value]) -> RayResult | Annihilated:
    """Run on canonical rays, stopping explicitly when a product is zero."""

    state = canonicalize(initial)
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of sedenion values")
    trace: list[TraceStep] = []
    for index, raw_event in enumerate(events, start=1):
        event = _checked(raw_event, where=f"event[{index - 1}]")
        before = state
        if event == zero():
            after = zero()
        else:
            event = canonicalize(event)
            after = mul(event, state)
        if after == zero():
            trace.append(TraceStep(event, before, after, Fraction(0)))
            return Annihilated(index, tuple(trace))
        state = canonicalize(after)
        trace.append(TraceStep(event, before, state, norm2(state)))
    return RayResult(state, tuple(trace))
