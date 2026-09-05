"""Exact rational sedenions in the 061.11 coordinate presentation."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TypeAlias

from topographo.core.exact import (
    ExactCayleyDicksonAlgebra,
    ExactValue,
    FractionInput,
)

DIMENSION = 16
CONVENTION = "(a,b)(c,d) = (ac - d*conj(b), conj(a)*d + c*b)"
Value: TypeAlias = ExactValue

_ALGEBRA = ExactCayleyDicksonAlgebra(DIMENSION)
_COORDINATE_SIGNS = (1,) + (-1,) * 7 + (1,) * 8


def value(coefficients: Sequence[FractionInput]) -> Value:
    """Construct a checked 061.11 value from exact rational coefficients."""

    try:
        return _ALGEBRA.value(coefficients)
    except ValueError as error:
        message = str(error).replace(
            f"values require {DIMENSION} coefficients",
            f"sedenion values require {DIMENSION} coefficients",
        )
        raise ValueError(message) from error


def checked(candidate: Value, *, where: str = "value") -> Value:
    """Validate the immutable representation used by the exact SSD layers."""

    return _ALGEBRA.checked(candidate, where=where)


def zero() -> Value:
    """Return the additive identity."""

    return _ALGEBRA.zero()


def one() -> Value:
    """Return the multiplicative identity."""

    return basis(0)


def basis(index: int) -> Value:
    """Return a basis vector in the public 061.11 presentation."""

    return _ALGEBRA.basis(index)


def add(left: Value, right: Value) -> Value:
    """Add two exact values coefficient-wise."""

    return _ALGEBRA.add(left, right)


def sub(left: Value, right: Value) -> Value:
    """Subtract two exact values coefficient-wise."""

    return _ALGEBRA.sub(left, right)


def scale(scalar: FractionInput, operand: Value) -> Value:
    """Scale an exact value by a rational."""

    return _ALGEBRA.scale(scalar, operand)


def conj(operand: Value) -> Value:
    """Return Cayley--Dickson conjugation under the pinned convention."""

    return _ALGEBRA.conj(operand)


def to_core_coordinates(operand: Value) -> Value:
    """Apply ``Phi(a, b) = (conj(a), b)`` to a 061.11 value."""

    operand = checked(operand, where="operand")
    return tuple(
        sign * coefficient
        for sign, coefficient in zip(_COORDINATE_SIGNS, operand, strict=True)
    )


def from_core_coordinates(operand: Value) -> Value:
    """Convert a core-coordinate value to the public 061.11 presentation."""

    operand = checked(operand, where="operand")
    return tuple(
        sign * coefficient
        for sign, coefficient in zip(_COORDINATE_SIGNS, operand, strict=True)
    )


def mul(left: Value, right: Value) -> Value:
    """Multiply two 061.11 values through the shared exact core backend."""

    core_left = to_core_coordinates(checked(left, where="left operand"))
    core_right = to_core_coordinates(checked(right, where="right operand"))
    return from_core_coordinates(_ALGEBRA.mul(core_left, core_right))


def norm2(operand: Value) -> Fraction:
    """Return the exact Euclidean squared norm."""

    return _ALGEBRA.norm2(operand)


def equal(left: Value, right: Value) -> bool:
    """Compare two checked exact values."""

    return _ALGEBRA.equal(left, right)
