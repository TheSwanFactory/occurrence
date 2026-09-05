"""Exact rational backend for the shared Cayley--Dickson kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TypeAlias

from topographo.core.cayley_dickson import SignedBasisTable, signed_basis_table

ExactValue: TypeAlias = tuple[Fraction, ...]
FractionInput: TypeAlias = Fraction | int | str


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


@dataclass(frozen=True)
class ExactCayleyDicksonAlgebra:
    """Exact rational arithmetic derived from the canonical signed-basis table.

    Values use this backend's core coordinates. Presentation-specific coordinate
    maps belong in higher layers and must adapt through this class rather than
    defining another multiplication table.
    """

    dim: int
    _table: SignedBasisTable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_table", signed_basis_table(self.dim))

    def value(self, coefficients: Sequence[FractionInput]) -> ExactValue:
        """Construct a checked value with exactly ``dim`` coefficients."""

        if isinstance(coefficients, (str, bytes)) or not isinstance(
            coefficients, Sequence
        ):
            raise TypeError(f"coefficients must be a sequence of {self.dim} rationals")
        if len(coefficients) != self.dim:
            raise ValueError(
                f"values require {self.dim} coefficients; got {len(coefficients)}"
            )
        return tuple(
            _fraction(coefficient, where=f"coefficient[{index}]")
            for index, coefficient in enumerate(coefficients)
        )

    def checked(self, candidate: ExactValue, *, where: str = "value") -> ExactValue:
        """Validate the immutable exact representation used by this backend."""

        if not isinstance(candidate, tuple):
            raise TypeError(
                f"{where} must be an immutable tuple of {self.dim} Fractions"
            )
        if len(candidate) != self.dim:
            raise ValueError(
                f"{where} must have {self.dim} coefficients; got {len(candidate)}"
            )
        for index, coefficient in enumerate(candidate):
            if not isinstance(coefficient, Fraction):
                raise TypeError(
                    f"{where}[{index}] must be Fraction; "
                    f"got {type(coefficient).__name__}"
                )
        return candidate

    def zero(self) -> ExactValue:
        """Return the additive identity."""

        return (Fraction(0),) * self.dim

    def one(self) -> ExactValue:
        """Return the multiplicative identity."""

        return self.basis(0)

    def basis(self, index: int) -> ExactValue:
        """Return a core-coordinate basis vector."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("basis index must be an integer")
        if not 0 <= index < self.dim:
            raise ValueError(f"basis index must be in [0, {self.dim}); got {index}")
        return tuple(
            Fraction(int(position == index)) for position in range(self.dim)
        )

    def add(self, left: ExactValue, right: ExactValue) -> ExactValue:
        """Add two values coefficient-wise."""

        left = self.checked(left, where="left operand")
        right = self.checked(right, where="right operand")
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def sub(self, left: ExactValue, right: ExactValue) -> ExactValue:
        """Subtract two values coefficient-wise."""

        left = self.checked(left, where="left operand")
        right = self.checked(right, where="right operand")
        return tuple(a - b for a, b in zip(left, right, strict=True))

    def scale(self, scalar: FractionInput, operand: ExactValue) -> ExactValue:
        """Scale a value by an exact rational."""

        factor = _fraction(scalar, where="scale")
        operand = self.checked(operand, where="operand")
        return tuple(factor * coefficient for coefficient in operand)

    def conj(self, operand: ExactValue) -> ExactValue:
        """Return Cayley--Dickson conjugation in core coordinates."""

        operand = self.checked(operand, where="operand")
        return operand[:1] + tuple(-coefficient for coefficient in operand[1:])

    def mul(self, left: ExactValue, right: ExactValue) -> ExactValue:
        """Multiply through the package's canonical signed-basis table."""

        left = self.checked(left, where="left operand")
        right = self.checked(right, where="right operand")
        product = [Fraction(0) for _ in range(self.dim)]
        for left_index, left_coefficient in enumerate(left):
            if not left_coefficient:
                continue
            for right_index, right_coefficient in enumerate(right):
                if not right_coefficient:
                    continue
                result, sign = self._table[left_index][right_index]
                product[result] += sign * left_coefficient * right_coefficient
        return tuple(product)

    def norm2(self, operand: ExactValue) -> Fraction:
        """Return the exact Euclidean squared norm."""

        operand = self.checked(operand, where="operand")
        return sum(
            (coefficient * coefficient for coefficient in operand), Fraction(0)
        )

    def equal(self, left: ExactValue, right: ExactValue) -> bool:
        """Compare two checked values."""

        return self.checked(left, where="left operand") == self.checked(
            right, where="right operand"
        )
