"""Typed expressions and ordered left-action programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from topographo.ssd import exact


@dataclass(frozen=True)
class Literal:
    """An exact value embedded in an expression tree."""

    value: exact.Value

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", exact.checked(self.value, where="literal"))


@dataclass(frozen=True)
class Conjugate:
    """Conjugation of one explicitly nested expression."""

    argument: Expression


@dataclass(frozen=True)
class Add:
    """Addition of two explicitly nested expressions."""

    left: Expression
    right: Expression


@dataclass(frozen=True)
class Subtract:
    """Subtraction of two explicitly nested expressions."""

    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply:
    """Multiplication of two explicitly nested expressions."""

    left: Expression
    right: Expression


Expression: TypeAlias = Literal | Conjugate | Add | Subtract | Multiply


@dataclass(frozen=True, init=False)
class Program:
    """An immutable sequence of events applied by ordered left action."""

    events: tuple[exact.Value, ...]

    def __init__(self, events: Sequence[exact.Value]) -> None:
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise TypeError("events must be a sequence of sedenion values")
        object.__setattr__(
            self,
            "events",
            tuple(
                exact.checked(event, where=f"event[{index}]")
                for index, event in enumerate(events)
            ),
        )


def as_program(events: Program | Sequence[exact.Value]) -> Program:
    """Return an ordered program, preserving an existing typed instance."""

    return events if isinstance(events, Program) else Program(events)


def evaluate(expression: Expression) -> exact.Value:
    """Evaluate a typed expression tree without flattening or reassociation."""

    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Conjugate):
        return exact.conj(evaluate(expression.argument))
    if isinstance(expression, Add):
        return exact.add(evaluate(expression.left), evaluate(expression.right))
    if isinstance(expression, Subtract):
        return exact.sub(evaluate(expression.left), evaluate(expression.right))
    if isinstance(expression, Multiply):
        return exact.mul(evaluate(expression.left), evaluate(expression.right))
    raise TypeError(f"expression must be a typed node; got {type(expression).__name__}")
