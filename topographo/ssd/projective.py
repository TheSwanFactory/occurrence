"""Rational projective state policy and explicit annihilation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from topographo.ssd import exact
from topographo.ssd.machine import Annihilated, Completed, execute
from topographo.ssd.program import Program, as_program


def canonicalize(operand: exact.Value) -> exact.Value:
    """Choose the representative whose first nonzero coefficient is one.

    States are rational projective points: ``v`` and ``q*v`` are equivalent for
    every nonzero rational ``q``, including negative scalars. Zero has no
    projective representative.
    """

    operand = exact.checked(operand, where="projective value")
    for coefficient in operand:
        if coefficient:
            return exact.scale(1 / coefficient, operand)
    raise ValueError("cannot canonicalize the zero vector")


def equivalent(left: exact.Value, right: exact.Value) -> bool:
    """Return whether two nonzero vectors determine the same projective point."""

    return canonicalize(left) == canonicalize(right)


@dataclass(frozen=True)
class ProjectivePolicy:
    """Canonicalize nonzero states and stop explicitly at a zero product."""

    def prepare_initial(self, initial: exact.Value) -> exact.Value:
        return canonicalize(initial)

    def prepare_event(self, event: exact.Value, index: int) -> exact.Value:
        event = exact.checked(event, where=f"event[{index}]")
        return event if event == exact.zero() else canonicalize(event)

    def resolve_product(self, product: exact.Value) -> exact.Value | None:
        return None if product == exact.zero() else canonicalize(product)


def run_projective(
    initial: exact.Value,
    events: Program | Sequence[exact.Value],
) -> Completed | Annihilated:
    """Run an ordered program on rational projective points."""

    return execute(initial, as_program(events), ProjectivePolicy())
