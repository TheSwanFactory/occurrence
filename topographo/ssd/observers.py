"""Derived observations over machine transitions and outcomes."""

from __future__ import annotations

from fractions import Fraction

from topographo.ssd import exact
from topographo.ssd.machine import Transition


def squared_norm(transition: Transition) -> Fraction:
    """Measure the exact squared norm of a transition's resulting state."""

    return exact.norm2(transition.after)
