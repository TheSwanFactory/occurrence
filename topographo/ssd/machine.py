"""Ordered left-action execution and policy-independent outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from topographo.ssd import exact
from topographo.ssd.program import Program, as_program


@dataclass(frozen=True, init=False)
class Transition:
    """The policy-selected event, state before it, and state after it.

    Squared norm remains derived rather than stored. The read-only ``norm2``
    property and optional fourth constructor argument preserve the original
    ``TraceStep`` compatibility surface without adding it to transition data.
    """

    event: exact.Value
    before: exact.Value
    after: exact.Value

    def __init__(
        self,
        event: exact.Value,
        before: exact.Value,
        after: exact.Value,
        norm2: Fraction | None = None,
    ) -> None:
        object.__setattr__(self, "event", exact.checked(event, where="event"))
        object.__setattr__(self, "before", exact.checked(before, where="before"))
        object.__setattr__(self, "after", exact.checked(after, where="after"))
        if norm2 is not None and norm2 != self.norm2:
            raise ValueError("norm2 does not match the transition's after state")

    @property
    def norm2(self) -> Fraction:
        """Return the derived squared norm for legacy ``TraceStep`` callers."""

        return exact.norm2(self.after)


@dataclass(frozen=True)
class Completed:
    """A successful endpoint and its complete ordered transition trace."""

    state: exact.Value
    trace: tuple[Transition, ...]


@dataclass(frozen=True)
class Annihilated:
    """A terminal zero product, including its one-based event step."""

    step: int
    trace: tuple[Transition, ...]


class StatePolicy(Protocol):
    """State preparation and zero handling around the common action engine."""

    def prepare_initial(self, initial: exact.Value) -> exact.Value: ...

    def prepare_event(self, event: exact.Value, index: int) -> exact.Value: ...

    def resolve_product(self, product: exact.Value) -> exact.Value | None: ...


@dataclass(frozen=True)
class RawVectorPolicy:
    """Keep exact vectors unchanged and allow execution through zero."""

    def prepare_initial(self, initial: exact.Value) -> exact.Value:
        return exact.checked(initial, where="initial state")

    def prepare_event(self, event: exact.Value, index: int) -> exact.Value:
        return exact.checked(event, where=f"event[{index}]")

    def resolve_product(self, product: exact.Value) -> exact.Value:
        return exact.checked(product, where="product")


def execute(
    initial: exact.Value,
    program: Program,
    policy: StatePolicy,
) -> Completed | Annihilated:
    """Execute one typed program through a supplied state-space policy.

    For events ``[z1, z2]`` the only algebraic steps are ``z1 * initial`` and
    then ``z2 * previous``. Events are never multiplied together or reassociated.
    """

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    state = policy.prepare_initial(initial)
    trace: list[Transition] = []
    for step, raw_event in enumerate(program.events, start=1):
        event = policy.prepare_event(raw_event, step - 1)
        before = state
        product = exact.mul(event, state)
        after = policy.resolve_product(product)
        if after is None:
            trace.append(Transition(event, before, product))
            return Annihilated(step, tuple(trace))
        trace.append(Transition(event, before, after))
        state = after
    return Completed(state, tuple(trace))


def run(
    initial: exact.Value,
    events: Program | Sequence[exact.Value],
) -> Completed:
    """Run an ordered program in raw-vector state space."""

    outcome = execute(initial, as_program(events), RawVectorPolicy())
    if isinstance(outcome, Annihilated):  # pragma: no cover - policy invariant
        raise RuntimeError("raw-vector execution cannot terminate on zero")
    return outcome
