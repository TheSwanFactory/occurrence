"""Classify / seal / resolve witness (Theory 39 fold-sufficiency).

Public classification is the FIPS incidence label (P, delta) shared by the three
Events of a certified basic block (Theory 40 / 068 finite geometry). Sealed
EventDenotation objects carry the evaluator ray. Classify-only PublicEventView
objects cannot execute Event action.

Provenance for the collision fixture: the finite basic Event design and
(P, delta) classifier used by experiments/mutual-frames/audit.py (Theory 068.02
census), which inherits the Fano-indexed Pasch / FIPS label conventions of
Theory 40. Two distinct Events from one certified block share (P, delta) while
retaining distinct sealed evaluations — the minimal public-collision witness
required by Issue 005 §1.3 (S1–S3).
"""

from __future__ import annotations

from dataclasses import dataclass

from topographo.ssd import exact
from topographo.ssd.frames import is_event, ray


@dataclass(frozen=True)
class PublicEventView:
    """Classify-only public incidence; insufficient for Event action."""

    label: tuple[int, int]


@dataclass(frozen=True)
class EventDenotation:
    """Sealed projective Event ray with opaque evaluator binding."""

    _ray: exact.Value
    binding: str = "default"

    def __post_init__(self) -> None:
        value = ray(exact.checked(self._ray, where="EventDenotation"))
        if not is_event(value):
            raise ValueError("EventDenotation requires a certified Event ray")
        object.__setattr__(self, "_ray", value)

    @property
    def ray(self) -> exact.Value:
        return self._ray


@dataclass(frozen=True)
class MissingSealedDenotation:
    """Classify-only or bare public view offered where a seal is required."""

    detail: str = ""


def public_label(event: exact.Value) -> tuple[int, int]:
    """Return the FIPS (P, delta) incidence label for a basic Event ray."""

    event = ray(exact.checked(event, where="public_label"))
    if not is_event(event):
        raise ValueError("public_label requires an Event")
    i = next(idx for idx in range(1, 8) if event[idx])
    j = next(idx for idx in range(1, 8) if event[8 + idx])
    p = i ^ j
    sigma = exact.mul(exact.basis(i), exact.basis(j))[p]
    return p, int(event[8 + j] * sigma)


def classify(denotation: EventDenotation | PublicEventView | exact.Value) -> PublicEventView:
    """Project a sealed denotation (or ray) to its public incidence view."""

    if isinstance(denotation, PublicEventView):
        return denotation
    if isinstance(denotation, EventDenotation):
        return PublicEventView(public_label(denotation.ray))
    return PublicEventView(public_label(denotation))


def seal(event: exact.Value, *, binding: str = "default") -> EventDenotation:
    """Seal an Event ray into an opaque evaluator denotation."""

    return EventDenotation(event, binding=binding)


def resolve(denotation: EventDenotation) -> exact.Value:
    """Recover the sealed evaluator ray for Event action."""

    if not isinstance(denotation, EventDenotation):
        raise TypeError("resolve requires EventDenotation")
    return denotation.ray


def evaluate_left(denotation: EventDenotation, state: exact.Value) -> exact.Value | None:
    """Event action under a sealed denotation; None on projective annihilation."""

    from topographo.ssd.frames import action

    return action(resolve(denotation), ray(state) if state != exact.zero() else state)


# ---------------------------------------------------------------------------
# Theory-39 public-collision witness (S1–S3)
# ---------------------------------------------------------------------------

def _collision_pair() -> tuple[EventDenotation, EventDenotation]:
    """Two sealed Events from one FIPS block sharing (P, delta)."""

    from topographo.ssd import fips_basic

    block = fips_basic.BLOCKS[0]
    a, b, _c = block
    left, right = seal(a, binding="witness-a"), seal(b, binding="witness-b")
    assert classify(left) == classify(right)
    assert resolve(left) != resolve(right)
    return left, right


@dataclass(frozen=True)
class CollisionWitness:
    """Documented S1 fixture: same public class, distinct sealed evaluations."""

    left: EventDenotation
    right: EventDenotation
    provenance: str = (
        "Theory 39 fold-sufficiency via Theory 40/068 FIPS (P,delta) collision: "
        "two Events of one certified basic block share public incidence while "
        "retaining distinct evaluator rays (experiments/mutual-frames/audit.py)."
    )


def collision_witness() -> CollisionWitness:
    left, right = _collision_pair()
    return CollisionWitness(left=left, right=right)
