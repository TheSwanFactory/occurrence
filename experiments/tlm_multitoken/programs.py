"""Strict OT program trees over exposed input signatures ``E^n R`` (Issue 008).

The `017.04` disposition certifies a **partial role-typed planar term calculus**
closed under partial substitution. Written as a grammar over the exposed-input
word, with ``e`` an Event leaf and ``r`` the retained leaf:

```text
E -> e | Cyc(E, E)
R -> r | Occ(E, R) | Sand(E, R)
```

Planar means the leaves are consumed left to right in the order presented; no
term reorders its inputs. A program for signature ``E^n R`` is therefore any
parse of the word ``e^n r`` deriving ``R``, and the exposed inputs are each used
exactly once.

Shape counts follow from the grammar. With ``c(k) = Catalan(k-1)`` event terms
over ``k`` consecutive Event leaves:

```text
t(0) = 1
t(n) = sum_{i=1..n} |{Occ, Sand}| * c(i) * t(n-i)
```

giving 2, 6, 20, 70 programs for ``E^1 R`` .. ``E^4 R`` with the full
constructor set, and 1, 2, 5, 14 when ``Sand`` is withheld. The 017 two-program
family is exactly the ``E^2 R`` Occ/Cyc pair: ``Occ(a, Occ(b, r))`` and
``Occ(Cyc(a, b), r)``.

Evaluation runs on the exact rational path in ``native.py``. Undefinedness
propagates as ``None``; nothing here is approximate and nothing is scored
against a float endpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

from experiments.tlm_multitoken import native
from experiments.tlm_multitoken.native import Ray

__all__ = [
    "CONSTRUCTORS",
    "Cyc",
    "EventLeaf",
    "EventTerm",
    "OCC_CYC_ONLY",
    "Occ",
    "RetainedLeaf",
    "RetainedTerm",
    "Sand",
    "Term",
    "count_programs",
    "enumerate_event_terms",
    "enumerate_programs",
    "evaluate",
    "event_leaf_indices",
    "kinds_used",
    "render",
    "signature",
]


@dataclass(frozen=True)
class EventLeaf:
    """The ``index``-th exposed Event input."""

    index: int


@dataclass(frozen=True)
class RetainedLeaf:
    """The single exposed retained input."""


@dataclass(frozen=True)
class Cyc:
    """``Cyc : (E,E) ⇀ E`` — forced third, undefined off the FIPS domain."""

    left: EventTerm
    right: EventTerm


@dataclass(frozen=True)
class Occ:
    """``Occ : (E,R) ⇀ R`` — projective left action."""

    event: EventTerm
    retained: RetainedTerm


@dataclass(frozen=True)
class Sand:
    """``Sand : (E,R) ⇀ R`` — bilateral action."""

    event: EventTerm
    retained: RetainedTerm


EventTerm: TypeAlias = "EventLeaf | Cyc"
RetainedTerm: TypeAlias = "RetainedLeaf | Occ | Sand"
Term: TypeAlias = "EventTerm | RetainedTerm"

#: Every certified constructor.
CONSTRUCTORS: frozenset[str] = frozenset({"Occ", "Cyc", "Sand"})

#: The 017 subset, kept so the two-program family is reproducible exactly.
OCC_CYC_ONLY: frozenset[str] = frozenset({"Occ", "Cyc"})

_RETAINED_HEADS: tuple[str, ...] = ("Occ", "Sand")


def signature(n_events: int) -> str:
    """Render the exposed-input signature, e.g. ``"E^3 R"``."""
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    if n_events == 0:
        return "R"
    if n_events == 1:
        return "E R"
    return f"E^{n_events} R"


@lru_cache(maxsize=None)
def enumerate_event_terms(
    start: int, count: int, constructors: frozenset[str] = CONSTRUCTORS
) -> tuple[EventTerm, ...]:
    """All type-``E`` terms spanning Event leaves ``start .. start+count-1``."""
    if count < 1:
        raise ValueError("an event term spans at least one Event leaf")
    if count == 1:
        return (EventLeaf(start),)
    if "Cyc" not in constructors:
        return ()
    out: list[EventTerm] = []
    for split in range(1, count):
        lefts = enumerate_event_terms(start, split, constructors)
        rights = enumerate_event_terms(start + split, count - split, constructors)
        for left in lefts:
            for right in rights:
                out.append(Cyc(left, right))
    return tuple(out)


@lru_cache(maxsize=None)
def enumerate_programs(
    n_events: int, constructors: frozenset[str] = CONSTRUCTORS
) -> tuple[RetainedTerm, ...]:
    """All type-``R`` programs for signature ``E^n R``, in a stable order."""
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    return _programs_from(0, n_events, constructors)


@lru_cache(maxsize=None)
def _programs_from(
    start: int, remaining: int, constructors: frozenset[str]
) -> tuple[RetainedTerm, ...]:
    if remaining == 0:
        return (RetainedLeaf(),)
    heads = [head for head in _RETAINED_HEADS if head in constructors]
    if not heads:
        return ()
    out: list[RetainedTerm] = []
    for take in range(1, remaining + 1):
        event_terms = enumerate_event_terms(start, take, constructors)
        if not event_terms:
            continue
        tails = _programs_from(start + take, remaining - take, constructors)
        for event_term in event_terms:
            for tail in tails:
                for head in heads:
                    out.append(
                        Occ(event_term, tail) if head == "Occ" else Sand(event_term, tail)
                    )
    return tuple(out)


def count_programs(n_events: int, constructors: frozenset[str] = CONSTRUCTORS) -> int:
    """Shape count for a signature, without materializing the terms."""
    return len(enumerate_programs(n_events, constructors))


def render(term: Term) -> str:
    """Human-readable term, e.g. ``Occ(Cyc(a1,a2),Occ(a3,r))``."""
    match term:
        case EventLeaf(index=index):
            return f"a{index + 1}"
        case RetainedLeaf():
            return "r"
        case Cyc(left=left, right=right):
            return f"Cyc({render(left)},{render(right)})"
        case Occ(event=event, retained=retained):
            return f"Occ({render(event)},{render(retained)})"
        case Sand(event=event, retained=retained):
            return f"Sand({render(event)},{render(retained)})"
    raise TypeError(f"not a term: {term!r}")


def event_leaf_indices(term: Term) -> tuple[int, ...]:
    """Event leaf indices in left-to-right order. Planarity check."""
    match term:
        case EventLeaf(index=index):
            return (index,)
        case RetainedLeaf():
            return ()
        case Cyc(left=left, right=right):
            return event_leaf_indices(left) + event_leaf_indices(right)
        case Occ(event=event, retained=retained) | Sand(event=event, retained=retained):
            return event_leaf_indices(event) + event_leaf_indices(retained)
    raise TypeError(f"not a term: {term!r}")


def kinds_used(term: Term) -> frozenset[str]:
    """Which constructors appear in the term."""
    match term:
        case EventLeaf() | RetainedLeaf():
            return frozenset()
        case Cyc(left=left, right=right):
            return frozenset({"Cyc"}) | kinds_used(left) | kinds_used(right)
        case Occ(event=event, retained=retained):
            return frozenset({"Occ"}) | kinds_used(event) | kinds_used(retained)
        case Sand(event=event, retained=retained):
            return frozenset({"Sand"}) | kinds_used(event) | kinds_used(retained)
    raise TypeError(f"not a term: {term!r}")


def evaluate(term: Term, events: Sequence[Ray], retained: Ray) -> Ray | None:
    """Exact native evaluation. ``None`` is undefinedness, never an error."""
    match term:
        case EventLeaf(index=index):
            return events[index]
        case RetainedLeaf():
            return retained
        case Cyc(left=left, right=right):
            a = evaluate(left, events, retained)
            if a is None:
                return None
            b = evaluate(right, events, retained)
            if b is None:
                return None
            return native.cyc(a, b)
        case Occ(event=event, retained=tail):
            e = evaluate(event, events, retained)
            if e is None:
                return None
            x = evaluate(tail, events, retained)
            if x is None:
                return None
            return native.occ(e, x)
        case Sand(event=event, retained=tail):
            e = evaluate(event, events, retained)
            if e is None:
                return None
            x = evaluate(tail, events, retained)
            if x is None:
                return None
            return native.sand(e, x)
    raise TypeError(f"not a term: {term!r}")


def evaluate_all(
    terms: Iterable[Term], events: Sequence[Ray], retained: Ray
) -> tuple[Ray | None, ...]:
    """Evaluate several programs on one input tuple."""
    return tuple(evaluate(term, events, retained) for term in terms)
