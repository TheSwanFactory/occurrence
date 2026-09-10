"""Ladder-C ambient generalized-`Mul` control for `008.04`. **Not OT-native.**

`008.01` section 5.C and `008.04` section 5.C ask for a capacity-matched control
over ambient tree alternatives, explicitly fenced as non-native, scored
separately, and never used to enlarge the strict OT domain.

The construction is deliberately minimal, so the comparison isolates one thing.
Take the *same* twenty planar tree shapes and evaluate every node with ambient
sedenion multiplication:

```text
Occ (e,r)  ->  [e * r]            same formula as native
Sand(e,r)  ->  [(e * r) * e]      same formula as native
Cyc (a,b)  ->  [a * b]            AMBIENT: no FIPS admissibility precondition,
                                  no forced-third table, no Event-role guard
```

So the single difference from ``programs.evaluate`` is that the strict ``Cyc``
domain — the 336 certified ordered pairs out of 7056 — is discarded, along with
the Event-role refinement on constructor slots. Everything else, including the
exact rational arithmetic and the projective endpoint identity, is unchanged.
This is the same move `017.21` made with ``exact_p_grp_ambient``.

Two consequences worth stating before any number is read:

* On the eight ``Cyc``-free bracketings ambient and native agree exactly, so the
  `008.02` sign-bit target endpoint is **identical** under both semantics. The
  control does not change the relation being learned.
* What it does change is availability. Ambient trees are defined almost
  everywhere, so the ambient learner chooses among about twenty alternatives
  where the native learner chooses among about eight. The ambient chance level
  is therefore lower, not higher.

Fence: ambient generalized ``Mul`` is the ambient sedenion algebra. It is not OT
occurrence, it is not a certified constructor, and an ambient success does not
license a strict-domain claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from experiments.tlm_multitoken.probe_program_space import retained_ray
from experiments.tlm_multitoken.programs import (
    Cyc,
    EventLeaf,
    Occ,
    RetainedLeaf,
    Sand,
    Term,
    event_leaf_indices,
    kinds_used,
)
from experiments.tlm_multitoken.programs import evaluate as native_evaluate
from experiments.tlm_multitoken.task import Pool, Record
from topographo.ssd import exact, fips_basic, projective

__all__ = [
    "AMBIENT_FENCE",
    "AmbientEvaluator",
    "AmbientPool",
    "ambient_evaluate",
    "materialize_ambient",
]

AMBIENT_FENCE = (
    "ambient generalized Mul is the ambient sedenion algebra, not OT occurrence; "
    "the strict Cyc admissibility precondition and the Event-role guard are "
    "dropped; ambient success does not enlarge the strict OT domain"
)

Ray = exact.Value


def _mul_or_none(left: Ray, right: Ray) -> Ray | None:
    product = exact.mul(left, right)
    return None if product == exact.zero() else product


def ambient_evaluate(term: Term, events, retained: Ray) -> Ray | None:
    """Evaluate a planar tree with unguarded ambient multiplication."""
    match term:
        case EventLeaf(index=index):
            return events[index]
        case RetainedLeaf():
            return retained
        case Cyc(left=left, right=right):
            a = ambient_evaluate(left, events, retained)
            b = ambient_evaluate(right, events, retained) if a is not None else None
            return None if a is None or b is None else _mul_or_none(a, b)
        case Occ(event=event, retained=tail):
            e = ambient_evaluate(event, events, retained)
            x = ambient_evaluate(tail, events, retained) if e is not None else None
            return None if e is None or x is None else _mul_or_none(e, x)
        case Sand(event=event, retained=tail):
            e = ambient_evaluate(event, events, retained)
            x = ambient_evaluate(tail, events, retained) if e is not None else None
            if e is None or x is None:
                return None
            inner = _mul_or_none(e, x)
            return None if inner is None else _mul_or_none(inner, e)
    raise TypeError(f"not a term: {term!r}")


class AmbientEvaluator:
    """Ambient evaluation, memoized on subterms consuming at most two Events."""

    def __init__(self, retained: Ray) -> None:
        self.retained = retained
        self._cache: dict[tuple[Term, tuple[Ray, ...]], Ray | None] = {}
        self.calls = 0

    def __call__(self, term: Term, events: tuple[Ray, ...]) -> Ray | None:
        consumed = event_leaf_indices(term)
        key: tuple[Term, tuple[Ray, ...]] | None = None
        if len(consumed) <= 2:
            key = (term, tuple(events[k] for k in consumed))
            if key in self._cache:
                return self._cache[key]
        self.calls += 1
        value = ambient_evaluate(term, events, self.retained)
        if key is not None:
            self._cache[key] = value
        return value


@dataclass
class AmbientPool:
    """Ambient mirror of a native :class:`~task.Pool`, record for record.

    ``records`` are ambient-evaluated but carry the *same* ``event_indices``,
    ``sign_pattern`` and ``target_program`` as the native pool, so every split
    built on the native pool transfers unchanged. ``native_class_of_pick`` lets
    an ambient-trained policy be rescored under exact native semantics without
    re-evaluating anything.
    """

    records: tuple[Record, ...]
    n_endpoint_classes: int
    evaluator_calls: int
    defined_histogram: dict[str, int]
    agrees_with_native_on_cyc_free: bool
    fence: str = AMBIENT_FENCE

    def as_dict(self) -> dict:
        return {
            "n_records": len(self.records),
            "n_endpoint_classes": self.n_endpoint_classes,
            "defined_histogram": self.defined_histogram,
            "mean_legal_programs": (
                sum(len(r.legal) for r in self.records) / len(self.records)
                if self.records
                else 0.0
            ),
            "target_defined_share": (
                sum(1 for r in self.records if r.target_defined) / len(self.records)
                if self.records
                else 0.0
            ),
            "agrees_with_native_on_cyc_free_programs": (
                self.agrees_with_native_on_cyc_free
            ),
            "fence": self.fence,
        }


def materialize_ambient(pool: Pool, *, events=None) -> AmbientPool:
    """Re-evaluate the native pool's inputs under ambient multiplication."""
    all_events = tuple(fips_basic.EVENTS) if events is None else tuple(events)
    retained = retained_ray()
    programs = pool.programs
    cyc_free = tuple(
        k for k, program in enumerate(programs) if "Cyc" not in kinds_used(program)
    )

    evaluate = AmbientEvaluator(retained)
    class_of: dict[tuple, int] = {}
    records: list[Record] = []
    hist: Counter[int] = Counter()
    agrees = True

    for native_record in pool.records:
        combo = tuple(all_events[k] for k in native_record.event_indices)
        endpoints = [evaluate(program, combo) for program in programs]
        classes: list[int] = []
        for value in endpoints:
            if value is None:
                classes.append(-1)
                continue
            key = tuple(str(c) for c in projective.canonicalize(value))
            classes.append(class_of.setdefault(key, len(class_of)))
        defined = tuple(value is not None for value in endpoints)
        hist[sum(defined)] += 1
        pattern = 0
        for k, ok in enumerate(defined):
            if ok:
                pattern |= 1 << k
        records.append(
            Record(
                event_indices=native_record.event_indices,
                sign_pattern=native_record.sign_pattern,
                target_program=native_record.target_program,
                target_class=classes[native_record.target_program],
                defined=defined,
                endpoint_class=tuple(classes),
                availability_pattern=pattern,
            )
        )

    # Spot-check the claim that ambient and native coincide off Cyc: the two
    # semantics must give the same ray for every Cyc-free program.
    for native_record in pool.records[: min(64, len(pool.records))]:
        combo = tuple(all_events[k] for k in native_record.event_indices)
        for k in cyc_free:
            ambient_value = ambient_evaluate(programs[k], combo, retained)
            native_value = native_evaluate(programs[k], combo, retained)
            if (ambient_value is None) != (native_value is None):
                agrees = False
                break
            if ambient_value is not None and not projective.equivalent(
                ambient_value, native_value
            ):
                agrees = False
                break

    return AmbientPool(
        records=tuple(records),
        n_endpoint_classes=len(class_of),
        evaluator_calls=evaluate.calls,
        defined_histogram={str(k): hist[k] for k in sorted(hist)},
        agrees_with_native_on_cyc_free=agrees,
    )
