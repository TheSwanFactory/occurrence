#!/usr/bin/env python3
"""008.01 section 3 signature search: is ``E^3 R`` the smallest usable signature?

Answers the three conditions `008.01` section 3 puts on a candidate exposed-input
signature, by exhaustive exact evaluation rather than sampling:

1. at least two strict legal program classes jointly defined on a useful domain;
2. exact native endpoints that differ on a non-negligible set;
3. no deterministic rule over local branch availability reproduces the target.

Condition 3 is the one `017.24` failed, where GroupedFirst matched the learned
policy exactly. It is settled here by a **ceiling**, not by a comparison against
one hand-written rule. Any availability rule is by definition a function of the
availability pattern alone, so its program choice is constant on each pattern
class. The best accuracy achievable by *any* availability rule is therefore

```text
sum over patterns of  max over programs of  |{x in pattern : endpoint(p,x) = target(x)}|
divided by the number of inputs
```

which upper-bounds GroupedFirst, ForceSeq, and every rule of that family at once.

Target relation
---------------
A latent bracketing grammar, specified independently of any policy and of
availability, as `008.01` section 4 requires. Each basic Event is
``e_i + s e_{8+j}``; the grammar reads the **sign bits** ``s`` of the exposed
Events and selects a program from the enumerated list for the signature:

```text
bits  = 2^(n-1)*b(s1) + ... + b(sn),   b(s) = 1 if s > 0 else 0
index = bits mod n_programs
```

The sign bits are a structural property of the certified Event design. They are
not availability, not derivable from which branches happen to be legal, and they
require context-sensitive program choice. Where the selected program is
undefined on an input the target is undefined there, and that count is reported
rather than hidden.

Held-out splits over sign combinations and Event identities are a downstream
concern; this probe only establishes which signatures admit such a target.

Constructor sets
----------------
Each signature is swept twice, which is what isolates the `017.24` result:

* ``Occ, Cyc`` — the 017 family. At ``E^2 R`` the two programs are exactly
  ``P_seq`` and ``P_grp``, so this cell is directly comparable to 017.
* ``Occ, Cyc, Sand`` — the full `017.04` calculus, available for the first time
  on the exact path.

Fence: configured exact arithmetic and program-space geometry. No training, no
learned policy, no claim about physical Event supply.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments.tlm_multitoken import native
from experiments.tlm_multitoken.programs import (
    CONSTRUCTORS,
    OCC_CYC_ONLY,
    Cyc,
    EventLeaf,
    RetainedLeaf,
    Term,
    enumerate_programs,
    event_leaf_indices,
    kinds_used,
    render,
    signature,
)
from topographo.ssd import exact, fips_basic

Ray = native.Ray

RETAINED_LABEL = "e4"

#: Documented acceptance thresholds for the three `008.01` section 3 conditions.
#: A "useful common domain" needs a real share of inputs with a genuine choice.
MIN_JOINT_DOMAIN_SHARE = 0.10
#: "Non-negligible" disagreement among jointly defined programs.
MIN_DISAGREEMENT_RATE = 0.10
#: `008.01` section 8 wants a learned policy to *materially* exceed every
#: availability rule. A 3-point gap is not material, so the headroom above the
#: availability ceiling must be at least this wide for the signature to qualify.
MIN_LEARNED_HEADROOM = 0.10


def retained_ray() -> Ray:
    return exact.basis(4)


# --- Event decoding --------------------------------------------------------


def decode_event(event: Ray) -> tuple[int, int, int]:
    """Return ``(i, j, s)`` for the certified Event ``e_i + s e_{8+j}``."""
    lower = [k for k in range(1, 8) if event[k]]
    upper = [k for k in range(8, 16) if event[k]]
    if len(lower) != 1 or len(upper) != 1:
        raise ValueError("not a basic Event in the certified design")
    i = lower[0]
    j = upper[0] - 8
    s = 1 if event[upper[0]] > 0 else -1
    return i, j, s


def sign_bit(event: Ray) -> int:
    return 1 if decode_event(event)[2] > 0 else 0


# --- memoized exact evaluation --------------------------------------------


class Evaluator:
    """Exact evaluation with subterm memoization on terms consuming <= 2 Events."""

    def __init__(self, retained: Ray) -> None:
        self.retained = retained
        self._cache: dict[tuple[Term, tuple[Ray, ...]], Ray | None] = {}
        self.calls = 0

    def __call__(self, term: Term, events: tuple[Ray, ...]) -> Ray | None:
        consumed = event_leaf_indices(term)
        key: tuple[Term, tuple[Ray, ...]] | None = None
        if len(consumed) <= 2:
            key = (term, tuple(events[k] for k in consumed))
            hit = self._cache.get(key, _MISS)
            if hit is not _MISS:
                return hit  # type: ignore[return-value]
        value = self._compute(term, events)
        if key is not None:
            self._cache[key] = value
        return value

    def _compute(self, term: Term, events: tuple[Ray, ...]) -> Ray | None:
        self.calls += 1
        match term:
            case EventLeaf(index=index):
                return events[index]
            case RetainedLeaf():
                return self.retained
            case Cyc(left=left, right=right):
                a = self(left, events)
                if a is None:
                    return None
                b = self(right, events)
                if b is None:
                    return None
                return native.cyc(a, b)
            case _:
                event_term = term.event  # type: ignore[union-attr]
                tail = term.retained  # type: ignore[union-attr]
                e = self(event_term, events)
                if e is None:
                    return None
                x = self(tail, events)
                if x is None:
                    return None
                op = native.occ if type(term).__name__ == "Occ" else native.sand
                return op(e, x)


_MISS = object()


# --- baselines -------------------------------------------------------------


def grouped_first_order(programs: tuple[Term, ...]) -> tuple[int, ...]:
    """Program indices most-grouped first: more Cyc nodes, then earlier grouping.

    This is the `017.24` GroupedFirst rule generalized to a wider program space:
    prefer the most grouped legal branch, fall back toward the right comb.
    """

    return tuple(
        sorted(
            range(len(programs)),
            key=lambda k: (
                -cyc_nodes(programs[k]),
                "Sand" in kinds_used(programs[k]),
                k,
            ),
        )
    )


def cyc_nodes(term: Term) -> int:
    """Structural count of ``Cyc`` nodes, the grouping depth of a bracketing."""
    match term:
        case EventLeaf() | RetainedLeaf():
            return 0
        case Cyc(left=left, right=right):
            return 1 + cyc_nodes(left) + cyc_nodes(right)
        case _:
            return cyc_nodes(term.event) + cyc_nodes(term.retained)  # type: ignore[union-attr]


def force_seq_index(programs: tuple[Term, ...], n_events: int) -> int:
    """Index of the pure right comb ``Occ(a1, Occ(a2, ... Occ(an, r)))``."""
    target = RetainedLeaf()
    for k in reversed(range(n_events)):
        from experiments.tlm_multitoken.programs import Occ

        target = Occ(EventLeaf(k), target)
    return programs.index(target)


def cyc_free_indices(programs: tuple[Term, ...]) -> tuple[int, ...]:
    return tuple(k for k, p in enumerate(programs) if "Cyc" not in kinds_used(p))


# --- the sweep -------------------------------------------------------------


@dataclass
class SignatureResult:
    n_events: int
    signature: str
    constructors: str
    n_programs: int
    n_cyc_free_programs: int
    n_inputs: int
    defined_histogram: dict[int, int]
    per_program_defined: dict[str, int]
    inputs_with_two_or_more_defined: int
    inputs_with_endpoint_disagreement: int
    endpoint_disagreement_rate_given_two: float
    n_availability_patterns: int
    largest_pattern_share: float
    max_distinct_endpoints_in_a_pattern: int
    target_defined: int
    target_program_pool: list[str]
    availability_ceiling: float
    learned_headroom: float
    grouped_first_accuracy: float
    force_seq_accuracy: float
    random_legal_accuracy: float
    supplied_bracketing_accuracy: float
    conditions: dict[str, bool]


def sweep(
    n_events: int,
    events: tuple[Ray, ...],
    retained: Ray,
    constructors: frozenset[str],
) -> SignatureResult:
    programs = enumerate_programs(n_events, constructors)
    if not programs:
        raise ValueError("constructor set yields no programs")
    gf_order = grouped_first_order(programs)
    fs_index = force_seq_index(programs, n_events)
    evaluate = Evaluator(retained)
    pool = tuple(range(len(programs)))

    n_inputs = 0
    defined_hist: Counter[int] = Counter()
    per_program_defined: Counter[int] = Counter()
    two_plus = 0
    disagreeing = 0
    target_defined = 0

    # pattern -> program index -> hits against the target
    pattern_hits: dict[frozenset[int], Counter[int]] = defaultdict(Counter)
    pattern_total: Counter[frozenset[int]] = Counter()
    pattern_distinct_endpoints: dict[frozenset[int], int] = {}

    gf_hits = fs_hits = supplied_hits = 0
    random_legal_expectation = 0.0

    for combo in itertools.product(events, repeat=n_events):
        n_inputs += 1
        endpoints = [evaluate(program, combo) for program in programs]
        defined = [k for k, value in enumerate(endpoints) if value is not None]
        defined_hist[len(defined)] += 1
        for k in defined:
            per_program_defined[k] += 1

        # distinct endpoint classes among the defined programs
        classes: list[Ray] = []
        for k in defined:
            if not any(native.rays_equal(endpoints[k], seen) for seen in classes):
                classes.append(endpoints[k])
        if len(defined) >= 2:
            two_plus += 1
            if len(classes) >= 2:
                disagreeing += 1

        pattern = frozenset(defined)
        pattern_total[pattern] += 1
        pattern_distinct_endpoints[pattern] = max(
            pattern_distinct_endpoints.get(pattern, 0), len(classes)
        )

        # latent bracketing grammar over the sign bits
        bits = 0
        for event in combo:
            bits = (bits << 1) | sign_bit(event)
        chosen = pool[bits % len(pool)]
        target = endpoints[chosen]
        if target is None:
            continue
        target_defined += 1

        for k in defined:
            if native.rays_equal(endpoints[k], target):
                pattern_hits[pattern][k] += 1

        # GroupedFirst: first legal program in most-grouped order
        for k in gf_order:
            if endpoints[k] is not None:
                gf_hits += int(native.rays_equal(endpoints[k], target))
                break
        # ForceSeq: the right comb, whether or not it matches
        if endpoints[fs_index] is not None:
            fs_hits += int(native.rays_equal(endpoints[fs_index], target))
        # Supplied bracketing: the grammar is handed over, so this is exact
        supplied_hits += 1
        # Random legal selector: expected hit rate on this input
        if defined:
            matching = sum(
                1 for k in defined if native.rays_equal(endpoints[k], target)
            )
            random_legal_expectation += matching / len(defined)

    ceiling = 0
    for pattern, total in pattern_total.items():
        hits = pattern_hits.get(pattern)
        ceiling += max(hits.values()) if hits else 0
        del total

    denominator = target_defined or 1
    availability_ceiling = ceiling / denominator
    # A policy that knows the grammar scores 1.0 wherever the target is defined,
    # so the headroom a learned policy could win is exactly 1 - ceiling.
    headroom = 1.0 - availability_ceiling
    conditions = {
        "two_or_more_legal_classes_jointly_defined": (
            two_plus / n_inputs >= MIN_JOINT_DOMAIN_SHARE if n_inputs else False
        ),
        "endpoints_differ_on_a_non_negligible_set": (
            two_plus > 0 and disagreeing / two_plus >= MIN_DISAGREEMENT_RATE
        ),
        "no_availability_rule_reproduces_the_target": headroom >= MIN_LEARNED_HEADROOM,
    }

    return SignatureResult(
        n_events=n_events,
        signature=signature(n_events),
        constructors="+".join(sorted(constructors)),
        n_programs=len(programs),
        n_cyc_free_programs=len(cyc_free_indices(programs)),
        n_inputs=n_inputs,
        defined_histogram=dict(sorted(defined_hist.items())),
        per_program_defined={
            render(programs[k]): per_program_defined[k] for k in range(len(programs))
        },
        inputs_with_two_or_more_defined=two_plus,
        inputs_with_endpoint_disagreement=disagreeing,
        endpoint_disagreement_rate_given_two=(disagreeing / two_plus) if two_plus else 0.0,
        n_availability_patterns=len(pattern_total),
        largest_pattern_share=(max(pattern_total.values()) / n_inputs) if n_inputs else 0.0,
        max_distinct_endpoints_in_a_pattern=(
            max(pattern_distinct_endpoints.values()) if pattern_distinct_endpoints else 0
        ),
        target_defined=target_defined,
        target_program_pool=[
            render(programs[pool[bits % len(pool)]]) for bits in range(2**n_events)
        ],
        availability_ceiling=availability_ceiling,
        learned_headroom=headroom,
        grouped_first_accuracy=gf_hits / denominator,
        force_seq_accuracy=fs_hits / denominator,
        random_legal_accuracy=random_legal_expectation / denominator,
        supplied_bracketing_accuracy=supplied_hits / denominator,
        conditions=conditions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=int,
        default=0,
        help="use only the first N basic Events (0 = all 84)",
    )
    parser.add_argument(
        "--signatures",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Event counts to sweep (2 reproduces the 017 reference)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    events = fips_basic.EVENTS if args.events <= 0 else fips_basic.EVENTS[: args.events]
    retained = retained_ray()

    sets = {"Occ+Cyc": OCC_CYC_ONLY, "Occ+Cyc+Sand": CONSTRUCTORS}
    results: list[SignatureResult] = []
    for n in sorted(args.signatures):
        for constructors in sets.values():
            results.append(sweep(n, tuple(events), retained, constructors))

    def smallest_for(constructors: frozenset[str]) -> str | None:
        label = "+".join(sorted(constructors))
        return next(
            (
                r.signature
                for r in results
                if r.constructors == label and all(r.conditions.values())
            ),
            None,
        )

    report = {
        "retained_context": RETAINED_LABEL,
        "n_events_used": len(events),
        "target_relation": (
            "latent bracketing grammar over Event sign bits; "
            "index = (bits) mod n_programs over the enumerated program list"
        ),
        "availability_ceiling_definition": (
            "best accuracy achievable by ANY function of the availability pattern; "
            "upper bounds GroupedFirst, ForceSeq and every availability rule"
        ),
        "acceptance_thresholds": {
            "min_joint_domain_share": MIN_JOINT_DOMAIN_SHARE,
            "min_disagreement_rate": MIN_DISAGREEMENT_RATE,
            "min_learned_headroom": MIN_LEARNED_HEADROOM,
        },
        "smallest_qualifying_signature": {
            "Occ+Cyc": smallest_for(OCC_CYC_ONLY),
            "Occ+Cyc+Sand": smallest_for(CONSTRUCTORS),
        },
        "signatures": [dict(vars(result)) for result in results],
        "fence": (
            "configured exact arithmetic and program-space geometry; no training, "
            "no learned policy, no physical Event supply claim"
        ),
    }

    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    summary = {
        "smallest_qualifying_signature": report["smallest_qualifying_signature"],
        "cells": [
            {
                "signature": r.signature,
                "constructors": r.constructors,
                "programs": r.n_programs,
                "inputs": r.n_inputs,
                "target_defined_frac": round(r.target_defined / r.n_inputs, 4),
                "disagreement_given_two": round(r.endpoint_disagreement_rate_given_two, 4),
                "availability_patterns": r.n_availability_patterns,
                "availability_ceiling": round(r.availability_ceiling, 4),
                "learned_headroom": round(r.learned_headroom, 4),
                "grouped_first": round(r.grouped_first_accuracy, 4),
                "force_seq": round(r.force_seq_accuracy, 4),
                "random_legal": round(r.random_legal_accuracy, 4),
                "conditions": r.conditions,
            }
            for r in results
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
