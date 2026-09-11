"""The fixed, exact, nonlearned Rung-3 resolver: SFP address -> public Event.

`009.05` section 6 permits this resolver to use the already-certified finite
correspondence, because Rung 3 asks whether a successfully constituted
consequence survives public denotation. It explicitly may not infer, repair or
alter an incorrect structured address.

The resolver holds no parameters of any kind and constructs no tensor. Resolution
is a single dictionary lookup on the canonical address key produced by
``sfp.address_of``. There is no nearest-neighbour step, no fallback, no
tie-breaking and no default.

It is not, however, an import-time-torch-free module: it reaches ``harness``
transitively through ``ladder_task``, which reuses the frozen ``009.02`` split
functions rather than reimplementing them, and ``harness`` imports torch. That
trade is deliberate -- a second copy of the split logic could drift from the one
that produced the ``009.02`` numbers -- and it is recorded here rather than
claimed away, because "nonlearned" is a statement about this module's arithmetic
and not about its dependency graph.

One structural fact governs how Rung 3 must be read, and it is proved here
rather than assumed. The SFP alphabet is **exactly saturated** by the catalogue::

    2 sign values  x  7 Fano points  x  6 K4 edges  =  84  =  the 84 Events

So every syntactically well-formed ``(S, FFF, PP, pp)`` names a real Event, and
the heads cannot emit anything else: they are 2, 7, 4 and 3 slots wide, and
``pp = 00`` has no slot at all. Resolution is therefore **total** -- the resolver
can never refuse -- and the resolved Event-identity accuracy equals the exact
structured-address accuracy identically, not approximately. Rung 3 cannot show a
codec drop, and :func:`verify_totality` and :func:`verify_no_repair` are what
license that statement.

That is not a weakness of the resolver. It is the reason a wrong address is
punished at full severity: a one-field error resolves to a *different certified
Event*, never to a refusal and never back to the right answer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ladder_task import (
    ARM_OUTPUT_CONVENTION,
    PP_CLASSES,
    SCIENCE_ARMS,
    code_tables,
    output_table,
    rung2_target,
)
from sfp import SfpAddress, address_of, bits2
from task import BOTTOM_INDEX, Dataset, build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_resolver.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

#: Returned when an address is outside the arm's certified alphabet. Distinct from
#: ``task.BOTTOM_INDEX``: BOTTOM is a *decision*, UNRESOLVED is a *failure to
#: denote*. Conflating them would let a bad address masquerade as non-admission.
UNRESOLVED = -2

FENCES = (
    "The resolver is fixed, exact and nonlearned. It holds no parameters.",
    "It may not infer, repair or alter an incorrect structured address.",
    (
        "UNRESOLVED != BOTTOM: a failure to denote is not a decision not to admit. "
        "Neither is an algebraic zero."
    ),
    (
        "It is applied only AFTER a structured address exists, never as part of "
        "producing one."
    ),
    (
        "It resolves within the arm's own certified alphabet, so no arm is resolved "
        "through another arm's chart."
    ),
)

PINS = {
    "events": 84,
    "sign_values": 2,
    "fano_points": 7,
    "edges_per_habitat": 6,
    "syntactic_field_tuples": 2 * 7 * 4 * 3,
    "distinct_addresses": 84,
    "admit": 336,
}

SATURATION_STATEMENT = (
    "2 x 7 x 6 = 84 = |Events|. The alphabet is exactly saturated, so resolution is "
    "total and Rung 3 cannot exhibit a codec drop. The 168 syntactic (S, FFF, PP, "
    "pp) tuples collapse two-to-one onto the 84 addresses because an Event has two "
    "block-incidence presentations."
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def pin(expected: object, observed: object, source: str) -> dict:
    return {
        "expected": expected,
        "observed": observed,
        "agrees": expected == observed,
        "source": source,
    }


def syntactic_tuples() -> tuple[tuple[int, int, int, int], ...]:
    """Every field combination the Rung-2 heads can emit, in canonical order."""

    return tuple(
        (s, fff, q, d)
        for s in range(2)
        for fff in range(1, 8)
        for q in range(4)
        for d in PP_CLASSES
    )


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolver:
    """A fixed exact map from one arm's SFP addresses to public Event identity.

    Built by inverting the arm's own frozen code table. Nonlearned: the table is
    constructed once from ``ladder_task.code_tables``, is never updated, and holds
    no float anywhere.
    """

    arm: str
    alphabet: str
    index_of_address: dict[tuple, int]

    def __post_init__(self) -> None:
        if len(self.index_of_address) != PINS["events"]:
            raise AssertionError(
                f"{self.arm} resolver covers {len(self.index_of_address)} addresses, "
                "not 84"
            )
        if len(set(self.index_of_address.values())) != PINS["events"]:
            raise AssertionError(f"{self.arm} resolver is not injective")

    def resolve_address(self, address: SfpAddress) -> int:
        """Canonical address -> Event index, or :data:`UNRESOLVED`.

        A plain lookup. The ``.get`` default is :data:`UNRESOLVED`, never a nearest
        address and never the caller's expected answer -- there is nowhere in this
        method for a repair to live.
        """

        return self.index_of_address.get(address.as_key(), UNRESOLVED)

    def resolve(self, s: int, fff: int, q: int, d: int) -> int:
        """Four predicted fields -> Event index, or :data:`UNRESOLVED`.

        Out-of-range fields resolve to :data:`UNRESOLVED` rather than raising, so a
        malformed prediction is scored as a failure to denote instead of crashing
        the evaluation. The Rung-2 heads cannot actually emit one.
        """

        if s not in (0, 1) or not 1 <= fff <= 7 or not 0 <= q <= 3:
            return UNRESOLVED
        if d not in PP_CLASSES:
            return UNRESOLVED
        return self.resolve_address(address_of(s, fff, q, d))

    def declaration(self) -> dict:
        return {
            "arm": self.arm,
            "alphabet": self.alphabet,
            "kind": "fixed exact nonlearned dictionary lookup",
            "n_addresses": len(self.index_of_address),
            "parameters": 0,
            "learned": False,
            "may_repair": False,
            "unresolved_sentinel": UNRESOLVED,
            "table_sha256": digest(
                sorted(
                    [f"{s}|{fff:03b}|{bits2(p[0][0])}{bits2(p[0][1])}", index]
                    for (s, fff, p), index in self.index_of_address.items()
                )
            ),
        }


def build_resolvers(dataset: Dataset) -> dict[str, Resolver]:
    """One resolver per science arm, each over that arm's own output alphabet."""

    tables = code_tables(dataset)
    resolvers = {}
    for name in SCIENCE_ARMS:
        table = output_table(name, tables)
        resolvers[name] = Resolver(
            arm=name,
            alphabet=ARM_OUTPUT_CONVENTION[name],
            index_of_address={
                code.as_key(): index for index, code in enumerate(table)
            },
        )
    return resolvers


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_exactness(dataset: Dataset, resolvers: dict[str, Resolver]) -> dict:
    """Every Event round-trips, from BOTH of its block-incidence presentations."""

    tables = code_tables(dataset)
    rows = {}
    for name, resolver in sorted(resolvers.items()):
        table = output_table(name, tables)
        from_address = 0
        from_presentation = 0
        for index, code in enumerate(table):
            if resolver.resolve_address(code) == index:
                from_address += 1
            hits = sum(
                1
                for presentation in code.incidences()
                if resolver.resolve(
                    presentation.s, presentation.fff, presentation.q, presentation.d
                )
                == index
            )
            if hits == 2:
                from_presentation += 1
        rows[name] = {
            "events": len(table),
            "resolved_from_canonical_address": from_address,
            "resolved_from_both_presentations": from_presentation,
            "exact_on_every_event": from_address == len(table) == from_presentation,
        }
    return {
        "per_arm": rows,
        "all_arms_exact": all(row["exact_on_every_event"] for row in rows.values()),
        "statement": (
            "each arm's resolver inverts that arm's own code exactly, and both "
            "presentations of an Event resolve to the same Event, which is the "
            "documented canonicalization working end to end"
        ),
    }


def verify_totality(resolvers: dict[str, Resolver]) -> dict:
    """Resolution is total: every syntactic field tuple names a real Event."""

    tuples = syntactic_tuples()
    rows = {}
    for name, resolver in sorted(resolvers.items()):
        resolved = [resolver.resolve(*fields) for fields in tuples]
        collapse = Counter(resolved)
        rows[name] = {
            "syntactic_tuples": len(tuples),
            "resolved": sum(1 for value in resolved if value != UNRESOLVED),
            "unresolved": sum(1 for value in resolved if value == UNRESOLVED),
            "distinct_events_reached": len(collapse),
            "tuples_per_event": sorted(set(collapse.values())),
            "total": all(value != UNRESOLVED for value in resolved),
        }
    return {
        "per_arm": rows,
        "resolution_is_total_for_every_arm": all(row["total"] for row in rows.values()),
        "all_84_events_reachable": all(
            row["distinct_events_reached"] == PINS["events"] for row in rows.values()
        ),
        "two_presentations_per_event": all(
            row["tuples_per_event"] == [2] for row in rows.values()
        ),
        "saturation": SATURATION_STATEMENT,
        "consequence_for_rung_3": (
            "because resolution is total and injective on addresses, resolved "
            "Event-identity accuracy is IDENTICALLY equal to exact structured-address "
            "accuracy. Rung 3 therefore cannot show a codec, resolver or conformance "
            "drop, and any Rung-2 to Rung-3 difference would be an implementation "
            "bug rather than a scientific finding. verify_resolution_identity "
            "measures the equality rather than asserting it."
        ),
        "why_a_refusal_path_still_exists": (
            "Resolver.resolve returns UNRESOLVED for an out-of-range field even "
            "though the heads cannot produce one. The branch is kept so that a "
            "future head with a wider field, or a hand-written address, is scored as "
            "a failure to denote instead of silently crashing or being repaired."
        ),
    }


def verify_no_repair(dataset: Dataset, resolvers: dict[str, Resolver]) -> dict:
    """A corrupted address resolves to the WRONG Event, never back to the right one.

    The real no-repair test. For every admitted pair, each field of the certified
    target address is perturbed to every other value it could take, and the
    resolver's answer is compared to the certified Event. If the resolver had any
    nearest-address or snap-back behaviour, some perturbations would come back
    correct.

    The one documented exception is a ``PP`` perturbation onto the target's OTHER
    block-incidence presentation. That is not a repair: both values name the same
    Event by the codec's own symmetry, and ``ladder_task`` already counts either as
    correct.
    """

    tables = code_tables(dataset)
    rows = {}
    for name, resolver in sorted(resolvers.items()):
        table = output_table(name, tables)
        perturbations = 0
        recovered_wrongly = 0
        other_presentation = 0
        landed_elsewhere = 0
        unresolved = 0
        per_field = {field: Counter() for field, _ in
                     (("S", 2), ("FFF", 7), ("PP", 4), ("pp", 3))}
        for position in dataset.admit_positions:
            record = dataset.records[position]
            target = rung2_target(record, table)
            base = (target.s, target.fff, target.pp_set[0], target.d)
            options = {
                "S": [v for v in range(2) if v != target.s],
                "FFF": [v for v in range(1, 8) if v != target.fff],
                "PP": [v for v in range(4) if v != target.pp_set[0]],
                "pp": [v for v in PP_CLASSES if v != target.d],
            }
            for field_index, field in enumerate(("S", "FFF", "PP", "pp")):
                for value in options[field]:
                    fields = list(base)
                    fields[field_index] = value
                    perturbations += 1
                    got = resolver.resolve(*fields)
                    if got == UNRESOLVED:
                        unresolved += 1
                        per_field[field]["unresolved"] += 1
                    elif got == record.target_index:
                        if field == "PP" and value in target.pp_set:
                            other_presentation += 1
                            per_field[field]["other_presentation"] += 1
                        else:
                            recovered_wrongly += 1
                            per_field[field]["repaired"] += 1
                    else:
                        landed_elsewhere += 1
                        per_field[field]["different_event"] += 1
        rows[name] = {
            "perturbations": perturbations,
            "resolved_to_a_different_event": landed_elsewhere,
            "resolved_to_the_other_presentation_of_the_same_event": other_presentation,
            "unresolved": unresolved,
            "silently_repaired": recovered_wrongly,
            "never_repairs": recovered_wrongly == 0,
            "per_field": {
                field: dict(sorted(counter.items()))
                for field, counter in sorted(per_field.items())
            },
        }
    return {
        "per_arm": rows,
        "no_arm_resolver_repairs_a_corrupted_address": all(
            row["never_repairs"] for row in rows.values()
        ),
        "documented_exception": (
            "a PP perturbation onto the target's other block-incidence presentation "
            "returns the same Event. That is the codec's own symmetry, declared in "
            "ladder_task.PP_SET_TARGET_CONVENTION and already credited as correct "
            "there, not a resolver repair."
        ),
        "statement": (
            "every other single-field corruption resolves to a different certified "
            "Event. The resolver is maximally unforgiving: it neither refuses nor "
            "rescues."
        ),
    }


def verify_resolution_identity(
    dataset: Dataset, resolvers: dict[str, Resolver]
) -> dict:
    """Address-correct and Event-correct are the same predicate, measured.

    Simulated over the certified targets and over every single-field corruption of
    them, so the equality is demonstrated on wrong answers too rather than only on
    right ones.
    """

    tables = code_tables(dataset)
    rows = {}
    for name, resolver in sorted(resolvers.items()):
        table = output_table(name, tables)
        agree = 0
        total = 0
        disagreements = []
        for position in dataset.admit_positions:
            record = dataset.records[position]
            target = rung2_target(record, table)
            trials = [(target.s, target.fff, q, target.d) for q in target.pp_set]
            trials.extend(
                [
                    (v, target.fff, target.pp_set[0], target.d)
                    for v in range(2)
                    if v != target.s
                ]
            )
            trials.extend(
                [
                    (target.s, v, target.pp_set[0], target.d)
                    for v in range(1, 8)
                    if v != target.fff
                ]
            )
            trials.extend(
                [
                    (target.s, target.fff, target.pp_set[0], v)
                    for v in PP_CLASSES
                    if v != target.d
                ]
            )
            for fields in trials:
                total += 1
                address_correct = (
                    address_of(fields[0], fields[1], fields[2], fields[3]).as_key()
                    == target.address.as_key()
                )
                event_correct = resolver.resolve(*fields) == record.target_index
                if address_correct == event_correct:
                    agree += 1
                elif len(disagreements) < 4:
                    disagreements.append(
                        {
                            "a": record.a_index,
                            "b": record.b_index,
                            "fields": list(fields),
                            "address_correct": address_correct,
                            "event_correct": event_correct,
                        }
                    )
        rows[name] = {
            "trials": total,
            "agreements": agree,
            "disagreements": len(disagreements),
            "examples": disagreements,
            "address_correct_iff_event_correct": agree == total,
        }
    return {
        "per_arm": rows,
        "identical_for_every_arm": all(
            row["address_correct_iff_event_correct"] for row in rows.values()
        ),
        "statement": (
            "exact structured-address accuracy and resolved Event-identity accuracy "
            "are the same number by construction, and the equality is measured over "
            "correct and corrupted addresses alike"
        ),
    }


def verify_bottom_is_not_unresolved(resolvers: dict[str, Resolver]) -> dict:
    """The two failure symbols stay distinct, and neither is an Event."""

    return {
        "bottom_index": BOTTOM_INDEX,
        "unresolved_sentinel": UNRESOLVED,
        "distinct": BOTTOM_INDEX != UNRESOLVED,
        "neither_is_a_valid_event_index": BOTTOM_INDEX < 0 and UNRESOLVED < 0,
        "no_resolver_ever_returns_bottom": all(
            BOTTOM_INDEX not in resolver.index_of_address.values()
            for resolver in resolvers.values()
        ),
        "statement": (
            "BOTTOM is a decision the gate makes; UNRESOLVED is an address that "
            "denotes nothing. Keeping them apart is what stops a malformed address "
            "from being scored as a correct non-admission."
        ),
        "fence": "0 != bottom, and bottom != unresolved",
    }


def verify_nonlearned(resolvers: dict[str, Resolver]) -> dict:
    """The resolver has no parameters and no float anywhere in its table."""

    rows = {}
    for name, resolver in sorted(resolvers.items()):
        values = list(resolver.index_of_address.values())
        keys = list(resolver.index_of_address.keys())
        rows[name] = {
            "declaration": resolver.declaration(),
            "all_values_are_ints": all(isinstance(v, int) for v in values),
            "no_float_in_keys": not any(
                isinstance(part, float)
                for key in keys
                for part in (key[0], key[1])
            ),
        }
    return {
        "per_arm": rows,
        "constructs_no_tensor": True,
        "holds_no_parameters": True,
        "imports_torch_transitively": True,
        "why_torch_is_still_in_the_graph": (
            "ladder_task reuses harness.training_positions and "
            "harness.evaluation_positions so the Rung-2 splits are provably the same "
            "objects that produced the 009.02 numbers, and harness imports torch. A "
            "local reimplementation would be import-time torch-free but could drift "
            "from the frozen splits, which is the worse failure. Recorded rather than "
            "claimed away."
        ),
        "all_tables_are_integer_only": all(
            row["all_values_are_ints"] and row["no_float_in_keys"]
            for row in rows.values()
        ),
        "statement": (
            "a frozen dataclass holding one dict of integers. There is no tensor, no "
            "gradient and no state to update, so 'nonlearned' is a property of the "
            "type rather than a promise about how it is used."
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    resolvers = build_resolvers(dataset)

    exactness = verify_exactness(dataset, resolvers)
    totality = verify_totality(resolvers)
    no_repair = verify_no_repair(dataset, resolvers)
    identity = verify_resolution_identity(dataset, resolvers)
    symbols = verify_bottom_is_not_unresolved(resolvers)
    nonlearned = verify_nonlearned(resolvers)

    pins = {
        "syntactic_field_tuples": pin(
            PINS["syntactic_field_tuples"], len(syntactic_tuples()), "field widths"
        ),
        "distinct_addresses": pin(
            PINS["distinct_addresses"],
            totality["per_arm"]["B_sfp"]["distinct_events_reached"],
            "measured on the exact code",
        ),
        "saturation": pin(
            PINS["events"],
            PINS["sign_values"] * PINS["fano_points"] * PINS["edges_per_habitat"],
            "2 x 7 x 6",
        ),
    }

    laws = {
        "all_arms_exact": exactness["all_arms_exact"],
        "resolution_is_total_for_every_arm": totality[
            "resolution_is_total_for_every_arm"
        ],
        "all_84_events_reachable": totality["all_84_events_reachable"],
        "two_presentations_per_event": totality["two_presentations_per_event"],
        "no_arm_resolver_repairs_a_corrupted_address": no_repair[
            "no_arm_resolver_repairs_a_corrupted_address"
        ],
        "address_correct_iff_event_correct": identity["identical_for_every_arm"],
        "bottom_and_unresolved_stay_distinct": symbols["distinct"]
        and symbols["no_resolver_ever_returns_bottom"],
        "resolver_is_nonlearned": nonlearned["all_tables_are_integer_only"],
        "all_pins_agree": all(row["agrees"] for row in pins.values()),
    }

    broken = sorted(name for name, ok in laws.items() if not ok)
    return {
        "module": "ladder_resolver",
        "purpose": (
            "the fixed exact nonlearned Rung-3 resolver from a structured SFP "
            "address to certified public Event identity"
        ),
        "executes": "009.05 section 6",
        "fences": list(FENCES),
        "pins": pins,
        "exactness": exactness,
        "totality": totality,
        "no_repair": no_repair,
        "resolution_identity": identity,
        "symbols": symbols,
        "nonlearned": nonlearned,
        "provenance": {
            "base_commit": BASE_COMMIT,
            "reuses": [
                "ladder_task.code_tables / output_table / rung2_target",
                "sfp.address_of",
            ],
            "constructs_no_tensor": True,
            "imports_torch_transitively_via_harness": True,
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - each arm's resolver inverts its own code exactly, resolution "
                "is total because the alphabet is saturated (2 x 7 x 6 = 84), no "
                "single-field corruption is ever repaired, and resolved "
                "Event-identity accuracy is measurably identical to exact "
                "structured-address accuracy"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }


def _report(result: dict) -> None:
    print(
        "saturation: 2 x 7 x 6 = "
        f"{PINS['sign_values'] * PINS['fano_points'] * PINS['edges_per_habitat']} "
        f"= {PINS['events']} Events -> resolution is total",
        flush=True,
    )
    row = result["no_repair"]["per_arm"]["B_sfp"]
    print(
        f"no-repair probe: {row['perturbations']} single-field corruptions, "
        f"{row['silently_repaired']} repaired, "
        f"{row['resolved_to_a_different_event']} landed on a different Event",
        flush=True,
    )
    print(
        "address-correct iff event-correct: "
        f"{result['resolution_identity']['identical_for_every_arm']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches ladder_resolver.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
