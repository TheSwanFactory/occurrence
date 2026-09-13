"""011.01 sections 3.2, 3.3 and 7: the exact ceilings and the cheap baselines.

Three things live here, and they are kept apart on purpose.

The **nonlearned exact ceiling** is the Gate-0 transparent reconstruction run as a
predictor. ``011.01`` section 3.2 requires it to recover the same-``FFF`` mate,
the Fano completion and the whole seven-line plane at exactly ``1.0000``, and
section 7.1 requires it to be a ceiling rather than a competing arm. It is
imported from ``fano_task``; the learned path in ``fano_heads`` does not import
it, and ``fano_heads.answer_rule_audit`` checks that mechanically.

The **exact control ceilings** are rational numbers, not measured chance.
``011.01`` section 3.3 forbids substituting empirical chance for the calculation,
so ``fano_task.localized_ceiling_census`` constructs, for every query and every
ordered pair of candidate answers, an automorphism of the control graph fixing
the query and carrying one candidate to the other. Every scored candidate lies in
one orbit, so any equivariant scorer must tie, and the best achievable rates are
``1/13`` for the mate and ``1/66`` for the two-element completion set.

The **cheap baselines** are the deterministic selectors ``011.01`` section 7 asks
for. They matter most where they succeed: the mate query is *cheaply exact in the
main arm*, because a mate shares six axis nodes with the query and a non-mate
shares five, and a two-line rule finds that. This module says so plainly. Query A
is therefore not the load-bearing evidence for global discovery -- it is the
easier half, and the honest reading of a positive result rests on Query B, where
every cheap rule reported here fails outright.

Torch-free. No tensors, no model, no training.
"""

from __future__ import annotations

import argparse
import math
import platform
from collections import Counter
from pathlib import Path

from fano_task import (
    EXPECTED_CEILINGS,
    Namespace,
    Observation,
    Query,
    build_episodes,
    build_localized_observation,
    build_namespaces,
    build_observation,
    localized_ceiling_census,
    pin,
    plane_audit,
    rd,
    reconstruct,
    render,
)
from task import build_dataset

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_fano_artifacts"
OUTPUT = ARTIFACTS / "fano_baselines.json"

BASE_COMMIT = "d4efc739ffb9ba58a4e79b1b98b3bd0e5aa8cbb2"

#: How close a measured rate must sit to an exact rational ceiling to count as
#: "at" it, and why the tolerance is not zero.
#:
#: A ceiling like ``1/13`` is the *expected* rate under uniform renaming. The
#: frozen namespace draw is finite -- 48 namespaces -- so a selector that is
#: exactly at its ceiling still lands ``O(1/48)`` away from the rational value.
#: Worse, a degenerate tie-break's hits arrive in per-namespace blocks: on the
#: completion query a single namespace contributes either 0 or 12 hits, so the
#: effective sample size is the namespace count, not the query count. The
#: tolerance is declared here, before any measurement, and the decision rule is
#: the absolute comparison; the exact binomial tail is reported alongside as a
#: diagnostic with that dependence stated rather than hidden.
CEILING_TOLERANCE = 0.01

#: How high a cheap selector may score on the load-bearing query before it stops
#: being cheap. Two orders below the section 8 class-A threshold of ``0.95``.
CHEAP_RULE_BOUND = 0.10

#: The deterministic selectors ``011.01`` section 7 requires, and what each is.
SELECTORS = {
    "uniform": (
        "the analytic rate of choosing the answer set uniformly at random from "
        "the scored candidates; reported as an exact rational, not sampled"
    ),
    "lowest_index": (
        "the opaque-index tie-break: take the lowest-numbered scored candidates. "
        "A learner that has learned nothing and ties everywhere decides this way, "
        "so it is the floor a null model actually reaches on frozen namespaces"
    ),
    "shared_axis_count": (
        "score each candidate by how many axis nodes it shares with the query "
        "habitats. The natural cheap rule on this observation, and the one worth "
        "reporting: it is exact for the mate and it is wrong for completion, "
        "because it selects the two queries' own mates"
    ),
    "event_two_hop": (
        "score each candidate by the number of length-four H--E--axis--E--H walks "
        "reaching it from the query habitats; a pure connectivity count with no "
        "notion of the recovered L/R bijection"
    ),
}


# ---------------------------------------------------------------------------
# comparing a measured rate against an exact rational ceiling
# ---------------------------------------------------------------------------

def _log_binomial_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binomial_tail(n: int, k: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, summed in log space.

    Log space rather than ``comb`` times powers, because ``comb(4032, 3000)``
    times ``p**3000`` overflows one way and underflows the other.
    """

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.exp(_log_binomial_pmf(n, i, p)) for i in range(k, n + 1))


def ceiling_compatibility(
    observed_rate: float | None,
    queries: int,
    ceiling: float,
    *,
    blocks: int | None = None,
) -> dict:
    """Is a measured rate at, below, or materially above an exact ceiling?

    The decision is the absolute comparison against ``CEILING_TOLERANCE``. The
    exact binomial tail is reported too, at the query count and -- when the
    selector's hits arrive in per-namespace blocks -- at the block count, which is
    the honest effective sample size for a degenerate tie-break.
    """

    if observed_rate is None:
        return {"observed": None, "ceiling": rd(ceiling), "materially_exceeds": None}
    excess = observed_rate - ceiling
    hits = round(observed_rate * queries)
    out = {
        "observed": rd(observed_rate),
        "ceiling": rd(ceiling),
        "excess": rd(excess),
        "tolerance": CEILING_TOLERANCE,
        "hits": hits,
        "queries": queries,
        "binomial_tail_at_query_count": rd(binomial_tail(queries, hits, ceiling)),
        "materially_exceeds": excess > CEILING_TOLERANCE,
        "at_or_below_ceiling": excess <= CEILING_TOLERANCE,
    }
    if blocks:
        out["blocks"] = blocks
        out["binomial_tail_at_block_count"] = rd(
            binomial_tail(blocks, round(observed_rate * blocks), ceiling)
        )
        out["block_note"] = (
            "a degenerate tie-break's hits arrive in per-namespace blocks, so the "
            "block count is the effective sample size and the query-count tail "
            "overstates significance"
        )
    return out


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

def _axis_sets(observation: Observation) -> tuple[tuple[frozenset[int], ...], ...]:
    """Per habitat, its left-axis and right-axis support in slot coordinates."""

    return (observation.left_support(), observation.right_support())


def _slot_of_habitat_node(namespace: Namespace) -> dict[int, int]:
    return {node: slot for slot, node in enumerate(namespace.habitat)}


def _decide(
    scores: dict[int, float], scored: tuple[int, ...], size: int
) -> tuple[int, ...]:
    """Top-``size`` scored candidates, ties broken by low opaque index."""

    order = sorted(scored, key=lambda node: (-scores.get(node, 0.0), node))
    return tuple(sorted(order[:size]))


def shared_axis_scores(
    observation: Observation, namespace: Namespace, query: Query
) -> dict[int, float]:
    left, right = _axis_sets(observation)
    slot_of = _slot_of_habitat_node(namespace)
    query_slots = [slot_of[node] for node in query.query_nodes]
    out: dict[int, float] = {}
    for node in query.scored_nodes:
        slot = slot_of[node]
        out[node] = float(
            sum(
                len(left[slot] & left[other]) + len(right[slot] & right[other])
                for other in query_slots
            )
        )
    return out


def event_two_hop_scores(
    observation: Observation, namespace: Namespace, query: Query
) -> dict[int, float]:
    """Length-four ``H--E--axis--E--H`` walk counts from the query habitats."""

    slot_of = _slot_of_habitat_node(namespace)
    query_slots = [slot_of[node] for node in query.query_nodes]
    left_of = observation.left_of_event()
    right_of = observation.right_of_event()
    habitat_of = observation.habitat_of_event()

    by_left: dict[int, list[int]] = {}
    by_right: dict[int, list[int]] = {}
    for event in range(observation.n_events):
        by_left.setdefault(left_of[event], []).append(event)
        by_right.setdefault(right_of[event], []).append(event)

    tally: Counter = Counter()
    for slot in query_slots:
        for event in observation.events_of_habitat()[slot]:
            for peer in by_left[left_of[event]]:
                tally[habitat_of[peer]] += 1
            for peer in by_right[right_of[event]]:
                tally[habitat_of[peer]] += 1
    out: dict[int, float] = {}
    for node in query.scored_nodes:
        out[node] = float(tally[slot_of[node]])
    return out


# ---------------------------------------------------------------------------
# the nonlearned exact ceiling
# ---------------------------------------------------------------------------

def nonlearned_ceiling(observation: Observation) -> dict:
    """The Gate-0 reconstruction run as a predictor. Section 3.2 wants ``1.0000``.

    Scored over every frozen namespace, so a ceiling that held only in the
    canonical node order would be caught. The reconstruction never reads a hidden
    field; agreement with the certified labels is the thing being measured.
    """

    reconstruction = reconstruct(observation)
    plane = plane_audit(reconstruction)
    namespaces = build_namespaces(observation)
    queries = build_episodes(observation, namespaces)
    by_namespace = {namespace.label: namespace for namespace in namespaces}

    hits = Counter()
    totals = Counter()
    for query in queries:
        totals[query.kind] += 1
        if not reconstruction.valid:
            # A failed reconstruction predicts nothing. This is the control's
            # case: 14 singleton classes, no bijection, no lines. Scoring it as
            # zero rather than raising is what makes the section 3.3
            # "non-discriminating" check reportable.
            continue
        namespace = by_namespace[query.namespace]
        slot_of = _slot_of_habitat_node(namespace)
        slots = tuple(slot_of[node] for node in query.query_nodes)
        if query.kind == "mate":
            predicted = (namespace.habitat[reconstruction.mate(slots[0])],)
        else:
            predicted = tuple(
                sorted(
                    namespace.habitat[slot]
                    for slot in reconstruction.completion(*slots)
                )
            )
        hits[query.kind] += tuple(sorted(predicted)) == tuple(sorted(query.target_nodes))

    return {
        "reconstruction_valid": reconstruction.valid,
        "namespaces": len(namespaces),
        "mate": pin(
            totals["mate"],
            hits["mate"],
            "reconstruct().mate over every frozen namespace",
        ),
        "completion": pin(
            totals["completion"],
            hits["completion"],
            "reconstruct().completion over every frozen namespace",
        ),
        "mate_accuracy": rd(hits["mate"] / totals["mate"]) if totals["mate"] else None,
        "completion_exact_set_accuracy": (
            rd(hits["completion"] / totals["completion"]) if totals["completion"] else None
        ),
        "whole_plane": {
            "lines_recovered": len(reconstruction.lines),
            "automorphism_order": plane["automorphism_order"]["observed"],
            "equivalent_to_certified_plane": plane["equivalent_to_certified_plane"],
            "accuracy": 1.0
            if plane["equivalent_to_certified_plane"] and reconstruction.valid
            else 0.0,
        },
        "role": (
            "011.01 section 7.1 NONLEARNED CEILING. It is not a competing learned "
            "baseline and it is not imported by the learned path"
        ),
    }


# ---------------------------------------------------------------------------
# cheap selectors, measured
# ---------------------------------------------------------------------------

def cheap_selectors(observation: Observation) -> dict:
    """Every section 7 deterministic selector, scored on the frozen namespaces."""

    namespaces = build_namespaces(observation)
    queries = build_episodes(observation, namespaces)
    by_namespace = {namespace.label: namespace for namespace in namespaces}

    rows: dict[str, Counter] = {name: Counter() for name in SELECTORS}
    totals: Counter = Counter()
    mate_pair_hits = 0
    reconstruction = reconstruct(observation)
    for query in queries:
        namespace = by_namespace[query.namespace]
        size = len(query.target_nodes)
        wanted = tuple(sorted(query.target_nodes))
        totals[query.kind] += 1

        flat = {node: 0.0 for node in query.scored_nodes}
        rows["lowest_index"][query.kind] += _decide(flat, query.scored_nodes, size) == wanted

        shared = shared_axis_scores(observation, namespace, query)
        rows["shared_axis_count"][query.kind] += (
            _decide(shared, query.scored_nodes, size) == wanted
        )

        walks = event_two_hop_scores(observation, namespace, query)
        rows["event_two_hop"][query.kind] += _decide(walks, query.scored_nodes, size) == wanted

        if query.kind == "completion" and reconstruction.valid:
            # What shared_axis_count actually selects, named explicitly so the
            # failure has a mechanism rather than just a number.
            top = _decide(shared, query.scored_nodes, size)
            slot_of = _slot_of_habitat_node(namespace)
            mates = tuple(
                sorted(
                    namespace.habitat[reconstruction.mate(slot_of[node])]
                    for node in query.query_nodes
                )
            )
            mate_pair_hits += top == mates

    analytic = {
        "mate": rd(1.0 / (observation.n_habitats - 1)),
        "completion": rd(1.0 / math.comb(observation.n_habitats - 2, 2)),
    }
    out: dict[str, dict] = {
        "uniform": {
            "description": SELECTORS["uniform"],
            "mate": analytic["mate"],
            "completion": analytic["completion"],
            "derivation": (
                "one of 13 scored habitats for the mate; one of the "
                f"{math.comb(observation.n_habitats - 2, 2)} two-element subsets "
                "of the 12 habitats outside the query for completion"
            ),
        }
    }
    for name in ("lowest_index", "shared_axis_count", "event_two_hop"):
        out[name] = {
            "description": SELECTORS[name],
            "mate": rd(rows[name]["mate"] / totals["mate"]) if totals["mate"] else None,
            "completion": (
                rd(rows[name]["completion"] / totals["completion"])
                if totals["completion"]
                else None
            ),
        }
    out["shared_axis_count"]["completion_selects_the_two_queries_own_mates"] = (
        rd(mate_pair_hits / totals["completion"]) if totals["completion"] else None
    )
    out["shared_axis_count"]["mechanism"] = (
        "a mate shares 6 axis nodes with the query and every other habitat shares "
        "5, so the count is exact for the mate; for completion the same count "
        "ranks the two queries' own mates at 11 and everything else at 10, which "
        "is a coherent answer to a different question"
    )
    out["queries_scored"] = dict(sorted(totals.items()))
    return out


def cheap_rule_verdict(main: dict, control: dict, namespaces: int) -> dict:
    """Which query carries the evidence, stated before any learner is trained."""

    selectors = ("lowest_index", "shared_axis_count", "event_two_hop")
    control_checks = {
        f"{name}/{kind}": ceiling_compatibility(
            control[name][kind],
            control["queries_scored"][kind],
            EXPECTED_CEILINGS[f"localized_{kind}"],
            blocks=namespaces,
        )
        for name in selectors
        for kind in ("mate", "completion")
    }
    return {
        "mate_is_cheap_in_the_main_arm": main["shared_axis_count"]["mate"] == 1.0,
        "structure_aware_cheap_rules_score_zero_on_completion": all(
            main[name]["completion"] == 0.0
            for name in ("shared_axis_count", "event_two_hop")
        ),
        "no_cheap_selector_is_material_on_completion": max(
            main[name]["completion"] for name in selectors
        )
        < CHEAP_RULE_BOUND,
        "cheap_rule_bound": CHEAP_RULE_BOUND,
        "worst_cheap_completion_rate_main": max(
            main[name]["completion"] for name in selectors
        ),
        "control_selectors_vs_exact_ceilings": control_checks,
        "no_control_selector_materially_exceeds_its_ceiling": all(
            row["at_or_below_ceiling"] for row in control_checks.values()
        ),
        "load_bearing_query": "completion",
        "statement": (
            "Query A is exactly solvable in the main arm by a two-line shared-axis "
            "count, so it is the easier half and it is NOT the load-bearing "
            "evidence for global discovery. On Query B both structure-aware cheap "
            "rules score exactly 0.0000 in the main arm, and the index tie-break "
            "sits near uniform, so a learner that recovers completion is not "
            "reproducing a cheap rule. In the control every selector sits at its "
            "exact information ceiling within the declared tolerance, which is "
            "what a graph where all candidates tie should produce"
        ),
        "index_tie_break_note": (
            "the lowest_index rate on completion is a property of the frozen "
            "namespace draw, not a rule: a namespace contributes 12 hits if its "
            "two lowest-numbered habitat nodes happen to be mates and 0 otherwise, "
            "so the rate moves in steps of 12/4032 and its effective sample size "
            "is the 48 namespaces"
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    main = build_observation(build_dataset())
    control = build_localized_observation(main)

    ceiling = nonlearned_ceiling(main)
    control_ceiling = nonlearned_ceiling(control)
    control_census = localized_ceiling_census(control)
    main_cheap = cheap_selectors(main)
    control_cheap = cheap_selectors(control)
    namespaces = len(build_namespaces(main))
    verdict_rows = cheap_rule_verdict(main_cheap, control_cheap, namespaces)

    ladder = {
        "mate": {
            "uniform": main_cheap["uniform"]["mate"],
            "lowest_index": main_cheap["lowest_index"]["mate"],
            "localized_control_exact_ceiling": control_census["mate"]["exact_ceiling"],
            "shared_axis_count_main": main_cheap["shared_axis_count"]["mate"],
            "nonlearned_exact_ceiling_main": ceiling["mate_accuracy"],
        },
        "completion": {
            "uniform": main_cheap["uniform"]["completion"],
            "lowest_index": main_cheap["lowest_index"]["completion"],
            "shared_axis_count_main": main_cheap["shared_axis_count"]["completion"],
            "event_two_hop_main": main_cheap["event_two_hop"]["completion"],
            "localized_control_exact_ceiling": control_census["completion"][
                "exact_ceiling"
            ],
            "nonlearned_exact_ceiling_main": ceiling["completion_exact_set_accuracy"],
        },
    }

    laws = {
        "nonlearned_mate_is_exact": ceiling["mate"]["agrees"],
        "nonlearned_completion_is_exact": ceiling["completion"]["agrees"],
        "nonlearned_plane_is_exact": ceiling["whole_plane"]["accuracy"] == 1.0,
        "nonlearned_mate_rate_is_one": ceiling["mate_accuracy"] == 1.0,
        "nonlearned_completion_rate_is_one": ceiling["completion_exact_set_accuracy"]
        == 1.0,
        "reconstruction_fails_on_the_control": not control_ceiling["reconstruction_valid"],
        "control_mate_ceiling_is_one_thirteenth": control_census["mate"][
            "exact_ceiling"
        ]
        == rd(EXPECTED_CEILINGS["localized_mate"]),
        "control_completion_ceiling_is_one_sixty_sixth": control_census["completion"][
            "exact_ceiling"
        ]
        == rd(EXPECTED_CEILINGS["localized_completion"]),
        "control_ceilings_are_single_orbit_proofs": control_census["mate"]["single_orbit"]
        and control_census["completion"]["single_orbit"],
        "mate_is_cheap_in_the_main_arm": verdict_rows["mate_is_cheap_in_the_main_arm"],
        "structure_aware_cheap_rules_score_zero_on_completion": verdict_rows[
            "structure_aware_cheap_rules_score_zero_on_completion"
        ],
        "no_cheap_selector_is_material_on_completion": verdict_rows[
            "no_cheap_selector_is_material_on_completion"
        ],
        "no_control_selector_materially_exceeds_its_ceiling": verdict_rows[
            "no_control_selector_materially_exceeds_its_ceiling"
        ],
    }
    broken = sorted(name for name, holds in laws.items() if not holds)

    return {
        "module": "fano_baselines",
        "purpose": (
            "011.01 sections 3.2, 3.3 and 7: the nonlearned exact main-arm "
            "ceiling, the exact localized-control information ceilings, and the "
            "deterministic cheap selectors"
        ),
        "selectors": dict(sorted(SELECTORS.items())),
        "nonlearned_ceiling": {"main": ceiling, "localized_control": control_ceiling},
        "localized_control_information_ceiling": control_census,
        "cheap_selectors": {"main": main_cheap, "localized_control": control_cheap},
        "ladder": ladder,
        "which_query_carries_the_evidence": verdict_rows,
        "relational_laws": laws,
        "provenance": {
            "base_commit": BASE_COMMIT,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": "not imported; this module scores no tensors",
        },
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "the nonlearned reconstruction recovers the mate, the completion "
                "and the whole seven-line plane at exactly 1.0000 in every frozen "
                "namespace; the localized control's exact ceilings are 1/13 and "
                "1/66 from constructed orbit witnesses; and in the main arm the "
                "mate is cheaply exact while both structure-aware cheap rules "
                "score exactly 0.0000 on completion and the index tie-break stays "
                "near uniform"
            )
            if not broken
            else "baseline audit FAILED: " + ", ".join(broken),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and compare byte for byte instead of writing",
    )
    args = parser.parse_args()

    payload = audit()
    text = render(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: {OUTPUT} does not exist; run without --check")
        if OUTPUT.read_text() != text:
            raise SystemExit(f"FAIL: {OUTPUT} is not byte-identical to the rebuild")
        if not payload["verdict"]["agrees"]:
            raise SystemExit(f"FAIL: {payload['verdict']['statement']}")
        print(f"PASS: {OUTPUT.name} re-derives byte for byte")
        print(f"PASS: {payload['verdict']['statement']}")
        return
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")
    print(payload["verdict"]["statement"])


if __name__ == "__main__":
    main()
