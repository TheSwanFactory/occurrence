"""011.01 section 8: the predeclared thresholds, scored as written, and the class.

The thresholds are not this module's invention and were not chosen after the run.
They are transcribed from ``011.01`` section 8, which states them numerically, so
there is no freedom here to move one. They are scored on the frozen run cells with
percentile bootstrap intervals over seeds, using the ``009.02`` bootstrap
**imported verbatim** rather than reimplemented.

Section 8's closing instruction is "do not round a near miss into A", and this
module is written so that it cannot: the class is derived from the threshold table
by a chain of explicit conditions, and the summary sentence is generated from that
derivation rather than typed by hand.

Torch, transitively, through the ``009.02`` analysis import. Not in CI.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

from analysis import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CI_LEVEL, bootstrap_ci
from fano_baselines import CEILING_TOLERANCE, ceiling_compatibility
from fano_task import EXPECTED_CEILINGS, rd, render

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_fano_artifacts"
OUTPUT = ARTIFACTS / "fano_analysis.json"

TASK_ARTIFACT = ARTIFACTS / "fano_task.json"
HEADS_ARTIFACT = ARTIFACTS / "fano_heads.json"
BASELINES_ARTIFACT = ARTIFACTS / "fano_baselines.json"
SWEEP_ARTIFACT = ARTIFACTS / "fano_sweep.json"

BASE_COMMIT = "d4efc739ffb9ba58a4e79b1b98b3bd0e5aa8cbb2"

#: ``011.01`` section 8, class A, transcribed. Frozen before the run and not
#: touched after it.
CLASS_A_THRESHOLDS = {
    "mate_exact_accuracy": 0.95,
    "fano_completion_exact_set_accuracy": 0.95,
    "fano_plane_valid_rate": 0.90,
    "certified_plane_equivalent_rate": 0.90,
    "transport_rate": 0.99,
}

#: How far above its exact ceiling the control may sit before the run is invalid.
#: Shared with ``fano_baselines`` so one tolerance governs every ceiling
#: comparison in this turn.
CONTROL_TOLERANCE = CEILING_TOLERANCE

#: What "materially above the control" means for the class-C and class-E tests.
#: Section 8 does not give a number, so one is declared: a factor of two over the
#: exact ceiling, which on a ``1/13`` ceiling is a rate above ``0.1538``.
MATERIAL_FACTOR = 2.0

#: How far a stalled seed's train accuracy may sit from its test accuracy and
#: still be called a fit failure rather than a generalization failure. A stalled
#: run that does no better on the data it optimized than on data it never saw has
#: not overfitted -- it has not fitted.
FIT_GAP_TOLERANCE = 0.02


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"FAIL: {path} does not exist; run its module first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# the threshold table
# ---------------------------------------------------------------------------

def per_seed(runs: list[dict], block: str, arm: str, metric: str) -> list[float]:
    rows = [run for run in runs if run["block"] == block and run["arm"] == arm]
    if metric in ("mate_exact_accuracy", "fano_completion_exact_set_accuracy"):
        return [row[metric] for row in rows]
    return [row["recovery"][metric] for row in rows] if rows and "recovery" in rows[0] else []


def transport_per_seed(runs: list[dict], block: str, arm: str) -> list[float]:
    rows = [run for run in runs if run["block"] == block and run["arm"] == arm]
    return [row["transport"]["transport_rate"] for row in rows if "transport" in row]


def threshold_table(runs: list[dict], block: str) -> dict:
    """Every class-A threshold, scored on the main arm with a bootstrap interval."""

    out = {}
    for metric, threshold in CLASS_A_THRESHOLDS.items():
        values = (
            transport_per_seed(runs, block, "main")
            if metric == "transport_rate"
            else per_seed(runs, block, "main", metric)
        )
        if not values:
            out[metric] = {"threshold": threshold, "observed": None, "passes": False}
            continue
        mean = sum(values) / len(values)
        interval = bootstrap_ci(values)
        out[metric] = {
            "threshold": threshold,
            "observed": rd(mean),
            "ci_low": interval["low"],
            "ci_high": interval["high"],
            "seeds": interval["n"],
            "per_seed": [rd(value) for value in values],
            "passes": mean >= threshold,
            "ci_low_passes": interval["low"] is not None and interval["low"] >= threshold,
        }
    return out


def seed_partition(runs: list[dict], block: str, arm: str) -> dict:
    """The per-seed outcome distribution, because the mean hides it.

    A block whose seeds are bimodal -- some at the exact loss minimum, some at the
    uniform-choice loss -- has a mean that describes neither. The partition is
    reported so the reader sees the shape and not just the average.
    """

    rows = [run for run in runs if run["block"] == block and run["arm"] == arm]
    if not rows:
        return {}
    converged = [
        row
        for row in rows
        if row["mate_exact_accuracy"] == 1.0
        and row["fano_completion_exact_set_accuracy"] == 1.0
    ]
    stalled = [row for row in rows if row not in converged]
    gaps = []
    for row in stalled:
        for kind in ("mate", "completion"):
            gaps.append(
                abs(
                    row["per_split"]["train"][kind]["accuracy"]
                    - row["per_split"]["test"][kind]["accuracy"]
                )
            )
    return {
        "seeds": len(rows),
        "converged_seeds": [row["seed"] for row in converged],
        "stalled_seeds": [row["seed"] for row in stalled],
        "converged_count": len(converged),
        "final_loss_per_seed": [rd(row["final_loss"]) for row in rows],
        "loss_minimum": 0.346574,
        "seeds_at_the_exact_loss_minimum": sum(
            1 for row in rows if row["final_loss"] < 0.35
        ),
        "stalled_train_versus_test": [
            {
                "seed": row["seed"],
                "train": {
                    kind: row["per_split"]["train"][kind]["accuracy"]
                    for kind in ("mate", "completion")
                },
                "test": {
                    kind: row["per_split"]["test"][kind]["accuracy"]
                    for kind in ("mate", "completion")
                },
            }
            for row in stalled
        ],
        "worst_stalled_train_test_gap": rd(max(gaps)) if gaps else None,
        "stalled_gap_tolerance": FIT_GAP_TOLERANCE,
        "stalled_seeds_are_fit_failures_not_generalization_failures": bool(gaps)
        and max(gaps) <= FIT_GAP_TOLERANCE,
        "converged_seed_metrics": {
            "mate_exact_accuracy": 1.0 if converged else None,
            "fano_completion_exact_set_accuracy": 1.0 if converged else None,
            "fano_plane_valid_rate": (
                rd(
                    sum(row["recovery"]["fano_plane_valid_rate"] for row in converged)
                    / len(converged)
                )
                if converged
                else None
            ),
            "transport_rate": (
                rd(
                    sum(row["transport"]["transport_rate"] for row in converged)
                    / len(converged)
                )
                if converged
                else None
            ),
        },
        "reading": (
            "the converged-seed figures are a CONDITIONAL statement about the "
            "seeds that fitted, not a result. Section 8's thresholds are scored on "
            "all frozen run cells and that is what the class is derived from"
        ),
    }


def control_table(runs: list[dict], block: str, test_namespaces: int) -> dict:
    """The control against its exact information ceilings."""

    rows = [run for run in runs if run["block"] == block and run["arm"] == "localized"]
    if not rows:
        return {}
    out = {}
    for kind, metric in (
        ("mate", "mate_exact_accuracy"),
        ("completion", "fano_completion_exact_set_accuracy"),
    ):
        values = [row[metric] for row in rows]
        mean = sum(values) / len(values)
        queries = sum(
            row["per_split"]["test"][kind]["queries"] for row in rows
        )
        out[kind] = {
            **ceiling_compatibility(
                mean,
                queries,
                EXPECTED_CEILINGS[f"localized_{kind}"],
                blocks=test_namespaces * len(rows),
            ),
            "per_seed": [rd(value) for value in values],
            "bootstrap": bootstrap_ci(values),
        }
    plane = [row["recovery"]["fano_plane_valid_rate"] for row in rows]
    out["fano_plane_valid_rate"] = {
        "observed": rd(sum(plane) / len(plane)),
        "per_seed": [rd(value) for value in plane],
        "expected": 0.0,
        "agrees": all(value == 0.0 for value in plane),
    }
    return out


# ---------------------------------------------------------------------------
# anti-cheating roll-up
# ---------------------------------------------------------------------------

def conformance(task: dict, heads: dict, baselines: dict, sweep: dict) -> dict:
    """Section 7's mechanical audits, rolled into one pass/fail ledger."""

    swap_rates = [
        run["left_right_swap"]["identical_rate"]
        for run in sweep["runs"]
        if "left_right_swap" in run
    ]
    checks = {
        "gate_0_passes": task["gate_0"]["passes"],
        "every_gate_0_pin_agrees": all(
            entry["agrees"] for entry in task["pins"].values()
        ),
        "no_hidden_field_on_the_learned_path": heads["answer_rule"]["clean"],
        "no_identity_parameter": heads["anti_identity"]["no_forbidden_dimension"]
        and heads["anti_identity"]["no_embedding_module"]
        and heads["anti_identity"]["no_buffer"],
        "parameter_count_is_node_count_invariant": heads["parameter_independence"][
            "parameter_count_is_node_count_invariant"
        ],
        "completion_query_is_swap_invariant": heads["swap_symmetry"][
            "scores_bitwise_identical"
        ]["agrees"],
        "decision_covariance_is_exact_on_decisive_rows": heads[
            "permutation_covariance"
        ]["decision_covariance_is_exact_on_decisive_rows"],
        "edge_paths_agree": heads["edge_paths"]["edge_paths_agree"],
        "control_differs_only_in_cross_habitat_sharing": task["gate_0"][
            "localized_control"
        ]["locality"]["within_habitat_left_and_right_patterns_identical"],
        "control_shares_no_axis_across_habitats": task["gate_0"]["localized_control"][
            "locality"
        ]["left_nodes_shared_across_habitats"]["control"]
        == 0,
        "control_does_not_identify_the_plane": not task["gate_0"]["localized_control"][
            "reconstruction_valid"
        ],
        "nonlearned_ceiling_is_exact": baselines["nonlearned_ceiling"]["main"][
            "mate"
        ]["agrees"]
        and baselines["nonlearned_ceiling"]["main"]["completion"]["agrees"],
        "global_left_right_swap_moves_no_answer": bool(swap_rates)
        and all(rate == 1.0 for rate in swap_rates),
    }
    return {
        "checks": checks,
        "failures": sorted(name for name, holds in checks.items() if not holds),
        "clean": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# the disposition
# ---------------------------------------------------------------------------

def disposition(
    table: dict, controls: dict, conformance_ledger: dict, partition: dict
) -> dict:
    """Derive A/B/C/D/E from the table. Nothing here is typed by hand."""

    mate = table["mate_exact_accuracy"]
    completion = table["fano_completion_exact_set_accuracy"]
    plane = table["fano_plane_valid_rate"]
    equivalent = table["certified_plane_equivalent_rate"]
    transport = table["transport_rate"]

    control_ok = all(
        controls[kind]["at_or_below_ceiling"] for kind in ("mate", "completion")
    )
    material_mate = mate["observed"] is not None and mate["observed"] >= (
        MATERIAL_FACTOR * EXPECTED_CEILINGS["localized_mate"]
    )
    material_completion = completion["observed"] is not None and completion[
        "observed"
    ] >= (MATERIAL_FACTOR * EXPECTED_CEILINGS["localized_completion"])

    queries_pass = mate["passes"] and completion["passes"]
    structure_pass = plane["passes"] and equivalent["passes"]
    transport_pass = transport["passes"]

    if not conformance_ledger["clean"] or not control_ok:
        klass = "D"
        because = (
            "an anti-cheating or conformance condition failed: "
            + ", ".join(conformance_ledger["failures"] or ["control exceeds its ceiling"])
        )
    elif not material_mate and not material_completion:
        klass = "E"
        because = (
            "the main learner did not rise materially above the exact control "
            "ceiling on either query"
        )
    elif queries_pass and structure_pass and transport_pass:
        klass = "A"
        because = "every class-A threshold passes on the frozen run cells"
    elif queries_pass and not structure_pass:
        klass = "B"
        because = (
            "the query thresholds pass but the plane-valid or certified-equivalence "
            "threshold misses"
        )
    else:
        klass = "C"
        because = (
            "the same-FFF pairing is materially learned above the control ceiling "
            f"({mate['observed']:.4f} against an exact {EXPECTED_CEILINGS['localized_mate']:.4f}) "
            "but the class-A thresholds are not all met on the frozen run cells: "
            + ", ".join(
                name
                for name, entry in table.items()
                if not entry["passes"]
            )
        )

    return {
        "class": klass,
        "because": because,
        "queries_pass": queries_pass,
        "structure_pass": structure_pass,
        "transport_pass": transport_pass,
        "control_at_or_below_ceiling": control_ok,
        "materially_above_control": {
            "mate": material_mate,
            "completion": material_completion,
            "material_factor": MATERIAL_FACTOR,
        },
        "missed_thresholds": sorted(
            name for name, entry in table.items() if not entry["passes"]
        ),
        "not_rounded_into_a": (
            "section 8 forbids rounding a near miss into A. The class above is "
            "derived from the threshold table by explicit conditions, and the "
            f"seed partition shows why the mean misses: {partition.get('converged_count')} "
            f"of {partition.get('seeds')} seeds reach the exact loss minimum and "
            "score 1.0000 on every metric, and the rest do not fit at all"
        ),
        "stalled_seeds_are_fit_failures": partition.get(
            "stalled_seeds_are_fit_failures_not_generalization_failures"
        ),
    }


def recommendation(disposition_row: dict, partition: dict, repair: dict) -> dict:
    return {
        "class": disposition_row["class"],
        "what_is_established": (
            "Gate 0 establishes that the anonymous typed observation determines the "
            "certified seven-point Fano quotient up to GL(3,2), with an exact "
            "nonlearned ceiling of 1.0000 and exact control ceilings of 1/13 and "
            "1/66 from constructed orbit witnesses. The learner reaches the ceiling "
            "on a majority of frozen seeds, with the whole-plane reconstruction and "
            "fresh-name transport at 1.0000 on exactly those seeds"
        ),
        "what_is_not_established": (
            "H_discovery_global_finite is NOT banked at class A. The seed-mean "
            "misses the section 8 thresholds because three of eight seeds do not "
            "fit, and section 8 forbids rounding that into A"
        ),
        "the_open_question_is_optimization_not_identifiability": (
            "every stalled seed shows train accuracy equal to test accuracy, so no "
            "generalization claim is at stake. The one permitted repair changed the "
            "message form and did not move the plateau, which locates the failure "
            "outside the message mechanism"
        ),
        "repair_outcome": repair,
        "suggested_next_turn": (
            "one further local turn on optimization only, with the observation, the "
            "control, the thresholds and the recovery audit unchanged. The declared "
            "architecture reaches the exact loss minimum whenever it escapes the "
            "plateau, so the question is initialization and conditioning, not "
            "representation. That is an Owner decision because 011.01 has now spent "
            "its one permitted repair"
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    task = load(TASK_ARTIFACT)
    heads = load(HEADS_ARTIFACT)
    baselines = load(BASELINES_ARTIFACT)
    sweep = load(SWEEP_ARTIFACT)
    runs = sweep["runs"]
    blocks = tuple(dict.fromkeys(run["block"] for run in runs))
    test_namespaces = task["episodes"]["main"]["namespaces"]["per_split"]["test"]

    tables = {block: threshold_table(runs, block) for block in blocks}
    controls = {block: control_table(runs, block, test_namespaces) for block in blocks}
    partitions = {
        block: seed_partition(runs, block, "main") for block in blocks
    }
    ledger = conformance(task, heads, baselines, sweep)

    primary = "primary" if "primary" in blocks else blocks[0]
    repair_block = "repair" if "repair" in blocks else None
    repair_outcome = {
        "declared": sweep["diagnosed_pathology"]["repair"],
        "ran": repair_block is not None,
    }
    if repair_block:
        repair_outcome.update(
            {
                "mate_exact_accuracy": tables[repair_block]["mate_exact_accuracy"][
                    "observed"
                ],
                "fano_completion_exact_set_accuracy": tables[repair_block][
                    "fano_completion_exact_set_accuracy"
                ]["observed"],
                "seeds_at_the_exact_loss_minimum": partitions[repair_block][
                    "seeds_at_the_exact_loss_minimum"
                ],
                "moved_the_plateau": partitions[repair_block]["converged_count"]
                > partitions[primary]["converged_count"],
                "judged_block": (
                    "the repaired block is judged if and only if it improves on the "
                    "primary; the primary is reported either way"
                ),
            }
        )

    judged = primary
    if repair_block and repair_outcome.get("moved_the_plateau"):
        judged = repair_block

    verdict = disposition(
        tables[judged], controls[judged], ledger, partitions[judged]
    )

    ladder = {
        "mate": {
            "uniform": baselines["ladder"]["mate"]["uniform"],
            "localized_control_exact_ceiling": baselines["ladder"]["mate"][
                "localized_control_exact_ceiling"
            ],
            "localized_control_learner": controls[judged]["mate"]["observed"],
            "main_learner": tables[judged]["mate_exact_accuracy"]["observed"],
            "nonlearned_exact_ceiling": baselines["ladder"]["mate"][
                "nonlearned_exact_ceiling_main"
            ],
        },
        "completion": {
            "uniform": baselines["ladder"]["completion"]["uniform"],
            "cheap_rules_main": baselines["ladder"]["completion"][
                "shared_axis_count_main"
            ],
            "localized_control_exact_ceiling": baselines["ladder"]["completion"][
                "localized_control_exact_ceiling"
            ],
            "localized_control_learner": controls[judged]["completion"]["observed"],
            "main_learner": tables[judged]["fano_completion_exact_set_accuracy"][
                "observed"
            ],
            "nonlearned_exact_ceiling": baselines["ladder"]["completion"][
                "nonlearned_exact_ceiling_main"
            ],
        },
    }

    return {
        "module": "fano_analysis",
        "purpose": (
            "011.01 section 8: the predeclared thresholds scored on the frozen run "
            "cells with bootstrap intervals, the control against its exact "
            "ceilings, the anti-cheating ledger, and the disposition"
        ),
        "thresholds": CLASS_A_THRESHOLDS,
        "threshold_source": (
            "011.01 section 8, transcribed numerically; no threshold was chosen or "
            "moved by this module"
        ),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "imported_from": "analysis.bootstrap_ci (the 009.02 implementation)",
        },
        "blocks": list(blocks),
        "judged_block": judged,
        "threshold_table": tables,
        "control_vs_exact_ceiling": controls,
        "seed_partition": partitions,
        "conformance": ledger,
        "ladder": ladder,
        "repair": repair_outcome,
        "disposition": verdict,
        "recommendation": recommendation(verdict, partitions[judged], repair_outcome),
        "control_tolerance": CONTROL_TOLERANCE,
        "provenance": {
            "base_commit": BASE_COMMIT,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "inputs": {
                "fano_task": task["digests"]["main_observation_sha256"],
                "fano_sweep_runs": sweep["digests"]["runs_digest"],
            },
        },
        "verdict": {
            "class": verdict["class"],
            "statement": (
                f"disposition {verdict['class']}: {verdict['because']}"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare byte for byte")
    args = parser.parse_args()

    payload = audit()
    text = render(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: {OUTPUT} does not exist; run without --check")
        if OUTPUT.read_text() != text:
            raise SystemExit(f"FAIL: {OUTPUT} is not byte-identical to the rebuild")
        print(f"PASS: {OUTPUT.name} re-derives byte for byte")
        print(f"PASS: {payload['verdict']['statement']}")
        return
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")
    print(payload["verdict"]["statement"])
    for name, entry in payload["threshold_table"][payload["judged_block"]].items():
        mark = "PASS" if entry["passes"] else "MISS"
        print(
            f"  {mark}  {name:38s} {entry['observed']:.4f} "
            f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]  >= {entry['threshold']}"
        )
    counts = Counter()
    for block, partition in payload["seed_partition"].items():
        counts[block] = partition["converged_count"]
    print("  converged seeds per block:", dict(counts))


if __name__ == "__main__":
    main()
