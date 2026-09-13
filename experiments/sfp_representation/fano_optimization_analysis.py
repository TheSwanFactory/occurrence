"""011.03 analysis: reliability, unchanged thresholds, controls, and class.

The disposition is derived from the committed TRAIN-only search and, when the
8/8 gate opened, the locked fresh evaluation.  No threshold or control tolerance
is introduced here: the 011.01 values are imported unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fano_analysis import CLASS_A_THRESHOLDS
from fano_baselines import CEILING_TOLERANCE
from fano_optimization_core import file_sha256
from fano_optimization_evaluate import check as check_evaluation
from fano_optimization_manifest import DEV_SEEDS, FRESH_SEEDS, object_sha256
from fano_optimization_search import check as check_search

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_optimization_artifacts"
BASELINE = ARTIFACTS / "baseline_replay.json"
CANDIDATE_MANIFEST = ARTIFACTS / "candidate_manifest.json"
CAPSULE_MANIFEST = ARTIFACTS / "train_capsule.json"
SEARCH = ARTIFACTS / "development_search.json"
SELECTED = ARTIFACTS / "selected_optimization.json"
EVALUATION = ARTIFACTS / "fresh_evaluation.json"
OUTPUT = ARTIFACTS / "optimization_analysis.json"

FROZEN_ARTIFACTS = {
    "task": ROOT / "011_fano_artifacts" / "fano_task.json",
    "heads": ROOT / "011_fano_artifacts" / "fano_heads.json",
    "baselines": ROOT / "011_fano_artifacts" / "fano_baselines.json",
}

DISPOSITION_NAMES = {
    "A": "optimization closure; bank global finite discovery",
    "B": "mean thresholds pass but reliability does not",
    "C": "improved but still below scientific thresholds",
    "D": "invalid",
    "E": "optimization-only closure fails before fresh evaluation",
}


def render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def load(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"required artifact does not exist: {path}")
    return json.loads(path.read_text())


def _fresh_metrics(evaluation: dict) -> dict:
    summary = evaluation["summary"]["fresh/main"]
    sources = {
        "mate_exact_accuracy": "mate_exact_accuracy",
        "fano_completion_exact_set_accuracy": (
            "fano_completion_exact_set_accuracy"
        ),
        "mate_pairing_valid_rate": "mate_pairing_valid_rate",
        "fano_plane_valid_rate": "fano_plane_valid_rate",
        "certified_plane_equivalent_rate": "certified_plane_equivalent_rate",
        "transport_rate": "transport_rate",
        "left_right_swap_identical_rate": "left_right_swap_identical_rate",
    }
    return {
        name: {
            "observed": summary[source]["mean"],
            "per_seed": summary[source]["per_seed"],
            **(
                {
                    "threshold": CLASS_A_THRESHOLDS[name],
                    "passes": summary[source]["mean"]
                    >= CLASS_A_THRESHOLDS[name],
                }
                if name in CLASS_A_THRESHOLDS
                else {}
            ),
        }
        for name, source in sources.items()
    }


def _legacy_determinism(search: dict, evaluation: dict) -> dict:
    selected_id = search["selected_candidate_id"]
    development = {
        row["seed"]: row
        for row in search["runs"]
        if row["candidate_id"] == selected_id
    }
    legacy = {
        row["seed"]: row["training"]
        for row in evaluation["runs"]
        if row["block"] == "legacy_bridge" and row["arm"] == "main"
    }
    fields = (
        "steps_to_convergence",
        "updates_completed",
        "final_train_loss_hex",
        "initial_state_sha256",
        "final_state_sha256",
        "parameters_finite",
        "deterministic_algorithms",
    )
    per_seed = {}
    for seed in DEV_SEEDS:
        comparisons = {
            field: development[seed][field] == legacy[seed][field]
            for field in fields
        }
        comparisons["train_fit"] = development[seed]["train_fit"] == legacy[seed][
            "train_fit"
        ]
        per_seed[str(seed)] = {
            "comparisons": comparisons,
            "exact": all(comparisons.values()),
        }
    return {
        "fields": list(fields) + ["train_fit"],
        "per_seed": per_seed,
        "all_8_retrainings_are_exact": all(
            row["exact"] for row in per_seed.values()
        ),
    }


def _source_conformance(capsule: dict) -> dict:
    observed = {
        name: file_sha256(ROOT / name)
        for name in capsule["frozen_source_sha256"]
    }
    expected = capsule["frozen_source_sha256"]
    return {
        "expected": expected,
        "observed": observed,
        "unchanged": observed == expected,
    }


def _conformance(
    frozen: dict[str, dict],
    capsule: dict,
    search: dict,
    evaluation: dict,
    determinism: dict,
) -> dict:
    fresh_main = [
        row
        for row in evaluation["runs"]
        if row["block"] == "fresh" and row["arm"] == "main"
    ]
    fresh_control = [
        row
        for row in evaluation["runs"]
        if row["block"] == "fresh" and row["arm"] == "localized"
    ]
    controls = evaluation["control_vs_exact_ceiling"]
    source = _source_conformance(capsule)
    commitment = evaluation["commitment"]
    checks = {
        "gate_0_passes": frozen["task"]["gate_0"]["passes"],
        "every_gate_0_pin_agrees": all(
            row["agrees"] for row in frozen["task"]["pins"].values()
        ),
        "frozen_011_02_sources_unchanged": source["unchanged"],
        "original_answer_rule_audit_clean": frozen["heads"]["answer_rule"][
            "clean"
        ],
        "optimization_learned_path_audit_clean": search["learned_path_audit"][
            "clean"
        ],
        "selection_boundary_is_train_only": search[
            "selection_boundary_audit"
        ]["clean"],
        "selected_candidate_reached_8_of_8": search[
            "selected_converged_count"
        ]
        == len(DEV_SEEDS),
        "selected_artifact_was_committed": commitment["checks"][
            "selected_artifact_committed_byte_for_byte"
        ],
        "no_fresh_seed_in_development_search": commitment["checks"][
            "no_fresh_seed_in_development_search"
        ],
        "fresh_output_absent_at_commitment": commitment["checks"][
            "fresh_output_absent"
        ],
        "legacy_retraining_is_bit_exact": determinism[
            "all_8_retrainings_are_exact"
        ],
        "deterministic_algorithms_all_fresh_main": all(
            row["training"]["deterministic_algorithms"] for row in fresh_main
        ),
        "deterministic_algorithms_all_fresh_control": all(
            row["training"]["deterministic_algorithms"] for row in fresh_control
        ),
        "fresh_seed_block_is_exact": sorted(row["seed"] for row in fresh_main)
        == list(FRESH_SEEDS),
        "control_seed_block_is_exact": sorted(
            row["seed"] for row in fresh_control
        )
        == list(FRESH_SEEDS),
        "global_left_right_swap_moves_no_answer": all(
            row["left_right_swap"]["identical_rate"] == 1.0
            for row in [*fresh_main, *fresh_control]
        ),
        "validity_entails_equivalence": all(
            row["recovery"]["fano_plane_valid_rate"]
            == row["recovery"]["certified_plane_equivalent_rate"]
            for row in fresh_main
        ),
        "localized_mate_within_exact_ceiling": controls["mate"][
            "at_or_below_ceiling"
        ],
        "localized_completion_within_exact_ceiling": controls["completion"][
            "at_or_below_ceiling"
        ],
        "localized_control_never_recovers_plane": controls[
            "never_recovers_a_plane"
        ],
    }
    return {
        "checks": checks,
        "failures": sorted(name for name, holds in checks.items() if not holds),
        "clean": all(checks.values()),
        "frozen_source_sha256": source,
    }


def _disposition(
    search: dict,
    metrics: dict,
    reliability: dict,
    controls: dict,
    conformance: dict,
) -> dict:
    scientific = {
        name: metrics[name]["passes"] for name in CLASS_A_THRESHOLDS
    }
    thresholds_pass = all(scientific.values())
    control_pass = all(
        controls[name]["at_or_below_ceiling"] for name in ("mate", "completion")
    ) and controls["never_recovers_a_plane"]
    reliable = reliability["fresh_converged_count"] == len(FRESH_SEEDS)

    if not conformance["clean"] or not control_pass:
        klass = "D"
        because = "a conformance, determinism, anti-cheating, or control check failed"
    elif reliable and thresholds_pass:
        klass = "A"
        because = (
            "16/16 fresh seeds converge and every unchanged scientific threshold, "
            "control ceiling, and audit passes"
        )
    elif thresholds_pass:
        klass = "B"
        because = (
            "all original scientific thresholds pass but fewer than 16/16 fresh "
            "seeds meet the frozen convergence definition"
        )
    else:
        klass = "C"
        because = (
            "the preregistered optimization improves development convergence from "
            "5/8 to 8/8 but one or more unchanged scientific thresholds miss on "
            "the fresh block"
        )
    return {
        "class": klass,
        "name": DISPOSITION_NAMES[klass],
        "because": because,
        "fresh_reliability_passes": reliable,
        "scientific_thresholds": scientific,
        "all_scientific_thresholds_pass": thresholds_pass,
        "control_passes": control_pass,
        "missed_thresholds": sorted(
            name for name, passes in scientific.items() if not passes
        ),
        "development_improvement": {
            "baseline_converged": 5,
            "selected_converged": search["selected_converged_count"],
        },
        "recommendation": (
            "recommend that the Owner bank H_discovery_global_finite as SUPPORTED "
            "on the frozen finite observation"
            if klass == "A"
            else "do not bank H_discovery_global_finite from this execution"
        ),
    }


def analysis() -> dict:
    baseline = load(BASELINE)
    manifest = load(CANDIDATE_MANIFEST)
    capsule = load(CAPSULE_MANIFEST)
    search = load(SEARCH)
    frozen = {name: load(path) for name, path in FROZEN_ARTIFACTS.items()}

    if not search["eligible_for_fresh_evaluation"]:
        body = {
            "schema": "occurrence.011.03.optimization-analysis.v1",
            "baseline_replay": baseline,
            "candidate_manifest_sha256": manifest["manifest_sha256"],
            "development_search_sha256": search["search_sha256"],
            "fresh_evaluation_ran": False,
            "disposition": {
                "class": "E",
                "name": DISPOSITION_NAMES["E"],
                "because": "no preregistered candidate reached 8/8 development convergence",
                "recommendation": (
                    "do not bank H_discovery_global_finite from this execution"
                ),
            },
        }
        return {**body, "analysis_sha256": object_sha256(body)}

    selected = load(SELECTED)
    evaluation = load(EVALUATION)
    metrics = _fresh_metrics(evaluation)
    determinism = _legacy_determinism(search, evaluation)
    conformance = _conformance(
        frozen,
        capsule,
        search,
        evaluation,
        determinism,
    )
    reliability_rows = [
        row
        for row in evaluation["runs"]
        if row["block"] == "fresh" and row["arm"] == "main"
    ]
    reliability = {
        "required": len(FRESH_SEEDS),
        "fresh_converged_count": sum(
            row["training"]["converged"] for row in reliability_rows
        ),
        "per_seed": {
            str(row["seed"]): {
                "converged": row["training"]["converged"],
                "steps_to_convergence": row["training"][
                    "steps_to_convergence"
                ],
                "final_train_loss": row["training"]["final_train_loss"],
                "mate_train_exact": row["training"]["train_fit"]["mate"][
                    "exact"
                ],
                "completion_train_exact": row["training"]["train_fit"][
                    "completion"
                ]["exact"],
            }
            for row in reliability_rows
        },
    }
    controls = evaluation["control_vs_exact_ceiling"]
    disposition = _disposition(
        search,
        metrics,
        reliability,
        controls,
        conformance,
    )
    body = {
        "schema": "occurrence.011.03.optimization-analysis.v1",
        "baseline_replay": baseline,
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "train_capsule_manifest_sha256": capsule["manifest_sha256"],
        "development_search_sha256": search["search_sha256"],
        "selected_optimization_sha256": selected[
            "selected_optimization_sha256"
        ],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "pre_evaluation_commit": evaluation["commitment"][
            "pre_evaluation_commit"
        ],
        "fresh_evaluation_ran": True,
        "thresholds": CLASS_A_THRESHOLDS,
        "control_tolerance": CEILING_TOLERANCE,
        "fresh_metrics": metrics,
        "fresh_reliability": reliability,
        "localized_control": controls,
        "legacy_determinism": determinism,
        "conformance": conformance,
        "disposition": disposition,
        "interpretation": {
            "identifiability": "already exact at Gate 0",
            "representation": "already achievable by the frozen architecture",
            "optimization": "the only variable changed in 011.03",
            "generalization": (
                "scored only after the optimization prescription was committed"
            ),
        },
        "finite_claim_only": (
            "the seven-point Fano / FFF incidence quotient can be learned from the "
            "anonymous finite FIPS habitat support-incidence observation, up to "
            "its natural symmetry"
        ),
    }
    return {**body, "analysis_sha256": object_sha256(body)}


def write() -> dict:
    payload = analysis()
    OUTPUT.write_text(render(payload))
    print(
        f"wrote {OUTPUT}; disposition {payload['disposition']['class']} — "
        f"{payload['disposition']['name']}"
    )
    return payload


def check() -> dict:
    check_search()
    search = load(SEARCH)
    if search["eligible_for_fresh_evaluation"]:
        check_evaluation()
    expected = analysis()
    if not OUTPUT.exists() or load(OUTPUT) != expected:
        raise SystemExit("FAIL: optimization analysis does not re-derive")
    if object_sha256(
        {
            key: value
            for key, value in expected.items()
            if key != "analysis_sha256"
        }
    ) != expected["analysis_sha256"]:
        raise SystemExit("FAIL: optimization analysis hash differs")
    print(
        f"PASS: {OUTPUT.name} re-derives; disposition "
        f"{expected['disposition']['class']}"
    )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        write()


if __name__ == "__main__":
    main()
