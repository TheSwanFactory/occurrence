"""Analyze Issue 014.05a trajectories with the accepted 014.03a definitions.

Run once after aggregation for temporal diagnostics, then run the transplant and
run this analyzer again.  Pass two adds component evidence and rewrites the same
``analysis.json`` and ``comparison_to_01403a.json`` without changing temporal
quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISSUE_DIR = HERE.parent
PREDECESSOR = ISSUE_DIR / "014.03a-Code-attachments"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"BLOCKED: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_module("run_014_tied_for_analysis", HERE / "run_014_tied.py")
OLD = load_module("analyze_014_predecessor", PREDECESSOR / "analyze_014.py")


def digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> dict:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw = path.read_bytes()
    loaded = json.loads(raw.decode("utf-8"))
    if digest(loaded) != digest(payload):
        raise SystemExit(f"BLOCKED: readback mismatch for {path.name}")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"BLOCKED: missing required artifact {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def classify_run(row: dict) -> dict:
    """Do not force a nonfitting run into predecessor reading M4."""

    if row["t_fit"] is None:
        return {
            "readings_applicable": [],
            "reading": "capacity-unqualified",
            "ordering": {
                "t_fit": None,
                "t_repr": row["t_repr"],
                "t_gen": row["t_gen"],
                "t_repr_before_t_gen": None,
            },
        }
    return OLD.classify(row)


def fit_by_8192(row: dict) -> bool:
    return row["t_fit"] is not None and row["t_fit"] <= 8192


def summarize(rows: list[dict], arm_capacity_qualified: bool, basis: str) -> dict:
    def reached(field: str) -> list[int]:
        return [row[field] for row in rows if row[field] is not None]

    readings: dict[str, int] = {}
    for row in rows:
        readings[row["reading"]] = readings.get(row["reading"], 0) + 1
    result = {
        "seeds": len(rows),
        "fit_by_update_8192": sum(fit_by_8192(row) for row in rows),
        "arm_capacity_qualified": arm_capacity_qualified,
        "arm_capacity_qualification_basis": basis,
        "individual_capacity_unqualified": sum(
            row["reading"] == "capacity-unqualified" for row in rows
        ),
        "reading_histogram": readings,
        "delayed_generalization_flagged": sum(
            row["delayed_generalization_flag"] for row in rows
        ),
        "swap_invariance_bit_exact_everywhere": all(
            row["swap_invariance_bit_exact_at_every_checkpoint"] for row in rows
        ),
    }
    for field in (
        "t_fit",
        "t_gen",
        "t_global",
        "t_repr",
        "t_repr_global",
        "t_repr_exact",
    ):
        values = reached(field)
        result[f"{field}_reached"] = len(values)
        result[f"median_{field}"] = statistics.median(values) if values else None
    for field in (
        "role_test_accuracy",
        "novel_test_accuracy",
        "probe_role_accuracy",
        "probe_novel_accuracy",
    ):
        result[f"final_{field}_median"] = statistics.median(
            row["final"][field] for row in rows
        )
    result["seeds_with_exact_56_block_readout"] = sum(
        row["final"]["full_56_block_relation_exact"] for row in rows
    )
    return result


def transplant_reading() -> dict | None:
    path = HERE / "component_transplants.json"
    if not path.is_file():
        return None
    artifact = load_json(path)
    rows = [row for row in artifact["runs"] if row.get("hybrids")]
    if not rows:
        return {
            "seeds": 0,
            "reading": "no fitted matched_regularized seed was eligible",
            "fence": artifact["interpretation_fence"],
        }
    keys = (
        "early_representation_early_downstream",
        "late_representation_late_downstream",
        "late_representation_early_downstream",
        "early_representation_late_downstream",
    )
    return {
        "seeds": len(rows),
        "means": {
            key: {
                split: statistics.mean(
                    row["hybrids"][key][split] for row in rows
                )
                for split in (
                    "TRAIN",
                    "ROLE_TEST",
                    "NOVEL_TEST",
                    "shared_event_embedding_norm",
                )
            }
            for key in keys
        },
        "all_replays_exact_including_shared_norm": all(
            row["replay_check"][
                "behavioral_fields_exact_including_shared_norm"
            ]
            for row in rows
        ),
        "fence": artifact["interpretation_fence"],
    }


def overall_tied_classification(
    temporal: list[dict], summary: dict, fallback_status: str
) -> dict:
    """Apply the task's T0-T4 hierarchy conservatively and mechanically."""

    primary_qualified = [
        arm
        for arm in RUN.PRIMARY_ARMS
        if summary[arm]["arm_capacity_qualified"]
    ]
    if primary_qualified:
        effective_arms = primary_qualified
        capacity_basis = "primary arm reached at least 7/8 exact TRAIN fits by update 8192"
    elif "capacity_fallback" in summary and summary["capacity_fallback"][
        "arm_capacity_qualified"
    ]:
        effective_arms = ["capacity_fallback"]
        capacity_basis = "fallback was capacity-qualified by the pre-scored 4/4 TRAIN-only calibration"
    else:
        return {
            "classification": "T0",
            "rule": "tied model cannot reliably fit TRAIN; no grokking conclusion",
            "effective_arms": [],
            "capacity_basis": fallback_status,
        }

    rows = [
        row
        for row in temporal
        if row["arm"] in effective_arms and row["t_fit"] is not None
    ]
    if not rows:
        return {
            "classification": "T0",
            "rule": "no scored run in the effective capacity arm fit TRAIN; no grokking conclusion",
            "effective_arms": effective_arms,
            "capacity_basis": capacity_basis,
        }
    if any(
        row["t_global"] is not None
        or row["t_repr_global"] is not None
        or row["t_repr_exact"] is not None
        for row in rows
    ):
        label = "T4"
        rule = "NOVEL_TEST, t_repr_global, or exact full-relation evidence emerged"
    elif any(
        row["t_gen"] is not None
        and row["t_repr"] is not None
        and row["t_repr"] <= row["t_gen"]
        for row in rows
    ):
        label = "T3"
        rule = "ROLE_TEST and readable representation emerged with t_repr no later than t_gen"
    elif any(row["t_gen"] is not None for row in rows):
        label = "T2"
        rule = "ROLE_TEST emerged without stronger unseen-block evidence and without the conservative T3 ordering"
    else:
        label = "T1"
        rule = "capacity-qualified tied model fit TRAIN but gained no ROLE_TEST consequence; tying alone was insufficient under this signal"
    return {
        "classification": label,
        "rule": rule,
        "effective_arms": effective_arms,
        "capacity_basis": capacity_basis,
        "precedence": "T0 capacity gate; otherwise T4 > T3 > T2 > T1",
        "near_note": "the task gives no numeric 'near' window, so T3 uses only the predeclared unambiguous condition t_repr <= t_gen and does not invent a post-hoc tolerance",
    }


def build_comparison(
    tied_classification: dict, summary: dict, temporal: list[dict]
) -> dict:
    predecessor_analysis_path = PREDECESSOR / "analysis.json"
    predecessor_diagnostic_path = PREDECESSOR / "negative_result_diagnostic.json"
    predecessor = load_json(predecessor_analysis_path)
    diagnostic = load_json(predecessor_diagnostic_path)
    old_summary = predecessor["summary_by_arm"]
    return {
        "module": "analyze_014_tied.py",
        "dispatch_uri": RUN.DISPATCH_URI,
        "spec_uri": RUN.SPEC_URI,
        "owner_uri": RUN.OWNER_URI,
        "inputs": {
            "predecessor_analysis": {
                "path": "../014.03a-Code-attachments/analysis.json",
                "sha256": file_digest(predecessor_analysis_path),
            },
            "predecessor_negative_diagnostic": {
                "path": "../014.03a-Code-attachments/negative_result_diagnostic.json",
                "sha256": file_digest(predecessor_diagnostic_path),
            },
            "tied_analysis_source": "same-pass temporal diagnostics",
        },
        "architectural_contrast": {
            "014.03a_untied": "independent input_emb[84,64] and cand_emb[84,64] Event Parameters",
            "014.05a_tied": "one event_emb[84,64] Parameter used by exact object identity for query and candidate scoring",
            "single_conceptual_change": "Event identity tying across query and answer roles",
            "predecessor_diagnostic_status": diagnostic["status"],
        },
        "accepted_untied_outcome": {
            "reading": "uniform M4",
            "regularized": old_summary["regularized"],
            "weight_decay_zero": old_summary["weight_decay_zero"],
        },
        "tied_outcome": {
            "overall": tied_classification,
            "summary_by_arm": summary,
            "individual_readings": [
                {key: row[key] for key in ("seed", "arm", "reading", "t_fit", "t_gen", "t_global", "t_repr", "t_repr_global", "t_repr_exact")}
                for row in temporal
            ],
        },
        "classification_definitions": {
            "T0": "tied model cannot reliably fit TRAIN; capacity/optimization boundary, no grokking conclusion",
            "T1": "tied model fits TRAIN but gains no role consequence; tying alone insufficient under this benchmark/training signal",
            "T2": "ROLE_TEST but not stronger NOVEL_TEST evidence; seen-triad role-neutral consequence only",
            "T3": "ROLE_TEST plus readable representation with t_repr conservatively no later than t_gen",
            "T4": "NOVEL_TEST, t_repr_global, or exact full-relation evidence; stronger finite relation-level organization",
        },
        "claim_boundary": {
            "sufficiency_only": "a positive result tests sufficiency relative to one untied predecessor; it does not establish necessity",
            "no_causal_overclaim": "architectural contrast plus trajectory association does not by itself identify a unique causal circuit",
            "negative_boundary": "a negative result does not establish that no architecture can grok this benchmark",
        },
        "required_fences": [
            "does not establish that all grokking is algebra discovery",
            "does not establish OT superiority to transformers or LLMs",
            "does not test H_native",
            "does not establish language-scale or open-ended discovery",
            "does not require preferred coordinates neuron-by-neuron",
            "does not establish low dimensionality as necessary",
            "probe decodability alone does not prove causal use",
            "does not establish tied embeddings as the only route to a role-neutral relation",
            "does not establish physical Event/Outcome semantics",
            "does not claim tying is necessary from a positive result",
            "does not claim all grokking works this way",
            "a negative does not establish that no architecture can grok the benchmark",
        ],
    }


def analyze() -> None:
    RUN.require_execution_gates()
    contract = load_json(HERE / "tied_benchmark_contract.json")
    for uri, path in RUN.ACCEPTED_INPUTS.items():
        pinned = contract["pinned_inputs"].get(uri)
        if pinned is None or pinned["sha256"] != file_digest(path):
            raise SystemExit(
                f"BLOCKED: accepted comparison input changed since contracts: {uri}"
            )
    controls = load_json(HERE / "probe_controls.json")
    calibration = load_json(HERE / "capacity_fallback_selection.json")
    if not controls["gate"]["passed"]:
        raise SystemExit("BLOCKED-PROBE-REGRESSION")
    probe_artifact = load_json(HERE / "representation_probe_trajectory.json")
    probe_index = {
        (row["seed"], row["arm"]): row["checkpoints"]
        for row in probe_artifact["runs"]
    }
    arms = {
        arm: load_json(HERE / f"trajectory_{arm}.json")
        for arm in RUN.PRIMARY_ARMS
    }
    fallback_path = HERE / "trajectory_capacity_fallback.json"
    fallback_status = "capacity decision artifact missing"
    if fallback_path.is_file():
        fallback_artifact = load_json(fallback_path)
        fallback_status = fallback_artifact["status"]
        if fallback_artifact.get("invoked") and fallback_artifact["status"] == "AGGREGATED":
            arms["capacity_fallback"] = fallback_artifact

    temporal = []
    for arm, artifact in arms.items():
        for run in artifact["runs"]:
            key = (run["seed"], arm)
            if key not in probe_index:
                raise SystemExit(f"BLOCKED: missing probe trajectory for {key}")
            row = OLD.temporal_for_run(run["behavioral"], probe_index[key])
            row.update({"seed": run["seed"], "arm": arm})
            row.update(classify_run(row))
            temporal.append(row)
    temporal.sort(key=lambda row: (row["arm"], row["seed"]))

    summary = {}
    for arm in arms:
        rows = [row for row in temporal if row["arm"] == arm]
        if arm in RUN.PRIMARY_ARMS:
            count = sum(fit_by_8192(row) for row in rows)
            qualified = count >= 7
            basis = f"{count}/8 scored seeds fit by update 8192; primary requires >=7/8"
        else:
            qualified = calibration["status"] == "QUALIFIED"
            basis = "capacity-qualified from the pre-scored 4/4 TRAIN-only calibration; nonfitting scored runs remain individually capacity-unqualified"
        summary[arm] = summarize(rows, qualified, basis)

    tied_classification = overall_tied_classification(
        temporal, summary, fallback_status
    )
    component = transplant_reading()
    comparison = build_comparison(tied_classification, summary, temporal)
    comparison_written = write_json(
        HERE / "comparison_to_01403a.json", comparison
    )
    payload = {
        "module": "analyze_014_tied.py",
        "execution_identity": RUN.EXECUTION_IDENTITY,
        "dispatch_uri": RUN.DISPATCH_URI,
        "spec_uri": RUN.SPEC_URI,
        "owner_uri": RUN.OWNER_URI,
        "inputs": {
            "tied_benchmark_contract_sha256": file_digest(
                HERE / "tied_benchmark_contract.json"
            ),
            "tied_probe_contract_sha256": file_digest(
                HERE / "tied_probe_contract.json"
            ),
            "probe_controls_sha256": file_digest(HERE / "probe_controls.json"),
            "capacity_fallback_selection_sha256": file_digest(
                HERE / "capacity_fallback_selection.json"
            ),
            "comparison_to_01403a_sha256": comparison_written["sha256"],
        },
        "parameter_identity_audit": contract["architecture"][
            "parameter_identity_audit"
        ],
        "probe_gate": controls["gate"],
        "capacity_fallback": {
            "calibration_status": calibration["status"],
            "selected_candidate": calibration["selected_candidate"],
            "scored_decision_status": fallback_status,
        },
        "temporal_definitions": contract["temporal_definitions"],
        "temporal_diagnostics": temporal,
        "summary_by_arm": summary,
        "overall_tied_classification": tied_classification,
        "component_transplant_reading": component,
        "separated_claims": {
            "A_MEMORIZATION": "see t_fit and per-arm capacity qualification; individual nonfits are capacity-unqualified, never M4",
            "B_ROLE_CONSEQUENCE": "see t_gen and ROLE_TEST trajectories",
            "C_GLOBAL_EXTRAPOLATION": "see t_global and NOVEL_TEST trajectories",
            "D_REPRESENTATION_READABILITY": "see t_repr, t_repr_global, and t_repr_exact from fresh TRAIN-only probes",
            "E_LOAD_BEARING_LOCATION": "see component_transplant_reading after pass 2; null on pass 1",
        },
        "fences": comparison["required_fences"],
    }
    written = write_json(HERE / "analysis.json", payload)
    print(
        f"analysis.json {written['sha256'][:16]}...; comparison "
        f"{comparison_written['sha256'][:16]}...; "
        f"overall={tied_classification['classification']}"
    )
    for arm, row in summary.items():
        print(
            f"{arm}: fit={row['t_fit_reached']}/{row['seeds']} "
            f"gen={row['t_gen_reached']} global={row['t_global_reached']} "
            f"repr={row['t_repr_reached']} readings={row['reading_histogram']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze tied 014.05a trajectories; run before and after transplant"
    )
    parser.parse_args(argv)
    analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
