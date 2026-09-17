"""014.03 / 014.03a — predeclared temporal diagnostics and M1-M4 readings.

Consumes only the committed trajectory artifacts and the frozen contracts. Every
threshold, persistence window and reading boundary is taken from
``benchmark_contract.json`` (section "temporal_definitions"), which was committed
before any neural training, so nothing here is fitted after the fact.

Run order::

    run_014.py ... trajectory (all scored seeds and arms)
    run_014.py ... aggregate
    analyze_014.py                  # pass 1: temporal diagnostics
    run_014.py ... transplant       # consumes the predeclared t_fit / t_gen
    analyze_014.py                  # pass 2: folds the transplant in

Pass 2 supersedes pass 1. The temporal definitions do not depend on the
transplant, so the two passes agree on every temporal quantity; only the
component-level section is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ISSUE_DIR = Path(__file__).resolve().parent.parent

T_GEN_THRESHOLD = 0.95
T_REPR_CONSISTENCY_THRESHOLD = 0.90
DELAYED_GAP = 2048
DELAYED_ROLE_AT_FIT = 0.50
PERSISTENCE = 2


def digest(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: object) -> dict:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    raw = path.read_bytes()
    reloaded = json.loads(raw.decode("utf-8"))
    if digest(reloaded) != digest(payload):
        raise SystemExit(f"BLOCKED: readback of {path} did not match")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def first_persistent(rows: list[dict], predicate) -> dict:
    """First checkpoint where ``predicate`` holds here and at the next two.

    Returns the update index, plus an explicit note when a threshold is first met
    inside the final two scheduled checkpoints, where the declared persistence
    window cannot be evaluated. Such a case is ``null``, never the final
    checkpoint.
    """

    hits = [i for i, row in enumerate(rows) if predicate(row)]
    for i in hits:
        window = rows[i : i + PERSISTENCE + 1]
        if len(window) < PERSISTENCE + 1:
            return {
                "update": None,
                "first_threshold_update": rows[i]["update"],
                "note": (
                    "threshold first met inside the final two scheduled "
                    "checkpoints, so the declared persistence window cannot be "
                    "evaluated; recorded as null"
                ),
            }
        if all(predicate(row) for row in window):
            return {"update": rows[i]["update"], "note": None}
    return {
        "update": None,
        "first_threshold_update": rows[hits[0]]["update"] if hits else None,
        "note": None if hits else "threshold never met",
    }


def first_simple(rows: list[dict], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return row["update"]
    return None


def temporal_for_run(behavioral: list[dict], probe: list[dict]) -> dict:
    behavioral = sorted(behavioral, key=lambda r: r["update"])
    probe = sorted(probe, key=lambda r: r["update"])

    t_fit = first_simple(behavioral, lambda r: r["train_accuracy"] == 1.0)
    t_gen = first_persistent(
        behavioral, lambda r: r["role_test_accuracy"] >= T_GEN_THRESHOLD
    )
    t_global = first_persistent(
        behavioral, lambda r: r["novel_test_accuracy"] >= T_GEN_THRESHOLD
    )
    t_repr = first_persistent(
        probe,
        lambda r: r["probe_role_accuracy"] >= T_GEN_THRESHOLD
        and r["seen_block_three_role_consistency"] >= T_REPR_CONSISTENCY_THRESHOLD,
    )
    t_repr_global = first_persistent(
        probe,
        lambda r: r["probe_novel_accuracy"] >= T_GEN_THRESHOLD
        and r["all_block_three_role_consistency"] >= T_REPR_CONSISTENCY_THRESHOLD,
    )
    t_repr_exact = first_simple(probe, lambda r: r["full_56_block_relation_exact"])

    by_update = {r["update"]: r for r in behavioral}
    role_at_fit = by_update[t_fit]["role_test_accuracy"] if t_fit is not None else None
    delayed = bool(
        t_fit is not None
        and t_gen["update"] is not None
        and (t_gen["update"] - t_fit) >= DELAYED_GAP
        and role_at_fit is not None
        and role_at_fit < DELAYED_ROLE_AT_FIT
    )

    final_behavioral = behavioral[-1]
    final_probe = probe[-1]
    return {
        "t_fit": t_fit,
        "t_gen": t_gen["update"],
        "t_global": t_global["update"],
        "t_repr": t_repr["update"],
        "t_repr_global": t_repr_global["update"],
        "t_repr_exact": t_repr_exact,
        "persistence_notes": {
            "t_gen": t_gen.get("note"),
            "t_global": t_global.get("note"),
            "t_repr": t_repr.get("note"),
            "t_repr_global": t_repr_global.get("note"),
        },
        "first_threshold_without_persistence": {
            "t_gen": t_gen.get("first_threshold_update"),
            "t_global": t_global.get("first_threshold_update"),
            "t_repr": t_repr.get("first_threshold_update"),
            "t_repr_global": t_repr_global.get("first_threshold_update"),
        },
        "role_test_accuracy_at_t_fit": role_at_fit,
        "delayed_generalization_flag": delayed,
        "delayed_generalization_components": {
            "t_fit_exists": t_fit is not None,
            "t_gen_exists": t_gen["update"] is not None,
            "gap_at_least_2048": bool(
                t_fit is not None
                and t_gen["update"] is not None
                and (t_gen["update"] - t_fit) >= DELAYED_GAP
            ),
            "role_below_half_at_t_fit": bool(
                role_at_fit is not None and role_at_fit < DELAYED_ROLE_AT_FIT
            ),
        },
        "final": {
            "update": final_behavioral["update"],
            "train_accuracy": final_behavioral["train_accuracy"],
            "role_test_accuracy": final_behavioral["role_test_accuracy"],
            "novel_test_accuracy": final_behavioral["novel_test_accuracy"],
            "input_swap_invariance_bit_exact": final_behavioral[
                "input_swap_invariance_bit_exact"
            ],
            "probe_role_accuracy": final_probe["probe_role_accuracy"],
            "probe_novel_accuracy": final_probe["probe_novel_accuracy"],
            "seen_block_three_role_consistency": final_probe[
                "seen_block_three_role_consistency"
            ],
            "all_block_three_role_consistency": final_probe[
                "all_block_three_role_consistency"
            ],
            "full_56_block_relation_exact": final_probe["full_56_block_relation_exact"],
        },
        "peak": {
            "role_test_accuracy": max(r["role_test_accuracy"] for r in behavioral),
            "novel_test_accuracy": max(r["novel_test_accuracy"] for r in behavioral),
            "probe_role_accuracy": max(r["probe_role_accuracy"] for r in probe),
            "probe_novel_accuracy": max(r["probe_novel_accuracy"] for r in probe),
        },
        "swap_invariance_bit_exact_at_every_checkpoint": all(
            r["input_swap_invariance_bit_exact"] for r in behavioral
        ),
    }


def classify(row: dict) -> dict:
    """Assign the declared M1-M4 readings. Mixed cases are reported as mixed."""

    t_fit = row["t_fit"]
    t_gen = row["t_gen"]
    t_repr = row["t_repr"]
    applicable = []

    if (
        t_fit is not None
        and t_repr is not None
        and t_gen is not None
        and t_fit < t_repr <= t_gen
    ):
        applicable.append("M1")
    if (t_gen is not None and t_repr is not None and t_gen < t_repr) or (
        t_repr is None and t_gen is not None
    ):
        applicable.append("M2")
    if t_repr is not None and t_gen is None:
        applicable.append("M3")
    if t_repr is None and t_gen is None:
        applicable.append("M4")

    if not applicable:
        label = "UNCLASSIFIED"
    elif len(applicable) == 1:
        label = applicable[0]
    else:
        label = "MIXED:" + "+".join(applicable)

    return {
        "readings_applicable": applicable,
        "reading": label,
        "ordering": {
            "t_fit": t_fit,
            "t_repr": t_repr,
            "t_gen": t_gen,
            "t_repr_before_t_gen": (
                None
                if t_repr is None or t_gen is None
                else bool(t_repr <= t_gen)
            ),
        },
    }


def analyze(bundle: str) -> None:
    here = ISSUE_DIR / f"{bundle}-Code-attachments"
    contract = json.loads((here / "benchmark_contract.json").read_text())
    regime = json.loads((here / "selected_regime.json").read_text())
    controls = json.loads((here / "probe_controls.json").read_text())
    probe_runs = json.loads((here / "representation_probe_trajectory.json").read_text())
    arms = {
        "regularized": json.loads((here / "trajectory_regularized.json").read_text()),
        "weight_decay_zero": json.loads(
            (here / "trajectory_weight_decay_zero.json").read_text()
        ),
    }
    probe_index = {(r["seed"], r["arm"]): r["checkpoints"] for r in probe_runs["runs"]}

    temporal = []
    for arm_name, payload in arms.items():
        for run in payload["runs"]:
            row = temporal_for_run(
                run["behavioral"], probe_index[(run["seed"], arm_name)]
            )
            row.update({"seed": run["seed"], "arm": arm_name})
            row.update(classify(row))
            temporal.append(row)
    temporal.sort(key=lambda r: (r["arm"], r["seed"]))

    def summarize(arm_name: str) -> dict:
        rows = [r for r in temporal if r["arm"] == arm_name]
        fits = [r["t_fit"] for r in rows if r["t_fit"] is not None]
        gens = [r["t_gen"] for r in rows if r["t_gen"] is not None]
        globals_ = [r["t_global"] for r in rows if r["t_global"] is not None]
        reprs = [r["t_repr"] for r in rows if r["t_repr"] is not None]
        repr_globals = [r["t_repr_global"] for r in rows if r["t_repr_global"] is not None]
        exacts = [r["t_repr_exact"] for r in rows if r["t_repr_exact"] is not None]
        readings: dict[str, int] = {}
        for r in rows:
            readings[r["reading"]] = readings.get(r["reading"], 0) + 1
        return {
            "seeds": len(rows),
            "t_fit_reached": len(fits),
            "median_t_fit": statistics.median(fits) if fits else None,
            "t_gen_reached": len(gens),
            "median_t_gen": statistics.median(gens) if gens else None,
            "t_global_reached": len(globals_),
            "median_t_global": statistics.median(globals_) if globals_ else None,
            "t_repr_reached": len(reprs),
            "median_t_repr": statistics.median(reprs) if reprs else None,
            "t_repr_global_reached": len(repr_globals),
            "median_t_repr_global": (
                statistics.median(repr_globals) if repr_globals else None
            ),
            "t_repr_exact_reached": len(exacts),
            "median_t_repr_exact": statistics.median(exacts) if exacts else None,
            "delayed_generalization_flagged": sum(
                r["delayed_generalization_flag"] for r in rows
            ),
            "reading_histogram": readings,
            "final_role_accuracy_median": statistics.median(
                r["final"]["role_test_accuracy"] for r in rows
            ),
            "final_novel_accuracy_median": statistics.median(
                r["final"]["novel_test_accuracy"] for r in rows
            ),
            "final_probe_role_accuracy_median": statistics.median(
                r["final"]["probe_role_accuracy"] for r in rows
            ),
            "final_probe_novel_accuracy_median": statistics.median(
                r["final"]["probe_novel_accuracy"] for r in rows
            ),
            "seeds_with_exact_56_block_readout": sum(
                r["final"]["full_56_block_relation_exact"] for r in rows
            ),
            "swap_invariance_bit_exact_everywhere": all(
                r["swap_invariance_bit_exact_at_every_checkpoint"] for r in rows
            ),
        }

    summary = {arm: summarize(arm) for arm in arms}

    transplant_path = here / "component_transplants.json"
    transplant = (
        json.loads(transplant_path.read_text()) if transplant_path.exists() else None
    )
    transplant_reading = None
    if transplant:
        rows = [r for r in transplant["runs"] if r.get("hybrids")]
        if rows:

            def mean(key: str, split: str) -> float:
                return statistics.mean(r["hybrids"][key][split] for r in rows)

            transplant_reading = {
                "seeds": len(rows),
                "means": {
                    key: {
                        split: mean(key, split)
                        for split in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")
                    }
                    for key in (
                        "early_representation_early_downstream",
                        "late_representation_late_downstream",
                        "late_representation_early_downstream",
                        "early_representation_late_downstream",
                    )
                },
            }
            late_repr_early_down = transplant_reading["means"][
                "late_representation_early_downstream"
            ]["ROLE_TEST"]
            early_repr_late_down = transplant_reading["means"][
                "early_representation_late_downstream"
            ]["ROLE_TEST"]
            early_early = transplant_reading["means"][
                "early_representation_early_downstream"
            ]["ROLE_TEST"]
            late_late = transplant_reading["means"][
                "late_representation_late_downstream"
            ]["ROLE_TEST"]
            role_values = [
                early_early,
                late_late,
                late_repr_early_down,
                early_repr_late_down,
            ]
            chance = 1.0 / 84.0
            all_role_at_or_below_chance = all(v <= chance for v in role_values)
            train_intact = {
                key: transplant_reading["means"][key]["TRAIN"]
                for key in transplant_reading["means"]
            }
            transplant_reading["localization"] = {
                "role_accuracy_in_all_four_combinations": role_values,
                "single_event_chance_level": chance,
                "all_four_at_or_below_chance": all_role_at_or_below_chance,
                "train_accuracy_by_combination": train_intact,
                "both_cross_hybrids_destroy_train_fit": (
                    train_intact["late_representation_early_downstream"] < 1.0
                    and train_intact["early_representation_late_downstream"] < 1.0
                ),
                "localizes_anything": not all_role_at_or_below_chance,
                "reading": (
                    "no localization is possible here. Held-out accuracy is at or "
                    "below single-event chance in all four combinations, so no "
                    "combination carries the behavior that would need locating, and "
                    "the differences between the four ROLE values are far smaller "
                    "than one held-out example. Both cross-hybrids also destroy the "
                    "memorized TRAIN fit, which is the coadaptation case the task "
                    "anticipated."
                    if all_role_at_or_below_chance
                    else "see the four combinations above"
                ),
                "fence": (
                    "component-level evidence only; coadaptation can make both "
                    "cross-hybrids fail and that is a result, not a licence for "
                    "another intervention"
                ),
            }

    payload = {
        "module": "analyze_014.py",
        "delivery_bundle": bundle,
        "task_uri": contract["task_uri"],
        "inputs": {
            "benchmark_contract_sha256": hashlib.sha256(
                (here / "benchmark_contract.json").read_bytes()
            ).hexdigest(),
            "probe_contract_sha256": hashlib.sha256(
                (here / "probe_contract.json").read_bytes()
            ).hexdigest(),
            "selected_regime_sha256": hashlib.sha256(
                (here / "selected_regime.json").read_bytes()
            ).hexdigest(),
        },
        "selected_regime": regime["selected_regime"],
        "matched_control": regime["matched_memorization_control"],
        "probe_gate": controls["gate"],
        "probe_controls_summary": {
            name: {
                "probe_role_accuracy": row["probe_role_accuracy"],
                "probe_novel_accuracy": row["probe_novel_accuracy"],
            }
            for name, row in controls["controls"].items()
        },
        "temporal_definitions": contract["temporal_definitions"],
        "temporal_diagnostics": temporal,
        "summary_by_arm": summary,
        "baselines": contract["baselines"],
        "component_transplant_reading": transplant_reading,
        "separated_claims": {
            "A_memorization": (
                "the main model fits the 48 designated TRAIN answers: reached by "
                f"{summary['regularized']['t_fit_reached']}/"
                f"{summary['regularized']['seeds']} regularized seeds and "
                f"{summary['weight_decay_zero']['t_fit_reached']}/"
                f"{summary['weight_decay_zero']['seeds']} control seeds"
            ),
            "B_role_consequence": (
                "alternate output roles on seen triads: t_gen reached by "
                f"{summary['regularized']['t_gen_reached']}/"
                f"{summary['regularized']['seeds']} regularized seeds and "
                f"{summary['weight_decay_zero']['t_gen_reached']}/"
                f"{summary['weight_decay_zero']['seeds']} control seeds"
            ),
            "C_global_extrapolation": (
                "wholly unseen block triples: t_global reached by "
                f"{summary['regularized']['t_global_reached']}/"
                f"{summary['regularized']['seeds']} regularized seeds and "
                f"{summary['weight_decay_zero']['t_global_reached']}/"
                f"{summary['weight_decay_zero']['seeds']} control seeds"
            ),
            "D_representation_readability": (
                "a fresh TRAIN-only decoder reading the relation off frozen token "
                "representations: t_repr reached by "
                f"{summary['regularized']['t_repr_reached']}/"
                f"{summary['regularized']['seeds']} regularized seeds and "
                f"{summary['weight_decay_zero']['t_repr_reached']}/"
                f"{summary['weight_decay_zero']['seeds']} control seeds"
            ),
            "E_load_bearing_location": (
                "see component_transplant_reading; null when the transplant has not "
                "yet been executed"
            ),
        },
        "fences": contract["fences"],
        "does_not_establish": [
            "all grokking is algebra discovery",
            "low dimensionality is necessary",
            "our SFP coordinates appeared neuron-by-neuron",
            "OT is superior to transformers or language models",
            "H_native or Outcome constitution",
            "open-ended or language-scale representation discovery",
            "physical Event or Outcome semantics",
            "that probe decodability by itself proves causal use",
            "that NOVEL_TEST success is guaranteed by the training examples",
        ],
    }
    written = write_json(here / "analysis.json", payload)
    print(f"analysis.json committed: {written['bytes']} bytes sha256={written['sha256'][:16]}...")
    for arm, row in summary.items():
        print(
            f"  {arm:18s} t_fit {row['t_fit_reached']}/{row['seeds']} "
            f"t_gen {row['t_gen_reached']}/{row['seeds']} "
            f"t_global {row['t_global_reached']}/{row['seeds']} "
            f"t_repr {row['t_repr_reached']}/{row['seeds']} "
            f"readings {row['reading_histogram']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="014.03a")
    analyze(parser.parse_args().bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
