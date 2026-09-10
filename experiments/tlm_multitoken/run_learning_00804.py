#!/usr/bin/env python3
"""`008.04` Ladder A: learned strict composition at ``E^3 R`` with Occ+Cyc+Sand.

Answers one question, narrowly:

> On the exact strict-OT program space for ``E^3 R`` with all three certified
> constructors, can a learned program-selection policy generalize on held-out
> compositions better than every relevant deterministic availability /
> fixed-execution baseline, under exact native endpoint scoring?

What runs, in order:

1. the `008.03` section 2.3 program-space viability gate, against the `008.02`
   pins, **before** any training;
2. every deterministic baseline on every split, both halves;
3. Ladder A arm 1 — the flat policy, one principled architecture pass;
4. Ladder A arm 2 — the structural policy, the single `008.04` section 10 repair;
5. Ladder C — the ambient generalized-`Mul` control, fenced, rescored natively;
6. the `reporting.py` interface-layering and baseline contract;
7. a threshold-driven disposition.

Ladder D (recovered denotations) is **not** run here and must not be run unless
this result is positive; that is `008.04` section 5.D.

Usage::

    PYTHONPATH=. python experiments/tlm_multitoken/run_learning_00804.py \
        --pool-size 40000 --out experiments/tlm_multitoken/00804_artifacts

Fence: strict constructors on the exact rational path; frozen true denotations;
learning restricted to program selection; no result decoder; no physical Event
supply claim; no language-model claim; ambient `Mul` is a control, never native.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.tlm_multitoken import ambient, policy, reporting, task
from experiments.tlm_multitoken.reporting import (
    REQUIRED_BASELINES,
    REQUIRED_DIAGNOSTICS,
    ArmReport,
    Hypothesis,
    InterfaceLayer,
)

#: `008.04` section 9 materiality bar. Same 0.10 absolute gap the `008.02`
#: headroom threshold and the `017.17` verdict rule already use.
MATERIAL_MARGIN = 0.10

#: The split the disposition is read off. The others are diagnostics.
PRIMARY_SPLIT = "motif"

#: Splits carrying the ambient control. The primary compositional split and the
#: i.i.d. split are enough to see whether discarding the strict domain helps.
AMBIENT_SPLITS = ("motif", "random")

FENCE = (
    "strict Occ/Cyc/Sand on the exact rational path; true/frozen denotations; "
    "learning restricted to strict program selection; no learned result decoder; "
    "ambient generalized Mul is a fenced control and is never OT-native; no "
    "physical Event supply, no physical program-tree selection, no modular "
    "grokking, no language-model claim, no generic computational superiority"
)


def sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _per_pattern_success(
    pool: task.Pool, split: task.Split, runs: list[policy.PolicyRun]
) -> dict:
    """Held-out exact-native success broken out by withheld sign-bit pattern.

    The seed-to-seed spread on a held-out-motif split is not noise around a
    single number: a policy can recover one withheld motif and miss the other,
    which shows up here and nowhere in the aggregate.
    """
    out: dict[str, dict] = {}
    patterns = sorted({pool.records[i].sign_pattern for i in split.test})
    for pattern in patterns:
        indices = [i for i in split.test if pool.records[i].sign_pattern == pattern]
        per_seed = []
        for run in runs:
            hits = sum(
                1
                for i in indices
                if pool.records[i].matches(
                    run.test_picks_by_input.get(pool.records[i].event_indices)
                )
            )
            per_seed.append(hits / (len(indices) or 1))
        target = pool.program_names[pool.records[indices[0]].target_program]
        out[f"{pattern:03b}"] = {
            "target_program": target,
            "n_test": len(indices),
            "exact_native_success": policy.mean_std(per_seed),
        }
    return out


def _summarize_runs(runs: list[policy.PolicyRun]) -> dict:
    """Seed-aggregate one (architecture, split) cell."""
    return {
        "architecture": runs[0].architecture,
        "split": runs[0].split,
        "param_count": runs[0].param_count,
        "n_train": runs[0].train_score.n,
        "n_test": runs[0].test_score.n,
        "train_exact_native_success": policy.mean_std(
            [r.train_score.exact_native_success for r in runs]
        ),
        "test_exact_native_success": policy.mean_std(
            [r.test_score.exact_native_success for r in runs]
        ),
        "test_tree_selection_accuracy": policy.mean_std(
            [r.test_score.tree_selection_accuracy for r in runs]
        ),
        "test_endpoint_accuracy_given_selected_defined": policy.mean_std(
            [r.test_score.endpoint_accuracy_given_selected_defined for r in runs]
        ),
        "test_endpoint_match_via_different_tree": policy.mean_std(
            [r.test_score.endpoint_match_via_different_tree for r in runs]
        ),
        "generalization_gap": policy.mean_std([r.generalization_gap for r in runs]),
        "test_selected_defined_share_masked": policy.mean_std(
            [r.test_score.selected_defined_share for r in runs]
        ),
        "test_legal_share_unmasked_argmax": policy.mean_std(
            [r.unmasked_legal_share_test for r in runs]
        ),
        "test_exact_native_success_unmasked_argmax": policy.mean_std(
            [r.unmasked_success_test for r in runs]
        ),
        "wall_sec": policy.mean_std([r.wall_sec for r in runs]),
        "seeds": [r.seed for r in runs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=40000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "00804_artifacts",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    config = task.TaskConfig(pool_size=args.pool_size)
    train_config = policy.TrainConfig(steps=args.steps, seeds=tuple(args.seeds))

    # --- 1. materialize and gate, before any training ----------------------
    print("materializing the exact pool ...", flush=True)
    pool = task.materialize_pool(config)
    gate = task.viability_gate(pool)
    print(json.dumps(gate.as_dict()["conformance_against_008_02_pins"], indent=2))
    if not gate.passed:
        print("VIABILITY GATE FAILED — refusing to train", flush=True)
        (args.out / "viability_gate_failure.json").write_text(
            json.dumps(gate.as_dict(), indent=2) + "\n"
        )
        return 2

    splits = task.build_splits(pool)
    split_records = {
        split.name: {
            "kind": split.kind,
            "rationale": split.rationale,
            "n_train": len(split.train),
            "n_test": len(split.test),
            "digest_sha256": task.split_digest(pool, split),
            "legal_program_counts_train": task.legal_program_counts(pool, split.train),
            "legal_program_counts_test": task.legal_program_counts(pool, split.test),
            **split.metadata,
        }
        for split in splits
    }

    # --- 2. deterministic baselines ---------------------------------------
    print("scoring deterministic baselines ...", flush=True)
    baselines: dict[str, list[task.BaselineScore]] = {
        split.name: task.score_baselines(pool, split) for split in splits
    }
    bars = {
        name: task.best_deterministic(scores)
        for name, scores in baselines.items()
    }

    # --- 3/4. Ladder A -----------------------------------------------------
    phi = policy.build_phi(pool)
    runs: list[policy.PolicyRun] = []
    cells: dict[str, dict] = {}
    for architecture in ("flat", "structural"):
        for split in splits:
            cell = [
                policy.train_and_score(
                    pool,
                    split,
                    architecture=architecture,
                    seed=seed,
                    config=train_config,
                    phi=phi,
                )
                for seed in train_config.seeds
            ]
            runs.extend(cell)
            summary = _summarize_runs(cell)
            if split.kind == "compositional_heldout_motif":
                summary["per_heldout_pattern"] = _per_pattern_success(
                    pool, split, cell
                )
            cells[f"{architecture}|{split.name}"] = summary
            print(
                f"  {architecture:11s} {split.name:13s} "
                f"train={summary['train_exact_native_success']['mean']:.4f} "
                f"test={summary['test_exact_native_success']['mean']:.4f} "
                f"+-{summary['test_exact_native_success']['std']:.4f} "
                f"bar={bars[split.name].exact_native_success:.4f} "
                f"({bars[split.name].name})",
                flush=True,
            )

    # --- 5. Ladder C: ambient generalized-Mul control ----------------------
    print("ambient generalized-Mul control ...", flush=True)
    amb = ambient.materialize_ambient(pool)
    ambient_pool = task.Pool(
        config=pool.config,
        programs=pool.programs,
        program_names=pool.program_names,
        records=amb.records,
        features=pool.features,
        n_endpoint_classes=amb.n_endpoint_classes,
        evaluator_calls=amb.evaluator_calls,
    )
    ambient_scored = sorted(ambient_pool.scored)
    ambient_cells: dict[str, dict] = {}
    ambient_runs: list[policy.PolicyRun] = []
    for architecture in ("flat", "structural"):
        for split in splits:
            if split.name not in AMBIENT_SPLITS:
                continue
            # Both semantics must be scored on the same inputs, so restrict to the
            # records whose target is defined ambiently as well as natively.
            restricted = task.restrict_split(split, ambient_scored, suffix="")
            cell = [
                policy.train_and_score(
                    ambient_pool,
                    restricted,
                    architecture=architecture,
                    seed=seed,
                    config=train_config,
                    phi=phi,
                    rescore_pool=pool,
                )
                for seed in train_config.seeds
            ]
            ambient_runs.extend(cell)
            key = f"{architecture}|{split.name}"
            summary = _summarize_runs(cell)
            summary["dropped_test_records"] = restricted.metadata["dropped_test"]
            summary["native_rescoring"] = {
                "train_exact_native_success": policy.mean_std(
                    [r.rescore_train_score.exact_native_success for r in cell]
                ),
                "test_exact_native_success": policy.mean_std(
                    [r.rescore_test_score.exact_native_success for r in cell]
                ),
                "test_tree_selection_accuracy": policy.mean_std(
                    [r.rescore_test_score.tree_selection_accuracy for r in cell]
                ),
                "test_share_selected_program_undefined_natively": policy.mean_std(
                    [1.0 - r.rescore_test_score.selected_defined_share for r in cell]
                ),
                "note": (
                    "policy trained wholly inside the ambient program space; the "
                    "program it selects is then executed under exact strict "
                    "native semantics, and an only-ambiently-defined program "
                    "scores as a miss"
                ),
            }
            ambient_cells[key] = summary
            print(
                f"  ambient {architecture:11s} {split.name:13s} "
                f"ambient_test={summary['test_exact_native_success']['mean']:.4f} "
                f"native_rescored_test="
                f"{summary['native_rescoring']['test_exact_native_success']['mean']:.4f}",
                flush=True,
            )

    # --- 6. reporting contract --------------------------------------------
    arms = [
        ArmReport(
            arm="A1_flat_learned_policy",
            signature="E^3 R",
            layers=frozenset({InterfaceLayer.SELECTED_TREE}),
            claims=frozenset({Hypothesis.H_REL}),
            baselines=REQUIRED_BASELINES,
            target_is_availability_rule=False,
        ),
        ArmReport(
            arm="A2_structural_learned_policy",
            signature="E^3 R",
            layers=frozenset({InterfaceLayer.SELECTED_TREE}),
            claims=frozenset({Hypothesis.H_REL}),
            baselines=REQUIRED_BASELINES,
            target_is_availability_rule=False,
        ),
        ArmReport(
            arm="B_deterministic_baselines",
            signature="E^3 R",
            layers=frozenset({InterfaceLayer.SUPPLIED_TREE}),
            claims=frozenset({Hypothesis.H_WEAK}),
            baselines=REQUIRED_BASELINES,
        ),
        ArmReport(
            arm="C_ambient_generalized_mul_control",
            signature="E^3 R",
            layers=frozenset({InterfaceLayer.SELECTED_TREE}),
            claims=frozenset(),
            baselines=REQUIRED_BASELINES,
        ),
    ]
    diagnostics = {
        "legal_program_count_by_signature": {
            split.name: split_records[split.name]["legal_program_counts_test"]
            for split in splits
        },
        "endpoint_disagreement_rate": gate.endpoint_disagreement_given_two,
        "deterministic_baseline_accuracy": {
            name: {s.name: s.exact_native_success for s in scores if s.half == "test"}
            for name, scores in baselines.items()
        },
        "learned_train_exact_success": {
            key: cell["train_exact_native_success"]["mean"] for key, cell in cells.items()
        },
        "learned_test_exact_success": {
            key: cell["test_exact_native_success"]["mean"] for key, cell in cells.items()
        },
        "generalization_gap": {
            key: cell["generalization_gap"]["mean"] for key, cell in cells.items()
        },
        "tree_selection_frequencies": {
            f"{run.architecture}|{run.split}|seed{run.seed}": run.picks_test
            for run in runs
        },
        "per_program_confusion": {
            f"{run.architecture}|{run.split}|seed{run.seed}": run.test_score.per_program_regret
            for run in runs
        },
        "success_by_n_legal_alternatives": {
            f"{run.architecture}|{run.split}|seed{run.seed}": run.test_score.success_by_n_legal
            for run in runs
        },
        "undefinedness_handling": {
            "target_defined_share": gate.target_defined_share,
            "masked_argmax_selected_defined_share": {
                key: cell["test_selected_defined_share_masked"]["mean"]
                for key, cell in cells.items()
            },
            "unmasked_argmax_legal_share": {
                key: cell["test_legal_share_unmasked_argmax"]["mean"]
                for key, cell in cells.items()
            },
        },
        "parameter_counts": {key: cell["param_count"] for key, cell in cells.items()},
        "seeds": list(train_config.seeds),
    }
    missing = [key for key in REQUIRED_DIAGNOSTICS if key not in diagnostics]
    reporting.validate_suite(arms, diagnostics)

    # --- 7. disposition ---------------------------------------------------
    disposition = _disposition(cells, bars, train_config)

    report = {
        "task": "008.04",
        "result": "008.05",
        "question": (
            "does a learned policy over strict OT-admissible program trees beat "
            "every relevant deterministic baseline on held-out compositions at "
            "E^3 R with Occ+Cyc+Sand, under exact native endpoint scoring?"
        ),
        "ladder": {
            "A": "true/frozen denotations + learned strict tree policy (run)",
            "B": "deterministic / non-learning baselines (run)",
            "C": "ambient generalized-Mul control, fenced non-native (run)",
            "D": "recovered denotations (NOT run; gated on A being positive)",
        },
        "implementation": {
            "arithmetic": "topographo.ssd.exact rational sedenions",
            "constructors": "experiments.tlm_multitoken.native (Occ, Cyc, Sand)",
            "programs": "experiments.tlm_multitoken.programs planar term calculus",
            "evaluator": "exact; projective endpoint identity via projective.canonicalize",
            "surrogate": train_config.as_dict()["surrogate"],
            "device": "cpu",
            "host": platform.node(),
            "deps": {"torch": torch.__version__, "numpy": np.__version__},
        },
        "task_definition": {
            **config.as_dict(),
            "target_relation": (
                "008.02 latent bracketing grammar over Event sign bits; "
                "bits = 4*b(s1)+2*b(s2)+b(s3); index = bits mod 20; the reachable "
                "pool is the eight Cyc-free bracketings and bit d governs the "
                "retained head at depth d, so the sign of the LAST Event governs "
                "the OUTERMOST head"
            ),
            "target_is_availability_rule": False,
            "target_inspects_model_scores": False,
            "target_uses_a_decoder": False,
            "n_programs": len(pool.programs),
            "program_names": list(pool.program_names),
            "cyc_free_programs": list(task.cyc_free_program_names(pool)),
            "n_endpoint_classes_in_pool": pool.n_endpoint_classes,
        },
        "viability_gate": gate.as_dict(),
        "splits": split_records,
        "baselines": {
            name: [s.as_dict() for s in scores] for name, scores in baselines.items()
        },
        "best_deployable_deterministic_baseline_test": {
            name: {
                "name": score.name,
                "exact_native_success": score.exact_native_success,
            }
            for name, score in bars.items()
        },
        "learned": cells,
        "ambient_control": {
            "pool": amb.as_dict(),
            "cells": ambient_cells,
            "fence": ambient.AMBIENT_FENCE,
        },
        "reporting_contract": {
            "arms": [
                {
                    "arm": arm.arm,
                    "layers": sorted(layer.value for layer in arm.layers),
                    "claims": sorted(claim.value for claim in arm.claims),
                    "valid": arm.valid,
                }
                for arm in arms
            ],
            "missing_required_diagnostics": missing,
            "validated": True,
        },
        "diagnostics": diagnostics,
        "seed_results": [run.as_dict() for run in runs],
        "ambient_seed_results": [run.as_dict() for run in ambient_runs],
        "disposition": disposition,
        "stop_rule": {
            "architecture_passes": 1,
            "repairs": 1,
            "repair": "structural program-feature head",
            "repair_motivation": (
                "the flat head has no output column that can win for a program "
                "whose index never appears as a training target, so its held-out "
                "score on a held-out-motif split is bounded near zero by "
                "construction rather than by learning capacity"
            ),
            "further_tuning": "none; returning the result per 008.04 section 10",
        },
        "fence": FENCE,
        "wall_sec_total": time.time() - started,
    }
    report["manifest_sha256"] = sha256_json(
        {
            "task_definition": report["task_definition"],
            "splits": {k: v["digest_sha256"] for k, v in split_records.items()},
            "viability_gate": gate.as_dict(),
        }
    )

    tiny = {
        "manifest_sha256": report["manifest_sha256"],
        "primary_split": PRIMARY_SPLIT,
        "availability_ceiling_pool": gate.availability_ceiling,
        "learned_headroom_pool": gate.learned_headroom,
        "best_deterministic_test": {
            name: [score.name, round(score.exact_native_success, 4)]
            for name, score in bars.items()
        },
        "learned_test": {
            key: round(cell["test_exact_native_success"]["mean"], 4)
            for key, cell in cells.items()
        },
        "margins": disposition["margins"],
        "verdict": disposition["verdict"],
        "positive": disposition["positive"],
        "negative_classification": disposition["negative_classification"],
        "ladder_D_warranted": disposition["ladder_D_warranted"],
    }

    (args.out / "learning_00804_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    (args.out / "learning_00804_report_tiny.json").write_text(
        json.dumps(tiny, indent=2) + "\n"
    )
    (args.out / "split_metadata.json").write_text(
        json.dumps({"config": config.as_dict(), "splits": split_records}, indent=2) + "\n"
    )
    print(json.dumps(tiny, indent=2))
    print(f"wrote {args.out}")
    print(f"elapsed {time.time() - started:.1f}s")
    return 0


def _disposition(
    cells: dict[str, dict],
    bars: dict[str, task.BaselineScore],
    train_config: policy.TrainConfig,
) -> dict:
    """Apply the `008.04` section 9 success / falsification rule mechanically.

    Evidence for learned strict native composition requires the learned held-out
    exact-native success to be *materially* above **every relevant deployable
    deterministic baseline**, and the gain must survive more than one seed. A
    learned policy that only ties the strongest deterministic rule is a negative
    result for the intended claim even when its absolute accuracy is 1.000, and
    that is exactly what happens on the splits a lookup table can solve.
    """
    margins: dict[str, dict] = {}
    for key, cell in cells.items():
        split = cell["split"]
        bar = bars[split]
        per_seed = cell["test_exact_native_success"]["values"]
        beating = [v for v in per_seed if v - bar.exact_native_success >= MATERIAL_MARGIN]
        margins[key] = {
            "split": split,
            "learned_test_mean": cell["test_exact_native_success"]["mean"],
            "learned_test_std": cell["test_exact_native_success"]["std"],
            "learned_test_per_seed": per_seed,
            "best_deterministic": bar.name,
            "best_deterministic_test": bar.exact_native_success,
            "margin": cell["test_exact_native_success"]["mean"]
            - bar.exact_native_success,
            "material_margin_threshold": MATERIAL_MARGIN,
            "seeds_beating_materially": len(beating),
            "material_on_more_than_one_seed": len(beating) > 1,
        }

    primary = {
        key: value for key, value in margins.items() if value["split"] == PRIMARY_SPLIT
    }
    winners = {
        key: value
        for key, value in primary.items()
        if value["margin"] >= MATERIAL_MARGIN and value["material_on_more_than_one_seed"]
    }
    positive = bool(winners)

    if positive:
        best_key = max(winners, key=lambda k: winners[k]["margin"])
        verdict = (
            f"POSITIVE on the primary {PRIMARY_SPLIT} split: "
            f"{best_key} reaches "
            f"{winners[best_key]['learned_test_mean']:.4f} held-out exact-native "
            f"success against {winners[best_key]['best_deterministic']} at "
            f"{winners[best_key]['best_deterministic_test']:.4f}, a margin of "
            f"{winners[best_key]['margin']:+.4f} on "
            f"{winners[best_key]['seeds_beating_materially']} of "
            f"{len(train_config.seeds)} seeds"
        )
        classification = None
    else:
        best_key = max(primary, key=lambda k: primary[k]["margin"]) if primary else ""
        verdict = (
            f"NEGATIVE on the primary {PRIMARY_SPLIT} split: best margin "
            f"{primary[best_key]['margin']:+.4f} against "
            f"{primary[best_key]['best_deterministic']}"
            if primary
            else "NEGATIVE: no primary-split cell scored"
        )
        classification = "policy learning"

    saturated = sorted(
        {
            value["split"]
            for value in margins.values()
            if value["best_deterministic_test"] >= 1.0 - 1e-12
        }
    )
    return {
        "material_margin": MATERIAL_MARGIN,
        "primary_split": PRIMARY_SPLIT,
        "margins": margins,
        "positive": positive,
        "verdict": verdict,
        "negative_classification": classification,
        "splits_saturated_by_a_deterministic_baseline": saturated,
        "saturated_split_note": (
            "on these splits a deterministic train-fitted lookup over the Event "
            "sign pattern already scores 1.000, so no learned policy can "
            "materially exceed it; per 008.04 section 9 those splits are "
            "negative for the intended claim and are reported as controls"
        ),
        "ladder_D_warranted": positive,
        "ladder_D_note": (
            "008.04 section 5.D gates recovered denotations on Ladder A being "
            "materially positive against the deterministic baselines"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
