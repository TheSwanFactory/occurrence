"""Execute the Issue 009 representation-ablation sweep (Task 10).

Runs the frozen harness over every arm, fold and seed and writes one flat
artifact of per-run rows plus light aggregates. It performs **no** analysis: the
paired A/B/C/D effects and the disposition are Task 11's work, deliberately kept
out of here so the numbers and their interpretation are separable.

Two declared configurations, and only two
-----------------------------------------
``primary``
    the default protocol, ``mask_held_out_candidates=False``. This is the one
    common architecture the stop rule allows.

``repair``
    ``mask_held_out_candidates=True``, the single motivated repair the stop rule
    permits, run over the **whole primary LOHO family** rather than a spot check
    so its verdict rests on the full 14 folds x 8 seeds.

The repair addresses a demonstrated representation-interface defect that
``harness.py`` diagnosed and reported without fixing: on a held-out-habitat fold
the correct forced third is itself a held-out Event, which is scored as a
candidate during training but is never the right answer, so cross-entropy learns
to suppress exactly the slots the test set needs. Masking those candidate slots
*during training only* removes a false negative signal; it does not add
information about the relation. It changes no fold membership, no label, no
ground truth, introduces no per-Event output identity, and leaks no SFP rule.

A coordinator spot check over 3 LOHO folds x 2 seeds found the repair did **not**
lift forced-third accuracy off the floor. This sweep evaluates it properly so
that finding is recorded on full evidence rather than a probe. No further repair
is attempted: the stop rule allows exactly one, and hunting for a second after
seeing results would be tuning.

Fences
------
::

    Labels come from the certified native FIPS relation; this driver only consumes them.
    Correctness is exact certified Event identity, never float proximity.
    Admission is reported as balanced accuracy; raw accuracy is inadmissible at
    92.9% CROSS_HABITAT.
    Arm E is a diagnostic memorization ceiling (token_dim == catalogue size), never
    a representation arm and never a substitute for Arm C.
    RANDOM_CONTROL is a regression control only and was found non-dispositive by the
    Task 8 leakage search; it is reported, never used for the disposition.
    One architecture plus at most one motivated repair. No hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import arms as arms_module
import folds as folds_module
import harness
import task as task_module

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "sweep.json"

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"

FENCES = (
    (
        "Labels come from the certified native FIPS relation; this driver only "
        "consumes them."
    ),
    "Correctness is exact certified Event identity, never float proximity.",
    (
        "Admission is reported as balanced accuracy; raw accuracy is inadmissible "
        "at 92.9% CROSS_HABITAT."
    ),
    (
        "Arm E is a diagnostic memorization ceiling (token_dim == catalogue size), "
        "never a representation arm and never a substitute for Arm C."
    ),
    (
        "RANDOM_CONTROL is a regression control only and was found non-dispositive "
        "by the Task 8 leakage search; it is reported, never used for the "
        "disposition."
    ),
    "One architecture plus at most one motivated repair. No hyperparameter tuning.",
)

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

#: Wall-clock and FLOP-timing keys are excluded from the replay comparison.
NON_REPLAYABLE_KEYS = ("wall_sec", "sweep_wall_sec", "started_at", "host")


def digest(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def strip_non_replayable(node: object) -> object:
    """Drop wall-clock-like keys recursively so replay can be compared."""

    if isinstance(node, dict):
        return {
            key: strip_non_replayable(value)
            for key, value in node.items()
            if key not in NON_REPLAYABLE_KEYS
        }
    if isinstance(node, list):
        return [strip_non_replayable(value) for value in node]
    return node


def configurations() -> dict[str, harness.HarnessConfig]:
    """The two declared configurations. Nothing else is run."""

    base = harness.HarnessConfig(seeds=SEEDS)
    return {
        "primary": base,
        "repair": replace(
            base,
            families=("LOHO",),
            train=replace(base.train, mask_held_out_candidates=True),
        ),
    }


def run_configuration(
    name: str,
    config: harness.HarnessConfig,
    dataset: task_module.Dataset,
    all_arms: tuple,
    all_folds: tuple,
) -> tuple[list[harness.RunResult], float]:
    """Every (arm, fold, seed) run for one configuration."""

    chosen = [f for f in all_folds if f.family in config.families]
    results: list[harness.RunResult] = []
    started = time.perf_counter()
    total = len(all_arms) * len(chosen) * len(config.seeds)
    done = 0
    for arm in all_arms:
        for fold in chosen:
            for seed in config.seeds:
                results.append(
                    harness.train_one(arm, fold, seed, config, dataset=dataset)
                )
                done += 1
                if done % 40 == 0 or done == total:
                    print(
                        f"[{name}] {done}/{total} "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
    return results, time.perf_counter() - started


def group_aggregates(results: list[harness.RunResult]) -> dict:
    """Aggregate by (arm, family), reusing the harness's own aggregator."""

    keys = sorted({(r.arm, r.fold_family) for r in results})
    return {
        f"{arm}|{family}": harness.aggregate(
            [r for r in results if r.arm == arm and r.fold_family == family]
        )
        for arm, family in keys
    }


def sweep() -> dict:
    dataset = task_module.build_dataset()
    all_arms = arms_module.all_arms(dataset)
    all_folds = folds_module.all_folds(dataset)
    configs = configurations()

    blocks: dict[str, dict] = {}
    wall: dict[str, float] = {}
    for name, config in configs.items():
        results, elapsed = run_configuration(
            name, config, dataset, all_arms, all_folds
        )
        wall[name] = elapsed
        rows = [r.as_dict() for r in results]
        blocks[name] = {
            "config": config.as_dict(),
            "n_runs": len(results),
            "families": list(config.families),
            "seeds": list(config.seeds),
            "runs": rows,
            "aggregates": group_aggregates(results),
            "runs_digest": digest(strip_non_replayable(rows)),
        }

    payload = {
        "issue": "009 - SFP consequence representation versus native FIPS realization",
        "task": "Task 10 - execute the sweep",
        "module": "experiments/sfp_representation/run_sweep.py",
        "scope": (
            "execution only: per-run rows plus light aggregates. Paired A/B/C/D "
            "effects and the disposition are Task 11 and are deliberately absent."
        ),
        "fences": list(FENCES),
        "configurations_declared": {
            "primary": (
                "the default protocol, mask_held_out_candidates=False; the one "
                "common architecture the stop rule allows"
            ),
            "repair": (
                "mask_held_out_candidates=True, the single motivated repair the "
                "stop rule permits, evaluated over the whole primary LOHO family"
            ),
            "no_third_configuration": (
                "no further repair is attempted; the stop rule allows exactly one "
                "and searching for a second after seeing results would be tuning"
            ),
        },
        "repair_rationale": (
            "harness.py diagnosed and reported without fixing: on a held-out-habitat "
            "fold the correct forced third is itself a held-out Event, scored as a "
            "candidate during training but never the right answer, so cross-entropy "
            "suppresses exactly the slots the test set needs. Masking those candidate "
            "slots during training only removes a false negative signal rather than "
            "adding information about the relation. It alters no fold membership, no "
            "label and no ground truth, adds no per-Event output identity, and leaks "
            "no SFP rule."
        ),
        "blocks": blocks,
        "arm_roles": {
            arm.name: (
                "diagnostic_memorization_ceiling"
                if arm.name == "E_opaque"
                else "science"
            )
            for arm in all_arms
        },
        "upstream_digests": {
            "catalogue": dataset.catalogue.sha256(),
            "dataset": dataset.sha256(),
            "arms": {arm.name: arm.sha256() for arm in all_arms},
            "folds": [[f.family, f.name, f.sha256()] for f in all_folds],
        },
        "provenance": {
            "base_commit": BASE_COMMIT,
            "topographo_version": importlib.metadata.version("topographo"),
            "host": platform.node(),
            "deps": {
                "torch": harness.torch.__version__,
                "numpy": harness.np.__version__,
            },
        },
        "sweep_wall_sec": {name: round(value, 2) for name, value in wall.items()},
    }
    payload["digests"] = {
        "blocks": {name: block["runs_digest"] for name, block in blocks.items()},
        "manifest": digest(
            strip_non_replayable({k: v for k, v in payload.items() if k != "digests"})
        ),
    }
    return payload


def _report(payload: dict) -> None:
    for name, block in sorted(payload["blocks"].items()):
        print(
            f"\n=== {name}: {block['n_runs']} runs over families "
            f"{block['families']} ===",
            flush=True,
        )
        print(
            f"{'arm|family':<26} {'joint':>8} {'forced3':>8} {'bal_adm':>8} "
            f"{'shd_acc':>8} {'gap':>8}",
            flush=True,
        )
        for key, agg in sorted(block["aggregates"].items()):
            def mean_of(metric: str) -> float:
                node = agg.get(metric)
                if isinstance(node, dict) and "mean" in node:
                    return float(node["mean"])
                return float("nan")

            print(
                f"{key:<26} {mean_of('joint_exact_success'):>8.4f} "
                f"{mean_of('positive_forced_third_exact_accuracy'):>8.4f} "
                f"{mean_of('admission_balanced_accuracy'):>8.4f} "
                f"{mean_of('same_habitat_disjoint_nonadmission_accuracy'):>8.4f} "
                f"{mean_of('generalization_gap'):>8.4f}",
                flush=True,
            )
    print(f"\nmanifest sha256: {payload['digests']['manifest']}", flush=True)
    print(f"wall: {payload['sweep_wall_sec']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = sweep()
    text = render(payload)
    _report(payload)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        stored = json.loads(OUTPUT.read_text())
        if strip_non_replayable(stored) != strip_non_replayable(payload):
            raise SystemExit(
                "FAIL: re-derived sweep differs from "
                f"{OUTPUT.name} outside the non-replayable keys"
            )
        print(f"PASS: replay matches {OUTPUT.name} (wall-clock excluded)", flush=True)
        return

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)


if __name__ == "__main__":
    main()
