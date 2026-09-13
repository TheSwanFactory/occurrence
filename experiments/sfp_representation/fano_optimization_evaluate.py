"""011.03 locked evaluation of one committed optimization prescription.

The executable refuses to run without the exact selected-artifact hash, verifies
that the selected artifact and TRAIN-only search are present in ``HEAD``, records
that commit as the pre-evaluation commitment, and only then admits the fixed fresh
seeds.  It retrains the selected prescription for the legacy main bridge, the
fresh main block, and the fresh localized-control block while reusing every frozen
011.02 scoring and representation-recovery audit unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fano_baselines import ceiling_compatibility
from fano_heads import type_channels
from fano_optimization_core import train_prescription
from fano_optimization_manifest import (
    DEV_SEEDS,
    FRESH_SEEDS,
    object_sha256,
)
from fano_sweep import (
    build_batches,
    left_right_swap_audit,
    recovery_audit,
    score_batch,
    transport_audit,
)
from fano_task import (
    EXPECTED_CEILINGS,
    build_localized_observation,
    build_namespaces,
    build_observation,
    rd,
)
from task import build_dataset

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
ARTIFACTS = ROOT / "011_optimization_artifacts"
SELECTED = ARTIFACTS / "selected_optimization.json"
SEARCH = ARTIFACTS / "development_search.json"
OUTPUT = ARTIFACTS / "fresh_evaluation.json"
PARTIAL = ARTIFACTS / "fresh_evaluation.partial.json"
EVALUATION_WORKERS = 4

_WORKER_SELECTED: dict | None = None
_WORKER_DATA: dict[str, tuple] = {}


def render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _strip_wall(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_wall(item)
            for key, item in value.items()
            if key != "wall_sec"
        }
    if isinstance(value, list):
        return [_strip_wall(item) for item in value]
    return value


def _selected_body(selected: dict) -> dict:
    return {
        key: value
        for key, value in selected.items()
        if key != "selected_optimization_sha256"
    }


def load_selected(expected_sha256: str | None = None) -> dict:
    selected = json.loads(SELECTED.read_text())
    observed = object_sha256(_selected_body(selected))
    if observed != selected["selected_optimization_sha256"]:
        raise ValueError("selected optimization artifact hash is invalid")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"selected lock {expected_sha256} does not equal committed artifact {observed}"
        )
    if selected["development_summary"]["converged_count"] != len(DEV_SEEDS):
        raise ValueError("selected optimization did not reach the mandatory 8/8")
    if selected["fresh_evaluation_seeds"] != list(FRESH_SEEDS):
        raise ValueError("selected optimization carries the wrong fresh seed block")
    return selected


def _git_bytes(*args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def _committed_file(commit: str, path: Path) -> bytes:
    return _git_bytes("show", f"{commit}:{_relative(path)}")


def capture_commitment(selected: dict) -> dict:
    head = _git_bytes("rev-parse", "HEAD").decode().strip()
    for path in (SELECTED, SEARCH):
        if _committed_file(head, path) != path.read_bytes():
            raise ValueError(f"{path.name} is not committed byte-for-byte at HEAD")
    search = json.loads(SEARCH.read_text())
    development_seeds = sorted({row["seed"] for row in search["runs"]})
    touched_fresh = sorted(set(development_seeds) & set(FRESH_SEEDS))
    if development_seeds != list(DEV_SEEDS) or touched_fresh:
        raise ValueError("development search seed ledger is not exactly 0..7")
    if search["selected_candidate_sha256"] != selected["selected_candidate_sha256"]:
        raise ValueError("selected artifact differs from the committed search result")
    return {
        "pre_evaluation_commit": head,
        "pre_evaluation_commit_time": _git_bytes(
            "show", "-s", "--format=%cI", head
        ).decode().strip(),
        "selected_artifact_path": _relative(SELECTED),
        "selected_artifact_git_blob_sha": _git_bytes(
            "rev-parse", f"{head}:{_relative(SELECTED)}"
        ).decode().strip(),
        "selected_optimization_sha256": selected[
            "selected_optimization_sha256"
        ],
        "development_search_sha256": selected["development_search_sha256"],
        "development_seed_ledger": development_seeds,
        "fresh_seed_intersection_before_commit": touched_fresh,
        "fresh_output_existed_before_evaluation": OUTPUT.exists() or PARTIAL.exists(),
        "checks": {
            "selected_artifact_committed_byte_for_byte": True,
            "search_artifact_committed_byte_for_byte": True,
            "development_seed_ledger_is_exact": True,
            "no_fresh_seed_in_development_search": not touched_fresh,
            "fresh_output_absent": not OUTPUT.exists() and not PARTIAL.exists(),
        },
    }


def verify_recorded_commitment(commitment: dict, selected: dict) -> None:
    commit = commitment["pre_evaluation_commit"]
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY,
        check=False,
    )
    if result.returncode:
        raise ValueError("recorded pre-evaluation commit is not an ancestor of HEAD")
    if _committed_file(commit, SELECTED) != SELECTED.read_bytes():
        raise ValueError("selected artifact differs from the pre-evaluation commit")
    committed_search = json.loads(_committed_file(commit, SEARCH))
    seeds = sorted({row["seed"] for row in committed_search["runs"]})
    if seeds != list(DEV_SEEDS) or set(seeds) & set(FRESH_SEEDS):
        raise ValueError("pre-evaluation search includes an impermissible seed")
    if commitment["selected_optimization_sha256"] != selected[
        "selected_optimization_sha256"
    ]:
        raise ValueError("recorded commitment has the wrong selected hash")


def _plain_train_batches(batches: list[object]) -> list[dict]:
    return [
        {
            "label": batch.label,
            "split": batch.split,
            "adjacency": batch.adjacency,
            "per_kind": [
                {
                    "kind": kind.kind,
                    "head": kind.head,
                    "marks": kind.marks,
                    "targets": kind.targets,
                    "scored": kind.scored,
                    "sizes": list(kind.sizes),
                }
                for kind in batch.per_kind
            ],
        }
        for batch in batches
    ]


def _arm_data(arm: str) -> tuple:
    if arm in _WORKER_DATA:
        return _WORKER_DATA[arm]
    main = build_observation(build_dataset())
    observation = main if arm == "main" else build_localized_observation(main)
    namespaces = build_namespaces(observation)
    batches = build_batches(observation, namespaces)
    row = (observation, namespaces, batches)
    _WORKER_DATA[arm] = row
    return row


def _aggregate_split(model: object, batches: list[object]) -> dict:
    rows = [score_batch(model, batch) for batch in batches]
    hits: Counter = Counter()
    totals: Counter = Counter()
    decisive: Counter = Counter()
    margins: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for kind, entry in row["per_kind"].items():
            hits[kind] += entry["hits"]
            totals[kind] += entry["queries"]
            decisive[kind] += entry["decisive_rows"]
            margins[kind].append(entry["median_margin"])
    return {
        kind: {
            "hits": hits[kind],
            "queries": totals[kind],
            "accuracy": rd(hits[kind] / totals[kind]),
            "decisive_rows": decisive[kind],
            "decisive_fraction": rd(decisive[kind] / totals[kind]),
            "median_margin": rd(
                sorted(margins[kind])[len(margins[kind]) // 2]
            ),
        }
        for kind in sorted(totals)
    }


def _worker_init(selected_path: str, expected_sha256: str) -> None:
    global _WORKER_SELECTED
    del selected_path
    _WORKER_SELECTED = load_selected(expected_sha256)


def _run_cell(block: str, arm: str, seed: int) -> dict:
    if _WORKER_SELECTED is None:
        raise RuntimeError("evaluation worker has no selected prescription")
    allowed = set(DEV_SEEDS) if block == "legacy_bridge" else set(FRESH_SEEDS)
    if seed not in allowed:
        raise ValueError(f"{block} refuses seed {seed}")
    observation, namespaces, batches = _arm_data(arm)
    model, training = train_prescription(
        _plain_train_batches(batches["train"]),
        _WORKER_SELECTED["selected_prescription"],
        seed=seed,
    )
    per_split = {
        split: _aggregate_split(model, batches[split])
        for split in ("train", "validation", "test")
    }
    type_rows = type_channels(observation)
    test_namespaces = tuple(
        namespace for namespace in namespaces if namespace.split == "test"
    )
    transport_namespaces = tuple(
        namespace for namespace in namespaces if namespace.split == "transport"
    )
    recovery_rows = [
        recovery_audit(model, observation, namespace, type_rows=type_rows)
        for namespace in test_namespaces
    ]
    recovery = {
        "per_namespace": recovery_rows,
        "namespaces": len(recovery_rows),
        "mate_pairing_valid_rate": rd(
            sum(row["mate_pairing_valid"] for row in recovery_rows)
            / len(recovery_rows)
        ),
        "fano_plane_valid_rate": rd(
            sum(row["fano_plane_valid"] for row in recovery_rows)
            / len(recovery_rows)
        ),
        "certified_plane_equivalent_rate": rd(
            sum(row["certified_plane_equivalent"] for row in recovery_rows)
            / len(recovery_rows)
        ),
    }
    transport = transport_audit(
        model,
        observation,
        test_namespaces[0],
        transport_namespaces,
        type_rows=type_rows,
    )
    swap = left_right_swap_audit(
        model,
        observation,
        test_namespaces[0],
        type_rows=type_rows,
    )
    return {
        "block": block,
        "arm": arm,
        "seed": seed,
        "selected_candidate_id": _WORKER_SELECTED["selected_candidate_id"],
        "selected_candidate_sha256": _WORKER_SELECTED[
            "selected_candidate_sha256"
        ],
        "training": training,
        "per_split": per_split,
        "mate_exact_accuracy": per_split["test"]["mate"]["accuracy"],
        "fano_completion_exact_set_accuracy": per_split["test"]["completion"][
            "accuracy"
        ],
        "recovery": recovery,
        "transport": transport,
        "left_right_swap": swap,
        "wall_sec": training["wall_sec"],
    }


def _spread(values: list[float]) -> dict:
    return {
        "mean": rd(sum(values) / len(values)),
        "min": rd(min(values)),
        "max": rd(max(values)),
        "per_seed": [rd(value) for value in values],
    }


def summarize(runs: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["block"], run["arm"])].append(run)
    out = {}
    for (block, arm), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        out[f"{block}/{arm}"] = {
            "seeds": [row["seed"] for row in rows],
            "converged_train_count": sum(
                row["training"]["converged"] for row in rows
            ),
            "final_train_loss": _spread(
                [row["training"]["final_train_loss"] for row in rows]
            ),
            "mate_exact_accuracy": _spread(
                [row["mate_exact_accuracy"] for row in rows]
            ),
            "fano_completion_exact_set_accuracy": _spread(
                [row["fano_completion_exact_set_accuracy"] for row in rows]
            ),
            "mate_pairing_valid_rate": _spread(
                [row["recovery"]["mate_pairing_valid_rate"] for row in rows]
            ),
            "fano_plane_valid_rate": _spread(
                [row["recovery"]["fano_plane_valid_rate"] for row in rows]
            ),
            "certified_plane_equivalent_rate": _spread(
                [
                    row["recovery"]["certified_plane_equivalent_rate"]
                    for row in rows
                ]
            ),
            "transport_rate": _spread(
                [row["transport"]["transport_rate"] for row in rows]
            ),
            "left_right_swap_identical_rate": _spread(
                [row["left_right_swap"]["identical_rate"] for row in rows]
            ),
        }
    return out


def control_checks(runs: list[dict], summary: dict) -> dict:
    rows = [
        row
        for row in runs
        if row["block"] == "fresh" and row["arm"] == "localized"
    ]
    entry = summary["fresh/localized"]
    out = {}
    for kind, metric in (
        ("mate", "mate_exact_accuracy"),
        ("completion", "fano_completion_exact_set_accuracy"),
    ):
        out[kind] = ceiling_compatibility(
            entry[metric]["mean"],
            sum(row["per_split"]["test"][kind]["queries"] for row in rows),
            EXPECTED_CEILINGS[f"localized_{kind}"],
            blocks=8 * len(rows),
        )
    out["plane_valid_rate"] = entry["fano_plane_valid_rate"]["mean"]
    out["never_recovers_a_plane"] = all(
        row["recovery"]["fano_plane_valid_rate"] == 0.0 for row in rows
    )
    return out


def _evaluation_payload(
    selected: dict,
    commitment: dict,
    runs: list[dict],
) -> dict:
    ordered = sorted(runs, key=lambda row: (row["block"], row["arm"], row["seed"]))
    expected_cells = {
        ("legacy_bridge", "main", seed) for seed in DEV_SEEDS
    } | {
        ("fresh", arm, seed)
        for arm in ("main", "localized")
        for seed in FRESH_SEEDS
    }
    observed_cells = {(row["block"], row["arm"], row["seed"]) for row in ordered}
    if observed_cells != expected_cells or len(ordered) != len(expected_cells):
        raise ValueError("evaluation rows are not the exact 40 committed cells")
    summary = summarize(ordered)
    controls = control_checks(ordered, summary)
    body = {
        "schema": "occurrence.011.03.locked-evaluation.v1",
        "commitment": commitment,
        "selected_optimization_sha256": selected[
            "selected_optimization_sha256"
        ],
        "selected_candidate_id": selected["selected_candidate_id"],
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "legacy_bridge_seeds": list(DEV_SEEDS),
        "fresh_evaluation_seeds": list(FRESH_SEEDS),
        "cells": {
            "legacy_bridge/main": len(DEV_SEEDS),
            "fresh/main": len(FRESH_SEEDS),
            "fresh/localized": len(FRESH_SEEDS),
        },
        "workers": EVALUATION_WORKERS,
        "runs": ordered,
        "summary": summary,
        "control_vs_exact_ceiling": controls,
        "fresh_main_converged_count": summary["fresh/main"][
            "converged_train_count"
        ],
        "runs_digest": object_sha256(_strip_wall(ordered)),
        "non_replayable_keys": ["wall_sec"],
    }
    return {**body, "evaluation_sha256": object_sha256(_strip_wall(body))}


def _partial_payload(selected: dict, commitment: dict, runs: list[dict]) -> dict:
    return {
        "schema": "occurrence.011.03.locked-evaluation-partial.v1",
        "selected_optimization_sha256": selected[
            "selected_optimization_sha256"
        ],
        "commitment": commitment,
        "completed_blocks": sorted({(row["block"], row["arm"]) for row in runs}),
        "runs": sorted(runs, key=lambda row: (row["block"], row["arm"], row["seed"])),
    }


def run_evaluation(expected_sha256: str, *, resume: bool) -> dict:
    selected = load_selected(expected_sha256)
    if resume:
        if not PARTIAL.exists():
            raise SystemExit(f"FAIL: cannot resume; {PARTIAL} does not exist")
        partial = json.loads(PARTIAL.read_text())
        if partial["selected_optimization_sha256"] != expected_sha256:
            raise SystemExit("FAIL: partial evaluation uses another selected hash")
        commitment = partial["commitment"]
        verify_recorded_commitment(commitment, selected)
        runs = partial["runs"]
    else:
        if OUTPUT.exists() or PARTIAL.exists():
            raise SystemExit("FAIL: evaluation output already exists")
        commitment = capture_commitment(selected)
        if not all(commitment["checks"].values()):
            raise SystemExit("FAIL: pre-evaluation commitment checks did not pass")
        runs = []

    completed = {(row["block"], row["arm"]) for row in runs}
    blocks = (
        ("legacy_bridge", "main", DEV_SEEDS),
        ("fresh", "main", FRESH_SEEDS),
        ("fresh", "localized", FRESH_SEEDS),
    )
    with ProcessPoolExecutor(
        max_workers=EVALUATION_WORKERS,
        initializer=_worker_init,
        initargs=(str(SELECTED), expected_sha256),
    ) as executor:
        for block, arm, seeds in blocks:
            if (block, arm) in completed:
                print(f"resume {block}/{arm}")
                continue
            futures = {
                executor.submit(_run_cell, block, arm, seed): seed for seed in seeds
            }
            rows = [future.result() for future in as_completed(futures)]
            rows.sort(key=lambda row: row["seed"])
            runs.extend(rows)
            converged = sum(row["training"]["converged"] for row in rows)
            print(f"{block}/{arm}: {converged}/{len(rows)} TRAIN-converged")
            PARTIAL.write_text(render(_partial_payload(selected, commitment, runs)))

    payload = _evaluation_payload(selected, commitment, runs)
    OUTPUT.write_text(render(payload))
    if PARTIAL.exists():
        PARTIAL.unlink()
    print(
        f"wrote {OUTPUT}; fresh main convergence "
        f"{payload['fresh_main_converged_count']}/{len(FRESH_SEEDS)}"
    )
    return payload


def check() -> dict:
    if not OUTPUT.exists():
        raise SystemExit(f"FAIL: {OUTPUT} does not exist")
    stored = json.loads(OUTPUT.read_text())
    selected = load_selected(stored["selected_optimization_sha256"])
    verify_recorded_commitment(stored["commitment"], selected)
    rebuilt = _evaluation_payload(selected, stored["commitment"], stored["runs"])
    if stored != rebuilt:
        raise SystemExit("FAIL: locked evaluation artifact does not re-derive")
    print(
        f"PASS: {OUTPUT.name} re-derives at pre-evaluation commit "
        f"{stored['commitment']['pre_evaluation_commit']}"
    )
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selected-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if args.resume or args.selected_sha256:
            raise SystemExit("FAIL: --check takes no other option")
        check()
        return
    if not args.selected_sha256:
        raise SystemExit("FAIL: evaluation requires --selected-sha256 LOCK")
    run_evaluation(args.selected_sha256, resume=args.resume)


if __name__ == "__main__":
    main()
