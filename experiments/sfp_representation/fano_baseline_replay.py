"""Replay and verify the frozen 011.02 primary/main Phase-0 baseline.

This executable is deliberately separate from the 011.03 optimization search.  It
runs only the eight frozen primary/main cells, derives their normalized row digest,
headline metrics and converged/stalled partition, and compares each field with the
committed Phase-0 record.  No optimization candidate or fresh seed is available on
this path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fano_analysis import seed_partition
from fano_sweep import SweepConfig, run_sweep, summarize
from ladder_sweep import strip_non_replayable
from task import digest

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "011_optimization_artifacts" / "baseline_replay.json"
FROZEN_SWEEP = ROOT / "011_fano_artifacts" / "fano_sweep.json"
SEEDS = tuple(range(8))
EXPECTED_DIGEST = "6f172373d64a51b659ef9466139e34baf5d01a374a480966722179c2e0b38dd1"

HEADLINE_FIELDS = (
    "mate_exact_accuracy",
    "fano_completion_exact_set_accuracy",
    "mate_pairing_valid_rate",
    "fano_plane_valid_rate",
    "certified_plane_equivalent_rate",
    "transport_rate",
    "left_right_swap_identical_rate",
)

FROZEN_SOURCE_SHA256 = {
    "fano_task.py": "fb616d5d0739005cffe10f372a00bdebcada7903151b2ddeede7881c64fd1260",
    "fano_heads.py": "25aef2177e069fca0a3f339dc75269c6b3685f4c34c04e5adcb7b83f8d351963",
    "fano_baselines.py": "22f646bdb864b5e035772fbfd583bca87e95a034000f749e7623259565cd95ad",
    "fano_sweep.py": "5afad0da005ae3b3f4fdadc2e7594163898748df9f534e55cb1135f2df741b64",
    "fano_analysis.py": "dbde797966e5a8dc08cb1d583feaf04ce56e94f82a59aa1c4eb0533a21383220",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fail(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise SystemExit(f"FAIL: {label}: observed {observed!r}, expected {expected!r}")


def _ordered_primary_main(rows: list[dict]) -> list[dict]:
    selected = sorted(
        (
            row
            for row in rows
            if row.get("block") == "primary" and row.get("arm") == "main"
        ),
        key=lambda row: row["seed"],
    )
    _fail("primary/main row count", len(selected), len(SEEDS))
    _fail("primary/main seed ledger", [row["seed"] for row in selected], list(SEEDS))
    for row in selected:
        seed = row["seed"]
        _fail(f"seed {seed} message mode", row["config"]["message_mode"], "node")
        _fail(f"seed {seed} final training step", row["loss_trace"][-1]["step"], 799)
        for field in ("recovery", "transport", "left_right_swap"):
            if field not in row:
                raise SystemExit(f"FAIL: seed {seed} lacks full-audit field {field!r}")
    return selected


def _derive(rows: list[dict]) -> dict:
    ordered = _ordered_primary_main(rows)
    summary = summarize(ordered)["primary/main"]
    partition = seed_partition(ordered, "primary", "main")
    return {
        "rows_digest": digest(
            [strip_non_replayable(row) for row in ordered]
        ),
        "headline": {
            field: summary[field]["mean"] for field in HEADLINE_FIELDS
        },
        "final_loss_per_seed": partition["final_loss_per_seed"],
        "converged_seeds": partition["converged_seeds"],
        "stalled_seeds": partition["stalled_seeds"],
    }


def _validate_frozen_sources() -> None:
    for name, expected in FROZEN_SOURCE_SHA256.items():
        _fail(f"frozen source {name}", _file_sha256(ROOT / name), expected)


def _validate_derived(label: str, derived: dict, baseline: dict) -> None:
    _fail(f"{label} rows digest", derived["rows_digest"], EXPECTED_DIGEST)
    _fail(
        f"{label} digest vs baseline replayable_rows_digest",
        derived["rows_digest"],
        baseline["replayable_rows_digest"],
    )
    _fail(f"{label} headline", derived["headline"], baseline["headline"])
    _fail(
        f"{label} final loss ledger",
        derived["final_loss_per_seed"],
        baseline["final_loss_per_seed"],
    )
    _fail(
        f"{label} converged seeds",
        derived["converged_seeds"],
        baseline["converged_seeds"],
    )
    _fail(
        f"{label} stalled seeds",
        derived["stalled_seeds"],
        baseline["stalled_seeds"],
    )


def check_committed() -> dict:
    baseline = _load(BASELINE)
    _validate_frozen_sources()
    committed = _derive(_load(FROZEN_SWEEP)["runs"])
    _validate_derived("committed 011.02", committed, baseline)
    _fail(
        "baseline reproduced_committed_rows_digest",
        baseline["reproduced_committed_rows_digest"],
        committed["rows_digest"],
    )
    _fail(
        "baseline reproduces_byte_normalized_rows",
        baseline["reproduces_byte_normalized_rows"],
        True,
    )
    return baseline


def replay() -> None:
    baseline = check_committed()
    config = SweepConfig(
        block="primary",
        message_mode="node",
        seeds=SEEDS,
        arms=("main",),
        steps=800,
        full_audits=True,
    )
    rebuilt = _derive(run_sweep((config,))["runs"])
    _validate_derived("fresh Phase-0 replay", rebuilt, baseline)
    _fail(
        "fresh replay vs committed rows",
        rebuilt["rows_digest"],
        baseline["reproduced_committed_rows_digest"],
    )
    print(
        "PASS: exact 011.02 primary/main Phase-0 replay; "
        f"5/8 converged; rows {rebuilt['rows_digest']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--committed-only",
        action="store_true",
        help="validate the frozen committed rows without retraining",
    )
    args = parser.parse_args()
    if args.committed_only:
        baseline = check_committed()
        print(
            "PASS: committed 011.02 primary/main rows and Phase-0 record agree; "
            f"rows {baseline['replayable_rows_digest']}"
        )
        return
    replay()


if __name__ == "__main__":
    main()
