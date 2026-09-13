"""011.03 preregistration: finite optimization-only candidates and selector.

This module is stdlib-only.  It freezes every candidate before comparative
training begins, including the exact 011.02 baseline, the exact convergence
criterion, development/fresh seed blocks, and the Owner-specified lexicographic
selector.  It contains no model, data, target, score, or evaluation code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_optimization_artifacts"
OUTPUT = ARTIFACTS / "candidate_manifest.json"

SCHEMA = "occurrence.011.03.optimization-candidates.v1"
DEV_SEEDS = tuple(range(8))
FRESH_SEEDS = tuple(range(1000, 1016))
EXACT_LOSS_MINIMUM = math.log(2.0) / 2.0
LOSS_TOLERANCE = 1e-4
MAX_CANDIDATES = 24
MAX_STEPS = 1600

BASELINE = {
    "initialization": {
        "scheme": "torch_default",
        "gain": 1.0,
        "linear_bias": "torch_default",
    },
    "optimizer": {
        "name": "Adam",
        "learning_rate": 0.003,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    },
    "schedule": {
        "name": "constant",
        "warmup_steps": 0,
        "minimum_lr_factor": 1.0,
    },
    "gradient_clip_norm": None,
    "max_steps": 800,
    "early_stopping": False,
}

KNOB_PATHS = (
    "initialization.scheme",
    "initialization.gain",
    "initialization.linear_bias",
    "optimizer.name",
    "optimizer.learning_rate",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.epsilon",
    "optimizer.weight_decay",
    "schedule.name",
    "schedule.warmup_steps",
    "schedule.minimum_lr_factor",
    "gradient_clip_norm",
    "max_steps",
    "early_stopping",
)


def canonical_text(value: object) -> str:
    """Canonical JSON used for every manifest and candidate hash."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_text(value).encode()).hexdigest()


def _nested_get(row: dict, path: str) -> object:
    value: object = row
    for part in path.split("."):
        if not isinstance(value, dict):
            raise TypeError(f"{path!r} does not name a manifest leaf")
        value = value[part]
    return value


def _merge(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _candidate(candidate_id: str, changes: dict) -> dict:
    prescription = _merge(BASELINE, changes)
    changed = [
        path
        for path in KNOB_PATHS
        if _nested_get(prescription, path) != _nested_get(BASELINE, path)
    ]
    canonical = canonical_text(prescription)
    return {
        "candidate_id": candidate_id,
        "prescription": prescription,
        "canonical_json": canonical,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "changed_knobs": changed,
        "changed_knob_count": len(changed),
    }


def candidates() -> list[dict]:
    """The complete finite set, frozen before the first comparative run."""

    extended = {"max_steps": 1600, "early_stopping": True}
    lr_fast = {
        **extended,
        "optimizer": {"learning_rate": 0.006},
    }
    responsive = {
        **extended,
        "optimizer": {
            "learning_rate": 0.003,
            "beta1": 0.5,
            "beta2": 0.99,
        },
    }
    tiny_epsilon = {
        **extended,
        "optimizer": {
            "learning_rate": 0.003,
            "beta1": 0.8,
            "beta2": 0.99,
            "epsilon": 1e-10,
        },
    }
    return [
        _candidate("baseline", {}),
        _candidate("default-extended", extended),
        _candidate("default-lr006", lr_fast),
        _candidate("default-responsive", responsive),
        _candidate("default-tiny-epsilon", tiny_epsilon),
        _candidate(
            "default-warmup-cosine",
            {
                **extended,
                "optimizer": {"learning_rate": 0.01},
                "schedule": {
                    "name": "warmup_cosine",
                    "warmup_steps": 100,
                    "minimum_lr_factor": 0.03,
                },
            },
        ),
        _candidate(
            "default-adamw",
            {
                **extended,
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 0.003,
                    "weight_decay": 0.0001,
                },
            },
        ),
        _candidate(
            "default-lr006-clip5",
            {**lr_fast, "gradient_clip_norm": 5.0},
        ),
        _candidate(
            "default-bias-zero",
            {
                **extended,
                "initialization": {"linear_bias": 0.0},
            },
        ),
        _candidate(
            "default-bias-positive001",
            {
                **extended,
                "initialization": {"linear_bias": 0.01},
            },
        ),
        _candidate(
            "default-bias-positive01",
            {
                **extended,
                "initialization": {"linear_bias": 0.1},
            },
        ),
        _candidate(
            "default-bias-positive001-responsive",
            {
                **responsive,
                "initialization": {"linear_bias": 0.01},
            },
        ),
        _candidate(
            "scaled-default-050",
            {
                **extended,
                "initialization": {
                    "scheme": "scaled_default",
                    "gain": 0.5,
                    "linear_bias": 0.0,
                },
            },
        ),
        _candidate(
            "scaled-default-150",
            {
                **extended,
                "initialization": {
                    "scheme": "scaled_default",
                    "gain": 1.5,
                    "linear_bias": 0.01,
                },
            },
        ),
        _candidate(
            "scaled-default-200",
            {
                **extended,
                "initialization": {
                    "scheme": "scaled_default",
                    "gain": 2.0,
                    "linear_bias": 0.01,
                },
            },
        ),
        _candidate(
            "xavier-100-zero",
            {
                **extended,
                "initialization": {
                    "scheme": "xavier_uniform",
                    "gain": 1.0,
                    "linear_bias": 0.0,
                },
            },
        ),
        _candidate(
            "xavier-100-positive001",
            {
                **extended,
                "initialization": {
                    "scheme": "xavier_uniform",
                    "gain": 1.0,
                    "linear_bias": 0.01,
                },
            },
        ),
        _candidate(
            "xavier-100-positive001-lr006",
            {
                **lr_fast,
                "initialization": {
                    "scheme": "xavier_uniform",
                    "gain": 1.0,
                    "linear_bias": 0.01,
                },
            },
        ),
        _candidate(
            "xavier-1414-positive001",
            {
                **extended,
                "initialization": {
                    "scheme": "xavier_uniform",
                    "gain": math.sqrt(2.0),
                    "linear_bias": 0.01,
                },
                "optimizer": {"learning_rate": 0.001},
            },
        ),
        _candidate(
            "xavier-1414-positive001-responsive",
            {
                **tiny_epsilon,
                "initialization": {
                    "scheme": "xavier_uniform",
                    "gain": math.sqrt(2.0),
                    "linear_bias": 0.01,
                },
            },
        ),
        _candidate(
            "orthogonal-100-zero",
            {
                **extended,
                "initialization": {
                    "scheme": "orthogonal",
                    "gain": 1.0,
                    "linear_bias": 0.0,
                },
            },
        ),
        _candidate(
            "orthogonal-1414-positive001",
            {
                **extended,
                "initialization": {
                    "scheme": "orthogonal",
                    "gain": math.sqrt(2.0),
                    "linear_bias": 0.01,
                },
                "optimizer": {"learning_rate": 0.001},
            },
        ),
        _candidate(
            "kaiming-zero-lr001",
            {
                **extended,
                "initialization": {
                    "scheme": "kaiming_uniform",
                    "gain": 1.0,
                    "linear_bias": 0.0,
                },
                "optimizer": {"learning_rate": 0.001},
            },
        ),
        _candidate(
            "kaiming-positive001-lr001",
            {
                **extended,
                "initialization": {
                    "scheme": "kaiming_uniform",
                    "gain": 1.0,
                    "linear_bias": 0.01,
                },
                "optimizer": {"learning_rate": 0.001},
            },
        ),
    ]


def manifest_body() -> dict:
    rows = candidates()
    if len(rows) > MAX_CANDIDATES:
        raise AssertionError(f"{len(rows)} candidates exceeds cap {MAX_CANDIDATES}")
    if rows[0]["prescription"] != BASELINE:
        raise AssertionError("the first candidate is not the exact baseline")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise AssertionError("candidate prescriptions are not unique")
    if any(row["prescription"]["max_steps"] > MAX_STEPS for row in rows):
        raise AssertionError("a candidate exceeds the 2x training-step cap")
    return {
        "schema": SCHEMA,
        "task": "011.03-GPT-optimization-reliability-closure-task.md",
        "frozen_architecture": {
            "message_mode": "node",
            "rounds": 4,
            "width": 32,
            "aggregation": "sum",
            "residual": True,
            "layer_norm": True,
            "heads": 2,
        },
        "fixed_optimizer_runtime": {
            "amsgrad": False,
            "capturable": False,
            "differentiable": False,
            "foreach": False,
            "fused": False,
            "maximize": False,
        },
        "development": {
            "arm": "main",
            "split": "train",
            "seeds": list(DEV_SEEDS),
            "candidate_count": len(rows),
            "candidate_cap": MAX_CANDIDATES,
            "maximum_steps": MAX_STEPS,
            "fresh_or_held_out_metrics_permitted": False,
        },
        "fresh_seed_commitment": list(FRESH_SEEDS),
        "convergence": {
            "exact_loss_minimum": EXACT_LOSS_MINIMUM,
            "derivation": "(0 + ln(2)) / 2",
            "absolute_loss_tolerance": LOSS_TOLERANCE,
            "loss_threshold": EXACT_LOSS_MINIMUM + LOSS_TOLERANCE,
            "mate_exact_fit": "112/112 TRAIN decisions",
            "completion_exact_fit": "672/672 TRAIN exact-set decisions",
            "finite_parameters_and_logits": True,
            "steps_to_convergence": (
                "number of completed optimizer updates at the first state meeting "
                "the loss and both exact-fit checks"
            ),
        },
        "selection_rule": [
            "number of converged development seeds, higher",
            "median steps-to-convergence among converged seeds, lower",
            "worst final excess loss over the exact minimum, lower",
            "number of optimization knobs changed from baseline, lower",
            "canonical candidate JSON lexical order, lower",
        ],
        "changed_knob_definition": list(KNOB_PATHS),
        "baseline": BASELINE,
        "candidates": rows,
    }


def manifest() -> dict:
    body = manifest_body()
    return {**body, "manifest_sha256": object_sha256(body)}


def render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def validate_loaded(value: dict) -> None:
    expected = manifest()
    if value != expected:
        raise ValueError("candidate manifest does not match the frozen source")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if object_sha256(body) != value["manifest_sha256"]:
        raise ValueError("candidate manifest hash is invalid")
    for row in value["candidates"]:
        if canonical_text(row["prescription"]) != row["canonical_json"]:
            raise ValueError(f"candidate {row['candidate_id']} canonical JSON differs")
        if hashlib.sha256(row["canonical_json"].encode()).hexdigest() != row["sha256"]:
            raise ValueError(f"candidate {row['candidate_id']} hash differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = manifest()
    text = render(payload)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            raise SystemExit(f"FAIL: {OUTPUT} does not match the frozen manifest")
        validate_loaded(json.loads(OUTPUT.read_text()))
        print(
            f"PASS: {OUTPUT.name} freezes {len(payload['candidates'])} candidates "
            f"at {payload['manifest_sha256']}"
        )
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(
        f"wrote {OUTPUT} with {len(payload['candidates'])} candidates; "
        f"sha256 {payload['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
