"""Build the content-addressed main-arm TRAIN-only capsule for 011.03.

This trusted preparation step is the only new 011.03 code that constructs task
queries before selection.  It selects the eight frozen TRAIN namespaces first,
then builds tensors only for those namespaces.  The search executable cannot
import this module and consumes only the resulting capsule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from fano_optimization_core import capsule_digest, file_sha256, load_capsule
from fano_optimization_manifest import OUTPUT as CANDIDATE_MANIFEST
from fano_optimization_manifest import object_sha256, validate_loaded
from fano_sweep import build_batches
from fano_task import build_namespaces, build_observation
from task import build_dataset

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_optimization_artifacts"
CAPSULE = ARTIFACTS / "train_capsule.pt"
OUTPUT = ARTIFACTS / "train_capsule.json"
FROZEN_ARTIFACT = ROOT / "011_fano_artifacts" / "fano_task.json"

FROZEN_SOURCES = (
    "fano_task.py",
    "fano_heads.py",
    "fano_baselines.py",
    "fano_sweep.py",
    "fano_analysis.py",
)


def _plain_batch(batch: object) -> dict:
    return {
        "label": batch.label,
        "split": batch.split,
        "adjacency": batch.adjacency.detach().cpu().contiguous(),
        "per_kind": [
            {
                "kind": kind.kind,
                "head": kind.head,
                "marks": kind.marks.detach().cpu().contiguous(),
                "targets": kind.targets.detach().cpu().contiguous(),
                "scored": kind.scored.detach().cpu().contiguous(),
                "sizes": list(kind.sizes),
            }
            for kind in batch.per_kind
        ],
    }


def build_capsule() -> dict:
    frozen = json.loads(FROZEN_ARTIFACT.read_text())
    observation = build_observation(build_dataset())
    namespaces = tuple(
        namespace
        for namespace in build_namespaces(observation)
        if namespace.split == "train"
    )
    if len(namespaces) != 8:
        raise AssertionError(f"found {len(namespaces)} TRAIN namespaces, expected 8")
    batches_by_split = build_batches(observation, namespaces)
    if set(batches_by_split) != {"train"}:
        raise AssertionError(
            f"TRAIN capsule preparation built splits {sorted(batches_by_split)}"
        )
    batches = [_plain_batch(batch) for batch in batches_by_split["train"]]
    if observation.sha256() != frozen["digests"]["main_observation_sha256"]:
        raise AssertionError("main observation digest moved")
    return {
        "schema": "occurrence.011.03.train-capsule.v1",
        "arm": "main",
        "split": "train",
        "observation_sha256": observation.sha256(),
        "namespace_labels": [namespace.label for namespace in namespaces],
        "namespace_sha256": [namespace.sha256() for namespace in namespaces],
        "batch_count": len(batches),
        "mate_queries": sum(
            len(batch["per_kind"][0]["sizes"]) for batch in batches
        ),
        "completion_queries": sum(
            len(batch["per_kind"][1]["sizes"]) for batch in batches
        ),
        "batches": batches,
    }


def capsule_manifest(capsule: dict) -> dict:
    candidate = json.loads(CANDIDATE_MANIFEST.read_text())
    validate_loaded(candidate)
    body = {
        "schema": "occurrence.011.03.train-capsule-manifest.v1",
        "candidate_manifest_sha256": candidate["manifest_sha256"],
        "capsule_file": CAPSULE.relative_to(ROOT).as_posix(),
        "capsule_file_sha256": file_sha256(CAPSULE),
        "capsule_content_sha256": capsule_digest(capsule),
        "selection_surface": {
            "arm": capsule["arm"],
            "split": capsule["split"],
            "batch_count": capsule["batch_count"],
            "mate_queries": capsule["mate_queries"],
            "completion_queries": capsule["completion_queries"],
            "namespace_labels": capsule["namespace_labels"],
            "contains_any_other_split": False,
            "contains_control": False,
        },
        "frozen_source_sha256": {
            name: file_sha256(ROOT / name) for name in FROZEN_SOURCES
        },
        "frozen_task_artifact_sha256": file_sha256(FROZEN_ARTIFACT),
        "main_observation_sha256": capsule["observation_sha256"],
        "torch_serialization_note": (
            "the plain file hash identifies the committed container; the content "
            "hash independently covers every metadata field and exact tensor byte"
        ),
    }
    return {**body, "manifest_sha256": object_sha256(body)}


def render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def check() -> dict:
    if not CAPSULE.exists() or not OUTPUT.exists():
        raise SystemExit("FAIL: TRAIN capsule artifacts do not exist")
    stored_manifest = json.loads(OUTPUT.read_text())
    stored_capsule = load_capsule(CAPSULE)
    if file_sha256(CAPSULE) != stored_manifest["capsule_file_sha256"]:
        raise SystemExit("FAIL: TRAIN capsule file hash differs")
    if capsule_digest(stored_capsule) != stored_manifest["capsule_content_sha256"]:
        raise SystemExit("FAIL: TRAIN capsule content hash differs")
    rebuilt = build_capsule()
    if capsule_digest(rebuilt) != stored_manifest["capsule_content_sha256"]:
        raise SystemExit("FAIL: frozen code does not rebuild the TRAIN capsule")
    expected_manifest = capsule_manifest(stored_capsule)
    if expected_manifest != stored_manifest:
        raise SystemExit("FAIL: TRAIN capsule manifest differs")
    print(
        f"PASS: {CAPSULE.name} contains only 8 main/TRAIN batches; "
        f"content {stored_manifest['capsule_content_sha256']}"
    )
    return stored_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check()
        return
    if not CANDIDATE_MANIFEST.exists():
        raise SystemExit(
            f"FAIL: write the frozen candidate manifest first: {CANDIDATE_MANIFEST}"
        )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    capsule = build_capsule()
    torch.save(capsule, CAPSULE)
    payload = capsule_manifest(capsule)
    OUTPUT.write_text(render(payload))
    print(
        f"wrote {CAPSULE} and {OUTPUT}; "
        f"content {payload['capsule_content_sha256']}"
    )


if __name__ == "__main__":
    main()
