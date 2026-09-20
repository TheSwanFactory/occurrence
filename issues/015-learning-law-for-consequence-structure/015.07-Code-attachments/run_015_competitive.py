#!/usr/bin/env python3
"""Execute Issue 015.06's matched competitive-consequence experiment.

Information exposure: this production experiment uses the complete curated
Issue-015.03 packet (84 opaque Event IDs, 48 TRAIN records, and 96
evaluation-only ROLE_TEST records), the authorized Issue-015.05/015.06 texts,
and the accepted Issue-015.04 artifacts. No NOVEL_TEST data, Issue 016 content,
SFP/Fano/habitat coordinates, global completion relation, or certified
completion tables are accessed. Both optimization objectives receive only the
48 TRAIN positives and deterministic structures derived from them. Current
ROLE_TEST metrics are first computed by ``score`` after an externally published
and read-back precommit.

The required staged invocation is intentionally strict::

    python run_015_competitive.py prepare \
        --benchmark-source PATH_OR_HTTPS_OR_QUILT_URI \
        --predecessor-trajectory-source PATH_OR_HTTPS_OR_QUILT_URI \
        --out-dir DIR --base-revision REVISION --base-entry-count COUNT
    # publish/read back DIR/precommit.json without changing it
    python run_015_competitive.py score \
        --benchmark-source PATH_OR_HTTPS_OR_QUILT_URI \
        --predecessor-trajectory-source PATH_OR_HTTPS_OR_QUILT_URI \
        --out-dir DIR --precommit-sha256 SHA256 \
        --precommit-package-revision REVISION
    python run_015_competitive.py verify \
        --predecessor-trajectory-source PATH_OR_HTTPS_OR_QUILT_URI \
        --out-dir DIR --result RESULT_MD

``prepare`` performs identity, substrate, proof, scorer, initialization, and
baseline-reference audits, but computes no current ROLE metric. ``score`` uses
one frozen scorer and optimizer with two frozen TRAIN-only objectives. It does
not tune, stop early, add seeds, or rerun based on outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

SCHEMA_PREFIX = "occurrence.gpt.01507"
INFORMATION_EXPOSURE = (
    "Production execution with the complete curated Issue-015.03 packet: 84 "
    "opaque Event IDs, 48 TRAIN records, and 96 evaluation-only ROLE_TEST "
    "records; the authorized Issue-015.05/015.06 texts; and the accepted "
    "Issue-015.04 artifacts. No NOVEL_TEST data, Issue 016 content, "
    "SFP/Fano/habitat coordinates, global completion relation, or certified "
    "completion tables were accessed. Both optimization objectives received "
    "only the 48 TRAIN positives and deterministic structures derived from "
    "them; current ROLE_TEST metrics were first computed after an externally "
    "published and read-back precommit."
)

TASK_REVISION = "522ff97ad90d2462595398ed0271ff6219d15ce68a0ca654c7cbd4490445aaac"
TASK_PATH = (
    "issues/015-learning-law-for-consequence-structure/"
    "015.06-GPT-role-neutral-competitive-consequence-learning-Coder.md"
)
TASK_URI = f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}&path={TASK_PATH}"
TASK_SHA256 = "71538e0d59c50d818969e5b4a752b863826de2568eb3640610e9be35301e6614"

OWNER_PATH = (
    "issues/015-learning-law-for-consequence-structure/"
    "015.05-GPT-Owner-adjudication-of-role-neutral-score-result.md"
)
OWNER_URI = f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}&path={OWNER_PATH}"
OWNER_SHA256 = "b2ad42ddc793f5d9aac6a2ac9db6fd579ad03601d214b652f44b2ece60b9685a"

BENCHMARK_REVISION = "6ceb230dae9e069011062ca88b13198042cee20b0ba8dc85ec4f3cf1623f5731"
BENCHMARK_PATH = (
    "issues/015-learning-law-for-consequence-structure/"
    "015.03-Code-attachments/role-learning-benchmark.json"
)
BENCHMARK_URI = (
    f"quilt+s3://protology#package=occurrence/gpt@{BENCHMARK_REVISION}"
    f"&path={BENCHMARK_PATH}"
)
BENCHMARK_SHA256 = "c9b7a0931ff9cd5e3084048bfa190dbaf7aef22f514d6fc2f0851b2abdc70dac"
EXPECTED_TRAIN_DIGEST = "ca7178877eed4f4a537deb9fb56e9a66f81a39395298a828b5fb058c6a7c9088"
EXPECTED_ROLE_DIGEST = "f000b04566a83a3ff7a9536fe4e1d5f8889b51df269879d757b1b2b60e6b7f37"

PREDECESSOR_REVISION = "27612778252e0858328b4325154d07f068eafe6cc961dfe36c0861cbf79b5bf4"
PREDECESSOR_BASE = (
    f"quilt+s3://protology#package=occurrence/gpt@{PREDECESSOR_REVISION}&path="
    "issues/015-learning-law-for-consequence-structure/"
)
PREDECESSOR_RESULT_PATH = "015.04-Coder-role-neutral-triad-learning-result-GPT.md"
PREDECESSOR_PRECOMMIT_PATH = "015.04-Code-attachments/precommit.json"
PREDECESSOR_ANALYSIS_PATH = "015.04-Code-attachments/analysis.json"
PREDECESSOR_TRAJECTORY_PATH = (
    "015.04-Code-attachments/trajectory_triad_symmetric.json"
)
PREDECESSOR_RUNNER_PATH = "015.04-Code-attachments/run_015_role.py"
PREDECESSOR_RESULT_SHA256 = (
    "a9b33345e911e68f53c1abe995cb5bfe99c9dd466d310a34d5c2715e8d57de79"
)
PREDECESSOR_PRECOMMIT_SHA256 = (
    "718bd4c756eb95999422fd0f516517fa5ec227e3e2ddb77c273ab1809eccec8f"
)
PREDECESSOR_ANALYSIS_SHA256 = (
    "d5de3f676accd94724ee8a4bd5c13a4eee778eca799d9fd1114502a1c8387987"
)
PREDECESSOR_TRAJECTORY_SHA256 = (
    "3f796d234067c469d1cb4378199d5fe790cb84cc372ec5e10bd8897952e13a24"
)
PREDECESSOR_RUNNER_SHA256 = (
    "163118e4a5dc6af9589ac98cac7734c1d5bd627d32904741b43e28bf604abadd"
)
PREDECESSOR_RESULT_URI = PREDECESSOR_BASE + PREDECESSOR_RESULT_PATH
PREDECESSOR_PRECOMMIT_URI = PREDECESSOR_BASE + PREDECESSOR_PRECOMMIT_PATH
PREDECESSOR_ANALYSIS_URI = PREDECESSOR_BASE + PREDECESSOR_ANALYSIS_PATH
PREDECESSOR_TRAJECTORY_URI = PREDECESSOR_BASE + PREDECESSOR_TRAJECTORY_PATH
PREDECESSOR_RUNNER_URI = PREDECESSOR_BASE + PREDECESSOR_RUNNER_PATH

N_EVENTS = 84
EMBED_DIM = 64
HIDDEN_DIM = 256
LEGAL_CANDIDATES = 82
POSITIVE_TRIADS = 48
ROLE_RECORDS = 96
CONSTITUENT_PAIRS = 144
CORRUPTIONS_PER_BRANCH = 81
CORRUPTIONS_PER_POSITIVE = 243
RELATION_CLASSES = 244
RELATION_TEMPERATURE = 1.0
SCORED_SEEDS = (3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007)
HORIZON = 16384
CHECKPOINTS = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    *range(1536, HORIZON + 1, 512),
)
ARMS = ("query_ce", "relational_competition")
DEFAULT_THREADS = 10

OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "betas": [0.9, 0.98],
    "eps": 1e-8,
    "schedule": "constant",
    "full_batch": True,
}

PRECOMMIT_NAME = "precommit.json"
QUERY_NAME = "trajectory_query_ce.json"
RELATION_NAME = "trajectory_relational_competition.json"
ANALYSIS_NAME = "analysis.json"
RUNNER_NAME = "run_015_competitive.py"
DEFAULT_RESULT_NAME = "015.07-Coder-role-neutral-competitive-consequence-result-GPT.md"
EXPECTED_ATTACHMENT_FILES = {
    PRECOMMIT_NAME,
    QUERY_NAME,
    RELATION_NAME,
    ANALYSIS_NAME,
    RUNNER_NAME,
}

ACCEPTED_ENVIRONMENT = {
    "python": "3.12.11 (main, Jun  4 2025, 17:42:58) [Clang 20.1.4 ]",
    "python_executable": "/Users/ernest/GitHub/occurrence/.venv/bin/python",
    "platform": "macOS-26.6.2-arm64-arm-64bit",
    "machine": "arm64",
    "processor": "arm",
    "torch": "2.14.0",
    "numpy": "2.5.1",
    "device": "cpu",
    "cpu_count": 14,
    "torch_num_threads": 10,
    "torch_num_interop_threads": 1,
    "requested_threads": 10,
}

BASELINE_POLICY = {
    "comparison": "exact recursive equality of every shared stored value",
    "shared_scope": (
        "seed and initialization hash; optimizer; completed horizon and early-stop "
        "flag; every checkpoint update, designated-query CE/accuracy, ROLE metrics, "
        "all-three-completion metrics, ROLE margins, parameter norms, and Event "
        "embedding diagnostics; derived shared temporal and final behavior"
    ),
    "new_relation_diagnostics_excluded": True,
    "timestamps_and_new_provenance_excluded": True,
    "material_mismatch_definition": "any shared-value deviation",
    "environment_differences_are_reported_but_do_not_waive_a_deviation": True,
    "failure_disposition": "C0/BLOCKED-BASELINE-REPLICATION; no causal interpretation",
    "tuning_to_recover_replication": False,
}


class ContractError(RuntimeError):
    """Raised when a frozen scientific or execution contract is violated."""


@dataclass(frozen=True, order=True)
class Record:
    """One unordered pair-to-designated-third record."""

    a: int
    b: int
    target: int

    def as_triple(self) -> list[int]:
        return [self.a, self.b, self.target]

    @property
    def query(self) -> tuple[int, int]:
        return (self.a, self.b)

    @property
    def unordered_triad(self) -> tuple[int, int, int]:
        return tuple(sorted((self.a, self.b, self.target)))


@dataclass(frozen=True)
class Batch:
    """A fixed collection of queries and their 82 legal candidate classes."""

    records: tuple[Record, ...]
    pair_a: Tensor
    pair_b: Tensor
    candidates: Tensor
    target_classes: Tensor


@dataclass(frozen=True)
class RelationBatch:
    """The 48 fixed 244-way positive-versus-local-corruption relations."""

    positive_triads: tuple[tuple[int, int, int], ...]
    candidates: Tensor
    target_classes: Tensor


class SharedScorer(nn.Module):
    """The exact parameterization accepted for Issue 015.04 Arm T."""

    def __init__(self) -> None:
        super().__init__()
        self.event = nn.Embedding(N_EVENTS, EMBED_DIM)
        self.linear1 = nn.Linear(4 * EMBED_DIM, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.event.weight, gain=1.0)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)

    def base_score(self, pair_a: Tensor, pair_b: Tensor, candidate: Tensor) -> Tensor:
        ea = self.event(pair_a)
        eb = self.event(pair_b)
        ec = self.event(candidate)
        features = torch.cat((ea + eb, ea * eb, torch.abs(ea - eb), ec), dim=-1)
        return self.linear2(F.gelu(self.linear1(features), approximate="none")).squeeze(-1)

    def triad_score(self, first: Tensor, second: Tensor, third: Tensor) -> Tensor:
        """Score canonical unordered triads in one fixed decomposition order."""
        ordered, _ = torch.sort(torch.stack((first, second, third), dim=-1), dim=-1)
        x, y, z = ordered.unbind(dim=-1)
        score_xy_z = self.base_score(x, y, z)
        score_xz_y = self.base_score(x, z, y)
        score_yz_x = self.base_score(y, z, x)
        return ((score_xy_z + score_xz_y) + score_yz_x) / 3.0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise ContractError(f"{label} must be 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ContractError(f"{label} must be 64 hexadecimal characters") from error


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def compact_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(),
            parse_constant=reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"cannot read valid finite JSON from {path}: {error}") from error


def read_quilt_source(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme != "quilt+s3" or not parsed.netloc:
        raise ContractError(f"invalid quilt+s3 source: {source}")
    parameters = parse_qs(parsed.fragment, strict_parsing=True)
    if set(parameters) != {"package", "path"}:
        raise ContractError("quilt+s3 source requires exactly package and path")
    package_reference = parameters["package"]
    logical_paths = parameters["path"]
    if len(package_reference) != 1 or len(logical_paths) != 1:
        raise ContractError("quilt+s3 source has repeated package or path")
    if "@" not in package_reference[0]:
        raise ContractError("quilt+s3 package must include an immutable revision")
    package_name, revision = package_reference[0].rsplit("@", 1)
    require_sha256(revision, "quilt+s3 package revision")
    logical_path = logical_paths[0]
    try:
        import quilt3  # type: ignore[import-not-found]
    except ImportError as error:
        raise ContractError(
            "quilt+s3 input requires optional quilt3; alternatively pass a local "
            "path or HTTPS URL"
        ) from error

    try:
        package = quilt3.Package.browse(
            package_name,
            registry=f"s3://{parsed.netloc}",
            top_hash=revision,
        )
        entry = package[logical_path]
        get_bytes = getattr(entry, "get_bytes", None)
        if callable(get_bytes):
            raw = get_bytes()
            if isinstance(raw, bytes):
                return raw
        with tempfile.TemporaryDirectory(prefix="occurrence-01507-") as directory:
            destination = Path(directory) / Path(logical_path).name
            fetched = entry.fetch(str(destination))
            candidates = [destination]
            if isinstance(fetched, (str, Path)):
                candidates.insert(0, Path(fetched))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.read_bytes()
    except ContractError:
        raise
    except Exception as error:
        raise ContractError(f"failed to read immutable Quilt source {source}: {error}") from error
    raise ContractError(f"Quilt entry did not yield bytes: {source}")


def read_source(source: str) -> bytes:
    if source.startswith("quilt+s3://"):
        return read_quilt_source(source)
    if source.startswith(("https://", "http://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            return response.read()
    return Path(source).read_bytes()


def configure_determinism(threads: int) -> None:
    if threads < 1:
        raise ContractError("threads must be positive")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def environment_record(threads: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "requested_threads": threads,
    }


def provenance_record(uri: str, revision: str, path: str, content_hash: str) -> dict[str, str]:
    return {
        "uri": uri,
        "revision": revision,
        "path": path,
        "content_sha256": content_hash,
    }


def pinned_inputs() -> dict[str, Any]:
    return {
        "handed_task": provenance_record(TASK_URI, TASK_REVISION, TASK_PATH, TASK_SHA256),
        "owner_adjudication": provenance_record(
            OWNER_URI, TASK_REVISION, OWNER_PATH, OWNER_SHA256
        ),
        "benchmark": provenance_record(
            BENCHMARK_URI,
            BENCHMARK_REVISION,
            BENCHMARK_PATH,
            BENCHMARK_SHA256,
        ),
        "accepted_predecessor": {
            "revision": PREDECESSOR_REVISION,
            "result": provenance_record(
                PREDECESSOR_RESULT_URI,
                PREDECESSOR_REVISION,
                (
                    "issues/015-learning-law-for-consequence-structure/"
                    + PREDECESSOR_RESULT_PATH
                ),
                PREDECESSOR_RESULT_SHA256,
            ),
            "precommit": provenance_record(
                PREDECESSOR_PRECOMMIT_URI,
                PREDECESSOR_REVISION,
                (
                    "issues/015-learning-law-for-consequence-structure/"
                    + PREDECESSOR_PRECOMMIT_PATH
                ),
                PREDECESSOR_PRECOMMIT_SHA256,
            ),
            "analysis": provenance_record(
                PREDECESSOR_ANALYSIS_URI,
                PREDECESSOR_REVISION,
                (
                    "issues/015-learning-law-for-consequence-structure/"
                    + PREDECESSOR_ANALYSIS_PATH
                ),
                PREDECESSOR_ANALYSIS_SHA256,
            ),
            "triad_symmetric_trajectory": provenance_record(
                PREDECESSOR_TRAJECTORY_URI,
                PREDECESSOR_REVISION,
                (
                    "issues/015-learning-law-for-consequence-structure/"
                    + PREDECESSOR_TRAJECTORY_PATH
                ),
                PREDECESSOR_TRAJECTORY_SHA256,
            ),
            "runner": provenance_record(
                PREDECESSOR_RUNNER_URI,
                PREDECESSOR_REVISION,
                (
                    "issues/015-learning-law-for-consequence-structure/"
                    + PREDECESSOR_RUNNER_PATH
                ),
                PREDECESSOR_RUNNER_SHA256,
            ),
        },
    }


def parse_records(raw: Sequence[dict[str, Any]], label: str) -> tuple[Record, ...]:
    parsed: list[Record] = []
    for index, item in enumerate(raw):
        if set(item) != {"query", "target"}:
            raise ContractError(f"{label}[{index}] has unexpected fields")
        query = item["query"]
        target = item["target"]
        if not isinstance(query, list) or len(query) != 2:
            raise ContractError(f"{label}[{index}] query is not a two-item list")
        if not all(type(value) is int for value in (*query, target)):
            raise ContractError(f"{label}[{index}] contains non-integer IDs")
        a, b = query
        if not (0 <= a < N_EVENTS and 0 <= b < N_EVENTS and 0 <= target < N_EVENTS):
            raise ContractError(f"{label}[{index}] contains an out-of-range ID")
        if not a < b:
            raise ContractError(f"{label}[{index}] query is not canonically unordered")
        if len({a, b, target}) != 3:
            raise ContractError(f"{label}[{index}] does not contain three distinct Events")
        parsed.append(Record(a, b, target))
    return tuple(parsed)


def records_digest(records: Sequence[Record]) -> str:
    compact = json.dumps(
        [record.as_triple() for record in records], separators=(",", ":")
    ).encode()
    return sha256_bytes(compact)


def alternate_roles(train: Sequence[Record]) -> tuple[Record, ...]:
    derived: list[Record] = []
    for record in train:
        a, b, c = record.a, record.b, record.target
        pair_for_a = sorted((b, c))
        pair_for_b = sorted((a, c))
        derived.append(Record(pair_for_a[0], pair_for_a[1], a))
        derived.append(Record(pair_for_b[0], pair_for_b[1], b))
    return tuple(derived)


def all_role_completions(train: Sequence[Record]) -> tuple[Record, ...]:
    records: list[Record] = []
    for original in train:
        a, b, c = original.a, original.b, original.target
        records.append(original)
        pair_for_a = sorted((b, c))
        pair_for_b = sorted((a, c))
        records.append(Record(pair_for_a[0], pair_for_a[1], a))
        records.append(Record(pair_for_b[0], pair_for_b[1], b))
    return tuple(records)


def validate_benchmark(
    raw_bytes: bytes,
) -> tuple[dict[str, Any], tuple[Record, ...], tuple[Record, ...]]:
    actual_hash = sha256_bytes(raw_bytes)
    if actual_hash != BENCHMARK_SHA256:
        raise ContractError(
            f"BLOCKED-BENCHMARK-MISMATCH content hash {actual_hash} != {BENCHMARK_SHA256}"
        )
    packet = json.loads(raw_bytes)
    if packet.get("schema") != "occurrence.gpt.015.role-learning-benchmark.v1":
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH schema")
    universe = packet.get("token_universe", {})
    if universe.get("count") != N_EVENTS or universe.get("ids") != list(range(N_EVENTS)):
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH Event universe")
    if any("novel" in key.lower() for key in packet):
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH NOVEL data present")

    train = parse_records(packet.get("train", []), "train")
    role = parse_records(packet.get("role_test", []), "role_test")
    if len(train) != POSITIVE_TRIADS or len(role) != ROLE_RECORDS:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH record counts")
    if packet.get("sizes") != {"train": POSITIVE_TRIADS, "role_test": ROLE_RECORDS}:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH declared sizes")

    train_digest = records_digest(train)
    role_digest = records_digest(role)
    if train_digest != EXPECTED_TRAIN_DIGEST:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH TRAIN digest")
    if role_digest != EXPECTED_ROLE_DIGEST:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH ROLE_TEST digest")

    train_pairs = {record.query for record in train}
    role_pairs = {record.query for record in role}
    if (
        len(train_pairs) != POSITIVE_TRIADS
        or len(role_pairs) != ROLE_RECORDS
        or train_pairs & role_pairs
    ):
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH query-pair disjointness")

    derived = alternate_roles(train)
    derived_set = set(derived)
    role_set = set(role)
    if len(derived) != ROLE_RECORDS or len(derived_set) != ROLE_RECORDS:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH derived ROLE_TEST cardinality")
    if derived_set != role_set:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH derived ROLE_TEST equality")

    constraints = packet.get("constraints", {})
    expected_constraints = {
        "train_and_role_query_pairs_disjoint": True,
        "input_pairs_unordered": True,
        "main_training_may_use_train_only": True,
        "role_test_is_evaluation_only": True,
        "novel_test_included": False,
    }
    if constraints != expected_constraints:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH constraints")

    validation = {
        "status": "PASS",
        "content_sha256": actual_hash,
        "event_count": len(universe["ids"]),
        "train_count": len(train),
        "role_test_count": len(role),
        "train_digest": train_digest,
        "role_test_digest": role_digest,
        "all_train_records_three_distinct_events": True,
        "all_role_records_three_distinct_events": True,
        "train_unique_query_pairs": len(train_pairs),
        "role_unique_query_pairs": len(role_pairs),
        "train_and_role_query_pairs_disjoint": True,
        "derived_alternate_role_count": len(derived_set),
        "derived_alternate_roles_equal_role_test": True,
        "novel_test_present": False,
    }
    return validation, train, role


def canonical_triad(values: Iterable[int]) -> tuple[int, int, int]:
    ordered = tuple(sorted(values))
    if len(ordered) != 3 or len(set(ordered)) != 3:
        raise ContractError("triad must contain exactly three distinct Events")
    if not all(0 <= value < N_EVENTS for value in ordered):
        raise ContractError("triad contains an out-of-range Event")
    return ordered


def constituent_pairs(triad: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    x, y, z = triad
    return ((x, y), (x, z), (y, z))


def neighborhood_for(triad: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    negatives: list[tuple[int, int, int]] = []
    excluded = set(triad)
    for pair in constituent_pairs(triad):
        for replacement in range(N_EVENTS):
            if replacement not in excluded:
                negatives.append(canonical_triad((*pair, replacement)))
    if len(negatives) != CORRUPTIONS_PER_POSITIVE:
        raise ContractError("relation neighborhood does not contain 243 corruptions")
    if len(set(negatives)) != CORRUPTIONS_PER_POSITIVE:
        raise ContractError("relation neighborhood corruptions are not unique")
    return tuple(negatives)


def triads_digest(triads: Sequence[tuple[int, int, int]]) -> str:
    return sha256_bytes(compact_json_bytes([list(triad) for triad in triads]))


def build_relation_substrate_from_positives(
    positives: Sequence[tuple[int, int, int]],
) -> tuple[dict[str, Any], tuple[tuple[tuple[int, int, int], ...], ...]]:
    frozen_positives = tuple(canonical_triad(triad) for triad in positives)
    if len(frozen_positives) != POSITIVE_TRIADS:
        raise ContractError("BLOCKED-RELATION-SUBSTRATE: expected 48 positive triads")
    if len(set(frozen_positives)) != POSITIVE_TRIADS:
        raise ContractError("BLOCKED-RELATION-SUBSTRATE: positive triads are not unique")

    all_pairs = [pair for triad in frozen_positives for pair in constituent_pairs(triad)]
    if len(all_pairs) != CONSTITUENT_PAIRS or len(set(all_pairs)) != CONSTITUENT_PAIRS:
        raise ContractError(
            "BLOCKED-RELATION-SUBSTRATE: expected exactly 144 unique constituent pairs"
        )

    positive_set = set(frozen_positives)
    neighborhoods: list[tuple[tuple[int, int, int], ...]] = []
    neighborhood_audits: list[dict[str, Any]] = []
    collision_count = 0
    for index, positive in enumerate(frozen_positives):
        negatives = neighborhood_for(positive)
        collisions = sorted(set(negatives) & positive_set)
        collision_count += len(collisions)
        branch_audits = []
        for branch_index, pair in enumerate(constituent_pairs(positive)):
            start = branch_index * CORRUPTIONS_PER_BRANCH
            stop = start + CORRUPTIONS_PER_BRANCH
            branch = negatives[start:stop]
            branch_audits.append(
                {
                    "preserved_pair": list(pair),
                    "replacement_order": "ascending Event ID outside positive triad",
                    "corruption_count": len(branch),
                    "unique_corruption_count": len(set(branch)),
                    "sha256": triads_digest(branch),
                }
            )
        neighborhood_audits.append(
            {
                "positive_index": index,
                "positive_triad": list(positive),
                "branches": branch_audits,
                "corruption_count": len(negatives),
                "unique_corruption_count": len(set(negatives)),
                "positive_collision_count": len(collisions),
                "sha256": triads_digest(negatives),
            }
        )
        neighborhoods.append(negatives)

    if collision_count:
        raise ContractError(
            "BLOCKED-RELATION-SUBSTRATE: a local corruption collides with an observed positive"
        )

    substrate = {
        "derivation_data": "48 TRAIN records only",
        "positive_order": "TRAIN record order; each triad canonicalized ascending",
        "positive_triads": [list(triad) for triad in frozen_positives],
        "positive_count": len(frozen_positives),
        "unique_positive_count": len(set(frozen_positives)),
        "positive_triads_sha256": triads_digest(frozen_positives),
        "constituent_pair_order": "(x,y), (x,z), (y,z) for canonical x<y<z",
        "constituent_pairs": [list(pair) for pair in all_pairs],
        "constituent_pair_count": len(all_pairs),
        "unique_constituent_pair_count": len(set(all_pairs)),
        "neighborhood_definition": (
            "N(T) is the union of three pair-preserving one-vertex replacement "
            "branches; replacements range over the 81 Events outside T"
        ),
        "branches_per_positive": 3,
        "corruptions_per_branch": CORRUPTIONS_PER_BRANCH,
        "corruptions_per_positive": CORRUPTIONS_PER_POSITIVE,
        "relation_classes_per_positive": RELATION_CLASSES,
        "total_corruption_entries": sum(len(item) for item in neighborhoods),
        "all_neighborhoods_exactly_243_unique": True,
        "positive_collision_count": collision_count,
        "no_positive_collisions": True,
        "neighborhood_audits": neighborhood_audits,
        "role_records_used": False,
        "novel_data_used": False,
    }
    return substrate, tuple(neighborhoods)


def build_relation_substrate(
    train: Sequence[Record],
) -> tuple[
    dict[str, Any],
    tuple[tuple[int, int, int], ...],
    tuple[tuple[tuple[int, int, int], ...], ...],
]:
    if len(train) != POSITIVE_TRIADS:
        raise ContractError("relation substrate did not receive exactly 48 TRAIN records")
    positives = tuple(record.unordered_triad for record in train)
    substrate, neighborhoods = build_relation_substrate_from_positives(positives)
    return substrate, positives, neighborhoods


def implication_proof_audit(
    train: Sequence[Record],
    positives: Sequence[tuple[int, int, int]],
    neighborhoods: Sequence[Sequence[tuple[int, int, int]]],
) -> dict[str, Any]:
    if len(train) != POSITIVE_TRIADS:
        raise ContractError("proof audit did not receive exactly 48 TRAIN records")
    if len(positives) != POSITIVE_TRIADS or len(neighborhoods) != POSITIVE_TRIADS:
        raise ContractError("proof audit received an invalid relation substrate")

    alternate_queries = 0
    alternate_wrong_candidates = 0
    all_presentations = 0
    all_presentation_wrong_candidates = 0
    for index, record in enumerate(train):
        positive = canonical_triad((record.a, record.b, record.target))
        if positive != tuple(positives[index]):
            raise ContractError("proof audit positive order drift")
        negative_set = set(neighborhoods[index])
        a, b, c = record.a, record.b, record.target
        alternate = (
            Record(*sorted((a, c)), b),
            Record(*sorted((b, c)), a),
        )
        presentations = (record, *alternate)
        for presentation_index, presentation in enumerate(presentations):
            if canonical_triad(
                (presentation.a, presentation.b, presentation.target)
            ) != positive:
                raise ContractError("proof audit designated completion is not positive T")
            wrong = [
                candidate
                for candidate in range(N_EVENTS)
                if candidate not in {
                    presentation.a,
                    presentation.b,
                    presentation.target,
                }
            ]
            if len(wrong) != CORRUPTIONS_PER_BRANCH:
                raise ContractError("proof audit wrong-candidate cardinality drift")
            for candidate in wrong:
                corruption = canonical_triad(
                    (presentation.a, presentation.b, candidate)
                )
                if corruption not in negative_set:
                    raise ContractError(
                        "objective-to-consequence proof failed mechanically: "
                        f"positive={index}, presentation={presentation_index}, "
                        f"candidate={candidate}"
                    )
            all_presentations += 1
            all_presentation_wrong_candidates += len(wrong)
            if presentation_index > 0:
                alternate_queries += 1
                alternate_wrong_candidates += len(wrong)

    if alternate_queries != ROLE_RECORDS or alternate_wrong_candidates != 7776:
        raise ContractError("proof audit alternate-role totals drifted")
    if all_presentations != 144 or all_presentation_wrong_candidates != 11664:
        raise ContractError("proof audit all-presentation totals drifted")

    return {
        "status": "PASS",
        "statement": (
            "If, for every observed triad T, its score is strictly greater than "
            "the score of every T' in N(T), then all 96 ROLE_TEST completions "
            "derived from those 48 triads are predicted correctly by pair-to-third argmax."
        ),
        "proof": (
            "Fix T={a,b,c}. For alternate query {a,c} with target b, every legal "
            "wrong candidate d is distinct from a, b, and c, so {a,c,d} is the "
            "pair-preserving corruption in N(T). Strict domination s(T)>s({a,c,d}) "
            "for every one of the 81 wrong d makes b the unique argmax. The same "
            "argument for query {b,c} makes a the unique argmax. Applying both "
            "arguments to each of the 48 TRAIN-derived positive triads proves all "
            "96 derived alternate-role completions. The argument uses only Event "
            "distinctness, candidate exclusion, N(T), and score symmetry."
        ),
        "mechanical_check_data": "TRAIN only",
        "role_record_argument_or_loader": False,
        "alternate_queries_checked": alternate_queries,
        "wrong_legal_candidates_per_alternate_query": CORRUPTIONS_PER_BRANCH,
        "alternate_wrong_candidate_memberships_checked": alternate_wrong_candidates,
        "all_three_presentations_checked": all_presentations,
        "all_presentation_wrong_candidate_memberships_checked": (
            all_presentation_wrong_candidates
        ),
        "every_designated_completion_maps_to_positive": True,
        "every_wrong_completion_maps_to_its_positive_N_T": True,
        "strict_domination_forces_unique_argmax": True,
    }


def make_batch(records: Sequence[Record]) -> Batch:
    frozen = tuple(records)
    pair_a = torch.tensor([record.a for record in frozen], dtype=torch.long)
    pair_b = torch.tensor([record.b for record in frozen], dtype=torch.long)
    all_ids = torch.arange(N_EVENTS, dtype=torch.long).expand(len(frozen), -1)
    legal = (all_ids != pair_a[:, None]) & (all_ids != pair_b[:, None])
    candidates = all_ids[legal].reshape(len(frozen), LEGAL_CANDIDATES)
    targets = torch.tensor([record.target for record in frozen], dtype=torch.long)
    matches = candidates == targets[:, None]
    if not bool(torch.all(matches.sum(dim=1) == 1)):
        raise ContractError("target does not occur exactly once among legal candidates")
    target_classes = matches.to(torch.int64).argmax(dim=1)
    return Batch(frozen, pair_a, pair_b, candidates, target_classes)


def query_logits(model: SharedScorer, batch: Batch) -> Tensor:
    """The exact accepted 015.04 triad-symmetric designated-query path."""
    rows, classes = batch.candidates.shape
    pair_a = batch.pair_a[:, None].expand(-1, classes).reshape(-1)
    pair_b = batch.pair_b[:, None].expand(-1, classes).reshape(-1)
    candidate = batch.candidates.reshape(-1)
    values = model.triad_score(pair_a, pair_b, candidate)
    return values.reshape(rows, classes)


def make_relation_batch(
    positives: Sequence[tuple[int, int, int]],
    neighborhoods: Sequence[Sequence[tuple[int, int, int]]],
) -> RelationBatch:
    frozen_positives = tuple(positives)
    if len(frozen_positives) != POSITIVE_TRIADS or len(neighborhoods) != POSITIVE_TRIADS:
        raise ContractError("invalid relation batch cardinality")
    rows = []
    for positive, negatives in zip(frozen_positives, neighborhoods, strict=True):
        if len(negatives) != CORRUPTIONS_PER_POSITIVE:
            raise ContractError("invalid relation neighborhood cardinality")
        rows.append((positive, *negatives))
    candidates = torch.tensor(rows, dtype=torch.long)
    if candidates.shape != (POSITIVE_TRIADS, RELATION_CLASSES, 3):
        raise ContractError("invalid relation candidate tensor shape")
    targets = torch.zeros(POSITIVE_TRIADS, dtype=torch.long)
    return RelationBatch(frozen_positives, candidates, targets)


def relation_logits(model: SharedScorer, batch: RelationBatch) -> Tensor:
    rows, classes, width = batch.candidates.shape
    if width != 3:
        raise ContractError("relation candidates are not triads")
    flattened = batch.candidates.reshape(-1, 3)
    values = model.triad_score(
        flattened[:, 0],
        flattened[:, 1],
        flattened[:, 2],
    )
    return values.reshape(rows, classes) / RELATION_TEMPERATURE


def make_model(seed: int) -> SharedScorer:
    torch.manual_seed(seed)
    return SharedScorer().to(device="cpu", dtype=torch.float32)


def tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    tensors = []
    for name, parameter in model.named_parameters():
        tensors.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "count": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "initialization": (
                    "xavier_uniform_gain_1.0" if name.endswith("weight") else "zeros"
                ),
            }
        )
    return {
        "tensors": tensors,
        "tensor_count": len(tensors),
        "trainable_scalar_count": sum(item["count"] for item in tensors),
        "event_tables": 1,
        "candidate_specific_bias": False,
        "role_specific_event_table": False,
        "block_indexed_parameter": False,
        "pair_indexed_lookup": False,
        "positional_role_label": False,
    }


def verify_matched_initialization(seeds: Iterable[int]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for seed in seeds:
        query_model = make_model(seed)
        relation_model = make_model(seed)
        exact = all(
            torch.equal(left, right)
            for left, right in zip(
                query_model.state_dict().values(),
                relation_model.state_dict().values(),
                strict=True,
            )
        )
        query_hash = state_hash(query_model)
        relation_hash = state_hash(relation_model)
        if not exact or query_hash != relation_hash:
            raise ContractError(f"matched initialization failed for seed {seed}")
        results[str(seed)] = {
            "byte_identical": True,
            "query_ce_sha256": query_hash,
            "relational_competition_sha256": relation_hash,
        }
    return results


def audit_triad_permutation_identity(chunk_size: int = 4096) -> dict[str, Any]:
    """Evaluate every distinct Event triple under all six input permutations."""
    model = make_model(0)
    triples = list(itertools.combinations(range(N_EVENTS), 3))
    permutations = tuple(itertools.permutations((0, 1, 2)))
    compared = 0
    with torch.inference_mode():
        for start in range(0, len(triples), chunk_size):
            chunk = triples[start : start + chunk_size]
            columns = [
                torch.tensor([triple[index] for triple in chunk], dtype=torch.long)
                for index in range(3)
            ]
            reference = model.triad_score(columns[0], columns[1], columns[2])
            for permutation in permutations:
                observed = model.triad_score(
                    columns[permutation[0]],
                    columns[permutation[1]],
                    columns[permutation[2]],
                )
                if not torch.equal(reference, observed):
                    raise ContractError(
                        "triad permutation identity failed at triple offset "
                        f"{start}, permutation {permutation}"
                    )
                compared += len(chunk)
    if len(triples) != math.comb(N_EVENTS, 3) or len(triples) != 95284:
        raise ContractError("unordered Event triple cardinality drift")
    return {
        "status": "PASS",
        "seed": 0,
        "unordered_distinct_triples": len(triples),
        "permutations_per_triple": len(permutations),
        "bit_identical_score_comparisons": compared,
        "all_six_permutations_bit_identical": True,
        "canonicalization": "ascending Event ID",
        "decomposition_accumulation_order": (
            "((g(x,y;z) + g(x,z;y)) + g(y,z;x)) / 3.0 for x<y<z"
        ),
        "dtype": "torch.float32",
        "device": "cpu",
    }


def exact_accuracy(score_matrix: Tensor, target_classes: Tensor) -> float:
    predictions = score_matrix.argmax(dim=1)
    return float((predictions == target_classes).to(torch.float64).mean().item())


def ordinary_median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def parameter_norms(model: nn.Module) -> dict[str, Any]:
    per_tensor: dict[str, float] = {}
    total_squared = 0.0
    for name, parameter in model.named_parameters():
        squared = float(torch.sum(parameter.detach().double() ** 2).item())
        per_tensor[name] = math.sqrt(squared)
        total_squared += squared
    return {"per_tensor_l2": per_tensor, "total_l2": math.sqrt(total_squared)}


def embedding_diagnostics(model: SharedScorer) -> dict[str, float]:
    embedding = model.event.weight.detach().double()
    singular = torch.linalg.svdvals(embedding)
    energy = singular.square()
    probabilities = energy / energy.sum()
    positive = probabilities[probabilities > 0]
    effective_rank = torch.exp(-(positive * torch.log(positive)).sum())
    return {
        "frobenius_norm": float(torch.linalg.vector_norm(embedding).item()),
        "effective_rank": float(effective_rank.item()),
    }


def score_definitions() -> dict[str, Any]:
    return {
        "base": (
            "p=concat(E[a]+E[b], E[a]*E[b], abs(E[a]-E[b]), E[c]); "
            "g(a,b;c)=Linear(256,256)->GELU(exact)->Linear(256,1)"
        ),
        "triad_symmetric": (
            "for canonical x<y<z, s({x,y,z})="
            "((g(x,y;z)+g(x,z;y))+g(y,z;x))/3"
        ),
        "base_pair_swap_invariant": True,
        "triad_role_permutation_invariant_by_constitution": True,
        "shared_by_both_arms": True,
        "only_primary_arm_difference": "TRAIN-only objective",
    }


def objective_definitions() -> dict[str, Any]:
    return {
        "query_ce": {
            "definition": (
                "mean of 48 designated-query cross-entropies; each row scores "
                "the target and 81 other legal candidates with s({a,b,c})"
            ),
            "rows": POSITIVE_TRIADS,
            "classes_per_row": LEGAL_CANDIDATES,
            "positive_class": "designated TRAIN target position",
            "role_augmentation": False,
            "backpropagated_diagnostics": ["query_cross_entropy"],
        },
        "relational_competition": {
            "definition": (
                "mean of 48 relation-level cross-entropies; each row is "
                "[s(T), s(T'_1), ..., s(T'_243)] for frozen N(T)"
            ),
            "rows": POSITIVE_TRIADS,
            "classes_per_row": RELATION_CLASSES,
            "positive_class": 0,
            "temperature": RELATION_TEMPERATURE,
            "margin": None,
            "weights": None,
            "sampling": None,
            "hard_negative_mining": False,
            "role_records_used": False,
            "backpropagated_diagnostics": ["relation_cross_entropy"],
        },
    }


def temporal_definitions() -> dict[str, Any]:
    return {
        "t_fit": "first scheduled checkpoint with designated-query accuracy == 1.0",
        "t_relation95": (
            "first scheduled checkpoint with >=46/48 strict local relation "
            "dominations at that checkpoint and the next two scheduled checkpoints"
        ),
        "t_relation_exact": (
            "first scheduled checkpoint with 48/48 strict local relation "
            "dominations at that checkpoint and the next two scheduled checkpoints"
        ),
        "t_role95": (
            "first scheduled checkpoint with ROLE_TEST accuracy >=0.95 at that "
            "checkpoint and the next two scheduled checkpoints"
        ),
        "t_role_exact": (
            "first scheduled checkpoint with ROLE_TEST accuracy ==1.0 at that "
            "checkpoint and the next two scheduled checkpoints"
        ),
        "smoothing": False,
        "never_reached": None,
    }


def interpretation_bins() -> dict[str, str]:
    return {
        "C0": "BLOCKED-BASELINE-REPLICATION -> no causal interpretation",
        "C1": (
            "R reaches sustained t_relation95 in <7/8 seeds -> the tested "
            "scorer/optimizer does not reliably realize the relational competition "
            "objective; optimization/capacity boundary"
        ),
        "C2": (
            "R reaches sustained t_relation95 in >=7/8 seeds but sustained "
            "t_role95 in <7/8 -> contradiction with the proved objective-to-"
            "consequence implication; treat as implementation/audit failure, not "
            "a scientific result"
        ),
        "C3": (
            "R reaches sustained t_role95 in >=7/8 seeds while replicated Q "
            "remains <7/8 -> role-neutral competitive relation learning is "
            "sufficient, relative to designated-query CE, for robust seen-triad consequence"
        ),
        "C4": (
            "both Q and R reach sustained t_role95 in >=7/8 seeds -> ROLE succeeds "
            "but the objective contrast is not isolated"
        ),
        "C5": (
            "Q reaches sustained t_role95 in >=7/8 seeds while R does not -> "
            "predecessor baseline behavior changed or relational competition is "
            "harmful; inspect without positive reinterpretation"
        ),
    }


def interpretation_precedence() -> list[str]:
    return [
        "C0 first if exact baseline replication fails",
        "C4 if Q and R both robustly reach t_role95",
        "C5 if Q alone robustly reaches t_role95",
        "C3 if R alone robustly reaches t_role95",
        "C1 if neither reaches robust t_role95 and R reaches t_relation95 in <7/8",
        "C2 otherwise (R relation95 robust but ROLE95 not robust)",
    ]


def scientific_fences() -> list[str]:
    return [
        "No discovery or evaluation of the global 56-block relation.",
        "No NOVEL blocks or NOVEL metric.",
        "No SFP/Fano/habitat coordinates.",
        "No claim for a preferred, unique, optimal, or necessary OT learning law.",
        "No claim that Event embeddings contain a unique readable algebra.",
        "No necessity claim for relational competition or symmetry.",
        "No claim that every consequence-learning problem reduces to permutation invariance.",
        "No language-scale, physical Event, or physical Outcome semantics claim.",
        "No claim that R discovers its local alternatives spontaneously; N(T) is licensed TRAIN-derived structure.",
    ]


def validate_predecessor_trajectory(raw_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    actual_hash = sha256_bytes(raw_bytes)
    if actual_hash != PREDECESSOR_TRAJECTORY_SHA256:
        raise ContractError(
            "BLOCKED-BASELINE-REFERENCE-MISMATCH content hash "
            f"{actual_hash} != {PREDECESSOR_TRAJECTORY_SHA256}"
        )
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH invalid JSON") from error
    if document.get("schema") != "occurrence.gpt.01504.trajectory.v1":
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH schema")
    if document.get("arm") != "triad_symmetric":
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH arm")
    if document.get("seeds") != list(SCORED_SEEDS):
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH seeds")
    if document.get("horizon_updates") != HORIZON:
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH horizon")
    if document.get("checkpoint_schedule") != list(CHECKPOINTS):
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH checkpoints")
    if document.get("novel_metrics_computed") is not False:
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH NOVEL audit")
    runs = document.get("runs")
    if not isinstance(runs, list) or [run.get("seed") for run in runs] != list(SCORED_SEEDS):
        raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH run order")
    for run in runs:
        if run.get("optimizer") != OPTIMIZER:
            raise ContractError(
                f"BLOCKED-BASELINE-REFERENCE-MISMATCH optimizer seed={run.get('seed')}"
            )
        if run.get("updates_completed") != HORIZON or run.get("early_stopped") is not False:
            raise ContractError(
                f"BLOCKED-BASELINE-REFERENCE-MISMATCH execution seed={run.get('seed')}"
            )
        checkpoints = run.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ContractError("BLOCKED-BASELINE-REFERENCE-MISMATCH checkpoint type")
        if [item.get("update") for item in checkpoints] != list(CHECKPOINTS):
            raise ContractError(
                f"BLOCKED-BASELINE-REFERENCE-MISMATCH schedule seed={run.get('seed')}"
            )
    audit = {
        "status": "PASS",
        "content_sha256": actual_hash,
        "schema": document["schema"],
        "arm": document["arm"],
        "seed_count": len(runs),
        "seeds": document["seeds"],
        "horizon_updates": document["horizon_updates"],
        "checkpoint_count_per_run": len(CHECKPOINTS),
        "checkpoint_schedule_exact": True,
        "optimizer_exact": True,
        "all_runs_completed_horizon": True,
        "novel_metrics_computed": False,
    }
    return audit, document


def build_precommit(
    validation: dict[str, Any],
    substrate: dict[str, Any],
    proof_audit: dict[str, Any],
    initialization: dict[str, Any],
    permutation_audit: dict[str, Any],
    baseline_audit: dict[str, Any],
    threads: int,
    base_revision: str,
    base_entry_count: int,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.precommit.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "status": "PRECOMMITTED-BEFORE-CURRENT-ROLE-EVALUATION",
        "pinned_inputs": pinned_inputs(),
        "benchmark_validation": validation,
        "accepted_baseline_trajectory_validation": baseline_audit,
        "environment": environment_record(threads),
        "deterministic_settings": {
            "pythonhashseed": "0",
            "python_random_seeded": True,
            "numpy_random_seeded": True,
            "torch_seed_per_run": True,
            "torch_use_deterministic_algorithms": True,
            "device": "cpu",
            "dtype": "float32",
            "thread_count_frozen": threads,
        },
        "runner": {
            "path": (
                "issues/015-learning-law-for-consequence-structure/"
                "015.07-Code-attachments/run_015_competitive.py"
            ),
            "content_sha256": sha256_file(Path(__file__)),
        },
        "parameter_inventory": parameter_inventory(make_model(0)),
        "matched_initialization_audit": initialization,
        "score_definitions": score_definitions(),
        "triad_permutation_identity_audit": permutation_audit,
        "candidate_mask": {
            "universe_classes": N_EVENTS,
            "masked": ["query Event a", "query Event b"],
            "legal_classes_per_query": LEGAL_CANDIDATES,
            "identical_between_arms": True,
        },
        "relation_substrate": substrate,
        "objective_to_consequence_implication": proof_audit,
        "objective_definitions": objective_definitions(),
        "optimizer": OPTIMIZER,
        "hyperparameter_search": False,
        "scored_plan": {
            "arms": list(ARMS),
            "seeds": list(SCORED_SEEDS),
            "horizon_updates": HORIZON,
            "early_stopping": False,
            "extra_seeds": False,
            "outcome_dependent_reruns": False,
            "checkpoints": list(CHECKPOINTS),
            "matched_initialization": True,
            "shared_scorer": True,
            "optimizer_identical": True,
        },
        "checkpoint_metrics": [
            "designated-query cross-entropy and accuracy",
            "relation-level cross-entropy",
            "strict relation domination fraction/count",
            "minimum/median/mean local-relation margin",
            "ROLE_TEST accuracy/count/exact 96/96 boolean",
            "fraction/count of 48 triads with all three completions correct",
            "minimum/median/mean ROLE_TEST completion margin",
            "per-tensor and total parameter L2 norms",
            "Event embedding Frobenius norm and entropy effective rank",
        ],
        "metric_definitions": {
            "strict_relation_domination": "s(T) > max_{T' in N(T)} s(T')",
            "local_relation_margin": "s(T) minus max corruption score in N(T)",
            "role_margin": "correct score minus maximum other legal-candidate score",
            "margin_median": "ordinary sample median (mean of the middle two for even n)",
            "embedding_effective_rank": (
                "exp(-sum_i p_i log p_i)), p_i=s_i^2/sum_j s_j^2 for the "
                "64 singular values of the 84x64 Event embedding"
            ),
        },
        "temporal_definitions": temporal_definitions(),
        "baseline_replication_policy": BASELINE_POLICY,
        "interpretation_bins": interpretation_bins(),
        "interpretation_precedence": interpretation_precedence(),
        "scientific_fences": scientific_fences(),
        "role_metrics_computed_before_this_precommit": False,
        "publication": {
            "path": (
                "issues/015-learning-law-for-consequence-structure/"
                "015.07-Code-attachments/precommit.json"
            ),
            "publish_against": "latest",
            "workflow": "occurrence",
            "base_revision_observed": base_revision,
            "base_entry_count_observed": base_entry_count,
            "expected_manifest_delta": 1,
            "expected_entry_count_after_publish": base_entry_count + 1,
            "immutable_after_role_metrics": True,
            "final_bundle_additional_entries": 5,
        },
    }


def validate_frozen_precommit(
    precommit: dict[str, Any],
    threads: int,
    *,
    require_current_environment: bool = True,
) -> None:
    expected_top_level = {
        "schema",
        "created_at_utc",
        "information_exposure",
        "status",
        "pinned_inputs",
        "benchmark_validation",
        "accepted_baseline_trajectory_validation",
        "environment",
        "deterministic_settings",
        "runner",
        "parameter_inventory",
        "matched_initialization_audit",
        "score_definitions",
        "triad_permutation_identity_audit",
        "candidate_mask",
        "relation_substrate",
        "objective_to_consequence_implication",
        "objective_definitions",
        "optimizer",
        "hyperparameter_search",
        "scored_plan",
        "checkpoint_metrics",
        "metric_definitions",
        "temporal_definitions",
        "baseline_replication_policy",
        "interpretation_bins",
        "interpretation_precedence",
        "scientific_fences",
        "role_metrics_computed_before_this_precommit",
        "publication",
    }
    if set(precommit) != expected_top_level:
        raise ContractError("precommit top-level fields drift")
    if precommit.get("schema") != f"{SCHEMA_PREFIX}.precommit.v1":
        raise ContractError("invalid precommit schema")
    if precommit.get("status") != "PRECOMMITTED-BEFORE-CURRENT-ROLE-EVALUATION":
        raise ContractError("invalid precommit status")
    if not isinstance(precommit.get("created_at_utc"), str):
        raise ContractError("invalid precommit creation timestamp")
    if precommit.get("information_exposure") != INFORMATION_EXPOSURE:
        raise ContractError("precommit information exposure drift")
    if precommit.get("pinned_inputs") != pinned_inputs():
        raise ContractError("precommit pinned-input drift")

    if type(threads) is not int or threads < 1:
        raise ContractError("invalid frozen thread count")
    expected_determinism = {
        "pythonhashseed": "0",
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_seed_per_run": True,
        "torch_use_deterministic_algorithms": True,
        "device": "cpu",
        "dtype": "float32",
        "thread_count_frozen": threads,
    }
    if precommit.get("deterministic_settings") != expected_determinism:
        raise ContractError("precommit deterministic-settings drift")
    if require_current_environment and precommit.get("environment") != environment_record(
        threads
    ):
        raise ContractError("runtime environment differs from frozen precommit")

    expected_runner = {
        "path": (
            "issues/015-learning-law-for-consequence-structure/"
            "015.07-Code-attachments/run_015_competitive.py"
        ),
        "content_sha256": sha256_file(Path(__file__)),
    }
    if precommit.get("runner") != expected_runner:
        raise ContractError("runner changed after precommit")
    if precommit.get("parameter_inventory") != parameter_inventory(make_model(0)):
        raise ContractError("precommit parameter inventory drift")
    if precommit.get("matched_initialization_audit") != verify_matched_initialization(
        SCORED_SEEDS
    ):
        raise ContractError("precommit matched-initialization audit drift")
    if precommit.get("score_definitions") != score_definitions():
        raise ContractError("precommit score-definition drift")

    expected_permutation = {
        "status": "PASS",
        "seed": 0,
        "unordered_distinct_triples": 95284,
        "permutations_per_triple": 6,
        "bit_identical_score_comparisons": 571704,
        "all_six_permutations_bit_identical": True,
        "canonicalization": "ascending Event ID",
        "decomposition_accumulation_order": (
            "((g(x,y;z) + g(x,z;y)) + g(y,z;x)) / 3.0 for x<y<z"
        ),
        "dtype": "torch.float32",
        "device": "cpu",
    }
    if precommit.get("triad_permutation_identity_audit") != expected_permutation:
        raise ContractError("precommit permutation identity audit drift")
    expected_mask = {
        "universe_classes": N_EVENTS,
        "masked": ["query Event a", "query Event b"],
        "legal_classes_per_query": LEGAL_CANDIDATES,
        "identical_between_arms": True,
    }
    if precommit.get("candidate_mask") != expected_mask:
        raise ContractError("precommit candidate-mask drift")

    expected_benchmark = {
        "status": "PASS",
        "content_sha256": BENCHMARK_SHA256,
        "event_count": N_EVENTS,
        "train_count": POSITIVE_TRIADS,
        "role_test_count": ROLE_RECORDS,
        "train_digest": EXPECTED_TRAIN_DIGEST,
        "role_test_digest": EXPECTED_ROLE_DIGEST,
        "all_train_records_three_distinct_events": True,
        "all_role_records_three_distinct_events": True,
        "train_unique_query_pairs": POSITIVE_TRIADS,
        "role_unique_query_pairs": ROLE_RECORDS,
        "train_and_role_query_pairs_disjoint": True,
        "derived_alternate_role_count": ROLE_RECORDS,
        "derived_alternate_roles_equal_role_test": True,
        "novel_test_present": False,
    }
    if precommit.get("benchmark_validation") != expected_benchmark:
        raise ContractError("precommit benchmark-validation drift")
    expected_baseline = {
        "status": "PASS",
        "content_sha256": PREDECESSOR_TRAJECTORY_SHA256,
        "schema": "occurrence.gpt.01504.trajectory.v1",
        "arm": "triad_symmetric",
        "seed_count": len(SCORED_SEEDS),
        "seeds": list(SCORED_SEEDS),
        "horizon_updates": HORIZON,
        "checkpoint_count_per_run": len(CHECKPOINTS),
        "checkpoint_schedule_exact": True,
        "optimizer_exact": True,
        "all_runs_completed_horizon": True,
        "novel_metrics_computed": False,
    }
    if precommit.get("accepted_baseline_trajectory_validation") != expected_baseline:
        raise ContractError("precommit baseline-reference validation drift")

    positives = precommit.get("relation_substrate", {}).get("positive_triads", [])
    if not isinstance(positives, list):
        raise ContractError("precommit positive-triad encoding drift")
    reconstructed, neighborhoods = build_relation_substrate_from_positives(
        tuple(tuple(item) for item in positives)
    )
    if reconstructed != precommit.get("relation_substrate"):
        raise ContractError("precommit relation substrate is not mechanically reproducible")
    synthetic_train = tuple(Record(*triad) for triad in reconstructed["positive_triads"])
    expected_proof = implication_proof_audit(
        synthetic_train,
        tuple(tuple(item) for item in reconstructed["positive_triads"]),
        neighborhoods,
    )
    if precommit.get("objective_to_consequence_implication") != expected_proof:
        raise ContractError("precommit implication proof audit drift")

    if precommit.get("optimizer") != OPTIMIZER:
        raise ContractError("precommit optimizer drift")
    if precommit.get("hyperparameter_search") is not False:
        raise ContractError("precommit permits hyperparameter search")
    expected_plan = {
        "arms": list(ARMS),
        "seeds": list(SCORED_SEEDS),
        "horizon_updates": HORIZON,
        "early_stopping": False,
        "extra_seeds": False,
        "outcome_dependent_reruns": False,
        "checkpoints": list(CHECKPOINTS),
        "matched_initialization": True,
        "shared_scorer": True,
        "optimizer_identical": True,
    }
    if precommit.get("scored_plan") != expected_plan:
        raise ContractError("precommit scored-plan drift")
    if precommit.get("objective_definitions") != objective_definitions():
        raise ContractError("precommit objective-definition drift")
    expected_checkpoint_metrics = [
        "designated-query cross-entropy and accuracy",
        "relation-level cross-entropy",
        "strict relation domination fraction/count",
        "minimum/median/mean local-relation margin",
        "ROLE_TEST accuracy/count/exact 96/96 boolean",
        "fraction/count of 48 triads with all three completions correct",
        "minimum/median/mean ROLE_TEST completion margin",
        "per-tensor and total parameter L2 norms",
        "Event embedding Frobenius norm and entropy effective rank",
    ]
    if precommit.get("checkpoint_metrics") != expected_checkpoint_metrics:
        raise ContractError("precommit checkpoint-metric drift")
    expected_metric_definitions = {
        "strict_relation_domination": "s(T) > max_{T' in N(T)} s(T')",
        "local_relation_margin": "s(T) minus max corruption score in N(T)",
        "role_margin": "correct score minus maximum other legal-candidate score",
        "margin_median": "ordinary sample median (mean of the middle two for even n)",
        "embedding_effective_rank": (
            "exp(-sum_i p_i log p_i)), p_i=s_i^2/sum_j s_j^2 for the "
            "64 singular values of the 84x64 Event embedding"
        ),
    }
    if precommit.get("metric_definitions") != expected_metric_definitions:
        raise ContractError("precommit metric-definition drift")
    if precommit.get("temporal_definitions") != temporal_definitions():
        raise ContractError("precommit temporal-definition drift")
    if precommit.get("baseline_replication_policy") != BASELINE_POLICY:
        raise ContractError("precommit baseline-policy drift")
    if precommit.get("interpretation_bins") != interpretation_bins():
        raise ContractError("precommit interpretation-bin drift")
    if precommit.get("interpretation_precedence") != interpretation_precedence():
        raise ContractError("precommit interpretation-precedence drift")
    if precommit.get("scientific_fences") != scientific_fences():
        raise ContractError("precommit scientific-fence drift")
    if precommit.get("role_metrics_computed_before_this_precommit") is not False:
        raise ContractError("precommit does not certify ROLE metric ordering")

    publication = precommit.get("publication")
    if not isinstance(publication, dict):
        raise ContractError("precommit publication record missing")
    base_revision = publication.get("base_revision_observed")
    base_count = publication.get("base_entry_count_observed")
    require_sha256(base_revision, "precommit publication base revision")
    if type(base_count) is not int or base_count < 0:
        raise ContractError("precommit publication base count is invalid")
    expected_publication = {
        "path": (
            "issues/015-learning-law-for-consequence-structure/"
            "015.07-Code-attachments/precommit.json"
        ),
        "publish_against": "latest",
        "workflow": "occurrence",
        "base_revision_observed": base_revision,
        "base_entry_count_observed": base_count,
        "expected_manifest_delta": 1,
        "expected_entry_count_after_publish": base_count + 1,
        "immutable_after_role_metrics": True,
        "final_bundle_additional_entries": 5,
    }
    if publication != expected_publication:
        raise ContractError("precommit publication contract drift")


def prepare(args: argparse.Namespace) -> int:
    configure_determinism(args.threads)
    require_sha256(args.base_revision, "base revision")
    if args.base_entry_count < 0:
        raise ContractError("base entry count must be nonnegative")

    raw_benchmark = read_source(args.benchmark_source)
    validation, train, _role_identity_only = validate_benchmark(raw_benchmark)
    substrate, positives, neighborhoods = build_relation_substrate(train)
    proof_audit = implication_proof_audit(train, positives, neighborhoods)

    raw_baseline = read_source(args.predecessor_trajectory_source)
    baseline_audit, _baseline = validate_predecessor_trajectory(raw_baseline)

    initialization = verify_matched_initialization(SCORED_SEEDS)
    print("auditing all scorer Event permutations", flush=True)
    permutation_audit = audit_triad_permutation_identity()
    precommit = build_precommit(
        validation,
        substrate,
        proof_audit,
        initialization,
        permutation_audit,
        baseline_audit,
        args.threads,
        args.base_revision,
        args.base_entry_count,
    )

    destination = args.out_dir / PRECOMMIT_NAME
    scored_outputs = [
        args.out_dir / QUERY_NAME,
        args.out_dir / RELATION_NAME,
        args.out_dir / ANALYSIS_NAME,
        args.out_dir.parent / DEFAULT_RESULT_NAME,
    ]
    if any(path.exists() for path in scored_outputs):
        raise ContractError("refusing prepare because scored outputs already exist")
    if destination.exists() and not args.force:
        raise ContractError(f"refusing to overwrite existing {destination}")
    write_json(destination, precommit)
    print(
        json.dumps(
            {
                "precommit": str(destination),
                "sha256": sha256_file(destination),
                "positive_triads": substrate["positive_count"],
                "corruptions_per_positive": substrate["corruptions_per_positive"],
                "role_metrics_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_checkpoint(
    model: SharedScorer,
    query_batch: Batch,
    role_batch: Batch,
    all_roles_batch: Batch,
    relation_batch: RelationBatch,
    update: int,
) -> dict[str, Any]:
    """Evaluate without mutation; shared metrics preserve 015.04 operation order."""
    model.eval()
    with torch.inference_mode():
        query_scores = query_logits(model, query_batch)
        role_scores = query_logits(model, role_batch)
        all_role_scores = query_logits(model, all_roles_batch)
        query_loss = float(F.cross_entropy(query_scores, query_batch.target_classes).item())
        query_accuracy = exact_accuracy(query_scores, query_batch.target_classes)
        role_predictions = role_scores.argmax(dim=1)
        role_hits = role_predictions == role_batch.target_classes
        role_accuracy = float(role_hits.to(torch.float64).mean().item())
        all_role_hits = (
            all_role_scores.argmax(dim=1) == all_roles_batch.target_classes
        ).reshape(POSITIVE_TRIADS, 3)
        triad_exact_count = int(torch.all(all_role_hits, dim=1).sum().item())

        row_ids = torch.arange(len(role_batch.records), dtype=torch.long)
        correct_scores = role_scores[row_ids, role_batch.target_classes]
        other_scores = role_scores.clone()
        other_scores[row_ids, role_batch.target_classes] = -torch.inf
        role_margins = (
            correct_scores - other_scores.max(dim=1).values
        ).double().tolist()

        relation_scores = relation_logits(model, relation_batch)
        relation_loss = float(
            F.cross_entropy(relation_scores, relation_batch.target_classes).item()
        )
        positive_scores = relation_scores[:, 0]
        maximum_corruptions = relation_scores[:, 1:].max(dim=1).values
        relation_margins = (positive_scores - maximum_corruptions).double().tolist()
        domination = positive_scores > maximum_corruptions
        domination_count = int(domination.sum().item())

    return {
        "update": update,
        "query_cross_entropy": query_loss,
        "query_accuracy": query_accuracy,
        "relation_cross_entropy": relation_loss,
        "strict_relation_domination_count": domination_count,
        "strict_relation_domination_total": POSITIVE_TRIADS,
        "strict_relation_domination_fraction": domination_count / POSITIVE_TRIADS,
        "strict_relation_domination_exact_48_of_48": domination_count == POSITIVE_TRIADS,
        "local_relation_margin": {
            "minimum": float(min(relation_margins)),
            "median": ordinary_median(relation_margins),
            "mean": float(statistics.fmean(relation_margins)),
        },
        "role_test_accuracy": role_accuracy,
        "role_test_correct": int(role_hits.sum().item()),
        "role_test_total": len(role_batch.records),
        "role_test_exact_96_of_96": bool(torch.all(role_hits).item()),
        "triads_all_three_roles_correct": triad_exact_count,
        "triads_total": POSITIVE_TRIADS,
        "fraction_triads_all_three_roles_correct": triad_exact_count / POSITIVE_TRIADS,
        "all_48_triads_exact": triad_exact_count == POSITIVE_TRIADS,
        "role_completion_margin": {
            "minimum": float(min(role_margins)),
            "median": ordinary_median(role_margins),
            "mean": float(statistics.fmean(role_margins)),
        },
        "parameter_norms": parameter_norms(model),
        "event_embedding": embedding_diagnostics(model),
    }


def training_objective(
    model: SharedScorer,
    arm: str,
    query_batch: Batch,
    relation_batch: RelationBatch,
) -> Tensor:
    """Return the sole backpropagated TRAIN-only objective for one arm."""
    if arm == "query_ce":
        return F.cross_entropy(query_logits(model, query_batch), query_batch.target_classes)
    if arm == "relational_competition":
        return F.cross_entropy(
            relation_logits(model, relation_batch), relation_batch.target_classes
        )
    raise ContractError(f"unknown arm {arm}")


def run_trajectory(
    model: SharedScorer,
    arm: str,
    train: tuple[Record, ...],
    role: tuple[Record, ...],
    positives: tuple[tuple[int, int, int], ...],
    neighborhoods: tuple[tuple[tuple[int, int, int], ...], ...],
    seed: int,
) -> dict[str, Any]:
    query_batch = make_batch(train)
    role_batch = make_batch(role)
    all_roles_batch = make_batch(all_role_completions(train))
    relation_batch = make_relation_batch(positives, neighborhoods)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=OPTIMIZER["learning_rate"],
        weight_decay=OPTIMIZER["weight_decay"],
        betas=tuple(OPTIMIZER["betas"]),
        eps=OPTIMIZER["eps"],
    )
    checkpoints: list[dict[str, Any]] = []
    checkpoint_set = set(CHECKPOINTS)
    checkpoints.append(
        evaluate_checkpoint(
            model,
            query_batch,
            role_batch,
            all_roles_batch,
            relation_batch,
            0,
        )
    )
    for update in range(1, HORIZON + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = training_objective(model, arm, query_batch, relation_batch)
        if not bool(torch.isfinite(loss)):
            raise ContractError(
                f"nonfinite scored loss: arm={arm}, seed={seed}, update={update}"
            )
        loss.backward()
        optimizer.step()
        if update in checkpoint_set:
            checkpoints.append(
                evaluate_checkpoint(
                    model,
                    query_batch,
                    role_batch,
                    all_roles_batch,
                    relation_batch,
                    update,
                )
            )
    if [item["update"] for item in checkpoints] != list(CHECKPOINTS):
        raise ContractError("trajectory checkpoint schedule drift")
    return {
        "seed": seed,
        "initialization_sha256": None,
        "optimizer": OPTIMIZER,
        "objective": objective_definitions()[arm],
        "updates_completed": HORIZON,
        "early_stopped": False,
        "checkpoints": checkpoints,
    }


def first_sustained(
    checkpoints: Sequence[dict[str, Any]], predicate: Any
) -> int | None:
    for index in range(len(checkpoints) - 2):
        window = checkpoints[index : index + 3]
        if all(predicate(item) for item in window):
            return int(checkpoints[index]["update"])
    return None


def temporal_quantities(checkpoints: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    t_fit = next(
        (
            int(item["update"])
            for item in checkpoints
            if item["query_accuracy"] == 1.0
        ),
        None,
    )
    return {
        "t_fit": t_fit,
        "t_relation95": first_sustained(
            checkpoints,
            lambda item: item["strict_relation_domination_count"] >= 46,
        ),
        "t_relation_exact": first_sustained(
            checkpoints,
            lambda item: item["strict_relation_domination_count"] == POSITIVE_TRIADS,
        ),
        "t_role95": first_sustained(
            checkpoints, lambda item: item["role_test_accuracy"] >= 0.95
        ),
        "t_role_exact": first_sustained(
            checkpoints, lambda item: item["role_test_accuracy"] == 1.0
        ),
    }


def predecessor_temporal_quantities(
    checkpoints: Sequence[dict[str, Any]],
) -> dict[str, int | None]:
    t_fit = next(
        (
            int(item["update"])
            for item in checkpoints
            if item["train_accuracy"] == 1.0
        ),
        None,
    )
    return {
        "t_fit": t_fit,
        "t_role95": first_sustained(
            checkpoints, lambda item: item["role_test_accuracy"] >= 0.95
        ),
        "t_role_exact": first_sustained(
            checkpoints, lambda item: item["role_test_accuracy"] == 1.0
        ),
        "t_triad_exact": first_sustained(
            checkpoints, lambda item: item["all_48_triads_exact"]
        ),
    }


def query_shared_temporal_quantities(
    checkpoints: Sequence[dict[str, Any]],
) -> dict[str, int | None]:
    values = temporal_quantities(checkpoints)
    return {
        "t_fit": values["t_fit"],
        "t_role95": values["t_role95"],
        "t_role_exact": values["t_role_exact"],
        "t_triad_exact": first_sustained(
            checkpoints, lambda item: item["all_48_triads_exact"]
        ),
    }


def trajectory_document(
    arm: str,
    runs: Sequence[dict[str, Any]],
    precommit_sha256: str,
    precommit_revision: str,
    environment: dict[str, Any],
    substrate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.trajectory.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "arm": arm,
        "score_definition": score_definitions()["triad_symmetric"],
        "objective": objective_definitions()[arm],
        "precommit": {
            "content_sha256": precommit_sha256,
            "package_readback_revision": precommit_revision,
            "read_back_before_any_current_role_metric": True,
        },
        "environment": environment,
        "seeds": list(SCORED_SEEDS),
        "horizon_updates": HORIZON,
        "checkpoint_schedule": list(CHECKPOINTS),
        "relation_substrate_sha256": substrate_sha256,
        "training_data_audit": {
            "train_records": POSITIVE_TRIADS,
            "role_records_in_objective": 0,
            "novel_records_in_objective": 0,
            "objective_function_role_argument": False,
            "objective_function_novel_argument": False,
            "checkpoint_diagnostics_backpropagated": False,
        },
        "runs": list(runs),
        "novel_metrics_computed": False,
    }


def compare_exact(
    expected: Any,
    observed: Any,
    path: str,
    deviations: list[dict[str, Any]],
) -> None:
    if type(expected) is not type(observed):
        deviations.append(
            {
                "path": path,
                "kind": "type",
                "expected": expected,
                "observed": observed,
            }
        )
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        for key in sorted(expected_keys - observed_keys):
            deviations.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "missing_observed_key",
                    "expected": expected[key],
                    "observed": None,
                }
            )
        for key in sorted(observed_keys - expected_keys):
            deviations.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "unexpected_observed_key",
                    "expected": None,
                    "observed": observed[key],
                }
            )
        for key in sorted(expected_keys & observed_keys):
            compare_exact(expected[key], observed[key], f"{path}.{key}", deviations)
        return
    if isinstance(expected, list):
        if len(expected) != len(observed):
            deviations.append(
                {
                    "path": path,
                    "kind": "list_length",
                    "expected": len(expected),
                    "observed": len(observed),
                }
            )
        for index, (left, right) in enumerate(zip(expected, observed, strict=False)):
            compare_exact(left, right, f"{path}[{index}]", deviations)
        return
    if expected != observed:
        deviations.append(
            {
                "path": path,
                "kind": "value",
                "expected": expected,
                "observed": observed,
            }
        )


def predecessor_shared_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "update",
        "train_cross_entropy",
        "train_accuracy",
        "role_test_accuracy",
        "role_test_correct",
        "role_test_total",
        "role_test_exact_96_of_96",
        "triads_all_three_roles_correct",
        "triads_total",
        "fraction_triads_all_three_roles_correct",
        "all_48_triads_exact",
        "role_completion_margin",
        "parameter_norms",
        "event_embedding",
    )
    return {key: checkpoint[key] for key in keys}


def query_as_predecessor_shared_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "update": checkpoint["update"],
        "train_cross_entropy": checkpoint["query_cross_entropy"],
        "train_accuracy": checkpoint["query_accuracy"],
        "role_test_accuracy": checkpoint["role_test_accuracy"],
        "role_test_correct": checkpoint["role_test_correct"],
        "role_test_total": checkpoint["role_test_total"],
        "role_test_exact_96_of_96": checkpoint["role_test_exact_96_of_96"],
        "triads_all_three_roles_correct": checkpoint[
            "triads_all_three_roles_correct"
        ],
        "triads_total": checkpoint["triads_total"],
        "fraction_triads_all_three_roles_correct": checkpoint[
            "fraction_triads_all_three_roles_correct"
        ],
        "all_48_triads_exact": checkpoint["all_48_triads_exact"],
        "role_completion_margin": checkpoint["role_completion_margin"],
        "parameter_norms": checkpoint["parameter_norms"],
        "event_embedding": checkpoint["event_embedding"],
    }


def baseline_replication_report(
    query_trajectory: dict[str, Any],
    predecessor: dict[str, Any],
    scoring_environment: dict[str, Any],
) -> dict[str, Any]:
    deviations: list[dict[str, Any]] = []
    compare_exact(
        predecessor["seeds"],
        query_trajectory["seeds"],
        "document.seeds",
        deviations,
    )
    compare_exact(
        predecessor["horizon_updates"],
        query_trajectory["horizon_updates"],
        "document.horizon_updates",
        deviations,
    )
    compare_exact(
        predecessor["checkpoint_schedule"],
        query_trajectory["checkpoint_schedule"],
        "document.checkpoint_schedule",
        deviations,
    )

    predecessor_runs = {run["seed"]: run for run in predecessor["runs"]}
    query_runs = {run["seed"]: run for run in query_trajectory["runs"]}
    temporal_comparison: dict[str, Any] = {}
    final_checkpoint_equal: dict[str, bool] = {}
    for seed in SCORED_SEEDS:
        expected_run = predecessor_runs.get(seed)
        observed_run = query_runs.get(seed)
        if expected_run is None or observed_run is None:
            deviations.append(
                {
                    "path": f"runs.{seed}",
                    "kind": "missing_run",
                    "expected": expected_run is not None,
                    "observed": observed_run is not None,
                }
            )
            continue
        for key in (
            "seed",
            "initialization_sha256",
            "optimizer",
            "updates_completed",
            "early_stopped",
        ):
            compare_exact(
                expected_run[key],
                observed_run[key],
                f"runs.{seed}.{key}",
                deviations,
            )
        compare_exact(
            len(expected_run["checkpoints"]),
            len(observed_run["checkpoints"]),
            f"runs.{seed}.checkpoint_count",
            deviations,
        )
        for index, (expected_checkpoint, observed_checkpoint) in enumerate(
            zip(
                expected_run["checkpoints"],
                observed_run["checkpoints"],
                strict=False,
            )
        ):
            compare_exact(
                predecessor_shared_checkpoint(expected_checkpoint),
                query_as_predecessor_shared_checkpoint(observed_checkpoint),
                f"runs.{seed}.checkpoints[{index}]",
                deviations,
            )
        expected_temporal = predecessor_temporal_quantities(
            expected_run["checkpoints"]
        )
        observed_temporal = query_shared_temporal_quantities(
            observed_run["checkpoints"]
        )
        temporal_comparison[str(seed)] = {
            "expected": expected_temporal,
            "observed": observed_temporal,
            "exact": expected_temporal == observed_temporal,
        }
        expected_final = predecessor_shared_checkpoint(
            expected_run["checkpoints"][-1]
        )
        observed_final = query_as_predecessor_shared_checkpoint(
            observed_run["checkpoints"][-1]
        )
        final_checkpoint_equal[str(seed)] = expected_final == observed_final

    environment_differences: list[dict[str, Any]] = []
    compare_exact(
        ACCEPTED_ENVIRONMENT,
        scoring_environment,
        "environment",
        environment_differences,
    )
    gate_passed = not deviations
    return {
        "status": "PASS" if gate_passed else "BLOCKED-BASELINE-REPLICATION",
        "reference_content_sha256": PREDECESSOR_TRAJECTORY_SHA256,
        "policy": BASELINE_POLICY,
        "accepted_environment": ACCEPTED_ENVIRONMENT,
        "scoring_environment": scoring_environment,
        "environment_difference_count": len(environment_differences),
        "environment_differences": environment_differences,
        "shared_value_deviation_count": len(deviations),
        "shared_value_deviations": deviations,
        "temporal_comparison_by_seed": temporal_comparison,
        "final_shared_checkpoint_exact_by_seed": final_checkpoint_equal,
        "all_shared_checkpoints_exact": gate_passed,
        "all_shared_temporals_exact": all(
            item["exact"] for item in temporal_comparison.values()
        ),
        "all_shared_final_behavior_exact": all(final_checkpoint_equal.values()),
        "causal_interpretation_permitted": gate_passed,
    }


def final_metrics(checkpoint: dict[str, Any]) -> dict[str, float | int]:
    return {
        "query_cross_entropy": checkpoint["query_cross_entropy"],
        "query_accuracy": checkpoint["query_accuracy"],
        "relation_cross_entropy": checkpoint["relation_cross_entropy"],
        "strict_relation_domination_count": checkpoint[
            "strict_relation_domination_count"
        ],
        "strict_relation_domination_fraction": checkpoint[
            "strict_relation_domination_fraction"
        ],
        "local_relation_margin_minimum": checkpoint["local_relation_margin"][
            "minimum"
        ],
        "local_relation_margin_median": checkpoint["local_relation_margin"]["median"],
        "local_relation_margin_mean": checkpoint["local_relation_margin"]["mean"],
        "role_test_correct": checkpoint["role_test_correct"],
        "role_test_accuracy": checkpoint["role_test_accuracy"],
        "triads_all_three_roles_correct": checkpoint[
            "triads_all_three_roles_correct"
        ],
        "fraction_triads_all_three_roles_correct": checkpoint[
            "fraction_triads_all_three_roles_correct"
        ],
        "role_margin_minimum": checkpoint["role_completion_margin"]["minimum"],
        "role_margin_median": checkpoint["role_completion_margin"]["median"],
        "role_margin_mean": checkpoint["role_completion_margin"]["mean"],
    }


def summarize_trajectories(
    trajectories: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    temporals: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    finals: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for run in trajectories[arm]["runs"]:
            key = str(run["seed"])
            temporals[arm][key] = temporal_quantities(run["checkpoints"])
            finals[arm][key] = final_metrics(run["checkpoints"][-1])

    counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        counts[arm] = {
            quantity: sum(
                temporals[arm][str(seed)][quantity] is not None
                for seed in SCORED_SEEDS
            )
            for quantity in (
                "t_fit",
                "t_relation95",
                "t_relation_exact",
                "t_role95",
                "t_role_exact",
            )
        }

    paired: list[dict[str, Any]] = []
    difference_fields = tuple(final_metrics(trajectories[ARMS[0]]["runs"][0]["checkpoints"][-1]))
    for seed in SCORED_SEEDS:
        key = str(seed)
        query_values = finals["query_ce"][key]
        relation_values = finals["relational_competition"][key]
        differences = {
            f"relational_minus_query_{field}": relation_values[field]
            - query_values[field]
            for field in difference_fields
        }
        paired.append({"seed": seed, **differences})

    summary = {}
    for field in paired[0]:
        if field == "seed":
            continue
        values = [float(item[field]) for item in paired]
        summary[f"{field}_mean"] = statistics.fmean(values)
        summary[f"{field}_median"] = ordinary_median(values)

    implication_violations: list[dict[str, Any]] = []
    relation_runs = trajectories["relational_competition"]["runs"]
    for run in relation_runs:
        for checkpoint in run["checkpoints"]:
            relation_count = checkpoint["strict_relation_domination_count"]
            role_count = checkpoint["role_test_correct"]
            triad_count = checkpoint["triads_all_three_roles_correct"]
            if role_count < 2 * relation_count:
                implication_violations.append(
                    {
                        "seed": run["seed"],
                        "update": checkpoint["update"],
                        "kind": "relation_count_implies_two_ROLE_completions_each",
                        "relation_count": relation_count,
                        "minimum_implied_role_count": 2 * relation_count,
                        "observed_role_count": role_count,
                    }
                )
            if triad_count < relation_count:
                implication_violations.append(
                    {
                        "seed": run["seed"],
                        "update": checkpoint["update"],
                        "kind": "relation_domination_implies_all_three_completions",
                        "relation_count": relation_count,
                        "minimum_implied_all_three_count": relation_count,
                        "observed_all_three_count": triad_count,
                    }
                )
            if relation_count == POSITIVE_TRIADS and role_count != ROLE_RECORDS:
                implication_violations.append(
                    {
                        "seed": run["seed"],
                        "update": checkpoint["update"],
                        "kind": "exact_relation_domination_implies_exact_ROLE",
                        "relation_count": relation_count,
                        "expected_role_count": ROLE_RECORDS,
                        "observed_role_count": role_count,
                    }
                )

    seed_exceptions: list[dict[str, Any]] = []
    for seed in SCORED_SEEDS:
        key = str(seed)
        query_temporal = temporals["query_ce"][key]
        relation_temporal = temporals["relational_competition"][key]
        reasons = []
        if (query_temporal["t_role95"] is not None) != (
            relation_temporal["t_role95"] is not None
        ):
            reasons.append("arms_differ_on_sustained_role95")
        if relation_temporal["t_relation95"] is not None and relation_temporal[
            "t_role95"
        ] is None:
            reasons.append("relation95_without_implied_role95")
        if relation_temporal["t_relation_exact"] is not None and relation_temporal[
            "t_role_exact"
        ] is None:
            reasons.append("relation_exact_without_implied_role_exact")
        if reasons:
            seed_exceptions.append(
                {
                    "seed": seed,
                    "reasons": reasons,
                    "query_ce": query_temporal,
                    "relational_competition": relation_temporal,
                }
            )

    return {
        "temporal_quantities": temporals,
        "seeds_reaching": counts,
        "final_metrics_by_arm_and_seed": finals,
        "paired_final_differences": paired,
        "paired_final_summary": summary,
        "objective_to_consequence_checkpoint_violations": implication_violations,
        "seed_level_exceptions": seed_exceptions,
    }


def determine_bin(
    baseline_passed: bool,
    query_role95_count: int,
    relation_relation95_count: int,
    relation_role95_count: int,
) -> str:
    if not baseline_passed:
        return "C0"
    query_robust = query_role95_count >= 7
    relation_robust = relation_role95_count >= 7
    if query_robust and relation_robust:
        return "C4"
    if query_robust and not relation_robust:
        return "C5"
    if relation_robust and not query_robust:
        return "C3"
    if relation_relation95_count < 7:
        return "C1"
    return "C2"


def build_analysis(
    precommit: dict[str, Any],
    precommit_sha256: str,
    precommit_revision: str,
    validation: dict[str, Any],
    trajectories: dict[str, dict[str, Any]],
    initialization_audit: dict[str, Any],
    baseline_report: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    summary = summarize_trajectories(trajectories)
    counts = summary["seeds_reaching"]
    bin_name = determine_bin(
        baseline_report["status"] == "PASS",
        counts["query_ce"]["t_role95"],
        counts["relational_competition"]["t_relation95"],
        counts["relational_competition"]["t_role95"],
    )
    if bin_name == "C0":
        status = "BLOCKED-BASELINE-REPLICATION"
    elif bin_name == "C2":
        status = "IMPLEMENTATION-AUDIT-FAILURE"
    else:
        status = "COMPLETE"
    return {
        "schema": f"{SCHEMA_PREFIX}.analysis.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "status": status,
        "benchmark_validation": validation,
        "precommit": {
            "content_sha256": precommit_sha256,
            "package_readback_revision": precommit_revision,
            "read_back_before_any_current_role_metric": True,
            "modified_after_role_metrics": False,
        },
        "environment": trajectories["query_ce"]["environment"],
        "optimizer": precommit["optimizer"],
        "matched_scored_initialization_audit": initialization_audit,
        "baseline_replication": baseline_report,
        "scored_execution_audit": {
            "seeds": list(SCORED_SEEDS),
            "arms": list(ARMS),
            "runs": len(SCORED_SEEDS) * len(ARMS),
            "horizon_updates_each": HORIZON,
            "all_runs_completed_horizon": True,
            "early_stops": 0,
            "extra_seeds": 0,
            "outcome_dependent_reruns": 0,
            "role_examples_in_training": 0,
            "novel_examples_in_training": 0,
            "query_role_augmented_examples": 0,
            "relation_corruptions_per_positive": CORRUPTIONS_PER_POSITIVE,
            "hyperparameter_searches": 0,
            "novel_metrics_computed": False,
        },
        **summary,
        "interpretation": {
            "bin": bin_name,
            "definition": interpretation_bins()[bin_name],
            "all_bins": interpretation_bins(),
            "precedence": interpretation_precedence(),
            "baseline_gate_passed": baseline_report["status"] == "PASS",
            "causal_interpretation_available": bin_name == "C3",
            "necessity_inferred": False,
        },
        "scientific_fences": scientific_fences(),
        "artifact_hashes": {
            PRECOMMIT_NAME: precommit_sha256,
            QUERY_NAME: sha256_file(out_dir / QUERY_NAME),
            RELATION_NAME: sha256_file(out_dir / RELATION_NAME),
            RUNNER_NAME: sha256_file(Path(__file__)),
            "accepted_predecessor_trajectory": PREDECESSOR_TRAJECTORY_SHA256,
        },
    }


def format_optional(value: int | None) -> str:
    return "—" if value is None else str(value)


def build_result_markdown(analysis: dict[str, Any], out_dir: Path) -> str:
    interpretation = analysis["interpretation"]
    baseline = analysis["baseline_replication"]
    counts = analysis["seeds_reaching"]
    temporals = analysis["temporal_quantities"]
    paired = {str(item["seed"]): item for item in analysis["paired_final_differences"]}
    lines = [
        "# 015.07 — Coder: role-neutral competitive consequence result",
        "",
        f"- **Status:** `{analysis['status']}`",
        f"- **Interpretation bin:** **{interpretation['bin']}**",
        f"- **Baseline replication gate:** `{baseline['status']}`",
        (
            "- **Precommit read back before current ROLE evaluation:** yes, "
            f"`{analysis['precommit']['package_readback_revision']}`"
        ),
        "",
        f"**Information exposure:** {INFORMATION_EXPOSURE}",
        "",
        "## Frozen contract",
        "",
        (
            "The exact Issue-015.03 packet passed all identity gates: 84 Events, 48 "
            "TRAIN records, 96 evaluation-only ROLE_TEST records, both semantic "
            "digests, pair disjointness, and exact equality between ROLE_TEST and the "
            "96 alternate completions derived from TRAIN. No NOVEL data or metric was "
            "loaded or computed."
        ),
        "",
        (
            "Both arms used the accepted 71,425-scalar triad-symmetric SharedScorer, "
            "byte-identical initialization, deterministic CPU float32 execution, the "
            "same fixed AdamW optimizer, eight seeds, checkpoints, and 16,384-update "
            "horizon. No ROLE_TEST record was used by either training objective."
        ),
        "",
        (
            "Q minimized only the mean 82-way CE over the 48 designated TRAIN queries. "
            "R minimized only the mean 244-way relation CE placing each positive T at "
            "class 0 against its 243 frozen pair-preserving one-vertex corruptions, "
            "with temperature 1 and no margin, weighting, sampling, or mining."
        ),
        "",
        "## Pre-score audits and implication",
        "",
        (
            "The 48 canonical positives were unique and had exactly 144 unique "
            "constituent pairs. Every N(T) contained exactly 243 unique corruptions in "
            "three 81-member branches and no corruption collided with a positive. The "
            "scorer passed all 95,284 distinct triples under all six permutations."
        ),
        "",
        (
            "For each alternate query and each of its 81 wrong legal candidates, the "
            "completed triad is in the corresponding N(T). Therefore strict domination "
            "of every N(T) makes the designated third the unique argmax for all 96 "
            "TRAIN-derived alternate-role queries. This was checked mechanically using "
            "TRAIN only before scoring."
        ),
        "",
        "## Baseline replication",
        "",
        (
            f"Gate status: **{baseline['status']}**. Shared-value deviations: "
            f"{baseline['shared_value_deviation_count']}. Environment differences: "
            f"{baseline['environment_difference_count']}."
        ),
        "",
        (
            "The gate compares every shared Q checkpoint value exactly to the accepted "
            "015.04 triad-symmetric trajectory. Environment differences are recorded "
            "but do not waive any shared-value deviation. No tuning is permitted."
        ),
        "",
        "## Scored trajectories",
        "",
        "| Arm | t_fit | t_relation95 | t_relation_exact | t_role95 | t_role_exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (
        ("query_ce", "Q — designated-query CE"),
        ("relational_competition", "R — relational competition"),
    ):
        arm_counts = counts[arm]
        lines.append(
            f"| {label} | {arm_counts['t_fit']}/8 | "
            f"{arm_counts['t_relation95']}/8 | "
            f"{arm_counts['t_relation_exact']}/8 | "
            f"{arm_counts['t_role95']}/8 | "
            f"{arm_counts['t_role_exact']}/8 |"
        )
    lines.extend(
        [
            "",
            (
                "Exact relation counts are the `t_relation_exact` column; exact ROLE "
                "counts are the `t_role_exact` column. Thresholds are unsmoothed and "
                "require the first qualifying checkpoint and the next two scheduled "
                "checkpoints, except t_fit, which is the first exact query-fit checkpoint."
            ),
            "",
            "| Seed | Q fit | Q rel95 | Q rel-exact | Q role95 | Q role-exact | R fit | R rel95 | R rel-exact | R role95 | R role-exact | Δ final ROLE R−Q | Δ final relation fraction R−Q | Δ final local margin mean R−Q | Δ final ROLE margin mean R−Q |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SCORED_SEEDS:
        key = str(seed)
        query = temporals["query_ce"][key]
        relation = temporals["relational_competition"][key]
        delta = paired[key]
        lines.append(
            f"| {seed} | {format_optional(query['t_fit'])} | "
            f"{format_optional(query['t_relation95'])} | "
            f"{format_optional(query['t_relation_exact'])} | "
            f"{format_optional(query['t_role95'])} | "
            f"{format_optional(query['t_role_exact'])} | "
            f"{format_optional(relation['t_fit'])} | "
            f"{format_optional(relation['t_relation95'])} | "
            f"{format_optional(relation['t_relation_exact'])} | "
            f"{format_optional(relation['t_role95'])} | "
            f"{format_optional(relation['t_role_exact'])} | "
            f"{delta['relational_minus_query_role_test_accuracy']:.6f} | "
            f"{delta['relational_minus_query_strict_relation_domination_fraction']:.6f} | "
            f"{delta['relational_minus_query_local_relation_margin_mean']:.6f} | "
            f"{delta['relational_minus_query_role_margin_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            (
                "All paired final accuracies, losses, strict-domination counts/fractions, "
                "triad-completion fractions, local-relation margins, ROLE margins, and "
                "their mean/median summaries are in `analysis.json`, together with every "
                "seed-level exception and every baseline deviation."
            ),
            "",
            "## Interpretation",
            "",
            f"**{interpretation['bin']}:** {interpretation['definition']}",
            "",
        ]
    )
    if interpretation["bin"] == "C3":
        lines.extend(
            [
                (
                    "On the frozen observed-triad substrate, using the same role-neutral "
                    "scorer, training each observed triad to dominate all locally "
                    "incompatible pair-preserving corruptions is sufficient relative to "
                    "designated-query cross-entropy to make the already-identifiable ROLE "
                    "consequences behaviorally available."
                ),
                "",
            ]
        )
    if interpretation["bin"] == "C0":
        lines.extend(
            [
                (
                    "The accepted Q baseline did not replicate exactly. No causal "
                    "interpretation of the objective contrast is made."
                ),
                "",
            ]
        )
    if interpretation["bin"] == "C2":
        lines.extend(
            [
                (
                    "This outcome contradicts the mechanically proved implication and is "
                    "treated as an implementation/audit failure, not a scientific result."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "No necessity is inferred.",
            "",
            "## Artifacts",
            "",
            f"- `{PRECOMMIT_NAME}` — SHA-256 `{analysis['precommit']['content_sha256']}`",
            f"- `{QUERY_NAME}` — SHA-256 `{sha256_file(out_dir / QUERY_NAME)}`",
            f"- `{RELATION_NAME}` — SHA-256 `{sha256_file(out_dir / RELATION_NAME)}`",
            f"- `{ANALYSIS_NAME}` — SHA-256 `{sha256_file(out_dir / ANALYSIS_NAME)}`",
            f"- `{RUNNER_NAME}` — SHA-256 `{sha256_file(Path(__file__))}`",
            "",
            "## Scientific fences",
            "",
        ]
    )
    lines.extend(f"- {fence}" for fence in scientific_fences())
    lines.extend(
        [
            "",
            (
                "A negative result in either arm does not refute role-neutral relational "
                "learning in other architectures."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def require_score_outputs_absent(out_dir: Path, result_path: Path) -> None:
    paths = [
        out_dir / QUERY_NAME,
        out_dir / RELATION_NAME,
        out_dir / ANALYSIS_NAME,
        result_path,
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ContractError(
            "refusing scored rerun because output files already exist: " + ", ".join(existing)
        )


def score(args: argparse.Namespace) -> int:
    configure_determinism(args.threads)
    precommit_path = args.out_dir / PRECOMMIT_NAME
    if not precommit_path.is_file():
        raise ContractError(f"missing frozen precommit {precommit_path}")
    actual_precommit_sha = sha256_file(precommit_path)
    require_sha256(args.precommit_sha256, "precommit SHA-256")
    if actual_precommit_sha != args.precommit_sha256:
        raise ContractError(
            f"precommit hash mismatch {actual_precommit_sha} != {args.precommit_sha256}"
        )
    require_sha256(args.precommit_package_revision, "precommit package readback revision")
    precommit = read_json(precommit_path)

    # This complete frozen-contract gate occurs before current evaluation records load.
    validate_frozen_precommit(precommit, args.threads)

    raw_baseline = read_source(args.predecessor_trajectory_source)
    baseline_audit, predecessor = validate_predecessor_trajectory(raw_baseline)
    if baseline_audit != precommit["accepted_baseline_trajectory_validation"]:
        raise ContractError("baseline reference differs from frozen precommit")

    raw_benchmark = read_source(args.benchmark_source)
    validation, train, role = validate_benchmark(raw_benchmark)
    if validation != precommit["benchmark_validation"]:
        raise ContractError("benchmark validation differs from frozen precommit")
    substrate, positives, neighborhoods = build_relation_substrate(train)
    if substrate != precommit["relation_substrate"]:
        raise ContractError("relation substrate differs from frozen precommit")
    proof_audit = implication_proof_audit(train, positives, neighborhoods)
    if proof_audit != precommit["objective_to_consequence_implication"]:
        raise ContractError("implication proof audit differs from frozen precommit")

    result_path = args.result or (args.out_dir.parent / DEFAULT_RESULT_NAME)
    expected_result_path = args.out_dir.parent / DEFAULT_RESULT_NAME
    if result_path.resolve() != expected_result_path.resolve():
        raise ContractError(f"result must use exact delivery path {expected_result_path}")
    require_score_outputs_absent(args.out_dir, result_path)

    runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    matched_initialization: dict[str, Any] = {}
    scoring_environment = environment_record(args.threads)
    for seed in SCORED_SEEDS:
        query_model = make_model(seed)
        relation_model = make_model(seed)
        query_hash = state_hash(query_model)
        relation_hash = state_hash(relation_model)
        byte_identical = all(
            torch.equal(left, right)
            for left, right in zip(
                query_model.state_dict().values(),
                relation_model.state_dict().values(),
                strict=True,
            )
        )
        if not byte_identical or query_hash != relation_hash:
            raise ContractError(f"scored initialization mismatch for seed {seed}")
        frozen_initialization = precommit["matched_initialization_audit"][str(seed)]
        if frozen_initialization != {
            "byte_identical": True,
            "query_ce_sha256": query_hash,
            "relational_competition_sha256": relation_hash,
        }:
            raise ContractError(f"scored initialization differs from precommit for seed {seed}")
        matched_initialization[str(seed)] = frozen_initialization

        print(f"scored seed={seed} arm=query_ce", flush=True)
        query_run = run_trajectory(
            query_model,
            "query_ce",
            train,
            role,
            positives,
            neighborhoods,
            seed,
        )
        query_run["initialization_sha256"] = query_hash
        runs["query_ce"].append(query_run)
        print(
            f"completed seed={seed} arm=query_ce final_role="
            f"{query_run['checkpoints'][-1]['role_test_accuracy']:.6f}",
            flush=True,
        )

        print(f"scored seed={seed} arm=relational_competition", flush=True)
        relation_run = run_trajectory(
            relation_model,
            "relational_competition",
            train,
            role,
            positives,
            neighborhoods,
            seed,
        )
        relation_run["initialization_sha256"] = relation_hash
        runs["relational_competition"].append(relation_run)
        print(
            f"completed seed={seed} arm=relational_competition final_role="
            f"{relation_run['checkpoints'][-1]['role_test_accuracy']:.6f}",
            flush=True,
        )

    substrate_sha = substrate["positive_triads_sha256"]
    query_document = trajectory_document(
        "query_ce",
        runs["query_ce"],
        actual_precommit_sha,
        args.precommit_package_revision,
        scoring_environment,
        substrate_sha,
    )
    relation_document = trajectory_document(
        "relational_competition",
        runs["relational_competition"],
        actual_precommit_sha,
        args.precommit_package_revision,
        scoring_environment,
        substrate_sha,
    )
    write_json(args.out_dir / QUERY_NAME, query_document)
    write_json(args.out_dir / RELATION_NAME, relation_document)

    baseline_report = baseline_replication_report(
        query_document, predecessor, scoring_environment
    )
    trajectories = {
        "query_ce": query_document,
        "relational_competition": relation_document,
    }
    analysis = build_analysis(
        precommit,
        actual_precommit_sha,
        args.precommit_package_revision,
        validation,
        trajectories,
        matched_initialization,
        baseline_report,
        args.out_dir,
    )
    write_json(args.out_dir / ANALYSIS_NAME, analysis)
    result_path.write_text(build_result_markdown(analysis, args.out_dir))
    print(
        json.dumps(
            {
                "analysis": str(args.out_dir / ANALYSIS_NAME),
                "baseline_replication": baseline_report["status"],
                "interpretation": analysis["interpretation"],
                "result": str(result_path),
                "trajectory_query_ce": str(args.out_dir / QUERY_NAME),
                "trajectory_relational_competition": str(
                    args.out_dir / RELATION_NAME
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def required_checkpoint_keys() -> set[str]:
    return {
        "update",
        "query_cross_entropy",
        "query_accuracy",
        "relation_cross_entropy",
        "strict_relation_domination_count",
        "strict_relation_domination_total",
        "strict_relation_domination_fraction",
        "strict_relation_domination_exact_48_of_48",
        "local_relation_margin",
        "role_test_accuracy",
        "role_test_correct",
        "role_test_total",
        "role_test_exact_96_of_96",
        "triads_all_three_roles_correct",
        "triads_total",
        "fraction_triads_all_three_roles_correct",
        "all_48_triads_exact",
        "role_completion_margin",
        "parameter_norms",
        "event_embedding",
    }


def finite_float_in_range(
    value: Any,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if type(value) is not float or not math.isfinite(value):
        return False
    if minimum is not None and value < minimum:
        return False
    return maximum is None or value <= maximum


def validate_margin_record(value: Any, label: str, failures: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"minimum", "median", "mean"}:
        failures.append(f"{label}: margin fields mismatch")
        return
    if not all(finite_float_in_range(item) for item in value.values()):
        failures.append(f"{label}: margin values must be finite floats")
        return
    if value["minimum"] > value["median"] or value["minimum"] > value["mean"]:
        failures.append(f"{label}: margin minimum is inconsistent")


def validate_parameter_record(value: Any, label: str, failures: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"per_tensor_l2", "total_l2"}:
        failures.append(f"{label}: parameter-norm fields mismatch")
        return
    expected_names = {
        "event.weight",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    }
    per_tensor = value["per_tensor_l2"]
    if not isinstance(per_tensor, dict) or set(per_tensor) != expected_names:
        failures.append(f"{label}: per-tensor norm inventory mismatch")
        return
    if not all(finite_float_in_range(item, 0.0) for item in per_tensor.values()):
        failures.append(f"{label}: per-tensor norms must be finite nonnegative floats")
    if not finite_float_in_range(value["total_l2"], 0.0):
        failures.append(f"{label}: total norm must be a finite nonnegative float")


def validate_embedding_record(value: Any, label: str, failures: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "frobenius_norm",
        "effective_rank",
    }:
        failures.append(f"{label}: Event-embedding fields mismatch")
        return
    if not finite_float_in_range(value["frobenius_norm"], 0.0):
        failures.append(f"{label}: embedding norm must be finite and nonnegative")
    if not finite_float_in_range(value["effective_rank"], 1.0, float(EMBED_DIM)):
        failures.append(f"{label}: embedding effective rank is out of range")


def validate_trajectory_document(
    document: dict[str, Any],
    arm: str,
    precommit_sha256: str,
    precommit_revision: str,
    substrate_sha256: str,
    expected_environment: dict[str, Any],
    initialization_audit: dict[str, Any],
    failures: list[str],
) -> None:
    label = arm
    expected_document_keys = {
        "schema",
        "created_at_utc",
        "information_exposure",
        "arm",
        "score_definition",
        "objective",
        "precommit",
        "environment",
        "seeds",
        "horizon_updates",
        "checkpoint_schedule",
        "relation_substrate_sha256",
        "training_data_audit",
        "runs",
        "novel_metrics_computed",
    }
    if not isinstance(document, dict) or set(document) != expected_document_keys:
        failures.append(f"{label}: document fields mismatch")
        return
    if document.get("schema") != f"{SCHEMA_PREFIX}.trajectory.v1":
        failures.append(f"{label}: schema mismatch")
    if type(document.get("created_at_utc")) is not str:
        failures.append(f"{label}: creation timestamp mismatch")
    if document.get("information_exposure") != INFORMATION_EXPOSURE:
        failures.append(f"{label}: information exposure mismatch")
    if document.get("arm") != arm:
        failures.append(f"{label}: arm mismatch")
    if document.get("score_definition") != score_definitions()["triad_symmetric"]:
        failures.append(f"{label}: score definition mismatch")
    if document.get("objective") != objective_definitions()[arm]:
        failures.append(f"{label}: objective mismatch")
    if document.get("environment") != expected_environment:
        failures.append(f"{label}: environment differs from frozen precommit")
    if document.get("seeds") != list(SCORED_SEEDS):
        failures.append(f"{label}: seed schedule mismatch")
    if document.get("horizon_updates") != HORIZON:
        failures.append(f"{label}: horizon mismatch")
    if document.get("checkpoint_schedule") != list(CHECKPOINTS):
        failures.append(f"{label}: checkpoint schedule mismatch")
    if document.get("relation_substrate_sha256") != substrate_sha256:
        failures.append(f"{label}: substrate hash mismatch")
    if document.get("novel_metrics_computed") is not False:
        failures.append(f"{label}: NOVEL metrics audit mismatch")
    expected_readback = {
        "content_sha256": precommit_sha256,
        "package_readback_revision": precommit_revision,
        "read_back_before_any_current_role_metric": True,
    }
    if document.get("precommit") != expected_readback:
        failures.append(f"{label}: precommit readback record mismatch")
    expected_training_audit = {
        "train_records": POSITIVE_TRIADS,
        "role_records_in_objective": 0,
        "novel_records_in_objective": 0,
        "objective_function_role_argument": False,
        "objective_function_novel_argument": False,
        "checkpoint_diagnostics_backpropagated": False,
    }
    if document.get("training_data_audit") != expected_training_audit:
        failures.append(f"{label}: training-data audit mismatch")

    runs = document.get("runs", [])
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        failures.append(f"{label}: runs must be a list of objects")
        return
    if [run.get("seed") for run in runs] != list(SCORED_SEEDS):
        failures.append(f"{label}: run count/order mismatch")
        return
    expected_run_keys = {
        "seed",
        "initialization_sha256",
        "optimizer",
        "objective",
        "updates_completed",
        "early_stopped",
        "checkpoints",
    }
    initialization_key = f"{arm}_sha256"
    for run in runs:
        seed = run.get("seed")
        run_label = f"{label}/{seed}"
        if set(run) != expected_run_keys:
            failures.append(f"{run_label}: run fields mismatch")
        expected_initialization = initialization_audit.get(str(seed), {}).get(
            initialization_key
        )
        if run.get("initialization_sha256") != expected_initialization:
            failures.append(f"{run_label}: initialization hash mismatch")
        try:
            require_sha256(run.get("initialization_sha256", ""), run_label)
        except ContractError as error:
            failures.append(str(error))
        if run.get("optimizer") != OPTIMIZER:
            failures.append(f"{run_label}: optimizer mismatch")
        if run.get("objective") != objective_definitions()[arm]:
            failures.append(f"{run_label}: run objective mismatch")
        if run.get("updates_completed") != HORIZON or run.get("early_stopped") is not False:
            failures.append(f"{run_label}: horizon or early-stop mismatch")
        checkpoints = run.get("checkpoints", [])
        if not isinstance(checkpoints, list) or not all(
            isinstance(item, dict) for item in checkpoints
        ):
            failures.append(f"{run_label}: checkpoints must be a list of objects")
            continue
        if [item.get("update") for item in checkpoints] != list(CHECKPOINTS):
            failures.append(f"{run_label}: checkpoint schedule mismatch")
            continue
        for checkpoint in checkpoints:
            checkpoint_label = f"{run_label}/{checkpoint.get('update')}"
            if set(checkpoint) != required_checkpoint_keys():
                failures.append(f"{checkpoint_label}: checkpoint fields mismatch")
                continue
            if type(checkpoint["update"]) is not int:
                failures.append(f"{checkpoint_label}: update is not an integer")
            float_domains = {
                "query_cross_entropy": (0.0, None),
                "query_accuracy": (0.0, 1.0),
                "relation_cross_entropy": (0.0, None),
                "strict_relation_domination_fraction": (0.0, 1.0),
                "role_test_accuracy": (0.0, 1.0),
                "fraction_triads_all_three_roles_correct": (0.0, 1.0),
            }
            for field, (minimum, maximum) in float_domains.items():
                if not finite_float_in_range(checkpoint[field], minimum, maximum):
                    failures.append(f"{checkpoint_label}: {field} is invalid")
            count_domains = {
                "strict_relation_domination_count": POSITIVE_TRIADS,
                "strict_relation_domination_total": POSITIVE_TRIADS,
                "role_test_correct": ROLE_RECORDS,
                "role_test_total": ROLE_RECORDS,
                "triads_all_three_roles_correct": POSITIVE_TRIADS,
                "triads_total": POSITIVE_TRIADS,
            }
            for field, maximum in count_domains.items():
                value = checkpoint[field]
                if type(value) is not int or not 0 <= value <= maximum:
                    failures.append(f"{checkpoint_label}: {field} is invalid")
            bool_fields = (
                "strict_relation_domination_exact_48_of_48",
                "role_test_exact_96_of_96",
                "all_48_triads_exact",
            )
            for field in bool_fields:
                if type(checkpoint[field]) is not bool:
                    failures.append(f"{checkpoint_label}: {field} is not boolean")

            relation_count = checkpoint["strict_relation_domination_count"]
            role_count = checkpoint["role_test_correct"]
            triad_count = checkpoint["triads_all_three_roles_correct"]
            if checkpoint["strict_relation_domination_total"] != POSITIVE_TRIADS:
                failures.append(f"{checkpoint_label}: relation total mismatch")
            if checkpoint["role_test_total"] != ROLE_RECORDS:
                failures.append(f"{checkpoint_label}: ROLE total mismatch")
            if checkpoint["triads_total"] != POSITIVE_TRIADS:
                failures.append(f"{checkpoint_label}: triad total mismatch")
            if type(relation_count) is int:
                if checkpoint["strict_relation_domination_fraction"] != (
                    relation_count / POSITIVE_TRIADS
                ):
                    failures.append(f"{checkpoint_label}: relation fraction mismatch")
                if checkpoint["strict_relation_domination_exact_48_of_48"] != (
                    relation_count == POSITIVE_TRIADS
                ):
                    failures.append(f"{checkpoint_label}: relation exact mismatch")
            if type(role_count) is int:
                if checkpoint["role_test_accuracy"] != role_count / ROLE_RECORDS:
                    failures.append(f"{checkpoint_label}: ROLE accuracy mismatch")
                if checkpoint["role_test_exact_96_of_96"] != (
                    role_count == ROLE_RECORDS
                ):
                    failures.append(f"{checkpoint_label}: ROLE exact mismatch")
            if type(triad_count) is int:
                if checkpoint["fraction_triads_all_three_roles_correct"] != (
                    triad_count / POSITIVE_TRIADS
                ):
                    failures.append(f"{checkpoint_label}: triad fraction mismatch")
                if checkpoint["all_48_triads_exact"] != (
                    triad_count == POSITIVE_TRIADS
                ):
                    failures.append(f"{checkpoint_label}: triad exact mismatch")
            validate_margin_record(
                checkpoint["local_relation_margin"],
                f"{checkpoint_label}/local_relation_margin",
                failures,
            )
            validate_margin_record(
                checkpoint["role_completion_margin"],
                f"{checkpoint_label}/role_completion_margin",
                failures,
            )
            validate_parameter_record(
                checkpoint["parameter_norms"],
                f"{checkpoint_label}/parameter_norms",
                failures,
            )
            validate_embedding_record(
                checkpoint["event_embedding"],
                f"{checkpoint_label}/event_embedding",
                failures,
            )


def exact_scored_execution_audit() -> dict[str, Any]:
    return {
        "seeds": list(SCORED_SEEDS),
        "arms": list(ARMS),
        "runs": len(SCORED_SEEDS) * len(ARMS),
        "horizon_updates_each": HORIZON,
        "all_runs_completed_horizon": True,
        "early_stops": 0,
        "extra_seeds": 0,
        "outcome_dependent_reruns": 0,
        "role_examples_in_training": 0,
        "novel_examples_in_training": 0,
        "query_role_augmented_examples": 0,
        "relation_corruptions_per_positive": CORRUPTIONS_PER_POSITIVE,
        "hyperparameter_searches": 0,
        "novel_metrics_computed": False,
    }


def verify(args: argparse.Namespace) -> int:
    expected_result = args.out_dir.parent / DEFAULT_RESULT_NAME
    failures: list[str] = []
    if args.result.resolve() != expected_result.resolve():
        failures.append(f"result path must be exactly {expected_result}")

    required_paths = {
        PRECOMMIT_NAME: args.out_dir / PRECOMMIT_NAME,
        QUERY_NAME: args.out_dir / QUERY_NAME,
        RELATION_NAME: args.out_dir / RELATION_NAME,
        ANALYSIS_NAME: args.out_dir / ANALYSIS_NAME,
        RUNNER_NAME: args.out_dir / RUNNER_NAME,
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise ContractError(f"missing required delivered files: {missing}")
    actual_attachment_files = {
        path.name for path in args.out_dir.iterdir() if path.is_file()
    }
    if actual_attachment_files != EXPECTED_ATTACHMENT_FILES:
        failures.append(
            "attachment filenames mismatch: "
            f"expected={sorted(EXPECTED_ATTACHMENT_FILES)} "
            f"observed={sorted(actual_attachment_files)}"
        )
    if not args.result.is_file():
        raise ContractError(f"missing required result {args.result}")

    precommit = read_json(required_paths[PRECOMMIT_NAME])
    query_document = read_json(required_paths[QUERY_NAME])
    relation_document = read_json(required_paths[RELATION_NAME])
    analysis = read_json(required_paths[ANALYSIS_NAME])
    result_text = args.result.read_text()

    precommit_sha = sha256_file(required_paths[PRECOMMIT_NAME])
    precommit_revision = analysis.get("precommit", {}).get(
        "package_readback_revision", ""
    )
    try:
        require_sha256(precommit_revision, "stored precommit readback revision")
        validate_frozen_precommit(
            precommit,
            precommit.get("deterministic_settings", {}).get("thread_count_frozen", 0),
            require_current_environment=False,
        )
    except ContractError as error:
        failures.append(f"precommit: {error}")
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2, sort_keys=True))
        return 2

    positives = tuple(
        tuple(item) for item in precommit.get("relation_substrate", {}).get("positive_triads", [])
    )
    try:
        reconstructed_substrate, _ = build_relation_substrate_from_positives(positives)
        if reconstructed_substrate != precommit.get("relation_substrate"):
            failures.append("relation substrate reconstruction mismatch")
    except ContractError as error:
        failures.append(f"relation substrate: {error}")
    substrate_sha = precommit.get("relation_substrate", {}).get(
        "positive_triads_sha256", ""
    )

    expected_environment = precommit.get("environment", {})
    initialization_audit = precommit.get("matched_initialization_audit", {})
    validate_trajectory_document(
        query_document,
        "query_ce",
        precommit_sha,
        precommit_revision,
        substrate_sha,
        expected_environment,
        initialization_audit,
        failures,
    )
    validate_trajectory_document(
        relation_document,
        "relational_competition",
        precommit_sha,
        precommit_revision,
        substrate_sha,
        expected_environment,
        initialization_audit,
        failures,
    )

    if query_document.get("environment") != relation_document.get("environment"):
        failures.append("Q/R trajectory environments are not identical")
    if query_document.get("score_definition") != relation_document.get("score_definition"):
        failures.append("Q/R score definitions are not identical")

    if analysis.get("schema") != f"{SCHEMA_PREFIX}.analysis.v1":
        failures.append("analysis schema mismatch")
    if analysis.get("information_exposure") != INFORMATION_EXPOSURE:
        failures.append("analysis information exposure mismatch")
    if analysis.get("precommit", {}).get("content_sha256") != precommit_sha:
        failures.append("analysis precommit hash mismatch")
    if analysis.get("precommit", {}).get("read_back_before_any_current_role_metric") is not True:
        failures.append("analysis precommit readback ordering mismatch")
    if analysis.get("precommit", {}).get("modified_after_role_metrics") is not False:
        failures.append("analysis precommit immutability mismatch")
    if analysis.get("scored_execution_audit") != exact_scored_execution_audit():
        failures.append("analysis scored-execution audit mismatch")
    if analysis.get("environment") != expected_environment:
        failures.append("analysis environment differs from frozen precommit")
    if analysis.get("matched_scored_initialization_audit") != initialization_audit:
        failures.append("analysis matched-initialization audit mismatch")
    for seed in SCORED_SEEDS:
        frozen = initialization_audit.get(str(seed), {})
        query_hash = frozen.get("query_ce_sha256")
        relation_hash = frozen.get("relational_competition_sha256")
        if query_hash != relation_hash:
            failures.append(f"seed {seed}: frozen Q/R initialization hashes differ")

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2, sort_keys=True))
        return 2

    trajectories = {
        "query_ce": query_document,
        "relational_competition": relation_document,
    }
    recomputed_summary = summarize_trajectories(trajectories)
    implication_violations = recomputed_summary[
        "objective_to_consequence_checkpoint_violations"
    ]
    if implication_violations:
        failures.append(
            "objective-to-consequence implication violated at "
            f"{len(implication_violations)} stored checkpoints"
        )
    for key, expected_value in recomputed_summary.items():
        if analysis.get(key) != expected_value:
            failures.append(f"analysis {key} mismatch")

    raw_baseline = read_source(args.predecessor_trajectory_source)
    try:
        _baseline_audit, predecessor = validate_predecessor_trajectory(raw_baseline)
        recomputed_baseline = baseline_replication_report(
            query_document,
            predecessor,
            query_document.get("environment", {}),
        )
        if analysis.get("baseline_replication") != recomputed_baseline:
            failures.append("analysis baseline replication gate mismatch")
    except ContractError as error:
        failures.append(f"baseline reference: {error}")
        recomputed_baseline = {"status": "BLOCKED-BASELINE-REPLICATION"}

    counts = recomputed_summary["seeds_reaching"]
    expected_bin = determine_bin(
        recomputed_baseline.get("status") == "PASS",
        counts["query_ce"]["t_role95"],
        counts["relational_competition"]["t_relation95"],
        counts["relational_competition"]["t_role95"],
    )
    if analysis.get("interpretation", {}).get("bin") != expected_bin:
        failures.append("analysis interpretation bin mismatch")
    if analysis.get("interpretation", {}).get("definition") != interpretation_bins()[
        expected_bin
    ]:
        failures.append("analysis interpretation definition mismatch")
    if analysis.get("interpretation", {}).get("necessity_inferred") is not False:
        failures.append("analysis improperly infers necessity")
    expected_status = (
        "BLOCKED-BASELINE-REPLICATION"
        if expected_bin == "C0"
        else "IMPLEMENTATION-AUDIT-FAILURE"
        if expected_bin == "C2"
        else "COMPLETE"
    )
    if analysis.get("status") != expected_status:
        failures.append("analysis status mismatch")

    expected_hashes = analysis.get("artifact_hashes", {})
    for name in (PRECOMMIT_NAME, QUERY_NAME, RELATION_NAME, RUNNER_NAME):
        path = required_paths[name]
        if expected_hashes.get(name) != sha256_file(path):
            failures.append(f"{name}: content hash mismatch")
    if expected_hashes.get("accepted_predecessor_trajectory") != (
        PREDECESSOR_TRAJECTORY_SHA256
    ):
        failures.append("accepted predecessor trajectory hash mismatch")

    expected_result_text = build_result_markdown(analysis, args.out_dir)
    if result_text != expected_result_text:
        failures.append("result markdown is not the exact analysis-derived document")

    required_result_statements = [
        "**Information exposure:**",
        f"**Baseline replication gate:** `{recomputed_baseline.get('status')}`",
        f"**Interpretation bin:** **{expected_bin}**",
        "No ROLE_TEST record was used by either training objective.",
        "No NOVEL data or metric was loaded or computed.",
        "No necessity is inferred.",
        f"`{PRECOMMIT_NAME}` — SHA-256 `{precommit_sha}`",
        f"`{QUERY_NAME}` — SHA-256 `{sha256_file(required_paths[QUERY_NAME])}`",
        f"`{RELATION_NAME}` — SHA-256 `{sha256_file(required_paths[RELATION_NAME])}`",
        f"`{ANALYSIS_NAME}` — SHA-256 `{sha256_file(required_paths[ANALYSIS_NAME])}`",
        f"`{RUNNER_NAME}` — SHA-256 `{sha256_file(required_paths[RUNNER_NAME])}`",
    ]
    for statement in required_result_statements:
        if statement not in result_text:
            failures.append(f"result markdown missing required statement: {statement}")
    if "Information exposure:" not in Path(__file__).read_text():
        failures.append("runner missing information exposure disclosure")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required_paths.items()
        },
        "result": {
            "bytes": args.result.stat().st_size,
            "sha256": sha256_file(args.result),
        },
        "baseline_replication": recomputed_baseline.get("status"),
        "interpretation": analysis.get("interpretation"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="validate frozen inputs and TRAIN-only relation contract; write precommit",
    )
    prepare_parser.add_argument("--benchmark-source", required=True)
    prepare_parser.add_argument("--predecessor-trajectory-source", required=True)
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    prepare_parser.add_argument("--base-revision", required=True)
    prepare_parser.add_argument("--base-entry-count", type=int, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    score_parser = subparsers.add_parser(
        "score",
        help="run frozen Q/R trajectories after immutable precommit readback",
    )
    score_parser.add_argument("--benchmark-source", required=True)
    score_parser.add_argument("--predecessor-trajectory-source", required=True)
    score_parser.add_argument("--out-dir", type=Path, required=True)
    score_parser.add_argument("--result", type=Path)
    score_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    score_parser.add_argument("--precommit-sha256", required=True)
    score_parser.add_argument("--precommit-package-revision", required=True)
    score_parser.set_defaults(function=score)

    verify_parser = subparsers.add_parser(
        "verify",
        help="mechanically verify final artifacts without training or ROLE recomputation",
    )
    verify_parser.add_argument("--predecessor-trajectory-source", required=True)
    verify_parser.add_argument("--out-dir", type=Path, required=True)
    verify_parser.add_argument("--result", type=Path, required=True)
    verify_parser.set_defaults(function=verify)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        return int(args.function(args))
    except ContractError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
