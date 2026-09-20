#!/usr/bin/env python3
"""Execute Issue 015.03's matched directional/triad-symmetric experiment.

Information exposure: this production experiment uses the complete curated Issue
015.03 packet (84 opaque Event IDs, 48 TRAIN records, and 96 ROLE_TEST records),
the handed task, repository/runtime conventions, and no NOVEL_TEST data. The
calibration function receives TRAIN records only; ROLE_TEST is first scored by
the ``score`` stage after an externally published and read-back precommit.

The required two-stage invocation is intentionally strict::

    python run_015_role.py prepare --benchmark-source PATH_OR_HTTPS --out-dir DIR
    # publish/read back DIR/precommit.json, without changing it
    python run_015_role.py score --benchmark-source PATH_OR_HTTPS --out-dir DIR \
        --precommit-sha256 SHA256 --precommit-package-revision REVISION
    python run_015_role.py verify --out-dir DIR --result RESULT_MD

Only the first command performs TRAIN-only optimizer calibration. The second
loads the frozen selection from precommit.json and never tunes or reruns based
on ROLE_TEST behavior.
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
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

SCHEMA_PREFIX = "occurrence.gpt.01504"
INFORMATION_EXPOSURE = (
    "Production execution with the complete curated Issue-015.03 packet: 84 "
    "opaque Event IDs, 48 TRAIN records, and 96 evaluation-only ROLE_TEST "
    "records. No NOVEL_TEST data, Issue 016 content, SFP/Fano/habitat "
    "coordinates, or certified completion tables were accessed. Optimizer "
    "calibration received TRAIN records only."
)
TASK_REVISION = "6ceb230dae9e069011062ca88b13198042cee20b0ba8dc85ec4f3cf1623f5731"
TASK_PATH = (
    "issues/015-learning-law-for-consequence-structure/"
    "015.03-GPT-role-neutral-triad-learning-Coder.md"
)
TASK_URI = f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}&path={TASK_PATH}"
TASK_SHA256 = "036595f43505a5185de3ee05976ae180aec45323a7292ab2006fb703826efd56"
BENCHMARK_PATH = (
    "issues/015-learning-law-for-consequence-structure/"
    "015.03-Code-attachments/role-learning-benchmark.json"
)
BENCHMARK_URI = (
    f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}"
    f"&path={BENCHMARK_PATH}"
)
BENCHMARK_SHA256 = "c9b7a0931ff9cd5e3084048bfa190dbaf7aef22f514d6fc2f0851b2abdc70dac"
EXPECTED_TRAIN_DIGEST = "ca7178877eed4f4a537deb9fb56e9a66f81a39395298a828b5fb058c6a7c9088"
EXPECTED_ROLE_DIGEST = "f000b04566a83a3ff7a9536fe4e1d5f8889b51df269879d757b1b2b60e6b7f37"

N_EVENTS = 84
EMBED_DIM = 64
HIDDEN_DIM = 256
LEGAL_CANDIDATES = 82
CALIBRATION_SEEDS = (0, 1, 2, 3)
SCORED_SEEDS = (3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007)
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003, 0.01)
WEIGHT_DECAYS = (0.0, 0.001, 0.01, 0.1, 1.0)
CALIBRATION_CAP = 4096
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
ARMS = ("directional", "triad_symmetric")
DEFAULT_THREADS = 10

PRECOMMIT_NAME = "precommit.json"
DIRECTIONAL_NAME = "trajectory_directional.json"
TRIAD_NAME = "trajectory_triad_symmetric.json"
ANALYSIS_NAME = "analysis.json"
DEFAULT_RESULT_NAME = "015.04-Coder-role-neutral-triad-learning-result-GPT.md"


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


class SharedScorer(nn.Module):
    """The one parameterization shared byte-for-byte by both score semantics."""

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


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_source(source: str) -> bytes:
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


def validate_benchmark(raw_bytes: bytes) -> tuple[dict[str, Any], tuple[Record, ...], tuple[Record, ...]]:
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
    if len(train) != 48 or len(role) != 96:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH record counts")
    if packet.get("sizes") != {"train": 48, "role_test": 96}:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH declared sizes")

    train_digest = records_digest(train)
    role_digest = records_digest(role)
    if train_digest != EXPECTED_TRAIN_DIGEST:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH TRAIN digest")
    if role_digest != EXPECTED_ROLE_DIGEST:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH ROLE_TEST digest")

    train_pairs = {record.query for record in train}
    role_pairs = {record.query for record in role}
    if len(train_pairs) != 48 or len(role_pairs) != 96 or train_pairs & role_pairs:
        raise ContractError("BLOCKED-BENCHMARK-MISMATCH query-pair disjointness")

    derived = alternate_roles(train)
    derived_set = set(derived)
    role_set = set(role)
    if len(derived) != 96 or len(derived_set) != 96 or derived_set != role_set:
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


def logits(model: SharedScorer, batch: Batch, arm: str) -> Tensor:
    rows, classes = batch.candidates.shape
    pair_a = batch.pair_a[:, None].expand(-1, classes).reshape(-1)
    pair_b = batch.pair_b[:, None].expand(-1, classes).reshape(-1)
    candidate = batch.candidates.reshape(-1)
    if arm == "directional":
        values = model.base_score(pair_a, pair_b, candidate)
    elif arm == "triad_symmetric":
        values = model.triad_score(pair_a, pair_b, candidate)
    else:
        raise ContractError(f"unknown arm {arm}")
    return values.reshape(rows, classes)


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
                    "xavier_uniform_gain_1.0"
                    if name.endswith("weight")
                    else "zeros"
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
        directional = make_model(seed)
        triad = make_model(seed)
        exact = all(
            torch.equal(left, right)
            for left, right in zip(
                directional.state_dict().values(), triad.state_dict().values(), strict=True
            )
        )
        left_hash = state_hash(directional)
        right_hash = state_hash(triad)
        if not exact or left_hash != right_hash:
            raise ContractError(f"matched initialization failed for seed {seed}")
        results[str(seed)] = {
            "byte_identical": True,
            "directional_sha256": left_hash,
            "triad_symmetric_sha256": right_hash,
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
                        "Arm-T permutation identity failed at triple offset "
                        f"{start}, permutation {permutation}"
                    )
                compared += len(chunk)
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


def train_until_fit(
    arm: str,
    train_records: tuple[Record, ...],
    seed: int,
    learning_rate: float,
    weight_decay: float,
    cap: int,
) -> dict[str, Any]:
    """TRAIN-only calibration worker; no packet or ROLE_TEST argument exists."""
    batch = make_batch(train_records)
    model = make_model(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    with torch.inference_mode():
        if exact_accuracy(logits(model, batch, arm), batch.target_classes) == 1.0:
            return {"first_fit_update": 0, "terminal": "fit"}
    for update in range(1, cap + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score_matrix = logits(model, batch, arm)
        loss = F.cross_entropy(score_matrix, batch.target_classes)
        if not bool(torch.isfinite(loss)):
            return {"first_fit_update": None, "terminal": "nonfinite", "update": update}
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            accuracy = exact_accuracy(logits(model, batch, arm), batch.target_classes)
        if accuracy == 1.0:
            return {"first_fit_update": update, "terminal": "fit"}
    return {"first_fit_update": None, "terminal": "cap", "update": cap}


def calibration_key(point: dict[str, Any], arms: Sequence[str]) -> tuple[float, int, float, float]:
    medians = []
    first_fits = []
    for arm in arms:
        values = [point["runs"][arm][str(seed)]["first_fit_update"] for seed in CALIBRATION_SEEDS]
        if any(value is None for value in values):
            raise ContractError("attempted to rank an unqualified calibration point")
        medians.append(float(statistics.median(values)))
        first_fits.extend(values)
    return (
        max(medians),
        max(first_fits),
        float(point["learning_rate"]),
        float(point["weight_decay"]),
    )


def calibrate(train_records: tuple[Record, ...]) -> dict[str, Any]:
    """Search the frozen grid using exactly 48 TRAIN records and no other data."""
    if len(train_records) != 48:
        raise ContractError("calibration did not receive exactly 48 TRAIN records")
    grid: list[dict[str, Any]] = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            point: dict[str, Any] = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "runs": {arm: {} for arm in ARMS},
            }
            print(
                f"calibration lr={learning_rate:g} wd={weight_decay:g}",
                flush=True,
            )
            for arm in ARMS:
                for seed in CALIBRATION_SEEDS:
                    point["runs"][arm][str(seed)] = train_until_fit(
                        arm,
                        train_records,
                        seed,
                        learning_rate,
                        weight_decay,
                        CALIBRATION_CAP,
                    )
            point["common_capacity_qualified"] = all(
                point["runs"][arm][str(seed)]["first_fit_update"] is not None
                for arm in ARMS
                for seed in CALIBRATION_SEEDS
            )
            point["arm_capacity_qualified"] = {
                arm: all(
                    point["runs"][arm][str(seed)]["first_fit_update"] is not None
                    for seed in CALIBRATION_SEEDS
                )
                for arm in ARMS
            }
            if point["common_capacity_qualified"]:
                point["common_selection_key"] = list(calibration_key(point, ARMS))
            for arm in ARMS:
                if point["arm_capacity_qualified"][arm]:
                    point.setdefault("arm_selection_keys", {})[arm] = list(
                        calibration_key(point, (arm,))
                    )
            grid.append(point)

    common = [point for point in grid if point["common_capacity_qualified"]]
    if common:
        selected = min(common, key=lambda point: calibration_key(point, ARMS))
        selection = {
            "status": "COMMON-CAPACITY",
            "matched_causal_comparison_available": True,
            "optimizer_by_arm": {
                arm: {
                    "learning_rate": selected["learning_rate"],
                    "weight_decay": selected["weight_decay"],
                }
                for arm in ARMS
            },
            "selection_key": list(calibration_key(selected, ARMS)),
        }
    else:
        per_arm: dict[str, dict[str, float]] = {}
        per_arm_keys: dict[str, list[float]] = {}
        for arm in ARMS:
            qualified = [point for point in grid if point["arm_capacity_qualified"][arm]]
            if not qualified:
                raise ContractError(f"no capacity-qualified calibration setting for {arm}")
            selected = min(qualified, key=lambda point: calibration_key(point, (arm,)))
            per_arm[arm] = {
                "learning_rate": selected["learning_rate"],
                "weight_decay": selected["weight_decay"],
            }
            per_arm_keys[arm] = list(calibration_key(selected, (arm,)))
        selection = {
            "status": "NO-COMMON-CAPACITY",
            "matched_causal_comparison_available": False,
            "optimizer_by_arm": per_arm,
            "selection_key_by_arm": per_arm_keys,
        }

    return {
        "training_data_audit": {
            "function": "calibrate(train_records)",
            "records_received": len(train_records),
            "digest_received": records_digest(train_records),
            "role_test_argument_or_loader": False,
            "role_test_metrics_computed_or_logged": False,
            "selection_uses_train_exact_accuracy_only": True,
        },
        "optimizer": "AdamW",
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "schedule": "constant",
        "full_batch": True,
        "seeds": list(CALIBRATION_SEEDS),
        "cap_updates": CALIBRATION_CAP,
        "learning_rate_grid": list(LEARNING_RATES),
        "weight_decay_grid": list(WEIGHT_DECAYS),
        "grid_points": grid,
        "common_capacity_qualified_count": len(common),
        "selection": selection,
    }


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


def score_definitions() -> dict[str, Any]:
    return {
        "base": (
            "p=concat(E[a]+E[b], E[a]*E[b], abs(E[a]-E[b]), E[c]); "
            "g(a,b;c)=Linear(256,256)->GELU(exact)->Linear(256,1)"
        ),
        "directional": "score_D(c|a,b)=g(a,b;c)",
        "triad_symmetric": (
            "for canonical x<y<z, score_T({x,y,z})="
            "((g(x,y;z)+g(x,z;y))+g(y,z;x))/3"
        ),
        "base_pair_swap_invariant": True,
        "triad_role_permutation_invariant_by_constitution": True,
        "only_primary_arm_difference": "score semantics",
    }


def interpretation_bins() -> dict[str, str]:
    return {
        "R0": (
            "NO-COMMON-CAPACITY: matched comparison unavailable; report only "
            "capacity diagnostics"
        ),
        "R1": (
            "neither arm has >=7/8 seeds reaching t_role95: this score family "
            "does not yield robust ROLE transfer; exact role symmetry alone is "
            "insufficient here"
        ),
        "R2": (
            "Arm T has >=7/8 seeds reaching t_role95 and Arm D has <7/8: "
            "role-neutral triad score constitution is sufficient relative to "
            "the matched directional scorer for robust seen-triad consequence"
        ),
        "R3": (
            "both arms have >=7/8 seeds reaching t_role95: the matched scalar-"
            "scoring family succeeds, but the experiment does not isolate role "
            "symmetry as the decisive cause"
        ),
        "R4": (
            "Arm D has >=7/8 seeds reaching t_role95 and Arm T has <7/8: role "
            "symmetry is not beneficial in this setup"
        ),
    }


def scientific_fences() -> list[str]:
    return [
        "No discovery or evaluation of the global 56-block relation.",
        "No NOVEL blocks or NOVEL metric.",
        "No SFP/Fano/habitat coordinates.",
        "No claim for a preferred OT learning law.",
        "No claim that Event embeddings contain a unique readable algebra.",
        "No necessity claim for symmetry.",
        "No claim that every consequence-learning problem reduces to permutation invariance.",
        "No language-scale, physical Event, or physical Outcome semantics claim.",
    ]


def build_precommit(
    validation: dict[str, Any],
    calibration: dict[str, Any],
    initialization: dict[str, Any],
    permutation_audit: dict[str, Any],
    threads: int,
    base_revision: str,
    base_entry_count: int,
) -> dict[str, Any]:
    inventory = parameter_inventory(make_model(0))
    return {
        "schema": f"{SCHEMA_PREFIX}.precommit.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "status": "PRECOMMITTED-BEFORE-ROLE-EVALUATION",
        "handed_task": {
            "uri": TASK_URI,
            "revision": TASK_REVISION,
            "path": TASK_PATH,
            "content_sha256": TASK_SHA256,
        },
        "benchmark": {
            "uri": BENCHMARK_URI,
            "revision": TASK_REVISION,
            "path": BENCHMARK_PATH,
            "content_sha256": BENCHMARK_SHA256,
            "validation": validation,
        },
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
        "parameter_inventory": inventory,
        "matched_initialization_audit": initialization,
        "score_definitions": score_definitions(),
        "triad_permutation_identity_audit": permutation_audit,
        "candidate_mask": {
            "universe_classes": N_EVENTS,
            "masked": ["query Event a", "query Event b"],
            "legal_classes_per_query": LEGAL_CANDIDATES,
            "identical_between_arms": True,
        },
        "training_signal": {
            "records": 48,
            "examples": "designated TRAIN query/target records only",
            "role_augmentation": False,
            "alternate_target_cycling": False,
            "auxiliary_or_contrastive_loss": False,
            "loss": "full-batch cross-entropy over 82 legal candidates",
        },
        "calibration": calibration,
        "scored_plan": {
            "seeds": list(SCORED_SEEDS),
            "horizon_updates": HORIZON,
            "early_stopping": False,
            "extra_seeds": False,
            "outcome_dependent_reruns": False,
            "checkpoints": list(CHECKPOINTS),
            "optimizer_by_arm": calibration["selection"]["optimizer_by_arm"],
            "comparison_kind": (
                "matched-causal"
                if calibration["selection"]["matched_causal_comparison_available"]
                else "capacity-diagnostic"
            ),
        },
        "checkpoint_metrics": [
            "TRAIN cross-entropy",
            "TRAIN accuracy",
            "ROLE_TEST accuracy",
            "ROLE_TEST exact 96/96 boolean",
            "fraction of 48 observed triples with all three role completions correct",
            "minimum/median/mean ROLE_TEST completion margin",
            "per-tensor and total parameter L2 norms",
            "Event embedding Frobenius norm",
            "Event embedding entropy effective rank",
        ],
        "metric_definitions": {
            "role_margin": "correct score minus maximum other legal-candidate score",
            "margin_median": "ordinary sample median (mean of middle two for n=96)",
            "embedding_effective_rank": (
                "exp(-sum_i p_i log p_i)), p_i=s_i^2/sum_j s_j^2 for the "
                "64 singular values of the 84x64 Event embedding"
            ),
        },
        "temporal_definitions": {
            "t_fit": "first scheduled checkpoint with TRAIN accuracy == 1.0",
            "t_role95": (
                "first scheduled checkpoint with ROLE_TEST accuracy >= 0.95 at "
                "that checkpoint and the next two scheduled checkpoints"
            ),
            "t_role_exact": (
                "first scheduled checkpoint with ROLE_TEST accuracy == 1.0 at "
                "that checkpoint and the next two scheduled checkpoints"
            ),
            "t_triad_exact": (
                "first scheduled checkpoint with all 48 observed triples having "
                "all three completions correct at that checkpoint and the next "
                "two scheduled checkpoints"
            ),
            "smoothing": False,
            "never_reached": None,
        },
        "interpretation_bins": interpretation_bins(),
        "scientific_fences": scientific_fences(),
        "role_metrics_computed_before_this_precommit": False,
        "publication": {
            "path": (
                "issues/015-learning-law-for-consequence-structure/"
                "015.04-Code-attachments/precommit.json"
            ),
            "publish_against": "latest",
            "workflow": "occurrence",
            "base_revision_observed": base_revision,
            "base_entry_count_observed": base_entry_count,
            "expected_manifest_delta": 1,
            "expected_entry_count_after_publish": base_entry_count + 1,
            "immutable_after_role_metrics": True,
        },
    }


def prepare(args: argparse.Namespace) -> int:
    configure_determinism(args.threads)
    raw = read_source(args.benchmark_source)
    validation, train, _role_evaluation_only = validate_benchmark(raw)
    initialization = verify_matched_initialization((*CALIBRATION_SEEDS, *SCORED_SEEDS))
    print("auditing all Arm-T Event permutations", flush=True)
    permutation_audit = audit_triad_permutation_identity()
    print("running TRAIN-only common-capacity calibration", flush=True)
    calibration = calibrate(train)
    precommit = build_precommit(
        validation,
        calibration,
        initialization,
        permutation_audit,
        args.threads,
        args.base_revision,
        args.base_entry_count,
    )
    destination = args.out_dir / PRECOMMIT_NAME
    if destination.exists() and not args.force:
        raise ContractError(f"refusing to overwrite existing {destination}")
    write_json(destination, precommit)
    print(
        json.dumps(
            {
                "precommit": str(destination),
                "sha256": sha256_file(destination),
                "selection": calibration["selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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


def evaluate_checkpoint(
    model: SharedScorer,
    arm: str,
    train_batch: Batch,
    role_batch: Batch,
    all_roles_batch: Batch,
    update: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        train_scores = logits(model, train_batch, arm)
        role_scores = logits(model, role_batch, arm)
        all_role_scores = logits(model, all_roles_batch, arm)
        train_loss = float(F.cross_entropy(train_scores, train_batch.target_classes).item())
        train_accuracy = exact_accuracy(train_scores, train_batch.target_classes)
        role_predictions = role_scores.argmax(dim=1)
        role_hits = role_predictions == role_batch.target_classes
        role_accuracy = float(role_hits.to(torch.float64).mean().item())
        all_role_hits = (
            all_role_scores.argmax(dim=1) == all_roles_batch.target_classes
        ).reshape(48, 3)
        triad_exact_count = int(torch.all(all_role_hits, dim=1).sum().item())

        row_ids = torch.arange(len(role_batch.records), dtype=torch.long)
        correct_scores = role_scores[row_ids, role_batch.target_classes]
        other_scores = role_scores.clone()
        other_scores[row_ids, role_batch.target_classes] = -torch.inf
        margins = (correct_scores - other_scores.max(dim=1).values).double().tolist()

    return {
        "update": update,
        "train_cross_entropy": train_loss,
        "train_accuracy": train_accuracy,
        "role_test_accuracy": role_accuracy,
        "role_test_correct": int(role_hits.sum().item()),
        "role_test_total": len(role_batch.records),
        "role_test_exact_96_of_96": bool(torch.all(role_hits).item()),
        "triads_all_three_roles_correct": triad_exact_count,
        "triads_total": 48,
        "fraction_triads_all_three_roles_correct": triad_exact_count / 48.0,
        "all_48_triads_exact": triad_exact_count == 48,
        "role_completion_margin": {
            "minimum": float(min(margins)),
            "median": ordinary_median(margins),
            "mean": float(statistics.fmean(margins)),
        },
        "parameter_norms": parameter_norms(model),
        "event_embedding": embedding_diagnostics(model),
    }


def run_trajectory(
    model: SharedScorer,
    arm: str,
    train: tuple[Record, ...],
    role: tuple[Record, ...],
    optimizer_config: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    train_batch = make_batch(train)
    role_batch = make_batch(role)
    all_roles_batch = make_batch(all_role_completions(train))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    checkpoints: list[dict[str, Any]] = []
    checkpoint_set = set(CHECKPOINTS)
    checkpoints.append(
        evaluate_checkpoint(model, arm, train_batch, role_batch, all_roles_batch, 0)
    )
    for update in range(1, HORIZON + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        score_matrix = logits(model, train_batch, arm)
        loss = F.cross_entropy(score_matrix, train_batch.target_classes)
        if not bool(torch.isfinite(loss)):
            raise ContractError(f"nonfinite scored loss: arm={arm}, seed={seed}, update={update}")
        loss.backward()
        optimizer.step()
        if update in checkpoint_set:
            checkpoints.append(
                evaluate_checkpoint(
                    model,
                    arm,
                    train_batch,
                    role_batch,
                    all_roles_batch,
                    update,
                )
            )
    if [item["update"] for item in checkpoints] != list(CHECKPOINTS):
        raise ContractError("trajectory checkpoint schedule drift")
    return {
        "seed": seed,
        "initialization_sha256": None,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": optimizer_config["learning_rate"],
            "weight_decay": optimizer_config["weight_decay"],
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "schedule": "constant",
            "full_batch": True,
        },
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


def determine_bin(
    selection_status: str,
    directional_count: int,
    triad_count: int,
) -> str:
    if selection_status == "NO-COMMON-CAPACITY":
        return "R0"
    directional_robust = directional_count >= 7
    triad_robust = triad_count >= 7
    if not directional_robust and not triad_robust:
        return "R1"
    if triad_robust and not directional_robust:
        return "R2"
    if directional_robust and triad_robust:
        return "R3"
    return "R4"


def trajectory_document(
    arm: str,
    runs: Sequence[dict[str, Any]],
    precommit_sha256: str,
    precommit_revision: str,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.trajectory.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "arm": arm,
        "score_definition": score_definitions()[arm],
        "precommit": {
            "content_sha256": precommit_sha256,
            "package_readback_revision": precommit_revision,
            "read_back_before_any_role_metric": True,
        },
        "seeds": list(SCORED_SEEDS),
        "horizon_updates": HORIZON,
        "checkpoint_schedule": list(CHECKPOINTS),
        "runs": list(runs),
        "novel_metrics_computed": False,
    }


def build_analysis(
    precommit: dict[str, Any],
    precommit_sha256: str,
    precommit_revision: str,
    validation: dict[str, Any],
    trajectories: dict[str, dict[str, Any]],
    initialization_audit: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    temporals: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    final_by_arm: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for run in trajectories[arm]["runs"]:
            seed_key = str(run["seed"])
            temporals[arm][seed_key] = temporal_quantities(run["checkpoints"])
            final = run["checkpoints"][-1]
            final_by_arm[arm][seed_key] = {
                "role_test_accuracy": final["role_test_accuracy"],
                "role_margin_minimum": final["role_completion_margin"]["minimum"],
                "role_margin_median": final["role_completion_margin"]["median"],
                "role_margin_mean": final["role_completion_margin"]["mean"],
                "fraction_triads_all_three_roles_correct": final[
                    "fraction_triads_all_three_roles_correct"
                ],
            }

    counts: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        counts[arm] = {
            quantity: sum(
                temporals[arm][str(seed)][quantity] is not None for seed in SCORED_SEEDS
            )
            for quantity in ("t_fit", "t_role95", "t_role_exact", "t_triad_exact")
        }

    paired: list[dict[str, Any]] = []
    for seed in SCORED_SEEDS:
        key = str(seed)
        directional = final_by_arm["directional"][key]
        triad = final_by_arm["triad_symmetric"][key]
        paired.append(
            {
                "seed": seed,
                "triad_minus_directional_final_role_accuracy": (
                    triad["role_test_accuracy"] - directional["role_test_accuracy"]
                ),
                "triad_minus_directional_final_margin_minimum": (
                    triad["role_margin_minimum"] - directional["role_margin_minimum"]
                ),
                "triad_minus_directional_final_margin_median": (
                    triad["role_margin_median"] - directional["role_margin_median"]
                ),
                "triad_minus_directional_final_margin_mean": (
                    triad["role_margin_mean"] - directional["role_margin_mean"]
                ),
            }
        )

    selection_status = precommit["calibration"]["selection"]["status"]
    bin_name = determine_bin(
        selection_status,
        counts["directional"]["t_role95"],
        counts["triad_symmetric"]["t_role95"],
    )
    exceptions = []
    for seed in SCORED_SEEDS:
        key = str(seed)
        d_reached = temporals["directional"][key]["t_role95"] is not None
        t_reached = temporals["triad_symmetric"][key]["t_role95"] is not None
        if d_reached != t_reached:
            exceptions.append(
                {
                    "seed": seed,
                    "directional_reached_t_role95": d_reached,
                    "triad_symmetric_reached_t_role95": t_reached,
                }
            )

    paired_accuracy = [item["triad_minus_directional_final_role_accuracy"] for item in paired]
    paired_margin = [item["triad_minus_directional_final_margin_mean"] for item in paired]
    return {
        "schema": f"{SCHEMA_PREFIX}.analysis.v1",
        "created_at_utc": utc_now(),
        "information_exposure": INFORMATION_EXPOSURE,
        "status": "COMPLETE",
        "benchmark_validation": validation,
        "precommit": {
            "content_sha256": precommit_sha256,
            "package_readback_revision": precommit_revision,
            "read_back_before_any_role_metric": True,
            "modified_after_role_metrics": False,
        },
        "calibration_selection": precommit["calibration"]["selection"],
        "comparison_kind": precommit["scored_plan"]["comparison_kind"],
        "matched_scored_initialization_audit": initialization_audit,
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
            "novel_metrics_computed": False,
        },
        "temporal_quantities": temporals,
        "seeds_reaching": counts,
        "final_metrics_by_arm_and_seed": final_by_arm,
        "paired_final_differences": paired,
        "paired_final_summary": {
            "role_accuracy_difference_mean": statistics.fmean(paired_accuracy),
            "role_accuracy_difference_median": ordinary_median(paired_accuracy),
            "role_margin_mean_difference_mean": statistics.fmean(paired_margin),
            "role_margin_mean_difference_median": ordinary_median(paired_margin),
        },
        "interpretation": {
            "bin": bin_name,
            "definition": interpretation_bins()[bin_name],
            "all_bins": interpretation_bins(),
            "seed_level_exceptions": exceptions,
            "necessity_inferred": False,
        },
        "scientific_fences": scientific_fences(),
        "artifact_hashes": {
            DIRECTIONAL_NAME: sha256_file(out_dir / DIRECTIONAL_NAME),
            TRIAD_NAME: sha256_file(out_dir / TRIAD_NAME),
            PRECOMMIT_NAME: precommit_sha256,
            Path(__file__).name: sha256_file(Path(__file__)),
        },
    }


def format_optional(value: int | None) -> str:
    return "—" if value is None else str(value)


def build_result_markdown(analysis: dict[str, Any], out_dir: Path) -> str:
    interpretation = analysis["interpretation"]
    counts = analysis["seeds_reaching"]
    selection = analysis["calibration_selection"]
    lines = [
        "# 015.04 — Coder: role-neutral triad learning result",
        "",
        f"- **Status:** `{analysis['status']}`",
        f"- **Interpretation bin:** **{interpretation['bin']}**",
        f"- **Comparison:** `{analysis['comparison_kind']}`",
        f"- **Precommit read back before ROLE evaluation:** yes, `{analysis['precommit']['package_readback_revision']}`",
        "",
        f"**Information exposure:** {INFORMATION_EXPOSURE}",
        "",
        "## Contract and benchmark",
        "",
        "The frozen packet passed all identity gates: 84 Events, 48 TRAIN records, "
        "96 ROLE_TEST records, the required TRAIN and ROLE_TEST digests, disjoint "
        "query pairs, three distinct Events per TRAIN record, and exact equality "
        "between ROLE_TEST and the 96 alternate completions derived from TRAIN. "
        "No NOVEL data or metric was loaded.",
        "",
        "Both arms used the same 71,425 trainable scalars (one 84×64 Event table "
        "and the 256→256→1 scorer), byte-identical matched initialization, candidate "
        "mask, 82-class full-batch cross-entropy, AdamW settings, and 48 designated "
        "TRAIN examples. Arm T passed an exhaustive all-triples/all-six-permutations "
        "bit-identity audit before training. No alternate-role example was trained.",
        "",
        "## TRAIN-only calibration",
        "",
        f"Selection status: `{selection['status']}`.",
        "",
        "```json",
        json.dumps(selection["optimizer_by_arm"], indent=2, sort_keys=True),
        "```",
        "",
        "The frozen 5×5 grid was evaluated with seeds 0–3 through the declared "
        "4096-update cap. Selection used only exact TRAIN fit times; calibration "
        "did not receive or score ROLE_TEST.",
        "",
        "## Scored trajectories",
        "",
        "All eight preregistered seeds (3000–3007) in both arms completed exactly "
        "16,384 full-batch updates, with no early stop, extra seed, or rerun. "
        "Thresholds are unsmoothed and require the first qualifying checkpoint plus "
        "the next two scheduled checkpoints.",
        "",
        "| Arm | t_fit | t_role95 | t_role_exact | t_triad_exact |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Directional | {counts['directional']['t_fit']}/8 | "
            f"{counts['directional']['t_role95']}/8 | "
            f"{counts['directional']['t_role_exact']}/8 | "
            f"{counts['directional']['t_triad_exact']}/8 |"
        ),
        (
            f"| Triad-symmetric | {counts['triad_symmetric']['t_fit']}/8 | "
            f"{counts['triad_symmetric']['t_role95']}/8 | "
            f"{counts['triad_symmetric']['t_role_exact']}/8 | "
            f"{counts['triad_symmetric']['t_triad_exact']}/8 |"
        ),
        "",
        "| Seed | D t_fit | D t_role95 | D t_role_exact | D t_triad_exact | T t_fit | T t_role95 | T t_role_exact | T t_triad_exact | Δ final ROLE (T−D) | Δ final mean margin (T−D) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    temporals = analysis["temporal_quantities"]
    paired = {str(item["seed"]): item for item in analysis["paired_final_differences"]}
    for seed in SCORED_SEEDS:
        key = str(seed)
        d = temporals["directional"][key]
        t = temporals["triad_symmetric"][key]
        delta = paired[key]
        lines.append(
            f"| {seed} | {format_optional(d['t_fit'])} | {format_optional(d['t_role95'])} | "
            f"{format_optional(d['t_role_exact'])} | {format_optional(d['t_triad_exact'])} | "
            f"{format_optional(t['t_fit'])} | {format_optional(t['t_role95'])} | "
            f"{format_optional(t['t_role_exact'])} | {format_optional(t['t_triad_exact'])} | "
            f"{delta['triad_minus_directional_final_role_accuracy']:.6f} | "
            f"{delta['triad_minus_directional_final_margin_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"**{interpretation['bin']}:** {interpretation['definition']}",
            "",
        ]
    )
    if interpretation["bin"] == "R2":
        lines.extend(
            [
                "On the frozen 48-triad substrate, constituting candidate scoring "
                "as an exactly role-neutral triad can make the already-identifiable "
                "alternate roles behaviorally available without training on those "
                "alternate-role examples.",
                "",
                "This is a finite sufficiency result relative to the matched directional "
                "control. It is not a necessity result.",
                "",
            ]
        )
    lines.extend(
        [
            "Seeds reaching exact ROLE and exact all-triad thresholds, paired final "
            "accuracy/margin differences, and seed-level exceptions are fully enumerated "
            "in `analysis.json`; checkpoint-level losses, accuracies, margins, parameter "
            "norms, embedding norms, and effective ranks are in the two trajectory files.",
            "",
            "## Artifacts",
            "",
            f"- `{PRECOMMIT_NAME}` — SHA-256 `{analysis['precommit']['content_sha256']}`",
            f"- `{DIRECTIONAL_NAME}` — SHA-256 `{sha256_file(out_dir / DIRECTIONAL_NAME)}`",
            f"- `{TRIAD_NAME}` — SHA-256 `{sha256_file(out_dir / TRIAD_NAME)}`",
            f"- `{ANALYSIS_NAME}` — contains the complete mechanical disposition",
            f"- `{Path(__file__).name}` — SHA-256 `{sha256_file(Path(__file__))}`",
            "",
            "## Scientific fences",
            "",
        ]
    )
    lines.extend(f"- {fence}" for fence in scientific_fences())
    lines.extend(
        [
            "",
            "A negative result in any arm does not refute role-neutral relational learning "
            "in other architectures.",
            "",
        ]
    )
    return "\n".join(lines)


def score(args: argparse.Namespace) -> int:
    configure_determinism(args.threads)
    precommit_path = args.out_dir / PRECOMMIT_NAME
    if not precommit_path.is_file():
        raise ContractError(f"missing frozen precommit {precommit_path}")
    actual_precommit_sha = sha256_file(precommit_path)
    if actual_precommit_sha != args.precommit_sha256:
        raise ContractError(
            f"precommit hash mismatch {actual_precommit_sha} != {args.precommit_sha256}"
        )
    if len(args.precommit_package_revision) != 64:
        raise ContractError("a 64-hex package readback revision is required")
    int(args.precommit_package_revision, 16)
    precommit = read_json(precommit_path)
    if precommit.get("status") != "PRECOMMITTED-BEFORE-ROLE-EVALUATION":
        raise ContractError("invalid precommit status")
    if precommit["role_metrics_computed_before_this_precommit"] is not False:
        raise ContractError("precommit does not certify ROLE metric ordering")
    if precommit["scored_plan"]["seeds"] != list(SCORED_SEEDS):
        raise ContractError("precommit scored seed drift")
    if precommit["scored_plan"]["horizon_updates"] != HORIZON:
        raise ContractError("precommit horizon drift")
    if precommit["scored_plan"]["checkpoints"] != list(CHECKPOINTS):
        raise ContractError("precommit checkpoint drift")

    raw = read_source(args.benchmark_source)
    validation, train, role = validate_benchmark(raw)
    optimizer_by_arm = precommit["calibration"]["selection"]["optimizer_by_arm"]
    runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    matched_initialization: dict[str, Any] = {}

    for seed in SCORED_SEEDS:
        directional = make_model(seed)
        triad = make_model(seed)
        left_hash = state_hash(directional)
        right_hash = state_hash(triad)
        byte_identical = all(
            torch.equal(left, right)
            for left, right in zip(
                directional.state_dict().values(), triad.state_dict().values(), strict=True
            )
        )
        if not byte_identical or left_hash != right_hash:
            raise ContractError(f"scored initialization mismatch for seed {seed}")
        matched_initialization[str(seed)] = {
            "byte_identical": True,
            "directional_sha256": left_hash,
            "triad_symmetric_sha256": right_hash,
        }

        print(f"scored seed={seed} arm=directional", flush=True)
        directional_run = run_trajectory(
            directional,
            "directional",
            train,
            role,
            optimizer_by_arm["directional"],
            seed,
        )
        directional_run["initialization_sha256"] = left_hash
        runs["directional"].append(directional_run)
        print(
            f"completed seed={seed} arm=directional final_role="
            f"{directional_run['checkpoints'][-1]['role_test_accuracy']:.6f}",
            flush=True,
        )

        print(f"scored seed={seed} arm=triad_symmetric", flush=True)
        triad_run = run_trajectory(
            triad,
            "triad_symmetric",
            train,
            role,
            optimizer_by_arm["triad_symmetric"],
            seed,
        )
        triad_run["initialization_sha256"] = right_hash
        runs["triad_symmetric"].append(triad_run)
        print(
            f"completed seed={seed} arm=triad_symmetric final_role="
            f"{triad_run['checkpoints'][-1]['role_test_accuracy']:.6f}",
            flush=True,
        )

    directional_doc = trajectory_document(
        "directional",
        runs["directional"],
        actual_precommit_sha,
        args.precommit_package_revision,
    )
    triad_doc = trajectory_document(
        "triad_symmetric",
        runs["triad_symmetric"],
        actual_precommit_sha,
        args.precommit_package_revision,
    )
    write_json(args.out_dir / DIRECTIONAL_NAME, directional_doc)
    write_json(args.out_dir / TRIAD_NAME, triad_doc)

    trajectories = {
        "directional": directional_doc,
        "triad_symmetric": triad_doc,
    }
    analysis = build_analysis(
        precommit,
        actual_precommit_sha,
        args.precommit_package_revision,
        validation,
        trajectories,
        matched_initialization,
        args.out_dir,
    )
    write_json(args.out_dir / ANALYSIS_NAME, analysis)
    result_path = args.result
    if result_path is None:
        result_path = args.out_dir.parent / DEFAULT_RESULT_NAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(build_result_markdown(analysis, args.out_dir))
    print(
        json.dumps(
            {
                "analysis": str(args.out_dir / ANALYSIS_NAME),
                "interpretation": analysis["interpretation"],
                "result": str(result_path),
                "trajectory_directional": str(args.out_dir / DIRECTIONAL_NAME),
                "trajectory_triad_symmetric": str(args.out_dir / TRIAD_NAME),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def verify(args: argparse.Namespace) -> int:
    precommit = read_json(args.out_dir / PRECOMMIT_NAME)
    directional = read_json(args.out_dir / DIRECTIONAL_NAME)
    triad = read_json(args.out_dir / TRIAD_NAME)
    analysis = read_json(args.out_dir / ANALYSIS_NAME)
    result_text = args.result.read_text()
    failures: list[str] = []

    documents = {
        PRECOMMIT_NAME: precommit,
        DIRECTIONAL_NAME: directional,
        TRIAD_NAME: triad,
        ANALYSIS_NAME: analysis,
    }
    for name, document in documents.items():
        if document.get("information_exposure") != INFORMATION_EXPOSURE:
            failures.append(f"{name}: missing or changed information exposure")
    if "**Information exposure:**" not in result_text:
        failures.append("result markdown: missing information exposure")
    if "Information exposure:" not in Path(__file__).read_text():
        failures.append("runner: missing information exposure")

    if precommit["benchmark"]["validation"]["status"] != "PASS":
        failures.append("precommit benchmark validation did not pass")
    if precommit["triad_permutation_identity_audit"][
        "all_six_permutations_bit_identical"
    ] is not True:
        failures.append("precommit permutation audit did not pass")
    if precommit["calibration"]["training_data_audit"][
        "role_test_metrics_computed_or_logged"
    ] is not False:
        failures.append("calibration contamination audit failed")

    for arm, document in (("directional", directional), ("triad_symmetric", triad)):
        if document["arm"] != arm:
            failures.append(f"{arm}: arm label mismatch")
        if document["seeds"] != list(SCORED_SEEDS):
            failures.append(f"{arm}: seed schedule mismatch")
        if document["checkpoint_schedule"] != list(CHECKPOINTS):
            failures.append(f"{arm}: checkpoint schedule mismatch")
        if len(document["runs"]) != len(SCORED_SEEDS):
            failures.append(f"{arm}: run count mismatch")
        for run in document["runs"]:
            if run["updates_completed"] != HORIZON or run["early_stopped"]:
                failures.append(f"{arm}/{run['seed']}: horizon or early-stop mismatch")
            observed = [item["update"] for item in run["checkpoints"]]
            if observed != list(CHECKPOINTS):
                failures.append(f"{arm}/{run['seed']}: checkpoints mismatch")
            recomputed = temporal_quantities(run["checkpoints"])
            recorded = analysis["temporal_quantities"][arm][str(run["seed"])]
            if recomputed != recorded:
                failures.append(f"{arm}/{run['seed']}: temporal analysis mismatch")

    expected_hashes = analysis["artifact_hashes"]
    for name in (PRECOMMIT_NAME, DIRECTIONAL_NAME, TRIAD_NAME, Path(__file__).name):
        path = Path(__file__) if name == Path(__file__).name else args.out_dir / name
        if sha256_file(path) != expected_hashes[name]:
            failures.append(f"{name}: content hash mismatch")

    if analysis["scored_execution_audit"] != {
        "all_runs_completed_horizon": True,
        "arms": list(ARMS),
        "early_stops": 0,
        "extra_seeds": 0,
        "horizon_updates_each": HORIZON,
        "novel_metrics_computed": False,
        "outcome_dependent_reruns": 0,
        "role_examples_in_training": 0,
        "runs": 16,
        "seeds": list(SCORED_SEEDS),
    }:
        failures.append("scored execution audit mismatch")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "files": {
            name: {
                "bytes": (args.out_dir / name).stat().st_size,
                "sha256": sha256_file(args.out_dir / name),
            }
            for name in (PRECOMMIT_NAME, DIRECTIONAL_NAME, TRIAD_NAME, ANALYSIS_NAME)
        },
        "result": {
            "bytes": args.result.stat().st_size,
            "sha256": sha256_file(args.result),
        },
        "runner": {
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__)),
        },
        "interpretation": analysis["interpretation"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="validate, audit, calibrate TRAIN-only, and write precommit"
    )
    prepare_parser.add_argument("--benchmark-source", required=True)
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    prepare_parser.add_argument("--base-revision", default=TASK_REVISION)
    prepare_parser.add_argument("--base-entry-count", type=int, default=329)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    score_parser = subparsers.add_parser(
        "score", help="run frozen scored trajectories after precommit readback"
    )
    score_parser.add_argument("--benchmark-source", required=True)
    score_parser.add_argument("--out-dir", type=Path, required=True)
    score_parser.add_argument("--result", type=Path)
    score_parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    score_parser.add_argument("--precommit-sha256", required=True)
    score_parser.add_argument("--precommit-package-revision", required=True)
    score_parser.set_defaults(function=score)

    verify_parser = subparsers.add_parser(
        "verify", help="verify final artifacts without training or ROLE recomputation"
    )
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
