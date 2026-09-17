"""Issue 014.05a tied-Event-identity experiment harness.

This executable reuses the accepted 014.03a relation reconstruction, benchmark
builder, exact probe, schedules, scoring semantics, serialization, and canonical
certificate.  It changes one conceptual model feature: one ``event_emb``
Parameter is used in both query and candidate roles.

Long-running stages are deliberately separate and resumable.  No scored training
is performed by importing this module.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISSUE_DIR = HERE.parent
REPO = ISSUE_DIR.parents[1]
PREDECESSOR = ISSUE_DIR / "014.03a-Code-attachments"
PREDECESSOR_RUN = PREDECESSOR / "run_014.py"
PREDECESSOR_CONTRACT = PREDECESSOR / "benchmark_contract.json"
PINS_DIR = HERE / "pins"
CHECKPOINTS = HERE / "checkpoints"

_spec = importlib.util.spec_from_file_location("run_014_predecessor", PREDECESSOR_RUN)
if _spec is None or _spec.loader is None:
    raise SystemExit("BLOCKED: cannot load accepted 014.03a runner")
PRE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = PRE
_spec.loader.exec_module(PRE)
PRE.BUNDLE = "014.03a"
PRE.HERE = HERE
PRE.PINS_DIR = PINS_DIR
PRE.CHECKPOINTS = CHECKPOINTS
PRE.RANDOM_CONTROL_DIM = 64

DISPATCH_URI = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "f5fbf710e80d4285730b84872581b4246ba77d973a89414ca20f473bf6f540c4"
    "&path=issues/014-grokking-algebraic-representation-readout/"
    "014.05a-GPT-tied-Event-identity-grokking-ablation-Coder.md"
)
SPEC_URI = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "064a638e189f38a63a351c8889c29464e87af59f0684ff31c10b564dfdab8fa3"
    "&path=issues/014-grokking-algebraic-representation-readout/"
    "014.05-GPT-tied-Event-identity-grokking-ablation-Coder.md"
)
OWNER_URI = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "064a638e189f38a63a351c8889c29464e87af59f0684ff31c10b564dfdab8fa3"
    "&path=issues/014-grokking-algebraic-representation-readout/"
    "014.04-GPT-Owner-adjudication-after-01403a.md"
)
EXECUTION_IDENTITY = "Kiro (GPT 5.6 Sol) acting in the generic Coder role"
ACCEPTED_BENCHMARK_SHA256 = (
    "26f816cc5ae514cd73ad8d3603da70ac08aee99bd8343b8a0fc6f3f730f7e2dd"
)

N_EVENTS = 84
N_BLOCKS = 56
EMB_DIM = 64
PAIR_FEATURES = 192
TRUNK_HIDDEN = 256
PROBE_SEED = 314159
PROBE_STEPS = 5000
PROBE_LR = 0.01
PROBE_BETAS = (0.9, 0.999)
PROBE_EPS = 1e-8
PROBE_WEIGHT_DECAY = 0.001
PROBE_GATE = 0.95
RANDOM_CONTROL_DIM = 64
RANDOM_CONTROL_SEED = PRE.RANDOM_CONTROL_SEED
SELECT_BETAS = (0.9, 0.98)
SELECT_EPS = 1e-8
FALLBACK_LRS = (0.0001, 0.0003, 0.001, 0.003, 0.01)
FALLBACK_WDS = (0.0, 0.001, 0.01, 0.1, 1.0)
CALIBRATION_SEEDS = (0, 1, 2, 3)
CALIBRATION_CAP = 8192
SCORED_SEEDS = tuple(range(2000, 2008))
HORIZON = 65536
BEHAVIORAL_CHECKPOINTS = PRE.BEHAVIORAL_CHECKPOINTS
PROBE_CHECKPOINTS = PRE.PROBE_CHECKPOINTS
PRIMARY_ARMS = ("matched_regularized", "matched_zero_wd")
ALL_ARMS = PRIMARY_ARMS + ("capacity_fallback",)
CONTRACT_NAMES = (
    "tied_benchmark_contract.json",
    "tied_probe_contract.json",
    "environment.json",
)

PRIMARY_CONFIGS = {
    "matched_regularized": {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 0.003,
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "weight_decay": 1.0,
        "weight_decay_semantics": "decoupled AdamW",
        "schedule": "constant",
        "full_batch": True,
    },
    "matched_zero_wd": {
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 0.003,
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "weight_decay_semantics": "decoupled AdamW (zero coefficient)",
        "schedule": "constant",
        "full_batch": True,
    },
}

NEW_PIN_FILES = {
    DISPATCH_URI: "014.05a-dispatch.md",
    SPEC_URI: "014.05-spec.md",
    OWNER_URI: "014.04-owner.md",
}
# These immutable S3 objects are small, single-part objects; their versioned ETags
# are exact MD5 byte identities obtained while resolving the three declared URIs.
EXPECTED_NEW_PIN_MD5 = {
    "014.05a-dispatch.md": "1650029eebf7ac10136fcc14b417c7eb",
    "014.05-spec.md": "eb01d1203a44de4fcc40236116d87551",
    "014.04-owner.md": "643768e2026e550330f9f6288ecd0821",
}
ACCEPTED_INPUTS = {
    "quilt+s3://protology#package=occurrence/gpt@1625721c0169a458ea75cd83bab276db294c24fcb8953358efd4ae0bf68cb867&path=issues/014-grokking-algebraic-representation-readout/014.03a-Coder-readable-algebraic-grokking-trajectory-result-GPT.md": ISSUE_DIR / "014.03a-Coder-readable-algebraic-grokking-trajectory-result-GPT.md",
    "quilt+s3://protology#package=occurrence/gpt@1625721c0169a458ea75cd83bab276db294c24fcb8953358efd4ae0bf68cb867&path=issues/014-grokking-algebraic-representation-readout/014.03a-Code-attachments/benchmark_contract.json": PREDECESSOR / "benchmark_contract.json",
    "quilt+s3://protology#package=occurrence/gpt@1625721c0169a458ea75cd83bab276db294c24fcb8953358efd4ae0bf68cb867&path=issues/014-grokking-algebraic-representation-readout/014.03a-Code-attachments/analysis.json": PREDECESSOR / "analysis.json",
    "quilt+s3://protology#package=occurrence/gpt@1625721c0169a458ea75cd83bab276db294c24fcb8953358efd4ae0bf68cb867&path=issues/014-grokking-algebraic-representation-readout/014.03a-Code-attachments/negative_result_diagnostic.json": PREDECESSOR / "negative_result_diagnostic.json",
    "quilt+s3://protology#package=occurrence/gpt@1625721c0169a458ea75cd83bab276db294c24fcb8953358efd4ae0bf68cb867&path=issues/014-grokking-algebraic-representation-readout/014.03a-Code-attachments/probe_controls.json": PREDECESSOR / "probe_controls.json",
    "quilt+s3://protology#package=occurrence/gpt@88e238a596901add8ef311c8f9cea5b5c54806976a7bcb114fd34f5cd8491f44&path=issues/014-grokking-algebraic-representation-readout/014.01-GPT-Owner-future-grokking-representation-readout-frame.md": PINS_DIR / "014.01-frame.md",
}
PINNED_INPUTS = {
    **PRE.PINNED_INPUTS,
    **NEW_PIN_FILES,
}
PIN_PLACEHOLDER_MARKER = "EXACT_RAW_PIN_REQUIRED"


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw = path.read_bytes()
    loaded = json.loads(raw.decode("utf-8"))
    if digest(loaded) != digest(payload):
        raise SystemExit(f"BLOCKED: readback mismatch for {path.name}")
    return {
        "path": str(path.relative_to(HERE)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload_digest": digest(payload),
        "readback_exact": True,
    }


def bootstrap_predecessor_pins() -> list[str]:
    """Copy the accepted 014.03a pins locally without touching that bundle."""

    PINS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in sorted(set(PRE.PINNED_INPUTS.values())):
        source = PREDECESSOR / "pins" / filename
        target = PINS_DIR / filename
        if not source.is_file():
            raise SystemExit(f"BLOCKED: predecessor pin missing: {source}")
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
            copied.append(filename)
    return copied


def require_exact_new_pins() -> None:
    missing = []
    placeholders = []
    mismatched = []
    for filename in NEW_PIN_FILES.values():
        path = PINS_DIR / filename
        if not path.is_file():
            missing.append(filename)
            continue
        raw = path.read_bytes()
        if PIN_PLACEHOLDER_MARKER.encode() in raw:
            placeholders.append(filename)
            continue
        observed_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        if observed_md5 != EXPECTED_NEW_PIN_MD5[filename]:
            mismatched.append(
                {
                    "file": filename,
                    "expected_versioned_s3_etag_md5": EXPECTED_NEW_PIN_MD5[filename],
                    "observed_md5": observed_md5,
                }
            )
    if missing or placeholders or mismatched:
        raise SystemExit(
            "BLOCKED-PINS: install exact raw immutable bytes before contracts; "
            f"missing={missing} placeholders={placeholders} mismatched={mismatched}"
        )


def pin_manifest() -> dict:
    require_exact_new_pins()
    manifest = {}
    for uri_or_label, filename in sorted(PINNED_INPUTS.items()):
        path = PINS_DIR / filename
        if not path.is_file():
            raise SystemExit(f"BLOCKED-PINS: missing pinned input {filename}")
        manifest[uri_or_label] = {
            "local_copy": f"pins/{filename}",
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        if filename in EXPECTED_NEW_PIN_MD5:
            manifest[uri_or_label]["uri_resolution"] = {
                "resolved_before_contract_state": True,
                "versioned_s3_etag_md5": EXPECTED_NEW_PIN_MD5[filename],
                "local_bytes_match_resolved_object": True,
            }
    for uri, path in sorted(ACCEPTED_INPUTS.items()):
        if not path.is_file():
            raise SystemExit(f"BLOCKED-PINS: missing accepted input {path}")
        manifest[uri] = {
            "local_copy": str(path.relative_to(ISSUE_DIR)),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
    return manifest


def set_determinism() -> None:
    PRE.set_determinism()


def xavier_uniform_(tensor, gain: float, generator) -> None:
    PRE.xavier_uniform_(tensor, gain, generator)


class TiedMainModel:
    """Factory namespace used only to keep the public builder straightforward."""


def build_main_model(seed: int):
    """Build the tied architecture with predecessor-aligned downstream RNG."""

    import torch

    class MainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.event_emb = torch.nn.Parameter(torch.empty(N_EVENTS, EMB_DIM))
            self.lin1 = torch.nn.Linear(PAIR_FEATURES, TRUNK_HIDDEN)
            self.lin2 = torch.nn.Linear(TRUNK_HIDDEN, EMB_DIM)
            self.cand_bias = torch.nn.Parameter(torch.zeros(N_EVENTS))
            self.scale = math.sqrt(EMB_DIM)
            self._last_query_parameter_id: int | None = None
            self._last_candidate_parameter_id: int | None = None

        def initialize(self, seed_value: int) -> None:
            generator = torch.Generator().manual_seed(seed_value)
            xavier_uniform_(self.event_emb, 1.0, generator)
            # Frozen alignment shim: consume the predecessor candidate-table draw
            # before downstream weights, without retaining or training that tensor.
            discarded = torch.empty(N_EVENTS, EMB_DIM)
            xavier_uniform_(discarded, 1.0, generator)
            del discarded
            xavier_uniform_(self.lin1.weight, 1.0, generator)
            xavier_uniform_(self.lin2.weight, 1.0, generator)
            with torch.no_grad():
                self.lin1.bias.zero_()
                self.lin2.bias.zero_()
                self.cand_bias.zero_()

        def forward(self, left, right):
            self._last_query_parameter_id = id(self.event_emb)
            e_left = self.event_emb[left]
            e_right = self.event_emb[right]
            features = torch.cat(
                (e_left + e_right, e_left * e_right, (e_left - e_right).abs()),
                dim=-1,
            )
            query = self.lin2(torch.nn.functional.gelu(self.lin1(features)))
            self._last_candidate_parameter_id = id(self.event_emb)
            return query @ self.event_emb.t() / self.scale + self.cand_bias

        def representation(self):
            return self.event_emb.detach().clone()

    model = MainModel()
    model.initialize(seed)
    return model


def parameter_identity_audit(model=None) -> dict:
    """Mechanically prove one Event Parameter serves both model roles."""

    import torch

    model = build_main_model(0) if model is None else model
    named = list(model.named_parameters(remove_duplicate=False))
    event_shaped = [(name, parameter) for name, parameter in named if tuple(parameter.shape) == (84, 64)]
    event_named = [(name, parameter) for name, parameter in named if "emb" in name.lower() or "event" in name.lower()]
    left = torch.tensor([0, 1], dtype=torch.long)
    right = torch.tensor([2, 3], dtype=torch.long)
    logits = model(left, right)
    expected_names = {
        "event_emb",
        "lin1.weight",
        "lin1.bias",
        "lin2.weight",
        "lin2.bias",
        "cand_bias",
    }
    tied_fresh = build_main_model(0)
    predecessor_fresh = PRE.build_main_model(0)
    predecessor_alignment = {
        "event_emb_matches_predecessor_input_emb": torch.equal(
            tied_fresh.event_emb, predecessor_fresh.input_emb
        ),
        "lin1_weight_matches": torch.equal(
            tied_fresh.lin1.weight, predecessor_fresh.lin1.weight
        ),
        "lin1_bias_matches": torch.equal(
            tied_fresh.lin1.bias, predecessor_fresh.lin1.bias
        ),
        "lin2_weight_matches": torch.equal(
            tied_fresh.lin2.weight, predecessor_fresh.lin2.weight
        ),
        "lin2_bias_matches": torch.equal(
            tied_fresh.lin2.bias, predecessor_fresh.lin2.bias
        ),
        "candidate_bias_matches": torch.equal(
            tied_fresh.cand_bias, predecessor_fresh.cand_bias
        ),
    }
    unique_parameter_objects = len({id(parameter) for _, parameter in named}) == len(named)
    storage_aliases = [
        [name_a, name_b]
        for index, (name_a, parameter_a) in enumerate(named)
        for name_b, parameter_b in named[index + 1 :]
        if parameter_a.untyped_storage().data_ptr() == parameter_b.untyped_storage().data_ptr()
    ]
    checks = {
        "exactly_one_84x64_parameter": len(event_shaped) == 1,
        "event_parameter_name_exact": [name for name, _ in event_shaped] == ["event_emb"],
        "only_event_embedding_named_parameter": [name for name, _ in event_named] == ["event_emb"],
        "candidate_uses_exact_event_parameter_object": model._last_candidate_parameter_id == id(model.event_emb),
        "query_uses_exact_event_parameter_object": model._last_query_parameter_id == id(model.event_emb),
        "query_and_candidate_object_ids_equal": model._last_query_parameter_id == model._last_candidate_parameter_id,
        "representation_is_detached_clone_84x64": (
            tuple(model.representation().shape) == (84, 64)
            and not model.representation().requires_grad
            and model.representation().untyped_storage().data_ptr()
            != model.event_emb.untyped_storage().data_ptr()
        ),
        "named_parameter_set_exact": {name for name, _ in named} == expected_names,
        "no_duplicate_parameter_object": unique_parameter_objects,
        "no_parameter_storage_alias": not storage_aliases,
        "predecessor_seed_alignment": all(predecessor_alignment.values()),
        "logit_shape_2x84": tuple(logits.shape) == (2, 84),
    }
    return {
        "method": "named-parameter, shape, object-id instrumentation, storage-alias, representation-clone, forward-shape, and predecessor RNG-alignment audit",
        "named_parameters": [{"name": name, "shape": list(parameter.shape)} for name, parameter in named],
        "event_shaped_parameters": [name for name, _ in event_shaped],
        "query_parameter_object_id": model._last_query_parameter_id,
        "candidate_parameter_object_id": model._last_candidate_parameter_id,
        "event_parameter_object_id": id(model.event_emb),
        "storage_aliases": storage_aliases,
        "predecessor_seed_alignment": predecessor_alignment,
        "checks": checks,
        "passed": all(checks.values()),
    }


def accepted_benchmark_identity() -> tuple[dict, object, object, dict]:
    """Reconstruct only the predecessor-fixed split and compare every identity field."""

    if file_digest(PREDECESSOR_CONTRACT) != ACCEPTED_BENCHMARK_SHA256:
        raise SystemExit("BLOCKED-BENCHMARK-MISMATCH: accepted contract raw-byte SHA-256")
    accepted = json.loads(PREDECESSOR_CONTRACT.read_text(encoding="utf-8"))
    relation = PRE.reconstruct()
    rebuilt = PRE.build_benchmark(relation)
    rebuilt_split = {
        "TRAIN": [list(row) for row in rebuilt.train],
        "ROLE_TEST": [list(row) for row in rebuilt.role_test],
        "NOVEL_TEST": [list(row) for row in rebuilt.novel_test],
    }
    checks = {
        "accepted_contract_raw_sha256": file_digest(PREDECESSOR_CONTRACT) == ACCEPTED_BENCHMARK_SHA256,
        "events_84": relation.certification["counts"]["events"] == 84 == accepted["relation"]["counts"]["events"],
        "blocks_56": len(relation.blocks) == 56 == accepted["relation"]["counts"]["blocks"],
        "blocks_exact": [list(row) for row in relation.blocks] == accepted["relation_reconstruction"]["blocks"],
        "train_exact": rebuilt_split["TRAIN"] == accepted["split"]["TRAIN"],
        "role_test_exact": rebuilt_split["ROLE_TEST"] == accepted["split"]["ROLE_TEST"],
        "novel_test_exact": rebuilt_split["NOVEL_TEST"] == accepted["split"]["NOVEL_TEST"],
        "split_sizes_exact": {name: len(rows) for name, rows in rebuilt_split.items()} == accepted["split"]["sizes"],
        "split_digests_exact": {name: digest(rows) for name, rows in rebuilt_split.items()} == accepted["split"]["digests"],
        "novel_positions_exact": list(rebuilt.novel_blocks) == accepted["holdout"]["novel_block_positions"],
        "novel_triples_exact": rebuilt.holdout["novel_block_triples"] == accepted["holdout"]["novel_block_triples"],
        "seen_positions_exact": rebuilt.holdout["seen_block_positions"] == accepted["holdout"]["seen_block_positions"],
        "designated_roles_exact": {str(k): v for k, v in sorted(rebuilt.designated_role.items())} == accepted["role_assignment"]["designated_role_by_block_position"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise SystemExit(f"BLOCKED-BENCHMARK-MISMATCH: {failed}")
    return accepted, relation, rebuilt, checks


def architecture_record(audit: dict) -> dict:
    return {
        "event_embedding": [84, 64],
        "trainable_event_parameter_count": 1,
        "pair_features": "concat(E[a]+E[b], E[a]*E[b], abs(E[a]-E[b])); 192 units",
        "trunk": "Linear(192,256) -> GELU -> Linear(256,64)",
        "scoring": "dot(q, event_emb[c])/sqrt(64) + candidate bias over all 84 Events",
        "candidate_bias": [84],
        "loss": "full 84-way cross entropy on designated TRAIN targets",
        "forbidden": ["second Event table", "role-specific Event projection", "per-pair parameter"],
        "initialization": {
            "event_emb_and_linear_weights": "Xavier uniform gain 1.0",
            "all_biases": "zero",
            "rng_alignment": "after event_emb initialization consume and discard one Xavier [84,64] draw before linear weights, matching the predecessor downstream draw order",
        },
        "representation": "detached clone of event_emb, shape [84,64]",
        "parameter_identity_audit": audit,
    }


def probe_contract_record(pins: dict) -> dict:
    return {
        "module": "run_014_tied.py (stage: contracts)",
        "dispatch_uri": DISPATCH_URI,
        "spec_uri": SPEC_URI,
        "owner_uri": OWNER_URI,
        "pinned_inputs": pins,
        "representation": "z_x = detached clone of event_emb[x], 64 dimensions",
        "form": {
            "features": "concat(z_a+z_b, z_a*z_b, abs(z_a-z_b)); 192 units",
            "hidden": "Linear(192,64) -> GELU",
            "candidate": "Linear(64,64,bias=False)",
            "score": "dot(hidden, candidate(z_c))/sqrt(64)",
            "candidate_set": "all 84 Events",
        },
        "optimizer": {
            "implementation": "torch.optim.AdamW",
            "semantics": "decoupled weight decay",
            "lr": PROBE_LR,
            "betas": list(PROBE_BETAS),
            "eps": PROBE_EPS,
            "weight_decay": PROBE_WEIGHT_DECAY,
            "steps": PROBE_STEPS,
            "seed": PROBE_SEED,
            "batching": "full-batch TRAIN only",
            "main_gradients": "forbidden; representation detached",
            "identity_indexed_parameters": "forbidden",
        },
        "controls": {
            "B_certified_sfp": "accepted summed Arm B primary positive control",
            "C_scrambled": "accepted summed matched non-automorphic Arm C diagnostic",
            "random_gaussian": f"seed {RANDOM_CONTROL_SEED}, shape [84,64] diagnostic",
            "secondary_predecessor_controls": "accepted concat variants may be reported, explicitly secondary",
        },
        "gate": {"ROLE_TEST": PROBE_GATE, "NOVEL_TEST": PROBE_GATE, "failure": "BLOCKED-PROBE-REGRESSION"},
        "reported_metrics": [
            "probe_train_accuracy",
            "probe_role_accuracy",
            "probe_novel_accuracy",
            "seen_block_three_role_consistency",
            "all_block_three_role_consistency",
            "full_56_block_relation_exact",
        ],
    }


def temporal_definitions() -> dict:
    return PRE.temporal_definitions_record()


def fallback_manifest() -> dict:
    return {
        "optimizer": "torch.optim.AdamW",
        "schedule": "constant",
        "full_batch": True,
        "betas": list(SELECT_BETAS),
        "eps": SELECT_EPS,
        "learning_rate_grid": list(FALLBACK_LRS),
        "weight_decay_grid": list(FALLBACK_WDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "calibration_cap": CALIBRATION_CAP,
        "qualification": "all 4/4 calibration seeds exact TRAIN fit by update 8192",
        "ranking_qualified_only": [
            "nonzero weight decay preferred",
            "weight decay descending",
            "median first exact-fit update ascending",
            "learning rate ascending",
        ],
        "failure": "NO-TIED-CAPACITY when neither primary qualifies and no fallback candidate qualified",
    }


def environment_record(pins: dict, train_rows: list[list[int]], prior_hashes: dict) -> dict:
    import torch

    def run(command: list[str]) -> str:
        try:
            return subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    record = {
        "execution_identity": EXECUTION_IDENTITY,
        "dispatch_uri": DISPATCH_URI,
        "spec_uri": SPEC_URI,
        "owner_uri": OWNER_URI,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "topographo": importlib.metadata.version("topographo"),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "repository": {"root": str(REPO), "commit": run(["git", "rev-parse", "HEAD"]), "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]), "dirty": bool(run(["git", "status", "--porcelain"]))},
        "determinism": {"torch_use_deterministic_algorithms": True, "torch_threads": 1, "torch_interop_threads": 1, "device": "cpu", "dtype": "float32", "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "pinned_inputs": pins,
        "contract_readback_proof": {
            "required_exact_files": list(CONTRACT_NAMES),
            "tied_benchmark_contract.json": prior_hashes["tied_benchmark_contract.json"],
            "tied_probe_contract.json": prior_hashes["tied_probe_contract.json"],
            "environment_self_readback": "validated by canonical payload readback immediately after write and by every gated stage",
        },
        "train_only_selector_payload": {"TRAIN": train_rows, "TRAIN_digest": digest(train_rows), "note": "the calibration stage reads only this environment-contained TRAIN payload; no held-out arrays"},
    }
    record["environment_payload_digest"] = digest(record)
    return record


def contract_readback() -> dict:
    """Prove the exact three pre-run files read back before any gated work."""

    missing = [name for name in CONTRACT_NAMES if not (HERE / name).is_file()]
    if missing:
        raise SystemExit(f"BLOCKED-CONTRACT-READBACK: missing {missing}; run contracts")
    environment = json.loads((HERE / "environment.json").read_text(encoding="utf-8"))
    environment_digest = environment.pop("environment_payload_digest", None)
    if environment_digest is None or digest(environment) != environment_digest:
        raise SystemExit(
            "BLOCKED-CONTRACT-READBACK: environment payload digest changed"
        )
    proof = environment.get("contract_readback_proof", {})
    if proof.get("required_exact_files") != list(CONTRACT_NAMES):
        raise SystemExit("BLOCKED-CONTRACT-READBACK: exact three-file proof absent")
    observed = {}
    for name in CONTRACT_NAMES[:2]:
        raw = (HERE / name).read_bytes()
        observed[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if observed[name] != proof.get(name):
            raise SystemExit(f"BLOCKED-CONTRACT-READBACK: {name} bytes changed")
    environment_raw = (HERE / "environment.json").read_bytes()
    json.loads(environment_raw.decode("utf-8"))
    observed["environment.json"] = {"bytes": len(environment_raw), "sha256": hashlib.sha256(environment_raw).hexdigest()}
    return {"passed": True, "exact_files": list(CONTRACT_NAMES), "readbacks": observed}


def contract_fingerprint() -> dict:
    readback = contract_readback()
    return {
        name: row["sha256"] for name, row in readback["readbacks"].items()
    }


def stage_bootstrap_pins() -> None:
    copied = bootstrap_predecessor_pins()
    PINS_DIR.mkdir(parents=True, exist_ok=True)
    for uri, filename in NEW_PIN_FILES.items():
        path = PINS_DIR / filename
        if not path.exists():
            path.write_text(f"{PIN_PLACEHOLDER_MARKER}\nReplace this file with exact raw bytes from:\n{uri}\n", encoding="utf-8")
    print(f"predecessor pins copied: {len(copied)}; exact new pins required before contracts")


def stage_contracts() -> None:
    existing_contracts = [name for name in CONTRACT_NAMES if (HERE / name).exists()]
    downstream_artifacts = sorted(
        path.name
        for path in HERE.glob("*.json")
        if path.name not in CONTRACT_NAMES
    )
    raw_checkpoints = sorted(path.name for path in CHECKPOINTS.glob("*.json"))
    if existing_contracts or downstream_artifacts or raw_checkpoints:
        raise SystemExit(
            "BLOCKED-CONTRACT-FROZEN: contracts are create-once and cannot be "
            "regenerated after any contract/downstream state exists; "
            f"contracts={existing_contracts} downstream={downstream_artifacts} "
            f"checkpoints={raw_checkpoints}"
        )
    set_determinism()
    bootstrap_predecessor_pins()
    pins = pin_manifest()
    accepted, relation, benchmark, equality = accepted_benchmark_identity()
    audit = parameter_identity_audit()
    if not audit["passed"]:
        raise SystemExit(f"BLOCKED-MODEL-IDENTITY-AUDIT: {canonical(audit)}")
    split = {name: accepted["split"][name] for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")}
    benchmark_payload = {
        "module": "run_014_tied.py (stage: contracts)",
        "execution_identity": EXECUTION_IDENTITY,
        "dispatch_uri": DISPATCH_URI,
        "spec_uri": SPEC_URI,
        "owner_uri": OWNER_URI,
        "pinned_inputs": pins,
        "accepted_benchmark_source": {"path": "../014.03a-Code-attachments/benchmark_contract.json", "raw_sha256_expected": ACCEPTED_BENCHMARK_SHA256, "raw_sha256_observed": file_digest(PREDECESSOR_CONTRACT)},
        "benchmark_reconstruction_policy": "reproduce the predecessor deterministic selection only; no search for an alternative split",
        "benchmark_identity_audit": equality,
        "relation": relation.certification,
        "relation_reconstruction": {"blocks": [list(row) for row in relation.blocks], "blocks_sha256": digest([list(row) for row in relation.blocks]), "ordered_edges_sha256": digest([list(row) for row in relation.ordered_edges])},
        "holdout": {"novel_block_positions": benchmark.holdout["novel_block_positions"], "novel_block_triples": benchmark.holdout["novel_block_triples"], "seen_block_positions": benchmark.holdout["seen_block_positions"], "authorized_condition": "accepted fixed split retains all 14 certified components and isolates no Event; no global-connectivity search or reinterpretation"},
        "role_assignment": accepted["role_assignment"],
        "split": {**split, "sizes": accepted["split"]["sizes"], "digests": accepted["split"]["digests"], "query_pairs_disjoint_across_sets": True},
        "architecture": architecture_record(audit),
        "matched_primary_arms": PRIMARY_CONFIGS,
        "capacity_fallback": fallback_manifest(),
        "scored_seeds": list(SCORED_SEEDS),
        "training_horizon": HORIZON,
        "checkpoint_schedules": {"behavioral": list(BEHAVIORAL_CHECKPOINTS), "representation_probe": list(PROBE_CHECKPOINTS), "no_early_stopping": True},
        "temporal_definitions": temporal_definitions(),
        "thresholds": {"behavioral_and_probe": 0.95, "block_consistency": 0.90, "delayed_gap": 2048, "role_at_fit_below": 0.50, "primary_capacity": "at least 7/8 exact TRAIN fits by update 8192"},
        "baselines": PRE.baselines(relation, benchmark),
        "information_boundary": accepted["information_boundary"],
        "normalization_and_hash_rules": {"json": "indent=2, sort_keys=True, trailing newline", "payload_digest": "SHA-256 canonical compact sorted JSON UTF-8", "raw_file_digest": "SHA-256 exact bytes", "floats": "exact Python JSON float representations", "readback": "every write immediately reloaded and canonically compared"},
        "fences": ["all grokking is not thereby algebra discovery", "no claim of OT superiority to transformers or LLMs", "does not test H_native", "no language-scale or open-ended discovery claim", "preferred coordinates need not appear neuron-by-neuron", "low dimensionality is not necessary", "decodability alone does not prove causal use", "tied embeddings are not claimed to be the only route to a role-neutral relation", "no physical Event/Outcome semantics", "a negative does not establish that no architecture can grok the benchmark"],
    }
    probe_payload = probe_contract_record(pins)
    bench_rb = write_json(HERE / CONTRACT_NAMES[0], benchmark_payload)
    probe_rb = write_json(HERE / CONTRACT_NAMES[1], probe_payload)
    environment_payload = environment_record(pins, split["TRAIN"], {CONTRACT_NAMES[0]: {"bytes": bench_rb["bytes"], "sha256": bench_rb["sha256"]}, CONTRACT_NAMES[1]: {"bytes": probe_rb["bytes"], "sha256": probe_rb["sha256"]}})
    write_json(HERE / CONTRACT_NAMES[2], environment_payload)
    proof = contract_readback()
    print(f"contracts committed and exact-three readback passed: {canonical(proof)}")


def load_contract() -> dict:
    contract_readback()
    return json.loads((HERE / "tied_benchmark_contract.json").read_text(encoding="utf-8"))


def contract_split(contract: dict) -> dict:
    return {name: tuple(tuple(row) for row in contract["split"][name]) for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")}


def as_tensors(rows):
    return PRE.as_tensors(rows)


def build_probe(dim: int, seed: int = PROBE_SEED):
    return PRE.build_probe(dim, seed)


def fit_probe(z, splits: dict, blocks: list[list[int]], seen_positions: list[int]) -> dict:
    PRE.BUNDLE = "014.03a"
    return PRE.fit_probe(z, splits, blocks, seen_positions)


def stage_probe_controls() -> None:
    set_determinism()
    contract = load_contract()
    splits = contract_split(contract)
    blocks = contract["relation_reconstruction"]["blocks"]
    seen = contract["holdout"]["seen_block_positions"]
    controls = PRE.control_vectors(PRE.arm_control_codes())
    controls["random_gaussian"] = __import__("torch").randn(N_EVENTS, RANDOM_CONTROL_DIM, generator=__import__("torch").Generator().manual_seed(RANDOM_CONTROL_SEED))
    results = {}
    for name, vectors in controls.items():
        started = time.time()
        results[name] = fit_probe(vectors, splits, blocks, seen)
        results[name]["elapsed_seconds"] = round(time.time() - started, 3)
        results[name]["control_status"] = "primary" if name in ("B_certified_sfp", "C_scrambled", "random_gaussian") else "secondary predecessor control"
        print(name, results[name]["probe_role_accuracy"], results[name]["probe_novel_accuracy"])
    positive = results["B_certified_sfp"]
    passed = positive["probe_role_accuracy"] >= PROBE_GATE and positive["probe_novel_accuracy"] >= PROBE_GATE
    payload = {"module": "run_014_tied.py (stage: probe-controls)", "contract_readback": contract_readback(), "contract_fingerprint": contract_fingerprint(), "tied_probe_contract_sha256": file_digest(HERE / "tied_probe_contract.json"), "controls": results, "gate": {"rule": "certified summed Arm B ROLE and NOVEL must both be >= 0.95", "observed_role": positive["probe_role_accuracy"], "observed_novel": positive["probe_novel_accuracy"], "passed": passed, "status": "PASS" if passed else "BLOCKED-PROBE-REGRESSION"}}
    write_json(HERE / "probe_controls.json", payload)
    if not passed:
        raise SystemExit("BLOCKED-PROBE-REGRESSION")


def exact_fit_accuracy(model, left, right, target) -> float:
    import torch

    with torch.no_grad():
        return float((model(left, right).argmax(dim=1) == target).float().mean())


def calibrate_candidate_train_only(left, right, target, learning_rate: float, decay: float, seed: int) -> dict:
    import torch

    model = build_main_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=SELECT_BETAS, eps=SELECT_EPS, weight_decay=decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    first_exact = None
    final_loss = None
    for update in range(1, CALIBRATION_CAP + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(left, right), target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if exact_fit_accuracy(model, left, right, target) == 1.0:
            first_exact = update
            break
    return {"seed": seed, "first_exact_fit_update": first_exact, "reached_exact_fit": first_exact is not None, "final_train_loss": final_loss}


def select_capacity_fallback(left, right, target) -> dict:
    candidates = []
    for learning_rate in FALLBACK_LRS:
        for decay in FALLBACK_WDS:
            seeds = [calibrate_candidate_train_only(left, right, target, learning_rate, decay, seed) for seed in CALIBRATION_SEEDS]
            fits = [row["first_exact_fit_update"] for row in seeds if row["reached_exact_fit"]]
            candidates.append({"learning_rate": learning_rate, "weight_decay": decay, "seeds": seeds, "qualified_4_of_4": len(fits) == 4, "median_first_exact_fit_update": statistics.median(fits) if fits else None})
            print(f"lr={learning_rate} wd={decay} fits={len(fits)}/4")
    qualified = [row for row in candidates if row["qualified_4_of_4"]]
    ranked = sorted(qualified, key=lambda row: (0 if row["weight_decay"] != 0.0 else 1, -row["weight_decay"], row["median_first_exact_fit_update"], row["learning_rate"]))
    return {"candidates": candidates, "ranking": ranked, "selected": ranked[0] if ranked else None}


FORBIDDEN_SELECTOR_SUBSTRINGS = ("heldout", "role_test", "novel_test", "role_accuracy", "novel_accuracy", "probe", "sfp", "certif", "equival", "oracle", "third_of_pair", "block_of_pair", "relation", "arm_control", "holdout")
SELECTOR_ROOTS = ("select_capacity_fallback", "calibrate_candidate_train_only", "build_main_model", "exact_fit_accuracy", "xavier_uniform_")


def audit_selector_boundary() -> dict:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    reachable = set()
    queue = list(SELECTOR_ROOTS)
    while queue:
        name = queue.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                queue.append(node.func.id)
    findings = []
    identifiers = set()
    for name in sorted(reachable):
        for node in ast.walk(functions[name]):
            tokens = []
            if isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            elif isinstance(node, (ast.arg, ast.keyword)) and node.arg:
                tokens.append(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                tokens.append(node.value)
            for token in tokens:
                identifiers.add(token)
                for forbidden in FORBIDDEN_SELECTOR_SUBSTRINGS:
                    if forbidden in token.lower():
                        findings.append({"function": name, "token": token, "forbidden": forbidden})
    return {"method": "AST call-graph walk over selector roots and all identifiers/strings", "selector_roots": list(SELECTOR_ROOTS), "reachable_functions": sorted(reachable), "forbidden_substrings": list(FORBIDDEN_SELECTOR_SUBSTRINGS), "identifiers_scanned": len(identifiers), "violations": findings, "clean": not findings}


def stage_capacity_calibration() -> None:
    set_determinism()
    contract_readback()
    if (HERE / "trajectory_matched_regularized.json").exists() or (HERE / "trajectory_matched_zero_wd.json").exists() or any(CHECKPOINTS.glob("trajectory_*_seed*.json")):
        raise SystemExit("BLOCKED: capacity calibration must precede every scored trajectory")
    audit = audit_selector_boundary()
    if not audit["clean"]:
        raise SystemExit(f"BLOCKED: selector boundary violation {canonical(audit)}")
    environment = json.loads((HERE / "environment.json").read_text(encoding="utf-8"))
    train_only = environment["train_only_selector_payload"]
    if digest(train_only["TRAIN"]) != train_only["TRAIN_digest"]:
        raise SystemExit("BLOCKED: TRAIN-only selector payload digest mismatch")
    left, right, target = as_tensors(tuple(tuple(row) for row in train_only["TRAIN"]))
    started = time.time()
    result = select_capacity_fallback(left, right, target)
    selected = result["selected"]
    payload = {"module": "run_014_tied.py (stage: capacity-calibration)", "contract_fingerprint": contract_fingerprint(), "task_rule": "TRAIN-only 5x5 AdamW grid; candidate qualifies only on 4/4 exact TRAIN fit by update 8192; rank qualified only by nonzero wd preferred, wd descending, median first-fit ascending, lr ascending", "manifest": fallback_manifest(), "selector_boundary_audit": audit, "sealed_heldout_touched": False, "heldout_computed_loaded_or_logged": False, "results": result["candidates"], "ranking_qualified_only": [{key: row[key] for key in ("learning_rate", "weight_decay", "median_first_exact_fit_update")} for row in result["ranking"]], "selected_candidate": None if selected is None else {"optimizer": "torch.optim.AdamW", "learning_rate": selected["learning_rate"], "weight_decay": selected["weight_decay"], "betas": list(SELECT_BETAS), "eps": SELECT_EPS, "schedule": "constant", "full_batch": True}, "status": "QUALIFIED" if selected else "NO-QUALIFIED-CANDIDATE", "elapsed_seconds": round(time.time() - started, 3)}
    write_json(HERE / "capacity_fallback_selection.json", payload)


def require_execution_gates() -> tuple[dict, dict]:
    readback = contract_readback()
    fingerprint = contract_fingerprint()
    controls_path = HERE / "probe_controls.json"
    calibration_path = HERE / "capacity_fallback_selection.json"
    if not controls_path.is_file():
        raise SystemExit(
            "BLOCKED-PROBE-REGRESSION: passing probe_controls.json required"
        )
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if not controls["gate"]["passed"]:
        raise SystemExit(
            "BLOCKED-PROBE-REGRESSION: passing probe_controls.json required"
        )
    if controls.get("contract_fingerprint") != fingerprint:
        raise SystemExit("BLOCKED: probe controls are bound to different contracts")
    if not calibration_path.is_file():
        raise SystemExit(
            "BLOCKED: capacity_fallback_selection.json required before trajectories"
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("contract_fingerprint") != fingerprint:
        raise SystemExit("BLOCKED: calibration is bound to different contracts")
    return readback, calibration


def parameter_digest(model) -> str:
    return PRE.parameter_digest(model)


def effective_rank(matrix) -> float:
    return PRE.effective_rank(matrix)


def behavioral_metrics(model, tensors: dict) -> dict:
    import torch

    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        left, right, target = tensors["TRAIN"]
        scores = model(left, right)
        swapped = model(right, left)
        row = {"train_cross_entropy": float(loss_fn(scores, target)), "train_accuracy": float((scores.argmax(dim=1) == target).float().mean()), "input_swap_invariance_bit_exact": bool(torch.equal(scores, swapped)), "input_swap_max_abs_difference": float((scores - swapped).abs().max())}
        for name, key in (("ROLE_TEST", "role_test_accuracy"), ("NOVEL_TEST", "novel_test_accuracy")):
            split_left, split_right, split_target = tensors[name]
            row[key] = float((model(split_left, split_right).argmax(dim=1) == split_target).float().mean())
        row.update({"parameter_norm": float(torch.sqrt(sum((parameter.detach() ** 2).sum() for parameter in model.parameters()))), "shared_event_embedding_norm": float(model.event_emb.detach().norm()), "shared_event_embedding_effective_rank": effective_rank(model.event_emb), "candidate_bias_norm": float(model.cand_bias.detach().norm()), "trunk_norm": float(torch.sqrt((model.lin1.weight.detach() ** 2).sum() + (model.lin2.weight.detach() ** 2).sum())), "trunk_lin1_effective_rank": effective_rank(model.lin1.weight), "trunk_lin2_effective_rank": effective_rank(model.lin2.weight)})
    return row


def config_for_arm(arm: str, calibration: dict) -> dict:
    if arm in PRIMARY_CONFIGS:
        return PRIMARY_CONFIGS[arm]
    decision_path = HERE / "trajectory_capacity_fallback.json"
    if not decision_path.is_file():
        raise SystemExit("BLOCKED: capacity-decision must authorize fallback")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "AUTHORIZED" or not decision.get("invoked"):
        raise SystemExit("BLOCKED: fallback scored runs are not authorized")
    if calibration.get("selected_candidate") is None:
        raise SystemExit("BLOCKED: no calibrated fallback candidate")
    return calibration["selected_candidate"]


def run_trajectory(seed: int, arm: str, horizon: int = HORIZON, capture_updates: tuple[int, ...] = (), skip_probe: bool = False) -> dict:
    import torch

    set_determinism()
    _, calibration = require_execution_gates()
    contract = load_contract()
    config = config_for_arm(arm, calibration)
    splits = contract_split(contract)
    tensors = {name: as_tensors(rows) for name, rows in splits.items()}
    blocks = contract["relation_reconstruction"]["blocks"]
    seen = contract["holdout"]["seen_block_positions"]
    model = build_main_model(seed)
    audit = parameter_identity_audit(model)
    if not audit["passed"]:
        raise SystemExit("BLOCKED-MODEL-IDENTITY-AUDIT")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"], weight_decay=config["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    behavioral = []
    probes = []
    captures = {}
    started = time.time()

    def observe(update: int) -> None:
        if update in BEHAVIORAL_CHECKPOINTS:
            row = behavioral_metrics(model, tensors)
            row["update"] = update
            behavioral.append(row)
        if update in PROBE_CHECKPOINTS and not skip_probe:
            row = fit_probe(model.representation(), splits, blocks, seen)
            row["update"] = update
            probes.append(row)
        if update in capture_updates:
            captures[update] = {name: value.detach().clone() for name, value in model.state_dict().items()}

    observe(0)
    left, right, target = tensors["TRAIN"]
    for update in range(1, horizon + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(left, right), target)
        loss.backward()
        optimizer.step()
        observe(update)
    return {"seed": seed, "arm": arm, "config": config, "horizon": horizon, "initial_parameter_digest": parameter_digest(build_main_model(seed)), "final_parameter_digest": parameter_digest(model), "model_parameter_identity_audit": audit, "behavioral": behavioral, "probe": probes, "captures": captures, "probe_steps": PROBE_STEPS, "elapsed_seconds": round(time.time() - started, 3), "early_stopping": False}


def stage_trajectory(seed: int, arm: str) -> None:
    if seed not in SCORED_SEEDS:
        raise SystemExit(f"BLOCKED: seed must be one of {SCORED_SEEDS}")
    path = CHECKPOINTS / f"trajectory_{arm}_seed{seed}.json"
    if path.exists():
        raise SystemExit(
            "BLOCKED: scored record already exists; no rerun-until-favorable: "
            f"{path.name}"
        )
    result = run_trajectory(seed, arm)
    result.pop("captures")
    write_json(path, result)
    print(f"committed {path.name}; elapsed={result['elapsed_seconds']}s")


def load_raw_run(seed: int, arm: str) -> dict:
    path = CHECKPOINTS / f"trajectory_{arm}_seed{seed}.json"
    if not path.is_file():
        raise SystemExit(f"BLOCKED: missing required scored run {path.name}")
    run = json.loads(path.read_text(encoding="utf-8"))
    if (
        run.get("seed") != seed
        or run.get("arm") != arm
        or run.get("horizon") != HORIZON
        or run.get("early_stopping") is not False
        or run.get("probe_steps") != PROBE_STEPS
    ):
        raise SystemExit(f"BLOCKED: malformed scored run header {path.name}")
    calibration = json.loads(
        (HERE / "capacity_fallback_selection.json").read_text(encoding="utf-8")
    )
    expected_config = (
        PRIMARY_CONFIGS[arm]
        if arm in PRIMARY_CONFIGS
        else calibration.get("selected_candidate")
    )
    if run.get("config") != expected_config:
        raise SystemExit(f"BLOCKED: wrong optimizer config in {path.name}")
    if run.get("initial_parameter_digest") != parameter_digest(build_main_model(seed)):
        raise SystemExit(f"BLOCKED: wrong initial parameter digest in {path.name}")
    audit = run.get("model_parameter_identity_audit", {})
    if (
        not audit.get("passed")
        or audit.get("event_shaped_parameters") != ["event_emb"]
        or not all(audit.get("checks", {}).values())
    ):
        raise SystemExit(f"BLOCKED-MODEL-IDENTITY-AUDIT: {path.name}")
    behavioral = run.get("behavioral", [])
    probes = run.get("probe", [])
    if [row.get("update") for row in behavioral] != list(BEHAVIORAL_CHECKPOINTS):
        raise SystemExit(f"BLOCKED: behavioral schedule mismatch in {path.name}")
    if [row.get("update") for row in probes] != list(PROBE_CHECKPOINTS):
        raise SystemExit(f"BLOCKED: probe schedule mismatch in {path.name}")
    behavioral_fields = {
        "update",
        "train_cross_entropy",
        "train_accuracy",
        "role_test_accuracy",
        "novel_test_accuracy",
        "input_swap_invariance_bit_exact",
        "input_swap_max_abs_difference",
        "parameter_norm",
        "shared_event_embedding_norm",
        "shared_event_embedding_effective_rank",
        "candidate_bias_norm",
        "trunk_norm",
        "trunk_lin1_effective_rank",
        "trunk_lin2_effective_rank",
    }
    probe_fields = {
        "update",
        "probe_train_accuracy",
        "probe_role_accuracy",
        "probe_novel_accuracy",
        "seen_block_three_role_consistency",
        "all_block_three_role_consistency",
        "full_56_block_relation_exact",
        "probe_final_train_loss",
        "decoded_distinct_triples",
        "representation_dim",
    }
    if any(not behavioral_fields <= set(row) for row in behavioral):
        raise SystemExit(f"BLOCKED: behavioral schema mismatch in {path.name}")
    if any(not probe_fields <= set(row) for row in probes):
        raise SystemExit(f"BLOCKED: probe schema mismatch in {path.name}")
    if any(row["representation_dim"] != 64 for row in probes):
        raise SystemExit(f"BLOCKED: probe representation width mismatch in {path.name}")
    return run


def fit_by_8192(run: dict) -> bool:
    return any(row["update"] <= 8192 and row["train_accuracy"] == 1.0 for row in run["behavioral"])


def stage_capacity_decision() -> None:
    _, calibration = require_execution_gates()
    if any(CHECKPOINTS.glob("trajectory_capacity_fallback_seed*.json")):
        raise SystemExit("BLOCKED: fallback runs exist before capacity decision")
    counts = {arm: sum(fit_by_8192(load_raw_run(seed, arm)) for seed in SCORED_SEEDS) for arm in PRIMARY_ARMS}
    primary_qualified = any(count >= 7 for count in counts.values())
    base = {"module": "run_014_tied.py (stage: capacity-decision)", "decision_inputs": "primary TRAIN accuracy only through update 8192", "primary_fit_by_8192": counts, "criterion": "fallback forbidden if either primary has at least 7/8 exact TRAIN fits by update 8192", "calibration_selection_sha256": file_digest(HERE / "capacity_fallback_selection.json")}
    if primary_qualified:
        payload = {**base, "invoked": False, "status": "NOT-INVOKED", "reason": "at least one primary arm is capacity-qualified; fallback disallowed"}
    elif calibration.get("selected_candidate") is not None:
        payload = {**base, "invoked": True, "status": "AUTHORIZED", "reason": "neither primary capacity-qualified and a 4/4 TRAIN-only candidate exists", "authorized_runs": [{"seed": seed, "arm": "capacity_fallback"} for seed in SCORED_SEEDS], "config": calibration["selected_candidate"]}
    else:
        payload = {**base, "invoked": False, "status": "NO-TIED-CAPACITY", "reason": "neither primary capacity-qualified and no 4/4 calibration candidate exists"}
    write_json(HERE / "trajectory_capacity_fallback.json", payload)
    print(payload["status"])


def aggregate_arm(arm: str) -> tuple[list[dict], list[dict]]:
    behavioral = []
    probes = []
    for seed in SCORED_SEEDS:
        run = load_raw_run(seed, arm)
        probes.append({"seed": seed, "arm": arm, "checkpoints": run["probe"]})
        behavioral.append({key: value for key, value in run.items() if key not in ("probe",)})
    return behavioral, probes


def stage_aggregate() -> None:
    require_execution_gates()
    shared = {"dispatch_uri": DISPATCH_URI, "spec_uri": SPEC_URI, "tied_benchmark_contract_sha256": file_digest(HERE / "tied_benchmark_contract.json"), "capacity_fallback_selection_sha256": file_digest(HERE / "capacity_fallback_selection.json"), "scored_seeds": list(SCORED_SEEDS), "horizon": HORIZON, "behavioral_checkpoints": list(BEHAVIORAL_CHECKPOINTS), "no_early_stopping": True, "no_reruns_after_seeing_results": True}
    all_probes = []
    for arm in PRIMARY_ARMS:
        runs, probes = aggregate_arm(arm)
        all_probes.extend(probes)
        write_json(HERE / f"trajectory_{arm}.json", {**shared, "arm": arm, "runs": runs})
    decision_path = HERE / "trajectory_capacity_fallback.json"
    if not decision_path.is_file():
        raise SystemExit(
            "BLOCKED: capacity-decision artifact is mandatory before aggregate"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") == "AUTHORIZED":
        runs, probes = aggregate_arm("capacity_fallback")
        all_probes.extend(probes)
        write_json(
            decision_path,
            {
                **shared,
                "arm": "capacity_fallback",
                "invoked": True,
                "status": "AGGREGATED",
                "decision": decision,
                "runs": runs,
            },
        )
    elif decision.get("status") not in ("NOT-INVOKED", "NO-TIED-CAPACITY"):
        raise SystemExit(
            f"BLOCKED: invalid or already-consumed capacity decision {decision.get('status')}"
        )
    elif any(CHECKPOINTS.glob("trajectory_capacity_fallback_seed*.json")):
        raise SystemExit(
            "BLOCKED: fallback files exist despite a not-invoked decision"
        )
    write_json(HERE / "representation_probe_trajectory.json", {**shared, "tied_probe_contract_sha256": file_digest(HERE / "tied_probe_contract.json"), "probe_checkpoints": list(PROBE_CHECKPOINTS), "runs": all_probes})
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_digest(path)} for path in sorted(CHECKPOINTS.glob("trajectory_*_seed*.json"))}
    write_json(CHECKPOINTS / "manifest.json", {"purpose": "raw one-file-per-seed/arm evidence", "schema": {"behavioral": "exact old schedule with TRAIN/ROLE/NOVEL/swap/norm/rank fields", "probe": "exact old schedule with six frozen readout fields", "model_parameter_identity_audit": "must pass in every run"}, "files": files, "resume": "python run_014_tied.py trajectory --seed <seed> --arm <arm>; existing records are never overwritten; aggregate regenerates summaries", "self_audit": "aggregate requires all 16 primaries and, when authorized, exactly eight fallback records"})


def stage_transplant() -> None:
    import torch

    analysis_path = HERE / "analysis.json"
    if not analysis_path.is_file():
        raise SystemExit("BLOCKED: run analyze_014_tied.py pass 1 first")
    temporal = json.loads(analysis_path.read_text(encoding="utf-8"))["temporal_diagnostics"]
    contract = load_contract()
    tensors = {name: as_tensors(rows) for name, rows in contract_split(contract).items()}
    results = []
    for row in temporal:
        if row["arm"] != "matched_regularized" or row["t_fit"] is None:
            continue
        early = row["t_fit"]
        late = row["t_gen"] if row["t_gen"] is not None and row["t_gen"] != early else HORIZON
        replay = run_trajectory(row["seed"], "matched_regularized", horizon=late, capture_updates=(early, late), skip_probe=True)
        scored = load_raw_run(row["seed"], "matched_regularized")
        expected = {item["update"]: item for item in scored["behavioral"] if item["update"] <= late}
        actual = {item["update"]: item for item in replay["behavioral"]}
        mismatches = [update for update in sorted(expected) if actual.get(update) != expected[update]]
        if mismatches:
            raise SystemExit(f"BLOCKED: replay behavioral mismatch including shared norm at {mismatches}")
        captures = replay["captures"]
        model = build_main_model(row["seed"])

        def evaluate(
            representation_source: int,
            downstream_source: int,
            captures_for_seed=captures,
            model_for_seed=model,
        ) -> dict:
            state = {
                "event_emb": captures_for_seed[representation_source][
                    "event_emb"
                ].clone()
            }
            for key in (
                "lin1.weight",
                "lin1.bias",
                "lin2.weight",
                "lin2.bias",
                "cand_bias",
            ):
                state[key] = captures_for_seed[downstream_source][key].clone()
            model_for_seed.load_state_dict(state)
            parameter_identity = parameter_identity_audit(model_for_seed)
            if not parameter_identity["passed"]:
                raise SystemExit("BLOCKED-MODEL-IDENTITY-AUDIT during transplant")
            with torch.no_grad():
                metrics = {}
                for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST"):
                    left, right, target = tensors[name]
                    metrics[name] = float(
                        (
                            model_for_seed(left, right).argmax(dim=1) == target
                        ).float().mean()
                    )
                metrics["shared_event_embedding_norm"] = float(
                    model_for_seed.event_emb.norm()
                )
            return metrics

        results.append({"seed": row["seed"], "early_update": early, "late_update": late, "basis": "t_fit/t_gen" if late != HORIZON else "t_fit/final because no distinct t_gen", "replay_check": {"behavioral_fields_exact_including_shared_norm": True, "checkpoints_compared": len(expected), "mismatches": []}, "hybrids": {"early_representation_early_downstream": evaluate(early, early), "late_representation_late_downstream": evaluate(late, late), "late_representation_early_downstream": evaluate(late, early), "early_representation_late_downstream": evaluate(early, late)}})
    write_json(HERE / "component_transplants.json", {"module": "run_014_tied.py (stage: transplant)", "partition": {"REPRESENTATION": "event_emb only", "DOWNSTREAM": "lin1 + lin2 + candidate bias"}, "method": "deterministic replay and four combinations without retraining", "interpretation_fence": "component-level evidence only; coadaptation may make both hybrids fail and does not authorize another intervention", "runs": results})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue 014.05a tied Event identity staged experiment harness")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("bootstrap-pins", help="copy 014.03a pins and create fail-closed placeholders for the three new immutable pins")
    sub.add_parser("contracts", help="verify exact predecessor benchmark and write/read back the exact three tied contracts")
    sub.add_parser("probe-controls", help="run the frozen AdamW representation controls and gate trajectories")
    sub.add_parser("capacity-calibration", help="run the TRAIN-only 5x5 fallback grid on seeds 0..3")
    trajectory = sub.add_parser("trajectory", help="run one exact 65536-update scored seed/arm record")
    trajectory.add_argument("--seed", type=int, required=True, choices=SCORED_SEEDS)
    trajectory.add_argument("--arm", required=True, choices=ALL_ARMS)
    sub.add_parser("capacity-decision", help="require all 16 primaries and authorize or forbid eight fallback runs using TRAIN only")
    sub.add_parser("aggregate", help="aggregate all primary and any authorized fallback records plus checkpoint manifest")
    sub.add_parser("transplant", help="replay fitted matched-regularized seeds and evaluate four tied component transplants")
    args = parser.parse_args(argv)
    if args.stage == "bootstrap-pins":
        stage_bootstrap_pins()
    elif args.stage == "contracts":
        stage_contracts()
    elif args.stage == "probe-controls":
        stage_probe_controls()
    elif args.stage == "capacity-calibration":
        stage_capacity_calibration()
    elif args.stage == "trajectory":
        stage_trajectory(args.seed, args.arm)
    elif args.stage == "capacity-decision":
        stage_capacity_decision()
    elif args.stage == "aggregate":
        stage_aggregate()
    elif args.stage == "transplant":
        stage_transplant()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
