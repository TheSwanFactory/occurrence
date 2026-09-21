#!/usr/bin/env python3
"""Exact blind audit of component-preserving local linear saturation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import re
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

TASK_REVISION = "1ccda27f8265fe39e42cb5813f1a11b453e36065b1fe4f251ee7b9d092153f14"
INPUT_REVISION = "1339739309f484ad738449f8733b2df12869f970030cdc9878e60de0753fa772"
PARENT_REVISION = "9443b8c03e9491fad7a2ea39b47069d594e09b997d5eeececa1fe9276db7f5f9"
PACKAGE_PREFIX = "quilt+s3://protology#package=occurrence/gpt@"
TASK_PATH = (
    "issues/016-global-relational-completion/"
    "016.10-GPT-blind-component-preserving-local-saturation-Coder.md"
)
INPUT_PATH = (
    "issues/016-global-relational-completion/"
    "016.01-Code-attachments/observed-symmetry-input.json"
)
PARENT_RESULT_PATH = (
    "issues/016-global-relational-completion/"
    "016.08-Coder-component-locality-versus-symmetry-GPT.md"
)
PARENT_JSON_PATH = (
    "issues/016-global-relational-completion/"
    "016.08-Code-attachments/component_locality_audit.json"
)
PARENT_RUNNER_PATH = (
    "issues/016-global-relational-completion/"
    "016.08-Code-attachments/audit_component_locality.py"
)
TASK_URI = f"{PACKAGE_PREFIX}{TASK_REVISION}&path={TASK_PATH}"
INPUT_URI = f"{PACKAGE_PREFIX}{INPUT_REVISION}&path={INPUT_PATH}"
PARENT_RESULT_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_RESULT_PATH}"
PARENT_JSON_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_JSON_PATH}"
PARENT_RUNNER_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_RUNNER_PATH}"

EXPECTED_HASHES = {
    "task": "c2900e29124afb029e6fe4c09bc56d9bdc40f89ec6bc30c932a639598d058cb1",
    "input": "992616907f4d16018b579076374db5eea03f188b754cdee3b205fd9cc084de59",
    "parent_result": "32979723a52e7c459c10cece9832a3670de8d7282246af7946228fc8484eb193",
    "parent_json": "d184301132337fbfb583bd3b47ac6365559a50507f7cf50dc9fac69f90c18631",
    "parent_runner": "235c342fa2c2ffd09bbfde72e04294a4eaa53fc4d5f38922cbef85b6f3e6c05a",
}
INPUT_SCHEMA = "occurrence.gpt.016.observed-symmetry-completion-input.v1"
PARENT_SCHEMA = "occurrence.gpt.01608.component-locality-audit.v1"
REPORT_SCHEMA = "occurrence.gpt.01611.local-saturation-audit.v1"

ATTACHMENT_DIR = Path(__file__).resolve().parent
ISSUE_DIR = ATTACHMENT_DIR.parent
DEFAULT_JSON = ATTACHMENT_DIR / "local_saturation_audit.json"
DEFAULT_RESULT = ISSUE_DIR / "016.11-Coder-component-preserving-local-saturation-GPT.md"

INFORMATION_EXPOSURE = (
    "The immutable 016.10 task was located through mcp_open package revision and "
    "directory-browse calls. The manifest listed path names and its message restated "
    "the handed agreement, the second unresolved-pair orbit, and the observed-template "
    "relation that are all present in the authorized 016.08 parent result; it exposed "
    "no certified target, certified full relation, target reveal, or additional "
    "Owner-only scientific conclusion and was not material contamination. Repository "
    "inspection was restricted to git state, tool configuration, and publication "
    "mechanics; one historical publication helper contained an unrelated Issue-011 "
    "message with no Issue-016 completion information. The only Issue-016 scientific "
    "contents read were this immutable task, the pinned blind 016.01 packet, and the "
    "pinned 016.08 result, JSON certificate, and runner. The mutable Issue-016 README "
    "and 016.03, 016.06, and 016.09 contents were not read. No certified target, "
    "certified full relation, preferred global structure, external completion table, "
    "withheld coordinate system, or forbidden geometric terminology was exposed or used."
)

Event = int
Pair = tuple[int, int]
Block = tuple[int, int, int]
Permutation = tuple[int, ...]


class SaturationAuditError(RuntimeError):
    """An exact finite-audit invariant failed."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_quilt_source(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme != "quilt+s3" or not parsed.netloc:
        raise SaturationAuditError(f"invalid quilt+s3 source: {source}")
    parameters = parse_qs(parsed.fragment, strict_parsing=True)
    if set(parameters) != {"package", "path"}:
        raise SaturationAuditError("quilt+s3 source requires exactly package and path")
    package_values = parameters["package"]
    path_values = parameters["path"]
    if len(package_values) != 1 or len(path_values) != 1:
        raise SaturationAuditError("quilt+s3 source repeats package or path")
    if "@" not in package_values[0]:
        raise SaturationAuditError("quilt+s3 package must include an immutable revision")
    package_name, revision = package_values[0].rsplit("@", 1)
    if len(revision) != 64 or any(char not in "0123456789abcdef" for char in revision):
        raise SaturationAuditError("quilt+s3 revision must be 64 lowercase hex digits")
    try:
        import quilt3  # type: ignore[import-not-found]
    except ImportError as error:
        raise SaturationAuditError("immutable Quilt input requires quilt3") from error
    try:
        package = quilt3.Package.browse(
            package_name,
            registry=f"s3://{parsed.netloc}",
            top_hash=revision,
        )
        entry = package[path_values[0]]
        get_bytes = getattr(entry, "get_bytes", None)
        if callable(get_bytes):
            raw = get_bytes()
            if isinstance(raw, bytes):
                return raw
        with tempfile.TemporaryDirectory(prefix="occurrence-01611-") as directory:
            destination = Path(directory) / Path(path_values[0]).name
            fetched = entry.fetch(str(destination))
            candidates = [destination]
            if isinstance(fetched, (str, Path)):
                candidates.insert(0, Path(fetched))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.read_bytes()
    except SaturationAuditError:
        raise
    except Exception as error:
        raise SaturationAuditError(
            f"failed to read immutable Quilt source: {source}"
        ) from error
    raise SaturationAuditError(f"Quilt entry yielded no bytes: {source}")


def read_source(source: str) -> bytes:
    if source.startswith("quilt+s3://"):
        return read_quilt_source(source)
    return Path(source).read_bytes()


def read_and_gate(source: str, expected_hash: str, label: str) -> bytes:
    try:
        raw = read_source(source)
    except Exception as error:
        raise SaturationAuditError(f"could not read {label} from {source}: {error}") from error
    actual_hash = sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise SaturationAuditError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return raw


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SaturationAuditError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise SaturationAuditError(f"{label} must be a JSON object")
    return value


def extract_json_member(raw: bytes, key: str) -> Any:
    """Decode one top-level member without decoding later packet members."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SaturationAuditError("blind packet is not UTF-8") from error
    match = re.search(rf'"{re.escape(key)}"\s*:', text)
    if match is None:
        raise SaturationAuditError(f"blind packet lacks member {key!r}")
    start = match.end()
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise SaturationAuditError(f"could not decode packet member {key!r}") from error
    return value


def normalize_pair(values: Sequence[int]) -> Pair:
    if len(values) != 2 or len(set(values)) != 2:
        raise SaturationAuditError(f"not an unordered Event pair: {values}")
    return tuple(sorted(values))  # type: ignore[return-value]


def normalize_block(values: Sequence[int]) -> Block:
    if len(values) != 3 or len(set(values)) != 3:
        raise SaturationAuditError(f"not a three-Event block: {values}")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise SaturationAuditError(f"block contains a non-integer Event: {values}")
    return tuple(sorted(values))  # type: ignore[return-value]


def pairs(block: Sequence[int]) -> tuple[Pair, Pair, Pair]:
    return tuple(itertools.combinations(sorted(block), 2))  # type: ignore[return-value]


def is_linear(blocks: Iterable[Block]) -> bool:
    owned: set[Pair] = set()
    for block in blocks:
        for pair in pairs(block):
            if pair in owned:
                return False
            owned.add(pair)
    return True


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def canonical_component_signature(
    events: Sequence[int], component_blocks: Sequence[Block]
) -> tuple[tuple[Any, ...], dict[str, Any], dict[int, int]]:
    ordered_events = tuple(sorted(events))
    position = {event: index for index, event in enumerate(ordered_events)}
    local_blocks = tuple(
        sorted(tuple(sorted(position[event] for event in block)) for block in component_blocks)
    )
    best_encoding: tuple[Block, ...] | None = None
    best_images: Permutation | None = None
    bijections_tested = 0
    for images in itertools.permutations(range(len(ordered_events))):
        bijections_tested += 1
        encoding = tuple(
            sorted(tuple(sorted(images[index] for index in block)) for block in local_blocks)
        )
        if best_encoding is None or (encoding, images) < (best_encoding, best_images):
            best_encoding = encoding
            best_images = images
    if best_encoding is None or best_images is None:
        raise SaturationAuditError("could not canonicalize an observed component")
    signature_core = {
        "event_vertex_count": len(ordered_events),
        "block_vertex_count": len(local_blocks),
        "incidence_edge_count": 3 * len(local_blocks),
        "canonical_blocks": [list(block) for block in best_encoding],
    }
    compact = json.dumps(signature_core, sort_keys=True, separators=(",", ":")).encode()
    signature = {
        **signature_core,
        "serialization": compact.decode(),
        "sha256": sha256_bytes(compact),
        "all_event_bijections_tested": bijections_tested,
    }
    labels = {event: best_images[index] for index, event in enumerate(ordered_events)}
    key = (len(ordered_events), len(local_blocks), best_encoding)
    return key, signature, labels


def automorphisms(blocks: Sequence[Block], size: int) -> tuple[Permutation, ...]:
    block_set = set(blocks)
    return tuple(
        images
        for images in itertools.permutations(range(size))
        if {
            tuple(sorted(images[event] for event in block))
            for block in blocks
        }
        == block_set
    )


def partition_pair_orbits(
    pair_domain: Sequence[Pair], group: Sequence[Permutation]
) -> list[list[Pair]]:
    domain = set(pair_domain)
    remaining = set(pair_domain)
    result: list[list[Pair]] = []
    while remaining:
        seed = min(remaining)
        orbit = {
            tuple(sorted((permutation[seed[0]], permutation[seed[1]])))
            for permutation in group
        }
        if not orbit <= domain:
            raise SaturationAuditError("an observed-class pair orbit escaped its domain")
        result.append(sorted(orbit))
        remaining -= orbit
    return result


def enumerate_maximal_completions(
    events: Sequence[int], observed_blocks: Sequence[Block]
) -> dict[str, Any]:
    event_tuple = tuple(sorted(events))
    observed = tuple(sorted(observed_blocks))
    if not is_linear(observed):
        raise SaturationAuditError("observed component is not linear")
    all_internal = tuple(itertools.combinations(event_tuple, 3))
    observed_set = set(observed)
    initially_admissible = tuple(
        block
        for block in all_internal
        if block not in observed_set and is_linear((*observed, block))
    )
    compatibility_edges = [
        [list(left), list(right)]
        for left, right in itertools.combinations(initially_admissible, 2)
        if is_linear((*observed, left, right))
    ]
    maximal: list[dict[str, Any]] = []
    linear_subset_count = 0
    subset_count = 1 << len(initially_admissible)
    for mask in range(subset_count):
        additions = tuple(
            block
            for index, block in enumerate(initially_admissible)
            if mask & (1 << index)
        )
        family = tuple(sorted((*observed, *additions)))
        if not is_linear(family):
            continue
        linear_subset_count += 1
        family_set = set(family)
        addable = tuple(
            block
            for block in all_internal
            if block not in family_set and is_linear((*family, block))
        )
        if not addable:
            maximal.append(
                {
                    "completion_id": len(maximal),
                    "blocks": [list(block) for block in family],
                    "added_blocks": [list(block) for block in additions],
                    "final_block_count": len(family),
                    "linearity_verified": True,
                    "maximality_all_internal_3_subsets_tested": len(all_internal),
                    "remaining_addable_blocks": [],
                }
            )
    if not maximal:
        raise SaturationAuditError("finite enumeration found no maximal completion")
    return {
        "all_internal_3_subset_count": len(all_internal),
        "initially_admissible_internal_blocks": [
            list(block) for block in initially_admissible
        ],
        "initially_admissible_internal_block_count": len(initially_admissible),
        "A0_classification": (
            "empty"
            if not initially_admissible
            else "singleton"
            if len(initially_admissible) == 1
            else "multiple"
        ),
        "initial_compatibility_graph": {
            "nodes": [list(block) for block in initially_admissible],
            "edges": compatibility_edges,
            "edge_definition": (
                "Two initially admissible blocks are adjacent exactly when adding "
                "both to the observed component remains linear."
            ),
        },
        "candidate_addition_subsets_exhaustively_tested": subset_count,
        "linear_candidate_addition_subset_count": linear_subset_count,
        "nonlinear_candidate_addition_subset_count": subset_count - linear_subset_count,
        "maximal_completion_count": len(maximal),
        "block_count_histogram_across_maximal_completions": histogram(
            item["final_block_count"] for item in maximal
        ),
        "maximal_completions": maximal,
        "observed_component_already_maximal": not initially_admissible,
        "maximal_completion_unique": len(maximal) == 1,
        "finite_uniqueness_certificate": {
            "candidate_universe": "every 3-subset of the existing component Event set",
            "all_internal_3_subsets_tested": len(all_internal),
            "all_subsets_of_initially_admissible_blocks_tested": subset_count,
            "maximal_families_found": len(maximal),
            "every_reported_family_rechecked_against_every_internal_3_subset": True,
        },
    }


def build_raw_components(blocks: Sequence[Block]) -> list[dict[str, Any]]:
    universe = tuple(sorted({event for block in blocks for event in block}))
    adjacency = {event: set() for event in universe}
    for block in blocks:
        for left, right in itertools.combinations(block, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(universe)
    components: list[dict[str, Any]] = []
    while remaining:
        start = min(remaining)
        seen = {start}
        queue = deque([start])
        while queue:
            event = queue.popleft()
            for neighbor in sorted(adjacency[event]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        remaining -= seen
        events = tuple(sorted(seen))
        component_blocks = tuple(sorted(block for block in blocks if set(block) <= seen))
        key, signature, labels = canonical_component_signature(events, component_blocks)
        components.append(
            {
                "events": events,
                "blocks": component_blocks,
                "class_key": key,
                "observed_signature": signature,
                "observed_labels": labels,
            }
        )
    components.sort(key=lambda item: item["events"])
    return components


def map_blocks(blocks: Sequence[Sequence[int]], labels: dict[int, int]) -> list[list[int]]:
    return sorted(
        [sorted(labels[event] for event in block) for block in blocks]
    )


def complete(observed_triads: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Complete each observed component using only the observed triads."""
    observed = tuple(sorted(normalize_block(block) for block in observed_triads))
    if not observed or len(set(observed)) != len(observed):
        raise SaturationAuditError("observed triads are empty or duplicated")
    if not is_linear(observed):
        raise SaturationAuditError("observed triads are not linear")
    universe = tuple(sorted({event for block in observed for event in block}))
    raw_components = build_raw_components(observed)
    ordered_class_keys = sorted({item["class_key"] for item in raw_components})
    class_by_key = {key: index for index, key in enumerate(ordered_class_keys)}

    component_records: list[dict[str, Any]] = []
    for component_id, raw in enumerate(raw_components):
        enumeration = enumerate_maximal_completions(raw["events"], raw["blocks"])
        canonical_initial = map_blocks(
            enumeration["initially_admissible_internal_blocks"],
            raw["observed_labels"],
        )
        canonical_maximal_additions = sorted(
            map_blocks(item["added_blocks"], raw["observed_labels"])
            for item in enumeration["maximal_completions"]
        )
        profile_core = {
            "canonical_observed_blocks": raw["observed_signature"]["canonical_blocks"],
            "canonical_initially_admissible_blocks": canonical_initial,
            "canonical_maximal_added_block_sets": canonical_maximal_additions,
            "block_count_histogram": enumeration[
                "block_count_histogram_across_maximal_completions"
            ],
        }
        record = {
            "component_id": component_id,
            "observed_class_id": class_by_key[raw["class_key"]],
            "events": list(raw["events"]),
            "observed_blocks": [list(block) for block in raw["blocks"]],
            "observed_canonical_signature": raw["observed_signature"],
            "canonical_label_by_event": [
                {"event": event, "canonical_label": raw["observed_labels"][event]}
                for event in raw["events"]
            ],
            **enumeration,
            "canonical_completion_profile": profile_core,
            "canonical_completion_profile_sha256": sha256_bytes(canonical_json(profile_core)),
        }
        component_records.append(record)

    class_records: list[dict[str, Any]] = []
    for class_id in range(len(ordered_class_keys)):
        members = [
            component
            for component in component_records
            if component["observed_class_id"] == class_id
        ]
        profile_hashes = sorted(
            {component["canonical_completion_profile_sha256"] for component in members}
        )
        if len(profile_hashes) != 1:
            raise SaturationAuditError(
                f"observed class {class_id} has non-isomorphic completion profiles"
            )
        representative = members[0]
        class_records.append(
            {
                "observed_class_id": class_id,
                "component_ids": [component["component_id"] for component in members],
                "multiplicity": len(members),
                "observed_canonical_signature": representative[
                    "observed_canonical_signature"
                ],
                "canonical_completion_profile": representative[
                    "canonical_completion_profile"
                ],
                "canonical_completion_profile_sha256": profile_hashes[0],
                "all_members_have_same_profile_up_to_isomorphism": True,
            }
        )

    observed_class_by_completed_key = {
        tuple_class["observed_canonical_signature"]["sha256"]: tuple_class[
            "observed_class_id"
        ]
        for tuple_class in class_records
    }
    template_comparisons: list[dict[str, Any]] = []
    for class_record in class_records:
        members = [
            component
            for component in component_records
            if component["observed_class_id"] == class_record["observed_class_id"]
        ]
        representative = members[0]
        distinct_types: dict[str, dict[str, Any]] = {}
        for completion in representative["maximal_completions"]:
            completed_blocks = tuple(normalize_block(block) for block in completion["blocks"])
            _, signature, labels = canonical_component_signature(
                representative["events"], completed_blocks
            )
            distinct_types.setdefault(
                signature["sha256"],
                {
                    "signature": signature,
                    "labels": labels,
                    "completion_ids": [],
                },
            )["completion_ids"].append(completion["completion_id"])
        for type_id, type_record in enumerate(
            sorted(distinct_types.values(), key=lambda item: item["signature"]["sha256"])
        ):
            signature = type_record["signature"]
            matched_class = observed_class_by_completed_key.get(signature["sha256"])
            identity_map = [
                {"source_completed_label": label, "observed_template_label": label}
                for label in range(signature["event_vertex_count"])
            ]
            target_blocks = (
                None
                if matched_class is None
                else next(
                    item["observed_canonical_signature"]["canonical_blocks"]
                    for item in class_records
                    if item["observed_class_id"] == matched_class
                )
            )
            certificate_valid = (
                matched_class is not None
                and signature["canonical_blocks"] == target_blocks
            )
            template_comparisons.append(
                {
                    "source_observed_class_id": class_record["observed_class_id"],
                    "maximal_completion_type_id": type_id,
                    "representative_component_id": representative["component_id"],
                    "representative_completion_ids": type_record["completion_ids"],
                    "completed_canonical_signature": signature,
                    "isomorphic_to_already_observed_class": matched_class is not None,
                    "matched_observed_class_id": matched_class,
                    "new_completion_type": matched_class is None,
                    "explicit_isomorphism_certificate": (
                        None
                        if matched_class is None
                        else {
                            "canonical_event_map": identity_map,
                            "representative_event_to_observed_template_label": [
                                {
                                    "event": event,
                                    "observed_template_label": type_record["labels"][event],
                                }
                                for event in representative["events"]
                            ],
                            "source_completed_canonical_blocks": signature[
                                "canonical_blocks"
                            ],
                            "target_observed_canonical_blocks": target_blocks,
                            "block_image_equality_verified": certificate_valid,
                        }
                    ),
                }
            )

    choice_ranges = [
        range(component["maximal_completion_count"])
        for component in component_records
    ]
    global_count = 1
    for choices in choice_ranges:
        global_count *= len(choices)
    global_completions: list[dict[str, Any]] = []
    for choice_tuple in itertools.product(*choice_ranges):
        final_blocks = sorted(
            block
            for component, completion_id in zip(
                component_records, choice_tuple, strict=True
            )
            for block in component["maximal_completions"][completion_id]["blocks"]
        )
        added_blocks = sorted(
            block
            for component, completion_id in zip(
                component_records, choice_tuple, strict=True
            )
            for block in component["maximal_completions"][completion_id]["added_blocks"]
        )
        global_completions.append(
            {
                "global_completion_id": len(global_completions),
                "component_completion_choices": [
                    {
                        "component_id": component["component_id"],
                        "completion_id": completion_id,
                    }
                    for component, completion_id in zip(
                        component_records, choice_tuple, strict=True
                    )
                ],
                "final_blocks": final_blocks,
                "added_blocks": added_blocks,
                "final_block_count": len(final_blocks),
                "linearity_verified": is_linear(
                    tuple(normalize_block(block) for block in final_blocks)
                ),
            }
        )
    if len(global_completions) != global_count:
        raise SaturationAuditError("global Cartesian product count mismatch")

    unresolved_records: list[dict[str, Any]] = []
    for component in component_records:
        events = tuple(component["events"])
        observed_blocks = tuple(
            normalize_block(block) for block in component["observed_blocks"]
        )
        observed_pairs = {pair for block in observed_blocks for pair in pairs(block)}
        label_map = {
            item["event"]: item["canonical_label"]
            for item in component["canonical_label_by_event"]
        }
        for pair in itertools.combinations(events, 2):
            if pair in observed_pairs:
                continue
            per_completion_thirds: list[set[int]] = []
            for completion in component["maximal_completions"]:
                thirds = {
                    next(event for event in block if event not in pair)
                    for block in completion["added_blocks"]
                    if set(pair) <= set(block)
                }
                per_completion_thirds.append(thirds)
            possible = set().union(*per_completion_thirds)
            forced = set.intersection(*per_completion_thirds)
            unresolved_records.append(
                {
                    "component_id": component["component_id"],
                    "observed_class_id": component["observed_class_id"],
                    "pair": list(pair),
                    "canonical_pair": sorted(label_map[event] for event in pair),
                    "appears_in_some_added_block": bool(possible),
                    "appears_in_every_maximal_completion": all(
                        bool(thirds) for thirds in per_completion_thirds
                    ),
                    "possible_third_events": sorted(possible),
                    "forced_third_events": sorted(forced),
                    "canonical_possible_third_events": sorted(
                        label_map[event] for event in possible
                    ),
                    "canonical_forced_third_events": sorted(
                        label_map[event] for event in forced
                    ),
                }
            )

    unresolved_by_class: list[dict[str, Any]] = []
    for class_record in class_records:
        class_id = class_record["observed_class_id"]
        canonical_blocks = tuple(
            normalize_block(block)
            for block in class_record["observed_canonical_signature"]["canonical_blocks"]
        )
        size = class_record["observed_canonical_signature"]["event_vertex_count"]
        used_pairs = {pair for block in canonical_blocks for pair in pairs(block)}
        unresolved_pairs = tuple(
            pair for pair in itertools.combinations(range(size), 2) if pair not in used_pairs
        )
        group = automorphisms(canonical_blocks, size)
        pair_orbits = partition_pair_orbits(unresolved_pairs, group)
        representative_component = class_record["component_ids"][0]
        representative_records = {
            tuple(record["canonical_pair"]): record
            for record in unresolved_records
            if record["component_id"] == representative_component
        }
        orbit_records = []
        for orbit_id, pair_orbit in enumerate(pair_orbits):
            pair_records = []
            for pair in pair_orbit:
                record = representative_records[pair]
                pair_records.append(
                    {
                        "canonical_pair": list(pair),
                        "appears_in_some_added_block": record[
                            "appears_in_some_added_block"
                        ],
                        "appears_in_every_maximal_completion": record[
                            "appears_in_every_maximal_completion"
                        ],
                        "canonical_possible_third_events": record[
                            "canonical_possible_third_events"
                        ],
                        "canonical_forced_third_events": record[
                            "canonical_forced_third_events"
                        ],
                    }
                )
            orbit_records.append(
                {
                    "orbit_id": orbit_id,
                    "canonical_pairs": [list(pair) for pair in pair_orbit],
                    "pair_count": len(pair_orbit),
                    "pair_records": pair_records,
                    "some_completion_count": sum(
                        record["appears_in_some_added_block"] for record in pair_records
                    ),
                    "every_completion_count": sum(
                        record["appears_in_every_maximal_completion"]
                        for record in pair_records
                    ),
                }
            )
        unresolved_by_class.append(
            {
                "observed_class_id": class_id,
                "component_ids": class_record["component_ids"],
                "automorphism_group_order_used_only_after_completion_enumeration": len(group),
                "unresolved_pair_orbits": orbit_records,
            }
        )

    return {
        "constructor": {
            "name": "complete",
            "semantic_interface": "complete(observed_triads) -> completion_result",
            "constitution": {
                "component_preservation": (
                    "Every added block is contained in one connected component of the "
                    "typed incidence graph reconstructed from the observed triads."
                ),
                "local_linear_maximality": (
                    "Within every component, the final family is linear and no other "
                    "3-subset of that component's existing Event set can be added."
                ),
            },
            "symmetry_used_to_define_completion": False,
            "greedy_completion_used": False,
        },
        "observed_relation": {
            "event_count_inferred_from_observed_triads": len(universe),
            "observed_block_count": len(observed),
            "observed_blocks": [list(block) for block in observed],
            "observed_linearity_verified": True,
        },
        "typed_incidence_graph": {
            "event_vertex_count": len(universe),
            "block_vertex_count": len(observed),
            "incidence_edge_count": 3 * len(observed),
            "connected_component_count": len(component_records),
            "component_isomorphism_class_count": len(class_records),
            "construction": (
                "Bipartite Event/block incidence is reconstructed from observed triads "
                "alone; equivalent Event adjacency is used for deterministic BFS."
            ),
        },
        "observed_component_classes": class_records,
        "components": component_records,
        "global_component_preserving_maximal_completions": {
            "definition": (
                "The disjoint union of one exhaustively enumerated maximal component-"
                "local linear completion from each observed component."
            ),
            "count": global_count,
            "minimum_final_block_count": min(
                item["final_block_count"] for item in global_completions
            ),
            "maximum_final_block_count": max(
                item["final_block_count"] for item in global_completions
            ),
            "unique": global_count == 1,
            "all_completions": global_completions,
            "explicit_witnesses": global_completions[: min(2, global_count)],
            "exact_final_block_set_if_unique": (
                global_completions[0]["final_blocks"] if global_count == 1 else None
            ),
            "exact_added_block_set_if_unique": (
                global_completions[0]["added_blocks"] if global_count == 1 else None
            ),
        },
        "observed_template_relation": template_comparisons,
        "all_observed_internal_unresolved_pairs": {
            "definition": (
                "Every pair internal to an observed component but absent from every "
                "observed block; occurrence consequences use added blocks only."
            ),
            "record_count": len(unresolved_records),
            "records": unresolved_records,
            "by_observed_class_and_pair_orbit": unresolved_by_class,
            "orbit_compression_applied_after_exhaustive_completion": True,
        },
    }


def validate_parent_decomposition(
    completion: dict[str, Any], parent: dict[str, Any]
) -> None:
    parent_components = parent.get("observed_component_decomposition", {}).get(
        "components", []
    )
    own_components = completion["components"]
    if len(parent_components) != len(own_components):
        raise SaturationAuditError("component count disagrees with pinned 016.08")
    for own, expected in zip(own_components, parent_components, strict=True):
        checks = {
            "component_id": own["component_id"] == expected.get("component_id"),
            "class_id": own["observed_class_id"]
            == expected.get("canonical_class_id"),
            "events": own["events"] == expected.get("events"),
            "blocks": own["observed_blocks"] == expected.get("blocks"),
            "signature": own["observed_canonical_signature"]["sha256"]
            == next(
                item["signature"]["sha256"]
                for item in parent["observed_component_decomposition"][
                    "component_classes"
                ]
                if item["class_id"] == own["observed_class_id"]
            ),
        }
        if not all(checks.values()):
            raise SaturationAuditError(
                f"component {own['component_id']} disagrees with 016.08: {checks}"
            )


def evaluate_handed_queries(
    completion: dict[str, Any], queries: Sequence[Pair], parent: dict[str, Any]
) -> dict[str, Any]:
    global_completions = completion[
        "global_component_preserving_maximal_completions"
    ]["all_completions"]
    parent_records = {
        tuple(record["query"]): record
        for record in parent.get("handed_query_audit", {}).get("records", [])
    }
    if set(parent_records) != set(queries):
        raise SaturationAuditError("handed query domain disagrees with pinned 016.08")
    records = []
    for query in queries:
        per_completion: list[set[int]] = []
        for global_completion in global_completions:
            thirds = {
                next(event for event in block if event not in query)
                for block in global_completion["final_blocks"]
                if set(query) <= set(block)
            }
            per_completion.append(thirds)
        candidate = set().union(*per_completion)
        forced = set.intersection(*per_completion)
        parent_record = parent_records[query]
        parent_component = parent_record["C_component"]
        parent_hki = parent_record["C_HKI"]
        comparison = {
            "C_sat_equals_01608_C_component": sorted(candidate) == parent_component,
            "F_sat_equals_01608_C_component": sorted(forced) == parent_component,
            "C_sat_equals_01608_C_HKI": sorted(candidate) == parent_hki,
            "F_sat_equals_01608_C_HKI": sorted(forced) == parent_hki,
            "01608_sets_equal_singletons": (
                parent_component == parent_hki and len(parent_component) == 1
            ),
        }
        records.append(
            {
                "query": list(query),
                "C_sat": sorted(candidate),
                "F_sat": sorted(forced),
                "completed_in_every_global_completion": all(
                    bool(thirds) for thirds in per_completion
                ),
                "forced_completion_is_singleton": len(forced) == 1,
                "pinned_01608_C_component": parent_component,
                "pinned_01608_C_HKI": parent_hki,
                "comparison": comparison,
                "reproduces_01608_singleton": all(comparison.values()),
            }
        )
    return {
        "loaded_only_after_query_independent_completion_frozen": True,
        "target_labels_loaded": False,
        "query_count": len(records),
        "records": records,
        "candidate_size_histogram": histogram(len(item["C_sat"]) for item in records),
        "forced_size_histogram": histogram(len(item["F_sat"]) for item in records),
        "completed_in_every_count": sum(
            item["completed_in_every_global_completion"] for item in records
        ),
        "forced_singleton_count": sum(
            item["forced_completion_is_singleton"] for item in records
        ),
        "exact_01608_singleton_reproduction_count": sum(
            item["reproduces_01608_singleton"] for item in records
        ),
    }


def mechanical_constructor_audit(frozen_sha256: str) -> dict[str, Any]:
    signature = inspect.signature(complete)
    parameters = list(signature.parameters.values())
    forbidden = {
        "queries",
        "unseen_queries",
        "targets",
        "target_labels",
        "component_class",
        "desired_final_block_count",
        "preferred_component_template",
        "symmetry_generators",
    }
    return {
        "callable": "complete",
        "inspect_signature": str(signature),
        "parameter_count": len(parameters),
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "has_default": parameter.default is not inspect.Parameter.empty,
            }
            for parameter in parameters
        ],
        "sole_parameter_is_observed_triads": (
            len(parameters) == 1
            and parameters[0].name == "observed_triads"
            and parameters[0].default is inspect.Parameter.empty
        ),
        "forbidden_parameter_names_present": sorted(
            forbidden & set(signature.parameters)
        ),
        "closure_free_variables": list(complete.__code__.co_freevars),
        "completion_call_argument_count": 1,
        "completion_call_argument_source": (
            "the observed_triads top-level member decoded in isolation from the "
            "hash-gated blind packet"
        ),
        "packet_unseen_queries_member_decoded_before_call": False,
        "packet_unseen_queries_member_decoded_before_freeze": False,
        "completion_frozen_as_canonical_json_before_query_decode": True,
        "frozen_query_independent_completion_sha256": frozen_sha256,
        "event_order": [
            "hash-gate all five authorized immutable source byte strings",
            "decode only the blind packet observed_triads member",
            "call complete(observed_triads)",
            "freeze the complete query-independent result as canonical JSON bytes",
            "decode the blind packet unseen_queries member",
            "decode pinned 016.08 JSON and evaluate comparisons",
        ],
    }


def build_report(raw_sources: dict[str, bytes]) -> dict[str, Any]:
    observed_value = extract_json_member(raw_sources["input"], "observed_triads")
    if not isinstance(observed_value, list):
        raise SaturationAuditError("observed_triads member is not a list")

    query_independent = complete(observed_value)
    frozen_bytes = canonical_json(query_independent)
    frozen_sha256 = sha256_bytes(frozen_bytes)
    completion = json.loads(frozen_bytes)

    queries_value = extract_json_member(raw_sources["input"], "unseen_queries")
    packet = parse_json(raw_sources["input"], "blind input packet after completion freeze")
    parent = parse_json(raw_sources["parent_json"], "pinned 016.08 certificate")
    if packet.get("schema") != INPUT_SCHEMA:
        raise SaturationAuditError("blind packet schema mismatch")
    if parent.get("schema") != PARENT_SCHEMA:
        raise SaturationAuditError("pinned 016.08 certificate schema mismatch")
    queries = tuple(normalize_pair(query) for query in queries_value)
    if len(queries) != 24 or len(set(queries)) != 24:
        raise SaturationAuditError("blind packet does not contain 24 distinct queries")
    if packet.get("sizes") != {"observed_triads": 48, "unseen_queries": 24}:
        raise SaturationAuditError("blind packet size declaration mismatch")
    if packet.get("token_universe", {}).get("ids") != list(range(84)):
        raise SaturationAuditError("blind packet Event universe mismatch")
    observed_pairs = {
        pair
        for block in completion["observed_relation"]["observed_blocks"]
        for pair in pairs(block)
    }
    if any(query in observed_pairs for query in queries):
        raise SaturationAuditError("a handed query is already observed")

    validate_parent_decomposition(completion, parent)
    handed = evaluate_handed_queries(completion, queries, parent)
    constructor_audit = mechanical_constructor_audit(frozen_sha256)
    global_record = completion["global_component_preserving_maximal_completions"]
    any_nonunique = any(
        not component["maximal_completion_unique"]
        for component in completion["components"]
    )
    if any_nonunique:
        bin_code = "S0"
        bin_name = "NONUNIQUE-LOCAL-SATURATION"
    elif handed["exact_01608_singleton_reproduction_count"] != 24:
        bin_code = "S1"
        bin_name = "UNIQUE-LOCAL-SATURATION-NONMATCH"
    else:
        bin_code = "S2"
        bin_name = "UNIQUE-LOCAL-SATURATION-MATCH"

    expected_component_profile = {
        0: {
            "count": 8,
            "initial_admissible": 1,
            "maximal_completions": 1,
            "final_blocks": 4,
        },
        1: {
            "count": 6,
            "initial_admissible": 0,
            "maximal_completions": 1,
            "final_blocks": 4,
        },
    }
    profile_checks = {}
    for class_id, expected in expected_component_profile.items():
        members = [
            component
            for component in completion["components"]
            if component["observed_class_id"] == class_id
        ]
        profile_checks[f"class_{class_id}"] = (
            len(members) == expected["count"]
            and all(
                component["initially_admissible_internal_block_count"]
                == expected["initial_admissible"]
                and component["maximal_completion_count"]
                == expected["maximal_completions"]
                and component["maximal_completions"][0]["final_block_count"]
                == expected["final_blocks"]
                for component in members
            )
        )
    checks = {
        "all_five_authorized_source_hashes_gated": all(
            sha256_bytes(raw_sources[name]) == EXPECTED_HASHES[name]
            for name in EXPECTED_HASHES
        ),
        "constructor_has_only_observed_triads_parameter": constructor_audit[
            "sole_parameter_is_observed_triads"
        ]
        and not constructor_audit["forbidden_parameter_names_present"],
        "query_list_decoded_only_after_completion_freeze": True,
        "observed_relation_is_48_block_linear": (
            completion["observed_relation"]["observed_block_count"] == 48
            and completion["observed_relation"]["observed_linearity_verified"]
        ),
        "decomposition_is_14_components_2_classes": (
            completion["typed_incidence_graph"]["connected_component_count"] == 14
            and completion["typed_incidence_graph"][
                "component_isomorphism_class_count"
            ]
            == 2
        ),
        "class_multiplicities_are_8_and_6": [
            item["multiplicity"]
            for item in completion["observed_component_classes"]
        ]
        == [8, 6],
        "all_components_have_6_events": all(
            len(component["events"]) == 6 for component in completion["components"]
        ),
        "decomposition_matches_pinned_01608": True,
        "every_component_profile_isomorphic_within_observed_class": all(
            item["all_members_have_same_profile_up_to_isomorphism"]
            for item in completion["observed_component_classes"]
        ),
        "class_0_exact_profile": profile_checks["class_0"],
        "class_1_exact_profile": profile_checks["class_1"],
        "all_14_component_completions_unique": not any_nonunique,
        "all_reported_component_completions_linear_and_maximal": all(
            completion_item["linearity_verified"]
            and completion_item["remaining_addable_blocks"] == []
            and completion_item[
                "maximality_all_internal_3_subsets_tested"
            ]
            == 20
            for component in completion["components"]
            for completion_item in component["maximal_completions"]
        ),
        "global_completion_count_is_exact_cartesian_product": (
            global_record["count"]
            == len(global_record["all_completions"])
            == 1
        ),
        "final_count_is_derived_not_supplied": (
            global_record["minimum_final_block_count"]
            == global_record["maximum_final_block_count"]
            == 56
        ),
        "exactly_8_blocks_added": len(
            global_record["exact_added_block_set_if_unique"]
        )
        == 8,
        "all_observed_template_isomorphism_certificates_verify": all(
            item["explicit_isomorphism_certificate"] is not None
            and item["explicit_isomorphism_certificate"][
                "block_image_equality_verified"
            ]
            for item in completion["observed_template_relation"]
        ),
        "all_24_queries_completed_forced_singletons": (
            handed["completed_in_every_count"] == 24
            and handed["forced_singleton_count"] == 24
        ),
        "all_24_queries_reproduce_01608": (
            handed["exact_01608_singleton_reproduction_count"] == 24
        ),
        "all_internal_unresolved_pairs_exhausted": (
            completion["all_observed_internal_unresolved_pairs"]["record_count"]
            == 66
        ),
        "no_target_labels_loaded": not handed["target_labels_loaded"],
        "no_sampling_or_greedy_search": True,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise SaturationAuditError(f"mechanical checks failed: {failed}")

    return {
        "schema": REPORT_SCHEMA,
        "information_exposure": INFORMATION_EXPOSURE,
        "provenance": {
            "authorized_inputs": {
                name: {"pinned_uri": uri, "sha256": EXPECTED_HASHES[name]}
                for name, uri in (
                    ("task", TASK_URI),
                    ("input", INPUT_URI),
                    ("parent_result", PARENT_RESULT_URI),
                    ("parent_json", PARENT_JSON_URI),
                    ("parent_runner", PARENT_RUNNER_URI),
                )
            },
            "runner_sha256": sha256_file(Path(__file__)),
            "finite_computation_dependencies": (
                "Python standard library; quilt3 only for immutable input transport"
            ),
            "sampling": None,
        },
        "query_independent_construction_audit": constructor_audit,
        "query_independent_completion": completion,
        "handed_query_audit_after_freeze": handed,
        "interpretation": {
            "bin_code": bin_code,
            "bin_name": bin_name,
            "allowed_S2_statement": (
                "on the frozen observed 48-block substrate, component preservation "
                "plus local linear maximality is sufficient to reconstruct the same "
                "24 conditional completions previously selected by H_KI."
                if bin_code == "S2"
                else None
            ),
            "necessity_or_physical_justification_claimed": False,
        },
        "logical_fence": {
            "A_fact": "The 48 observed blocks have disconnected components.",
            "B_constitution": "New blocks are required to stay inside those components.",
            "C_constitution": "Each component is required to be locally maximal linear.",
            "D_consequence": (
                "The resulting completion, if unique, is forced by constitutions B+C."
            ),
            "non_implication": "A does not imply B or C.",
            "attribution": (
                "The result is generated by the 48 observed blocks under explicit "
                "component-preservation and local-maximality constitutions, not by "
                "the observed blocks alone."
            ),
        },
        "forbidden_additions_audit": {
            "certified_targets_used": False,
            "certified_full_relation_used": False,
            "desired_final_block_count_used": False,
            "global_maximality_over_84_events_used": False,
            "cross_component_blocks_used": False,
            "equal_degree_axiom_used": False,
            "automorphism_orbit_closure_used_as_completion_rule": False,
            "external_geometry_or_coordinates_used": False,
            "completion_table_used": False,
            "neural_learning_used": False,
            "symmetry_use": (
                "Only after exhaustive completion, to compress unresolved-pair "
                "consequences and verify isomorphic profiles."
            ),
        },
        "mechanical_checks": checks,
        "publication_contract": {
            "expected_manifest_entry_delta": 3,
            "paths": [
                "issues/016-global-relational-completion/016.11-Coder-component-preserving-local-saturation-GPT.md",
                "issues/016-global-relational-completion/016.11-Code-attachments/local_saturation_audit.json",
                "issues/016-global-relational-completion/016.11-Code-attachments/audit_local_saturation.py",
            ],
            "workflow": "occurrence",
            "publish_against": "latest",
        },
    }


def format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def format_blocks(blocks: Sequence[Sequence[int]]) -> str:
    return "{" + ", ".join(format_set(block) for block in blocks) + "}"


def render_markdown(report: dict[str, Any], json_bytes: bytes) -> bytes:
    completion = report["query_independent_completion"]
    global_record = completion["global_component_preserving_maximal_completions"]
    handed = report["handed_query_audit_after_freeze"]
    unresolved = completion["all_observed_internal_unresolved_pairs"]
    interpretation = report["interpretation"]
    lines = [
        "Information exposure:",
        "",
        report["information_exposure"],
        "",
        "# 016.11 — Component-preserving local-saturation result",
        "",
        "## Verdict",
        "",
        (
            f"**{interpretation['bin_code']} — {interpretation['bin_name']}.** Exact "
            "enumeration finds one maximal component-local linear completion for each "
            "of all 14 observed components, hence exactly one global component-preserving "
            "maximal completion. It contains 56 blocks, adding eight blocks to the "
            "observed 48. All 24 handed pairs are completed in that unique result and "
            "reproduce both pinned 016.08 singleton projections exactly."
        ),
        "",
        "## 1. Query-independent construction boundary",
        "",
        (
            "The constructor is `complete(observed_triads) -> completion_result`. Runtime "
            "inspection found exactly one required parameter, `observed_triads`, no "
            "closure variables, and no forbidden input parameter. The runner hash-gates "
            "the packet, decodes only `observed_triads`, runs and canonically freezes the "
            "entire completion, and only then decodes `unseen_queries` and the parent "
            "comparison records. The frozen query-independent construction SHA-256 is "
            f"`{report['query_independent_construction_audit']['frozen_query_independent_completion_sha256']}`."
        ),
        "",
        "## 2. Reconstructed components",
        "",
        (
            "From the 48 observed triads alone, the typed incidence graph has 84 Event "
            "vertices, 48 block vertices, 144 incidence edges, 14 connected components, "
            "and two canonical component isomorphism classes. Class 0 has eight members; "
            "class 1 has six; every component has six Events."
        ),
        "",
        "| Component | Observed class | Events | Observed blocks |",
        "|---:|---:|---|---|",
    ]
    for component in completion["components"]:
        lines.append(
            f"| {component['component_id']} | {component['observed_class_id']} | "
            f"`{format_set(component['events'])}` | "
            f"`{format_blocks(component['observed_blocks'])}` |"
        )
    lines.extend(
        [
            "",
            "## 3. Exact component-preserving linear completion",
            "",
            (
                "For each component C, the runner enumerates all 20 three-subsets of "
                "V(C). It forms A0(C) from every block individually compatible with the "
                "observed family, enumerates every subset of A0(C), retains each linear "
                "family, and calls it maximal only after checking that none of all 20 "
                "internal three-subsets can be added. No greedy order, target, query, "
                "desired count, component-class choice, template, or symmetry generator "
                "enters this definition."
            ),
            "",
            "## 4. Per-component exact classification",
            "",
            "| C | Class | A0(C) | M(C) count | Block-count histogram | Added-block sets (one per completion) | Already maximal | Unique |",
            "|---:|---:|---|---:|---|---|---|---|",
        ]
    )
    for component in completion["components"]:
        additions = [
            item["added_blocks"] for item in component["maximal_completions"]
        ]
        lines.append(
            f"| {component['component_id']} | {component['observed_class_id']} | "
            f"`{component['initially_admissible_internal_blocks']}` | "
            f"{component['maximal_completion_count']} | "
            f"`{component['block_count_histogram_across_maximal_completions']}` | "
            f"`{additions}` | "
            f"{str(component['observed_component_already_maximal']).lower()} | "
            f"{str(component['maximal_completion_unique']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                "Every class-0 component has exactly one initially admissible block and "
                "exactly one maximal completion, obtained by adding it. Every class-1 "
                "component has A0(C)=∅ and is already maximal. Canonical profile hashes "
                "are identical within each observed class; the JSON records all Event "
                "sets, observed blocks, A0 lists, complete M(C) families, and finite "
                "uniqueness certificates."
            ),
            "",
            "## 5. One-step versus maximal completion",
            "",
            (
                "No component has multiple initially admissible blocks: the A0 size "
                "histogram over components is `{0: 6, 1: 8}`. Thus every compatibility "
                "graph has zero edges—six have no nodes and eight have one node—and no "
                "choice dependence occurs. This conclusion is a result of exhaustive "
                "enumeration, not an assumption that all A0 blocks can be added together."
            ),
            "",
            "## 6. Global component-preserving maximal completions",
            "",
            (
                f"The exact Cartesian product count is **{global_record['count']}**. "
                f"Minimum and maximum final counts are both **{global_record['minimum_final_block_count']}**. "
                "The exact eight added blocks are:"
            ),
            "",
            f"`{format_blocks(global_record['exact_added_block_set_if_unique'])}`",
            "",
            "The exact unique final block set is:",
            "",
            f"`{format_blocks(global_record['exact_final_block_set_if_unique'])}`",
            "",
            "No total-block-count premise was supplied; 56 is derived by the finite search.",
            "",
            "## 7. Relation to already-observed component templates",
            "",
        ]
    )
    for comparison in completion["observed_template_relation"]:
        certificate = comparison["explicit_isomorphism_certificate"]
        lines.append(
            f"- Observed class {comparison['source_observed_class_id']} completion type "
            f"{comparison['maximal_completion_type_id']} is isomorphic to already-observed "
            f"class {comparison['matched_observed_class_id']}. The explicit canonical "
            f"Event map is `{certificate['canonical_event_map']}` and maps source blocks "
            f"`{certificate['source_completed_canonical_blocks']}` exactly to target blocks "
            f"`{certificate['target_observed_canonical_blocks']}`; verification is true."
        )
    lines.extend(
        [
            "",
            (
                "This comparison was performed only after maximal completions were "
                "enumerated; matching an observed class was not a completion requirement."
            ),
            "",
            "## 8. Handed queries, loaded only after completion freeze",
            "",
            "| Query | C_sat | F_sat | Every completion | Forced singleton | 016.08 C_component | 016.08 C_HKI | Match |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in handed["records"]:
        lines.append(
            f"| `{record['query']}` | `{record['C_sat']}` | `{record['F_sat']}` | "
            f"{str(record['completed_in_every_global_completion']).lower()} | "
            f"{str(record['forced_completion_is_singleton']).lower()} | "
            f"`{record['pinned_01608_C_component']}` | "
            f"`{record['pinned_01608_C_HKI']}` | "
            f"{str(record['reproduces_01608_singleton']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                "C_sat and F_sat are singleton for 24/24 queries; every query is "
                "completed in every global completion, and all 24 exactly match both "
                "pinned 016.08 sets. No target labels were loaded."
            ),
            "",
            "## 9. Every observed internal unresolved pair",
            "",
            (
                f"The runner exhausts **{unresolved['record_count']}** observed-internal "
                "unresolved pairs across all 14 components. The exact per-component "
                "records and third-Event sets are in the JSON. Canonical class/orbit "
                "profiles are:"
            ),
            "",
        ]
    )
    for class_record in unresolved["by_observed_class_and_pair_orbit"]:
        lines.append(f"### Observed class {class_record['observed_class_id']}")
        lines.append("")
        for orbit in class_record["unresolved_pair_orbits"]:
            lines.append(
                f"- Orbit {orbit['orbit_id']}: pairs `{orbit['canonical_pairs']}`; "
                f"some-added count {orbit['some_completion_count']}/{orbit['pair_count']}; "
                f"every-added count {orbit['every_completion_count']}/{orbit['pair_count']}; "
                f"records `{orbit['pair_records']}`."
            )
        lines.append("")
    lines.extend(
        [
            (
                "Local saturation therefore distinguishes the two class-0 unresolved-"
                "pair orbits without using the handed-query selection: one orbit is "
                "forced into the unique added block, while the other remains absent in "
                "the unique maximal completion. The class-1 unresolved orbit also "
                "remains absent because those components are already maximal."
            ),
            "",
            "## 10. Interpretation bin",
            "",
            f"**{interpretation['bin_code']} — {interpretation['bin_name']}**.",
            "",
            f"> {interpretation['allowed_S2_statement']}",
            "",
            (
                "This does not establish that component preservation or local maximality "
                "is necessary or physically justified."
            ),
            "",
            "## 11. Logical fence",
            "",
            "A. The 48 observed blocks have disconnected components. **Fact.**",
            "",
            "B. New blocks are required to stay inside those components. **Constitution.**",
            "",
            "C. Each component is required to be locally maximal linear. **Constitution.**",
            "",
            "D. The unique result is forced by B+C. **Consequence.**",
            "",
            (
                "A does not imply B or C. The completion is generated by the observed "
                "48-block substrate under the explicit B+C constitution, not by the "
                "observed blocks alone."
            ),
            "",
            "## 12. Forbidden-addition audit",
            "",
            (
                "The computation uses no certified target or full relation, desired "
                "count, global maximality, cross-component block, equal-degree axiom, "
                "automorphism closure as a completion rule, external geometry, withheld "
                "coordinate system, completion table, or neural learning. Automorphisms "
                "are used only after exhaustive enumeration to compress pair orbits and "
                "verify isomorphic profiles."
            ),
            "",
            "## 13. Exactness, reproduction, and publication",
            "",
            (
                "Every component search enumerates all subsets of A0(C), and every "
                "reported maximal family is independently checked against all 20 internal "
                "three-subsets. Every mechanical check in the JSON is true."
            ),
            "",
            "```bash",
            "python issues/016-global-relational-completion/016.11-Code-attachments/audit_local_saturation.py",
            "python issues/016-global-relational-completion/016.11-Code-attachments/audit_local_saturation.py --verify",
            "```",
            "",
            f"JSON certificate SHA-256: `{sha256_bytes(json_bytes)}`.",
            "",
            (
                "Publication contract: publish against latest with workflow "
                "`occurrence`; exactly three delivered paths; expected manifest entry-"
                "count delta **+3**."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_sources(args: argparse.Namespace) -> dict[str, bytes]:
    sources = {
        "task": args.task,
        "input": args.input,
        "parent_result": args.parent_result,
        "parent_json": args.parent_json,
        "parent_runner": args.parent_runner,
    }
    return {
        name: read_and_gate(source, EXPECTED_HASHES[name], name.replace("_", " "))
        for name, source in sources.items()
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    report = build_report(build_sources(args))
    json_bytes = canonical_json(report)
    markdown_bytes = render_markdown(report, json_bytes)
    json_path = Path(args.json)
    result_path = Path(args.result)
    if args.verify:
        failures = []
        for path, expected in ((json_path, json_bytes), (result_path, markdown_bytes)):
            if not path.is_file():
                failures.append(f"missing {path}")
            elif path.read_bytes() != expected:
                failures.append(f"byte mismatch {path}")
        return {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "interpretation": report["interpretation"]["bin_code"],
            "global_completion_count": report["query_independent_completion"][
                "global_component_preserving_maximal_completions"
            ]["count"],
            "handed_matches": report["handed_query_audit_after_freeze"][
                "exact_01608_singleton_reproduction_count"
            ],
            "json_sha256": sha256_bytes(json_bytes),
            "result_sha256": sha256_bytes(markdown_bytes),
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(json_bytes)
    result_path.write_bytes(markdown_bytes)
    return {
        "status": "COMPLETE",
        "json": str(json_path),
        "result": str(result_path),
        "interpretation": report["interpretation"]["bin_code"],
        "global_completion_count": report["query_independent_completion"][
            "global_component_preserving_maximal_completions"
        ]["count"],
        "handed_matches": report["handed_query_audit_after_freeze"][
            "exact_01608_singleton_reproduction_count"
        ],
        "json_sha256": sha256_bytes(json_bytes),
        "result_sha256": sha256_bytes(markdown_bytes),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact blind component-preserving local-saturation audit"
    )
    parser.add_argument("--task", default=TASK_URI)
    parser.add_argument("--input", default=INPUT_URI)
    parser.add_argument("--parent-result", default=PARENT_RESULT_URI)
    parser.add_argument("--parent-json", default=PARENT_JSON_URI)
    parser.add_argument("--parent-runner", default=PARENT_RUNNER_URI)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        outcome = execute(parse_args(argv))
    except SaturationAuditError as error:
        print(f"local-saturation audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["status"] in {"COMPLETE", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
