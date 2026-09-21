#!/usr/bin/env python3
"""Exact blind audit of component locality versus pinned symmetry constitutions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

TASK_REVISION = "d5c058ac65f69eec36563eaaf2a1f02faf94de5be9976e9c076b689ff5e2dd24"
INPUT_REVISION = "1339739309f484ad738449f8733b2df12869f970030cdc9878e60de0753fa772"
PARENT_REVISION = "ea987ac8f7632ce3134e292612137c4a5bab987bd2f10f94edf3b1728e00ffe0"
PACKAGE_PREFIX = "quilt+s3://protology#package=occurrence/gpt@"
TASK_PATH = (
    "issues/016-global-relational-completion/"
    "016.07-GPT-blind-component-locality-versus-symmetry-Coder.md"
)
INPUT_PATH = (
    "issues/016-global-relational-completion/"
    "016.01-Code-attachments/observed-symmetry-input.json"
)
PARENT_RESULT_PATH = (
    "issues/016-global-relational-completion/"
    "016.05-Coder-canonical-symmetry-factor-ablation-GPT.md"
)
PARENT_JSON_PATH = (
    "issues/016-global-relational-completion/"
    "016.05-Code-attachments/symmetry_factor_ablation.json"
)
PARENT_RUNNER_PATH = (
    "issues/016-global-relational-completion/"
    "016.05-Code-attachments/audit_symmetry_factors.py"
)
TASK_URI = f"{PACKAGE_PREFIX}{TASK_REVISION}&path={TASK_PATH}"
INPUT_URI = f"{PACKAGE_PREFIX}{INPUT_REVISION}&path={INPUT_PATH}"
PARENT_RESULT_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_RESULT_PATH}"
PARENT_JSON_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_JSON_PATH}"
PARENT_RUNNER_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_RUNNER_PATH}"

EXPECTED_HASHES = {
    "task": "b170a7166dbd3a232c8adcebc4e35da0c7986f89563fc95e51deccb3b4e1f449",
    "input": "992616907f4d16018b579076374db5eea03f188b754cdee3b205fd9cc084de59",
    "parent_result": "3253005670493019946ef7ddeebb6c80e9cb7c5739fa66f99e1fd27b4241a337",
    "parent_json": "6f4b954622c1edaa625486c68074cb184ecf3cc1110be81b0bf4d2eb38bcc8b3",
    "parent_runner": "c77db27abfc1ad3af6acabc625c028d9d9f77633d98f44633417d47ca33da796",
}
INPUT_SCHEMA = "occurrence.gpt.016.observed-symmetry-completion-input.v1"
PARENT_SCHEMA = "occurrence.gpt.01605.canonical-symmetry-factor-ablation.v1"
REPORT_SCHEMA = "occurrence.gpt.01608.component-locality-audit.v1"

ATTACHMENT_DIR = Path(__file__).resolve().parent
ISSUE_DIR = ATTACHMENT_DIR.parent
DEFAULT_JSON = ATTACHMENT_DIR / "component_locality_audit.json"
DEFAULT_RESULT = ISSUE_DIR / "016.08-Coder-component-locality-versus-symmetry-GPT.md"

INFORMATION_EXPOSURE = (
    "The immutable 016.07 task was first located through a package-manifest browse. "
    "Its manifest message restated only the H_KI factor result and locality-audit "
    "question already contained in the authorized task; it exposed no certified "
    "target, full relation, or additional Owner conclusion and was not material "
    "contamination. Repository inspection was then restricted to non-scientific "
    "publication conventions. The only scientific sources read were the immutable "
    "016.07 task, the pinned blind 016.01 packet, and the pinned 016.05 result, JSON "
    "certificate, and runner. The mutable Issue-016 README and all 016.03 and 016.06 "
    "content were not read. No certified target, certified full relation, preferred "
    "global structure, external geometric terminology, or withheld coordinate "
    "system was exposed or used."
)

Event = int
Pair = tuple[int, int]
Block = tuple[int, int, int]
Permutation = tuple[int, ...]


class LocalityAuditError(RuntimeError):
    """An exact finite-audit invariant failed."""


class Blocked01605Reproduction(LocalityAuditError):
    """The independent computation did not reproduce the pinned 016.05 sets."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_quilt_source(source: str) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme != "quilt+s3" or not parsed.netloc:
        raise LocalityAuditError(f"invalid quilt+s3 source: {source}")
    parameters = parse_qs(parsed.fragment, strict_parsing=True)
    if set(parameters) != {"package", "path"}:
        raise LocalityAuditError("quilt+s3 source requires exactly package and path")
    package_values = parameters["package"]
    path_values = parameters["path"]
    if len(package_values) != 1 or len(path_values) != 1:
        raise LocalityAuditError("quilt+s3 source repeats package or path")
    if "@" not in package_values[0]:
        raise LocalityAuditError("quilt+s3 package must include an immutable revision")
    package_name, revision = package_values[0].rsplit("@", 1)
    if len(revision) != 64 or any(char not in "0123456789abcdef" for char in revision):
        raise LocalityAuditError("quilt+s3 revision must be 64 lowercase hex digits")
    try:
        import quilt3  # type: ignore[import-not-found]
    except ImportError as error:
        raise LocalityAuditError("immutable Quilt input requires quilt3") from error
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
        with tempfile.TemporaryDirectory(prefix="occurrence-01608-") as directory:
            destination = Path(directory) / Path(path_values[0]).name
            fetched = entry.fetch(str(destination))
            candidates = [destination]
            if isinstance(fetched, (str, Path)):
                candidates.insert(0, Path(fetched))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.read_bytes()
    except LocalityAuditError:
        raise
    except Exception as error:
        raise LocalityAuditError(f"failed to read immutable Quilt source: {source}") from error
    raise LocalityAuditError(f"Quilt entry yielded no bytes: {source}")


def read_source(source: str) -> bytes:
    if source.startswith("quilt+s3://"):
        return read_quilt_source(source)
    return Path(source).read_bytes()


def read_and_gate(source: str, expected_hash: str, label: str) -> bytes:
    try:
        raw = read_source(source)
    except Exception as error:
        raise LocalityAuditError(f"could not read {label} from {source}: {error}") from error
    actual_hash = sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise LocalityAuditError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return raw


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalityAuditError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise LocalityAuditError(f"{label} must be a JSON object")
    return value


def pairs(block: Sequence[int]) -> tuple[Pair, Pair, Pair]:
    return tuple(itertools.combinations(sorted(block), 2))  # type: ignore[return-value]


def action_pair(permutation: Permutation, pair: Pair) -> Pair:
    return tuple(sorted((permutation[pair[0]], permutation[pair[1]])))  # type: ignore[return-value]


def action_block(permutation: Permutation, block: Block) -> Block:
    return tuple(sorted(permutation[event] for event in block))  # type: ignore[return-value]


def orbit(
    seed: Pair | Block,
    generators: Sequence[Permutation],
    action: Any,
) -> tuple[Any, ...]:
    seen = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        for generator in generators:
            image = action(generator, current)
            if image not in seen:
                seen.add(image)
                queue.append(image)
    return tuple(sorted(seen))


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def set_comparison(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    left_set = set(left)
    right_set = set(right)
    return {
        "equal": left_set == right_set,
        "left_minus_right": sorted(left_set - right_set),
        "right_minus_left": sorted(right_set - left_set),
        "symmetric_difference": sorted(left_set ^ right_set),
    }


def validate_packet(packet: dict[str, Any]) -> tuple[tuple[int, ...], tuple[Block, ...], tuple[Pair, ...]]:
    if packet.get("schema") != INPUT_SCHEMA:
        raise LocalityAuditError("blind packet schema mismatch")
    universe = tuple(packet.get("token_universe", {}).get("ids", []))
    if universe != tuple(range(84)):
        raise LocalityAuditError("blind packet universe is not exactly 0..83")
    blocks = tuple(
        tuple(sorted(item)) for item in packet.get("observed_triads", [])
    )
    queries = tuple(
        tuple(sorted(item)) for item in packet.get("unseen_queries", [])
    )
    if len(blocks) != 48 or len(set(blocks)) != 48:
        raise LocalityAuditError("blind packet does not contain 48 distinct blocks")
    if any(len(block) != 3 or len(set(block)) != 3 for block in blocks):
        raise LocalityAuditError("observed block is not a three-Event set")
    observed_pairs = [pair for block in blocks for pair in pairs(block)]
    if len(observed_pairs) != 144 or len(set(observed_pairs)) != 144:
        raise LocalityAuditError("observed relation is not pair-unique")
    if len(queries) != 24 or len(set(queries)) != 24:
        raise LocalityAuditError("blind packet does not contain 24 distinct queries")
    if any(len(query) != 2 or query in set(observed_pairs) for query in queries):
        raise LocalityAuditError("a handed query is malformed or already observed")
    if packet.get("sizes") != {"observed_triads": 48, "unseen_queries": 24}:
        raise LocalityAuditError("blind packet size declaration mismatch")
    return universe, blocks, queries


def canonical_component_signature(
    events: Sequence[int], component_blocks: Sequence[Block]
) -> tuple[tuple[Any, ...], dict[str, Any], dict[int, int]]:
    ordered_events = tuple(sorted(events))
    position = {event: index for index, event in enumerate(ordered_events)}
    local_blocks = tuple(
        sorted(tuple(sorted(position[event] for event in block)) for block in component_blocks)
    )
    best_encoding: tuple[tuple[int, int, int], ...] | None = None
    best_images: tuple[int, ...] | None = None
    for images in itertools.permutations(range(len(ordered_events))):
        encoding = tuple(
            sorted(
                tuple(sorted(images[index] for index in block))
                for block in local_blocks
            )
        )
        if best_encoding is None or (encoding, images) < (best_encoding, best_images):
            best_encoding = encoding
            best_images = images
    if best_encoding is None or best_images is None:
        raise LocalityAuditError("could not canonicalize a component")
    signature_core = {
        "event_vertex_count": len(ordered_events),
        "block_vertex_count": len(local_blocks),
        "incidence_edge_count": 3 * len(local_blocks),
        "canonical_blocks": [list(block) for block in best_encoding],
    }
    compact = json.dumps(
        signature_core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    signature = {
        **signature_core,
        "serialization": compact.decode("utf-8"),
        "sha256": sha256_bytes(compact),
    }
    key = (len(ordered_events), len(local_blocks), best_encoding)
    labels = {
        event: best_images[index] for index, event in enumerate(ordered_events)
    }
    return key, signature, labels


def automorphisms(blocks: Sequence[Block], size: int) -> tuple[Permutation, ...]:
    block_set = set(blocks)
    return tuple(
        images
        for images in itertools.permutations(range(size))
        if {action_block(images, block) for block in blocks} == block_set
    )


def build_component_decomposition(
    universe: Sequence[int], blocks: Sequence[Block]
) -> tuple[dict[str, Any], dict[int, int], dict[int, int], dict[int, dict[int, int]]]:
    adjacency = {event: set() for event in universe}
    for block in blocks:
        for left, right in itertools.combinations(block, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    raw_components = []
    remaining = set(universe)
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
        component_events = tuple(sorted(seen))
        component_blocks = tuple(
            sorted(block for block in blocks if set(block) <= seen)
        )
        key, signature, labels = canonical_component_signature(
            component_events, component_blocks
        )
        raw_components.append(
            {
                "events": component_events,
                "blocks": component_blocks,
                "key": key,
                "signature": signature,
                "labels": labels,
            }
        )
    raw_components.sort(key=lambda item: item["events"])
    ordered_keys = sorted({item["key"] for item in raw_components})
    class_by_key = {key: class_id for class_id, key in enumerate(ordered_keys)}
    event_to_component: dict[int, int] = {}
    event_to_class: dict[int, int] = {}
    canonical_labels: dict[int, dict[int, int]] = {}
    component_records = []
    for component_id, item in enumerate(raw_components):
        class_id = class_by_key[item["key"]]
        local_blocks = tuple(tuple(block) for block in item["signature"]["canonical_blocks"])
        local_automorphisms = automorphisms(local_blocks, len(item["events"]))
        for event in item["events"]:
            event_to_component[event] = component_id
            event_to_class[event] = class_id
        canonical_labels[component_id] = item["labels"]
        component_records.append(
            {
                "component_id": component_id,
                "canonical_class_id": class_id,
                "events": list(item["events"]),
                "blocks": [list(block) for block in item["blocks"]],
                "canonical_label_by_event": [
                    {"event": event, "canonical_label": item["labels"][event]}
                    for event in item["events"]
                ],
                "automorphism_order": len(local_automorphisms),
                "all_event_bijections_tested": len(local_automorphisms)
                + (len(tuple(itertools.permutations(range(len(item["events"]))))) - len(local_automorphisms)),
            }
        )
    class_records = []
    for class_id, key in enumerate(ordered_keys):
        members = [
            record
            for record in component_records
            if record["canonical_class_id"] == class_id
        ]
        source = next(item for item in raw_components if item["key"] == key)
        class_records.append(
            {
                "class_id": class_id,
                "signature": source["signature"],
                "component_ids": [record["component_id"] for record in members],
                "multiplicity": len(members),
                "component_automorphism_order": members[0]["automorphism_order"],
            }
        )
    event_records = []
    component_by_id = {
        record["component_id"]: record for record in component_records
    }
    for event in universe:
        component = component_by_id[event_to_component[event]]
        event_records.append(
            {
                "event": event,
                "component_id": component["component_id"],
                "canonical_class_id": component["canonical_class_id"],
                "component_events": component["events"],
                "component_observed_blocks": component["blocks"],
            }
        )
    decomposition = {
        "construction": (
            "Connected components are computed from the typed Event/observed-block "
            "incidence graph using only observed blocks. Component IDs are assigned "
            "by increasing Event tuple. Each class signature is the lexicographically "
            "least block encoding over every Event bijection; class IDs sort those "
            "complete signatures. Handed queries are not consulted."
        ),
        "typed_incidence_graph": {
            "event_vertex_count": len(universe),
            "block_vertex_count": len(blocks),
            "incidence_edge_count": 3 * len(blocks),
            "connected_component_count": len(component_records),
            "component_isomorphism_class_count": len(class_records),
        },
        "component_classes": class_records,
        "components": component_records,
        "event_component_records": event_records,
    }
    return decomposition, event_to_component, event_to_class, canonical_labels


def validate_parent_structure(
    decomposition: dict[str, Any], parent_report: dict[str, Any]
) -> None:
    if parent_report.get("schema") != PARENT_SCHEMA:
        raise LocalityAuditError("pinned 016.05 certificate schema mismatch")
    parent_factor = parent_report.get("canonical_factor_system", {})
    parent_components = parent_factor.get("components", [])
    own_components = decomposition["components"]
    if len(parent_components) != len(own_components):
        raise LocalityAuditError("component count disagrees with pinned 016.05")
    for own, parent in zip(own_components, parent_components, strict=True):
        comparisons = {
            "component_id": own["component_id"] == parent.get("component_id"),
            "canonical_class_id": own["canonical_class_id"]
            == parent.get("canonical_class_id"),
            "events": own["events"] == parent.get("events"),
            "blocks": own["blocks"]
            == sorted(parent.get("blocks", [])),
            "automorphism_order": own["automorphism_order"]
            == parent.get("automorphism_order"),
        }
        if not all(comparisons.values()):
            raise LocalityAuditError(
                f"component {own['component_id']} disagrees with pinned 016.05: "
                f"{comparisons}"
            )
    parent_classes = parent_factor.get("component_classes", [])
    own_classes = decomposition["component_classes"]
    if len(parent_classes) != len(own_classes):
        raise LocalityAuditError("component-class count disagrees with pinned 016.05")
    for own, parent in zip(own_classes, parent_classes, strict=True):
        if (
            own["class_id"] != parent.get("class_id")
            or own["component_ids"] != parent.get("component_ids")
            or own["signature"] != parent.get("signature")
            or own["component_automorphism_order"]
            != parent.get("component_automorphism_order")
        ):
            raise LocalityAuditError(
                f"component class {own['class_id']} disagrees with pinned 016.05"
            )


def local_candidates(
    query: Pair, universe: Sequence[int], observed_pairs: set[Pair]
) -> tuple[int, ...]:
    candidates = []
    for candidate in universe:
        if candidate in query:
            continue
        triple = tuple(sorted((*query, candidate)))
        if all(pair not in observed_pairs for pair in pairs(triple)):
            candidates.append(candidate)
    return tuple(candidates)


def orbit_is_linear(
    orbit_blocks: Sequence[Block], observed_pairs: set[Pair]
) -> bool:
    owned: set[Pair] = set()
    for block in orbit_blocks:
        for pair in pairs(block):
            if pair in observed_pairs or pair in owned:
                return False
            owned.add(pair)
    return True


class CandidateComputer:
    def __init__(
        self,
        universe: Sequence[int],
        blocks: Sequence[Block],
        generators: dict[str, Sequence[Permutation]],
    ) -> None:
        self.universe = tuple(universe)
        self.observed_pairs = {pair for block in blocks for pair in pairs(block)}
        self.generators = generators
        self.orbit_cache: dict[tuple[str, Block], tuple[Block, ...]] = {}

    def local(self, query: Pair) -> tuple[int, ...]:
        return local_candidates(query, self.universe, self.observed_pairs)

    def symmetry(self, name: str, query: Pair) -> tuple[int, ...]:
        surviving = []
        for candidate in self.local(query):
            triple = tuple(sorted((*query, candidate)))
            key = (name, triple)
            if key not in self.orbit_cache:
                self.orbit_cache[key] = orbit(
                    triple, self.generators[name], action_block
                )
            if orbit_is_linear(self.orbit_cache[key], self.observed_pairs):
                surviving.append(candidate)
        return tuple(surviving)


def subgroup_record(parent_report: dict[str, Any], name: str) -> dict[str, Any]:
    records = [
        record for record in parent_report.get("subgroups", []) if record.get("name") == name
    ]
    if len(records) != 1:
        raise LocalityAuditError(f"pinned 016.05 lacks unique subgroup {name}")
    return records[0]


def pinned_candidate_map(
    parent_report: dict[str, Any], name: str
) -> dict[Pair, tuple[int, ...]]:
    record = subgroup_record(parent_report, name)
    return {
        tuple(item["query"]): tuple(item["C_H"])
        for item in record.get("queries", [])
    }


def generator_system(parent_report: dict[str, Any]) -> dict[str, tuple[Permutation, ...]]:
    factor = parent_report["canonical_factor_system"]
    catalog = {
        record["name"]: tuple(record["images"])
        for record in factor["generator_catalog"]
    }
    for name, permutation in catalog.items():
        if sorted(permutation) != list(range(84)):
            raise LocalityAuditError(f"pinned generator is not bijective: {name}")
    classes = {record["class_id"]: record for record in factor["component_classes"]}
    names = {
        "H_II": [],
        "H_IK": classes[1]["levels"]["K"]["generator_names"],
        "H_KI": classes[0]["levels"]["K"]["generator_names"],
    }
    return {
        subgroup: tuple(catalog[name] for name in selected)
        for subgroup, selected in names.items()
    }


def classify_locality_vs_symmetry(
    locality: Sequence[int], symmetry: Sequence[int]
) -> str:
    locality_set = set(locality)
    symmetry_set = set(symmetry)
    if locality_set == symmetry_set:
        return "locality-equivalent"
    if symmetry_set < locality_set:
        return "symmetry-stronger"
    if locality_set < symmetry_set:
        return "locality-stronger"
    return "incomparable"


def build_pair_record(
    query: Pair,
    computer: CandidateComputer,
    event_to_component: dict[int, int],
    event_to_class: dict[int, int],
    class_events: dict[int, set[int]],
    component_events: dict[int, set[int]],
    canonical_labels: dict[int, dict[int, int]],
    pinned_maps: dict[str, dict[Pair, tuple[int, ...]]] | None = None,
) -> dict[str, Any]:
    same_component = event_to_component[query[0]] == event_to_component[query[1]]
    local = computer.local(query)
    h_ii = computer.symmetry("H_II", query)
    h_ik = computer.symmetry("H_IK", query)
    h_ki = computer.symmetry("H_KI", query)
    if same_component:
        component_id = event_to_component[query[0]]
        class_id = event_to_class[query[0]]
        c_class = tuple(item for item in local if item in class_events[class_id])
        c_component = tuple(item for item in local if item in component_events[component_id])
        template_pair = tuple(
            sorted(canonical_labels[component_id][event] for event in query)
        )
    else:
        component_id = None
        class_id = None
        c_class = None
        c_component = None
        template_pair = None
    if pinned_maps is not None:
        failures = {}
        for name, computed in (("H_II", h_ii), ("H_IK", h_ik), ("H_KI", h_ki)):
            expected = pinned_maps[name].get(query)
            if expected != computed:
                failures[name] = {
                    "computed": list(computed),
                    "pinned": None if expected is None else list(expected),
                }
        pinned_local = pinned_maps["C_local"].get(query)
        if pinned_local != local:
            failures["C_local"] = {
                "computed": list(local),
                "pinned": None if pinned_local is None else list(pinned_local),
            }
        if failures:
            raise Blocked01605Reproduction(
                f"BLOCKED-01605-REPRODUCTION at query {query}: {failures}"
            )
    e_class = None if c_class is None else set_comparison(c_class, h_ik)
    e_component = None if c_component is None else set_comparison(c_component, h_ki)
    record = {
        "query": list(query),
        "same_observed_component": same_component,
        "component_id": component_id,
        "canonical_class_id": class_id,
        "canonical_template_pair": None if template_pair is None else list(template_pair),
        "C_local": list(local),
        "C_local_size": len(local),
        "C_HII": list(h_ii),
        "C_HII_size": len(h_ii),
        "C_class": None if c_class is None else list(c_class),
        "C_class_size": None if c_class is None else len(c_class),
        "C_HIK": list(h_ik),
        "C_HIK_size": len(h_ik),
        "C_component": None if c_component is None else list(c_component),
        "C_component_size": None if c_component is None else len(c_component),
        "C_HKI": list(h_ki),
        "C_HKI_size": len(h_ki),
        "E_class": e_class,
        "E_component": e_component,
        "component_vs_HKI_classification": (
            None
            if c_component is None
            else classify_locality_vs_symmetry(c_component, h_ki)
        ),
        "local_to_component_eliminations": (
            None if c_component is None else sorted(set(local) - set(c_component))
        ),
        "local_to_HKI_eliminations": sorted(set(local) - set(h_ki)),
    }
    return record


def partition_pair_domain(
    domain: Sequence[Pair], generators: Sequence[Permutation]
) -> list[list[Pair]]:
    domain_set = set(domain)
    remaining = set(domain)
    result = []
    while remaining:
        seed = min(remaining)
        current = orbit(seed, generators, action_pair)
        if not set(current) <= domain_set:
            raise LocalityAuditError("pair orbit escaped unresolved internal-pair domain")
        result.append(list(current))
        remaining -= set(current)
    return result


def build_template_analysis(
    decomposition: dict[str, Any], relevant_classes: Sequence[int]
) -> list[dict[str, Any]]:
    analyses = []
    for class_id in relevant_classes:
        class_record = next(
            item
            for item in decomposition["component_classes"]
            if item["class_id"] == class_id
        )
        template_blocks = tuple(
            tuple(block) for block in class_record["signature"]["canonical_blocks"]
        )
        event_count = class_record["signature"]["event_vertex_count"]
        used_pairs = {pair for block in template_blocks for pair in pairs(block)}
        all_pairs = set(itertools.combinations(range(event_count), 2))
        unresolved_pairs = tuple(sorted(all_pairs - used_pairs))
        group = automorphisms(template_blocks, event_count)
        pair_orbits = partition_pair_domain(unresolved_pairs, group)
        block_degrees = {
            event: sum(event in block for block in template_blocks)
            for event in range(event_count)
        }
        orbit_records = []
        for orbit_id, pair_orbit in enumerate(pair_orbits):
            candidate_sets = {
                pair: tuple(
                    candidate
                    for candidate in range(event_count)
                    if candidate not in pair
                    and all(
                        item not in used_pairs
                        for item in pairs(tuple(sorted((*pair, candidate))))
                    )
                )
                for pair in pair_orbit
            }
            completed_blocks = sorted(
                {
                    tuple(sorted((*pair, candidate)))
                    for pair, candidates in candidate_sets.items()
                    for candidate in candidates
                }
            )
            pair_degree_profiles = sorted(
                [block_degrees[pair[0]], block_degrees[pair[1]]]
                for pair in pair_orbit
            )
            intersections = sorted(
                len(set(left) & set(right))
                for left, right in itertools.combinations(pair_orbit, 2)
            )
            orbit_records.append(
                {
                    "orbit_id": orbit_id,
                    "pairs": [list(pair) for pair in pair_orbit],
                    "orbit_size": len(pair_orbit),
                    "pair_endpoint_observed_block_degree_profiles": pair_degree_profiles,
                    "pairwise_pair_intersection_sizes": intersections,
                    "component_local_candidate_counts": {
                        f"{pair[0]}-{pair[1]}": len(candidate_sets[pair])
                        for pair in pair_orbit
                    },
                    "component_local_candidate_count_histogram": histogram(
                        len(candidates) for candidates in candidate_sets.values()
                    ),
                    "canonical_completed_blocks": [
                        list(block) for block in completed_blocks
                    ],
                }
            )
        analyses.append(
            {
                "canonical_class_id": class_id,
                "canonical_blocks": [list(block) for block in template_blocks],
                "event_count": event_count,
                "observed_block_count": len(template_blocks),
                "used_unordered_pair_count": len(used_pairs),
                "unused_unordered_pair_count": len(unresolved_pairs),
                "used_unordered_pairs": [list(pair) for pair in sorted(used_pairs)],
                "unresolved_internal_pairs": [
                    list(pair) for pair in unresolved_pairs
                ],
                "automorphism_group_order": len(group),
                "automorphisms": [list(permutation) for permutation in group],
                "observed_block_degree_by_event": block_degrees,
                "unresolved_pair_orbits": orbit_records,
            }
        )
    return analyses


def summarize_handed(
    records: Sequence[dict[str, Any]],
    parent_report: dict[str, Any],
    pinned_hww: dict[Pair, tuple[int, ...]],
) -> dict[str, Any]:
    component_counts = Counter(str(record["component_id"]) for record in records)
    class_counts = Counter(str(record["canonical_class_id"]) for record in records)
    class_matches = sum(record["E_class"]["equal"] for record in records)
    component_matches = sum(record["E_component"]["equal"] for record in records)
    class_mismatches = [
        {
            "query": record["query"],
            **record["E_class"],
        }
        for record in records
        if not record["E_class"]["equal"]
    ]
    component_mismatches = [
        {
            "query": record["query"],
            **record["E_component"],
        }
        for record in records
        if not record["E_component"]["equal"]
    ]
    hww_class_matches = sum(
        tuple(record["C_class"]) == pinned_hww[tuple(record["query"])]
        for record in records
    )
    hww_component_matches = sum(
        tuple(record["C_component"]) == pinned_hww[tuple(record["query"])]
        for record in records
    )
    return {
        "records": list(records),
        "query_distribution": {
            "over_components": dict(sorted(component_counts.items(), key=lambda item: int(item[0]))),
            "over_component_classes": dict(
                sorted(class_counts.items(), key=lambda item: int(item[0]))
            ),
        },
        "candidate_size_histograms": {
            key: histogram(record[f"{key}_size"] for record in records)
            for key in (
                "C_local",
                "C_HII",
                "C_class",
                "C_HIK",
                "C_component",
                "C_HKI",
            )
        },
        "primary_equivalence_tests": {
            "E_class": {
                "definition": "C_class(q) == C_HIK(q)",
                "exact_match_count": class_matches,
                "domain_count": len(records),
                "mismatches": class_mismatches,
            },
            "E_component": {
                "definition": "C_component(q) == C_HKI(q)",
                "exact_match_count": component_matches,
                "domain_count": len(records),
                "mismatches": component_mismatches,
            },
        },
        "singleton_on_all_applicable_queries": {
            "L_class": all(record["C_class_size"] == 1 for record in records),
            "L_component": all(
                record["C_component_size"] == 1 for record in records
            ),
        },
        "reproduces_pinned_H_WW_singleton_projection": {
            "L_class": {
                "exact_match_count": hww_class_matches,
                "domain_count": len(records),
            },
            "L_component": {
                "exact_match_count": hww_component_matches,
                "domain_count": len(records),
            },
        },
        "pinned_01605_reproduction": {
            "status": "PASS",
            "independently_reproduced_constitutions": ["H_II", "H_IK", "H_KI"],
            "exact_query_set_matches": {
                name: sum(
                    tuple(
                        record[
                            {"H_II": "C_HII", "H_IK": "C_HIK", "H_KI": "C_HKI"}[
                                name
                            ]
                        ]
                    )
                    == pinned_candidate_map(parent_report, name)[
                        tuple(record["query"])
                    ]
                    for record in records
                )
                for name in ("H_II", "H_IK", "H_KI")
            },
            "C_local_exact_matches": len(records),
        },
    }


def summarize_exhaustive(
    records: Sequence[dict[str, Any]], pair_orbits: Sequence[Sequence[Pair]]
) -> dict[str, Any]:
    equality_class = sum(record["E_class"]["equal"] for record in records)
    equality_component = sum(record["E_component"]["equal"] for record in records)
    singleton = [record["query"] for record in records if record["C_component_size"] == 1]
    impossible = [record["query"] for record in records if record["C_component_size"] == 0]
    ambiguous = [record["query"] for record in records if record["C_component_size"] > 1]
    classifications = Counter(
        record["component_vs_HKI_classification"] for record in records
    )
    mismatch_records = [
        {
            "query": record["query"],
            "component_id": record["component_id"],
            "canonical_template_pair": record["canonical_template_pair"],
            "classification": record["component_vs_HKI_classification"],
            "C_component": record["C_component"],
            "C_HKI": record["C_HKI"],
            "E_component": record["E_component"],
            "local_to_component_eliminations": record[
                "local_to_component_eliminations"
            ],
            "local_to_HKI_eliminations": record["local_to_HKI_eliminations"],
        }
        for record in records
        if not record["E_component"]["equal"]
    ]
    return {
        "definition": (
            "Every unordered Event pair internal to every component in a handed-query "
            "class, excluding every pair contained in an observed block."
        ),
        "total_unresolved_internal_pair_count": len(records),
        "records": list(records),
        "H_KI_pair_orbit_decomposition": [
            [list(pair) for pair in item] for item in pair_orbits
        ],
        "H_KI_pair_orbit_size_histogram": histogram(len(item) for item in pair_orbits),
        "candidate_size_histograms": {
            key: histogram(record[f"{key}_size"] for record in records)
            for key in (
                "C_local",
                "C_class",
                "C_component",
                "C_HIK",
                "C_HKI",
            )
        },
        "exact_equality_counts": {
            "C_class_equals_C_HIK": {
                "matches": equality_class,
                "domain": len(records),
            },
            "C_component_equals_C_HKI": {
                "matches": equality_component,
                "domain": len(records),
            },
        },
        "component_locality_outcomes": {
            "singleton_pairs": singleton,
            "singleton_count": len(singleton),
            "impossible_pairs": impossible,
            "impossible_count": len(impossible),
            "ambiguous_pairs": ambiguous,
            "ambiguous_count": len(ambiguous),
        },
        "component_locality_vs_H_KI_classification": {
            "counts": {
                key: classifications.get(key, 0)
                for key in (
                    "locality-equivalent",
                    "symmetry-stronger",
                    "locality-stronger",
                    "incomparable",
                )
            },
            "mismatches_with_full_elimination_sets": mismatch_records,
        },
    }


def build_report(raw_sources: dict[str, bytes], source_names: dict[str, str]) -> dict[str, Any]:
    packet = parse_json(raw_sources["input"], "blind input packet")
    parent_report = parse_json(raw_sources["parent_json"], "pinned 016.05 certificate")
    universe, blocks, handed_queries = validate_packet(packet)
    decomposition, event_to_component, event_to_class, canonical_labels = (
        build_component_decomposition(universe, blocks)
    )
    validate_parent_structure(decomposition, parent_report)
    generator_sets = generator_system(parent_report)
    computer = CandidateComputer(universe, blocks, generator_sets)
    pinned_maps = {
        name: pinned_candidate_map(parent_report, name)
        for name in ("H_II", "H_IK", "H_KI")
    }
    pinned_maps["C_local"] = {
        tuple(item["query"]): tuple(item["C_local"])
        for item in subgroup_record(parent_report, "H_II")["queries"]
    }
    pinned_hww = pinned_candidate_map(parent_report, "H_WW")
    components = decomposition["components"]
    component_events = {
        item["component_id"]: set(item["events"]) for item in components
    }
    class_events = {
        item["class_id"]: {
            event
            for component_id in item["component_ids"]
            for event in component_events[component_id]
        }
        for item in decomposition["component_classes"]
    }
    handed_records = [
        build_pair_record(
            query,
            computer,
            event_to_component,
            event_to_class,
            class_events,
            component_events,
            canonical_labels,
            pinned_maps,
        )
        for query in handed_queries
    ]
    if not all(record["same_observed_component"] for record in handed_records):
        raise LocalityAuditError("a handed query crosses observed components")
    relevant_classes = sorted(
        {record["canonical_class_id"] for record in handed_records}
    )
    relevant_components = sorted(
        component_id
        for class_id in relevant_classes
        for component_id in decomposition["component_classes"][class_id]["component_ids"]
    )
    observed_pairs = computer.observed_pairs
    unresolved_pairs = tuple(
        sorted(
            pair
            for component_id in relevant_components
            for pair in itertools.combinations(sorted(component_events[component_id]), 2)
            if pair not in observed_pairs
        )
    )
    exhaustive_records = [
        build_pair_record(
            query,
            computer,
            event_to_component,
            event_to_class,
            class_events,
            component_events,
            canonical_labels,
        )
        for query in unresolved_pairs
    ]
    h_ki_pair_orbits = partition_pair_domain(
        unresolved_pairs, generator_sets["H_KI"]
    )
    templates = build_template_analysis(decomposition, relevant_classes)
    handed = summarize_handed(handed_records, parent_report, pinned_hww)
    exhaustive = summarize_exhaustive(exhaustive_records, h_ki_pair_orbits)

    handed["locality_constitution_definitions"] = {
        "applicability": (
            "L_class and L_component are defined only when both query endpoints lie "
            "in the same observed connected component; otherwise each is "
            "not_applicable."
        ),
        "L_class": (
            "C_class(q) = {z in C_local(q): z lies in any observed component "
            "whose canonical isomorphism class equals the query component's class}."
        ),
        "L_component": (
            "C_component(q) = {z in C_local(q): z lies in the same observed "
            "connected component as both query endpoints}."
        ),
        "orbit_closure_used": False,
    }
    reproduction = handed["pinned_01605_reproduction"]
    reproduction[
        "pinned_constitutions_with_independently_recomputed_candidate_projections"
    ] = reproduction.pop("independently_reproduced_constitutions")

    class_one_events = sorted(class_events[1])
    mismatch_records = exhaustive[
        "component_locality_vs_H_KI_classification"
    ]["mismatches_with_full_elimination_sets"]
    template_orbits = templates[0]["unresolved_pair_orbits"]
    singleton_template_orbit = next(
        item
        for item in template_orbits
        if item["component_local_candidate_count_histogram"] == {"1": 3}
    )
    impossible_template_orbit = next(
        item
        for item in template_orbits
        if item["component_local_candidate_count_histogram"] == {"0": 3}
    )
    singleton_template_pairs = {
        tuple(pair) for pair in singleton_template_orbit["pairs"]
    }
    impossible_template_pairs = {
        tuple(pair) for pair in impossible_template_orbit["pairs"]
    }

    def event_orbit_size(
        event: int, generators: Sequence[Permutation]
    ) -> int:
        seen = {event}
        queue = deque([event])
        while queue:
            current = queue.popleft()
            for generator in generators:
                image = generator[current]
                if image not in seen:
                    seen.add(image)
                    queue.append(image)
        return len(seen)

    def support_component_ids(
        generators: Sequence[Permutation], expected_class: int
    ) -> set[int] | None:
        supports = set()
        for generator in generators:
            moved = {
                event
                for event, image in enumerate(generator)
                if image != event
            }
            component_ids = {event_to_component[event] for event in moved}
            if (
                not moved
                or len(component_ids) != 1
                or {event_to_class[event] for event in moved} != {expected_class}
            ):
                return None
            supports.update(component_ids)
        return supports

    observed_block_set = set(blocks)
    selected_generators_preserve_B0 = all(
        {action_block(generator, block) for block in blocks}
        == observed_block_set
        for generators in generator_sets.values()
        for generator in generators
    )
    h_ki_supports = support_component_ids(generator_sets["H_KI"], 0)
    h_ik_supports = support_component_ids(generator_sets["H_IK"], 1)
    proof_checks = {
        "every_handed_pair_is_in_singleton_template_orbit": all(
            tuple(record["canonical_template_pair"]) in singleton_template_pairs
            for record in handed_records
        ),
        "every_component_mismatch_is_in_impossible_template_orbit": all(
            tuple(record["canonical_template_pair"]) in impossible_template_pairs
            for record in mismatch_records
        ),
        "singleton_orbit_pairs_overlap_pairwise": singleton_template_orbit[
            "pairwise_pair_intersection_sizes"
        ]
        == [1, 1, 1],
        "impossible_orbit_pairs_are_pairwise_disjoint": impossible_template_orbit[
            "pairwise_pair_intersection_sizes"
        ]
        == [0, 0, 0],
        "selected_generators_preserve_B0": selected_generators_preserve_B0,
        "H_KI_generators_have_independent_class_0_component_supports": (
            h_ki_supports
            == set(decomposition["component_classes"][0]["component_ids"])
        ),
        "H_IK_generators_have_independent_class_1_component_supports": (
            h_ik_supports
            == set(decomposition["component_classes"][1]["component_ids"])
        ),
        "H_KI_fixes_class_1_pointwise": all(
            event_orbit_size(event, generator_sets["H_KI"]) == 1
            for event in class_events[1]
        ),
        "H_KI_has_size_3_event_orbits_on_class_0": all(
            event_orbit_size(event, generator_sets["H_KI"]) == 3
            for event in class_events[0]
        ),
        "H_IK_fixes_class_0_pointwise": all(
            event_orbit_size(event, generator_sets["H_IK"]) == 1
            for event in class_events[0]
        ),
        "H_IK_is_transitive_on_each_class_1_component": all(
            event_orbit_size(event, generator_sets["H_IK"]) == 6
            for event in class_events[1]
        ),
        "every_exhaustive_pair_has_class_locality_equivalence": all(
            record["E_class"]["equal"] for record in exhaustive_records
        ),
        "every_component_mismatch_is_locality_stronger": all(
            record["classification"] == "locality-stronger"
            for record in mismatch_records
        ),
        "every_component_mismatch_has_no_local_candidate": all(
            record["C_component"] == [] for record in mismatch_records
        ),
        "every_component_mismatch_HKI_set_is_exactly_class_1": all(
            record["C_HKI"] == class_one_events for record in mismatch_records
        ),
        "all_relevant_components_exhaustively_checked": len(relevant_components) == 8
        and {record["component_id"] for record in exhaustive_records}
        == set(relevant_components),
    }
    symmetry_analysis = {
        "classification_domain": "all unresolved internal pairs in handed-query component classes",
        "counts": exhaustive["component_locality_vs_H_KI_classification"]["counts"],
        "structural_condition": (
            "The handed pairs belong to the unresolved template orbit whose three "
            "pairs overlap pairwise and form one three-Event set. An external Event "
            "fixed by H_KI then makes repeated cross-pairs across the extension orbit, "
            "so symmetry rejects it exactly as component locality does. The unhanded "
            "counterexample orbit is a three-pair matching: its pairs are pairwise "
            "disjoint. With any of the 36 class-1 Events fixed by H_KI, the three "
            "extension blocks share only that one Event and remain pair-unique. H_KI "
            "therefore keeps all 36 while component locality keeps none."
        ),
        "exact_template_proof": [
            (
                "For canonical blocks {0,1,2}, {0,3,4}, {1,3,5}, the six unused "
                "pairs split under the complete order-6 automorphism group into "
                "{0,5},{1,4},{2,3} and {2,4},{2,5},{4,5}."
            ),
            (
                "For the matching orbit {0,5},{1,4},{2,3}, no template Event is "
                "locally admissible. For every class-1 Event z, H_KI fixes z and the "
                "extension orbit is {0,5,z},{1,4,z},{2,3,z}; these blocks share no "
                "unordered pair, so all 36 such z survive. A candidate in another "
                "class-0 component moves independently, causing repeated query or "
                "cross-pairs, so no other candidate survives. Thus C_component is "
                "empty and C_HKI is exactly the class-1 Event set."
            ),
            (
                "For the overlapping orbit {2,4},{2,5},{4,5}, component locality "
                "allows exactly the remaining Event and yields the invariant block "
                "{2,4,5}. Any external fixed z produces blocks with repeated pairs "
                "{2,z}, {4,z}, or {5,z}; independently moving external candidates "
                "also collide. Thus C_component equals C_HKI and is singleton."
            ),
            (
                "For H_IK, every class-0 Event is fixed, so every locally admissible "
                "class-0 candidate has a one-block linear orbit. Every class-1 Event "
                "moves under its complete component automorphism group while the "
                "query is fixed, repeating the query pair. Hence C_class equals "
                "C_HIK on every one of the 48 pairs."
            ),
        ],
        "exhaustive_proof_checks": proof_checks,
    }
    checks = {
        "all_five_input_hashes_gated": all(
            sha256_bytes(raw_sources[name]) == EXPECTED_HASHES[name]
            for name in EXPECTED_HASHES
        ),
        "blind_packet_validated": len(universe) == 84
        and len(blocks) == 48
        and len(handed_queries) == 24,
        "component_decomposition_14_by_2": decomposition["typed_incidence_graph"][
            "connected_component_count"
        ]
        == 14
        and decomposition["typed_incidence_graph"][
            "component_isomorphism_class_count"
        ]
        == 2,
        "component_decomposition_matches_01605": True,
        "all_84_events_have_component_records": len(
            decomposition["event_component_records"]
        )
        == 84,
        "all_handed_queries_internal": all(
            record["same_observed_component"] for record in handed_records
        ),
        "01605_reproduction_pass": handed["pinned_01605_reproduction"]["status"]
        == "PASS",
        "handed_E_class_24_of_24": handed["primary_equivalence_tests"]["E_class"][
            "exact_match_count"
        ]
        == 24,
        "handed_E_component_24_of_24": handed["primary_equivalence_tests"][
            "E_component"
        ]["exact_match_count"]
        == 24,
        "exhausted_exactly_48_internal_pairs": len(exhaustive_records) == 48,
        "exhaustive_class_equivalence_48_of_48": exhaustive[
            "exact_equality_counts"
        ]["C_class_equals_C_HIK"]["matches"]
        == 48,
        "exhaustive_component_equivalence_24_of_48": exhaustive[
            "exact_equality_counts"
        ]["C_component_equals_C_HKI"]["matches"]
        == 24,
        "H_KI_has_16_three_pair_orbits": len(h_ki_pair_orbits) == 16
        and all(len(item) == 3 for item in h_ki_pair_orbits),
        "component_locality_24_singleton_24_impossible_0_ambiguous": exhaustive[
            "component_locality_outcomes"
        ]["singleton_count"]
        == 24
        and exhaustive["component_locality_outcomes"]["impossible_count"] == 24
        and exhaustive["component_locality_outcomes"]["ambiguous_count"] == 0,
        "classification_24_equivalent_24_locality_stronger": exhaustive[
            "component_locality_vs_H_KI_classification"
        ]["counts"]
        == {
            "locality-equivalent": 24,
            "symmetry-stronger": 0,
            "locality-stronger": 24,
            "incomparable": 0,
        },
        "template_proof_exhaustively_verified": all(proof_checks.values()),
        "no_stochastic_sampling": True,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise LocalityAuditError(f"mechanical checks failed: {failed}")
    return {
        "schema": REPORT_SCHEMA,
        "information_exposure": INFORMATION_EXPOSURE,
        "provenance": {
            "authorized_inputs": {
                name: {
                    "pinned_uri": uri,
                    "actual_source": source_names[name],
                    "sha256": sha256_bytes(raw_sources[name]),
                }
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
            "query_list_used_in_component_construction": False,
            "sampling": None,
        },
        "observed_component_decomposition": decomposition,
        "handed_query_audit": handed,
        "exhaustive_unresolved_internal_pair_audit": exhaustive,
        "component_template_analysis": templates,
        "symmetry_beyond_locality": symmetry_analysis,
        "logical_fence": {
            "A_fact": "The observed relation decomposes into connected components.",
            "B_constitution": "A new block is required not to merge observed components.",
            "C_consequence": "Under B, some completion candidates are eliminated.",
            "non_implication": (
                "A does not imply B. Missing blocks can cause observed disconnectedness; "
                "component locality remains an added completion principle."
            ),
            "scope": (
                "The handed-domain equality does not make the constitutions globally "
                "equivalent, and the exhaustive counterexamples disprove equivalence "
                "even across all unresolved internal pairs of the relevant observed "
                "component class."
            ),
        },
        "publication_contract": {
            "expected_manifest_entry_delta": 3,
            "paths": [
                "issues/016-global-relational-completion/016.08-Coder-component-locality-versus-symmetry-GPT.md",
                "issues/016-global-relational-completion/016.08-Code-attachments/component_locality_audit.json",
                "issues/016-global-relational-completion/016.08-Code-attachments/audit_component_locality.py",
            ],
            "workflow": "occurrence",
            "publish_against": "latest",
        },
        "mechanical_checks": checks,
    }


def format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def format_blocks(blocks: Sequence[Sequence[int]]) -> str:
    return "{" + ", ".join(format_set(block) for block in blocks) + "}"


def render_markdown(report: dict[str, Any], json_bytes: bytes) -> bytes:
    decomposition = report["observed_component_decomposition"]
    handed = report["handed_query_audit"]
    exhaustive = report["exhaustive_unresolved_internal_pair_audit"]
    templates = report["component_template_analysis"]
    symmetry = report["symmetry_beyond_locality"]
    lines = [
        "Information exposure:",
        "",
        report["information_exposure"],
        "",
        "# 016.08 — Component locality versus symmetry result",
        "",
        "## Verdict",
        "",
        (
            "On the handed 24-query domain, **E_class holds 24/24** and "
            "**E_component holds 24/24**. In the task's allowed wording: on the "
            "handed query domain, H_KI orbit closure and observed-component locality "
            "induce the same candidate sets. All 24 component-local sets are the same "
            "singletons as the pinned H_WW projection."
        ),
        "",
        (
            "The exhaustive relevant-class audit rejects a broader equivalence: "
            "**C_component = C_HKI for only 24/48 unresolved internal pairs**. The "
            "other 24 pairs are **locality-stronger**: component locality gives the "
            "empty set while H_KI keeps exactly all 36 class-1 Events. Thus the handed "
            "agreement selects one of two unresolved-pair orbits and is not a global "
            "equivalence. By contrast, C_class = C_HIK for all 48/48 pairs."
        ),
        "",
        "## 1. Observed connected components",
        "",
        (
            "The typed incidence graph reconstructed from the blind packet has **14 "
            "components** and **2 canonical isomorphism classes**. Components were "
            "constructed before and independently of the handed query list. The JSON "
            "field `observed_component_decomposition.event_component_records` gives "
            "the required component ID, class ID, full component Event set, and full "
            "component block set separately for each of all 84 Events."
        ),
        "",
        "| Component | Class | Events | Observed blocks | Aut order |",
        "|---:|---:|---|---|---:|",
    ]
    for component in decomposition["components"]:
        lines.append(
            f"| {component['component_id']} | {component['canonical_class_id']} | "
            f"`{format_set(component['events'])}` | "
            f"`{format_blocks(component['blocks'])}` | "
            f"{component['automorphism_order']} |"
        )
    lines.extend(
        [
            "",
            "Canonical classes:",
            "",
        ]
    )
    for class_record in decomposition["component_classes"]:
        lines.append(
            f"- Class {class_record['class_id']}: components "
            f"`{class_record['component_ids']}`; signature "
            f"`{class_record['signature']['sha256']}`; canonical blocks "
            f"`{format_blocks(class_record['signature']['canonical_blocks'])}`; "
            f"component automorphism order {class_record['component_automorphism_order']}."
        )
    lines.extend(
        [
            "",
            "## 2–5. Handed queries, exact baselines, and primary tests",
            "",
            (
                "Every handed pair lies in one class-0 component. The distribution is "
                f"`{handed['query_distribution']['over_components']}` over components "
                f"and `{handed['query_distribution']['over_component_classes']}` over classes."
            ),
            "",
            (
                "The non-group constitutions are defined only when both query "
                "endpoints lie in one observed component C; otherwise both are "
                "`not_applicable`. L_class is C_class(q) = {z in C_local(q): z "
                "belongs to any observed component canonically isomorphic to C}. "
                "L_component is C_component(q) = {z in C_local(q): z belongs to "
                "C}. Neither definition uses automorphism orbit closure."
            ),
            "",
            (
                "The runner independently recomputed the candidate projections "
                "C_local, C_HII, C_HIK, and C_HKI from observed pairs plus the "
                "pinned constitution generator images, then required exact equality "
                "with every corresponding pinned 016.05 set. Status: "
                f"**{handed['pinned_01605_reproduction']['status']}**."
            ),
            "",
        ]
    )
    for record in handed["records"]:
        lines.append(
            f"- `{record['query']}` — component {record['component_id']}, class "
            f"{record['canonical_class_id']}; C_local={format_set(record['C_local'])}; "
            f"C_HII={format_set(record['C_HII'])}; "
            f"C_class={format_set(record['C_class'])}; "
            f"C_HIK={format_set(record['C_HIK'])}; "
            f"C_component={format_set(record['C_component'])}; "
            f"C_HKI={format_set(record['C_HKI'])}."
        )
    lines.extend(
        [
            "",
            "Candidate-size histograms:",
            "",
            "```json",
            json.dumps(handed["candidate_size_histograms"], indent=2, sort_keys=True),
            "```",
            "",
            (
                "E_class: **24/24**, no mismatches. E_component: **24/24**, no "
                "mismatches. L_class is not singleton on all applicable queries; "
                "L_component is singleton on all 24. Against the pinned H_WW "
                "singleton projection, L_class matches 0/24 and L_component matches "
                "24/24."
            ),
            "",
            "## 6. Exhaustive unresolved internal-pair domain",
            "",
            (
                f"The eight relevant class-0 components contain **{exhaustive['total_unresolved_internal_pair_count']}** "
                "unresolved internal pairs: every internal unordered pair not already "
                "used by an observed block. H_KI decomposes them into **16 orbits of "
                "size 3**."
            ),
            "",
            "Candidate-size histograms:",
            "",
            "```json",
            json.dumps(exhaustive["candidate_size_histograms"], indent=2, sort_keys=True),
            "```",
            "",
            (
                "Exact equalities: C_class = C_HIK for **48/48**; C_component = "
                "C_HKI for **24/48**. Under component locality, 24 pairs are singleton, "
                "24 are impossible, and 0 remain ambiguous. The exact pair lists and "
                "all five complete candidate sets for every pair are in the JSON."
            ),
            "",
            "H_KI unresolved-pair orbits:",
            "",
        ]
    )
    for index, pair_orbit in enumerate(exhaustive["H_KI_pair_orbit_decomposition"]):
        lines.append(f"- Orbit {index}: `{pair_orbit}`")
    outcomes = exhaustive["component_locality_outcomes"]
    lines.extend(
        [
            "",
            "Component-local singleton pairs:",
            "",
            f"`{outcomes['singleton_pairs']}`",
            "",
            "Component-local impossible pairs:",
            "",
            f"`{outcomes['impossible_pairs']}`",
            "",
            "Component-local ambiguous pairs: none.",
            "",
            "## 7. Canonical component template",
            "",
        ]
    )
    for template in templates:
        lines.extend(
            [
                f"### Canonical class {template['canonical_class_id']}",
                "",
                f"Blocks: `{format_blocks(template['canonical_blocks'])}`.",
                "",
                (
                    f"Events: {template['event_count']}; observed blocks: "
                    f"{template['observed_block_count']}; used unordered pairs: "
                    f"{template['used_unordered_pair_count']}; unused unordered pairs: "
                    f"{template['unused_unordered_pair_count']}; automorphism-group "
                    f"order: {template['automorphism_group_order']}."
                ),
                "",
            ]
        )
        for orbit_record in template["unresolved_pair_orbits"]:
            lines.append(
                f"- Pair orbit {orbit_record['orbit_id']}: "
                f"`{orbit_record['pairs']}`; component-local size histogram "
                f"`{orbit_record['component_local_candidate_count_histogram']}`; "
                f"canonical completed blocks "
                f"`{orbit_record['canonical_completed_blocks']}`."
            )
    lines.extend(
        [
            "",
            (
                "There is exactly one unresolved-pair orbit whose members admit a "
                "unique same-component completion: `{[2,4], [2,5], [4,5]}` in "
                "canonical labels, completed by the canonical block `{2,4,5}`. The "
                "other orbit `{[0,5], [1,4], [2,3]}` has no component-local candidate."
            ),
            "",
            "## 8. Does symmetry contribute beyond locality?",
            "",
            "| Classification | Count |",
            "|---|---:|",
        ]
    )
    for name, count in symmetry["counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            symmetry["structural_condition"],
            "",
            "Exact template proof:",
            "",
        ]
    )
    for item in symmetry["exact_template_proof"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            (
                "The proof was paired with exhaustive verification over all eight "
                "components of the relevant canonical class; every structured proof "
                "check in the JSON is true. Full local→component and local→H_KI "
                "elimination sets, including every symmetric difference, are recorded "
                "for every mismatch."
            ),
            "",
            "## 9. Logical fence",
            "",
            "A. The observed relation decomposes into connected components. **Fact.**",
            "",
            "B. A new block may not merge observed components. **Added constitution.**",
            "",
            "C. Under B, some candidates are eliminated. **Consequence.**",
            "",
            (
                "A does not imply B. Missing blocks can themselves cause observed "
                "disconnectedness. Component locality therefore remains an added "
                "completion principle even where it reproduces a symmetry result; "
                "the exhaustive counterexamples here also show that it can be "
                "strictly stronger than H_KI."
            ),
            "",
            "## 10. Exactness, reproduction, and publication",
            "",
            (
                "All calculations are deterministic finite enumeration. No sampling, "
                "target labels, total-block premise, saturation premise, degree premise, "
                "external geometry, coordinates, completion table, or neural learner "
                "is used."
            ),
            "",
            "```bash",
            "python issues/016-global-relational-completion/016.08-Code-attachments/audit_component_locality.py",
            "python issues/016-global-relational-completion/016.08-Code-attachments/audit_component_locality.py --verify",
            "```",
            "",
            f"JSON certificate SHA-256: `{sha256_bytes(json_bytes)}`.",
            "",
            (
                "Publication contract: publish against latest with workflow "
                "`occurrence`; exactly three delivered paths; expected manifest "
                "entry-count delta **+3**."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_sources(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, str]]:
    source_names = {
        "task": args.task,
        "input": args.input,
        "parent_result": args.parent_result,
        "parent_json": args.parent_json,
        "parent_runner": args.parent_runner,
    }
    raw_sources = {
        name: read_and_gate(source, EXPECTED_HASHES[name], name.replace("_", " "))
        for name, source in source_names.items()
    }
    return raw_sources, source_names


def execute(args: argparse.Namespace) -> dict[str, Any]:
    raw_sources, source_names = build_sources(args)
    report = build_report(raw_sources, source_names)
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
            "json_sha256": sha256_bytes(json_bytes),
            "result_sha256": sha256_bytes(markdown_bytes),
            "handed_component_equivalence": "24/24",
            "exhaustive_component_equivalence": "24/48",
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(json_bytes)
    result_path.write_bytes(markdown_bytes)
    return {
        "status": "COMPLETE",
        "json": str(json_path),
        "result": str(result_path),
        "json_sha256": sha256_bytes(json_bytes),
        "result_sha256": sha256_bytes(markdown_bytes),
        "handed_component_equivalence": "24/24",
        "exhaustive_component_equivalence": "24/48",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact blind component-locality versus symmetry audit"
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
    except Blocked01605Reproduction as error:
        print(str(error), file=sys.stderr)
        return 2
    except LocalityAuditError as error:
        print(f"component-locality audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["status"] in {"COMPLETE", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
