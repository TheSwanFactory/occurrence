#!/usr/bin/env python3
"""Exact blind audit of canonical symmetry factors of the observed relation.

Information exposure: before the immutable 016.04 task could be retrieved through
an authenticated Quilt client, local project metadata and Issue-014/015 transport
references were inspected only to establish retrieval and publication conventions.
They exposed no Issue-016 withheld target or preferred structure. After retrieval,
scientific reads were restricted to the immutable 016.04 task, the pinned 016.01
blind packet, and the pinned 016.02 result, JSON certificate, and implementation.
The mutable Issue-016 README and every 016.03 path were not read.

The finite computation reuses the hash-gated 016.02 implementation for its input,
component-isomorphism, permutation, orbit, and linearity primitives. This runner
canonically orders component classes by an isomorphism-invariant signature, builds
all nine I/K/W direct-product subgroups from proved-complete wreath factors, and
recomputes every candidate orbit without stochastic sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

TASK_REVISION = "27eb95b6cb7383d8f5d8407ae28be6ac14ac07d9864c42ee222b903bd8c923bd"
TASK_PATH = (
    "issues/016-global-relational-completion/"
    "016.04-GPT-blind-canonical-symmetry-factor-ablation-Coder.md"
)
INPUT_REVISION = "1339739309f484ad738449f8733b2df12869f970030cdc9878e60de0753fa772"
INPUT_PATH = (
    "issues/016-global-relational-completion/"
    "016.01-Code-attachments/observed-symmetry-input.json"
)
PARENT_REVISION = "7569b774b9573bf02be0042a6225d1240e0aa62616c7adf421763cdb6c13edef"
PARENT_RESULT_PATH = (
    "issues/016-global-relational-completion/"
    "016.02-Coder-observed-symmetry-completion-audit-GPT.md"
)
PARENT_JSON_PATH = (
    "issues/016-global-relational-completion/"
    "016.02-Code-attachments/observed_symmetry_audit.json"
)
PACKAGE_PREFIX = "quilt+s3://protology#package=occurrence/gpt@"
TASK_URI = f"{PACKAGE_PREFIX}{TASK_REVISION}&path={TASK_PATH}"
INPUT_URI = f"{PACKAGE_PREFIX}{INPUT_REVISION}&path={INPUT_PATH}"
PARENT_RESULT_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_RESULT_PATH}"
PARENT_JSON_URI = f"{PACKAGE_PREFIX}{PARENT_REVISION}&path={PARENT_JSON_PATH}"

EXPECTED_HASHES = {
    "task": "9522cb89ee15507d2d53e646f33d9dfe83a1afc82cefcf9b7ff47bf7963354a9",
    "input": "992616907f4d16018b579076374db5eea03f188b754cdee3b205fd9cc084de59",
    "parent_result": "3d495ce5fd72fe30dc113e28467d7b995a6436f8d11384e763a0216c8d4c6b04",
    "parent_json": "ba58523d2310305ea823d15c47cb72510c6be42c89560b72e277c07bcfd0ad4c",
    "parent_runner": "1e8fe389550c142ecd03b75dafb176e0ef611f836a915a8a3e0c46cf41b03333",
}
PARENT_SCHEMA = "occurrence.gpt.01602.observed-symmetry-audit.v1"
REPORT_SCHEMA = "occurrence.gpt.01605.canonical-symmetry-factor-ablation.v1"
LEVELS = ("I", "K", "W")
LEVEL_RANK = {level: rank for rank, level in enumerate(LEVELS)}
EXPECTED_GROUP_ORDER = 6**8 * math.factorial(8) * 24**6 * math.factorial(6)
EXPECTED_COMPONENT_PROFILE = {(8, 6), (6, 24)}

ATTACHMENT_DIR = Path(__file__).resolve().parent
ISSUE_DIR = ATTACHMENT_DIR.parent
PARENT_RUNNER = ISSUE_DIR / "016.02-Code-attachments/audit_observed_symmetry.py"
DEFAULT_JSON = ATTACHMENT_DIR / "symmetry_factor_ablation.json"
DEFAULT_RESULT = ISSUE_DIR / "016.05-Coder-canonical-symmetry-factor-ablation-GPT.md"

INFORMATION_EXPOSURE = (
    "Before the immutable 016.04 task could be retrieved through an authenticated "
    "Quilt client, local project metadata and Issue-014/015 transport references "
    "were inspected only for retrieval and publication conventions; they exposed "
    "no Issue-016 withheld target or preferred structure. Thereafter the only "
    "scientific inputs read were the immutable 016.04 task, the pinned blind "
    "016.01 packet, and the pinned 016.02 result, JSON certificate, and "
    "implementation. The mutable Issue-016 README, all 016.03 content, certified "
    "NOVEL targets, the certified full relation, preferred Fano/SFP coordinates, "
    "and Owner conclusions were not read or otherwise exposed."
)

Event = int
Block = tuple[int, int, int]
Pair = tuple[int, int]
Permutation = tuple[int, ...]


class FactorAuditError(RuntimeError):
    """An exact factor-audit invariant failed."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_parent() -> ModuleType:
    if not PARENT_RUNNER.is_file():
        raise FactorAuditError(f"authorized parent runner missing: {PARENT_RUNNER}")
    actual_hash = sha256_file(PARENT_RUNNER)
    if actual_hash != EXPECTED_HASHES["parent_runner"]:
        raise FactorAuditError(
            "authorized parent runner hash mismatch: "
            f"expected {EXPECTED_HASHES['parent_runner']}, got {actual_hash}"
        )
    spec = importlib.util.spec_from_file_location(
        "occurrence_01602_observed_symmetry", PARENT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise FactorAuditError("could not load the authorized parent runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_and_gate(parent: ModuleType, source: str, expected_hash: str, label: str) -> bytes:
    try:
        raw = parent.read_input(source)
    except Exception as error:
        raise FactorAuditError(f"could not read {label} from {source}: {error}") from error
    actual_hash = sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise FactorAuditError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return raw


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FactorAuditError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise FactorAuditError(f"{label} must be a JSON object")
    return value


def component_signature(
    component: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return a complete deterministic isomorphism invariant for a small component."""
    events = tuple(component["events"])
    index = {event: position for position, event in enumerate(events)}
    local_blocks = tuple(
        tuple(sorted(index[event] for event in block)) for block in component["blocks"]
    )
    encodings = []
    for images in itertools.permutations(range(len(events))):
        encodings.append(
            tuple(
                sorted(
                    tuple(sorted(images[position] for position in block))
                    for block in local_blocks
                )
            )
        )
    canonical_blocks = min(encodings)
    key = (len(events), len(local_blocks), canonical_blocks)
    signature_value = {
        "event_vertex_count": len(events),
        "block_vertex_count": len(local_blocks),
        "incidence_edge_count": 3 * len(local_blocks),
        "canonical_blocks": [list(block) for block in canonical_blocks],
    }
    signature_bytes = json.dumps(
        signature_value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    signature_value["sha256"] = sha256_bytes(signature_bytes)
    signature_value["serialization"] = signature_bytes.decode("utf-8")
    return key, signature_value


def generator_record(
    name: str,
    kind: str,
    class_id: int,
    permutation: Permutation,
    parent: ModuleType,
    component_id: int | None = None,
    components: Sequence[int] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "class_id": class_id,
        "images": list(permutation),
        "cycles": parent.permutation_cycles(permutation),
    }
    if component_id is not None:
        record["component_id"] = component_id
    if components is not None:
        record["components"] = list(components)
    return record


def build_factor_system(
    parent: ModuleType,
    universe: Sequence[Event],
    blocks: Sequence[Block],
    parent_report: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Permutation],
    dict[int, dict[str, list[str]]],
    list[dict[str, Any]],
]:
    components = parent.event_components(universe, blocks)
    signature_groups: dict[tuple[Any, ...], list[int]] = {}
    signature_values: dict[tuple[Any, ...], dict[str, Any]] = {}
    for component_id, component in enumerate(components):
        key, value = component_signature(component)
        signature_groups.setdefault(key, []).append(component_id)
        signature_values[key] = value
    ordered_keys = sorted(signature_groups)
    if len(components) != 14 or len(ordered_keys) != 2:
        raise FactorAuditError("component decomposition is not the required 14 / 2")

    generator_permutations: dict[str, Permutation] = {}
    generator_records: list[dict[str, Any]] = []
    level_generators: dict[int, dict[str, list[str]]] = {}
    class_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    block_set = set(blocks)

    for class_id, key in enumerate(ordered_keys):
        component_ids = sorted(signature_groups[key])
        representative = components[component_ids[0]]
        internal_names: list[str] = []
        swap_names: list[str] = []
        automorphism_orders: set[int] = set()

        for component_id in component_ids:
            component = components[component_id]
            raw_automorphisms = parent.component_isomorphisms(component, component)
            localized = [
                parent.localize_images(
                    component["events"], images, component["events"]
                )
                for images in raw_automorphisms
            ]
            local_generators = parent.greedy_generators(localized)
            closure = parent.generated_closure(
                local_generators, len(component["events"])
            )
            if closure != set(localized):
                raise FactorAuditError("local generator closure is not exact")
            automorphism_orders.add(len(localized))
            local_record = {
                "component_id": component_id,
                "events": list(component["events"]),
                "block_vertex_ids": list(component["block_indices"]),
                "blocks": [list(block) for block in component["blocks"]],
                "canonical_class_id": class_id,
                "automorphism_order": len(localized),
                "all_event_bijections_tested": math.factorial(
                    len(component["events"])
                ),
                "enumerated_automorphisms": [list(item) for item in localized],
                "local_generator_count": len(local_generators),
            }
            component_records.append(local_record)
            for generator_id, local_generator in enumerate(local_generators):
                name = (
                    f"class_{class_id}_component_{component_id}_internal_"
                    f"{generator_id}"
                )
                permutation = parent.globalize_local(
                    component["events"], local_generator, len(universe)
                )
                generator_permutations[name] = permutation
                generator_records.append(
                    generator_record(
                        name,
                        "within_component_automorphism",
                        class_id,
                        permutation,
                        parent,
                        component_id=component_id,
                    )
                )
                internal_names.append(name)

        if len(automorphism_orders) != 1:
            raise FactorAuditError("one signature class has unequal local group orders")
        local_order = next(iter(automorphism_orders))
        for swap_id, (left_id, right_id) in enumerate(
            itertools.pairwise(component_ids)
        ):
            name = f"class_{class_id}_component_swap_{swap_id}"
            permutation = parent.component_swap(
                components[left_id], components[right_id], len(universe)
            )
            generator_permutations[name] = permutation
            generator_records.append(
                generator_record(
                    name,
                    "component_permutation",
                    class_id,
                    permutation,
                    parent,
                    components=(left_id, right_id),
                )
            )
            swap_names.append(name)

        multiplicity = len(component_ids)
        kernel_order = local_order**multiplicity
        wreath_order = kernel_order * math.factorial(multiplicity)
        level_generators[class_id] = {
            "I": [],
            "K": internal_names,
            "W": internal_names + swap_names,
        }
        representative_isomorphisms = {
            str(component_id): len(
                parent.component_isomorphisms(
                    representative, components[component_id]
                )
            )
            for component_id in component_ids
        }
        class_records.append(
            {
                "class_id": class_id,
                "signature": signature_values[key],
                "component_ids": component_ids,
                "multiplicity": multiplicity,
                "component_automorphism_order": local_order,
                "representative_component_id": component_ids[0],
                "representative_to_member_isomorphism_counts": (
                    representative_isomorphisms
                ),
                "all_members_isomorphic": all(
                    count > 0 for count in representative_isomorphisms.values()
                ),
                "levels": {
                    "I": {
                        "order": 1,
                        "formula": "1",
                        "generator_names": [],
                        "action": "fixes every Event in this class pointwise",
                    },
                    "K": {
                        "order": kernel_order,
                        "formula": f"{local_order}^{multiplicity}",
                        "generator_names": internal_names,
                        "action": (
                            "independent complete within-component automorphisms; "
                            "each connected component is fixed setwise"
                        ),
                    },
                    "W": {
                        "order": wreath_order,
                        "formula": (
                            f"{local_order}^{multiplicity} * {multiplicity}!"
                        ),
                        "generator_names": internal_names + swap_names,
                        "action": (
                            "complete class wreath action: independent internal "
                            "automorphisms and every component permutation"
                        ),
                    },
                },
                "strict_level_inclusions": {
                    "I<K": 1 < kernel_order,
                    "K<W": kernel_order < wreath_order,
                    "proof": (
                        "K contains a listed nonidentity internal automorphism; W "
                        "contains a listed adjacent component swap not stabilizing "
                        "each component setwise. Their exact orders are strictly "
                        "increasing."
                    ),
                },
            }
        )

    profile = {
        (record["multiplicity"], record["component_automorphism_order"])
        for record in class_records
    }
    if profile != EXPECTED_COMPONENT_PROFILE:
        raise FactorAuditError(f"unexpected canonical component profile: {profile}")

    generator_checks = []
    class_event_sets = {
        record["class_id"]: {
            event
            for component_id in record["component_ids"]
            for event in components[component_id]["events"]
        }
        for record in class_records
    }
    for record in generator_records:
        permutation = generator_permutations[record["name"]]
        class_events = class_event_sets[record["class_id"]]
        check = {
            "name": record["name"],
            "bijective": sorted(permutation) == list(universe),
            "preserves_B0": {
                parent.maps_block(permutation, block) for block in blocks
            }
            == block_set,
            "fixes_other_class_pointwise": all(
                permutation[event] == event
                for event in universe
                if event not in class_events
            ),
            "maps_own_class_to_itself": {
                permutation[event] for event in class_events
            }
            == class_events,
        }
        if not all(value for key_name, value in check.items() if key_name != "name"):
            raise FactorAuditError(f"generator verification failed: {record['name']}")
        generator_checks.append(check)

    group_order = math.prod(record["levels"]["W"]["order"] for record in class_records)
    parent_order = parent_report.get("observed_automorphism_group", {}).get(
        "group_order"
    )
    if group_order != EXPECTED_GROUP_ORDER or group_order != parent_order:
        raise FactorAuditError("canonical wreath decomposition disagrees with parent G")

    certificate = {
        "typed_incidence_graph": {
            "event_vertex_count": len(universe),
            "block_vertex_count": len(blocks),
            "incidence_edge_count": 3 * len(blocks),
            "connected_component_count": len(components),
            "component_isomorphism_class_count": len(class_records),
            "construction": (
                "Event and observed-block vertices are typed; incidence joins each "
                "block vertex to its three Event vertices. Every component signature "
                "is the lexicographically least block encoding over every Event "
                "bijection of that component, hence isomorphism invariant and "
                "complete for these exhaustively enumerated six-Event components."
            ),
        },
        "canonical_class_order": (
            "increasing lexicographic order of (Event count, block count, canonical "
            "block encoding)"
        ),
        "component_classes": class_records,
        "components": sorted(component_records, key=lambda item: item["component_id"]),
        "full_group_order": group_order,
        "full_group_formula": " * ".join(
            record["levels"]["W"]["formula"] for record in class_records
        ),
        "generator_catalog": generator_records,
        "generator_checks": generator_checks,
        "exactness_proof": [
            "Every typed-incidence automorphism permutes connected components only within an equal canonical signature class.",
            "Every Event bijection of every six-Event component was tested, so each displayed local automorphism group is complete.",
            "The deterministic internal generators close to each complete enumerated local group.",
            "Independent component supports give the direct-product kernel K_i with the displayed power order.",
            "Deterministic adjacent component swaps generate the complete symmetric component action, giving W_i = Aut(component) wreath S_m.",
            "The two class supports are disjoint, so every H_ab is the exact direct product of its selected class levels, not an approximated generator subgroup.",
        ],
    }
    return certificate, generator_permutations, level_generators, components


def subgroup_name(level_0: str, level_1: str) -> str:
    return f"H_{level_0}{level_1}"


def subgroup_leq(left: str, right: str) -> bool:
    return all(
        LEVEL_RANK[left[index]] <= LEVEL_RANK[right[index]]
        for index in (2, 3)
    )


def subgroup_levels(name: str) -> tuple[str, str]:
    if len(name) != 4 or not name.startswith("H_"):
        raise FactorAuditError(f"invalid subgroup name {name}")
    levels = (name[2], name[3])
    if any(level not in LEVELS for level in levels):
        raise FactorAuditError(f"invalid subgroup levels in {name}")
    return levels


def classify_candidate_size(size: int, local_size: int) -> str:
    if size == local_size:
        return "unchanged"
    if size == 0:
        return "symmetry-incompatible"
    if size == 1:
        return "singleton"
    if 1 < size < local_size:
        return "narrowed-but-ambiguous"
    raise FactorAuditError(f"nonexhaustive candidate classification {size}/{local_size}")


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def analyze_orbit(
    parent: ModuleType,
    seed: Block,
    generators: Sequence[Permutation],
    observed_owner: dict[Pair, Block],
) -> dict[str, Any]:
    orbit_blocks = parent.orbit(seed, generators, parent.action_block)
    observed_hits: list[tuple[Pair, Block, Block]] = []
    pair_owners: dict[Pair, list[Block]] = {}
    for block in orbit_blocks:
        for pair in parent.pairs(block):
            if pair in observed_owner:
                observed_hits.append((pair, block, observed_owner[pair]))
            pair_owners.setdefault(pair, []).append(block)
    internal_hits = [
        (pair, tuple(sorted(owners)))
        for pair, owners in sorted(pair_owners.items())
        if len(owners) > 1
    ]
    collision_certificates = []
    if observed_hits:
        pair, orbit_block, observed_block = min(observed_hits)
        collision_certificates.append(
            {
                "kind": "collision_with_B0",
                "pair": list(pair),
                "orbit_block": list(orbit_block),
                "observed_block": list(observed_block),
            }
        )
    if internal_hits:
        pair, owners = internal_hits[0]
        collision_certificates.append(
            {
                "kind": "internal_orbit_collision",
                "pair": list(pair),
                "orbit_blocks": [list(block) for block in owners[:2]],
            }
        )
    linear = not observed_hits and not internal_hits
    if linear:
        category = "none"
    elif observed_hits and internal_hits:
        category = "observed_and_internal"
    elif observed_hits:
        category = "collision_with_B0"
    else:
        category = "internal_orbit_collision"
    encoded_orbit = [list(block) for block in orbit_blocks]
    return {
        "_blocks": orbit_blocks,
        "orbit_size": len(orbit_blocks),
        "orbit_sha256": sha256_bytes(canonical_json(encoded_orbit)),
        "linear_with_B0": linear,
        "collision_category": category,
        "pair_collision_certificates": collision_certificates,
    }


class OrbitCache:
    def __init__(
        self,
        parent: ModuleType,
        generators: Sequence[Permutation],
        observed_owner: dict[Pair, Block],
    ) -> None:
        self.parent = parent
        self.generators = generators
        self.observed_owner = observed_owner
        self.by_member: dict[Block, dict[str, Any]] = {}

    def get(self, triple: Block) -> dict[str, Any]:
        if triple not in self.by_member:
            analysis = analyze_orbit(
                self.parent, triple, self.generators, self.observed_owner
            )
            for member in analysis["_blocks"]:
                self.by_member[member] = analysis
        return self.by_member[triple]


def public_orbit_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in analysis.items() if not key.startswith("_")}


def extension_certificate(
    constitution: str, candidate: int, analysis: dict[str, Any]
) -> dict[str, Any]:
    return {
        "constitution": constitution,
        "candidate": candidate,
        "orbit_size": analysis["orbit_size"],
        "orbit_sha256": analysis["orbit_sha256"],
        "extension_blocks": [list(block) for block in analysis["_blocks"]],
        "extension_is_orbit_closed": True,
        "extension_is_linear_with_B0": analysis["linear_with_B0"],
    }


def build_subgroups(
    parent: ModuleType,
    universe: Sequence[Event],
    blocks: Sequence[Block],
    queries: Sequence[Pair],
    factor_certificate: dict[str, Any],
    generator_permutations: dict[str, Permutation],
    level_generators: dict[int, dict[str, list[str]]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, Pair, int], dict[str, Any]],
    dict[str, dict[Pair, tuple[int, ...]]],
]:
    observed_owner = {
        pair: block for block in blocks for pair in parent.pairs(block)
    }
    observed_pairs = set(observed_owner)
    class_records = factor_certificate["component_classes"]
    local_sets = {
        query: parent.local_candidates(query, universe, observed_pairs)
        for query in queries
    }
    if any(len(candidates) != 79 for candidates in local_sets.values()):
        raise FactorAuditError("a handed query did not reproduce 79 local candidates")

    subgroup_records: list[dict[str, Any]] = []
    analysis_lookup: dict[tuple[str, Pair, int], dict[str, Any]] = {}
    candidate_maps: dict[str, dict[Pair, tuple[int, ...]]] = {}
    for level_0 in LEVELS:
        for level_1 in LEVELS:
            name = subgroup_name(level_0, level_1)
            selected_names = (
                level_generators[0][level_0] + level_generators[1][level_1]
            )
            generators = [generator_permutations[item] for item in selected_names]
            order = (
                class_records[0]["levels"][level_0]["order"]
                * class_records[1]["levels"][level_1]["order"]
            )
            if EXPECTED_GROUP_ORDER % order:
                raise FactorAuditError(f"nonintegral index for {name}")
            cache = OrbitCache(parent, generators, observed_owner)
            query_records = []
            candidate_map: dict[Pair, tuple[int, ...]] = {}
            for query in queries:
                local = local_sets[query]
                surviving = []
                eliminations = []
                for candidate in local:
                    triple = tuple(sorted((*query, candidate)))
                    analysis = cache.get(triple)
                    analysis_lookup[(name, query, candidate)] = analysis
                    if analysis["linear_with_B0"]:
                        surviving.append(candidate)
                    else:
                        if not analysis["pair_collision_certificates"]:
                            raise FactorAuditError(
                                f"elimination without collision certificate: "
                                f"{name}, {query}, {candidate}"
                            )
                        eliminations.append(
                            {
                                "candidate": candidate,
                                "triple": list(triple),
                                **public_orbit_analysis(analysis),
                            }
                        )
                candidate_set = tuple(surviving)
                candidate_map[query] = candidate_set
                ambiguity_witnesses = []
                if len(candidate_set) > 1:
                    for candidate in candidate_set[:2]:
                        ambiguity_witnesses.append(
                            extension_certificate(
                                name,
                                candidate,
                                analysis_lookup[(name, query, candidate)],
                            )
                        )
                if len(candidate_set) > 1 and len(ambiguity_witnesses) != 2:
                    raise FactorAuditError("ambiguous query lacks two extensions")
                singleton_certificate = None
                if len(candidate_set) == 1:
                    candidate = candidate_set[0]
                    singleton_certificate = extension_certificate(
                        name,
                        candidate,
                        analysis_lookup[(name, query, candidate)],
                    )
                query_records.append(
                    {
                        "query": list(query),
                        "C_local": list(local),
                        "C_local_size": len(local),
                        "C_H": list(candidate_set),
                        "C_H_size": len(candidate_set),
                        "classification": classify_candidate_size(
                            len(candidate_set), len(local)
                        ),
                        "eliminations": eliminations,
                        "ambiguity_witnesses": ambiguity_witnesses,
                        "singleton_extension_certificate": singleton_certificate,
                    }
                )

            query_orbits = parent.partition_orbits(
                queries, generators, parent.action_pair
            )
            if {query for orbit_value in query_orbits for query in orbit_value} != set(
                queries
            ):
                raise FactorAuditError(f"query orbit decomposition escaped at {name}")
            sizes = [len(candidate_map[query]) for query in queries]
            classifications = Counter(
                classify_candidate_size(size, 79) for size in sizes
            )
            subgroup_records.append(
                {
                    "name": name,
                    "class_levels": {"class_0": level_0, "class_1": level_1},
                    "order": order,
                    "index_in_G": EXPECTED_GROUP_ORDER // order,
                    "generator_count": len(selected_names),
                    "generator_names": selected_names,
                    "exact_group_certificate": (
                        "The selected generators are the union of the proved-complete "
                        "canonical level generators on two disjoint supports. Its "
                        "order is the product of the two exact level orders."
                    ),
                    "candidate_size_histogram": histogram(sizes),
                    "query_counts": {
                        "candidate_size_79": sizes.count(79),
                        "narrowed_but_ambiguous": classifications[
                            "narrowed-but-ambiguous"
                        ],
                        "singleton": classifications["singleton"],
                        "symmetry_incompatible": classifications[
                            "symmetry-incompatible"
                        ],
                    },
                    "query_orbit_decomposition": [
                        [list(query) for query in orbit_value]
                        for orbit_value in query_orbits
                    ],
                    "query_orbit_sizes": [len(item) for item in query_orbits],
                    "queries": query_records,
                }
            )
            candidate_maps[name] = candidate_map

    if len(subgroup_records) != 9 or len(candidate_maps) != 9:
        raise FactorAuditError("did not construct exactly nine constitutions")
    return subgroup_records, analysis_lookup, candidate_maps


def baseline_projection(candidate_map: dict[Pair, Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {"query": list(query), "candidate_set": list(candidate_map[query])}
        for query in sorted(candidate_map)
    ]


def apply_baseline_gate(
    parent_report: dict[str, Any],
    subgroup_records: list[dict[str, Any]],
    candidate_maps: dict[str, dict[Pair, tuple[int, ...]]],
) -> dict[str, Any]:
    parent_map = {
        tuple(record["query"]): tuple(record["C_G"])
        for record in parent_report.get("queries", [])
    }
    if set(parent_map) != set(candidate_maps["H_WW"]):
        raise FactorAuditError("BLOCKED-FULL-G-REPRODUCTION: query domains differ")
    parent_bytes = canonical_json(baseline_projection(parent_map))
    reproduced_bytes = canonical_json(
        baseline_projection(candidate_maps["H_WW"])
    )
    byte_identical = reproduced_bytes == parent_bytes
    all_singleton = all(len(value) == 1 for value in parent_map.values())
    if not byte_identical or not all_singleton:
        raise FactorAuditError(
            "BLOCKED-FULL-G-REPRODUCTION: canonical candidate projection differs"
        )
    baseline = candidate_maps["H_WW"]
    for subgroup in subgroup_records:
        candidate_map = candidate_maps[subgroup["name"]]
        equal_flags = {
            query: candidate_map[query] == baseline[query] for query in sorted(baseline)
        }
        subgroup["all_24_candidate_sets_equal_H_WW"] = all(equal_flags.values())
        subgroup["queries_equal_H_WW_count"] = sum(equal_flags.values())
    return {
        "status": "PASS",
        "projection": (
            "Canonical JSON list sorted by query, with records containing exactly "
            "query and candidate_set"
        ),
        "parent_projection_sha256": sha256_bytes(parent_bytes),
        "reproduced_projection_sha256": sha256_bytes(reproduced_bytes),
        "byte_identical": byte_identical,
        "all_24_parent_sets_singleton": all_singleton,
        "candidate_sets": baseline_projection(candidate_maps["H_WW"]),
    }


def removed_factors(
    name: str, class_records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    levels = subgroup_levels(name)
    result = []
    for class_id, level in enumerate(levels):
        record = class_records[class_id]
        multiplicity = record["multiplicity"]
        local_order = record["component_automorphism_order"]
        if level == "W":
            continue
        if level == "K":
            result.append(
                {
                    "class_id": class_id,
                    "removed": "component permutations",
                    "removed_factor_order": math.factorial(multiplicity),
                    "removed_factor_formula": f"{multiplicity}!",
                }
            )
        else:
            result.append(
                {
                    "class_id": class_id,
                    "removed": (
                        "all within-component automorphisms and all component "
                        "permutations"
                    ),
                    "removed_factor_order": (
                        local_order**multiplicity * math.factorial(multiplicity)
                    ),
                    "removed_factor_formula": (
                        f"{local_order}^{multiplicity} * {multiplicity}!"
                    ),
                }
            )
    return result


def analyze_sufficiency(
    subgroup_records: Sequence[dict[str, Any]],
    candidate_maps: dict[str, dict[Pair, tuple[int, ...]]],
    class_records: Sequence[dict[str, Any]],
    analysis_lookup: dict[tuple[str, Pair, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record_by_name = {record["name"]: record for record in subgroup_records}
    sufficient = sorted(
        name
        for name, record in record_by_name.items()
        if record["all_24_candidate_sets_equal_H_WW"]
    )
    minimal = []
    for name in sufficient:
        strict_sufficient_predecessors = [
            other
            for other in sufficient
            if other != name and subgroup_leq(other, name)
        ]
        if not strict_sufficient_predecessors:
            minimal.append(name)

    minimal_records = []
    for name in minimal:
        strict_predecessors = sorted(
            other
            for other in record_by_name
            if other != name and subgroup_leq(other, name)
        )
        predecessor_proof = [
            {
                "name": other,
                "24_query_sufficient": other in sufficient,
                "queries_equal_H_WW_count": record_by_name[other][
                    "queries_equal_H_WW_count"
                ],
            }
            for other in strict_predecessors
        ]
        if any(item["24_query_sufficient"] for item in predecessor_proof):
            raise FactorAuditError(f"claimed nonminimal sufficient subgroup {name}")
        singleton_certificates = []
        for query in sorted(candidate_maps[name]):
            values = candidate_maps[name][query]
            if len(values) != 1:
                raise FactorAuditError(f"sufficient subgroup {name} is not singleton")
            candidate = values[0]
            singleton_certificates.append(
                {
                    "query": list(query),
                    **extension_certificate(
                        name, candidate, analysis_lookup[(name, query, candidate)]
                    ),
                }
            )
        minimal_records.append(
            {
                "name": name,
                "order": record_by_name[name]["order"],
                "index_in_G": record_by_name[name]["index_in_G"],
                "removed_symmetry_factors_relative_to_G": removed_factors(
                    name, class_records
                ),
                "strict_predecessor_proof": predecessor_proof,
                "singleton_certificates": singleton_certificates,
            }
        )

    queries = sorted(candidate_maps["H_WW"])
    query_records = []
    minima_shapes = set()
    for query in queries:
        baseline = candidate_maps["H_WW"][query]
        query_sufficient = sorted(
            name
            for name, candidate_map in candidate_maps.items()
            if candidate_map[query] == baseline
        )
        query_minimal = [
            name
            for name in query_sufficient
            if not any(
                other != name
                and subgroup_leq(other, name)
                and other in query_sufficient
                for other in query_sufficient
            )
        ]
        query_minimal = sorted(query_minimal)
        minima_shapes.add(tuple(query_minimal))
        query_records.append(
            {
                "query": list(query),
                "H_WW_singleton": list(baseline),
                "sufficient_constitutions": query_sufficient,
                "inclusion_minimal_constitutions": query_minimal,
                "minimality_proofs": [
                    {
                        "name": name,
                        "strict_predecessors": [
                            {
                                "name": other,
                                "query_sufficient": other in query_sufficient,
                            }
                            for other in sorted(candidate_maps)
                            if other != name and subgroup_leq(other, name)
                        ],
                    }
                    for name in query_minimal
                ],
            }
        )
    return (
        {
            "definition": (
                "A tested constitution is 24-query sufficient exactly when all 24 "
                "candidate sets are singleton and equal the corresponding H_WW set."
            ),
            "sufficient_constitutions": sufficient,
            "inclusion_minimal_sufficient_constitutions": minimal,
            "minimal_records": minimal_records,
            "scope": (
                "Minimality is only within the tested canonical 3x3 factor lattice; "
                "it is not a claim about all subgroups of G."
            ),
        },
        {
            "queries": query_records,
            "different_queries_require_different_factor_levels": (
                len(minima_shapes) > 1
            ),
            "distinct_minimal_constitution_patterns": [
                list(item) for item in sorted(minima_shapes)
            ],
        },
    )


def cover_transitions() -> list[dict[str, str]]:
    transitions = []
    for other_level in LEVELS:
        transitions.extend(
            [
                {
                    "source": subgroup_name("I", other_level),
                    "target": subgroup_name("K", other_level),
                    "mechanism": "within_component_automorphisms_class_0",
                    "new_ingredient": "I_0 -> K_0",
                },
                {
                    "source": subgroup_name("K", other_level),
                    "target": subgroup_name("W", other_level),
                    "mechanism": "component_permutations_class_0",
                    "new_ingredient": "K_0 -> W_0",
                },
            ]
        )
    for other_level in LEVELS:
        transitions.extend(
            [
                {
                    "source": subgroup_name(other_level, "I"),
                    "target": subgroup_name(other_level, "K"),
                    "mechanism": "within_component_automorphisms_class_1",
                    "new_ingredient": "I_1 -> K_1",
                },
                {
                    "source": subgroup_name(other_level, "K"),
                    "target": subgroup_name(other_level, "W"),
                    "mechanism": "component_permutations_class_1",
                    "new_ingredient": "K_1 -> W_1",
                },
            ]
        )
    return transitions


def attribute_eliminations(
    candidate_maps: dict[str, dict[Pair, tuple[int, ...]]],
    analysis_lookup: dict[tuple[str, Pair, int], dict[str, Any]],
) -> dict[str, Any]:
    transition_records = []
    mechanism_records: dict[str, dict[str, Any]] = {}
    all_monotone = True
    every_elimination_certified = True
    for transition in cover_transitions():
        source = transition["source"]
        target = transition["target"]
        eliminations = []
        strict_queries = 0
        for query in sorted(candidate_maps[source]):
            source_set = set(candidate_maps[source][query])
            target_set = set(candidate_maps[target][query])
            monotone = target_set <= source_set
            all_monotone = all_monotone and monotone
            if not monotone:
                raise FactorAuditError(f"candidate monotonicity failed {source}->{target}")
            removed = sorted(source_set - target_set)
            if removed:
                strict_queries += 1
            for candidate in removed:
                analysis = analysis_lookup[(target, query, candidate)]
                if analysis["linear_with_B0"]:
                    raise FactorAuditError("transition elimination has a linear target orbit")
                certificates = analysis["pair_collision_certificates"]
                if not certificates:
                    every_elimination_certified = False
                    raise FactorAuditError("transition elimination lacks a certificate")
                eliminations.append(
                    {
                        "query": list(query),
                        "candidate": candidate,
                        "target_orbit_size": analysis["orbit_size"],
                        "target_orbit_sha256": analysis["orbit_sha256"],
                        "target_collision_category": analysis[
                            "collision_category"
                        ],
                        "pair_collision_certificates": certificates,
                    }
                )
        transition_record = {
            **transition,
            "strictly_shrunk_query_count": strict_queries,
            "elimination_count": len(eliminations),
            "eliminations": eliminations,
        }
        transition_records.append(transition_record)
        aggregate = mechanism_records.setdefault(
            transition["mechanism"],
            {
                "mechanism": transition["mechanism"],
                "transition_count": 0,
                "nonzero_transition_count": 0,
                "aggregate_elimination_count": 0,
                "representative_pair_collision_certificate": None,
            },
        )
        aggregate["transition_count"] += 1
        aggregate["aggregate_elimination_count"] += len(eliminations)
        if eliminations:
            aggregate["nonzero_transition_count"] += 1
            if aggregate["representative_pair_collision_certificate"] is None:
                aggregate["representative_pair_collision_certificate"] = {
                    "transition": f"{source}->{target}",
                    **eliminations[0],
                }
    return {
        "cover_transitions": transition_records,
        "mechanism_aggregates": [
            mechanism_records[name] for name in sorted(mechanism_records)
        ],
        "checks": {
            "all_cover_transitions_candidate_monotone": all_monotone,
            "every_transition_elimination_has_pair_collision_certificate": (
                every_elimination_certified
            ),
            "every_nonzero_mechanism_has_representative_certificate": all(
                record["aggregate_elimination_count"] == 0
                or record["representative_pair_collision_certificate"] is not None
                for record in mechanism_records.values()
            ),
        },
    }


def build_lattice_certificate(
    subgroup_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    names = sorted(record["name"] for record in subgroup_records)
    inclusions = []
    for left in names:
        for right in names:
            inclusions.append(
                {
                    "subgroup": left,
                    "supergroup": right,
                    "included": subgroup_leq(left, right),
                    "proper": left != right and subgroup_leq(left, right),
                }
            )
    return {
        "constitutions": names,
        "coordinate_order": "I < K < W independently in each canonical class",
        "inclusion_relation": inclusions,
        "cover_transitions": [
            {key: value for key, value in item.items() if key in {"source", "target"}}
            for item in cover_transitions()
        ],
        "identity_is_H_II": True,
        "full_G_is_H_WW": True,
    }


def validate_parent_report(
    parent_report: dict[str, Any], input_raw: bytes
) -> None:
    if parent_report.get("schema") != PARENT_SCHEMA:
        raise FactorAuditError("pinned parent report schema mismatch")
    provenance = parent_report.get("provenance", {})
    if provenance.get("input_raw_sha256") != sha256_bytes(input_raw):
        raise FactorAuditError("parent report was not computed from the handed input")
    group = parent_report.get("observed_automorphism_group", {})
    if group.get("group_order") != EXPECTED_GROUP_ORDER:
        raise FactorAuditError("parent full group order mismatch")
    if group.get("component_count") != 14:
        raise FactorAuditError("parent component count mismatch")
    if len(parent_report.get("queries", [])) != 24:
        raise FactorAuditError("parent query count mismatch")


def build_report(
    parent: ModuleType,
    raw_sources: dict[str, bytes],
) -> dict[str, Any]:
    parent_report = parse_json(raw_sources["parent_json"], "parent certificate")
    validate_parent_report(parent_report, raw_sources["input"])
    try:
        _packet, universe, blocks, queries = parent.validate_packet(raw_sources["input"])
    except Exception as error:
        raise FactorAuditError(f"parent input validation failed: {error}") from error

    factor_certificate, generator_permutations, level_generators, _components = (
        build_factor_system(parent, universe, blocks, parent_report)
    )
    subgroup_records, analysis_lookup, candidate_maps = build_subgroups(
        parent,
        universe,
        blocks,
        queries,
        factor_certificate,
        generator_permutations,
        level_generators,
    )
    baseline_gate = apply_baseline_gate(
        parent_report, subgroup_records, candidate_maps
    )
    sufficiency, query_wise = analyze_sufficiency(
        subgroup_records,
        candidate_maps,
        factor_certificate["component_classes"],
        analysis_lookup,
    )
    attribution = attribute_eliminations(candidate_maps, analysis_lookup)
    lattice = build_lattice_certificate(subgroup_records)

    checks = {
        "input_hash_gated": sha256_bytes(raw_sources["input"])
        == EXPECTED_HASHES["input"],
        "task_hash_gated": sha256_bytes(raw_sources["task"])
        == EXPECTED_HASHES["task"],
        "parent_result_hash_gated": sha256_bytes(raw_sources["parent_result"])
        == EXPECTED_HASHES["parent_result"],
        "parent_json_hash_gated": sha256_bytes(raw_sources["parent_json"])
        == EXPECTED_HASHES["parent_json"],
        "parent_runner_hash_gated": sha256_file(PARENT_RUNNER)
        == EXPECTED_HASHES["parent_runner"],
        "component_count_14": factor_certificate["typed_incidence_graph"][
            "connected_component_count"
        ]
        == 14,
        "two_canonical_component_classes": len(
            factor_certificate["component_classes"]
        )
        == 2,
        "component_profile_exact": {
            (
                record["multiplicity"],
                record["component_automorphism_order"],
            )
            for record in factor_certificate["component_classes"]
        }
        == EXPECTED_COMPONENT_PROFILE,
        "full_group_order_exact": factor_certificate["full_group_order"]
        == EXPECTED_GROUP_ORDER,
        "all_generator_checks_pass": all(
            all(value for key, value in record.items() if key != "name")
            for record in factor_certificate["generator_checks"]
        ),
        "all_class_level_inclusions_strict": all(
            record["strict_level_inclusions"]["I<K"]
            and record["strict_level_inclusions"]["K<W"]
            for record in factor_certificate["component_classes"]
        ),
        "exactly_nine_constitutions": len(subgroup_records) == 9,
        "H_II_identity": next(
            record for record in subgroup_records if record["name"] == "H_II"
        )["order"]
        == 1,
        "H_WW_equals_G": next(
            record for record in subgroup_records if record["name"] == "H_WW"
        )["order"]
        == EXPECTED_GROUP_ORDER,
        "all_local_candidate_sets_size_79": all(
            query_record["C_local_size"] == 79
            for subgroup in subgroup_records
            for query_record in subgroup["queries"]
        ),
        "full_G_baseline_byte_identical": baseline_gate["byte_identical"],
        "every_ambiguity_has_two_extensions": all(
            query_record["C_H_size"] <= 1
            or len(query_record["ambiguity_witnesses"]) == 2
            for subgroup in subgroup_records
            for query_record in subgroup["queries"]
        ),
        "every_elimination_has_pair_collision_certificate": all(
            elimination["pair_collision_certificates"]
            for subgroup in subgroup_records
            for query_record in subgroup["queries"]
            for elimination in query_record["eliminations"]
        ),
        "all_transition_checks_pass": all(attribution["checks"].values()),
        "minimal_sufficiency_proofs_complete": all(
            not any(
                predecessor["24_query_sufficient"]
                for predecessor in record["strict_predecessor_proof"]
            )
            for record in sufficiency["minimal_records"]
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise FactorAuditError(f"mechanical checks failed: {failed}")

    return {
        "schema": REPORT_SCHEMA,
        "information_exposure": INFORMATION_EXPOSURE,
        "provenance": {
            "task_uri": TASK_URI,
            "task_sha256": sha256_bytes(raw_sources["task"]),
            "input_uri": INPUT_URI,
            "input_sha256": sha256_bytes(raw_sources["input"]),
            "parent_result_uri": PARENT_RESULT_URI,
            "parent_result_sha256": sha256_bytes(raw_sources["parent_result"]),
            "parent_json_uri": PARENT_JSON_URI,
            "parent_json_sha256": sha256_bytes(raw_sources["parent_json"]),
            "parent_runner_path": str(PARENT_RUNNER.relative_to(ISSUE_DIR.parent)),
            "parent_runner_sha256": sha256_file(PARENT_RUNNER),
            "runner_sha256": sha256_file(Path(__file__)),
            "finite_computation_dependencies": (
                "Python standard library plus hash-gated 016.02 standard-library "
                "finite-combinatorics implementation; Quilt libraries only for "
                "immutable input transport"
            ),
        },
        "input_validation": {
            "universe_size": len(universe),
            "observed_block_count": len(blocks),
            "observed_constituent_pair_count": len(
                {pair for block in blocks for pair in parent.pairs(block)}
            ),
            "handed_query_count": len(queries),
            "queries_excluded_from_group_construction": True,
            "no_stochastic_sampling": True,
        },
        "canonical_factor_system": factor_certificate,
        "factor_lattice": lattice,
        "subgroups": subgroup_records,
        "full_G_baseline_gate": baseline_gate,
        "inclusion_minimal_sufficiency": sufficiency,
        "query_wise_minimal_symmetry": query_wise,
        "elimination_attribution": attribution,
        "scientific_scope": {
            "established": (
                "For these 24 handed queries, each tested subgroup's reported "
                "candidate set is the exact set of locally admissible candidates "
                "whose subgroup orbit can be adjoined to B0 linearly."
            ),
            "fence": (
                "Agreement with H_WW shows only that full-G invariance can be "
                "weakened to that tested intrinsic constitution without changing "
                "these 24 conditional completions. It does not establish necessity, "
                "a unique weakest subgroup, OT privilege, certified target truth, "
                "identification of every unseen query, a unique total/global block "
                "system, Fano/SFP/projective/XOR structure, or a learning result."
            ),
            "minimality_scope": (
                "Any minimality statement is only within the canonical nine-group "
                "I/K/W factor lattice."
            ),
            "generator_minimization_claim": None,
        },
        "mechanical_checks": checks,
    }


def format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def render_markdown(report: dict[str, Any], json_bytes: bytes) -> bytes:
    classes = report["canonical_factor_system"]["component_classes"]
    subgroup_records = report["subgroups"]
    sufficient = report["inclusion_minimal_sufficiency"]
    query_wise = report["query_wise_minimal_symmetry"]
    attribution = report["elimination_attribution"]
    subgroup_by_name = {record["name"]: record for record in subgroup_records}
    query_maps = {
        record["name"]: {
            tuple(query_record["query"]): query_record["C_H"]
            for query_record in record["queries"]
        }
        for record in subgroup_records
    }
    local_map = {
        tuple(record["query"]): record["C_local"]
        for record in subgroup_by_name["H_II"]["queries"]
    }

    lines = [
        "Information exposure:",
        "",
        report["information_exposure"],
        "",
        "# 016.05 — Canonical symmetry-factor ablation result",
        "",
        "## Verdict",
        "",
        (
            "The exact blind 3x3 factor audit completed with the pinned full-G "
            "baseline gate **PASS**. The inclusion-minimal 24-query sufficient "
            "constitution(s) within this lattice are: **"
            + ", ".join(
                sufficient["inclusion_minimal_sufficient_constitutions"]
            )
            + "**."
        ),
        "",
        (
            "This is minimality only within the nine canonically defined I/K/W "
            "constitutions. It is not a minimum over arbitrary subgroups or a claim "
            "that any surviving singleton is a certified target."
        ),
        "",
        "## Canonical observed-component decomposition",
        "",
        (
            "The typed incidence graph has **14 connected components** in **2 "
            "isomorphism classes**. Classes are named only by increasing complete "
            "isomorphism-invariant signature; no semantic names were assigned."
        ),
        "",
        "| Class | Signature SHA-256 | Components | Local | I order | K order | W order |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in classes:
        levels = record["levels"]
        lines.append(
            f"| {record['class_id']} | `{record['signature']['sha256']}` | "
            f"{record['multiplicity']} | {record['component_automorphism_order']} | "
            f"{levels['I']['order']} | {levels['K']['order']} | "
            f"{levels['W']['order']} |"
        )
    lines.extend(
        [
            "",
            (
                "Thus `|G| = "
                + report["canonical_factor_system"]["full_group_formula"]
                + " = "
                + str(report["canonical_factor_system"]["full_group_order"])
                + "`. Exhaustive component bijections, complete local automorphism "
                "enumerations, deterministic generators, all generator images and "
                "cycles, support checks, and the wreath-product completeness proof "
                "are in the JSON certificate. Both strict chains `I_i < K_i < W_i` "
                "are mechanically certified."
            ),
            "",
            "## Full factor-ablation table",
            "",
            "| Group | Order | [G:H] | Gens | Size histogram | 79 | Ambiguous | Singleton | Incompatible | = H_WW all 24 | Query orbit sizes |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for record in subgroup_records:
        counts = record["query_counts"]
        lines.append(
            f"| {record['name']} | {record['order']} | {record['index_in_G']} | "
            f"{record['generator_count']} | `{record['candidate_size_histogram']}` | "
            f"{counts['candidate_size_79']} | {counts['narrowed_but_ambiguous']} | "
            f"{counts['singleton']} | {counts['symmetry_incompatible']} | "
            f"{record['all_24_candidate_sets_equal_H_WW']} | "
            f"`{record['query_orbit_sizes']}` |"
        )

    lines.extend(
        [
            "",
            (
                "`H_II` is the identity and `H_WW = G`. Each row's deterministic "
                "generator names, exact direct-product order certificate, complete query "
                "orbit decomposition, candidate eliminations, and ambiguity/singleton "
                "extension witnesses are recorded in the JSON."
            ),
            "",
            "## Exact candidate sets for every query and constitution",
            "",
            (
                "The following is the complete exact matrix. `C_local` is shown once per "
                "query; every remaining set is computed independently by subgroup orbit "
                "closure."
            ),
            "",
        ]
    )
    for query in sorted(local_map):
        fragments = [f"C_local={format_set(local_map[query])}"]
        for name in sorted(query_maps):
            fragments.append(f"{name}={format_set(query_maps[name][query])}")
        lines.append(f"- `{list(query)}`: " + "; ".join(fragments) + ".")

    lines.extend(
        [
            "",
            "## Inclusion-minimal sufficient constitutions",
            "",
            (
                "24-query sufficient constitutions: `"
                + "`, `".join(sufficient["sufficient_constitutions"])
                + "`. Inclusion-minimal: `"
                + "`, `".join(
                    sufficient["inclusion_minimal_sufficient_constitutions"]
                )
                + "`."
            ),
            "",
        ]
    )
    for record in sufficient["minimal_records"]:
        removed = "; ".join(
            f"class {item['class_id']}: {item['removed']} "
            f"(factor {item['removed_factor_formula']})"
            for item in record["removed_symmetry_factors_relative_to_G"]
        ) or "none"
        predecessors = ", ".join(
            f"{item['name']} ({item['queries_equal_H_WW_count']}/24)"
            for item in record["strict_predecessor_proof"]
        ) or "none"
        lines.extend(
            [
                f"### {record['name']}",
                "",
                f"Order `{record['order']}`; index `[G:H] = {record['index_in_G']}`.",
                "",
                f"Removed relative to G: {removed}.",
                "",
                (
                    "Every strict predecessor fails 24-query sufficiency: "
                    f"{predecessors}. Exact 24 singleton orbit-closed extension "
                    "certificates are in the JSON."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Query-wise minimal symmetry",
            "",
            (
                "Different queries require different factor levels: **"
                + str(query_wise["different_queries_require_different_factor_levels"])
                + "**."
            ),
            "",
            "| Query | H_WW singleton | Inclusion-minimal tested constitutions |",
            "|---|---|---|",
        ]
    )
    for record in query_wise["queries"]:
        lines.append(
            f"| `{record['query']}` | `{record['H_WW_singleton']}` | "
            f"`{record['inclusion_minimal_constitutions']}` |"
        )

    lines.extend(
        [
            "",
            "## Elimination attribution by added ingredient",
            "",
            "| Mechanism | Cover transitions | Nonzero | Aggregate eliminations |",
            "|---|---:|---:|---:|",
        ]
    )
    for record in attribution["mechanism_aggregates"]:
        lines.append(
            f"| `{record['mechanism']}` | {record['transition_count']} | "
            f"{record['nonzero_transition_count']} | "
            f"{record['aggregate_elimination_count']} |"
        )
    lines.extend(
        [
            "",
            "| Transition | Strictly shrunk queries | Eliminations |",
            "|---|---:|---:|",
        ]
    )
    for record in attribution["cover_transitions"]:
        lines.append(
            f"| `{record['source']} -> {record['target']}` | "
            f"{record['strictly_shrunk_query_count']} | "
            f"{record['elimination_count']} |"
        )
    lines.extend(
        [
            "",
            (
                "Every eliminated `(transition, query, candidate)` is listed in the JSON "
                "with the target orbit's exact size/hash and a concrete repeated-pair "
                "collision. Each nonzero mechanism also has a representative exact "
                "certificate. Counts are transition-attributions and may count one "
                "candidate under more than one lattice context; they are not deduplicated "
                "claims of causal necessity."
            ),
            "",
            "## Full-G baseline gate",
            "",
            (
                "The canonical 24-query candidate projection extracted from pinned "
                "016.02 and the independently recomputed `H_WW` projection are "
                "byte-identical. Both have SHA-256 `"
                + report["full_G_baseline_gate"]["parent_projection_sha256"]
                + "`. Status: **PASS**."
            ),
            "",
            "## Scientific fences",
            "",
            report["scientific_scope"]["fence"],
            "",
            (
                "No arbitrary-generator minimization was performed or claimed. Every "
                "minimality statement is restricted to the intrinsic nine-group factor "
                "lattice. No stochastic sampling, hidden target, preferred algebra, or "
                "external scientific source was used."
            ),
            "",
            "## Artifacts and reproduction",
            "",
            f"- Task: `{TASK_URI}`",
            f"- Task SHA-256: `{report['provenance']['task_sha256']}`",
            f"- Input: `{INPUT_URI}`",
            f"- Input SHA-256: `{report['provenance']['input_sha256']}`",
            f"- Parent result SHA-256: `{report['provenance']['parent_result_sha256']}`",
            f"- Parent JSON SHA-256: `{report['provenance']['parent_json_sha256']}`",
            f"- Parent runner SHA-256: `{report['provenance']['parent_runner_sha256']}`",
            f"- `audit_symmetry_factors.py` SHA-256: `{report['provenance']['runner_sha256']}`",
            f"- `symmetry_factor_ablation.json` SHA-256: `{sha256_bytes(json_bytes)}`",
            "- Expected final Quilt manifest delta: `+3`.",
            "",
            "From the repository root, with Quilt credentials:",
            "",
            "```bash",
            "python issues/016-global-relational-completion/\\",
            "016.05-Code-attachments/audit_symmetry_factors.py",
            "```",
            "",
            (
                "Add `--verify` to recompute every finite certificate and compare both "
                "generated files byte-for-byte without rewriting them."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    parent = load_parent()
    raw_sources = {
        "task": read_and_gate(parent, args.task, EXPECTED_HASHES["task"], "task"),
        "input": read_and_gate(parent, args.input, EXPECTED_HASHES["input"], "input"),
        "parent_result": read_and_gate(
            parent,
            args.parent_result,
            EXPECTED_HASHES["parent_result"],
            "parent result",
        ),
        "parent_json": read_and_gate(
            parent,
            args.parent_json,
            EXPECTED_HASHES["parent_json"],
            "parent JSON",
        ),
    }
    report = build_report(parent, raw_sources)
    json_bytes = canonical_json(report)
    markdown_bytes = render_markdown(report, json_bytes)
    json_path = Path(args.json)
    result_path = Path(args.result)

    if args.verify:
        failures = []
        for path, expected in (
            (json_path, json_bytes),
            (result_path, markdown_bytes),
        ):
            if not path.is_file():
                failures.append(f"missing {path}")
            elif path.read_bytes() != expected:
                failures.append(f"byte mismatch {path}")
        return {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "json_sha256": sha256_bytes(json_bytes),
            "result_sha256": sha256_bytes(markdown_bytes),
            "minimal_sufficient": report["inclusion_minimal_sufficiency"][
                "inclusion_minimal_sufficient_constitutions"
            ],
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
        "minimal_sufficient": report["inclusion_minimal_sufficiency"][
            "inclusion_minimal_sufficient_constitutions"
        ],
        "full_G_baseline": report["full_G_baseline_gate"]["status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact blind canonical I/K/W symmetry-factor ablation"
    )
    parser.add_argument("--task", default=TASK_URI)
    parser.add_argument("--input", default=INPUT_URI)
    parser.add_argument("--parent-result", default=PARENT_RESULT_URI)
    parser.add_argument("--parent-json", default=PARENT_JSON_URI)
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outcome = execute(args)
    except FactorAuditError as error:
        print(f"factor audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["status"] in {"COMPLETE", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
