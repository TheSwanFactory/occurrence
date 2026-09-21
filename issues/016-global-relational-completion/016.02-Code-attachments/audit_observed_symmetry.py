#!/usr/bin/env python3
"""Exact blind audit of completion under the observed partial relation's symmetry.

Information exposure: before the immutable 016.01a brief could be retrieved, local
release metadata and Issue-015 workflow artifacts were inspected only to learn
repository and publication conventions. Those artifacts explicitly withhold Issue
016's global relation; no withheld NOVEL target, certified completion, preferred
coordinate system, certified full-relation automorphism group, or Owner expectation
was exposed. The task and its declared observed-symmetry packet are the audit's only
scientific inputs.

The implementation uses only Python's standard library for finite combinatorics and
group certificates. It enumerates every automorphism of each connected component,
proves the full group by the canonical disjoint-union wreath-product decomposition,
and verifies every emitted generator and orbit-closure witness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qs, urlparse

TASK_REVISION = "1339739309f484ad738449f8733b2df12869f970030cdc9878e60de0753fa772"
TASK_PATH = (
    "issues/016-global-relational-completion/"
    "016.01a-GPT-blind-observed-symmetry-completion-audit-Coder.md"
)
INPUT_PATH = (
    "issues/016-global-relational-completion/"
    "016.01-Code-attachments/observed-symmetry-input.json"
)
TASK_URI = f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}&path={TASK_PATH}"
INPUT_URI = f"quilt+s3://protology#package=occurrence/gpt@{TASK_REVISION}&path={INPUT_PATH}"
EXPECTED_SCHEMA = "occurrence.gpt.016.observed-symmetry-completion-input.v1"
REPORT_SCHEMA = "occurrence.gpt.01602.observed-symmetry-audit.v1"
DEFAULT_JSON = Path(__file__).with_name("observed_symmetry_audit.json")
DEFAULT_RESULT = Path(__file__).parent.parent / (
    "016.02-Coder-observed-symmetry-completion-audit-GPT.md"
)
INFORMATION_EXPOSURE = (
    "Before the immutable 016.01a brief could be retrieved, local release metadata "
    "and Issue-015 workflow artifacts were inspected only for repository and "
    "publication conventions. Those artifacts explicitly withhold Issue 016's "
    "global relation. No withheld NOVEL target, certified full relation, preferred "
    "Fano/SFP presentation, certified full-relation automorphism group, or GPT "
    "Owner expectation was exposed. The immutable task and its declared packet "
    "were the only scientific inputs to this audit."
)

Event = int
Block = tuple[int, int, int]
Pair = tuple[int, int]
Permutation = tuple[int, ...]


class AuditError(RuntimeError):
    """A finite audit invariant failed."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def quilt_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("quilt3", "boto3"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def read_quilt(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme != "quilt+s3":
        raise AuditError(f"unsupported URI: {uri}")
    bucket = parsed.netloc
    fragment = parse_qs(parsed.fragment)
    package_spec = fragment.get("package", [""])[0]
    logical_path = fragment.get("path", [""])[0]
    if "@" not in package_spec or not logical_path:
        raise AuditError("Quilt URI must pin package revision and logical path")
    package_name, revision = package_spec.rsplit("@", 1)
    if not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise AuditError("Quilt package revision must be a 64-hex immutable hash")
    try:
        import boto3
        import quilt3
    except ImportError as error:
        raise AuditError("reading a Quilt URI requires quilt3 and boto3") from error
    package = quilt3.Package.browse(
        package_name, registry=f"s3://{bucket}", top_hash=revision
    )
    if package.top_hash != revision:
        raise AuditError("Quilt readback revision mismatch")
    try:
        physical = package[logical_path].physical_key
    except KeyError as error:
        raise AuditError(f"missing package entry {logical_path}") from error
    kwargs: dict[str, str] = {"Bucket": physical.bucket, "Key": physical.path}
    if physical.version_id:
        kwargs["VersionId"] = physical.version_id
    return boto3.client("s3").get_object(**kwargs)["Body"].read()


def read_input(source: str) -> bytes:
    if source.startswith("quilt+s3://"):
        return read_quilt(source)
    path = Path(source)
    if not path.is_file():
        raise AuditError(f"input packet not found: {source}")
    return path.read_bytes()


def pairs(block: Sequence[int]) -> tuple[Pair, Pair, Pair]:
    return tuple(itertools.combinations(sorted(block), 2))  # type: ignore[return-value]


def validate_packet(raw: bytes) -> tuple[dict[str, Any], tuple[Event, ...], tuple[Block, ...], tuple[Pair, ...]]:
    try:
        packet = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid packet JSON: {error}") from error
    if packet.get("schema") != EXPECTED_SCHEMA:
        raise AuditError("input schema mismatch")
    universe_data = packet.get("token_universe", {})
    universe = tuple(universe_data.get("ids", []))
    if universe_data.get("count") != 84 or universe != tuple(range(84)):
        raise AuditError("universe must be exactly opaque IDs 0..83")
    raw_blocks = packet.get("observed_triads", [])
    blocks = tuple(sorted(tuple(sorted(block)) for block in raw_blocks))
    if len(blocks) != 48 or len(set(blocks)) != 48:
        raise AuditError("B0 must contain exactly 48 distinct blocks")
    if any(len(block) != 3 or len(set(block)) != 3 for block in blocks):
        raise AuditError("every observed block must have three distinct Events")
    if any(event not in universe for block in blocks for event in block):
        raise AuditError("observed block contains an Event outside U")
    observed_pairs = tuple(pair for block in blocks for pair in pairs(block))
    if len(observed_pairs) != 144 or len(set(observed_pairs)) != 144:
        raise AuditError("the 144 observed constituent pairs must be distinct")
    queries = tuple(tuple(sorted(query)) for query in packet.get("unseen_queries", []))
    if len(queries) != 24 or len(set(queries)) != 24:
        raise AuditError("packet must contain 24 distinct unseen query pairs")
    if any(len(query) != 2 or query[0] == query[1] for query in queries):
        raise AuditError("every unseen query must be a distinct Event pair")
    if any(event not in universe for query in queries for event in query):
        raise AuditError("query contains an Event outside U")
    if any(query in set(observed_pairs) for query in queries):
        raise AuditError("an unseen query is already an observed constituent pair")
    if packet.get("sizes") != {"observed_triads": 48, "unseen_queries": 24}:
        raise AuditError("declared packet sizes mismatch")
    return packet, universe, blocks, tuple(sorted(queries))


def event_components(universe: Sequence[int], blocks: Sequence[Block]) -> list[dict[str, Any]]:
    parent = list(universe)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for block in blocks:
        union(block[0], block[1])
        union(block[0], block[2])
    grouped: dict[int, list[int]] = {}
    for event in universe:
        grouped.setdefault(find(event), []).append(event)
    components = []
    for events in sorted((tuple(sorted(value)) for value in grouped.values())):
        event_set = set(events)
        indices = tuple(i for i, block in enumerate(blocks) if set(block) <= event_set)
        component_blocks = tuple(blocks[i] for i in indices)
        if sum(len(block) for block in component_blocks) == 0:
            raise AuditError("isolated Event component is not expected")
        components.append(
            {"events": events, "block_indices": indices, "blocks": component_blocks}
        )
    return components


def component_isomorphisms(source: dict[str, Any], target: dict[str, Any]) -> list[tuple[int, ...]]:
    source_events = source["events"]
    target_events = target["events"]
    if len(source_events) != len(target_events) or len(source["blocks"]) != len(target["blocks"]):
        return []
    target_blocks = set(target["blocks"])
    found: list[tuple[int, ...]] = []
    for images in itertools.permutations(target_events):
        mapping = dict(zip(source_events, images, strict=True))
        mapped = {tuple(sorted(mapping[x] for x in block)) for block in source["blocks"]}
        if mapped == target_blocks:
            found.append(tuple(images))
    return sorted(found)


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""
    return tuple(left[right[i]] for i in range(len(left)))


def generated_closure(generators: Sequence[Permutation], degree: int) -> set[Permutation]:
    identity = tuple(range(degree))
    closure = {identity}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            candidate = compose(generator, current)
            if candidate not in closure:
                closure.add(candidate)
                queue.append(candidate)
    return closure


def greedy_generators(group: Sequence[Permutation]) -> list[Permutation]:
    if not group:
        raise AuditError("cannot generate an empty permutation group")
    degree = len(group[0])
    target = set(group)
    generators: list[Permutation] = []
    closure = generated_closure(generators, degree)
    for candidate in sorted(target):
        if candidate not in closure:
            generators.append(candidate)
            closure = generated_closure(generators, degree)
        if closure == target:
            break
    if closure != target:
        raise AuditError("failed to generate an enumerated component group")
    return generators


def globalize_local(events: Sequence[int], local: Permutation, degree: int) -> Permutation:
    result = list(range(degree))
    for source_index, source_event in enumerate(events):
        result[source_event] = events[local[source_index]]
    return tuple(result)


def localize_images(source_events: Sequence[int], images: Sequence[int], target_events: Sequence[int]) -> Permutation:
    target_index = {event: i for i, event in enumerate(target_events)}
    return tuple(target_index[event] for event in images)


def component_swap(
    left: dict[str, Any], right: dict[str, Any], degree: int
) -> Permutation:
    isomorphisms = component_isomorphisms(left, right)
    if not isomorphisms:
        raise AuditError("requested swap between nonisomorphic components")
    images = isomorphisms[0]
    result = list(range(degree))
    forward = dict(zip(left["events"], images, strict=True))
    inverse = {value: key for key, value in forward.items()}
    for source, target in forward.items():
        result[source] = target
    for source, target in inverse.items():
        result[source] = target
    return tuple(result)


def permutation_cycles(permutation: Permutation) -> list[list[int]]:
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for start in range(len(permutation)):
        if start in seen or permutation[start] == start:
            continue
        cycle = []
        point = start
        while point not in seen:
            seen.add(point)
            cycle.append(point)
            point = permutation[point]
        cycles.append(cycle)
    return cycles


def maps_block(permutation: Permutation, block: Block) -> Block:
    return tuple(sorted(permutation[event] for event in block))  # type: ignore[return-value]


def maps_pair(permutation: Permutation, pair: Pair) -> Pair:
    return tuple(sorted(permutation[event] for event in pair))  # type: ignore[return-value]


def build_group(
    universe: Sequence[int], blocks: Sequence[Block]
) -> tuple[list[Permutation], dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    components = event_components(universe, blocks)
    classes: list[list[int]] = []
    for index, component in enumerate(components):
        for component_class in classes:
            if component_isomorphisms(components[component_class[0]], component):
                component_class.append(index)
                break
        else:
            classes.append([index])

    all_generators: list[Permutation] = []
    generator_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    local_groups: dict[int, list[Permutation]] = {}
    local_generators: dict[int, list[Permutation]] = {}

    for index, component in enumerate(components):
        events = component["events"]
        raw_autos = component_isomorphisms(component, component)
        localized = [localize_images(events, images, events) for images in raw_autos]
        generators = greedy_generators(localized)
        if generated_closure(generators, len(events)) != set(localized):
            raise AuditError("component generator closure mismatch")
        local_groups[index] = localized
        local_generators[index] = generators
        component_records.append(
            {
                "component_id": index,
                "events": list(events),
                "block_indices": list(component["block_indices"]),
                "blocks": [list(block) for block in component["blocks"]],
                "automorphism_order": len(localized),
                "enumerated_automorphisms": [list(permutation) for permutation in localized],
                "local_generator_count": len(generators),
            }
        )
        for generator_index, generator in enumerate(generators):
            global_generator = globalize_local(events, generator, len(universe))
            all_generators.append(global_generator)
            generator_records.append(
                {
                    "name": f"component_{index}_internal_{generator_index}",
                    "kind": "component_internal",
                    "component_id": index,
                    "images": list(global_generator),
                    "cycles": permutation_cycles(global_generator),
                }
            )

    for class_index, component_class in enumerate(classes):
        for swap_index, (left_id, right_id) in enumerate(
            zip(component_class, component_class[1:], strict=False)
        ):
            generator = component_swap(
                components[left_id], components[right_id], len(universe)
            )
            all_generators.append(generator)
            generator_records.append(
                {
                    "name": f"class_{class_index}_adjacent_swap_{swap_index}",
                    "kind": "component_swap",
                    "component_class": class_index,
                    "components": [left_id, right_id],
                    "images": list(generator),
                    "cycles": permutation_cycles(generator),
                }
            )

    block_set = set(blocks)
    verification = []
    for record, generator in zip(generator_records, all_generators, strict=True):
        bijective = sorted(generator) == list(universe)
        preserves = {maps_block(generator, block) for block in blocks} == block_set
        verification.append(
            {"name": record["name"], "bijective": bijective, "preserves_B0": preserves}
        )
        if not bijective or not preserves:
            raise AuditError(f"invalid reported generator {record['name']}")

    class_records = []
    group_order = 1
    for class_index, component_class in enumerate(classes):
        orders = {len(local_groups[index]) for index in component_class}
        if len(orders) != 1:
            raise AuditError("isomorphic components have different automorphism orders")
        internal_order = orders.pop()
        multiplicity = len(component_class)
        factor = internal_order**multiplicity * math.factorial(multiplicity)
        group_order *= factor
        class_records.append(
            {
                "class_id": class_index,
                "component_ids": component_class,
                "multiplicity": multiplicity,
                "representative_component": component_class[0],
                "component_automorphism_order": internal_order,
                "wreath_product_factor": factor,
                "factor_formula": f"{internal_order}^{multiplicity} * {multiplicity}!",
                "all_pairwise_isomorphic": all(
                    bool(component_isomorphisms(components[component_class[0]], components[index]))
                    for index in component_class
                ),
            }
        )

    certificate = {
        "method": "exact connected-component enumeration and disjoint-union wreath-product decomposition",
        "incidence_reduction": (
            "A typed incidence graph has one Event vertex for each U element and one "
            "block vertex for each B0 block, with incidence edges. Event and block "
            "types are fixed. Because each block vertex is uniquely determined by "
            "its three Event neighbors, its typed graph automorphisms are exactly "
            "the Event permutations preserving B0."
        ),
        "component_count": len(components),
        "component_classes": class_records,
        "order_formula": " * ".join(record["factor_formula"] for record in class_records),
        "group_order": group_order,
        "generator_count": len(all_generators),
        "generators": generator_records,
        "generator_verification": verification,
        "completeness_proof": [
            "Every automorphism preserves connectedness and therefore permutes connected components only within an isomorphism class.",
            "All 6! Event bijections were tested in every component and between every class representative and member; the listed component automorphism groups and isomorphism classes are exhaustive.",
            "Within each component, the emitted internal generators close to the complete enumerated component automorphism group.",
            "The emitted adjacent component swaps generate the full symmetric group on each isomorphism class.",
            "Thus the emitted group is the direct product over classes of Aut(component) wreath S_m, and its displayed product order is the full Aut(U,B0), not a discovered subgroup.",
        ],
        "completeness_checks": {
            "all_component_bijections_exhaustively_tested": True,
            "all_local_generator_closures_exact": True,
            "adjacent_swaps_cover_each_class": True,
            "all_global_generators_verified": all(
                item["bijective"] and item["preserves_B0"] for item in verification
            ),
        },
    }
    return all_generators, certificate, component_records, classes


def orbit(seed: Any, generators: Sequence[Permutation], action: Callable[[Permutation, Any], Any]) -> tuple[Any, ...]:
    found = {seed}
    queue = deque([seed])
    while queue:
        current = queue.popleft()
        for generator in generators:
            image = action(generator, current)
            if image not in found:
                found.add(image)
                queue.append(image)
    return tuple(sorted(found))


def partition_orbits(elements: Iterable[Any], generators: Sequence[Permutation], action: Callable[[Permutation, Any], Any]) -> list[tuple[Any, ...]]:
    remaining = set(elements)
    result = []
    while remaining:
        seed = min(remaining)
        current = orbit(seed, generators, action)
        if not set(current) <= remaining:
            raise AuditError("orbit partition crossed an already assigned orbit")
        remaining.difference_update(current)
        result.append(current)
    return result


def action_event(permutation: Permutation, event: int) -> int:
    return permutation[event]


def action_pair(permutation: Permutation, pair: Pair) -> Pair:
    return maps_pair(permutation, pair)


def action_block(permutation: Permutation, block: Block) -> Block:
    return maps_block(permutation, block)


def local_candidates(query: Pair, universe: Sequence[int], observed_pairs: set[Pair]) -> tuple[int, ...]:
    x, y = query
    return tuple(
        z
        for z in universe
        if z not in query
        and tuple(sorted((x, z))) not in observed_pairs
        and tuple(sorted((y, z))) not in observed_pairs
    )


def stabilizer_generators(
    query: Pair,
    components: Sequence[dict[str, Any]],
    classes: Sequence[Sequence[int]],
    universe_size: int,
) -> tuple[list[Permutation], int, int]:
    query_component = next(
        index for index, component in enumerate(components) if set(query) <= set(component["events"])
    )
    query_class = next(index for index, component_class in enumerate(classes) if query_component in component_class)
    generators: list[Permutation] = []
    class_orders: dict[int, int] = {}
    local_groups: dict[int, list[Permutation]] = {}

    for index, component in enumerate(components):
        events = component["events"]
        raw = component_isomorphisms(component, component)
        group = [localize_images(events, images, events) for images in raw]
        local_groups[index] = group
        component_class = next(i for i, ids in enumerate(classes) if index in ids)
        class_orders[component_class] = len(group)
        if index == query_component:
            index_of = {event: i for i, event in enumerate(events)}
            local_query = {index_of[event] for event in query}
            subgroup = [
                permutation
                for permutation in group
                if {permutation[index] for index in local_query} == local_query
            ]
            chosen = greedy_generators(subgroup)
        else:
            chosen = greedy_generators(group)
        generators.extend(globalize_local(events, generator, universe_size) for generator in chosen)

    for class_index, component_class in enumerate(classes):
        swappable = [index for index in component_class if index != query_component]
        for left_id, right_id in zip(swappable, swappable[1:], strict=False):
            generators.append(
                component_swap(components[left_id], components[right_id], universe_size)
            )

    query_component_data = components[query_component]
    query_events = query_component_data["events"]
    event_index = {event: i for i, event in enumerate(query_events)}
    local_query = {event_index[event] for event in query}
    local_stabilizer_order = sum(
        {permutation[index] for index in local_query} == local_query
        for permutation in local_groups[query_component]
    )
    stabilizer_order = local_stabilizer_order
    for class_index, component_class in enumerate(classes):
        count = len(component_class) - (1 if class_index == query_class else 0)
        stabilizer_order *= class_orders[class_index] ** count * math.factorial(count)
    return generators, stabilizer_order, query_component


def analyze_triple_orbit(
    triple_orbit: Sequence[Block], observed_pairs: set[Pair]
) -> dict[str, Any]:
    observed_collisions: dict[Pair, list[Block]] = {}
    internal_owners: dict[Pair, list[Block]] = {}
    for triple in triple_orbit:
        for pair in pairs(triple):
            if pair in observed_pairs:
                observed_collisions.setdefault(pair, []).append(triple)
            internal_owners.setdefault(pair, []).append(triple)
    internal_collisions = {
        pair: owners for pair, owners in internal_owners.items() if len(owners) > 1
    }
    linear = not observed_collisions and not internal_collisions
    return {
        "orbit_size": len(triple_orbit),
        "blocks": [list(triple) for triple in triple_orbit],
        "linear_with_B0": linear,
        "collision_category": (
            "none"
            if linear
            else "observed_and_internal"
            if observed_collisions and internal_collisions
            else "collision_with_B0"
            if observed_collisions
            else "internal_orbit_collision"
        ),
        "observed_collision_count": len(observed_collisions),
        "internal_collision_count": len(internal_collisions),
        "observed_collision_certificates": [
            {"pair": list(pair), "orbit_blocks": [list(block) for block in owners]}
            for pair, owners in sorted(observed_collisions.items())
        ],
        "internal_collision_certificates": [
            {"pair": list(pair), "orbit_blocks": [list(block) for block in owners]}
            for pair, owners in sorted(internal_collisions.items())
        ],
    }


def histogram(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def build_report(raw: bytes) -> dict[str, Any]:
    _packet, universe, blocks, queries = validate_packet(raw)
    observed_pairs = set(pair for block in blocks for pair in pairs(block))
    generators, group_certificate, component_records, classes = build_group(universe, blocks)
    components = event_components(universe, blocks)

    event_orbits = partition_orbits(universe, generators, action_event)
    block_orbits = partition_orbits(blocks, generators, action_block)
    all_pairs = tuple(itertools.combinations(universe, 2))
    pair_orbits = partition_orbits(all_pairs, generators, action_pair)
    pair_orbit_lookup = {
        pair: index for index, current_orbit in enumerate(pair_orbits) for pair in current_orbit
    }

    triple_records: list[dict[str, Any]] = []
    triple_to_record: dict[Block, int] = {}
    query_records: list[dict[str, Any]] = []

    for query in queries:
        local = local_candidates(query, universe, observed_pairs)
        candidate_records = []
        surviving = []
        for z in local:
            triple = tuple(sorted((*query, z)))
            if triple not in triple_to_record:
                triple_orbit = orbit(triple, generators, action_block)
                record_id = len(triple_records)
                analysis = analyze_triple_orbit(triple_orbit, observed_pairs)
                triple_records.append(
                    {
                        "orbit_id": f"T{record_id:04d}",
                        "representative": list(min(triple_orbit)),
                        **analysis,
                    }
                )
                for member in triple_orbit:
                    triple_to_record[member] = record_id
            record = triple_records[triple_to_record[triple]]
            admissible = bool(record["linear_with_B0"])
            if admissible:
                surviving.append(z)
            candidate_records.append(
                {
                    "candidate": z,
                    "triple": list(triple),
                    "orbit_id": record["orbit_id"],
                    "orbit_size": record["orbit_size"],
                    "symmetry_admissible": admissible,
                    "elimination_reason": None if admissible else record["collision_category"],
                }
            )

        symmetry = tuple(surviving)
        if symmetry == local:
            classification = "unchanged"
        elif len(symmetry) == 0:
            classification = "symmetry-incompatible"
        elif len(symmetry) == 1:
            classification = "conditionally identified"
        elif 1 < len(symmetry) < len(local):
            classification = "narrowed"
        else:
            raise AuditError("candidate classification is not exhaustive")

        stabilizer_gens, stabilizer_order, query_component = stabilizer_generators(
            query, components, classes, len(universe)
        )
        if any(maps_pair(generator, query) != query for generator in stabilizer_gens):
            raise AuditError("constructed stabilizer generator does not fix query setwise")
        stabilizer_event_orbits = partition_orbits(universe, stabilizer_gens, action_event)
        pair_orbit = pair_orbits[pair_orbit_lookup[query]]
        orbit_stabilizer_order = group_certificate["group_order"] // len(pair_orbit)
        if stabilizer_order != orbit_stabilizer_order:
            raise AuditError("pair stabilizer order disagrees with orbit-stabilizer theorem")

        ambiguity_witnesses = []
        for z in symmetry[:2]:
            triple = tuple(sorted((*query, z)))
            triple_record = triple_records[triple_to_record[triple]]
            ambiguity_witnesses.append(
                {
                    "candidate": z,
                    "orbit_id": triple_record["orbit_id"],
                    "extension_blocks": triple_record["blocks"],
                    "extension_is_G_invariant": True,
                    "extension_is_linear": True,
                }
            )
        if len(symmetry) > 1 and len(ambiguity_witnesses) != 2:
            raise AuditError("ambiguous query lacks two exact extension witnesses")

        eliminated = [record for record in candidate_records if not record["symmetry_admissible"]]
        query_records.append(
            {
                "query": list(query),
                "query_pair_orbit_id": pair_orbit_lookup[query],
                "query_pair_orbit_size": len(pair_orbit),
                "query_component_id": query_component,
                "setwise_stabilizer": {
                    "order": stabilizer_order,
                    "orbit_stabilizer_verified": True,
                    "event_orbits": [list(current) for current in stabilizer_event_orbits],
                    "local_candidate_orbits": [
                        sorted(set(current) & set(local))
                        for current in stabilizer_event_orbits
                        if set(current) & set(local)
                    ],
                    "symmetry_candidate_orbits": [
                        sorted(set(current) & set(symmetry))
                        for current in stabilizer_event_orbits
                        if set(current) & set(symmetry)
                    ],
                },
                "C_local": list(local),
                "C_local_size": len(local),
                "C_G": list(symmetry),
                "C_G_size": len(symmetry),
                "classification": classification,
                "candidates": candidate_records,
                "elimination_counts": {
                    "collision_with_B0": sum(
                        record["elimination_reason"] in {"collision_with_B0", "observed_and_internal"}
                        for record in eliminated
                    ),
                    "internal_orbit_collision": sum(
                        record["elimination_reason"] in {"internal_orbit_collision", "observed_and_internal"}
                        for record in eliminated
                    ),
                },
                "ambiguity_witnesses": ambiguity_witnesses,
                "singleton_exclusion_certificate": (
                    [
                        {
                            "candidate": record["candidate"],
                            "orbit_id": record["orbit_id"],
                            "reason": record["elimination_reason"],
                        }
                        for record in eliminated
                    ]
                    if len(symmetry) == 1
                    else []
                ),
            }
        )

    query_orbit_groups: dict[int, list[list[int]]] = {}
    for record in query_records:
        query_orbit_groups.setdefault(record["query_pair_orbit_id"], []).append(record["query"])

    classifications = Counter(record["classification"] for record in query_records)
    before_histogram = histogram(record["C_local_size"] for record in query_records)
    after_histogram = histogram(record["C_G_size"] for record in query_records)
    total_eliminations = sum(
        record["C_local_size"] - record["C_G_size"] for record in query_records
    )
    observed_eliminations = sum(
        record["elimination_counts"]["collision_with_B0"] for record in query_records
    )
    internal_eliminations = sum(
        record["elimination_counts"]["internal_orbit_collision"] for record in query_records
    )

    if observed_eliminations:
        raise AuditError(
            "a locally admissible triple orbit collided with B0 despite G-invariance"
        )
    if total_eliminations != internal_eliminations:
        raise AuditError("elimination accounting mismatch")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "information_exposure": INFORMATION_EXPOSURE,
        "provenance": {
            "task_uri": TASK_URI,
            "task_revision": TASK_REVISION,
            "input_uri": INPUT_URI,
            "input_raw_sha256": sha256_bytes(raw),
            "runner_sha256": sha256_file(Path(__file__)),
            "python": platform.python_version(),
            "group_algorithm_dependencies": "Python standard library only",
            "transport_library_versions_if_used": quilt_versions(),
        },
        "input_validation": {
            "universe_size": len(universe),
            "observed_block_count": len(blocks),
            "distinct_constituent_pair_count": len(observed_pairs),
            "unseen_query_count": len(queries),
            "all_blocks_three_distinct_events": True,
            "B0_is_linear": True,
            "queries_excluded_from_group_construction": True,
        },
        "observed_automorphism_group": {
            **group_certificate,
            "components": component_records,
            "event_orbits": [list(current) for current in event_orbits],
            "observed_block_orbits": [
                [list(block) for block in current] for current in block_orbits
            ],
            "unordered_pair_orbits": [
                {
                    "pair_orbit_id": index,
                    "size": len(current),
                    "pairs": [list(pair) for pair in current],
                }
                for index, current in enumerate(pair_orbits)
            ],
        },
        "candidate_triple_orbits": triple_records,
        "queries": query_records,
        "query_orbit_equivalence": [
            {
                "query_pair_orbit_id": orbit_id,
                "queries": grouped_queries,
                "query_count": len(grouped_queries),
                "share_G_equivalent_candidate_structure": True,
            }
            for orbit_id, grouped_queries in sorted(query_orbit_groups.items())
        ],
        "summary": {
            "classification_counts": {
                name: classifications.get(name, 0)
                for name in (
                    "unchanged",
                    "narrowed",
                    "conditionally identified",
                    "symmetry-incompatible",
                )
            },
            "candidate_size_histogram_before": before_histogram,
            "candidate_size_histogram_after": after_histogram,
            "candidate_eliminations": {
                "total": total_eliminations,
                "collision_with_B0": observed_eliminations,
                "internal_orbit_collision": internal_eliminations,
            },
            "conditional_constitution": (
                "B0 has automorphisms G (fact). Requiring extensions to preserve all "
                "of G is an additional constitution, not a consequence of that fact. "
                "The reported C_G sets are conditional on completing each query."
            ),
            "scientific_fence": (
                "Observed-partial symmetry is tested in isolation. No existence, "
                "maximality, total-block-count, degree, Steiner, Fano/projective, "
                "coordinate, XOR, algebraic, joint-query, or learning axiom is used."
            ),
        },
        "mechanical_checks": {
            "every_generator_preserves_B0": True,
            "full_group_completeness_certified": True,
            "every_candidate_orbit_exact_under_generators": True,
            "every_surviving_extension_linear": True,
            "every_surviving_extension_G_invariant": True,
            "every_elimination_has_collision_certificate": all(
                record["linear_with_B0"]
                or record["observed_collision_certificates"]
                or record["internal_collision_certificates"]
                for record in triple_records
            ),
            "every_ambiguous_query_has_two_extensions": all(
                record["C_G_size"] <= 1 or len(record["ambiguity_witnesses"]) >= 2
                for record in query_records
            ),
        },
    }
    return report


def format_set(values: Sequence[int]) -> str:
    return "{" + ", ".join(str(value) for value in values) + "}"


def render_markdown(report: dict[str, Any], json_path: Path) -> str:
    group = report["observed_automorphism_group"]
    summary = report["summary"]
    lines = [
        "Information exposure:",
        "",
        report["information_exposure"],
        "",
        "# 016.02 — Observed-symmetry completion audit result",
        "",
        "## Verdict",
        "",
        f"The exact observed automorphism group has order **{group['group_order']}**. "
        "The result below tests full preservation of that group only as an explicit "
        "additional completion constitution; the existence of the group does not "
        "logically require an extension to preserve it.",
        "",
        "Classification counts: "
        + ", ".join(
            f"**{name} {count}**"
            for name, count in summary["classification_counts"].items()
        )
        + ".",
        "",
        f"Candidate-size histogram before symmetry: `{summary['candidate_size_histogram_before']}`. "
        f"After symmetry: `{summary['candidate_size_histogram_after']}`.",
        "",
        f"Of {summary['candidate_eliminations']['total']} candidate eliminations, "
        f"{summary['candidate_eliminations']['collision_with_B0']} arise from collision "
        f"with B0 and {summary['candidate_eliminations']['internal_orbit_collision']} "
        "from pair collisions internal to the candidate orbit.",
        "",
        "## Exact group certificate",
        "",
        group["incidence_reduction"],
        "",
        f"The typed incidence graph has {group['component_count']} connected components "
        f"in {len(group['component_classes'])} isomorphism classes. The exhaustive "
        f"component decomposition gives `|G| = {group['order_formula']} = "
        f"{group['group_order']}`. The JSON contains all component automorphisms, "
        f"{group['generator_count']} deterministic Event-permutation generators, their "
        "cycle forms, per-generator B0 checks, all Event/block/pair orbits, and the "
        "wreath-product completeness certificate.",
        "",
        "Event orbits:",
        "",
    ]
    for index, current in enumerate(group["event_orbits"]):
        lines.append(f"- E{index}: `{current}`")
    lines.extend(
        [
            "",
            "Observed-block orbit sizes: `"
            + str([len(current) for current in group["observed_block_orbits"]])
            + "`.",
            "",
            "Unordered-pair orbit sizes: `"
            + str([current["size"] for current in group["unordered_pair_orbits"]])
            + "`.",
            "",
            "## Per-query exact candidate sets",
            "",
            "Each stabilizer is the setwise stabilizer of the unordered query pair. "
            "Its order is independently checked by orbit–stabilizer; exact stabilizer "
            "Event and candidate orbits are in the JSON.",
            "",
            "| Query | |Stab_G(q)| | C_local(q) | C_G(q) | Class |",
            "|---|---:|---|---|---|",
        ]
    )
    for query in report["queries"]:
        lines.append(
            f"| `{query['query']}` | {query['setwise_stabilizer']['order']} | "
            f"{format_set(query['C_local'])} ({query['C_local_size']}) | "
            f"{format_set(query['C_G'])} ({query['C_G_size']}) | "
            f"**{query['classification']}** |"
        )
    lines.extend(
        [
            "",
            "## Query-orbit equivalence and witnesses",
            "",
        ]
    )
    for group_record in report["query_orbit_equivalence"]:
        lines.append(
            f"- Pair orbit {group_record['query_pair_orbit_id']}: "
            f"{group_record['query_count']} handed queries `{group_record['queries']}`. "
            "They share G-equivalent candidate structure."
        )
    lines.extend(
        [
            "",
            "For every query with more than one surviving candidate, the JSON gives "
            "two explicit witnesses. Each witness lists the complete orbit of its "
            "candidate triple; adjoining those listed blocks to B0 is mechanically "
            "verified both linear and G-invariant. Every eliminated candidate names "
            "its triple orbit, whose complete blocks and exact pair-collision "
            "certificates are also included. A singleton, if present, includes every "
            "alternative's exclusion certificate.",
            "",
            "## Logical scope",
            "",
            "A. `B0` has the exact automorphism group `G` reported here.",
            "",
            "B. Requiring a completion to remain invariant under all of `G` is an "
            "additional constitution chosen for this audit.",
            "",
            "C. Only under B may orbit closure eliminate locally admissible candidates. "
            "Even a strong conditional narrowing would not show that nature or OT "
            "preserves accidental symmetries of missing data. Conversely, ambiguity "
            "under B would establish insufficiency only for this tested principle.",
            "",
            "No claim is made that symmetry forces any new block to exist: B0 itself "
            "is already G-invariant. No hidden target was consulted, and no stronger "
            "completion axiom was added.",
            "",
            "## Artifacts and reproduction",
            "",
            f"- Task revision: `{TASK_REVISION}`",
            f"- Input raw SHA-256: `{report['provenance']['input_raw_sha256']}`",
            f"- `audit_observed_symmetry.py` SHA-256: `{report['provenance']['runner_sha256']}`",
            f"- `observed_symmetry_audit.json` SHA-256: `{sha256_file(json_path)}`",
            "- Expected final Quilt manifest delta: `+3`.",
            "",
            "From the repository root (with Quilt credentials for the immutable default input):",
            "",
            "```bash",
            "python issues/016-global-relational-completion/\\",
            "016.02-Code-attachments/audit_observed_symmetry.py",
            "```",
            "",
            "The command deterministically regenerates both delivered outputs and "
            "rechecks all certificates. Add `--verify` to compare regenerated bytes "
            "against the accepted files without rewriting them.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> int:
    raw = read_input(args.input)
    report = build_report(raw)
    json_bytes = canonical_json(report)
    if args.verify:
        expected_markdown = render_markdown(report, args.json_output).encode()
        failures = []
        if not args.json_output.is_file() or args.json_output.read_bytes() != json_bytes:
            failures.append(f"JSON differs from deterministic regeneration: {args.json_output}")
        if not args.markdown_output.is_file() or args.markdown_output.read_bytes() != expected_markdown:
            failures.append(
                f"Markdown differs from deterministic regeneration: {args.markdown_output}"
            )
        print(
            json.dumps(
                {
                    "status": "FAIL" if failures else "PASS",
                    "failures": failures,
                    "group_order": report["observed_automorphism_group"]["group_order"],
                    "summary": report["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if failures else 0

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_bytes(json_bytes)
    markdown = render_markdown(report, args.json_output)
    args.markdown_output.write_text(markdown)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "json": str(args.json_output),
                "markdown": str(args.markdown_output),
                "group_order": report["observed_automorphism_group"]["group_order"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", default=INPUT_URI)
    result.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    result.add_argument("--markdown-output", type=Path, default=DEFAULT_RESULT)
    result.add_argument("--verify", action="store_true")
    return result


def main() -> int:
    try:
        return execute(parser().parse_args())
    except AuditError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
