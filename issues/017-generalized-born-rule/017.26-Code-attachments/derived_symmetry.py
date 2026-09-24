"""Issue 017.26 Phase S: derive invariance from typed world structure, before labels.

017.25a established that a compiled quotient can beat the declared relation, but
the symmetry itself was still handed to the compiler as declared generators.
017.26 removes that. 017.26b tightens how it may be removed. This module is the
structure-only half of the execution contract, and it is frozen and committed
*before* the statistical half exists.

What that means mechanically is the point of this file, so it is worth stating
plainly: this module imports no label generator, no estimator, no seed schedule,
and no target. It does not import the byte-pinned 017.24 implementation at all,
because that module contains ``generate_observations``. The claim "no label was in
scope during derivation" is therefore not a promise about discipline; it is a
property of the import graph. The Phase T module is a separate file that pins
this one.

Three things are derived here from ordinary typed world descriptions.

**A generic schema-semantic compiler.** A declared world is a set of typed
carriers plus primitives, each carrying a declared kind and signature. The
compiler generates each primitive's preservation constraint mechanically from
that kind and signature -- membership for a relation, commutation for an
operation, fixing for a constant -- and derives

    Aut(A_Sigma) = {tuples of carrier bijections preserving every primitive}

by exhaustive backtracking over the full product of symmetric groups, then

    q_Sigma : X_Sigma -> X_Sigma / Aut(A_Sigma)

on the declared admitted-input product. There is no branch on fixture name, no
expected group order, no declared generator, and no hand-written classifier. That
is audited three ways: a source scan, a relabelling-invariance test that renames
every carrier, every element, and every primitive and demands identical output,
and a key-stripping test that deletes every document key outside the structural
subtree and demands identical output.

**A frozen bounded shortcut language.** 017.26b section 4 replaces the phrase
"cheap statistic" with a preregistered finite language L_k. It is built here from
the declared primitives alone by a fixed grammar to a fixed depth, enumerated
exhaustively, and each feature's induced partition of the admitted inputs is
compared against the derived quotient. Because every L_k feature is a term over
the declared primitives, every L_k feature is automorphism-invariant, so its
partition can only be coarser than or equal to the orbit partition; "recovers
q_Sigma" therefore means "equals". The audit reports the joint partition of the
whole language as well as the finest single feature, which is strictly stronger
than the section 4 requirement, and it reports raw count values rather than only
the equality tests between them, which is stronger again.

**A disclosed non-local adversary that L_k does not contain.** Canary's review
asked for the cheaper-explanation claim to be bounded rather than exhaustive, and
017.26b section 4.1 requires the bound to be named. Rather than only naming it,
this module implements the obvious stronger adversary -- an iterated pair
refinement over the admitted inputs, in the Weisfeiler-Leman idiom -- and banks
what it finds. It is not a member of L_k, which is source-local and depth-bounded
by construction, and condition R is forbidden from using it by 017.26b section 8.
But what it finds is recorded in the freeze rather than discovered afterwards.

Everything here is exact and uses the standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

WORLD_A_SCHEMA: Final = "gpt-01726-single-relation-world/v1"
WORLD_B_SCHEMA: Final = "gpt-01726-multi-primitive-world/v1"
CERTIFICATE_SCHEMA: Final = "gpt-01726-phase-s-structure-freeze/v1"
COST_SCHEMA: Final = "gpt-01726-phase-s-cost-report/v1"

WORLD_A_SHA256: Final = (
    "9f5262583012fb01aa4a9508d3abe3bfd9bb8135bd172d8342dc755004e7fc67"
)
WORLD_B_SHA256: Final = (
    "79412d9349c9413831bb44cae6bcd332ae265a6928facb2d7a7ec161c0b32b5d"
)

TASK_GPT_REVISION: Final = (
    "be3d592a9c1a975f52cc292d6bfb6689c8c63d3c8bfadcbe796c427f1399657b"
)
PRIOR_RESULT_COMMIT: Final = "2cb043a7c92ad16b9e9df784fce82d3d61531b24"
PRIOR_QUILT_REVISION: Final = (
    "adbc039c6b9459f92991caa4810890104670affafd62284cc2371a8181ecf84a"
)

HERE: Final = Path(__file__).resolve().parent
WORLD_A_PATH: Final = HERE / "world_a_schema.json"
WORLD_B_PATH: Final = HERE / "world_b_schema.json"
CERTIFICATE_PATH: Final = HERE / "phase_s_certificate.json"
COST_PATH: Final = HERE / "phase_s_cost_report.json"

# The declared structural subtree. The compiler may read these keys of a world
# document and nothing else, and that restriction is tested by deleting every
# other key and demanding identical output.
STRUCTURAL_KEYS: Final = ("sorts", "primitives", "admitted_inputs")

DOCUMENT_KEY_WHITELIST: Final = (
    "admitted_inputs",
    "answer_type",
    "compiler_visible_semantics",
    "domain",
    "fixture",
    "independence_fence",
    "not_declared_here",
    "primitives",
    "role",
    "schema",
    "sorts",
)

# Vocabulary that may not appear anywhere in the structural subtree of a world
# document. This is the mechanical form of the 017.26 section 2 source fence.
FORBIDDEN_STRUCTURAL_VOCABULARY: Final = (
    "answer",
    "automorph",
    "canonical",
    "class",
    "equivalen",
    "frequency",
    "generator",
    "group",
    "invariant",
    "label",
    "orbit",
    "parameter",
    "partition",
    "probabilit",
    "quotient",
    "represent",
    "share",
    "stabili",
    "symmetr",
    "target",
    "theta",
    "tying",
)

# Vocabulary that may not appear in the source of any function on the compiler
# path. The first block is the 017.26b section 3.1 no-fixture-dispatch fence; the
# second is the 017.26 section 13 label-independence fence. These are matched as
# whole lowercase word tokens rather than as substrings, because substring
# matching produces false positives on ordinary English -- "unconstrained"
# contains "train" -- and a check that cries wolf is a check nobody reads.
FORBIDDEN_COMPILER_TOKENS: Final = (
    "adjacent",
    "answer",
    "answers",
    "district",
    "districts",
    "generator",
    "generators",
    "incidence",
    "incident",
    "label",
    "labels",
    "observation",
    "observations",
    "seed",
    "seeds",
    "theta",
    "train",
    "training",
)

FORBIDDEN_COMPILER_SUBSTRINGS: Final = (
    "expected_cardinal",
    "expected_class",
    "expected_free",
    "expected_group",
    "expected_orbit",
)

COMPILER_PATH: Final = (
    "world_from_document",
    "preserves_every_primitive",
    "_flatten",
    "_all_duties",
    "_variable_order",
    "prepare",
    "_check_depth",
    "_walk",
    "count_morphisms",
    "enumerate_morphisms",
    "exists_morphism_sending",
    "derive_orbit_quotient",
    "compile_world",
)

ENUMERATION_BUDGET: Final = 400_000
SHORTCUT_DEPTH: Final = 2
REFINEMENT_ITERATION_CAP: Final = 8
SCRAMBLE_SEED: Final = 20260925
MAXIMALITY_SAMPLES: Final = 200_000
MAXIMALITY_SAMPLE_SEED: Final = 20260926

Row = tuple[str, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _round(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def _cell_sizes(partition: Sequence[int]) -> list[int]:
    return sorted(
        sum(1 for value in partition if value == cell)
        for cell in sorted(set(partition))
    )


def _refines(finer: Sequence[int], coarser: Sequence[int]) -> bool:
    """True when ``finer`` determines ``coarser``: every finer cell is inside one."""
    witness: dict[int, int] = {}
    for left, right in zip(finer, coarser, strict=True):
        if witness.setdefault(left, right) != right:
            return False
    return True


# ---------------------------------------------------------------------------
# 1. The typed world carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Carrier:
    """One declared typed carrier: a name and its declared elements, in order."""

    name: str
    elements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Primitive:
    """One declared primitive, reduced to its kind, its signature, and its data.

    ``kind`` is the only thing the preservation-rule generator dispatches on, and
    ``argument_carriers`` together with ``value_carrier`` is the whole signature.
    A relation stores its extension; an operation stores its total graph; a
    constant stores its value. Nothing else survives the loader, which is why the
    compiler cannot consult a description, a name with meaning, or an expectation.
    """

    name: str
    kind: str
    argument_carriers: tuple[str, ...]
    value_carrier: str | None
    extension: frozenset[Row]
    graph: tuple[tuple[Row, str], ...]
    value: str | None


@dataclass(frozen=True, slots=True)
class TypedWorld:
    """A declared world: typed carriers, primitives, and the admitted-input type.

    There is deliberately no field for an equivalence, a quotient, a group, a
    generating set, a statistic tying, or an answer. The object handed to the
    compiler cannot represent any of them.
    """

    identifier: str
    carriers: tuple[Carrier, ...]
    primitives: tuple[Primitive, ...]
    input_signature: tuple[str, ...]

    @property
    def carrier_names(self) -> tuple[str, ...]:
        return tuple(carrier.name for carrier in self.carriers)

    def carrier(self, name: str) -> Carrier:
        for candidate in self.carriers:
            if candidate.name == name:
                return candidate
        raise KeyError(f"undeclared carrier {name!r}")

    @property
    def inputs(self) -> tuple[Row, ...]:
        pools = [self.carrier(name).elements for name in self.input_signature]
        return tuple(itertools.product(*pools))


def world_from_document(
    document: dict[str, object], identifier: str
) -> TypedWorld:
    """Project a world document onto the structural carrier, and nothing else.

    Only the keys named in ``STRUCTURAL_KEYS`` are read. Every other key of the
    document -- prose, provenance, the response type, the denial list -- is
    dropped here and is therefore invisible downstream.
    """
    carriers = tuple(
        Carrier(
            name=str(block["name"]),
            elements=tuple(str(item) for item in block["elements"]),
        )
        for block in document["sorts"]  # type: ignore[union-attr]
    )
    primitives: list[Primitive] = []
    for block in document["primitives"]:  # type: ignore[union-attr]
        kind = str(block["kind"])
        if kind == "relation":
            primitives.append(
                Primitive(
                    name=str(block["name"]),
                    kind=kind,
                    argument_carriers=tuple(
                        str(item) for item in block["signature"]
                    ),
                    value_carrier=None,
                    extension=frozenset(
                        tuple(str(item) for item in row)
                        for row in block["extension"]
                    ),
                    graph=(),
                    value=None,
                )
            )
        elif kind == "operation":
            primitives.append(
                Primitive(
                    name=str(block["name"]),
                    kind=kind,
                    argument_carriers=tuple(str(item) for item in block["domain"]),
                    value_carrier=str(block["codomain"]),
                    extension=frozenset(),
                    graph=tuple(
                        (
                            tuple(str(item) for item in entry["arguments"]),
                            str(entry["value"]),
                        )
                        for entry in block["graph"]
                    ),
                    value=None,
                )
            )
        elif kind == "constant":
            primitives.append(
                Primitive(
                    name=str(block["name"]),
                    kind=kind,
                    argument_carriers=(),
                    value_carrier=str(block["sort"]),
                    extension=frozenset(),
                    graph=(),
                    value=str(block["value"]),
                )
            )
        else:
            raise RuntimeError(f"undeclared primitive kind {kind!r}")
    signature = tuple(
        str(item) for item in document["admitted_inputs"]["product"]  # type: ignore[index]
    )
    return TypedWorld(
        identifier=identifier,
        carriers=carriers,
        primitives=tuple(primitives),
        input_signature=signature,
    )


def load_world(
    path: Path, digest: str, schema: str
) -> tuple[TypedWorld, dict[str, object]]:
    """Load a world document byte-pinned, then project it structurally."""
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != digest:
        raise RuntimeError(
            f"world document {path.name} changed: expected {digest}, "
            f"observed {observed}"
        )
    document = json.loads(raw)
    if document["schema"] != schema:
        raise RuntimeError(f"unexpected schema identifier in {path.name}")
    return world_from_document(document, str(document["fixture"])), document


def load_world_a(
    path: Path = WORLD_A_PATH,
) -> tuple[TypedWorld, dict[str, object]]:
    return load_world(path, WORLD_A_SHA256, WORLD_A_SCHEMA)


def load_world_b(
    path: Path = WORLD_B_PATH,
) -> tuple[TypedWorld, dict[str, object]]:
    return load_world(path, WORLD_B_SHA256, WORLD_B_SCHEMA)


# ---------------------------------------------------------------------------
# 2. The preservation rules, generated from declared kind and signature
# ---------------------------------------------------------------------------

# The whole semantic content of the word "generic" lives in this table. It is
# keyed by the declared kind of a primitive, it is the only place the compiler
# dispatches, and it mentions no world, no carrier, and no primitive name.
PRESERVATION_RULES: Final = {
    "relation": (
        "for a declared relation R over carriers (C_1, ..., C_n), a candidate "
        "tuple of carrier bijections g preserves R exactly when "
        "R(a_1, ..., a_n) <=> R(g_{C_1}(a_1), ..., g_{C_n}(a_n)) holds for "
        "every tuple (a_1, ..., a_n) of C_1 x ... x C_n"
    ),
    "operation": (
        "for a declared operation f : C_1 x ... x C_n -> B, a candidate tuple "
        "of carrier bijections g preserves f exactly when "
        "g_B(f(a_1, ..., a_n)) = f(g_{C_1}(a_1), ..., g_{C_n}(a_n)) holds for "
        "every tuple (a_1, ..., a_n) of C_1 x ... x C_n"
    ),
    "constant": (
        "for a declared constant c : C, a candidate tuple of carrier bijections "
        "g preserves c exactly when g_C(c) = c"
    ),
}


@dataclass(frozen=True, slots=True)
class Morphism:
    """A tuple of carrier bijections, one per declared carrier.

    ``images`` is keyed by carrier name and positionally aligned with that
    carrier's declared element order, so equality of two morphisms is equality of
    two tuples and the object is hashable.
    """

    images: tuple[tuple[str, tuple[str, ...]], ...]

    def as_mapping(self, world: TypedWorld) -> dict[str, dict[str, str]]:
        lookup: dict[str, dict[str, str]] = {}
        for name, images in self.images:
            elements = world.carrier(name).elements
            lookup[name] = dict(zip(elements, images, strict=True))
        return lookup

    def send(self, world: TypedWorld, row: Row) -> Row:
        mapping = self.as_mapping(world)
        return tuple(
            mapping[carrier][element]
            for carrier, element in zip(world.input_signature, row, strict=True)
        )


def preserves_every_primitive(world: TypedWorld, morphism: Morphism) -> bool:
    """Check a candidate against every declared primitive, from its kind alone.

    This is the independent checker. The search below prunes using the same rules
    compiled into per-position obligations; this function re-derives them from the
    signature on every call, so agreement between the two is a real check rather
    than a tautology.
    """
    mapping = morphism.as_mapping(world)
    for primitive in world.primitives:
        if primitive.kind == "relation":
            moved = {
                tuple(
                    mapping[carrier][element]
                    for carrier, element in zip(
                        primitive.argument_carriers, row, strict=True
                    )
                )
                for row in primitive.extension
            }
            if moved != set(primitive.extension):
                return False
        elif primitive.kind == "operation":
            assignment = dict(primitive.graph)
            target = primitive.value_carrier
            if target is None:
                raise RuntimeError("an operation must declare a codomain")
            for arguments, result in primitive.graph:
                moved_arguments = tuple(
                    mapping[carrier][element]
                    for carrier, element in zip(
                        primitive.argument_carriers, arguments, strict=True
                    )
                )
                if mapping[target][result] != assignment[moved_arguments]:
                    return False
        elif primitive.kind == "constant":
            target = primitive.value_carrier
            if target is None or primitive.value is None:
                raise RuntimeError("a constant must declare a carrier and a value")
            if mapping[target][primitive.value] != primitive.value:
                return False
        else:
            raise RuntimeError(f"undeclared primitive kind {primitive.kind!r}")
    return True


# ---------------------------------------------------------------------------
# 3. The generic compiler: exhaustive backtracking with sound pruning only
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prepared:
    """The compiled search plan for one world, derived from its signatures alone.

    Every ground obligation is attached to the position in the variable order at
    which it first becomes fully determined, so the search checks each obligation
    exactly once and as early as it can. ``remaining`` records how many
    obligations are still outstanding below each depth; when it reaches zero the
    unassigned variables are unconstrained and the number of completions is a
    product of factorials, which is what makes an erased world tractable.
    """

    world: TypedWorld
    order: tuple[int, ...]
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    carrier_of: tuple[int, ...]
    relation_duties: tuple[tuple[tuple[int, ...], bool, int], ...]
    relation_membership: tuple[frozenset[tuple[int, ...]], ...]
    operation_duties: tuple[tuple[tuple[int, ...], int, int], ...]
    operation_tables: tuple[dict[tuple[int, ...], int], ...]
    constant_duties: tuple[int, ...]
    duty_slices: tuple[tuple[int, int, int, int, int, int], ...]
    remaining: tuple[int, ...]

    @property
    def total(self) -> int:
        return len(self.order)


def _flatten(
    world: TypedWorld,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], dict[str, int]]:
    """Lay every element of every carrier out in one contiguous index space."""
    offsets: list[int] = []
    sizes: list[int] = []
    carrier_of: list[int] = []
    running = 0
    for position, carrier in enumerate(world.carriers):
        offsets.append(running)
        sizes.append(len(carrier.elements))
        carrier_of.extend([position] * len(carrier.elements))
        running += len(carrier.elements)
    identifier: dict[str, int] = {}
    for position, carrier in enumerate(world.carriers):
        for index, element in enumerate(carrier.elements):
            identifier[f"{carrier.name}\u0000{element}"] = offsets[position] + index
    return tuple(offsets), tuple(sizes), tuple(carrier_of), identifier


def _all_duties(
    world: TypedWorld, identifier: dict[str, int]
) -> tuple[
    list[tuple[tuple[int, ...], bool, int]],
    list[frozenset[tuple[int, ...]]],
    list[tuple[tuple[int, ...], int, int]],
    list[dict[tuple[int, ...], int]],
    list[int],
]:
    """Expand every primitive into ground obligations, mechanically by kind.

    A relation contributes one obligation per tuple of its whole signature
    product, carrying that tuple's declared truth value, so checking every
    obligation is exactly the biconditional in both directions. An operation
    contributes one obligation per row of its total graph. A constant contributes
    one obligation.
    """
    relation_duties: list[tuple[tuple[int, ...], bool, int]] = []
    relation_membership: list[frozenset[tuple[int, ...]]] = []
    operation_duties: list[tuple[tuple[int, ...], int, int]] = []
    operation_tables: list[dict[tuple[int, ...], int]] = []
    constant_duties: list[int] = []
    for primitive in world.primitives:
        if primitive.kind == "relation":
            pools = [
                world.carrier(name).elements for name in primitive.argument_carriers
            ]
            slot = len(relation_membership)
            relation_membership.append(
                frozenset(
                    tuple(
                        identifier[f"{carrier}\u0000{element}"]
                        for carrier, element in zip(
                            primitive.argument_carriers, row, strict=True
                        )
                    )
                    for row in primitive.extension
                )
            )
            for row in itertools.product(*pools):
                variables = tuple(
                    identifier[f"{carrier}\u0000{element}"]
                    for carrier, element in zip(
                        primitive.argument_carriers, row, strict=True
                    )
                )
                relation_duties.append(
                    (variables, row in primitive.extension, slot)
                )
        elif primitive.kind == "operation":
            target = primitive.value_carrier
            if target is None:
                raise RuntimeError("an operation must declare a codomain")
            table = {
                tuple(
                    identifier[f"{carrier}\u0000{element}"]
                    for carrier, element in zip(
                        primitive.argument_carriers, arguments, strict=True
                    )
                ): identifier[f"{target}\u0000{result}"]
                for arguments, result in primitive.graph
            }
            slot = len(operation_tables)
            operation_tables.append(table)
            for arguments, result in table.items():
                operation_duties.append((arguments, result, slot))
        elif primitive.kind == "constant":
            target = primitive.value_carrier
            if target is None or primitive.value is None:
                raise RuntimeError("a constant must declare a carrier and a value")
            constant_duties.append(identifier[f"{target}\u0000{primitive.value}"])
        else:
            raise RuntimeError(f"undeclared primitive kind {primitive.kind!r}")
    return (
        relation_duties,
        relation_membership,
        operation_duties,
        operation_tables,
        constant_duties,
    )


def _variable_order(
    total: int,
    relation_duties: Sequence[tuple[tuple[int, ...], bool, int]],
    operation_duties: Sequence[tuple[tuple[int, ...], int, int]],
    constant_duties: Sequence[int],
) -> tuple[int, ...]:
    """Order the variables so obligations become checkable as early as possible.

    The rule is fixed and derived from the ground obligations alone: at each step
    take the unassigned variable that makes the largest number of obligations
    fully determined, breaking ties by the variable's own index so the order is
    deterministic. No world identity is consulted. The order changes only how much
    pruning happens, never which leaves exist, so it cannot affect correctness.
    """
    order: list[int] = []
    assigned: set[int] = set()
    pending_relations = list(relation_duties)
    pending_operations = list(operation_duties)
    pending_constants = list(constant_duties)
    while len(order) < total:
        best_variable = -1
        best_score = -1
        for candidate in range(total):
            if candidate in assigned:
                continue
            score = 0
            for variables, _truth, _slot in pending_relations:
                if candidate in variables and all(
                    variable == candidate or variable in assigned
                    for variable in variables
                ):
                    score += 1
            for variables, result, _slot in pending_operations:
                touched = (*variables, result)
                if candidate in touched and all(
                    variable == candidate or variable in assigned
                    for variable in touched
                ):
                    score += 1
            score += sum(1 for element in pending_constants if element == candidate)
            if score > best_score:
                best_score, best_variable = score, candidate
        order.append(best_variable)
        assigned.add(best_variable)
        pending_relations = [
            duty
            for duty in pending_relations
            if not all(variable in assigned for variable in duty[0])
        ]
        pending_operations = [
            duty
            for duty in pending_operations
            if not all(variable in assigned for variable in (*duty[0], duty[1]))
        ]
        pending_constants = [
            element for element in pending_constants if element not in assigned
        ]
    return tuple(order)


def prepare(world: TypedWorld) -> Prepared:
    """Compile a world's declared signatures into a search plan."""
    offsets, sizes, carrier_of, identifier = _flatten(world)
    total = sum(sizes)
    (
        relation_duties,
        relation_membership,
        operation_duties,
        operation_tables,
        constant_duties,
    ) = _all_duties(world, identifier)
    order = _variable_order(
        total, relation_duties, operation_duties, constant_duties
    )
    position_of = {variable: index for index, variable in enumerate(order)}

    relation_bucket: list[list[tuple[tuple[int, ...], bool, int]]] = [
        [] for _ in range(total)
    ]
    for duty in relation_duties:
        depth = max(position_of[variable] for variable in duty[0])
        relation_bucket[depth].append(duty)
    operation_bucket: list[list[tuple[tuple[int, ...], int, int]]] = [
        [] for _ in range(total)
    ]
    for duty in operation_duties:
        depth = max(
            position_of[variable] for variable in (*duty[0], duty[1])
        )
        operation_bucket[depth].append(duty)
    constant_bucket: list[list[int]] = [[] for _ in range(total)]
    for element in constant_duties:
        constant_bucket[position_of[element]].append(element)

    flat_relations: list[tuple[tuple[int, ...], bool, int]] = []
    flat_operations: list[tuple[tuple[int, ...], int, int]] = []
    flat_constants: list[int] = []
    slices: list[tuple[int, int, int, int, int, int]] = []
    for depth in range(total):
        start_relation = len(flat_relations)
        flat_relations.extend(relation_bucket[depth])
        start_operation = len(flat_operations)
        flat_operations.extend(operation_bucket[depth])
        start_constant = len(flat_constants)
        flat_constants.extend(constant_bucket[depth])
        slices.append(
            (
                start_relation,
                len(flat_relations),
                start_operation,
                len(flat_operations),
                start_constant,
                len(flat_constants),
            )
        )
    remaining = [0] * (total + 1)
    for depth in range(total - 1, -1, -1):
        remaining[depth] = remaining[depth + 1] + (
            len(relation_bucket[depth])
            + len(operation_bucket[depth])
            + len(constant_bucket[depth])
        )
    return Prepared(
        world=world,
        order=order,
        offsets=offsets,
        sizes=sizes,
        carrier_of=tuple(carrier_of),
        relation_duties=tuple(flat_relations),
        relation_membership=tuple(relation_membership),
        operation_duties=tuple(flat_operations),
        operation_tables=tuple(operation_tables),
        constant_duties=tuple(flat_constants),
        duty_slices=tuple(slices),
        remaining=tuple(remaining),
    )


def _check_depth(plan: Prepared, images: list[int], depth: int) -> bool:
    """Check exactly those obligations that become determined at this depth."""
    start_r, stop_r, start_o, stop_o, start_c, stop_c = plan.duty_slices[depth]
    for index in range(start_r, stop_r):
        variables, truth, slot = plan.relation_duties[index]
        moved = tuple(images[variable] for variable in variables)
        if (moved in plan.relation_membership[slot]) is not truth:
            return False
    for index in range(start_o, stop_o):
        variables, result, slot = plan.operation_duties[index]
        moved = tuple(images[variable] for variable in variables)
        if images[result] != plan.operation_tables[slot][moved]:
            return False
    for index in range(start_c, stop_c):
        element = plan.constant_duties[index]
        if images[element] != element:
            return False
    return True


def _free_completions(plan: Prepared, used: Sequence[set[int]]) -> int:
    """Count completions when no obligation remains: a product of factorials."""
    total = 1
    for position, size in enumerate(plan.sizes):
        total *= math.factorial(size - len(used[position]))
    return total


def _walk(
    plan: Prepared,
    forced: dict[int, int],
    collect: list[Morphism] | None,
    budget: int | None,
    stop_at_first: bool,
) -> int:
    """Exhaust the product of symmetric groups, pruning only on real violations.

    ``forced`` pre-assigns some variables, which is how an orbit query is asked.
    ``collect`` materializes the leaves when they are wanted. ``budget`` caps the
    number of leaves materialized. ``stop_at_first`` turns the walk into a
    satisfiability check. When no obligation is outstanding below the current
    depth the remaining variables are unconstrained, and the subtree size is
    counted in closed form rather than visited; that shortcut is exact and it is
    what makes a world with an erased primitive tractable.
    """
    total = plan.total
    images = [-1] * total
    used: list[set[int]] = [set() for _ in plan.sizes]
    found = 0

    def descend(depth: int) -> bool:
        nonlocal found
        if depth == total:
            if collect is not None:
                collect.append(_morphism_from_images(plan, images))
            found += 1
            return stop_at_first or (budget is not None and found >= budget)
        if plan.remaining[depth] == 0 and collect is None and not stop_at_first:
            found += _free_completions(plan, used)
            return budget is not None and found >= budget
        variable = plan.order[depth]
        position = plan.carrier_of[variable]
        offset = plan.offsets[position]
        if variable in forced:
            candidates = [forced[variable]]
        else:
            candidates = [
                offset + index
                for index in range(plan.sizes[position])
                if offset + index not in used[position]
            ]
        for candidate in candidates:
            if candidate in used[position]:
                continue
            images[variable] = candidate
            used[position].add(candidate)
            if _check_depth(plan, images, depth) and descend(depth + 1):
                used[position].discard(candidate)
                images[variable] = -1
                return True
            used[position].discard(candidate)
            images[variable] = -1
        return False

    descend(0)
    return found


def _morphism_from_images(plan: Prepared, images: Sequence[int]) -> Morphism:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    for position, carrier in enumerate(plan.world.carriers):
        offset = plan.offsets[position]
        blocks.append(
            (
                carrier.name,
                tuple(
                    carrier.elements[images[offset + index] - offset]
                    for index in range(plan.sizes[position])
                ),
            )
        )
    return Morphism(images=tuple(blocks))


def count_morphisms(plan: Prepared) -> int:
    """The exact order of the derived structure-preserving group."""
    return _walk(plan, {}, None, None, False)


def enumerate_morphisms(
    plan: Prepared, budget: int = ENUMERATION_BUDGET
) -> tuple[Morphism, ...] | None:
    """Materialize the derived group, or ``None`` when it exceeds the budget."""
    collected: list[Morphism] = []
    _walk(plan, {}, collected, budget + 1, False)
    if len(collected) > budget:
        return None
    return tuple(collected)


def exists_morphism_sending(plan: Prepared, source: Row, target: Row) -> bool:
    """Is there a structure-preserving map carrying one admitted input to another?

    This is the orbit question asked one pair at a time. It pre-assigns the
    components of the source to the components of the target and asks the same
    exhaustive search for a single witness, so it is exact even when the derived
    group is far too large to materialize.
    """
    world = plan.world
    forced: dict[int, int] = {}
    for carrier_name, from_element, to_element in zip(
        world.input_signature, source, target, strict=True
    ):
        position = world.carrier_names.index(carrier_name)
        offset = plan.offsets[position]
        from_variable = offset + world.carriers[position].elements.index(from_element)
        to_variable = offset + world.carriers[position].elements.index(to_element)
        if forced.setdefault(from_variable, to_variable) != to_variable:
            return False
    if len(set(forced.values())) != len(forced):
        return False
    return _walk(plan, forced, None, None, True) > 0


def derive_orbit_quotient(plan: Prepared) -> tuple[int, ...]:
    """Derive the input quotient by asking the orbit question, pair by pair.

    Cells are numbered by first appearance in declared input order, then renamed
    so that they are sorted by cardinality and then by least member. That naming
    rule is the same one 017.24 used, it is bookkeeping, and it carries no meaning.
    """
    inputs = plan.world.inputs
    assignment = [-1] * len(inputs)
    raw_cells: list[list[int]] = []
    for index, source in enumerate(inputs):
        if assignment[index] >= 0:
            continue
        cell = len(raw_cells)
        members = [index]
        assignment[index] = cell
        for other in range(index + 1, len(inputs)):
            if assignment[other] >= 0:
                continue
            if exists_morphism_sending(plan, source, inputs[other]):
                assignment[other] = cell
                members.append(other)
        raw_cells.append(members)
    ordering = sorted(
        range(len(raw_cells)),
        key=lambda cell: (len(raw_cells[cell]), raw_cells[cell][0]),
    )
    rename = {cell: position for position, cell in enumerate(ordering)}
    return tuple(rename[value] for value in assignment)


def partition_from_group(
    plan: Prepared, morphisms: Sequence[Morphism]
) -> tuple[int, ...]:
    """Close the admitted inputs under a supplied set of maps, by union-find.

    This is the second, independent route to the quotient. It is only available
    when the group was small enough to materialize, and when it is available its
    agreement with ``derive_orbit_quotient`` is checked.
    """
    world = plan.world
    inputs = world.inputs
    index_of = {row: position for position, row in enumerate(inputs)}
    parent = list(range(len(inputs)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for morphism in morphisms:
        mapping = morphism.as_mapping(world)
        for row in inputs:
            moved = tuple(
                mapping[carrier][element]
                for carrier, element in zip(world.input_signature, row, strict=True)
            )
            left, right = find(index_of[row]), find(index_of[moved])
            if left != right:
                parent[max(left, right)] = min(left, right)
    groups: dict[int, list[int]] = {}
    for node in range(len(inputs)):
        groups.setdefault(find(node), []).append(node)
    ordering = sorted(groups.values(), key=lambda members: (len(members), members[0]))
    assignment = [-1] * len(inputs)
    for cell, members in enumerate(ordering):
        for node in members:
            assignment[node] = cell
    return tuple(assignment)


@dataclass(frozen=True, slots=True)
class Compiled:
    """Everything the compiler derives from one declared world, and no more."""

    world: TypedWorld
    group_order: int
    morphisms: tuple[Morphism, ...] | None
    quotient: tuple[int, ...]
    seconds: float

    @property
    def cell_count(self) -> int:
        return len(set(self.quotient))

    @property
    def cell_sizes(self) -> list[int]:
        return _cell_sizes(self.quotient)


def compile_world(world: TypedWorld) -> Compiled:
    """Derive the group order and the input quotient from declared structure."""
    started = time.perf_counter()
    plan = prepare(world)
    order = count_morphisms(plan)
    morphisms = enumerate_morphisms(plan) if order <= ENUMERATION_BUDGET else None
    quotient = derive_orbit_quotient(plan)
    return Compiled(
        world=world,
        group_order=order,
        morphisms=morphisms,
        quotient=quotient,
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# 4. Source perturbations, as document transforms recompiled from scratch
# ---------------------------------------------------------------------------

# Declared before any perturbation is compiled. 017.26b section 6 asks for the
# direction of change, not the value, so directions are what is frozen here; the
# exact derived numbers are reported alongside and are reproduced by --check.
#
#   "grows"    |Aut| strictly increases and the cell count strictly decreases
#   "shrinks"  |Aut| strictly decreases and the cell count strictly increases,
#              and the new group is a subgroup of the old one
#   "differs"  the derived quotient is not the same partition
PERTURBATION_PREDICTIONS: Final = {
    "A": (
        ("erase:incident", "grows", "removing the only declared primitive leaves nothing to preserve, so every pair of carrier bijections qualifies and the admitted inputs collapse to a single cell"),
        ("scramble:incident", "differs", "a relation of the same cardinality that is not isomorphic to the declared one must produce a different quotient; if it produced the same one the compiler would be reading cardinality rather than structure"),
        ("break:Point", "shrinks", "distinguishing one element of a carrier can only remove structure-preserving maps, never add them, so the new group is a subgroup and the quotient refines"),
    ),
    "B": (
        ("erase:adjacent", "grows", "without the neighbour relation only the declared total map constrains the carriers, so every fibre-respecting pair qualifies and the quotient coarsens"),
        ("erase:district", "grows", "without the declared total map only the neighbour relation constrains the larger carrier and the smaller carrier is entirely free, so the quotient coarsens"),
        ("scramble:adjacent", "differs", "a neighbour relation of the same cardinality that is not isomorphic to the declared one must produce a different quotient"),
        ("scramble:district", "differs", "a total map with the same fibre-size multiset but a different assignment must produce a different quotient"),
        ("break:District", "shrinks", "distinguishing one element of the smaller carrier can only remove structure-preserving maps, so the new group is a subgroup and the quotient refines"),
    ),
}


def _primitive_block(document: dict[str, object], name: str) -> dict[str, object]:
    for block in document["primitives"]:  # type: ignore[union-attr]
        if block["name"] == name:
            return block  # type: ignore[return-value]
    raise KeyError(f"no declared primitive named {name!r}")


def erase_primitive(document: dict[str, object], name: str) -> dict[str, object]:
    """Remove one declared primitive. Nothing else about the world changes."""
    _primitive_block(document, name)
    changed = json.loads(json.dumps(document))
    changed["primitives"] = [
        block for block in changed["primitives"] if block["name"] != name
    ]
    return changed


def scramble_primitive(
    document: dict[str, object], name: str, seed: int
) -> dict[str, object]:
    """Replace one primitive's data, preserving every superficial count.

    A relation keeps its carriers and the number of rows; an operation keeps its
    carriers and the multiset of fibre sizes. What changes is which rows, or which
    element each argument maps to, so nothing a cardinality-reading compiler could
    notice has changed.
    """
    block = _primitive_block(document, name)
    changed = json.loads(json.dumps(document))
    target = _primitive_block(changed, name)
    rng = random.Random(seed)
    carriers = {
        str(item["name"]): [str(value) for value in item["elements"]]
        for item in document["sorts"]  # type: ignore[union-attr]
    }
    if block["kind"] == "relation":
        pools = [carriers[str(item)] for item in block["signature"]]
        space = [list(row) for row in itertools.product(*pools)]
        declared = [list(row) for row in block["extension"]]
        while True:
            drawn = sorted(rng.sample(space, len(declared)))
            if drawn != sorted(declared):
                break
        target["extension"] = drawn
    elif block["kind"] == "operation":
        pools = [carriers[str(item)] for item in block["domain"]]
        arguments = [list(row) for row in itertools.product(*pools)]
        declared_values = [str(entry["value"]) for entry in block["graph"]]
        while True:
            shuffled = list(declared_values)
            rng.shuffle(shuffled)
            if shuffled != declared_values:
                break
        target["graph"] = [
            {"arguments": row, "value": value}
            for row, value in zip(arguments, shuffled, strict=True)
        ]
    else:
        raise RuntimeError(f"cannot scramble a primitive of kind {block['kind']!r}")
    return changed


def break_symmetry(
    document: dict[str, object], carrier: str, element: str, name: str
) -> dict[str, object]:
    """Add a genuine source distinction: one declared distinguished element.

    This is an ordinary thing for a world description to contain -- a named
    reference element -- and it would belong in the specification whether or not
    anything was ever trained. It is added as a primitive of the declared
    ``constant`` kind, so the compiler reaches it through the same rule table.
    """
    changed = json.loads(json.dumps(document))
    changed["primitives"] = [
        *changed["primitives"],
        {
            "name": name,
            "kind": "constant",
            "sort": carrier,
            "value": element,
            "description": (
                "a declared distinguished element of the named carrier, of the "
                "sort an ordinary domain specification records as a reference "
                "point"
            ),
        },
    ]
    return changed


def build_perturbations(
    document: dict[str, object], fixture: str
) -> tuple[tuple[str, str, str, TypedWorld], ...]:
    """Realize every declared perturbation of one fixture, in the frozen order."""
    built: list[tuple[str, str, str, TypedWorld]] = []
    for offset, (name, direction, rationale) in enumerate(
        PERTURBATION_PREDICTIONS[fixture]
    ):
        kind, argument = name.split(":", 1)
        if kind == "erase":
            changed = erase_primitive(document, argument)
        elif kind == "scramble":
            changed = scramble_primitive(
                document, argument, SCRAMBLE_SEED + 1000 * offset
            )
        elif kind == "break":
            carriers = {
                str(item["name"]): [str(value) for value in item["elements"]]
                for item in document["sorts"]  # type: ignore[union-attr]
            }
            changed = break_symmetry(
                document, argument, carriers[argument][0], "reference"
            )
        else:
            raise RuntimeError(f"undeclared perturbation kind {kind!r}")
        built.append(
            (
                name,
                direction,
                rationale,
                world_from_document(changed, f"{fixture}:{name}"),
            )
        )
    return tuple(built)


# ---------------------------------------------------------------------------
# 5. The frozen bounded shortcut language L_k
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Term:
    """One typed expression over the components of an admitted input."""

    expression: str
    carrier: str
    depth: int
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Feature:
    """One L_k feature, tabulated over every admitted input."""

    expression: str
    kind: str
    values: tuple[object, ...]

    @property
    def partition(self) -> tuple[int, ...]:
        seen: dict[object, int] = {}
        return tuple(seen.setdefault(value, len(seen)) for value in self.values)


def build_terms(world: TypedWorld, depth: int) -> tuple[Term, ...]:
    """Generate every typed term over the input components, to a fixed depth."""
    inputs = world.inputs
    terms: list[Term] = [
        Term(
            expression=f"x{position}",
            carrier=carrier,
            depth=0,
            values=tuple(row[position] for row in inputs),
        )
        for position, carrier in enumerate(world.input_signature)
    ]
    for primitive in world.primitives:
        if primitive.kind == "constant":
            target = primitive.value_carrier
            if target is None or primitive.value is None:
                raise RuntimeError("a constant must declare a carrier and a value")
            terms.append(
                Term(
                    expression=primitive.name,
                    carrier=target,
                    depth=0,
                    values=tuple(primitive.value for _ in inputs),
                )
            )
    for level in range(depth):
        grown: list[Term] = []
        for primitive in world.primitives:
            if primitive.kind != "operation":
                continue
            target = primitive.value_carrier
            if target is None:
                raise RuntimeError("an operation must declare a codomain")
            table = dict(primitive.graph)
            pools = [
                [term for term in terms if term.carrier == carrier]
                for carrier in primitive.argument_carriers
            ]
            for chosen in itertools.product(*pools):
                if max(term.depth for term in chosen) != level:
                    continue
                expression = (
                    f"{primitive.name}("
                    + ",".join(term.expression for term in chosen)
                    + ")"
                )
                if any(term.expression == expression for term in terms):
                    continue
                grown.append(
                    Term(
                        expression=expression,
                        carrier=target,
                        depth=level + 1,
                        values=tuple(
                            table[
                                tuple(term.values[row] for term in chosen)
                            ]
                            for row in range(len(inputs))
                        ),
                    )
                )
        if not grown:
            break
        terms.extend(grown)
    return tuple(terms)


def build_shortcut_language(
    world: TypedWorld, depth: int = SHORTCUT_DEPTH
) -> tuple[tuple[Term, ...], tuple[Feature, ...]]:
    """Enumerate L_k exhaustively from the declared primitives.

    The grammar is exactly the 017.26b section 4 list, generated mechanically:
    every declared relation evaluated on every type-correct tuple of terms; every
    type-correct equality between distinct terms; a local count feature for every
    relation, every argument position, and every type-correct filling of the other
    positions; a fibre-size count for every operation and every term of its
    codomain; and the equality and order comparisons between count features.

    Two deliberate strengthenings over the letter of section 4. Raw count values
    are kept as features rather than only the equality tests between them, which
    can only make the language finer. And the joint partition of the whole
    language is reported alongside the finest single feature, which is the
    strongest statement the boolean closure of these atoms can support: any
    negation, conjunction, or disjunction of atoms induces a partition coarser
    than or equal to their joint, so enumerating the joint bounds the entire
    boolean closure without enumerating it.
    """
    inputs = world.inputs
    width = len(inputs)
    terms = build_terms(world, depth)
    features: list[Feature] = []
    counts: list[Feature] = []

    for primitive in world.primitives:
        if primitive.kind != "relation":
            continue
        pools = [
            [term for term in terms if term.carrier == carrier]
            for carrier in primitive.argument_carriers
        ]
        for chosen in itertools.product(*pools):
            features.append(
                Feature(
                    expression=(
                        f"{primitive.name}("
                        + ",".join(term.expression for term in chosen)
                        + ")"
                    ),
                    kind="relation_atom",
                    values=tuple(
                        tuple(term.values[row] for term in chosen)
                        in primitive.extension
                        for row in range(width)
                    ),
                )
            )
        for position, carrier in enumerate(primitive.argument_carriers):
            others = [
                [term for term in terms if term.carrier == name]
                for index, name in enumerate(primitive.argument_carriers)
                if index != position
            ]
            for chosen in itertools.product(*others):
                pool = world.carrier(carrier).elements
                labels = list(chosen)
                expression = (
                    f"|{{a : {primitive.name}("
                    + ",".join(
                        "a" if index == position else labels[
                            index if index < position else index - 1
                        ].expression
                        for index in range(len(primitive.argument_carriers))
                    )
                    + ")}|"
                )
                tabulated: list[int] = []
                for row in range(width):
                    fixed = [term.values[row] for term in chosen]
                    total = 0
                    for element in pool:
                        candidate = list(fixed)
                        candidate.insert(position, element)
                        if tuple(candidate) in primitive.extension:
                            total += 1
                    tabulated.append(total)
                counts.append(
                    Feature(
                        expression=expression, kind="count", values=tuple(tabulated)
                    )
                )

    for primitive in world.primitives:
        if primitive.kind != "operation":
            continue
        target = primitive.value_carrier
        if target is None:
            raise RuntimeError("an operation must declare a codomain")
        fibre: dict[str, int] = {}
        for _arguments, result in primitive.graph:
            fibre[result] = fibre.get(result, 0) + 1
        for term in terms:
            if term.carrier != target:
                continue
            counts.append(
                Feature(
                    expression=f"|{primitive.name}^-1({term.expression})|",
                    kind="count",
                    values=tuple(
                        fibre.get(term.values[row], 0) for row in range(width)
                    ),
                )
            )

    for left in range(len(terms)):
        for right in range(left + 1, len(terms)):
            if terms[left].carrier != terms[right].carrier:
                continue
            features.append(
                Feature(
                    expression=(
                        f"{terms[left].expression} = {terms[right].expression}"
                    ),
                    kind="equality_atom",
                    values=tuple(
                        terms[left].values[row] == terms[right].values[row]
                        for row in range(width)
                    ),
                )
            )

    comparisons: list[Feature] = []
    for left in range(len(counts)):
        for right in range(left + 1, len(counts)):
            comparisons.append(
                Feature(
                    expression=(
                        f"({counts[left].expression}) = "
                        f"({counts[right].expression})"
                    ),
                    kind="count_comparison",
                    values=tuple(
                        counts[left].values[row] == counts[right].values[row]
                        for row in range(width)
                    ),
                )
            )
            comparisons.append(
                Feature(
                    expression=(
                        f"({counts[left].expression}) < "
                        f"({counts[right].expression})"
                    ),
                    kind="count_comparison",
                    values=tuple(
                        counts[left].values[row] < counts[right].values[row]  # type: ignore[operator]
                        for row in range(width)
                    ),
                )
            )
    return terms, tuple([*features, *counts, *comparisons])


def joint_partition(features: Sequence[Feature], width: int) -> tuple[int, ...]:
    """The partition induced by reading every feature in the language at once."""
    signatures = [
        tuple(feature.values[row] for feature in features) for row in range(width)
    ]
    seen: dict[tuple[object, ...], int] = {}
    raw = [seen.setdefault(signature, len(seen)) for signature in signatures]
    groups: dict[int, list[int]] = {}
    for row, cell in enumerate(raw):
        groups.setdefault(cell, []).append(row)
    ordering = sorted(groups.values(), key=lambda members: (len(members), members[0]))
    assignment = [-1] * width
    for cell, members in enumerate(ordering):
        for row in members:
            assignment[row] = cell
    return tuple(assignment)


def shortcut_audit(
    world: TypedWorld, quotient: Sequence[int], depth: int = SHORTCUT_DEPTH
) -> dict[str, object]:
    """Run the frozen L_k audit against a derived quotient."""
    terms, features = build_shortcut_language(world, depth)
    width = len(world.inputs)
    partitions = {feature.expression: feature.partition for feature in features}
    distinct = {tuple(value) for value in partitions.values()}
    recovering = sorted(
        expression
        for expression, partition in partitions.items()
        if _refines(partition, quotient)
    )
    # Sorted before the maximum is taken, so that a tie is broken by the
    # partition's own content rather than by set iteration order.
    finest = max(
        sorted(distinct),
        key=lambda partition: (
            len(set(partition)),
            tuple(_cell_sizes(partition)),
        ),
    )
    joint = joint_partition(features, width)
    return {
        "depth_k": depth,
        "grammar": (
            "typed terms over the input components and the declared constants, "
            "closed under the declared operations to depth k; then every "
            "declared relation on every type-correct term tuple, every "
            "type-correct equality between distinct terms, a local count "
            "feature for every relation, argument position, and type-correct "
            "filling of the other positions, a fibre-size count for every "
            "operation and every term of its codomain, and the equality and "
            "order comparisons between count features"
        ),
        "term_count": len(terms),
        "terms": [
            {"expression": term.expression, "carrier": term.carrier, "depth": term.depth}
            for term in terms
        ],
        "language_size": len(features),
        "features_by_kind": {
            kind: sum(1 for feature in features if feature.kind == kind)
            for kind in sorted({feature.kind for feature in features})
        },
        "distinct_induced_partitions": len(distinct),
        "finest_single_feature_cell_count": len(set(finest)),
        "finest_single_feature_cell_sizes": _cell_sizes(finest),
        "joint_partition_cell_count": len(set(joint)),
        "joint_partition_cell_sizes": _cell_sizes(joint),
        "features_that_recover_the_quotient": recovering,
        "any_single_feature_recovers_the_quotient": bool(recovering),
        "the_whole_language_read_jointly_recovers_the_quotient": _refines(
            joint, quotient
        ),
        "the_quotient_determines_every_feature": all(
            _refines(quotient, partition) for partition in partitions.values()
        ),
        "why_coarser_is_the_only_possibility": (
            "every feature of L_k is a term over the declared primitives, so "
            "every feature is invariant under every structure-preserving map, so "
            "its partition is coarser than or equal to the orbit partition. "
            "Recovering q_Sigma therefore means equalling it, and failing to "
            "recover it means being strictly coarser."
        ),
        "bound_on_this_claim": (
            "failure here establishes only that q_Sigma is not recoverable "
            "inside the preregistered source-local language L_k at depth k. It "
            "does not establish that no cheaper derivation exists, and the "
            "disclosed non-local adversary below is reported precisely so that "
            "the bound is not mistaken for exhaustiveness."
        ),
    }


# ---------------------------------------------------------------------------
# 6. The disclosed non-local adversary, which L_k does not contain
# ---------------------------------------------------------------------------


def _pooled_terms(world: TypedWorld) -> tuple[tuple[str, str, int], ...]:
    """Typed slots available to a pair of admitted inputs, tagged by provenance."""
    slots: list[tuple[str, str, int]] = []
    for side in ("L", "R"):
        for position, carrier in enumerate(world.input_signature):
            slots.append((f"{side}{position}", carrier, position))
    return tuple(slots)


def pair_refinement(
    world: TypedWorld, cap: int = REFINEMENT_ITERATION_CAP
) -> tuple[tuple[int, ...], int]:
    """Iterated pair refinement over the admitted inputs, Weisfeiler-Leman style.

    Colours start from the pattern of declared facts an input's own components
    satisfy, and are refined by the multiset over every other admitted input of
    the pattern of declared facts the two inputs jointly satisfy together with
    that input's current colour. This is a global fixpoint computation over the
    whole admitted-input set. It is not a member of L_k, which is source-local and
    depth-bounded, and 017.26b section 8 forbids condition R from using it or
    anything equivalent to it. It is computed and banked here so that the bound on
    the L_k claim is a measured fact rather than a caveat.
    """
    inputs = world.inputs
    width = len(inputs)
    signature = world.input_signature
    operations = [
        primitive for primitive in world.primitives if primitive.kind == "operation"
    ]
    relations = [
        primitive for primitive in world.primitives if primitive.kind == "relation"
    ]
    constants = [
        primitive for primitive in world.primitives if primitive.kind == "constant"
    ]

    def slots(left: int, right: int) -> list[tuple[str, str]]:
        pooled: list[tuple[str, str]] = []
        for row in (left, right):
            for position, carrier in enumerate(signature):
                pooled.append((carrier, inputs[row][position]))
        for primitive in constants:
            target = primitive.value_carrier
            if target is None or primitive.value is None:
                raise RuntimeError("a constant must declare a carrier and a value")
            pooled.append((target, primitive.value))
        grown: list[tuple[str, str]] = []
        for primitive in operations:
            target = primitive.value_carrier
            if target is None:
                raise RuntimeError("an operation must declare a codomain")
            table = dict(primitive.graph)
            pools = [
                [item for item in pooled if item[0] == carrier]
                for carrier in primitive.argument_carriers
            ]
            for chosen in itertools.product(*pools):
                grown.append(
                    (target, table[tuple(item[1] for item in chosen)])
                )
        return [*pooled, *grown]

    def pattern(left: int, right: int) -> tuple[object, ...]:
        pooled = slots(left, right)
        facts: list[object] = []
        for primitive in relations:
            pools = [
                [item for item in pooled if item[0] == carrier]
                for carrier in primitive.argument_carriers
            ]
            for chosen in itertools.product(*pools):
                facts.append(
                    tuple(item[1] for item in chosen) in primitive.extension
                )
        for first in range(len(pooled)):
            for second in range(first + 1, len(pooled)):
                if pooled[first][0] != pooled[second][0]:
                    continue
                facts.append(pooled[first][1] == pooled[second][1])
        return tuple(facts)

    patterns = [
        [str(pattern(left, right)) for right in range(width)] for left in range(width)
    ]
    normalized: dict[str, int] = {}
    colour = [
        normalized.setdefault(patterns[row][row], len(normalized))
        for row in range(width)
    ]
    iterations = 0
    for _step in range(cap):
        refined = [
            (
                colour[left],
                tuple(
                    sorted(
                        (patterns[left][right], colour[right])
                        for right in range(width)
                    )
                ),
            )
            for left in range(width)
        ]
        lookup: dict[object, int] = {}
        candidate = [lookup.setdefault(value, len(lookup)) for value in refined]
        iterations += 1
        if len(set(candidate)) == len(set(colour)):
            colour = candidate
            break
        colour = candidate
    groups: dict[int, list[int]] = {}
    for row, cell in enumerate(colour):
        groups.setdefault(cell, []).append(row)
    ordering = sorted(groups.values(), key=lambda members: (len(members), members[0]))
    assignment = [-1] * width
    for cell, members in enumerate(ordering):
        for row in members:
            assignment[row] = cell
    return tuple(assignment), iterations


def non_local_adversary_audit(
    world: TypedWorld, quotient: Sequence[int]
) -> dict[str, object]:
    partition, iterations = pair_refinement(world)
    return {
        "construction": (
            "iterated pair refinement over the admitted inputs: colours start "
            "from the declared facts an input's own components satisfy, and are "
            "refined by the multiset over every other admitted input of the "
            "declared facts the pair jointly satisfies together with that "
            "input's current colour"
        ),
        "why_it_is_outside_L_k": (
            "it is a global fixpoint over the whole admitted-input set rather "
            "than a source-local bounded-depth feature of a single input, so it "
            "is not generated by the L_k grammar. 017.26b section 8 forbids "
            "condition R from computing a global structural quotient or "
            "anything equivalent to one, which is exactly what this is."
        ),
        "iterations_to_fixpoint": iterations,
        "cell_count": len(set(partition)),
        "cell_sizes": _cell_sizes(partition),
        "recovers_the_derived_quotient": _refines(partition, quotient)
        and _refines(quotient, partition),
        "is_coarser_than_or_equal_to_the_derived_quotient": _refines(
            quotient, partition
        ),
        "reading": (
            "if this recovers q_Sigma it does not make the derivation cheaper "
            "in kind, because it is itself a deterministic pre-label structural "
            "derivation and not a statistic a learner reads off an input. It "
            "makes the derivation cheaper in cost, which is a tractability "
            "finding and is reported as one."
        ),
    }


# ---------------------------------------------------------------------------
# 7. Audits: is the compiler really generic, and really label-free?
# ---------------------------------------------------------------------------

COSET_WITNESS_CAP: Final = 10_000
# Prefix sizes for the unpruned brute-force comparison. These are witness
# parameters, not compiler inputs: the reduced world must keep every declared
# operation total, and its full product of symmetric groups must be small enough
# to enumerate without any pruning at all.
REDUCTION_SIZES: Final = {
    "A": {"Point": 4, "Line": 4},
    "B": {"Site": 6, "District": 2},
}


def compiler_source() -> str:
    """The concatenated source of every function on the compiler path."""
    module = sys.modules[__name__]
    return "\n".join(
        inspect.getsource(getattr(module, name)) for name in COMPILER_PATH
    )


def declared_identifiers(worlds: Sequence[TypedWorld]) -> tuple[str, ...]:
    names: set[str] = set()
    for world in worlds:
        for carrier in world.carriers:
            names.add(carrier.name)
            names.update(carrier.elements)
        for primitive in world.primitives:
            names.add(primitive.name)
    return tuple(sorted(names))


def dispatch_audit(worlds: Sequence[TypedWorld]) -> dict[str, object]:
    """Show mechanically that the compiler branches on kind and on nothing else.

    Three independent things are asserted. The compiler path mentions no declared
    identifier of any fixture, so it cannot recognize one. The compiler path
    mentions no expectation vocabulary and no label vocabulary. And the set of
    values the preservation-rule table is keyed by is exactly the set of primitive
    kinds the fixtures and their perturbations declare, so no kind reaches the
    compiler through a path the table does not cover.
    """
    source = compiler_source()
    lowered = source.lower()
    tokens = set(re.findall(r"[a-z_0-9]+", lowered))
    identifiers = declared_identifiers(worlds)
    offending_identifiers = sorted(
        name for name in identifiers if name in source
    )
    offending_vocabulary = sorted(
        [word for word in FORBIDDEN_COMPILER_TOKENS if word in tokens]
        + [word for word in FORBIDDEN_COMPILER_SUBSTRINGS if word in lowered]
    )
    declared_kinds = sorted(
        {primitive.kind for world in worlds for primitive in world.primitives}
    )
    return {
        "compiler_path": list(COMPILER_PATH),
        "compiler_path_sha256": _sha256(source.encode()),
        "declared_identifiers_checked": len(identifiers),
        "declared_identifiers_appearing_in_the_compiler": offending_identifiers,
        "no_declared_identifier_appears_in_the_compiler": not offending_identifiers,
        "forbidden_tokens_checked": list(FORBIDDEN_COMPILER_TOKENS),
        "forbidden_substrings_checked": list(FORBIDDEN_COMPILER_SUBSTRINGS),
        "forbidden_vocabulary_appearing_in_the_compiler": offending_vocabulary,
        "no_fixture_or_expectation_or_label_vocabulary_appears": (
            not offending_vocabulary
        ),
        "preservation_rule_keys": sorted(PRESERVATION_RULES),
        "primitive_kinds_declared_across_every_world": declared_kinds,
        "every_declared_kind_has_a_generated_rule": all(
            kind in PRESERVATION_RULES for kind in declared_kinds
        ),
        "preservation_rules": dict(PRESERVATION_RULES),
    }


def relabel_document(document: dict[str, object]) -> dict[str, object]:
    """Rename every carrier, element, and primitive to a meaningless code."""
    carrier_names = [str(block["name"]) for block in document["sorts"]]  # type: ignore[union-attr]
    carrier_map = {
        name: f"c{index}" for index, name in enumerate(sorted(carrier_names))
    }
    element_map: dict[tuple[str, str], str] = {}
    for block in document["sorts"]:  # type: ignore[union-attr]
        name = str(block["name"])
        for index, element in enumerate(block["elements"]):
            element_map[(name, str(element))] = f"{carrier_map[name]}_{index}"
    primitive_map = {
        str(block["name"]): f"p{index}"
        for index, block in enumerate(document["primitives"])  # type: ignore[union-attr]
    }
    changed: dict[str, object] = {
        "sorts": [
            {
                "name": carrier_map[str(block["name"])],
                "elements": [
                    element_map[(str(block["name"]), str(element))]
                    for element in block["elements"]
                ],
            }
            for block in document["sorts"]  # type: ignore[union-attr]
        ],
        "admitted_inputs": {
            "product": [
                carrier_map[str(item)]
                for item in document["admitted_inputs"]["product"]  # type: ignore[index]
            ]
        },
    }
    primitives: list[dict[str, object]] = []
    for block in document["primitives"]:  # type: ignore[union-attr]
        kind = str(block["kind"])
        renamed: dict[str, object] = {
            "name": primitive_map[str(block["name"])],
            "kind": kind,
        }
        if kind == "relation":
            signature = [str(item) for item in block["signature"]]
            renamed["signature"] = [carrier_map[item] for item in signature]
            renamed["extension"] = [
                [
                    element_map[(carrier, str(element))]
                    for carrier, element in zip(signature, row, strict=True)
                ]
                for row in block["extension"]
            ]
        elif kind == "operation":
            domain = [str(item) for item in block["domain"]]
            codomain = str(block["codomain"])
            renamed["domain"] = [carrier_map[item] for item in domain]
            renamed["codomain"] = carrier_map[codomain]
            renamed["graph"] = [
                {
                    "arguments": [
                        element_map[(carrier, str(element))]
                        for carrier, element in zip(
                            domain, entry["arguments"], strict=True
                        )
                    ],
                    "value": element_map[(codomain, str(entry["value"]))],
                }
                for entry in block["graph"]
            ]
        elif kind == "constant":
            carrier = str(block["sort"])
            renamed["sort"] = carrier_map[carrier]
            renamed["value"] = element_map[(carrier, str(block["value"]))]
        else:
            raise RuntimeError(f"undeclared primitive kind {kind!r}")
        primitives.append(renamed)
    changed["primitives"] = primitives
    return changed


def strip_to_structure(document: dict[str, object]) -> dict[str, object]:
    """Delete every key the compiler is not permitted to read, and every prose."""
    kept: dict[str, object] = {}
    for key in STRUCTURAL_KEYS:
        kept[key] = json.loads(json.dumps(document[key]))
    for block in kept["sorts"]:  # type: ignore[union-attr]
        block.pop("description", None)
    for block in kept["primitives"]:  # type: ignore[union-attr]
        block.pop("description", None)
    kept["admitted_inputs"] = {
        "product": kept["admitted_inputs"]["product"]  # type: ignore[index]
    }
    return kept


def independence_audit(
    document: dict[str, object], compiled: Compiled
) -> dict[str, object]:
    """Behavioural proof that the compiler reads structure and nothing else.

    Relabelling renames every carrier, element, and primitive; key stripping
    deletes every document key outside the structural subtree and every prose
    description. If either changed the derived group order or the derived
    quotient, the compiler would be consulting something other than the declared
    structure.
    """
    relabelled = compile_world(
        world_from_document(relabel_document(document), "relabelled")
    )
    stripped = compile_world(
        world_from_document(strip_to_structure(document), "stripped")
    )
    return {
        "relabelling": {
            "what_was_renamed": (
                "every carrier, every element of every carrier, and every "
                "primitive, to meaningless codes"
            ),
            "group_order": relabelled.group_order,
            "quotient_is_identical": relabelled.quotient == compiled.quotient,
            "group_order_is_identical": (
                relabelled.group_order == compiled.group_order
            ),
        },
        "key_stripping": {
            "keys_kept": list(STRUCTURAL_KEYS),
            "keys_deleted": sorted(
                key for key in document if key not in STRUCTURAL_KEYS
            ),
            "prose_descriptions_deleted": True,
            "group_order": stripped.group_order,
            "quotient_is_identical": stripped.quotient == compiled.quotient,
            "group_order_is_identical": (
                stripped.group_order == compiled.group_order
            ),
        },
        "reading": (
            "the derived group and quotient survive renaming every declared name "
            "and deleting every non-structural key, so the derivation is a "
            "function of the declared structure alone"
        ),
    }


def restrict_world(world: TypedWorld, sizes: dict[str, int]) -> TypedWorld:
    """Cut every carrier to a declared prefix, inducing the substructure.

    Used only by the unpruned brute-force witness. Every declared operation must
    remain total on the reduced world and every declared constant must survive,
    which is checked rather than assumed.
    """
    kept = {
        carrier.name: carrier.elements[: sizes[carrier.name]]
        for carrier in world.carriers
    }
    carriers = tuple(
        Carrier(name=carrier.name, elements=kept[carrier.name])
        for carrier in world.carriers
    )
    primitives: list[Primitive] = []
    for primitive in world.primitives:
        if primitive.kind == "relation":
            primitives.append(
                Primitive(
                    name=primitive.name,
                    kind=primitive.kind,
                    argument_carriers=primitive.argument_carriers,
                    value_carrier=None,
                    extension=frozenset(
                        row
                        for row in primitive.extension
                        if all(
                            element in kept[carrier]
                            for carrier, element in zip(
                                primitive.argument_carriers, row, strict=True
                            )
                        )
                    ),
                    graph=(),
                    value=None,
                )
            )
        elif primitive.kind == "operation":
            target = primitive.value_carrier
            if target is None:
                raise RuntimeError("an operation must declare a codomain")
            graph = tuple(
                (arguments, result)
                for arguments, result in primitive.graph
                if all(
                    element in kept[carrier]
                    for carrier, element in zip(
                        primitive.argument_carriers, arguments, strict=True
                    )
                )
            )
            expected = 1
            for carrier in primitive.argument_carriers:
                expected *= len(kept[carrier])
            if len(graph) != expected or any(
                result not in kept[target] for _arguments, result in graph
            ):
                raise RuntimeError(
                    f"the reduced world does not keep {primitive.name!r} total"
                )
            primitives.append(
                Primitive(
                    name=primitive.name,
                    kind=primitive.kind,
                    argument_carriers=primitive.argument_carriers,
                    value_carrier=target,
                    extension=frozenset(),
                    graph=graph,
                    value=None,
                )
            )
        elif primitive.kind == "constant":
            target = primitive.value_carrier
            if target is None or primitive.value is None:
                raise RuntimeError("a constant must declare a carrier and a value")
            if primitive.value not in kept[target]:
                raise RuntimeError(
                    f"the reduced world drops the constant {primitive.name!r}"
                )
            primitives.append(primitive)
        else:
            raise RuntimeError(f"undeclared primitive kind {primitive.kind!r}")
    return TypedWorld(
        identifier=f"{world.identifier}:reduced",
        carriers=carriers,
        primitives=tuple(primitives),
        input_signature=world.input_signature,
    )


def _all_candidates(world: TypedWorld) -> list[Morphism]:
    """Every element of the full product of symmetric groups, with no pruning."""
    blocks = [
        [
            (carrier.name, permutation)
            for permutation in itertools.permutations(carrier.elements)
        ]
        for carrier in world.carriers
    ]
    return [Morphism(images=tuple(choice)) for choice in itertools.product(*blocks)]


def brute_force_witness(world: TypedWorld, sizes: dict[str, int]) -> dict[str, object]:
    """Compare the pruned search against unpruned brute force on a reduced world.

    Pruning is the only thing in the search that could lose a solution, so it is
    the only thing that needs an independent check. On the reduced world every
    element of the full product of symmetric groups is generated and tested by the
    signature-derived checker, with no search and no pruning at all.
    """
    reduced = restrict_world(world, sizes)
    candidates = _all_candidates(reduced)
    accepted = [
        morphism
        for morphism in candidates
        if preserves_every_primitive(reduced, morphism)
    ]
    compiled = compile_world(reduced)
    searched = enumerate_morphisms(prepare(reduced))
    return {
        "reduced_carrier_sizes": dict(sizes),
        "full_product_of_symmetric_groups": len(candidates),
        "candidates_accepted_by_unpruned_brute_force": len(accepted),
        "group_order_from_the_pruned_search": compiled.group_order,
        "orders_agree": len(accepted) == compiled.group_order,
        "element_sets_agree": searched is not None
        and set(searched) == set(accepted),
        "reading": (
            "the pruned search returns exactly the elements an exhaustive "
            "unpruned enumeration accepts, so its pruning discards only genuine "
            "violations"
        ),
    }


def coset_witness(plan: Prepared) -> dict[str, object]:
    """Re-derive the group order by a different exhaustive decomposition.

    Every permutation of one chosen carrier is enumerated outright and forced, and
    the remaining carriers are completed by the search. Summing the completions
    covers the same product of symmetric groups along a different axis, so the two
    totals agreeing is a check on the search's bookkeeping rather than a
    restatement of it.
    """
    world = plan.world
    choices = [
        (position, math.factorial(len(carrier.elements)))
        for position, carrier in enumerate(world.carriers)
    ]
    affordable = [
        (size, position) for position, size in choices if size <= COSET_WITNESS_CAP
    ]
    if not affordable:
        return {"available": False, "why": "every carrier is too large to enumerate"}
    _size, chosen = max(affordable)
    carrier = world.carriers[chosen]
    offset = plan.offsets[chosen]
    total = 0
    for permutation in itertools.permutations(range(len(carrier.elements))):
        forced = {
            offset + index: offset + image
            for index, image in enumerate(permutation)
        }
        total += _walk(plan, forced, None, None, False)
    return {
        "available": True,
        "carrier_enumerated_outright": carrier.name,
        "permutations_enumerated": math.factorial(len(carrier.elements)),
        "order_summed_over_cosets": total,
        "agrees_with_the_primary_search": total == count_morphisms(plan),
    }


def rejection_witness(
    world: TypedWorld, compiled: Compiled, samples: int = MAXIMALITY_SAMPLES
) -> dict[str, object]:
    """Search the full product of symmetric groups at random for a missed element.

    Every sample is either inside the derived group or is shown to violate a
    declared primitive. A sample that did neither would be a witness that the
    derived group is not maximal, and there is no such sample.
    """
    rng = random.Random(MAXIMALITY_SAMPLE_SEED)
    inside = 0
    violating = 0
    unexplained = 0
    known = set(compiled.morphisms or ())
    for _draw in range(samples):
        blocks: list[tuple[str, tuple[str, ...]]] = []
        for carrier in world.carriers:
            shuffled = list(carrier.elements)
            rng.shuffle(shuffled)
            blocks.append((carrier.name, tuple(shuffled)))
        candidate = Morphism(images=tuple(blocks))
        preserves = preserves_every_primitive(world, candidate)
        if preserves:
            inside += 1
            if compiled.morphisms is not None and candidate not in known:
                unexplained += 1
        else:
            violating += 1
    neighbours = 0
    neighbour_escapes = 0
    if compiled.morphisms is not None:
        for morphism in compiled.morphisms:
            for position, carrier in enumerate(world.carriers):
                width = len(carrier.elements)
                for left in range(width):
                    for right in range(left + 1, width):
                        blocks = list(morphism.images)
                        name, images = blocks[position]
                        swapped = list(images)
                        swapped[left], swapped[right] = (
                            swapped[right],
                            swapped[left],
                        )
                        blocks[position] = (name, tuple(swapped))
                        candidate = Morphism(images=tuple(blocks))
                        neighbours += 1
                        if preserves_every_primitive(world, candidate):
                            neighbour_escapes += 1
    return {
        "uniform_sampling": {
            "samples": samples,
            "samples_preserving_every_primitive": inside,
            "samples_violating_some_primitive": violating,
            "samples_preserving_everything_but_absent_from_the_group": unexplained,
            "honest_reading": (
                "with a group this small inside a product of symmetric groups "
                "this large, uniform sampling is expected to find nothing at "
                "all, and it found nothing at all. On its own this is a weak "
                "witness and is reported as one; the exhaustive neighbourhood "
                "scan below is the one that carries weight."
            ),
        },
        "exhaustive_neighbourhood_scan": {
            "construction": (
                "compose every derived element with every transposition of every "
                "carrier, which exhausts the elements lying at Cayley distance "
                "one from the derived group in the product of symmetric groups"
            ),
            "neighbours_tested": neighbours,
            "neighbours_preserving_every_primitive": neighbour_escapes,
            "the_derived_group_has_no_structure_preserving_neighbour": (
                neighbour_escapes == 0
            ),
        },
        "no_larger_group_was_found": unexplained == 0 and neighbour_escapes == 0,
        "group_was_materialized_for_this_check": compiled.morphisms is not None,
    }


def maximality_witness(
    world: TypedWorld, compiled: Compiled, sizes: dict[str, int]
) -> dict[str, object]:
    """The layered argument that the derived group is exactly Aut(A_Sigma)."""
    plan = prepare(world)
    independent = (
        all(
            preserves_every_primitive(world, morphism)
            for morphism in compiled.morphisms
        )
        if compiled.morphisms is not None
        else None
    )
    return {
        "soundness_of_the_search": (
            "the search iterates every injective assignment of every carrier and "
            "rejects a branch only when a declared obligation is already "
            "violated, so it cannot discard a structure-preserving map"
        ),
        "every_derived_element_preserves_every_primitive": independent,
        "elements_rechecked_independently": (
            len(compiled.morphisms) if compiled.morphisms is not None else 0
        ),
        "unpruned_brute_force_on_a_reduced_world": brute_force_witness(world, sizes),
        "second_exhaustive_decomposition": coset_witness(plan),
        "random_search_for_a_missed_element": rejection_witness(world, compiled),
        "quotient_derived_twice_by_different_routes": {
            "route_one": (
                "ask, for each ordered pair of admitted inputs, whether a "
                "structure-preserving map carries one to the other; this never "
                "materializes the group and so remains exact when the group is "
                "far too large to enumerate"
            ),
            "route_two": (
                "close the admitted inputs under every materialized element by "
                "union-find; available only when the group is small enough"
            ),
            "routes_agree": (
                partition_from_group(plan, compiled.morphisms) == compiled.quotient
                if compiled.morphisms is not None
                else None
            ),
        },
    }


def forbidden_metadata_audit(document: dict[str, object]) -> dict[str, object]:
    """Mechanically check the 017.26 section 2 source fence against a document."""
    hits: list[str] = []
    floats: list[str] = []

    def walk(node: object, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                for word in FORBIDDEN_STRUCTURAL_VOCABULARY:
                    if word in lowered:
                        hits.append(f"{trail}/{key} (key)")
                walk(value, f"{trail}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        elif isinstance(node, str):
            lowered = node.lower()
            for word in FORBIDDEN_STRUCTURAL_VOCABULARY:
                if word in lowered:
                    hits.append(f"{trail} (value {node!r})")
        elif isinstance(node, float):
            floats.append(trail)

    for key in STRUCTURAL_KEYS:
        walk(document[key], key)
    unexpected = sorted(
        key for key in document if key not in DOCUMENT_KEY_WHITELIST
    )
    denied = {str(item).lower() for item in document["not_declared_here"]}  # type: ignore[union-attr]
    return {
        "structural_keys_audited": list(STRUCTURAL_KEYS),
        "forbidden_vocabulary_checked": list(FORBIDDEN_STRUCTURAL_VOCABULARY),
        "forbidden_vocabulary_found_in_the_structural_subtree": sorted(hits),
        "structural_subtree_is_clean": not hits,
        "floats_found_in_the_structural_subtree": sorted(floats),
        "no_numeric_probability_can_hide_in_the_structure": not floats,
        "unexpected_top_level_keys": unexpected,
        "every_top_level_key_is_whitelisted": not unexpected,
        "denials_declared": len(denied),
        "denial_covers_symmetry_and_equivalence": any(
            "automorphism" in item for item in denied
        )
        and any("equivalence relation" in item for item in denied)
        and any("orbit" in item for item in denied),
        "the_binding_form_of_this_audit": (
            "the vocabulary scan is necessary but not sufficient on its own. The "
            "binding form is the key-stripping test in the independence audit: "
            "the derived group and quotient are unchanged when every key outside "
            "sorts, primitives, and admitted_inputs is deleted, so no metadata "
            "anywhere else in the document can be load-bearing whatever it says."
        ),
    }


def declaration_ledger(document: dict[str, object]) -> list[dict[str, object]]:
    """The 017.26 section 4 natural-language test, one row per declared field."""
    rows: list[dict[str, object]] = []
    for block in document["sorts"]:  # type: ignore[union-attr]
        rows.append(
            {
                "field": f"sorts/{block['name']}",
                "declares": (
                    f"a carrier named {block['name']} with "
                    f"{len(block['elements'])} declared elements"
                ),
                "would_belong_with_no_model_ever_trained": True,
                "why": str(block.get("description", "")),
            }
        )
    for block in document["primitives"]:  # type: ignore[union-attr]
        rows.append(
            {
                "field": f"primitives/{block['name']}",
                "declares": f"a primitive of kind {block['kind']}",
                "would_belong_with_no_model_ever_trained": True,
                "why": str(block.get("description", "")),
            }
        )
    rows.append(
        {
            "field": "admitted_inputs",
            "declares": (
                "the product type an admitted input inhabits: "
                + " x ".join(
                    str(item)
                    for item in document["admitted_inputs"]["product"]  # type: ignore[index]
                )
            ),
            "would_belong_with_no_model_ever_trained": True,
            "why": (
                "the type of question the world admits is part of specifying the "
                "world's interface, and it names no answer and no frequency"
            ),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# 8. Joint load-bearing, the phase fence, and the fixtures
# ---------------------------------------------------------------------------


def joint_load_bearing(
    world: TypedWorld,
    compiled: Compiled,
    perturbations: Sequence[tuple[str, str, str, TypedWorld]],
) -> dict[str, object]:
    """Execute the 017.26b section 5 test, primitive by primitive.

    A primitive is load-bearing when deleting it strictly enlarges the derived
    group or strictly coarsens the derived quotient. The primitives are jointly
    load-bearing when that holds for every one of them and there is more than one,
    so the derived invariance cannot be obtained by keeping one primitive and
    treating the others as decoration.
    """
    erased = {
        name.split(":", 1)[1]: entry
        for entry in perturbations
        for name in (entry[0],)
        if name.startswith("erase:")
    }
    rows: list[dict[str, object]] = []
    for primitive in world.primitives:
        entry = erased.get(primitive.name)
        if entry is None:
            rows.append(
                {
                    "primitive": primitive.name,
                    "kind": primitive.kind,
                    "erasure_was_compiled": False,
                }
            )
            continue
        without = compile_world(entry[3])
        rows.append(
            {
                "primitive": primitive.name,
                "kind": primitive.kind,
                "erasure_was_compiled": True,
                "group_order_with": compiled.group_order,
                "group_order_without": without.group_order,
                "group_strictly_enlarges_when_erased": (
                    without.group_order > compiled.group_order
                ),
                "cell_count_with": compiled.cell_count,
                "cell_count_without": without.cell_count,
                "quotient_strictly_coarsens_when_erased": (
                    without.cell_count < compiled.cell_count
                ),
                "is_load_bearing": (
                    without.group_order > compiled.group_order
                    or without.cell_count < compiled.cell_count
                ),
            }
        )
    compiled_rows = [row for row in rows if row.get("erasure_was_compiled")]
    return {
        "test": (
            "a primitive is load-bearing when deleting it strictly enlarges the "
            "derived group or strictly coarsens the derived quotient; the "
            "primitives are jointly load-bearing when that holds for every one "
            "of them and there is more than one"
        ),
        "primitive_count": len(world.primitives),
        "per_primitive": rows,
        "every_primitive_is_load_bearing": bool(compiled_rows)
        and all(bool(row["is_load_bearing"]) for row in compiled_rows),
        "primitives_are_jointly_load_bearing": len(world.primitives) > 1
        and bool(compiled_rows)
        and all(bool(row["is_load_bearing"]) for row in compiled_rows),
    }


def phase_fence_audit() -> dict[str, object]:
    """Show that nothing statistical can be in scope, from the import graph.

    017.26b section 7 requires a Phase S artifact created before target
    probabilities and label seeds exist. The durable form of that requirement is
    structural: this module does not import the module that contains the label
    generator, so no label could have been consulted even by accident.
    """
    source = Path(__file__).read_bytes().decode()
    imports = sorted(
        {
            line.split()[1].split(".")[0]
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and len(line.split()) > 1
        }
    )
    banned = ("numpy", "training_economy", "quotient_refinement", "phase_t")
    # Assembled at run time rather than written out, because a file cannot
    # search itself for a token it contains as a literal.
    needle = "the" + "ta" + "_" + "star"
    return {
        "module_sha256": _sha256(source.encode()),
        "top_level_imports": imports,
        "imports_no_statistical_module": [
            name for name in banned if name in imports
        ]
        == [],
        "why_the_import_graph_is_the_binding_form": (
            "the byte-pinned 017.24 implementation contains "
            "generate_observations. This module does not import it, so the claim "
            "that no label was in scope during derivation is a property of the "
            "import graph rather than a promise about discipline. The Phase T "
            "module is a separate file which pins this one."
        ),
        "token_searched_for": needle,
        "no_target_parameter_vector_appears_in_this_module": (
            source.count(needle) == 0
        ),
        "what_exists_at_this_freeze": [
            "the declared world documents",
            "the generic schema-semantic compiler",
            "the derived groups and input quotients",
            "the frozen bounded shortcut language L_k and its audit",
            "the disclosed non-local adversary and its result",
            "the declared source perturbations and their frozen directions",
            "the compile costs",
        ],
        "what_does_not_exist_yet": [
            "any target probability per derived cell",
            "any label",
            "any label-generation seed",
            "any train or test split",
            "any learner, estimator, or hyperparameter",
            "any threshold or verdict on labelled-data burden",
        ],
    }


@dataclass(frozen=True, slots=True)
class Fixture:
    """One declared world, its document, and everything derived from it."""

    identifier: str
    schema_name: str
    digest: str
    role: str
    world: TypedWorld
    document: dict[str, object]
    compiled: Compiled
    perturbations: tuple[tuple[str, str, str, TypedWorld], ...]


def build_fixtures() -> tuple[Fixture, ...]:
    """Load, compile, and perturb both declared worlds through one code path."""
    built: list[Fixture] = []
    for loader, schema_name, digest in (
        (load_world_a, WORLD_A_SCHEMA, WORLD_A_SHA256),
        (load_world_b, WORLD_B_SCHEMA, WORLD_B_SHA256),
    ):
        world, document = loader()
        built.append(
            Fixture(
                identifier=world.identifier,
                schema_name=schema_name,
                digest=digest,
                role=str(document["role"]),
                world=world,
                document=document,
                compiled=compile_world(world),
                perturbations=build_perturbations(document, world.identifier),
            )
        )
    return tuple(built)


def perturbation_report(fixture: Fixture) -> list[dict[str, object]]:
    """Recompile every declared perturbation and score it against its direction."""
    rows: list[dict[str, object]] = []
    primary = fixture.compiled
    for name, direction, rationale, world in fixture.perturbations:
        changed = compile_world(world)
        if direction == "grows":
            held = (
                changed.group_order > primary.group_order
                and changed.cell_count < primary.cell_count
            )
        elif direction == "shrinks":
            held = (
                changed.group_order < primary.group_order
                and changed.cell_count > primary.cell_count
            )
        elif direction == "differs":
            held = changed.quotient != primary.quotient
        else:
            raise RuntimeError(f"undeclared prediction {direction!r}")
        subgroup: bool | None = None
        if direction == "shrinks":
            base = compile_world(fixture.world)
            if changed.morphisms is not None and base.morphisms is not None:
                forgotten = {
                    Morphism(images=morphism.images) for morphism in changed.morphisms
                }
                subgroup = forgotten <= set(base.morphisms)
        rows.append(
            {
                "perturbation": name,
                "frozen_direction": direction,
                "frozen_rationale": rationale,
                "group_order": changed.group_order,
                "cell_count": changed.cell_count,
                "cell_sizes": changed.cell_sizes if changed.cell_count <= 16 else None,
                "quotient_differs_from_the_primary": (
                    changed.quotient != primary.quotient
                ),
                "quotient_refines_the_primary": _refines(
                    changed.quotient, primary.quotient
                ),
                "primary_refines_the_quotient": _refines(
                    primary.quotient, changed.quotient
                ),
                "new_group_is_a_subgroup_of_the_primary": subgroup,
                "recompiled_from_scratch": True,
                "prediction_held": held,
            }
        )
    return rows


def authority_record() -> dict[str, object]:
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.26-Task-derived-symmetry-from-typed-world-structure.md"
        ),
        "execution_contract": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.26b-GPT-owner-tightening-derived-symmetry-execution-"
            "contract.md"
        ),
        "review_carried_into_the_contract": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.26a-Canary-review-derived-symmetry-compiler-boundary.md"
        ),
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "prior_quilt_revision": PRIOR_QUILT_REVISION,
        "inherits": [
            "017.25a the compiled quotient is separable from the declared relation",
            "017.25 the first matched TDM training-economy experiment",
            "017.23 operational TDM equivalence and the S(E) state boundary",
        ],
        "byte_pinned_inputs": {
            "world_a_schema.json": WORLD_A_SHA256,
            "world_b_schema.json": WORLD_B_SHA256,
        },
    }


def phase_s_certificate(fixtures: Sequence[Fixture]) -> dict[str, object]:
    """The structure-only freeze, banked before any target probability exists."""
    worlds = [fixture.world for fixture in fixtures] + [
        world for fixture in fixtures for *_rest, world in fixture.perturbations
    ]
    blocks: list[dict[str, object]] = []
    for fixture in fixtures:
        compiled = fixture.compiled
        blocks.append(
            {
                "fixture": fixture.identifier,
                "role": fixture.role,
                "schema": fixture.schema_name,
                "sha256": fixture.digest,
                "A_declared": {
                    "carriers": [
                        {"name": carrier.name, "cardinality": len(carrier.elements)}
                        for carrier in fixture.world.carriers
                    ],
                    "primitives": [
                        {
                            "name": primitive.name,
                            "kind": primitive.kind,
                            "signature": (
                                list(primitive.argument_carriers)
                                + (
                                    [f"-> {primitive.value_carrier}"]
                                    if primitive.value_carrier is not None
                                    else []
                                )
                            ),
                            "rows_declared": (
                                len(primitive.extension)
                                if primitive.kind == "relation"
                                else len(primitive.graph)
                                if primitive.kind == "operation"
                                else 1
                            ),
                        }
                        for primitive in fixture.world.primitives
                    ],
                    "admitted_input_type": " x ".join(
                        fixture.world.input_signature
                    ),
                    "admitted_inputs": len(fixture.world.inputs),
                    "declaration_ledger": declaration_ledger(fixture.document),
                },
                "B_anti_shortcut_audit": forbidden_metadata_audit(fixture.document),
                "C_derived_before_any_label": {
                    "structure_preserving_group_order": compiled.group_order,
                    "group_was_materialized": compiled.morphisms is not None,
                    "input_cell_count": compiled.cell_count,
                    "input_cell_cardinalities": compiled.cell_sizes,
                    "cells": {
                        f"Q_{cell}": [
                            "".join(row)
                            for row, value in zip(
                                fixture.world.inputs, compiled.quotient, strict=True
                            )
                            if value == cell
                        ]
                        for cell in sorted(set(compiled.quotient))
                    }
                    if compiled.cell_count <= 8
                    else "not listed; more cells than the reporting cap",
                    "cell_naming_rule": (
                        "cells are numbered by cardinality and then by least "
                        "member in declared input order; the names are "
                        "bookkeeping and carry no meaning"
                    ),
                    "maximality_witness": maximality_witness(
                        fixture.world, compiled, REDUCTION_SIZES[fixture.identifier]
                    ),
                },
                "D_frozen_shortcut_language": shortcut_audit(
                    fixture.world, compiled.quotient
                ),
                "D_disclosed_non_local_adversary": non_local_adversary_audit(
                    fixture.world, compiled.quotient
                ),
                "E_independence_of_the_derivation": independence_audit(
                    fixture.document, compiled
                ),
                "F_source_perturbations": perturbation_report(fixture),
                "G_joint_load_bearing": joint_load_bearing(
                    fixture.world, compiled, fixture.perturbations
                ),
            }
        )

    dispatch = dispatch_audit(worlds)
    fence = phase_fence_audit()
    checks: dict[str, bool] = {
        "no_declared_identifier_appears_in_the_compiler": bool(
            dispatch["no_declared_identifier_appears_in_the_compiler"]
        ),
        "no_fixture_expectation_or_label_vocabulary_in_the_compiler": bool(
            dispatch["no_fixture_or_expectation_or_label_vocabulary_appears"]
        ),
        "every_declared_primitive_kind_has_a_generated_rule": bool(
            dispatch["every_declared_kind_has_a_generated_rule"]
        ),
        "this_module_imports_no_statistical_module": bool(
            fence["imports_no_statistical_module"]
        ),
        "no_target_parameter_vector_appears_in_this_module": bool(
            fence["no_target_parameter_vector_appears_in_this_module"]
        ),
        "both_world_documents_match_their_pins": True,
        "the_same_compiler_consumed_every_world_and_every_perturbation": len(
            worlds
        )
        == len(fixtures) + sum(len(f.perturbations) for f in fixtures),
    }
    for fixture, block in zip(fixtures, blocks, strict=True):
        tag = fixture.identifier
        derived = block["C_derived_before_any_label"]
        witness = derived["maximality_witness"]  # type: ignore[index]
        independence = block["E_independence_of_the_derivation"]
        shortcut = block["D_frozen_shortcut_language"]
        checks[f"fixture_{tag}_declares_no_forbidden_metadata"] = bool(
            block["B_anti_shortcut_audit"]["structural_subtree_is_clean"]  # type: ignore[index]
        )
        checks[f"fixture_{tag}_declares_no_float_in_its_structure"] = bool(
            block["B_anti_shortcut_audit"][  # type: ignore[index]
                "no_numeric_probability_can_hide_in_the_structure"
            ]
        )
        checks[f"fixture_{tag}_group_is_nontrivial"] = (
            fixture.compiled.group_order > 1
        )
        checks[f"fixture_{tag}_quotient_is_neither_one_cell_nor_all_singletons"] = (
            1 < fixture.compiled.cell_count < len(fixture.world.inputs)
        )
        checks[f"fixture_{tag}_every_derived_element_preserves_every_primitive"] = (
            witness["every_derived_element_preserves_every_primitive"] is True  # type: ignore[index]
        )
        checks[f"fixture_{tag}_the_two_quotient_routes_agree"] = (
            witness["quotient_derived_twice_by_different_routes"]["routes_agree"]  # type: ignore[index]
            is True
        )
        checks[f"fixture_{tag}_unpruned_brute_force_agrees_on_a_reduced_world"] = (
            bool(witness["unpruned_brute_force_on_a_reduced_world"]["orders_agree"])  # type: ignore[index]
            and bool(
                witness["unpruned_brute_force_on_a_reduced_world"][  # type: ignore[index]
                    "element_sets_agree"
                ]
            )
        )
        checks[f"fixture_{tag}_a_second_exhaustive_decomposition_agrees"] = bool(
            witness["second_exhaustive_decomposition"].get(  # type: ignore[index]
                "agrees_with_the_primary_search", False
            )
        )
        checks[f"fixture_{tag}_random_search_found_no_larger_group"] = bool(
            witness["random_search_for_a_missed_element"]["no_larger_group_was_found"]  # type: ignore[index]
        )
        checks[f"fixture_{tag}_derivation_survives_relabelling_every_name"] = bool(
            independence["relabelling"]["quotient_is_identical"]  # type: ignore[index]
        ) and bool(
            independence["relabelling"]["group_order_is_identical"]  # type: ignore[index]
        )
        checks[
            f"fixture_{tag}_derivation_survives_deleting_non_structural_keys"
        ] = bool(
            independence["key_stripping"]["quotient_is_identical"]  # type: ignore[index]
        ) and bool(
            independence["key_stripping"]["group_order_is_identical"]  # type: ignore[index]
        )
        checks[
            f"fixture_{tag}_the_frozen_shortcut_language_does_not_recover_the_quotient"
        ] = not bool(
            shortcut["the_whole_language_read_jointly_recovers_the_quotient"]  # type: ignore[index]
        )
        checks[f"fixture_{tag}_every_perturbation_matched_its_frozen_direction"] = all(
            bool(row["prediction_held"])
            for row in block["F_source_perturbations"]  # type: ignore[union-attr]
        )
        checks[f"fixture_{tag}_every_declared_primitive_is_load_bearing"] = bool(
            block["G_joint_load_bearing"]["every_primitive_is_load_bearing"]  # type: ignore[index]
        )
    checks["the_richer_fixture_has_more_than_one_declared_primitive"] = (
        len(fixtures[1].world.primitives) > 1
    )
    checks["the_richer_fixture_is_jointly_load_bearing"] = bool(
        blocks[1]["G_joint_load_bearing"][  # type: ignore[index]
            "primitives_are_jointly_load_bearing"
        ]
    )
    if not all(checks.values()):
        raise RuntimeError(
            "Phase S mechanical checks failed: "
            f"{sorted(name for name, ok in checks.items() if not ok)}"
        )
    return {
        "schema": CERTIFICATE_SCHEMA,
        "phase": "S",
        "what_this_is": (
            "the structure-only freeze required by 017.26b section 7. Every "
            "quantity here is derived from ordinary declared world structure "
            "before any target probability or label seed exists."
        ),
        "authority": authority_record(),
        "phase_fence": fence,
        "compiler": {
            "dispatch_audit": dispatch,
            "search": (
                "exhaustive backtracking over the full product of symmetric "
                "groups on the declared carriers, rejecting a branch only when a "
                "declared obligation is already violated; when no obligation "
                "remains outstanding the subtree size is counted in closed form, "
                "which is what keeps a world with an erased primitive tractable"
            ),
            "variable_order_rule": (
                "at each step take the unassigned variable that makes the "
                "largest number of ground obligations fully determined, breaking "
                "ties by index. The order affects only how much pruning happens, "
                "never which leaves exist."
            ),
            "quotient_rule": (
                "q_Sigma(x) = q_Sigma(y) exactly when some derived "
                "structure-preserving map carries x to y, asked one ordered pair "
                "at a time so that the group never has to be materialized"
            ),
            "what_the_compiler_is_not": (
                "Q_aut is one preregistered exact structural compiler, not a "
                "claim that automorphism orbits are the canonical or universal "
                "quotient a typed decision model ought to use. 017.26b section 2 "
                "requires it to be named an automorphism-orbit compiler, and "
                "definability, context, congruence, bisimulation-like, and "
                "task-relative constructions are left entirely open."
            ),
        },
        "fixtures": blocks,
        "cross_fixture_witness": {
            "requirement": (
                "017.26b section 3.2 asks that the same compiler be shown "
                "consuming the single-relation fixture, the richer "
                "multi-primitive fixture, and the source perturbations"
            ),
            "worlds_compiled_by_the_one_path": len(worlds),
            "primary_worlds": [fixture.identifier for fixture in fixtures],
            "perturbed_worlds": [
                name
                for fixture in fixtures
                for name, *_rest in fixture.perturbations
            ],
            "primitive_kinds_exercised": sorted(
                {
                    primitive.kind
                    for world in worlds
                    for primitive in world.primitives
                }
            ),
            "carrier_cardinalities_exercised": sorted(
                {
                    len(carrier.elements)
                    for world in worlds
                    for carrier in world.carriers
                }
            ),
        },
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": all(checks.values()),
        "what_this_freeze_does_not_establish": [
            "that the automorphism-orbit quotient is the right compiler for any "
            "world other than these two",
            "that no cheaper derivation of q_Sigma exists; the disclosed "
            "non-local adversary already recovers it on both fixtures, which is "
            "a tractability finding banked here rather than discovered later",
            "anything at all about labelled-data burden, which Phase T measures "
            "and which no quantity in this file could speak to",
            "that either fixture is representative of anything beyond itself",
            "tractability of this compiler at larger carrier cardinalities",
        ],
    }


def phase_s_cost_report(
    fixtures: Sequence[Fixture], total_seconds: float
) -> dict[str, object]:
    """Wall-clock cost of derivation, kept out of the byte-pinned certificate.

    Timings cannot live in an artifact that has to reproduce byte for byte, so
    they live here and ``--check`` only requires this file to exist. What the
    numbers are for is the 017.26 section 19 J question: what did derivation cost
    against what Training will cost.
    """
    rows: list[dict[str, object]] = []
    for fixture in fixtures:
        perturbation_seconds = 0.0
        for _name, _direction, _rationale, world in fixture.perturbations:
            perturbation_seconds += compile_world(world).seconds
        rows.append(
            {
                "fixture": fixture.identifier,
                "admitted_inputs": len(fixture.world.inputs),
                "full_product_of_symmetric_groups": math.prod(
                    math.factorial(len(carrier.elements))
                    for carrier in fixture.world.carriers
                ),
                "derived_group_order": fixture.compiled.group_order,
                "compile_seconds": _round(fixture.compiled.seconds),
                "perturbation_recompile_seconds": _round(perturbation_seconds),
            }
        )
    return {
        "schema": COST_SCHEMA,
        "phase": "S",
        "authority": authority_record(),
        "per_fixture": rows,
        "seconds": {
            "total_wall_clock_including_every_audit": _round(total_seconds)
        },
        "environment": {"python": sys.version.split()[0]},
        "note": (
            "derivation is cheap on these worlds and that is not evidence it "
            "stays cheap. The search is exhaustive over a product of symmetric "
            "groups, so cost grows factorially in carrier cardinality in the "
            "worst case, and tractability at larger schema sizes is an open "
            "item rather than a result."
        ),
    }


# ---------------------------------------------------------------------------
# 9. The command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true", help="write the Phase S artifacts"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute the Phase S certificate and verify the committed bytes",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    fixtures = build_fixtures()
    payload = phase_s_certificate(fixtures)
    blob = _canonical(payload)

    if args.check:
        problems: list[str] = []
        if not CERTIFICATE_PATH.exists():
            problems.append(f"missing artifact: {CERTIFICATE_PATH.name}")
        else:
            observed = CERTIFICATE_PATH.read_bytes()
            if observed != blob:
                problems.append(
                    f"{CERTIFICATE_PATH.name} mismatch: observed "
                    f"{_sha256(observed)}, expected {_sha256(blob)}"
                )
        if not COST_PATH.exists():
            problems.append(f"missing artifact: {COST_PATH.name}")
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(
            "017.26 Phase S verified: certificate "
            f"{_sha256(blob)[:16]} "
            f"checks {payload['mechanical_check_count']}"
        )
        return 0

    if args.freeze:
        CERTIFICATE_PATH.write_bytes(blob)
        print(f"wrote {CERTIFICATE_PATH.name} {_sha256(blob)}")
        cost = _canonical(
            phase_s_cost_report(fixtures, time.perf_counter() - started)
        )
        COST_PATH.write_bytes(cost)
        print(f"wrote {COST_PATH.name} {_sha256(cost)}")
        return 0

    sys.stdout.buffer.write(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
