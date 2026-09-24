"""Issue 017.28 Phase S: which structural compiler do we actually need?

017.27 earned a narrow result -- ordinary typed source structure was sufficient
for a generic exact compiler to derive a useful input quotient before labels --
and then disclosed the next problem in the same breath. Full automorphism
enumeration was not necessary on either fixture. An iterated pair refinement
recovered the same quotient in two rounds of polynomial work.

So the question this module asks is not whether compilation helps. It is:

    what structural compiler is sufficient, canonical, and tractable?

The immediate scientific target is a *separation* between cheap fixed-point
refinement and exact automorphism-orbit compilation. Two compilers are put on
the same footing:

    Q_ref(Sigma)   the frozen fixed-point refinement of this module
    Q_aut(Sigma)   X_Sigma / Aut(A_Sigma), the byte-pinned 017.26 compiler

and a preregistered finite family of typed worlds is searched, in a frozen
order, for a world where they disagree.

Three things make this module the structural half of a fenced experiment.

**No target exists here.** This module imports the standard library and the
byte-pinned 017.26 Phase S module, and nothing else. The 017.26 module does not
import the 017.24 implementation, because that module contains
``generate_observations``. So no label generator, no target probability, no
seed schedule, and no estimator is reachable from this file by any path. The
claim is a property of the import graph, not a promise about discipline.

**The generator family is frozen before any comparison runs.** Six families,
each a deterministic function of frozen integer parameters, are enumerated in
an order that is not hand-chosen: families are sorted by declared primitive
count, then declared sort count, then name, and that rule is asserted
mechanically against the declared order. Within a family, parameters are sorted
by declared carrier size then by the parameter tuple, and that is asserted too.
No randomness is used anywhere in the family -- the quasigroup family
enumerates *reduced Latin squares in lexicographic order*, which is a
construction, not a choice.

**The search reports every world.** The complete ledger is banked, equality
cases included, and the first separation under the frozen order is designated
primary before any statistics exist.

A disclosure that belongs here rather than in a footnote. The parameter ranges
were set by an off-family tractability probe -- shuffled and intercalate-turned
cyclic Latin squares, and circulant graphs at sizes above the declared bound --
run before the family was frozen, so that the declared size bounds reflect
compile cost rather than compile results. None of the probe worlds is a member
of the family: the family's quasigroups are lexicographically reduced squares,
which the probe never generated. The probe also fixed the exact-compiler route:
materialize the group and close the admitted inputs under it when the group is
small enough, because the pair-by-pair orbit query costs an order of magnitude
more on a world whose group is nearly trivial, and 017.27 established that the
two routes agree.

Everything here is exact and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

CERTIFICATE_SCHEMA: Final = "gpt-01728-phase-s-compiler-scope/v1"
COST_SCHEMA: Final = "gpt-01728-phase-s-cost-report/v1"
WORLD_SCHEMA: Final = "gpt-01728-generated-typed-world/v1"

# The 017.26 Phase S module, byte-pinned. Q_aut comes through this pin and
# nowhere else, so the exact reference of this turn is provably the compiler that
# was banked before theta_star existed in 017.26.
PHASE_S_01726_SHA256: Final = (
    "e7074a60ed460b7b98d7f234e3af06aec169e5ae526e477c62e1e8799d0f2e3b"
)
PHASE_S_01726_CERTIFICATE_SHA256: Final = (
    "f2e36998f0e5715d785d5dd0a86790fc32604040334c8250d2fd6c845c29709a"
)

TASK_GPT_REVISION: Final = (
    "e5289c582121bda62394321bf73b4b476f9bb23c49eb6efd19044c14366009f1"
)
PRIOR_RESULT_COMMIT: Final = "2817b8c1a1d0e6b10d4ea6ac16d1bbce35df06a8"
PRIOR_QUILT_REVISION: Final = (
    "16f56ff72b50adf1361618dfa92b66b3501377d8d2abe19a749cf1cebcc9fe55"
)

HERE: Final = Path(__file__).resolve().parent
PHASE_S_01726_PATH: Final = (
    HERE.parent / "017.26-Code-attachments" / "derived_symmetry.py"
)
PHASE_S_01726_CERTIFICATE_PATH: Final = (
    HERE.parent / "017.26-Code-attachments" / "phase_s_certificate.json"
)
CERTIFICATE_PATH: Final = HERE / "phase_s_certificate.json"
COST_PATH: Final = HERE / "phase_s_cost_report.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_pinned(path: Path, digest: str, alias: str) -> object:
    """Import a module byte-pinned, refusing to run if its bytes have changed."""
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != digest:
        raise RuntimeError(
            f"pinned module {path.name} changed: expected {digest}, "
            f"observed {observed}"
        )
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the pinned module {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


exact = _load_pinned(
    PHASE_S_01726_PATH, PHASE_S_01726_SHA256, "derived_symmetry_01726_pinned"
)

# Declared bounds on the search, frozen before any world is compared. A world
# whose declared size exceeds a bound is skipped on its *declared size alone*,
# which is readable from the document without compiling anything, so no result
# can influence what is searched.
MAX_ADMITTED_INPUTS: Final = 144
MAX_TOTAL_CARRIER_ELEMENTS: Final = 40
MAX_WORLDS: Final = 200

# The frozen Q_ref fixed-point cap. The pinned 017.26 procedure uses 8; this
# module allows more and asserts that the fixpoint was reached inside 8, which is
# what makes the pinned call a valid witness on every searched world.
REFINEMENT_ROUND_CAP: Final = 32
PINNED_REFINEMENT_CAP: Final = 8

# Prefix sizes for the independent unpruned cross-check of Q_aut. Witness
# parameters, not compiler inputs.
CROSS_CHECK_REDUCTION: Final = 4

NUMERIC_TOLERANCE: Final = 1e-9

Row = tuple[str, ...]
Partition = tuple[int, ...]

_refines = exact._refines
_cell_sizes = exact._cell_sizes


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _round(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def _canonical_partition(colour: Sequence[int]) -> Partition:
    """Rename cells so they are sorted by cardinality, then by least member.

    This is the 017.24 naming rule, reused unchanged. It is bookkeeping and
    carries no meaning, but using the same rule as every prior turn is what lets
    two partitions from two turns be compared as literal tuples.
    """
    groups: dict[int, list[int]] = {}
    for position, cell in enumerate(colour):
        groups.setdefault(cell, []).append(position)
    ordering = sorted(groups.values(), key=lambda members: (len(members), members[0]))
    assignment = [-1] * len(colour)
    for cell, members in enumerate(ordering):
        for position in members:
            assignment[position] = cell
    return tuple(assignment)


def _strictly_coarser(coarse: Sequence[int], fine: Sequence[int]) -> bool:
    return _refines(fine, coarse) and not _refines(coarse, fine)


# ---------------------------------------------------------------------------
# 1. Compiler R: the frozen fixed-point refinement, specified then implemented
# ---------------------------------------------------------------------------

# 017.28 section 4 refuses to let the phrase "pair refinement" carry a
# comparison, and lists nine things that must be defined before a separation
# fixture may be selected. This table is that definition. It is the frozen
# specification of Q_ref, it is emitted verbatim into the certificate, and the
# implementation below is written against it.
Q_REF_SPECIFICATION: Final = {
    "name": "fixed-point pair refinement over the admitted inputs",
    "provenance": (
        "this is the formalization of the procedure 017.27 disclosed at its "
        "Phase S freeze as a bound on the L_k claim. The pinned 017.26 "
        "implementation of that procedure is called on every searched world and "
        "its partition is required to equal this module's independently written "
        "implementation, so the formalization cannot have drifted from what "
        "017.27 actually reported"
    ),
    "1_initial_colours": (
        "the colour of an admitted input x is the identifier of its reflexive "
        "atomic type tau(x, x), where tau is defined in item 2. Admitted inputs "
        "with the same reflexive atomic type start with the same colour and no "
        "other information is used to seed a colour"
    ),
    "2_seeding_facts": (
        "the atomic type tau(x, y) of an ordered pair of admitted inputs is "
        "computed from the *slot pool* of the pair: the typed components of x in "
        "declared input-signature order, then the typed components of y in the "
        "same order, then the value of every declared constant, then -- as one "
        "closure round -- the value of every declared operation on every "
        "type-correct tuple of the slots listed so far. tau(x, y) records, for "
        "every declared relation in declaration order, exactly which "
        "type-correct tuples of slot positions satisfy it, and which pairs of "
        "same-sorted slot positions hold equal elements. Nothing else is read: "
        "no primitive name, no carrier name, no element name, no document key "
        "outside the structural subtree"
    ),
    "3_context_objects": (
        "the whole admitted-input set X_Sigma. Every admitted input is a context "
        "object for every other one, so the construction is a global fixpoint "
        "over X_Sigma rather than a source-local feature of a single input"
    ),
    "4_accumulated_signature": (
        "at each round the signature of x is the pair (current colour of x, the "
        "sorted multiset over every y in X_Sigma of (identifier of tau(x, y), "
        "current colour of y)). The current colour of x is retained as the first "
        "component, which is what makes the refinement monotone"
    ),
    "5_colour_renaming_rule": (
        "within a round, distinct signatures are numbered by first appearance in "
        "declared admitted-input order. Because the numbering is one fixed "
        "bijection for the whole round, and the signature comparison is "
        "equality of multisets, the induced partition does not depend on the "
        "numbering. At the fixpoint the cells are renamed once more, sorted by "
        "cardinality and then by least member index, which is the 017.24 naming "
        "rule and is reused so that partitions from different turns are "
        "comparable as literal tuples"
    ),
    "6_fixed_point_criterion": (
        "stop when a round does not increase the number of colour classes. "
        "Because the previous colour is the first component of the signature the "
        "partition can only get finer, so an unchanged class count means an "
        "unchanged partition and the fixpoint has been reached. The round budget "
        f"is {REFINEMENT_ROUND_CAP}, and reaching the cap without a fixpoint is "
        "a recorded failure rather than a silent truncation"
    ),
    "7_multiple_carrier_sorts": (
        "every slot carries its declared sort. A relation ranges only over "
        "type-correct tuples of slots, an equality test is only formed between "
        "slots of the same sort, and an operation closes only over type-correct "
        "argument tuples. The admitted-input signature may repeat a sort or "
        "combine distinct sorts; nothing in the construction assumes two "
        "components, two sorts, or a binary primitive"
    ),
    "8_relations_operations_constants": (
        "a relation contributes the set of type-correct slot-position tuples in "
        "its extension. An operation contributes one new slot per type-correct "
        "argument tuple of the pre-closure pool, carrying its declared value, "
        "and those new slots then participate in relation and equality facts. A "
        "constant contributes one slot carrying its declared value. Primitives "
        "are referenced by declaration index rather than by name, so no name is "
        "load-bearing"
    ),
    "9_complexity": (
        "|X_Sigma|^2 atomic types, each costing sum over declared relations of "
        "(slot count)^(arity), plus at most the round budget passes of "
        "O(|X_Sigma|^2 log |X_Sigma|). Polynomial in |X_Sigma| and in the "
        "declared primitive sizes for bounded arity. Q_aut by contrast searches "
        "a product of symmetric groups and is factorial in carrier cardinality "
        "in the worst case"
    ),
    "relabelling_invariance": (
        "the atomic type is a set of slot *positions* and primitive *indices*, "
        "never a name or an element, and the slot layout is determined by the "
        "declared signature order. Renaming every carrier, element, and "
        "primitive therefore leaves every atomic type identifier unchanged, "
        "which is asserted mechanically on every searched world"
    ),
    "what_is_not_claimed": (
        "Q_ref is canonical only relative to this frozen construction. It has no "
        "intrinsic characterization of the kind Q_aut has, and agreement with "
        "Q_aut on some fixtures does not transfer Q_aut's mathematical "
        "canonicity to it"
    ),
}


@dataclass(frozen=True, slots=True)
class Refined:
    """What the refinement compiler derives from one declared world."""

    partition: Partition
    rounds: int
    cell_counts_by_round: tuple[int, ...]
    atomic_partition: Partition
    bounded_partition: Partition
    distinct_atomic_types: int
    slot_count: int
    reached_fixed_point: bool
    seconds: float

    @property
    def cell_count(self) -> int:
        return len(set(self.partition))

    @property
    def cell_sizes(self) -> list[int]:
        return _cell_sizes(self.partition)


def _slot_pool(world: object, left: Row, right: Row) -> tuple[tuple[str, str], ...]:
    """The typed slot pool of an ordered pair of admitted inputs.

    Components of the left input, then of the right, then declared constants,
    then one closure round under the declared operations. The order is fixed by
    the declared signature and declaration order, so it is invariant under
    renaming.
    """
    pooled: list[tuple[str, str]] = []
    for side in (left, right):
        for position, carrier in enumerate(world.input_signature):
            pooled.append((carrier, side[position]))
    for primitive in world.primitives:
        if primitive.kind != "constant":
            continue
        if primitive.value_carrier is None or primitive.value is None:
            raise RuntimeError("a constant must declare a carrier and a value")
        pooled.append((primitive.value_carrier, primitive.value))
    grown: list[tuple[str, str]] = []
    for primitive in world.primitives:
        if primitive.kind != "operation":
            continue
        if primitive.value_carrier is None:
            raise RuntimeError("an operation must declare a codomain")
        table = dict(primitive.graph)
        pools = [
            [slot for slot in pooled if slot[0] == carrier]
            for carrier in primitive.argument_carriers
        ]
        for chosen in itertools.product(*pools):
            grown.append(
                (primitive.value_carrier, table[tuple(slot[1] for slot in chosen)])
            )
    return tuple([*pooled, *grown])


def _atomic_type(world: object, left: Row, right: Row) -> tuple[object, ...]:
    """tau(x, y): which declared facts the slot pool of the pair satisfies.

    Encoded as sets of slot *positions*, keyed by primitive declaration *index*.
    No name and no element appears in the result, which is what makes the type
    invariant under relabelling by construction rather than by test.
    """
    slots = _slot_pool(world, left, right)
    facts: list[object] = []
    for index, primitive in enumerate(world.primitives):
        if primitive.kind != "relation":
            continue
        positions = [
            [slot for slot, item in enumerate(slots) if item[0] == carrier]
            for carrier in primitive.argument_carriers
        ]
        satisfied = frozenset(
            combination
            for combination in itertools.product(*positions)
            if tuple(slots[slot][1] for slot in combination) in primitive.extension
        )
        facts.append((index, satisfied))
    equal = frozenset(
        (first, second)
        for first in range(len(slots))
        for second in range(first + 1, len(slots))
        if slots[first][0] == slots[second][0] and slots[first][1] == slots[second][1]
    )
    facts.append((-1, equal))
    return tuple(facts)


def derive_refinement_quotient(
    world: object, cap: int = REFINEMENT_ROUND_CAP
) -> Refined:
    """Compiler R. Deterministic, polynomial, and blind to everything but source."""
    started = time.perf_counter()
    inputs = world.inputs
    width = len(inputs)

    identifier: dict[tuple[object, ...], int] = {}
    table = [[0] * width for _ in range(width)]
    for left in range(width):
        for right in range(width):
            table[left][right] = identifier.setdefault(
                _atomic_type(world, inputs[left], inputs[right]), len(identifier)
            )

    colour = _renumber([table[row][row] for row in range(width)])
    counts = [len(set(colour))]
    atomic = _canonical_partition(colour)
    bounded: Partition | None = None
    rounds = 0
    reached = False
    for _step in range(cap):
        signatures = [
            (
                colour[left],
                tuple(sorted((table[left][right], colour[right]) for right in range(width))),
            )
            for left in range(width)
        ]
        candidate = _renumber(signatures)
        rounds += 1
        if bounded is None:
            bounded = _canonical_partition(candidate)
        counts.append(len(set(candidate)))
        if len(set(candidate)) == len(set(colour)):
            colour = candidate
            reached = True
            break
        colour = candidate
    if bounded is None:
        bounded = atomic
    return Refined(
        partition=_canonical_partition(colour),
        rounds=rounds,
        cell_counts_by_round=tuple(counts),
        atomic_partition=atomic,
        bounded_partition=bounded,
        distinct_atomic_types=len(identifier),
        slot_count=len(_slot_pool(world, inputs[0], inputs[0])),
        reached_fixed_point=reached,
        seconds=time.perf_counter() - started,
    )


def _renumber(values: Sequence[object]) -> list[int]:
    lookup: dict[object, int] = {}
    return [lookup.setdefault(value, len(lookup)) for value in values]


# ---------------------------------------------------------------------------
# 2. Compiler A: the exact automorphism-orbit quotient, byte-pinned
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Exact:
    """What the exact compiler derives, and by which of its two routes."""

    partition: Partition
    group_order: int
    materialized: bool
    route: str
    seconds: float
    plan: object
    morphisms: object

    @property
    def cell_count(self) -> int:
        return len(set(self.partition))

    @property
    def cell_sizes(self) -> list[int]:
        return _cell_sizes(self.partition)


def derive_exact_quotient(world: object) -> Exact:
    """Compiler A. Q_aut(Sigma) = X_Sigma / Aut(A_Sigma), exactly.

    Two exact routes are available and 017.27 verified that they agree. When the
    derived group is small enough to materialize, the admitted inputs are closed
    under it by union-find, which costs O(|Aut| |X_Sigma|). Otherwise the orbit
    question is asked pair by pair, which is exact for any group size but pays a
    restricted exhaustive search per pair. The route is recorded, and on every
    designated fixture both routes are run and required to agree.
    """
    started = time.perf_counter()
    plan = exact.prepare(world)
    order = exact.count_morphisms(plan)
    morphisms = (
        exact.enumerate_morphisms(plan) if order <= exact.ENUMERATION_BUDGET else None
    )
    if morphisms is not None:
        partition = exact.partition_from_group(plan, morphisms)
        route = "materialize the derived group, then close the inputs by union-find"
    else:
        partition = exact.derive_orbit_quotient(plan)
        route = "ask the orbit question pair by pair, by restricted exhaustive search"
    return Exact(
        partition=tuple(partition),
        group_order=order,
        materialized=morphisms is not None,
        route=route,
        seconds=time.perf_counter() - started,
        plan=plan,
        morphisms=morphisms,
    )


COSET_WITNESS_CAP: Final = 200_000


def cross_check_exact(world: object, derived: Exact) -> dict[str, object]:
    """Re-derive Q_aut by the other exact route, and re-verify every element.

    Three independent things. The pair-by-pair orbit query is run and must return
    the same partition as the union-find closure, which is the 017.27 agreement
    re-established on this turn's worlds rather than inherited. Every
    materialized element is re-verified by the 017.26 independent checker, which
    re-derives the preservation rules from the declared signature instead of
    reusing the search's compiled obligations, so agreement between the two is a
    real check rather than a tautology. And Cayley distance one is exhausted: no
    composition of a derived element with a transposition preserves every
    primitive, which is a local maximality witness.
    """
    pairwise = tuple(exact.derive_orbit_quotient(derived.plan))
    report: dict[str, object] = {
        "pair_by_pair_orbit_partition_cell_count": len(set(pairwise)),
        "the_two_exact_routes_agree": pairwise == derived.partition,
    }
    if derived.morphisms is None:
        report["every_element_reverified_independently"] = None
        report["cayley_distance_one_neighbours_checked"] = 0
        report["no_neighbour_preserves_every_primitive"] = None
        report["cayley_witness_within_budget"] = False
        return report
    report["every_element_reverified_independently"] = all(
        exact.preserves_every_primitive(world, morphism)
        for morphism in derived.morphisms
    )
    known = set(derived.morphisms)
    transpositions = sum(
        len(carrier.elements) * (len(carrier.elements) - 1) // 2
        for carrier in world.carriers
    )
    if len(derived.morphisms) * max(transpositions, 1) > COSET_WITNESS_CAP:
        report["cayley_distance_one_neighbours_checked"] = 0
        report["no_neighbour_preserves_every_primitive"] = None
        report["cayley_witness_within_budget"] = False
        return report
    neighbours = 0
    escaped = 0
    for morphism in derived.morphisms:
        mapping = morphism.as_mapping(world)
        for carrier in world.carriers:
            elements = carrier.elements
            for first in range(len(elements)):
                for second in range(first + 1, len(elements)):
                    row = dict(mapping[carrier.name])
                    row[elements[first]] = mapping[carrier.name][elements[second]]
                    row[elements[second]] = mapping[carrier.name][elements[first]]
                    moved = {**mapping, carrier.name: row}
                    candidate = exact.Morphism(
                        images=tuple(
                            (
                                name,
                                tuple(
                                    moved[name][element]
                                    for element in world.carrier(name).elements
                                ),
                            )
                            for name in world.carrier_names
                        )
                    )
                    neighbours += 1
                    if candidate in known:
                        continue
                    if exact.preserves_every_primitive(world, candidate):
                        escaped += 1
    report["cayley_distance_one_neighbours_checked"] = neighbours
    report["no_neighbour_preserves_every_primitive"] = escaped == 0
    report["cayley_witness_within_budget"] = True
    return report


# ---------------------------------------------------------------------------
# 3. The frozen fixture generator family G
# ---------------------------------------------------------------------------


def _relation(name: str, carriers: Sequence[str], rows: Sequence[Row]) -> dict:
    return {
        "name": name,
        "kind": "relation",
        "signature": list(carriers),
        "extension": [list(row) for row in sorted(rows)],
    }


def _operation(
    name: str, domain: Sequence[str], codomain: str, graph: Sequence[tuple[Row, str]]
) -> dict:
    return {
        "name": name,
        "kind": "operation",
        "domain": list(domain),
        "codomain": codomain,
        "graph": [
            {"arguments": list(arguments), "value": value} for arguments, value in graph
        ],
    }


def _sort(name: str, prefix: str, size: int) -> dict:
    return {"name": name, "elements": [f"{prefix}{index}" for index in range(size)]}


def _document(
    identifier: str,
    family: str,
    parameters: dict[str, object],
    sorts: Sequence[dict],
    primitives: Sequence[dict],
    product: Sequence[str],
) -> dict:
    return {
        "schema": WORLD_SCHEMA,
        "fixture": identifier,
        "family": family,
        "parameters": parameters,
        "sorts": list(sorts),
        "primitives": list(primitives),
        "admitted_inputs": {"product": list(product)},
    }


def _symmetric_circulant(size: int, offsets: Sequence[int], prefix: str) -> list[Row]:
    pairs: set[tuple[int, int]] = set()
    for base in range(size):
        for offset in offsets:
            other = (base + offset) % size
            if base == other:
                continue
            pairs.add((base, other))
            pairs.add((other, base))
    return [(f"{prefix}{left}", f"{prefix}{right}") for left, right in sorted(pairs)]


def _directed_circulant(size: int, offsets: Sequence[int], prefix: str) -> list[Row]:
    pairs = sorted(
        {
            (base, (base + offset) % size)
            for base in range(size)
            for offset in offsets
            if (base + offset) % size != base
        }
    )
    return [(f"{prefix}{left}", f"{prefix}{right}") for left, right in pairs]


def build_circulant_graph(size: int, offsets: tuple[int, ...]) -> dict:
    """One carrier, one symmetric circulant relation, inputs the vertex square."""
    return _document(
        identifier=f"circulant_graph:n={size}:offsets={'-'.join(map(str, offsets))}",
        family="circulant_graph",
        parameters={"size": size, "offsets": list(offsets)},
        sorts=[_sort("Vertex", "v", size)],
        primitives=[
            _relation("edge", ("Vertex", "Vertex"), _symmetric_circulant(size, offsets, "v"))
        ],
        product=("Vertex", "Vertex"),
    )


def build_bipartite_incidence(size: int, offsets: tuple[int, ...]) -> dict:
    """Two carriers, one incidence relation, inputs the point-line product."""
    rows = [
        (f"p{base}", f"l{(base + offset) % size}")
        for base in range(size)
        for offset in offsets
    ]
    return _document(
        identifier=f"bipartite_incidence:n={size}:offsets={'-'.join(map(str, offsets))}",
        family="bipartite_incidence",
        parameters={"size": size, "offsets": list(offsets)},
        sorts=[_sort("Point", "p", size), _sort("Line", "l", size)],
        primitives=[_relation("incident", ("Point", "Line"), rows)],
        product=("Point", "Line"),
    )


def reduced_latin_squares(order: int, wanted: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """The first ``wanted`` reduced Latin squares of the order, in lex order.

    A reduced Latin square has its first row and first column in natural order.
    Backtracking fills the remaining cells in row-major order, trying symbols in
    increasing order, so the squares are produced in lexicographic order of the
    flattened matrix. This is a construction and not a choice: no random draw, no
    sampling, and no property of the square is consulted.
    """
    square = [[-1] * order for _ in range(order)]
    for index in range(order):
        square[0][index] = index
        square[index][0] = index
    found: list[tuple[tuple[int, ...], ...]] = []

    def place(row: int, column: int) -> None:
        if len(found) >= wanted:
            return
        if row == order:
            found.append(tuple(tuple(line) for line in square))
            return
        if column == order:
            place(row + 1, 1)
            return
        for value in range(order):
            if value in square[row][:column]:
                continue
            if any(square[above][column] == value for above in range(row)):
                continue
            square[row][column] = value
            place(row, column + 1)
            square[row][column] = -1
            if len(found) >= wanted:
                return

    place(1, 1)
    return tuple(found)


def build_quasigroup(order: int, index: int) -> dict:
    """Three carriers, one total binary operation given by a reduced Latin square.

    Aut of this world is exactly the autotopism group of the square: triples of
    bijections (alpha, beta, gamma) with gamma(L(r, c)) = L(alpha(r), beta(c)).
    Nothing about that is declared; it is what preservation of the declared
    operation means, and the compiler derives it from the signature.
    """
    square = reduced_latin_squares(order, index + 1)[index]
    graph = [
        ((f"r{row}", f"c{column}"), f"s{square[row][column]}")
        for row in range(order)
        for column in range(order)
    ]
    return _document(
        identifier=f"quasigroup:n={order}:square={index}",
        family="quasigroup",
        parameters={"order": order, "square_index": index},
        sorts=[
            _sort("Row", "r", order),
            _sort("Col", "c", order),
            _sort("Sym", "s", order),
        ],
        primitives=[_operation("op", ("Row", "Col"), "Sym", graph)],
        product=("Row", "Col"),
    )


def build_two_relation_graph(
    size: int, edge_offsets: tuple[int, ...], arc_offsets: tuple[int, ...]
) -> dict:
    """One carrier carrying two relations of different kinds of symmetry."""
    tag = f"{'-'.join(map(str, edge_offsets))}_{'-'.join(map(str, arc_offsets))}"
    return _document(
        identifier=f"two_relation_graph:n={size}:offsets={tag}",
        family="two_relation_graph",
        parameters={
            "size": size,
            "edge_offsets": list(edge_offsets),
            "arc_offsets": list(arc_offsets),
        },
        sorts=[_sort("Vertex", "v", size)],
        primitives=[
            _relation(
                "edge", ("Vertex", "Vertex"), _symmetric_circulant(size, edge_offsets, "v")
            ),
            _relation(
                "arc", ("Vertex", "Vertex"), _directed_circulant(size, arc_offsets, "v")
            ),
        ],
        product=("Vertex", "Vertex"),
    )


def build_corridor(sites: int, districts: int) -> dict:
    """Two carriers, a ring relation and a total typed map onto the smaller one."""
    if sites % districts:
        raise RuntimeError("the corridor family requires districts to divide sites")
    run = sites // districts
    graph = [((f"s{index}",), f"d{index // run}") for index in range(sites)]
    return _document(
        identifier=f"corridor:n={sites}:m={districts}",
        family="corridor",
        parameters={"sites": sites, "districts": districts},
        sorts=[_sort("Site", "s", sites), _sort("District", "d", districts)],
        primitives=[
            _relation("adjacent", ("Site", "Site"), _symmetric_circulant(sites, (1,), "s")),
            _operation("district", ("Site",), "District", graph),
        ],
        product=("Site", "District"),
    )


def build_composite_product(
    size: int, order: int, offsets: tuple[int, ...]
) -> dict:
    """A circulant-graph block and a quasigroup block, coupled by a typed map.

    This is the section 6 "products or compositions of smaller worlds" bullet
    realized as a genuine composition rather than a disjoint union: the declared
    link map ties the graph block to the quasigroup block, so a structure
    preserving map must respect both blocks and the coupling at once.
    """
    square = reduced_latin_squares(order, 1)[0]
    graph = [
        ((f"r{row}", f"c{column}"), f"s{square[row][column]}")
        for row in range(order)
        for column in range(order)
    ]
    link = [((f"v{index}",), f"r{index % order}") for index in range(size)]
    tag = "-".join(map(str, offsets))
    return _document(
        identifier=f"composite_product:n={size}:m={order}:offsets={tag}",
        family="composite_product",
        parameters={"size": size, "order": order, "offsets": list(offsets)},
        sorts=[
            _sort("Vertex", "v", size),
            _sort("Row", "r", order),
            _sort("Col", "c", order),
            _sort("Sym", "s", order),
        ],
        primitives=[
            _relation(
                "edge", ("Vertex", "Vertex"), _symmetric_circulant(size, offsets, "v")
            ),
            _operation("op", ("Row", "Col"), "Sym", graph),
            _operation("link", ("Vertex",), "Row", link),
        ],
        product=("Vertex", "Col"),
    )


CIRCULANT_PARAMETERS: Final = (
    (5, (1,)),
    (5, (2,)),
    (6, (1,)),
    (6, (2,)),
    (6, (1, 3)),
    (6, (2, 3)),
    (7, (1,)),
    (7, (1, 2)),
    (7, (1, 3)),
    (8, (1,)),
    (8, (1, 2)),
    (8, (1, 4)),
    (8, (2, 3)),
    (8, (1, 2, 3)),
    (9, (1,)),
    (9, (3,)),
    (9, (1, 3)),
    (10, (1,)),
    (10, (1, 5)),
    (10, (2, 5)),
    (12, (1, 6)),
    (12, (3, 4)),
)

INCIDENCE_PARAMETERS: Final = (
    (5, (0, 1)),
    (5, (0, 1, 2)),
    (6, (0, 1)),
    (6, (0, 1, 2)),
    (6, (0, 1, 3)),
    (7, (0, 1)),
    (7, (0, 1, 3)),
    (7, (0, 1, 2, 4)),
    (8, (0, 1, 3)),
    (8, (0, 1, 2, 4)),
    (9, (0, 1, 3)),
    (9, (0, 1, 2, 4)),
    (11, (0, 1, 3)),
)

QUASIGROUP_ORDERS: Final = (4, 5, 6, 7, 8)
QUASIGROUP_SQUARES_PER_ORDER: Final = 6
# Order 4 has only four reduced Latin squares in total, so the count is read off
# the enumerator rather than assumed.
QUASIGROUP_PARAMETERS: Final = tuple(
    (order, index)
    for order in QUASIGROUP_ORDERS
    for index in range(len(reduced_latin_squares(order, QUASIGROUP_SQUARES_PER_ORDER)))
)

TWO_RELATION_PARAMETERS: Final = (
    (6, (1,), (2,)),
    (6, (1,), (3,)),
    (7, (1,), (2,)),
    (7, (1,), (3,)),
    (8, (1,), (2,)),
    (8, (1,), (4,)),
    (8, (2,), (3,)),
    (9, (1,), (3,)),
    (10, (1,), (2,)),
    (10, (1,), (5,)),
)

CORRIDOR_PARAMETERS: Final = (
    (6, 2),
    (6, 3),
    (8, 2),
    (8, 4),
    (9, 3),
    (10, 2),
    (10, 5),
    (12, 3),
    (12, 4),
)

COMPOSITE_PARAMETERS: Final = (
    (6, 3, (1,)),
    (6, 3, (1, 3)),
    (8, 4, (1,)),
    (9, 3, (1, 3)),
    (10, 5, (1,)),
)


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """One frozen family of the generator, with its declared profile."""

    name: str
    sorts: int
    primitives: int
    rationale: str
    parameters: tuple[tuple[object, ...], ...]
    build: Callable[..., dict]


def _total_elements(document: dict) -> int:
    return sum(len(block["elements"]) for block in document["sorts"])


def _admitted_inputs(document: dict) -> int:
    sizes = {block["name"]: len(block["elements"]) for block in document["sorts"]}
    total = 1
    for name in document["admitted_inputs"]["product"]:
        total *= sizes[name]
    return total


def _flatten_parameters(parameters: Sequence[object]) -> tuple[int, ...]:
    values: list[int] = []
    for item in parameters:
        if isinstance(item, tuple):
            values.extend(int(value) for value in item)
        else:
            values.append(int(item))  # type: ignore[arg-type]
    return tuple(values)


def _family_key(spec: FamilySpec) -> tuple[int, int, str]:
    return (spec.primitives, spec.sorts, spec.name)


def _parameter_key(
    build: Callable[..., dict], parameters: Sequence[object]
) -> tuple[int, tuple[int, ...]]:
    return (_total_elements(build(*parameters)), _flatten_parameters(parameters))


_FAMILY_TABLE: Final = (
    FamilySpec(
        name="circulant_graph",
        sorts=1,
        primitives=1,
        rationale=(
            "the simplest regular-graph construction section 6 names: a single "
            "carrier, a single symmetric relation, and the admitted inputs the "
            "vertex square. Every vertex has the same declared degree, so no "
            "local feature distinguishes one from another and the compilers are "
            "compared on what they can do with global structure alone"
        ),
        parameters=CIRCULANT_PARAMETERS,
        build=build_circulant_graph,
    ),
    FamilySpec(
        name="bipartite_incidence",
        sorts=2,
        primitives=1,
        rationale=(
            "the incidence construction section 6 names, and the same "
            "construction shape as the 017.27 single-relation fixture. The "
            "equality replay itself reads the byte-pinned 017.27 document rather "
            "than a regenerated one, so the family is not required to contain "
            "that exact world and does not claim to"
        ),
        parameters=INCIDENCE_PARAMETERS,
        build=build_bipartite_incidence,
    ),
    FamilySpec(
        name="quasigroup",
        sorts=3,
        primitives=1,
        rationale=(
            "the typed-operation construction section 6 names, taken to three "
            "carriers. A Latin square is the maximally regular binary operation: "
            "every row and every column carries every symbol exactly once, so "
            "the declared local data is constant across the whole world, and the "
            "derived group is exactly the autotopism group of the square. This "
            "is where the two compilers are most likely to come apart, and "
            "saying so in advance is part of the preregistration"
        ),
        parameters=QUASIGROUP_PARAMETERS,
        build=build_quasigroup,
    ),
    FamilySpec(
        name="two_relation_graph",
        sorts=1,
        primitives=2,
        rationale=(
            "the multiple-relations bullet of section 6, with one symmetric and "
            "one directed relation on the same carrier, so joint preservation of "
            "primitives of the same kind but different symmetry is exercised"
        ),
        parameters=TWO_RELATION_PARAMETERS,
        build=build_two_relation_graph,
    ),
    FamilySpec(
        name="corridor",
        sorts=2,
        primitives=2,
        rationale=(
            "the distinguished-source-structure and typed-map bullets of "
            "section 6, and the same construction shape as the 017.27 "
            "multi-primitive fixture. The equality replay reads the byte-pinned "
            "017.27 document rather than a regenerated one"
        ),
        parameters=CORRIDOR_PARAMETERS,
        build=build_corridor,
    ),
    FamilySpec(
        name="composite_product",
        sorts=4,
        primitives=3,
        rationale=(
            "the products-and-compositions bullet of section 6: a graph block "
            "and a quasigroup block coupled by a declared typed map, so a "
            "structure-preserving map must respect two blocks and their coupling "
            "at once"
        ),
        parameters=COMPOSITE_PARAMETERS,
        build=build_composite_product,
    ),
)

FAMILY_ORDER_RULE: Final = (
    "the enumeration order is *computed* by a frozen rule, not written down and "
    "then checked. Families are sorted by declared primitive count, then declared "
    "carrier-sort count, then family name. Within a family, parameter tuples are "
    "sorted by total declared carrier elements, then by the parameter tuple "
    "flattened to a tuple of integers. The declared tables are therefore only "
    "sets of parameters; the sequence a reader sees is the rule applied to them, "
    "so no hand-arrangement of the order is possible. What is checked mechanically "
    "is that no two entries tie under the rule, so the computed order is total, "
    "and that each family's declared sort and primitive counts are exactly what "
    "its builder produces"
)

# The frozen order, computed rather than declared.
FAMILIES: Final = tuple(
    FamilySpec(
        name=spec.name,
        sorts=spec.sorts,
        primitives=spec.primitives,
        rationale=spec.rationale,
        parameters=tuple(
            sorted(spec.parameters, key=lambda item: _parameter_key(spec.build, item))
        ),
        build=spec.build,
    )
    for spec in sorted(_FAMILY_TABLE, key=_family_key)
)


def family_order_is_total() -> bool:
    keys = [_family_key(spec) for spec in FAMILIES]
    return keys == sorted(keys) and len(set(keys)) == len(keys)


def parameter_order_is_total(spec: FamilySpec) -> bool:
    keys = [_parameter_key(spec.build, parameters) for parameters in spec.parameters]
    return keys == sorted(keys) and len(set(keys)) == len(keys)


def declared_profile_matches_the_builder(spec: FamilySpec) -> bool:
    return all(
        len(spec.build(*parameters)["sorts"]) == spec.sorts
        and len(spec.build(*parameters)["primitives"]) == spec.primitives
        for parameters in spec.parameters
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One world the frozen family proposes, before anything is compiled."""

    position: int
    family: str
    parameters: tuple[object, ...]
    identifier: str
    document: dict
    admitted_inputs: int
    total_elements: int

    @property
    def within_declared_bounds(self) -> bool:
        return (
            self.admitted_inputs <= MAX_ADMITTED_INPUTS
            and self.total_elements <= MAX_TOTAL_CARRIER_ELEMENTS
        )


def enumerate_family() -> tuple[Candidate, ...]:
    """Realize the frozen family in the frozen order. Nothing is compiled here."""
    candidates: list[Candidate] = []
    for spec in FAMILIES:
        for parameters in spec.parameters:
            if len(candidates) >= MAX_WORLDS:
                return tuple(candidates)
            document = spec.build(*parameters)
            candidates.append(
                Candidate(
                    position=len(candidates),
                    family=spec.name,
                    parameters=parameters,
                    identifier=str(document["fixture"]),
                    document=document,
                    admitted_inputs=_admitted_inputs(document),
                    total_elements=_total_elements(document),
                )
            )
    return tuple(candidates)


# ---------------------------------------------------------------------------
# 4. The search protocol
# ---------------------------------------------------------------------------

SEARCH_PROTOCOL: Final = {
    "families": [spec.name for spec in FAMILIES],
    "family_order_rule": FAMILY_ORDER_RULE,
    "declared_world_budget": MAX_WORLDS,
    "declared_size_bounds": {
        "max_admitted_inputs": MAX_ADMITTED_INPUTS,
        "max_total_carrier_elements": MAX_TOTAL_CARRIER_ELEMENTS,
        "why": (
            "both bounds are read off a world document before anything is "
            "compiled, so a world can only be skipped for its declared size and "
            "never for what a compiler returned. The numbers were set by an "
            "off-family tractability probe run before the family was frozen"
        ),
    },
    "stop_rule": (
        "enumerate every world of the frozen family in the frozen order, "
        "compile both quotients on every world within the declared size bounds, "
        "and record every outcome including the equality cases. The first world "
        "under the frozen order with Q_ref != Q_aut is the primary separation "
        "fixture. The first such world whose Q_aut is not the discrete partition "
        "is additionally designated the graded separation fixture, because a "
        "world where the exact compiler degenerates to one parameter per "
        "admitted input cannot show an estimation trade-off. Both designations "
        "are made here, in Phase S, before any target exists. The family is not "
        "extended, reordered, or truncated after any comparison is seen"
    ),
    "randomness": (
        "none. No seed is drawn anywhere in the family or the search. The "
        "quasigroup family enumerates reduced Latin squares in lexicographic "
        "order by backtracking, which is a construction rather than a sample"
    ),
}


@dataclass(frozen=True, slots=True)
class Searched:
    """One searched world, with both quotients and every comparison."""

    candidate: Candidate
    world: object
    refined: Refined
    derived: Exact
    separates: bool
    refinement_is_no_finer_than_orbits: bool
    splits_an_orbit: bool


def search_world(candidate: Candidate) -> Searched:
    """Compile both quotients on one world and compare them exactly."""
    world = exact.world_from_document(candidate.document, candidate.identifier)
    refined = derive_refinement_quotient(world)
    derived = derive_exact_quotient(world)
    coarser = _refines(derived.partition, refined.partition)
    return Searched(
        candidate=candidate,
        world=world,
        refined=refined,
        derived=derived,
        separates=refined.partition != derived.partition,
        refinement_is_no_finer_than_orbits=coarser,
        splits_an_orbit=not coarser,
    )


def is_discrete(partition: Sequence[int]) -> bool:
    return len(set(partition)) == len(partition)


def ledger_row(found: Searched) -> dict[str, object]:
    """One row of the complete search ledger. Cheap enough to keep for every world."""
    candidate = found.candidate
    return {
        "position": candidate.position,
        "family": candidate.family,
        "world": candidate.identifier,
        "parameters": list(map(str, candidate.parameters)),
        "admitted_inputs": candidate.admitted_inputs,
        "total_carrier_elements": candidate.total_elements,
        "declared_primitives": [
            {
                "index": index,
                "kind": primitive.kind,
                "arity": len(primitive.argument_carriers),
                "declared_rows": (
                    len(primitive.extension)
                    if primitive.kind == "relation"
                    else len(primitive.graph)
                    if primitive.kind == "operation"
                    else 1
                ),
            }
            for index, primitive in enumerate(found.world.primitives)
        ],
        "Q_ref_cells": found.refined.cell_count,
        "Q_ref_cell_sizes": found.refined.cell_sizes,
        "Q_ref_rounds": found.refined.rounds,
        "Q_ref_cell_counts_by_round": list(found.refined.cell_counts_by_round),
        "Q_ref_reached_fixed_point": found.refined.reached_fixed_point,
        "Q_ref_distinct_atomic_types": found.refined.distinct_atomic_types,
        "Q_ref_slots_per_pair": found.refined.slot_count,
        "group_order": found.derived.group_order,
        "Q_aut_cells": found.derived.cell_count,
        "Q_aut_cell_sizes": found.derived.cell_sizes,
        "Q_aut_is_discrete": is_discrete(found.derived.partition),
        "Q_aut_route": found.derived.route,
        "Q_aut_group_materialized": found.derived.materialized,
        "partitions_are_equal": not found.separates,
        "Q_ref_is_no_finer_than_Q_aut": found.refinement_is_no_finer_than_orbits,
        "Q_ref_splits_an_automorphism_orbit": found.splits_an_orbit,
        "separates": found.separates,
        "statistical_degrees_of_freedom": {
            "under_Q_ref": found.refined.cell_count,
            "under_Q_aut": found.derived.cell_count,
            "under_a_saturated_model": candidate.admitted_inputs,
        },
    }


def timing_row(found: Searched) -> dict[str, object]:
    """Wall clock for one world.

    Timings live in the cost report and never in the certificate. The
    certificate is the immutable structural freeze and section 12 requires it to
    be byte-identical on recomputation, which a wall-clock number cannot be.
    """
    return {
        "position": found.candidate.position,
        "family": found.candidate.family,
        "world": found.candidate.identifier,
        "admitted_inputs": found.candidate.admitted_inputs,
        "total_carrier_elements": found.candidate.total_elements,
        "declared_primitive_rows": [
            len(primitive.extension)
            if primitive.kind == "relation"
            else len(primitive.graph)
            if primitive.kind == "operation"
            else 1
            for primitive in found.world.primitives
        ],
        "Q_ref_rounds": found.refined.rounds,
        "Q_ref_cells": found.refined.cell_count,
        "Q_aut_cells": found.derived.cell_count,
        "group_order": found.derived.group_order,
        "Q_ref_seconds": _round(found.refined.seconds),
        "Q_aut_seconds": _round(found.derived.seconds),
        "dof_under_Q_ref": found.refined.cell_count,
        "dof_under_Q_aut": found.derived.cell_count,
        "dof_under_a_saturated_model": found.candidate.admitted_inputs,
    }


# ---------------------------------------------------------------------------
# 5. Equality replay on the two 017.27 worlds
# ---------------------------------------------------------------------------


def load_01726_certificate() -> dict[str, object]:
    """Read the banked 017.26 Phase S freeze, byte-pinned."""
    raw = PHASE_S_01726_CERTIFICATE_PATH.read_bytes()
    observed = _sha256(raw)
    if observed != PHASE_S_01726_CERTIFICATE_SHA256:
        raise RuntimeError(
            f"the 017.26 Phase S certificate changed: expected "
            f"{PHASE_S_01726_CERTIFICATE_SHA256}, observed {observed}"
        )
    document = json.loads(raw)
    if document["phase"] != "S":
        raise RuntimeError("unexpected phase in the 017.26 Phase S certificate")
    return document


def banked_quotient(
    certificate: dict[str, object], identifier: str, world: object
) -> Partition:
    """Reconstruct a banked quotient from the 017.26 cell membership lists.

    This never calls a compiler. It parses the cells the 017.26 freeze recorded,
    which is the only way the equality replay can mean "the same partition
    017.27 reported" rather than "whatever the compiler says today".
    """
    for block in certificate["fixtures"]:
        if block["fixture"] != identifier:
            continue
        cells = block["C_derived_before_any_label"]["cells"]
        position = {"".join(row): index for index, row in enumerate(world.inputs)}
        assignment = [-1] * len(position)
        for name in sorted(cells, key=lambda key: int(key.split("_")[1])):
            for member in cells[name]:
                assignment[position[member]] = int(name.split("_")[1])
        if min(assignment) < 0:
            raise RuntimeError("the 017.26 freeze did not cover every admitted input")
        return tuple(assignment)
    raise RuntimeError(f"the 017.26 freeze has no fixture {identifier!r}")


REPLAY_REPORTED_ROUNDS: Final = 2


def equality_replay() -> dict[str, object]:
    """Section 5. Replay the two 017.27 worlds before searching for a separation."""
    certificate = load_01726_certificate()
    rows: list[dict[str, object]] = []
    for loader in (exact.load_world_a, exact.load_world_b):
        world, _document = loader()
        refined = derive_refinement_quotient(world)
        derived = derive_exact_quotient(world)
        pinned, pinned_rounds = exact.pair_refinement(world)
        banked = banked_quotient(certificate, world.identifier, world)
        rows.append(
            {
                "world": world.identifier,
                "admitted_inputs": len(world.inputs),
                "group_order": derived.group_order,
                "Q_aut_cells": derived.cell_count,
                "Q_aut_cell_sizes": derived.cell_sizes,
                "Q_ref_cells": refined.cell_count,
                "Q_ref_cell_sizes": refined.cell_sizes,
                "Q_ref_rounds": refined.rounds,
                "Q_ref_equals_Q_aut": refined.partition == derived.partition,
                "Q_aut_reproduces_the_017_26_banked_quotient": (
                    derived.partition == banked
                ),
                "Q_ref_reproduces_the_017_26_banked_quotient": (
                    refined.partition == banked
                ),
                "this_module_agrees_with_the_pinned_017_26_procedure": (
                    _canonical_partition(pinned) == refined.partition
                ),
                "pinned_procedure_rounds": pinned_rounds,
                "rounds_match_the_017_27_report": refined.rounds
                == REPLAY_REPORTED_ROUNDS,
            }
        )
    return {
        "requirement": (
            "017.28 section 5 requires Q_ref(Sigma_A) = Q_aut(Sigma_A) and "
            "Q_ref(Sigma_B) = Q_aut(Sigma_B) before a separation is sought, and "
            "requires the two-round convergence 017.27 reported to be reproduced "
            "under the now-frozen definition"
        ),
        "discrepancy_with_the_017_27_procedure": (
            "none. The frozen Q_ref of this module is the same construction, "
            "reimplemented independently with a different encoding of the atomic "
            "type -- sets of satisfied slot positions keyed by primitive "
            "declaration index, rather than a vector of booleans -- and the "
            "pinned 017.26 procedure is run alongside it on every world in this "
            "turn and required to return the same partition. The only deliberate "
            "difference is instrumentation: this implementation also exposes the "
            "round-0 colouring and the one-round colouring, which the 017.26 "
            "procedure does not return, and a larger round budget with an "
            "explicit fixpoint flag"
        ),
        "fixtures": rows,
        "both_worlds_reproduce_equality": all(
            bool(row["Q_ref_equals_Q_aut"]) for row in rows
        ),
        "both_worlds_reproduce_the_banked_quotient": all(
            bool(row["Q_aut_reproduces_the_017_26_banked_quotient"])
            and bool(row["Q_ref_reproduces_the_017_26_banked_quotient"])
            for row in rows
        ),
        "both_worlds_reproduce_two_round_convergence": all(
            bool(row["rounds_match_the_017_27_report"]) for row in rows
        ),
        "this_module_agrees_with_the_pinned_procedure_on_both": all(
            bool(row["this_module_agrees_with_the_pinned_017_26_procedure"])
            for row in rows
        ),
    }


# ---------------------------------------------------------------------------
# 6. What counts as a separation, and the exact witnesses
# ---------------------------------------------------------------------------


def _cells_of(partition: Sequence[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for position, cell in enumerate(partition):
        groups.setdefault(cell, []).append(position)
    return groups


def separation_witness(found: Searched) -> dict[str, object]:
    """Section 8. Exhibit the exact inputs refinement merges that orbits separate.

    The witness pair is chosen mechanically: the lowest-numbered Q_ref cell that
    contains more than one Q_aut orbit, and within it the two lowest-indexed
    admitted inputs lying in different orbits. The non-existence certificate is
    the pinned exhaustive search itself -- ``exists_morphism_sending`` returns
    false only after exhausting the whole product of symmetric groups under sound
    pruning -- and, when the group was materialized, it is confirmed a second way
    by applying every derived element to the source and checking that the target
    is never reached.
    """
    world = found.world
    inputs = world.inputs
    refined_cells = _cells_of(found.refined.partition)
    orbit_of = found.derived.partition

    chosen: tuple[int, int, int] | None = None
    merged: list[dict[str, object]] = []
    for cell in sorted(refined_cells):
        members = refined_cells[cell]
        orbits = sorted({orbit_of[index] for index in members})
        if len(orbits) < 2:
            continue
        merged.append(
            {
                "Q_ref_cell": f"R_{cell}",
                "size": len(members),
                "Q_aut_orbits_merged": len(orbits),
                "Q_aut_orbit_sizes_merged": sorted(
                    sum(1 for value in orbit_of if value == orbit) for orbit in orbits
                ),
            }
        )
        if chosen is None:
            first = members[0]
            second = next(
                index for index in members if orbit_of[index] != orbit_of[first]
            )
            chosen = (cell, first, second)

    if chosen is None:
        return {
            "separation_exists": False,
            "why": "no Q_ref cell contains more than one automorphism orbit",
        }

    cell, left, right = chosen
    source, target = inputs[left], inputs[right]
    reachable = exact.exists_morphism_sending(found.derived.plan, source, target)
    by_group: bool | None = None
    if found.derived.morphisms is not None:
        by_group = any(
            morphism.send(world, source) == target
            for morphism in found.derived.morphisms
        )
    orbit_members = [
        "".join(inputs[index])
        for index in range(len(inputs))
        if orbit_of[index] == orbit_of[left]
    ]
    other_members = [
        "".join(inputs[index])
        for index in range(len(inputs))
        if orbit_of[index] == orbit_of[right]
    ]
    return {
        "separation_exists": True,
        "world": found.candidate.identifier,
        "witness_pair": {
            "x": "".join(source),
            "y": "".join(target),
            "shared_Q_ref_cell": f"R_{cell}",
            "Q_aut_orbit_of_x": f"A_{orbit_of[left]}",
            "Q_aut_orbit_of_y": f"A_{orbit_of[right]}",
            "x_and_y_share_a_refinement_cell": found.refined.partition[left]
            == found.refined.partition[right],
            "no_structure_preserving_map_sends_x_to_y": not reachable,
            "confirmed_by_applying_every_derived_element": (
                None if by_group is None else not by_group
            ),
            "certificate": (
                "exists_morphism_sending exhausts the whole product of symmetric "
                "groups under sound pruning and returns false, so the negative "
                "is exact rather than a search failure"
            ),
            "Q_aut_orbit_of_x_members": orbit_members,
            "Q_aut_orbit_of_y_members": other_members,
            "colour_of_x_by_round": "identical to y at every round, by "
            "construction of the shared final cell",
        },
        "merged_cells": merged,
        "Q_ref_cell_count": found.refined.cell_count,
        "Q_ref_cell_sizes": found.refined.cell_sizes,
        "Q_aut_cell_count": found.derived.cell_count,
        "Q_aut_cell_sizes": found.derived.cell_sizes,
        "group_order": found.derived.group_order,
        "Q_ref_is_no_finer_than_Q_aut": found.refinement_is_no_finer_than_orbits,
        "Q_ref_is_strictly_coarser_than_Q_aut": _strictly_coarser(
            found.refined.partition, found.derived.partition
        ),
    }


# ---------------------------------------------------------------------------
# 7. Canonicality audit: representation invariance and determinism
# ---------------------------------------------------------------------------

CANONICALITY_SENSES: Final = {
    "representation_invariance": (
        "relabelling every carrier, element, and primitive does not change the "
        "quotient. Both compilers must satisfy this and both are tested. Because "
        "the relabelling used here renames positionally and preserves declared "
        "order, the transported partition must be the *identical tuple*, which is "
        "a stronger check than equality up to renaming"
    ),
    "algorithmic_determinism": (
        "given the same serialized schema the compiler returns byte-identical "
        "canonical output. Tested by recomputing every partition twice in one "
        "process, and by --check recomputing the whole certificate and demanding "
        "byte-identity"
    ),
    "mathematical_canonicity": (
        "Q_aut has an intrinsic characterization: it is the orbit partition of "
        "the full automorphism group of the declared structure, which is defined "
        "without reference to any algorithm. Q_ref has no such characterization. "
        "It is canonical only relative to the frozen construction of section 4 of "
        "this module, and agreement with Q_aut on any number of fixtures does not "
        "transfer Q_aut's canonicity to it"
    ),
}


def invariance_audit(
    found: Searched, include_exact: bool
) -> dict[str, object]:
    """Recompile a relabelled and a key-stripped copy of the world from scratch."""
    document = found.candidate.document
    relabelled = exact.world_from_document(
        exact.relabel_document(document), f"{found.candidate.identifier}:relabelled"
    )
    stripped = exact.world_from_document(
        exact.strip_to_structure(document), f"{found.candidate.identifier}:stripped"
    )
    relabelled_ref = derive_refinement_quotient(relabelled)
    stripped_ref = derive_refinement_quotient(stripped)
    repeat_ref = derive_refinement_quotient(found.world)
    report: dict[str, object] = {
        "world": found.candidate.identifier,
        "Q_ref_is_relabelling_invariant": relabelled_ref.partition
        == found.refined.partition,
        "Q_ref_ignores_every_non_structural_key": stripped_ref.partition
        == found.refined.partition,
        "Q_ref_is_deterministic_within_a_process": repeat_ref.partition
        == found.refined.partition,
        "Q_ref_round_count_is_relabelling_invariant": relabelled_ref.rounds
        == found.refined.rounds,
        "exact_compiler_audited_here": include_exact,
    }
    if include_exact:
        relabelled_exact = derive_exact_quotient(relabelled)
        stripped_exact = derive_exact_quotient(stripped)
        report["Q_aut_is_relabelling_invariant"] = (
            relabelled_exact.partition == found.derived.partition
            and relabelled_exact.group_order == found.derived.group_order
        )
        report["Q_aut_ignores_every_non_structural_key"] = (
            stripped_exact.partition == found.derived.partition
            and stripped_exact.group_order == found.derived.group_order
        )
    return report


# ---------------------------------------------------------------------------
# 8. The compiler ladder, measured rather than assumed
# ---------------------------------------------------------------------------

LADDER_LEVELS: Final = (
    "local_atomic_features",
    "bounded_refinement_one_round",
    "fixed_point_refinement_Q_ref",
    "exact_automorphism_orbit_Q_aut",
)


def ladder_row(found: Searched) -> dict[str, object]:
    """Verify, rather than assume, that the four levels are ordered on this world."""
    atomic = found.refined.atomic_partition
    bounded = found.refined.bounded_partition
    fixpoint = found.refined.partition
    orbits = found.derived.partition
    return {
        "world": found.candidate.identifier,
        "cells": {
            "local_atomic_features": len(set(atomic)),
            "bounded_refinement_one_round": len(set(bounded)),
            "fixed_point_refinement_Q_ref": len(set(fixpoint)),
            "exact_automorphism_orbit_Q_aut": len(set(orbits)),
        },
        "bounded_refines_atomic": _refines(bounded, atomic),
        "fixpoint_refines_bounded": _refines(fixpoint, bounded),
        "orbits_refine_fixpoint": _refines(orbits, fixpoint),
        "ladder_holds": (
            _refines(bounded, atomic)
            and _refines(fixpoint, bounded)
            and _refines(orbits, fixpoint)
        ),
        "strict_at": [
            name
            for name, finer, coarser in (
                ("bounded_over_atomic", bounded, atomic),
                ("fixpoint_over_bounded", fixpoint, bounded),
                ("orbits_over_fixpoint", orbits, fixpoint),
            )
            if _strictly_coarser(coarser, finer)
        ],
    }


PLAUSIBLE_ALTERNATIVES: Final = (
    {
        "compiler": "definability quotient",
        "idea": (
            "identify admitted inputs not separated by any formula of a declared "
            "logic over the schema signature"
        ),
        "status": "not implemented in this turn",
    },
    {
        "compiler": "congruence closure",
        "idea": (
            "quotient by the least congruence of the declared algebraic "
            "primitives containing a declared generating relation"
        ),
        "status": "not implemented in this turn",
    },
    {
        "compiler": "bisimulation-like equivalence",
        "idea": (
            "coinductive equivalence over a transition reading of the declared "
            "relations, which is the natural coarsest rather than finest "
            "fixpoint"
        ),
        "status": "not implemented in this turn",
    },
    {
        "compiler": "context equivalence",
        "idea": (
            "identify inputs indistinguishable by every declared context of a "
            "bounded shape, which generalizes the 017.26 L_k family upward"
        ),
        "status": "not implemented in this turn",
    },
    {
        "compiler": "task-relative observational quotient",
        "idea": (
            "identify inputs that no admissible observable of the decision "
            "problem distinguishes, which is the only level on this list that "
            "would be allowed to read the target"
        ),
        "status": "not implemented in this turn, and would break the phase fence",
    },
    {
        "compiler": "canonical forms",
        "idea": (
            "compute a canonical labelling of the declared structure and read "
            "orbits off it, which is the same partition as Q_aut by a different "
            "and usually faster algorithm"
        ),
        "status": "not implemented in this turn",
    },
)


# ---------------------------------------------------------------------------
# 9. Source counterfactuals on the separation fixture
# ---------------------------------------------------------------------------

COUNTERFACTUAL_DIRECTIONS: Final = (
    (
        "erase_the_first_declared_primitive",
        "the gap must disappear or change form, because a world with less "
        "declared structure has a larger group and a coarser orbit partition, "
        "and the refinement loses the same facts",
    ),
    (
        "distinguish_one_declared_element",
        "adding a declared constant can only remove structure-preserving maps, "
        "so Q_aut refines; whether the gap survives is the question",
    ),
    (
        "relabel_every_carrier",
        "nothing may change at all; this is the serialization-accident control",
    ),
    (
        "compose_with_a_declared_tag_map",
        "the composite_product coupling device applied to the separation "
        "fixture: a two-element carrier and a total typed map onto it. Real "
        "source structure is added, so Q_aut can only refine",
    ),
)


def compose_with_tag_map(document: dict) -> dict:
    """Add a two-element carrier and a declared total map onto it.

    This is the coupling device the ``composite_product`` family uses, applied to
    an already-generated world. It is ordinary source structure -- a declared
    binary attribute of the first carrier -- and it is added through the same
    ``operation`` kind the compiler already reaches by its rule table.
    """
    changed = json.loads(json.dumps(document))
    first = changed["sorts"][0]
    elements = [str(value) for value in first["elements"]]
    changed["sorts"] = [
        *changed["sorts"],
        {"name": "Tag", "elements": ["t0", "t1"]},
    ]
    changed["primitives"] = [
        *changed["primitives"],
        {
            "name": "tag",
            "kind": "operation",
            "domain": [first["name"]],
            "codomain": "Tag",
            "graph": [
                {"arguments": [element], "value": f"t{index % 2}"}
                for index, element in enumerate(elements)
            ],
        },
    ]
    return changed


def counterfactual_report(found: Searched) -> dict[str, object]:
    """Section 14. Alter only source structure, and recompute both compilers."""
    document = found.candidate.document
    baseline_gap = found.derived.cell_count - found.refined.cell_count
    first_primitive = str(document["primitives"][0]["name"])
    first_carrier = str(document["sorts"][0]["name"])
    first_element = str(document["sorts"][0]["elements"][0])

    transforms = {
        "erase_the_first_declared_primitive": exact.erase_primitive(
            document, first_primitive
        ),
        "distinguish_one_declared_element": exact.break_symmetry(
            document, first_carrier, first_element, "reference"
        ),
        "relabel_every_carrier": exact.relabel_document(document),
        "compose_with_a_declared_tag_map": compose_with_tag_map(document),
    }

    rows: list[dict[str, object]] = []
    for name, direction in COUNTERFACTUAL_DIRECTIONS:
        world = exact.world_from_document(
            transforms[name], f"{found.candidate.identifier}:{name}"
        )
        refined = derive_refinement_quotient(world)
        derived = derive_exact_quotient(world)
        separates = refined.partition != derived.partition
        gap = derived.cell_count - refined.cell_count
        if not separates:
            form = "disappeared"
        elif gap > baseline_gap:
            form = "widens"
        elif gap < baseline_gap:
            form = "narrows"
        elif (
            refined.cell_sizes == found.refined.cell_sizes
            and derived.cell_sizes == found.derived.cell_sizes
        ):
            form = "persists"
        else:
            form = "changes form"
        rows.append(
            {
                "counterfactual": name,
                "frozen_direction": direction,
                "admitted_inputs": len(world.inputs),
                "group_order": derived.group_order,
                "Q_ref_cells": refined.cell_count,
                "Q_ref_cell_sizes": refined.cell_sizes,
                "Q_aut_cells": derived.cell_count,
                "Q_aut_cell_sizes": derived.cell_sizes,
                "separates": separates,
                "Q_ref_is_no_finer_than_Q_aut": _refines(
                    derived.partition, refined.partition
                ),
                "cell_count_gap": gap,
                "gap": form,
            }
        )
    return {
        "baseline_world": found.candidate.identifier,
        "baseline_gap": baseline_gap,
        "baseline_Q_ref_cells": found.refined.cell_count,
        "baseline_Q_aut_cells": found.derived.cell_count,
        "rows": rows,
        "the_gap_survives_at_least_one_source_change": any(
            bool(row["separates"]) for row in rows
        ),
        "relabelling_changed_nothing": next(
            row["Q_ref_cells"] == found.refined.cell_count
            and row["Q_aut_cells"] == found.derived.cell_count
            and row["Q_ref_cell_sizes"] == found.refined.cell_sizes
            and row["Q_aut_cell_sizes"] == found.derived.cell_sizes
            for row in rows
            if row["counterfactual"] == "relabel_every_carrier"
        ),
        "reading": (
            "a gap that only survives the relabelling control is a "
            "serialization accident. A gap that survives erasure, a declared "
            "distinction, or a composition is a property of the typed world"
        ),
    }


# ---------------------------------------------------------------------------
# 10. Structural cost, as a first-class axis
# ---------------------------------------------------------------------------


def cost_report(timings: Sequence[dict[str, object]]) -> dict[str, object]:
    """Section 11. Compilation cost and its scaling inside each family."""
    families: dict[str, list[dict[str, object]]] = {}
    for row in timings:
        families.setdefault(str(row["family"]), []).append(dict(row))
    ratios = [
        float(row["Q_aut_seconds"]) / max(float(row["Q_ref_seconds"]), 1e-9)
        for row in timings
    ]
    worst = max(timings, key=lambda row: float(row["Q_aut_seconds"]))
    return {
        "per_world_columns_required_by_section_11": [
            "schema size as total declared carrier elements",
            "|X_Sigma| as admitted_inputs",
            "declared primitive sizes as declared_primitive_rows",
            "Q_ref rounds",
            "Q_ref time",
            "Q_aut time",
            "Q_ref cell count",
            "Q_aut orbit count",
            "statistical degrees of freedom under each",
        ],
        "per_family_scaling": {
            name: sorted(rows, key=lambda row: (row["admitted_inputs"], row["world"]))
            for name, rows in sorted(families.items())
        },
        "total_Q_ref_seconds": _round(
            sum(float(row["Q_ref_seconds"]) for row in timings)
        ),
        "total_Q_aut_seconds": _round(
            sum(float(row["Q_aut_seconds"]) for row in timings)
        ),
        "worst_single_world_Q_aut_seconds": _round(float(worst["Q_aut_seconds"])),
        "worst_single_world": worst["world"],
        "exact_over_refinement_time_ratio": {
            "median": _round(sorted(ratios)[len(ratios) // 2]),
            "max": _round(max(ratios)),
            "min": _round(min(ratios)),
        },
        "reading": (
            "these are measured wall-clock numbers on the searched sizes and "
            "nothing is extrapolated beyond them. The refinement is polynomial "
            "by construction; the exact compiler searches a product of symmetric "
            "groups and is factorial in carrier cardinality in the worst case, "
            "which is why the declared size bounds exist. Where the exact "
            "compiler looks cheap here it is because sound pruning happens to be "
            "effective on a highly structured world, not because the worst case "
            "improved"
        ),
    }


# ---------------------------------------------------------------------------
# 11. Source fences: nothing on the Q_ref path may mention a target
# ---------------------------------------------------------------------------

COMPILER_PATH: Final = (
    "_slot_pool",
    "_atomic_type",
    "derive_refinement_quotient",
    "_renumber",
    "_canonical_partition",
)

GENERATOR_PATH: Final = (
    "_relation",
    "_operation",
    "_sort",
    "_document",
    "_symmetric_circulant",
    "_directed_circulant",
    "build_circulant_graph",
    "build_bipartite_incidence",
    "reduced_latin_squares",
    "build_quasigroup",
    "build_two_relation_graph",
    "build_corridor",
    "build_composite_product",
    "enumerate_family",
)

FORBIDDEN_TOKENS: Final = (
    "answer",
    "answers",
    "automorphism",
    "expected",
    "label",
    "labels",
    "observation",
    "observations",
    "orbit",
    "orbits",
    "probability",
    "seed",
    "seeds",
    "theta",
    "train",
    "training",
)


def _path_source(names: Sequence[str]) -> str:
    module = sys.modules[__name__]
    return "\n".join(inspect.getsource(getattr(module, name)) for name in names)


def source_fence_audit() -> dict[str, object]:
    """Show mechanically that neither the refinement nor the generator can cheat.

    The refinement path must mention no target vocabulary and no orbit or
    automorphism vocabulary, because a refinement that consulted the exact
    answer would not be a refinement. The generator path must additionally
    mention no property of a generated world, because a family that branched on
    what it was about to produce would not be preregistered.
    """
    refinement = _path_source(COMPILER_PATH)
    generator = _path_source(GENERATOR_PATH)
    both = f"{refinement}\n{generator}"
    tokens = set(re.findall(r"[a-z_0-9]+", both.lower()))
    offending = sorted(word for word in FORBIDDEN_TOKENS if word in tokens)
    imported = sorted(
        name
        for name in sys.modules
        if name.startswith("training_economy") or name.startswith("quotient_refinement")
    )
    return {
        "refinement_path": list(COMPILER_PATH),
        "refinement_path_sha256": _sha256(refinement.encode()),
        "generator_path": list(GENERATOR_PATH),
        "generator_path_sha256": _sha256(generator.encode()),
        "forbidden_tokens_checked": list(FORBIDDEN_TOKENS),
        "forbidden_tokens_found": offending,
        "no_target_or_answer_vocabulary_on_either_path": not offending,
        "no_label_generator_is_reachable": not imported,
        "modules_matching_a_label_bearing_name": imported,
        "why_the_import_graph_is_the_binding_form": (
            "this module imports the standard library and the byte-pinned 017.26 "
            "Phase S module. The 017.26 module does not import the 017.24 "
            "implementation, because that module contains generate_observations. "
            "So no label, seed, estimator, or target probability is reachable "
            "from this file by any path, and that is a property of the import "
            "graph rather than a promise about discipline"
        ),
    }


# ---------------------------------------------------------------------------
# 12. The Phase S freeze
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.28-Task-compiler-scope-refinement-versus-exact-automorphism.md"
        ),
        "prior_result": (
            "quilt+s3://protology#package=occurrence/gpt@"
            f"{PRIOR_QUILT_REVISION}&path=issues/017-jev-pivot/"
            "017.27-Kiro-derived-symmetry-from-typed-world-structure-result.md"
        ),
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "byte_pinned_inputs": {
            "../017.26-Code-attachments/derived_symmetry.py": PHASE_S_01726_SHA256,
            "../017.26-Code-attachments/phase_s_certificate.json": (
                PHASE_S_01726_CERTIFICATE_SHA256
            ),
        },
    }


def _membership(partition: Sequence[int], world: object, prefix: str) -> dict[str, list[str]]:
    cells = _cells_of(partition)
    return {
        f"{prefix}_{cell}": ["".join(world.inputs[index]) for index in sorted(members)]
        for cell, members in sorted(cells.items())
    }


def designated_fixture_block(found: Searched, role: str) -> dict[str, object]:
    """Everything Phase T needs, banked here so it cannot be recomputed later."""
    return {
        "role": role,
        "world": found.candidate.identifier,
        "family": found.candidate.family,
        "parameters": list(map(str, found.candidate.parameters)),
        "position_in_the_frozen_order": found.candidate.position,
        "document": found.candidate.document,
        "document_sha256": _sha256(_canonical(found.candidate.document)),
        "admitted_inputs": [
            "".join(row) for row in found.world.inputs
        ],
        "group_order": found.derived.group_order,
        "Q_aut_cells": found.derived.cell_count,
        "Q_aut_cell_sizes": found.derived.cell_sizes,
        "Q_aut_is_discrete": is_discrete(found.derived.partition),
        "Q_aut_membership": _membership(found.derived.partition, found.world, "A"),
        "Q_ref_cells": found.refined.cell_count,
        "Q_ref_cell_sizes": found.refined.cell_sizes,
        "Q_ref_rounds": found.refined.rounds,
        "Q_ref_membership": _membership(found.refined.partition, found.world, "R"),
        "Q_ref_is_strictly_coarser_than_Q_aut": _strictly_coarser(
            found.refined.partition, found.derived.partition
        ),
        "statistical_degrees_of_freedom": {
            "under_Q_aut": found.derived.cell_count,
            "under_Q_ref": found.refined.cell_count,
            "under_a_saturated_model": found.candidate.admitted_inputs,
        },
    }


PHASE_T_CONTRACT: Final = {
    "runs_only_if": "Phase S found a clean separation",
    "fixtures": (
        "the primary separation fixture, and the graded separation fixture when "
        "it is a different world. Both are designated in this certificate, which "
        "is committed and banked before any target exists"
    ),
    "conditions": {
        "A": "one observable scalar per Q_aut cell; the exact automorphism compiler",
        "R": "one observable scalar per Q_ref cell; the refinement compiler",
        "H": (
            "a handed exact oracle that reads Q_aut cell membership out of this "
            "certificate by a path that never calls a compiler, and that must be "
            "bit-identical to A"
        ),
        "S": "one observable scalar per admitted input; the saturated model",
    },
    "target": (
        "the hidden statistical world must be indexed by Q_aut cells, and "
        "theta_star must be fixed by the rule inherited from 017.24a rather than "
        "chosen, so that no separation can be selected to flatter a condition. No "
        "probability, no seed, and no rule evaluation appears in this Phase S "
        "certificate"
    ),
    "two_losses": {
        "approximation": (
            "the exact irreducible proper-scoring floor of the best "
            "Q_ref-measurable predictor, computed in closed form from cell "
            "weights and theta_star"
        ),
        "estimation": (
            "Q_aut may carry more parameters, so at small n the coarser Q_ref may "
            "win by pooling. The crossover must be measured, and a result where "
            "refinement is better throughout the tested finite-data regime is a "
            "real result and must be reported as one"
        ),
    },
}

REQUIRED_ARTIFACTS: Final = (
    "frozen Q_ref specification -- Q_REF_SPECIFICATION in this module, emitted "
    "verbatim into this certificate",
    "generic Q_ref implementation -- derive_refinement_quotient in this module",
    "exact Q_aut implementation -- the byte-pinned 017.26 compiler",
    "fixture-family generator G -- FAMILIES in this module",
    "preregistered search order, budget, and stop rule -- SEARCH_PROTOCOL",
    "equality replay certificate for both 017.27 fixtures -- section B below",
    "complete search ledger -- section D below",
    "first-separation certificate -- section E below",
    "exact witness pair -- section E below",
    "representation-invariance and determinism tests -- section F below",
    "compiler cost and scaling report -- phase_s_cost_report.json",
    "source-counterfactual report -- section I below",
    "immutable Phase S freeze -- this file, committed and banked before theta*",
)


@dataclass(frozen=True, slots=True)
class Bundle:
    certificate: dict[str, object]
    cost: dict[str, object]


def run_phase_s() -> Bundle:
    """Execute the whole structural phase and assemble the freeze."""
    started = time.perf_counter()

    moment = time.perf_counter()
    certificate_01726 = load_01726_certificate()
    replay = equality_replay()
    replay_seconds = time.perf_counter() - moment

    candidates = enumerate_family()
    skipped = [
        {
            "position": candidate.position,
            "world": candidate.identifier,
            "admitted_inputs": candidate.admitted_inputs,
            "total_carrier_elements": candidate.total_elements,
            "reason": "declared size exceeds a declared bound",
        }
        for candidate in candidates
        if not candidate.within_declared_bounds
    ]

    moment = time.perf_counter()
    searched: list[Searched] = []
    ledger: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    pinned_agreement = 0
    for candidate in candidates:
        if not candidate.within_declared_bounds:
            continue
        found = search_world(candidate)
        pinned, _rounds = exact.pair_refinement(found.world, PINNED_REFINEMENT_CAP)
        if _canonical_partition(pinned) == found.refined.partition:
            pinned_agreement += 1
        searched.append(found)
        ledger.append(ledger_row(found))
        timings.append(timing_row(found))
    search_seconds = time.perf_counter() - moment

    separations = [found for found in searched if found.separates]
    primary = separations[0] if separations else None
    graded = next(
        (
            found
            for found in separations
            if not is_discrete(found.derived.partition)
        ),
        None,
    )

    moment = time.perf_counter()
    first_of_family: dict[str, str] = {}
    for found in searched:
        first_of_family.setdefault(found.candidate.family, found.candidate.identifier)
    audit_worlds = set(first_of_family.values())
    if primary is not None:
        audit_worlds.add(primary.candidate.identifier)
    if graded is not None:
        audit_worlds.add(graded.candidate.identifier)
    invariance = [
        invariance_audit(
            found, include_exact=found.candidate.identifier in audit_worlds
        )
        for found in searched
    ]
    cross_checks = [
        {"world": found.candidate.identifier, **cross_check_exact(found.world, found.derived)}
        for found in searched
        if found.candidate.identifier in audit_worlds
    ]
    audit_seconds = time.perf_counter() - moment

    ladder = [ladder_row(found) for found in searched]

    moment = time.perf_counter()
    witness = (
        separation_witness(primary)
        if primary is not None
        else {"separation_exists": False, "why": "no world in the frozen family separated"}
    )
    graded_witness = (
        separation_witness(graded)
        if graded is not None and graded is not primary
        else None
    )
    counterfactuals = (
        counterfactual_report(primary) if primary is not None else None
    )
    graded_counterfactuals = (
        counterfactual_report(graded)
        if graded is not None and graded is not primary
        else None
    )
    witness_seconds = time.perf_counter() - moment

    fences = source_fence_audit()
    cost = cost_report(timings)

    designated: list[dict[str, object]] = []
    if primary is not None:
        designated.append(designated_fixture_block(primary, "primary separation fixture"))
    if graded is not None and graded is not primary:
        designated.append(designated_fixture_block(graded, "graded separation fixture"))

    checks: dict[str, bool] = {
        "the_017_26_module_bytes_match_the_pin": True,
        "the_017_26_certificate_bytes_match_the_pin": True,
        "the_017_26_certificate_passed_every_check_of_its_own": bool(
            certificate_01726["all_mechanical_checks_pass"]
        ),
        "the_equality_replay_reproduces_Q_ref_equals_Q_aut_on_both_worlds": bool(
            replay["both_worlds_reproduce_equality"]
        ),
        "the_equality_replay_reproduces_the_017_26_banked_quotients": bool(
            replay["both_worlds_reproduce_the_banked_quotient"]
        ),
        "the_equality_replay_reproduces_two_round_convergence": bool(
            replay["both_worlds_reproduce_two_round_convergence"]
        ),
        "this_module_agrees_with_the_pinned_procedure_on_both_replay_worlds": bool(
            replay["this_module_agrees_with_the_pinned_procedure_on_both"]
        ),
        "the_computed_family_order_is_total": family_order_is_total(),
        "every_computed_parameter_order_is_total": all(
            parameter_order_is_total(spec) for spec in FAMILIES
        ),
        "every_family_profile_matches_its_builder": all(
            declared_profile_matches_the_builder(spec) for spec in FAMILIES
        ),
        "every_world_within_the_declared_bounds_was_compiled": len(searched)
        == len(candidates) - len(skipped),
        "the_world_budget_was_not_exceeded": len(candidates) <= MAX_WORLDS,
        "Q_ref_never_split_an_automorphism_orbit": all(
            not found.splits_an_orbit for found in searched
        ),
        "Q_ref_reached_its_fixed_point_on_every_world": all(
            found.refined.reached_fixed_point for found in searched
        ),
        "Q_ref_reached_its_fixed_point_inside_the_pinned_cap": all(
            found.refined.rounds <= PINNED_REFINEMENT_CAP for found in searched
        ),
        "this_module_agrees_with_the_pinned_procedure_on_every_searched_world": (
            pinned_agreement == len(searched)
        ),
        "the_compiler_ladder_holds_on_every_searched_world": all(
            bool(row["ladder_holds"]) for row in ladder
        ),
        "Q_ref_is_relabelling_invariant_on_every_searched_world": all(
            bool(row["Q_ref_is_relabelling_invariant"]) for row in invariance
        ),
        "Q_ref_ignores_every_non_structural_key_on_every_searched_world": all(
            bool(row["Q_ref_ignores_every_non_structural_key"]) for row in invariance
        ),
        "Q_ref_is_deterministic_on_every_searched_world": all(
            bool(row["Q_ref_is_deterministic_within_a_process"]) for row in invariance
        ),
        "Q_aut_is_relabelling_invariant_on_every_audited_world": all(
            bool(row["Q_aut_is_relabelling_invariant"])
            for row in invariance
            if row["exact_compiler_audited_here"]
        ),
        "Q_aut_ignores_every_non_structural_key_on_every_audited_world": all(
            bool(row["Q_aut_ignores_every_non_structural_key"])
            for row in invariance
            if row["exact_compiler_audited_here"]
        ),
        "the_two_exact_routes_agree_on_every_cross_checked_world": all(
            bool(row["the_two_exact_routes_agree"]) for row in cross_checks
        ),
        "every_materialized_element_reverified_independently": all(
            row["every_element_reverified_independently"] is not False
            for row in cross_checks
        ),
        "no_cayley_distance_one_neighbour_escaped": all(
            row["no_neighbour_preserves_every_primitive"] is not False
            for row in cross_checks
        ),
        "no_target_or_answer_vocabulary_on_the_refinement_or_generator_path": bool(
            fences["no_target_or_answer_vocabulary_on_either_path"]
        ),
        "no_label_generator_is_reachable_from_this_module": bool(
            fences["no_label_generator_is_reachable"]
        ),
    }
    if primary is not None:
        checks["the_separation_witness_pair_shares_a_refinement_cell"] = bool(
            witness["witness_pair"]["x_and_y_share_a_refinement_cell"]
        )
        checks["no_structure_preserving_map_sends_the_witness_x_to_y"] = bool(
            witness["witness_pair"]["no_structure_preserving_map_sends_x_to_y"]
        )
        checks["Q_ref_is_strictly_coarser_than_Q_aut_on_the_primary_fixture"] = bool(
            witness["Q_ref_is_strictly_coarser_than_Q_aut"]
        )
        checks["the_relabelling_counterfactual_changed_nothing"] = bool(
            counterfactuals["relabelling_changed_nothing"]
        )
        checks["every_designated_fixture_covers_every_admitted_input"] = all(
            sum(len(members) for members in block["Q_aut_membership"].values())
            == len(block["admitted_inputs"])
            and sum(len(members) for members in block["Q_ref_membership"].values())
            == len(block["admitted_inputs"])
            for block in designated
        )

    if not all(checks.values()):
        raise RuntimeError(
            "Phase S mechanical checks failed: "
            f"{sorted(name for name, ok in checks.items() if not ok)}"
        )

    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "phase": "S",
        "authority": authority_record(),
        "A_compiler_definitions": {
            "Q_ref": Q_REF_SPECIFICATION,
            "Q_aut": {
                "name": "exact automorphism-orbit quotient",
                "definition": (
                    "Q_aut(Sigma) = X_Sigma / Aut(A_Sigma), where Aut(A_Sigma) is "
                    "the group of tuples of typed carrier bijections preserving "
                    "every declared primitive by signature"
                ),
                "implementation": (
                    "the byte-pinned 017.26 compiler, unchanged. Two exact routes "
                    "are available and both are run on every designated world: "
                    "materialize the group and close the admitted inputs under it "
                    "by union-find, or ask the orbit question pair by pair by "
                    "restricted exhaustive search"
                ),
                "why_it_is_the_reference_and_not_the_preferred_compiler": (
                    "Q_aut is the exact reference for this comparison because it "
                    "has an intrinsic mathematical characterization. Nothing here "
                    "presumes it is the compiler a production system should use; "
                    "that is exactly what this turn is measuring"
                ),
                "enumeration_budget": exact.ENUMERATION_BUDGET,
            },
            "the_predicted_relation": (
                "every automorphism preserves every declared fact, so every "
                "atomic type is automorphism-invariant, so Q_ref should be no "
                "finer than Q_aut. That is verified on every searched world "
                "rather than assumed, and a Q_ref that ever split an orbit would "
                "be a compiler defect"
            ),
        },
        "B_equality_replay": replay,
        "C_search_preregistration": {
            "protocol": SEARCH_PROTOCOL,
            "families": [
                {
                    "name": spec.name,
                    "declared_carrier_sorts": spec.sorts,
                    "declared_primitives": spec.primitives,
                    "rationale": spec.rationale,
                    "parameter_count": len(spec.parameters),
                    "parameters": [list(map(str, item)) for item in spec.parameters],
                }
                for spec in FAMILIES
            ],
            "worlds_proposed": len(candidates),
            "worlds_skipped_for_declared_size": skipped,
            "worlds_compiled": len(searched),
            "stated_expectation_before_running": (
                "the parameter ranges and the family list were chosen with an "
                "explicit a-priori hypothesis, stated here rather than "
                "discovered later: the frozen Q_ref reads only the declared "
                "facts of a four-to-eight slot pool, so a world whose declared "
                "local data is constant everywhere is where it should go blind. "
                "A Latin square is the extreme case of that, which is why the "
                "quasigroup family is in G. Stating the hypothesis in the freeze "
                "is preregistration; it is not the same thing as choosing the "
                "instance after seeing the answer, and the ordering rule and the "
                "complete ledger are what make the difference auditable"
            ),
        },
        "D_search_ledger": ledger,
        "E_first_separation": {
            "separations_found": len(separations),
            "primary": witness,
            "graded": graded_witness,
            "designated_for_phase_T": designated,
            "phase_T_contract": PHASE_T_CONTRACT,
            "every_separating_world": [
                {
                    "position": found.candidate.position,
                    "world": found.candidate.identifier,
                    "family": found.candidate.family,
                    "Q_ref_cells": found.refined.cell_count,
                    "Q_aut_cells": found.derived.cell_count,
                    "Q_aut_is_discrete": is_discrete(found.derived.partition),
                }
                for found in separations
            ],
        },
        "F_canonicality": {
            "senses": CANONICALITY_SENSES,
            "rows": invariance,
            "exact_cross_checks": cross_checks,
            "exact_compiler_audit_rule": (
                "the exact compiler's invariance audit and its two-route cross "
                "check are run on the first parameter tuple of every family and "
                "on every designated separation fixture. The rule is declared "
                "rather than chosen per world, and the refinement compiler is "
                "audited on every searched world because it is cheap enough to be"
            ),
            "source_fences": fences,
        },
        "G_structural_cost": {
            "where_the_numbers_live": (
                "every wall-clock measurement is in phase_s_cost_report.json and "
                "none is in this certificate. This certificate is the immutable "
                "structural freeze and section 12 requires it to be byte-identical "
                "on recomputation, which a wall-clock number cannot be. What is "
                "structural about cost is here: the refinement round count, the "
                "atomic-type count, the derived group order, and the degrees of "
                "freedom each compiler leaves to Training"
            ),
            "structural_cost_by_world": [
                {
                    "world": row["world"],
                    "total_carrier_elements": row["total_carrier_elements"],
                    "admitted_inputs": row["admitted_inputs"],
                    "Q_ref_rounds": row["Q_ref_rounds"],
                    "Q_ref_distinct_atomic_types": row["Q_ref_distinct_atomic_types"],
                    "Q_ref_slots_per_pair": row["Q_ref_slots_per_pair"],
                    "group_order": row["group_order"],
                    "Q_aut_group_materialized": row["Q_aut_group_materialized"],
                    "statistical_degrees_of_freedom": row[
                        "statistical_degrees_of_freedom"
                    ],
                }
                for row in ledger
            ],
            "reading": cost["reading"],
        },
        "H_compiler_ladder": {
            "levels": list(LADDER_LEVELS),
            "rows": ladder,
            "the_ladder_is_verified_not_assumed": (
                "every consecutive pair of levels is checked with the refinement "
                "relation on every searched world, and the worlds where a step is "
                "strict are recorded. The ladder is a measured fact about this "
                "declared vocabulary, not a universal total ordering of compilers"
            ),
            "worlds_where_orbits_strictly_refine_the_fixpoint": [
                row["world"] for row in ladder if "orbits_over_fixpoint" in row["strict_at"]
            ],
            "plausible_alternatives_not_implemented": list(PLAUSIBLE_ALTERNATIVES),
        },
        "I_source_counterfactuals": {
            "primary": counterfactuals,
            "graded": graded_counterfactuals,
        },
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "phase_fence": {
            "no_target_probability_exists_in_this_phase": True,
            "no_label_seed_exists_in_this_phase": True,
            "no_estimator_exists_in_this_phase": True,
            "how_that_is_enforced": fences["why_the_import_graph_is_the_binding_form"],
        },
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": all(checks.values()),
        "byte_reproducibility": (
            "this payload contains no wall-clock number and no floating-point "
            "measurement of any kind, so recomputing it must reproduce it "
            "byte-for-byte. --check demands exactly that"
        ),
    }
    cost_payload = {
        "schema": COST_SCHEMA,
        "phase": "S",
        "authority": authority_record(),
        **cost,
        "environment": {"python": sys.version.split()[0]},
        "seconds": {
            "equality_replay": _round(replay_seconds),
            "search": _round(search_seconds),
            "canonicality_audits": _round(audit_seconds),
            "witnesses_and_counterfactuals": _round(witness_seconds),
            "total_wall_clock": _round(time.perf_counter() - started),
        },
    }
    return Bundle(certificate=certificate, cost=cost_payload)


def verify(bundle: Bundle) -> list[str]:
    problems: list[str] = []
    expected = _canonical(bundle.certificate)
    if not CERTIFICATE_PATH.exists():
        problems.append(f"missing artifact: {CERTIFICATE_PATH.name}")
    else:
        observed = CERTIFICATE_PATH.read_bytes()
        if observed != expected:
            problems.append(
                f"{CERTIFICATE_PATH.name} mismatch: observed "
                f"{_sha256(observed)}, expected {_sha256(expected)}"
            )
    if not COST_PATH.exists():
        problems.append(f"missing artifact: {COST_PATH.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute the structure freeze and verify it byte-for-byte",
    )
    parser.add_argument(
        "--write", action="store_true", help="write the structure freeze to disk"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a short human summary of the search instead of the payload",
    )
    args = parser.parse_args(argv)

    bundle = run_phase_s()

    if args.check:
        problems = verify(bundle)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        found = bundle.certificate["E_first_separation"]["separations_found"]
        print(
            "017.28 Phase S verified: certificate "
            f"{_sha256(_canonical(bundle.certificate))[:16]} "
            f"{bundle.certificate['mechanical_check_count']} checks "
            f"{found} separations"
        )
        return 0

    if args.write:
        for path, payload in (
            (CERTIFICATE_PATH, bundle.certificate),
            (COST_PATH, bundle.cost),
        ):
            blob = _canonical(payload)
            path.write_bytes(blob)
            print(f"wrote {path.name} {_sha256(blob)}")
        return 0

    if args.summary:
        certificate = bundle.certificate
        seconds = {
            str(row["world"]): row
            for rows in bundle.cost["per_family_scaling"].values()
            for row in rows
        }
        print(
            "worlds compiled: "
            f"{certificate['C_search_preregistration']['worlds_compiled']}"
        )
        print(f"separations:     {certificate['E_first_separation']['separations_found']}")
        for row in certificate["D_search_ledger"]:
            timing = seconds[str(row["world"])]
            flag = "SEPARATES" if row["separates"] else "equal"
            print(
                f"  {row['position']:3d} {row['world']:44s} "
                f"|X|={row['admitted_inputs']:4d} "
                f"|Aut|={row['group_order']:>12} "
                f"ref={row['Q_ref_cells']:3d} aut={row['Q_aut_cells']:3d} "
                f"t_ref={timing['Q_ref_seconds']:8.4f} "
                f"t_aut={timing['Q_aut_seconds']:9.4f} {flag}"
            )
        return 0

    sys.stdout.buffer.write(_canonical(bundle.certificate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
