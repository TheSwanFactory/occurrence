"""Tests for the 017.28 Phase S compiler-scope module.

The tests that matter most here are the independent ones. The central claim of
the turn is that two specific admitted inputs of one specific world share a
refinement cell and lie in different automorphism orbits, so the orbit claim is
re-derived by an algorithm that shares no code with the compiler: for a Latin
square, a choice of row and column bijection forces the symbol bijection, so the
whole autotopism group can be enumerated directly from the square without any
backtracking search at all.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "compiler_scope.py"

_spec = importlib.util.spec_from_file_location("compiler_scope_under_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
scope = importlib.util.module_from_spec(_spec)
sys.modules["compiler_scope_under_test"] = scope
_spec.loader.exec_module(scope)

exact = scope.exact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Pins and fences
# ---------------------------------------------------------------------------


def test_the_017_26_module_pin_holds() -> None:
    assert _sha256(scope.PHASE_S_01726_PATH) == scope.PHASE_S_01726_SHA256


def test_the_017_26_certificate_pin_holds() -> None:
    assert (
        _sha256(scope.PHASE_S_01726_CERTIFICATE_PATH)
        == scope.PHASE_S_01726_CERTIFICATE_SHA256
    )


def test_no_label_generator_is_reachable_in_a_clean_process() -> None:
    """The fence is about this module's import graph, so isolate the process.

    POST-FREEZE CORRECTION to this test file only, disclosed in the result. The
    module's ``source_fence_audit`` scans ``sys.modules``, which is process-global.
    Running the Phase T tests in the same pytest process legitimately loads the
    017.24 label generator -- Phase T is supposed to -- and that would make an
    in-process assertion here fail for a reason that has nothing to do with Phase
    S. Spawning a clean interpreter that loads only ``compiler_scope`` is the
    check that was always intended, and it is strictly stronger: it shows that
    importing Phase S *by itself* pulls in no label generator.
    ``compiler_scope.py`` and ``phase_s_certificate.json`` are untouched and
    their pins still hold.
    """
    import subprocess
    import textwrap

    program = textwrap.dedent(
        f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("only_scope", {str(MODULE_PATH)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules["only_scope"] = module
        spec.loader.exec_module(module)
        audit = module.source_fence_audit()
        print(json.dumps({{
            "reachable": audit["no_label_generator_is_reachable"],
            "matching": audit["modules_matching_a_label_bearing_name"],
            "tokens": audit["forbidden_tokens_found"],
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    audit = json.loads(completed.stdout.strip().splitlines()[-1])
    assert audit["reachable"]
    assert audit["matching"] == []
    assert audit["tokens"] == []


def test_the_banked_certificate_recorded_a_clean_fence() -> None:
    """The freeze was written by a clean process, and it says so."""
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    fences = certificate["F_canonicality"]["source_fences"]
    assert fences["no_label_generator_is_reachable"]
    assert fences["modules_matching_a_label_bearing_name"] == []


def test_no_target_vocabulary_on_the_refinement_or_generator_path() -> None:
    audit = scope.source_fence_audit()
    assert audit["forbidden_tokens_found"] == []
    assert audit["no_target_or_answer_vocabulary_on_either_path"]


def test_the_pinned_017_26_module_imports_no_label_generator() -> None:
    """The fence is the import graph, so the import graph is what is inspected.

    The 017.26 module's prose does mention ``generate_observations`` -- it
    explains at length that it does not import it -- so a text scan would be the
    wrong test. The right test parses the module and looks at what it actually
    imports.
    """
    import ast

    tree = ast.parse(scope.PHASE_S_01726_PATH.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert imported
    assert not any(
        name.startswith(("training_economy", "quotient_refinement"))
        for name in imported
    )
    # And it loads nothing dynamically either, so the static import list is the
    # whole import graph.
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "spec_from_file_location"
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# The frozen generator family
# ---------------------------------------------------------------------------


def test_the_specification_defines_every_required_item() -> None:
    for index in range(1, 10):
        assert any(
            key.startswith(f"{index}_") for key in scope.Q_REF_SPECIFICATION
        ), f"section 4 item {index} is not defined"


def test_the_computed_family_order_is_total() -> None:
    assert scope.family_order_is_total()


def test_every_computed_parameter_order_is_total() -> None:
    for spec in scope.FAMILIES:
        assert scope.parameter_order_is_total(spec), spec.name


def test_every_family_profile_matches_its_builder() -> None:
    for spec in scope.FAMILIES:
        assert scope.declared_profile_matches_the_builder(spec), spec.name


def test_the_family_order_is_the_rule_applied_to_the_table() -> None:
    expected = sorted(
        (spec.primitives, spec.sorts, spec.name) for spec in scope._FAMILY_TABLE
    )
    assert [
        (spec.primitives, spec.sorts, spec.name) for spec in scope.FAMILIES
    ] == expected


def test_the_family_draws_no_randomness() -> None:
    """Prose may say "no random draw"; the code may not make one."""
    import ast
    import re

    source = scope._path_source(scope.GENERATOR_PATH)
    assert not re.search(r"\brandom\b\s*\.", source)
    assert not re.search(r"\b(shuffle|sample|randrange|randint|choice)\s*\(", source)
    tree = ast.parse(MODULE_PATH.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "random" not in imported


def test_every_proposed_world_is_within_the_declared_bounds() -> None:
    candidates = scope.enumerate_family()
    assert candidates
    assert len(candidates) <= scope.MAX_WORLDS
    for candidate in candidates:
        assert candidate.admitted_inputs <= scope.MAX_ADMITTED_INPUTS
        assert candidate.total_elements <= scope.MAX_TOTAL_CARRIER_ELEMENTS


def test_every_proposed_world_loads_as_a_typed_world() -> None:
    for candidate in scope.enumerate_family():
        world = exact.world_from_document(candidate.document, candidate.identifier)
        assert len(world.inputs) == candidate.admitted_inputs
        assert world.primitives


@pytest.mark.parametrize(
    ("order", "total"), [(3, 1), (4, 4), (5, 56)]
)
def test_reduced_latin_square_counts_are_the_known_ones(order: int, total: int) -> None:
    assert len(scope.reduced_latin_squares(order, 10_000)) == total


def test_every_generated_square_is_a_reduced_latin_square() -> None:
    for order in (4, 5, 6):
        for square in scope.reduced_latin_squares(order, 6):
            assert square[0] == tuple(range(order))
            assert tuple(row[0] for row in square) == tuple(range(order))
            for row in square:
                assert sorted(row) == list(range(order))
            for column in range(order):
                assert sorted(row[column] for row in square) == list(range(order))


def test_reduced_latin_squares_are_produced_in_lexicographic_order() -> None:
    flattened = [
        tuple(value for row in square for value in row)
        for square in scope.reduced_latin_squares(6, 6)
    ]
    assert flattened == sorted(flattened)


# ---------------------------------------------------------------------------
# Compiler R on worlds outside the frozen family
# ---------------------------------------------------------------------------

OFF_FAMILY = (
    # a four-cycle, the smallest world the 017.26 tests also use
    {
        "sorts": [{"name": "V", "elements": ["a", "b", "c", "d"]}],
        "primitives": [
            {
                "name": "e",
                "kind": "relation",
                "signature": ["V", "V"],
                "extension": [
                    ["a", "b"],
                    ["b", "a"],
                    ["b", "c"],
                    ["c", "b"],
                    ["c", "d"],
                    ["d", "c"],
                    ["d", "a"],
                    ["a", "d"],
                ],
            }
        ],
        "admitted_inputs": {"product": ["V", "V"]},
    },
    # a three-carrier world with a typed map and a constant
    {
        "sorts": [
            {"name": "A", "elements": ["a0", "a1", "a2", "a3"]},
            {"name": "B", "elements": ["b0", "b1"]},
        ],
        "primitives": [
            {
                "name": "f",
                "kind": "operation",
                "domain": ["A"],
                "codomain": "B",
                "graph": [
                    {"arguments": ["a0"], "value": "b0"},
                    {"arguments": ["a1"], "value": "b0"},
                    {"arguments": ["a2"], "value": "b1"},
                    {"arguments": ["a3"], "value": "b1"},
                ],
            },
            {"name": "k", "kind": "constant", "sort": "A", "value": "a0"},
        ],
        "admitted_inputs": {"product": ["A", "B"]},
    },
)


@pytest.mark.parametrize("document", OFF_FAMILY)
def test_refinement_agrees_with_the_pinned_procedure_off_family(document: dict) -> None:
    world = exact.world_from_document(document, "off-family")
    mine = scope.derive_refinement_quotient(world)
    pinned, _rounds = exact.pair_refinement(world)
    assert scope._canonical_partition(pinned) == mine.partition


@pytest.mark.parametrize("document", OFF_FAMILY)
def test_refinement_is_no_finer_than_orbits_off_family(document: dict) -> None:
    world = exact.world_from_document(document, "off-family")
    mine = scope.derive_refinement_quotient(world)
    derived = scope.derive_exact_quotient(world)
    assert exact._refines(derived.partition, mine.partition)


@pytest.mark.parametrize("document", OFF_FAMILY)
def test_refinement_is_relabelling_invariant_off_family(document: dict) -> None:
    world = exact.world_from_document(document, "off-family")
    renamed = exact.world_from_document(
        exact.relabel_document(document), "off-family-renamed"
    )
    assert (
        scope.derive_refinement_quotient(renamed).partition
        == scope.derive_refinement_quotient(world).partition
    )


@pytest.mark.parametrize("document", OFF_FAMILY)
def test_refinement_is_deterministic_off_family(document: dict) -> None:
    world = exact.world_from_document(document, "off-family")
    first = scope.derive_refinement_quotient(world)
    second = scope.derive_refinement_quotient(world)
    assert first.partition == second.partition
    assert first.rounds == second.rounds
    assert first.distinct_atomic_types == second.distinct_atomic_types


def test_the_atomic_type_mentions_no_name_and_no_element() -> None:
    world = exact.world_from_document(OFF_FAMILY[1], "off-family")
    renamed = exact.world_from_document(
        exact.relabel_document(OFF_FAMILY[1]), "off-family-renamed"
    )
    for left in range(len(world.inputs)):
        for right in range(len(world.inputs)):
            assert scope._atomic_type(
                world, world.inputs[left], world.inputs[right]
            ) == scope._atomic_type(
                renamed, renamed.inputs[left], renamed.inputs[right]
            )


def test_the_ladder_holds_off_family() -> None:
    for document in OFF_FAMILY:
        world = exact.world_from_document(document, "off-family")
        refined = scope.derive_refinement_quotient(world)
        derived = scope.derive_exact_quotient(world)
        assert exact._refines(refined.bounded_partition, refined.atomic_partition)
        assert exact._refines(refined.partition, refined.bounded_partition)
        assert exact._refines(derived.partition, refined.partition)


def test_canonical_partition_sorts_by_size_then_least_member() -> None:
    assert scope._canonical_partition([5, 5, 5, 9, 9, 2]) == (2, 2, 2, 1, 1, 0)
    assert scope._canonical_partition([0, 1, 2]) == (0, 1, 2)


def test_both_exact_routes_agree_off_family() -> None:
    for document in OFF_FAMILY:
        world = exact.world_from_document(document, "off-family")
        derived = scope.derive_exact_quotient(world)
        report = scope.cross_check_exact(world, derived)
        assert report["the_two_exact_routes_agree"]
        assert report["every_element_reverified_independently"]


# ---------------------------------------------------------------------------
# The equality replay
# ---------------------------------------------------------------------------


def test_the_equality_replay_reproduces_the_017_27_numbers() -> None:
    replay = scope.equality_replay()
    assert replay["both_worlds_reproduce_equality"]
    assert replay["both_worlds_reproduce_the_banked_quotient"]
    assert replay["both_worlds_reproduce_two_round_convergence"]
    assert replay["this_module_agrees_with_the_pinned_procedure_on_both"]
    by_world = {row["world"]: row for row in replay["fixtures"]}
    assert by_world["A"]["group_order"] == 14
    assert by_world["A"]["Q_aut_cell_sizes"] == [7, 14, 14, 14]
    assert by_world["A"]["Q_ref_cell_sizes"] == [7, 14, 14, 14]
    assert by_world["B"]["group_order"] == 6
    assert by_world["B"]["Q_aut_cell_sizes"] == [3, 6, 6, 6, 6]
    assert by_world["B"]["Q_ref_cell_sizes"] == [3, 6, 6, 6, 6]


# ---------------------------------------------------------------------------
# The separation, verified by an algorithm that shares no code with the compiler
# ---------------------------------------------------------------------------


def autotopism_group(square: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Enumerate the autotopism group of a Latin square, without any search.

    An autotopism is a triple of bijections (alpha, beta, gamma) with
    ``gamma(L(r, c)) = L(alpha(r), beta(c))``. Row 0 of a Latin square contains
    every symbol exactly once, so a choice of alpha and beta *forces* gamma
    outright; the rest is verification. This shares no code with the backtracking
    compiler under test, so agreement between them is a real check.
    """
    order = len(square)
    found: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for alpha in itertools.permutations(range(order)):
        for beta in itertools.permutations(range(order)):
            gamma = [-1] * order
            for column in range(order):
                gamma[square[0][column]] = square[alpha[0]][beta[column]]
            if -1 in gamma or len(set(gamma)) != order:
                continue
            if all(
                gamma[square[row][column]] == square[alpha[row]][beta[column]]
                for row in range(1, order)
                for column in range(order)
            ):
                found.append((alpha, beta, tuple(gamma)))
    return found


def test_the_primary_separation_is_independently_verified() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    primary = certificate["E_first_separation"]["primary"]
    assert primary["separation_exists"]
    assert primary["world"] == "quasigroup:n=6:square=1"

    order, index = 6, 1
    square = scope.reduced_latin_squares(order, index + 1)[index]
    group = autotopism_group(square)
    assert len(group) == primary["group_order"]

    cells = [(row, column) for row in range(order) for column in range(order)]
    parent = {cell: cell for cell in cells}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for alpha, beta, _gamma in group:
        for row, column in cells:
            left, right = find((row, column)), find((alpha[row], beta[column]))
            if left != right:
                parent[max(left, right)] = min(left, right)
    orbits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        orbits.setdefault(find(cell), []).append(cell)
    assert sorted(len(members) for members in orbits.values()) == primary[
        "Q_aut_cell_sizes"
    ]
    assert len(orbits) == primary["Q_aut_cell_count"]

    witness = primary["witness_pair"]
    assert witness["x"] == "r0c0"
    assert witness["y"] == "r0c2"
    assert find((0, 0)) != find((0, 2))
    assert sorted(orbits[find((0, 0))]) == sorted(
        (int(name[1]), int(name[3])) for name in witness["Q_aut_orbit_of_x_members"]
    )


def test_the_primary_separation_refinement_cell_is_everything() -> None:
    document = scope.build_quasigroup(6, 1)
    world = exact.world_from_document(document, "primary")
    refined = scope.derive_refinement_quotient(world)
    assert refined.cell_count == 1
    assert refined.cell_sizes == [36]
    derived = scope.derive_exact_quotient(world)
    assert derived.cell_count == 2
    assert exact._refines(derived.partition, refined.partition)
    assert not exact._refines(refined.partition, derived.partition)


def test_the_orbit_distinction_is_the_row_block_column_block_agreement() -> None:
    """Name the invariant the refinement could not see, and check it is the orbit.

    This is diagnostic rather than load-bearing. The orbit partition is derived
    from declared structure alone; this test only records that on this square the
    two orbits are exactly the cells whose row pair and column pair agree, which
    is what makes the separation legible.
    """
    document = scope.build_quasigroup(6, 1)
    world = exact.world_from_document(document, "primary")
    derived = scope.derive_exact_quotient(world)
    position = {"".join(row): index for index, row in enumerate(world.inputs)}
    blocks = {
        derived.partition[position[f"r{row}c{column}"]]: (row // 2 == column // 2)
        for row in range(6)
        for column in range(6)
    }
    assert len(blocks) == 2
    assert set(blocks.values()) == {True, False}
    for row in range(6):
        for column in range(6):
            cell = derived.partition[position[f"r{row}c{column}"]]
            assert blocks[cell] == (row // 2 == column // 2)


def test_every_cell_of_the_primary_square_lies_in_exactly_one_intercalate() -> None:
    """The reason the refinement is blind, stated as a measured fact.

    The only non-trivial declared fact the atomic type of a pair of cells can
    report on a Latin square is whether the two cells span a 2x2 Latin subsquare.
    On this square that count is 1 for every cell, so the whole admitted-input
    set carries one atomic profile and the fixpoint is reached in one round with
    one cell.
    """
    square = scope.reduced_latin_squares(6, 2)[1]
    counts = set()
    for row in range(6):
        for column in range(6):
            counts.add(
                sum(
                    1
                    for other_row in range(6)
                    for other_column in range(6)
                    if other_row != row
                    and other_column != column
                    and square[row][column] == square[other_row][other_column]
                    and square[row][other_column] == square[other_row][column]
                )
            )
    assert counts == {1}


# ---------------------------------------------------------------------------
# The committed freeze
# ---------------------------------------------------------------------------


def test_the_committed_certificate_passes_its_own_checks() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    assert certificate["schema"] == scope.CERTIFICATE_SCHEMA
    assert certificate["phase"] == "S"
    assert certificate["all_mechanical_checks_pass"]
    assert certificate["mechanical_check_count"] >= 30
    assert all(certificate["mechanical_checks"].values())


FORBIDDEN_CERTIFICATE_KEYS = (
    "theta_star",
    "theta",
    "lambda",
    "label_seed",
    "base_seed",
    "estimator",
    "target_probability",
    "asymptotic_floor",
    "excess_nll",
)


def _keys(payload: object) -> list[str]:
    if isinstance(payload, dict):
        found: list[str] = []
        for key, value in payload.items():
            found.append(str(key))
            found.extend(_keys(value))
        return found
    if isinstance(payload, list):
        return [name for item in payload for name in _keys(item)]
    return []


def test_the_committed_certificate_declares_no_target() -> None:
    """Phase S must contain no target. Checked on keys, not on prose.

    The certificate necessarily *names* the vocabulary it forbids -- the source
    fence publishes its own token list -- so scanning the text for the word
    "probability" would fail on the fence itself. What must be absent is a key
    that carries a target, a seed, an estimator, or a loss.
    """
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    keys = {name.lower() for name in _keys(certificate)}
    for forbidden in FORBIDDEN_CERTIFICATE_KEYS:
        assert forbidden not in keys, forbidden
    fence = certificate["phase_fence"]
    assert fence["no_target_probability_exists_in_this_phase"]
    assert fence["no_label_seed_exists_in_this_phase"]
    assert fence["no_estimator_exists_in_this_phase"]


def test_the_ledger_covers_every_compiled_world() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    ledger = certificate["D_search_ledger"]
    preregistration = certificate["C_search_preregistration"]
    assert len(ledger) == preregistration["worlds_compiled"]
    assert [row["position"] for row in ledger] == sorted(
        row["position"] for row in ledger
    )
    assert all(row["Q_ref_is_no_finer_than_Q_aut"] for row in ledger)
    assert not any(row["Q_ref_splits_an_automorphism_orbit"] for row in ledger)


def test_the_first_separation_is_the_first_in_the_frozen_order() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    ledger = certificate["D_search_ledger"]
    separating = [row["position"] for row in ledger if row["separates"]]
    primary = certificate["E_first_separation"]["primary"]
    by_position = {row["position"]: row for row in ledger}
    assert by_position[min(separating)]["world"] == primary["world"]


def test_the_designated_fixture_membership_round_trips() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    for block in certificate["E_first_separation"]["designated_for_phase_T"]:
        world = exact.world_from_document(block["document"], block["world"])
        names = ["".join(row) for row in world.inputs]
        assert names == block["admitted_inputs"]
        for prefix, expected in (
            ("A", scope.derive_exact_quotient(world).partition),
            ("R", scope.derive_refinement_quotient(world).partition),
        ):
            membership = block[f"Q_{'aut' if prefix == 'A' else 'ref'}_membership"]
            rebuilt = [-1] * len(names)
            for key, members in membership.items():
                for member in members:
                    rebuilt[names.index(member)] = int(key.split("_")[1])
            assert tuple(rebuilt) == tuple(expected)


def test_the_counterfactual_relabelling_control_changed_nothing() -> None:
    certificate = json.loads(scope.CERTIFICATE_PATH.read_text())
    report = certificate["I_source_counterfactuals"]["primary"]
    assert report["relabelling_changed_nothing"]
    rows = {row["counterfactual"]: row for row in report["rows"]}
    assert rows["relabel_every_carrier"]["Q_ref_cells"] == report["baseline_Q_ref_cells"]
    assert rows["relabel_every_carrier"]["Q_aut_cells"] == report["baseline_Q_aut_cells"]


def test_the_cost_report_exists_and_is_consistent() -> None:
    cost = json.loads(scope.COST_PATH.read_text())
    assert cost["schema"] == scope.COST_SCHEMA
    assert cost["total_Q_ref_seconds"] > 0
    assert cost["total_Q_aut_seconds"] > 0
    assert cost["per_family_scaling"].keys() == {
        spec.name for spec in scope.FAMILIES
    }
