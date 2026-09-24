"""Adversarial tests for the Issue 017.26 Phase S structural freeze.

Written at the freeze, before any target probability or label exists. Everything
except the certificate test is unconditional, so the freeze commit is green
without pretending to have a result.

The claims this phase lives or dies by are checked independently of the module's
own audit.

Both derived group orders are re-derived by a second algorithm written here and
not shared with the compiler: for the single-relation fixture, an exhaustive walk
over the larger carrier's permutations with the other carrier's image recovered by
matching declared row sets; for the multi-primitive fixture, an exhaustive walk
over one carrier's permutations with the other carrier's image forced by the
declared total map.

Genericity is tested behaviourally rather than by reading the source. Three worlds
that do not exist anywhere in the repository are built here and handed to the same
compiler, including one that exercises a primitive kind the committed fixtures
only reach through a perturbation. If the compiler were recognizing the committed
fixtures, these would fail.

And every feature of the frozen shortcut language is checked to be invariant under
every derived structure-preserving map, which is the fact that makes "L_k does not
recover q_Sigma" mean "L_k is strictly coarser" rather than "L_k happens to
disagree".
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "derived_symmetry.py"
SPEC = importlib.util.spec_from_file_location("derived_symmetry_01726", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
phase_s = importlib.util.module_from_spec(SPEC)
sys.modules["derived_symmetry_01726"] = phase_s
SPEC.loader.exec_module(phase_s)


@pytest.fixture(scope="module")
def world_a():
    return phase_s.load_world_a()


@pytest.fixture(scope="module")
def world_b():
    return phase_s.load_world_b()


@pytest.fixture(scope="module")
def compiled_a(world_a):
    return phase_s.compile_world(world_a[0])


@pytest.fixture(scope="module")
def compiled_b(world_b):
    return phase_s.compile_world(world_b[0])


def _relation(world, name):
    for primitive in world.primitives:
        if primitive.name == name:
            return primitive
    raise KeyError(name)


def _count_injections(keys, candidates):
    """Count injective choices of ``key -> candidate``, by plain backtracking."""
    order = sorted(keys, key=lambda key: (len(candidates[key]), key))

    def walk(index, used):
        if index == len(order):
            return 1
        total = 0
        for image in candidates[order[index]]:
            if image in used:
                continue
            total += walk(index + 1, used | {image})
        return total

    return walk(0, frozenset())


# --- pins -------------------------------------------------------------------


def test_both_world_documents_are_byte_pinned(world_a, world_b):
    assert world_a[1]["schema"] == phase_s.WORLD_A_SCHEMA
    assert world_b[1]["schema"] == phase_s.WORLD_B_SCHEMA
    assert len(world_a[0].inputs) == 49
    assert len(world_b[0].inputs) == 27


def test_a_changed_world_document_is_refused(tmp_path):
    for path, loader in (
        (phase_s.WORLD_A_PATH, phase_s.load_world_a),
        (phase_s.WORLD_B_PATH, phase_s.load_world_b),
    ):
        corrupted = tmp_path / path.name
        corrupted.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(RuntimeError):
            loader(corrupted)


def test_the_committed_digests_are_the_literal_expected_strings():
    assert phase_s.WORLD_A_SHA256 == (
        "9f5262583012fb01aa4a9508d3abe3bfd9bb8135bd172d8342dc755004e7fc67"
    )
    assert phase_s.WORLD_B_SHA256 == (
        "79412d9349c9413831bb44cae6bcd332ae265a6928facb2d7a7ec161c0b32b5d"
    )


# --- the source fence -------------------------------------------------------


def test_neither_world_declares_any_symmetry_or_equivalence_metadata(
    world_a, world_b
):
    for _world, document in (world_a, world_b):
        audit = phase_s.forbidden_metadata_audit(document)
        assert audit["structural_subtree_is_clean"], audit[
            "forbidden_vocabulary_found_in_the_structural_subtree"
        ]
        assert audit["no_numeric_probability_can_hide_in_the_structure"]
        assert audit["every_top_level_key_is_whitelisted"]
        assert audit["denial_covers_symmetry_and_equivalence"]


def test_the_structural_subtree_carries_only_carriers_primitives_and_the_input_type(
    world_a, world_b
):
    allowed = {
        "relation": {"name", "kind", "signature", "description", "extension"},
        "operation": {
            "name",
            "kind",
            "domain",
            "codomain",
            "description",
            "graph",
        },
        "constant": {"name", "kind", "sort", "value", "description"},
    }
    for _world, document in (world_a, world_b):
        for block in document["sorts"]:
            assert set(block) <= {"name", "elements", "description"}
        for block in document["primitives"]:
            assert set(block) <= allowed[block["kind"]], block["name"]
        assert set(document["admitted_inputs"]) <= {
            "product",
            "description",
            "cardinality",
        }


def test_deleting_every_non_structural_key_changes_nothing(
    world_a, world_b, compiled_a, compiled_b
):
    for (_world, document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        stripped = phase_s.world_from_document(
            phase_s.strip_to_structure(document), "stripped"
        )
        again = phase_s.compile_world(stripped)
        assert again.group_order == compiled.group_order
        assert again.quotient == compiled.quotient


def test_renaming_every_declared_name_changes_nothing(
    world_a, world_b, compiled_a, compiled_b
):
    for (_world, document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        relabelled = phase_s.world_from_document(
            phase_s.relabel_document(document), "relabelled"
        )
        again = phase_s.compile_world(relabelled)
        assert again.group_order == compiled.group_order
        assert again.quotient == compiled.quotient


def test_reordering_the_document_keys_changes_nothing(world_b, compiled_b):
    document = json.loads(json.dumps(world_b[1]))
    document["primitives"] = list(reversed(document["primitives"]))
    again = phase_s.compile_world(
        phase_s.world_from_document(document, "reordered")
    )
    assert again.group_order == compiled_b.group_order
    assert again.quotient == compiled_b.quotient


# --- the derived group, re-derived by a second algorithm --------------------


def test_the_single_relation_group_order_is_confirmed_by_row_set_matching(
    world_a, compiled_a
):
    """Exhaust the larger carrier and recover the other by matching row sets.

    This is the 017.24-era algorithm, written out here so that the committed
    order does not rest on the generic search alone. It walks all 5040
    permutations of one carrier and, for each, counts the injective relabellings
    of the other carrier consistent with the declared rows.
    """
    world = world_a[0]
    left = world.carrier("Point").elements
    right = world.carrier("Line").elements
    extension = _relation(world, "incident").extension
    column = {
        element: frozenset(a for a, b in extension if b == element)
        for element in right
    }
    by_column: dict[frozenset[str], list[str]] = {}
    for element, members in column.items():
        by_column.setdefault(members, []).append(element)

    total = 0
    for permutation in itertools.permutations(left):
        mapping = dict(zip(left, permutation, strict=True))
        candidates: dict[str, list[str]] = {}
        feasible = True
        for element, members in column.items():
            image = frozenset(mapping[member] for member in members)
            matches = by_column.get(image)
            if not matches:
                feasible = False
                break
            candidates[element] = matches
        if feasible:
            total += _count_injections(list(right), candidates)
    assert total == 14
    assert compiled_a.group_order == total


def test_the_multi_primitive_group_order_is_confirmed_by_forcing_the_map(
    world_b, compiled_b
):
    """Exhaust the larger carrier; the smaller carrier's image is then forced.

    Because the declared total map is onto, a relabelling of the larger carrier
    determines at most one relabelling of the smaller one, so walking all 362,880
    permutations of the larger carrier is exhaustive over the whole product of
    symmetric groups without assuming anything the compiler assumes.
    """
    world = world_b[0]
    sites = world.carrier("Site").elements
    extension = _relation(world, "adjacent").extension
    operation = next(
        primitive for primitive in world.primitives if primitive.kind == "operation"
    )
    assignment = {arguments[0]: value for arguments, value in operation.graph}

    total = 0
    for permutation in itertools.permutations(sites):
        mapping = dict(zip(sites, permutation, strict=True))
        if {(mapping[a], mapping[b]) for a, b in extension} != set(extension):
            continue
        induced: dict[str, str] = {}
        consistent = True
        for site in sites:
            source = assignment[site]
            target = assignment[mapping[site]]
            if induced.setdefault(source, target) != target:
                consistent = False
                break
        if consistent and len(set(induced.values())) == len(induced):
            total += 1
    assert total == 6
    assert compiled_b.group_order == total


def test_the_derived_cell_structure_is_what_the_freeze_reports(
    compiled_a, compiled_b
):
    assert compiled_a.cell_sizes == [7, 14, 14, 14]
    assert compiled_b.cell_sizes == [3, 6, 6, 6, 6]
    assert sum(compiled_a.cell_sizes) == 49
    assert sum(compiled_b.cell_sizes) == 27


def test_the_two_routes_to_the_quotient_agree(world_a, world_b, compiled_a, compiled_b):
    for (world, _document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        plan = phase_s.prepare(world)
        assert compiled.morphisms is not None
        assert (
            phase_s.partition_from_group(plan, compiled.morphisms)
            == compiled.quotient
        )


def test_every_derived_element_preserves_every_primitive(
    world_a, world_b, compiled_a, compiled_b
):
    for (world, _document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        assert compiled.morphisms is not None
        for morphism in compiled.morphisms:
            assert phase_s.preserves_every_primitive(world, morphism)


def test_the_derived_group_has_no_structure_preserving_neighbour(
    world_a, world_b, compiled_a, compiled_b
):
    for (world, _document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        witness = phase_s.rejection_witness(world, compiled, samples=2000)
        assert witness["exhaustive_neighbourhood_scan"][
            "the_derived_group_has_no_structure_preserving_neighbour"
        ]


def test_unpruned_brute_force_agrees_on_a_reduced_world(world_a, world_b):
    for (world, _document) in (world_a, world_b):
        sizes = phase_s.REDUCTION_SIZES[world.identifier]
        witness = phase_s.brute_force_witness(world, sizes)
        assert witness["orders_agree"], witness
        assert witness["element_sets_agree"], witness


# --- genericity, tested on worlds that exist nowhere else -------------------


def _fresh_cycle_world(size, name="Vertex", constant=None):
    elements = [f"{name}{index}" for index in range(size)]
    extension = [
        [elements[index], elements[(index + 1) % size]] for index in range(size)
    ] + [[elements[(index + 1) % size], elements[index]] for index in range(size)]
    primitives = [
        {
            "name": "ring",
            "kind": "relation",
            "signature": [name, name],
            "extension": extension,
        }
    ]
    if constant is not None:
        primitives.append(
            {
                "name": "root",
                "kind": "constant",
                "sort": name,
                "value": elements[constant],
            }
        )
    return {
        "sorts": [{"name": name, "elements": elements}],
        "primitives": primitives,
        "admitted_inputs": {"product": [name, name]},
    }


def test_the_compiler_handles_a_world_it_has_never_seen():
    """A four-cycle the repository does not contain. Aut is dihedral of order 8."""
    world = phase_s.world_from_document(_fresh_cycle_world(4), "fresh_c4")
    compiled = phase_s.compile_world(world)
    assert compiled.group_order == 8
    assert compiled.cell_count == 3
    assert sum(compiled.cell_sizes) == 16


def test_the_compiler_handles_a_fresh_world_with_a_declared_constant():
    """Distinguishing one element cuts the dihedral group to a reflection pair."""
    plain = phase_s.compile_world(
        phase_s.world_from_document(_fresh_cycle_world(4), "fresh_c4")
    )
    pointed = phase_s.compile_world(
        phase_s.world_from_document(
            _fresh_cycle_world(4, constant=0), "fresh_c4_pointed"
        )
    )
    assert pointed.group_order == 2
    assert pointed.group_order < plain.group_order
    assert phase_s._refines(pointed.quotient, plain.quotient)


def test_the_compiler_handles_a_fresh_world_with_three_carriers_and_an_operation():
    """A fresh signature shape: three carriers, a relation, and a typed map."""
    document = {
        "sorts": [
            {"name": "Left", "elements": ["l0", "l1", "l2", "l3"]},
            {"name": "Right", "elements": ["r0", "r1"]},
            {"name": "Tag", "elements": ["t0", "t1"]},
        ],
        "primitives": [
            {
                "name": "pairs",
                "kind": "relation",
                "signature": ["Left", "Right"],
                "extension": [
                    ["l0", "r0"],
                    ["l1", "r0"],
                    ["l2", "r1"],
                    ["l3", "r1"],
                ],
            },
            {
                "name": "tag",
                "kind": "operation",
                "domain": ["Left"],
                "codomain": "Tag",
                "graph": [
                    {"arguments": ["l0"], "value": "t0"},
                    {"arguments": ["l1"], "value": "t1"},
                    {"arguments": ["l2"], "value": "t0"},
                    {"arguments": ["l3"], "value": "t1"},
                ],
            },
        ],
        "admitted_inputs": {"product": ["Left", "Tag"]},
    }
    world = phase_s.world_from_document(document, "fresh_three_carrier")
    compiled = phase_s.compile_world(world)
    candidates = phase_s._all_candidates(world)
    accepted = [
        morphism
        for morphism in candidates
        if phase_s.preserves_every_primitive(world, morphism)
    ]
    assert compiled.group_order == len(accepted)
    assert compiled.group_order > 1
    assert 1 < compiled.cell_count < len(world.inputs)


def test_an_undeclared_primitive_kind_is_refused():
    document = _fresh_cycle_world(4)
    document["primitives"][0]["kind"] = "predicate"
    with pytest.raises(RuntimeError):
        phase_s.world_from_document(document, "bad_kind")


def test_a_reduced_world_that_would_break_totality_is_refused(world_b):
    world = world_b[0]
    with pytest.raises(RuntimeError):
        phase_s.restrict_world(world, {"Site": 5, "District": 1})


# --- the frozen shortcut language ------------------------------------------


def test_every_shortcut_feature_is_invariant_under_every_derived_map(
    world_a, world_b, compiled_a, compiled_b
):
    """The fact that makes the L_k verdict mean what it says.

    Every L_k feature is a term over the declared primitives, so it must take the
    same value on an input and on that input's image under any structure-
    preserving map. Checking that directly means a feature can only be coarser
    than or equal to the orbit partition, never crosswise to it, so "does not
    recover q_Sigma" is exactly "is strictly coarser than q_Sigma".
    """
    for (world, _document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        _terms, features = phase_s.build_shortcut_language(world)
        position = {row: index for index, row in enumerate(world.inputs)}
        assert compiled.morphisms is not None
        for feature in features:
            for morphism in compiled.morphisms:
                for row in world.inputs:
                    moved = morphism.send(world, row)
                    assert (
                        feature.values[position[row]]
                        == feature.values[position[moved]]
                    ), (feature.expression, row, moved)


def test_the_shortcut_language_does_not_recover_either_quotient(
    world_a, world_b, compiled_a, compiled_b
):
    for (world, _document), compiled, finest in (
        (world_a, compiled_a, [21, 28]),
        (world_b, compiled_b, [9, 18]),
    ):
        audit = phase_s.shortcut_audit(world, compiled.quotient)
        assert not audit["any_single_feature_recovers_the_quotient"]
        assert not audit["the_whole_language_read_jointly_recovers_the_quotient"]
        assert audit["the_quotient_determines_every_feature"]
        assert audit["finest_single_feature_cell_sizes"] == finest
        assert audit["joint_partition_cell_sizes"] == finest
        assert audit["depth_k"] == phase_s.SHORTCUT_DEPTH


def test_on_the_single_relation_fixture_the_quotient_strictly_refines_the_relation(
    world_a, compiled_a
):
    """Continuity with 017.25a, re-derived rather than imported.

    The declared relation splits the admitted inputs 21 and 28. The derived
    quotient splits them 7, 14, 14, 14, and every derived cell lies wholly inside
    or wholly outside the relation. So the derived quotient is strictly finer than
    the one statistic 017.25a showed was already enough on its own fixture.
    """
    world = world_a[0]
    extension = _relation(world, "incident").extension
    relation = tuple(0 if row in extension else 1 for row in world.inputs)
    assert sorted(
        relation.count(cell) for cell in set(relation)
    ) == [21, 28]
    assert phase_s._refines(compiled_a.quotient, relation)
    assert not phase_s._refines(relation, compiled_a.quotient)


def test_the_disclosed_non_local_adversary_is_reported_not_hidden(
    world_a, world_b, compiled_a, compiled_b
):
    """The bound on the L_k claim is a measured fact banked in the freeze.

    An iterated pair refinement over the admitted inputs does recover both
    quotients. That is disclosed here, at the freeze, precisely so it cannot be
    presented later as a caveat discovered after the numbers were seen. It is not
    a member of L_k and 017.26b section 8 forbids condition R from using it.
    """
    for (world, _document), compiled in (
        (world_a, compiled_a),
        (world_b, compiled_b),
    ):
        audit = phase_s.non_local_adversary_audit(world, compiled.quotient)
        assert audit["recovers_the_derived_quotient"] is True
        assert audit["cell_sizes"] == compiled.cell_sizes


# --- the perturbations ------------------------------------------------------


def test_every_perturbation_matched_the_direction_frozen_before_it_was_compiled(
    world_a, world_b
):
    for world, document in (world_a, world_b):
        fixture = phase_s.Fixture(
            identifier=world.identifier,
            schema_name=document["schema"],
            digest="",
            role=str(document["role"]),
            world=world,
            document=document,
            compiled=phase_s.compile_world(world),
            perturbations=phase_s.build_perturbations(document, world.identifier),
        )
        rows = phase_s.perturbation_report(fixture)
        assert rows
        for row in rows:
            assert row["prediction_held"], row


def test_erasing_the_only_primitive_yields_the_full_product_of_symmetric_groups(
    world_a
):
    world, document = world_a
    erased = phase_s.world_from_document(
        phase_s.erase_primitive(document, "incident"), "erased"
    )
    compiled = phase_s.compile_world(erased)
    import math

    assert compiled.group_order == math.factorial(7) * math.factorial(7)
    assert compiled.cell_count == 1
    assert compiled.morphisms is None


def test_a_scramble_keeps_every_superficial_count_and_still_changes_the_quotient(
    world_a, compiled_a
):
    world, document = world_a
    changed = phase_s.scramble_primitive(document, "incident", 7)
    scrambled = phase_s.world_from_document(changed, "scrambled")
    original = _relation(world, "incident").extension
    replaced = _relation(scrambled, "incident").extension
    assert len(replaced) == len(original)
    assert replaced != original
    assert [len(c.elements) for c in scrambled.carriers] == [
        len(c.elements) for c in world.carriers
    ]
    assert phase_s.compile_world(scrambled).quotient != compiled_a.quotient


def test_the_symmetry_break_yields_a_subgroup_and_a_refinement(
    world_b, compiled_b
):
    world, document = world_b
    pointed = phase_s.world_from_document(
        phase_s.break_symmetry(document, "District", "D0", "reference"), "pointed"
    )
    compiled = phase_s.compile_world(pointed)
    assert compiled.group_order < compiled_b.group_order
    assert compiled.morphisms is not None and compiled_b.morphisms is not None
    assert set(compiled.morphisms) <= set(compiled_b.morphisms)
    assert phase_s._refines(compiled.quotient, compiled_b.quotient)


def test_every_primitive_of_the_richer_fixture_is_load_bearing(world_b, compiled_b):
    world, document = world_b
    perturbations = phase_s.build_perturbations(document, world.identifier)
    audit = phase_s.joint_load_bearing(world, compiled_b, perturbations)
    assert audit["primitive_count"] == 2
    assert audit["every_primitive_is_load_bearing"]
    assert audit["primitives_are_jointly_load_bearing"]
    for row in audit["per_primitive"]:
        assert row["is_load_bearing"], row


# --- the phase fence --------------------------------------------------------


def test_this_phase_cannot_have_seen_a_label_because_it_imports_no_generator():
    audit = phase_s.phase_fence_audit()
    assert audit["imports_no_statistical_module"]
    assert audit["no_target_parameter_vector_appears_in_this_module"]
    for banned in ("numpy", "training_economy", "quotient_refinement"):
        assert banned not in audit["top_level_imports"]


def test_the_compiler_path_names_no_fixture_and_no_expectation(world_a, world_b):
    audit = phase_s.dispatch_audit([world_a[0], world_b[0]])
    assert audit["no_declared_identifier_appears_in_the_compiler"], audit[
        "declared_identifiers_appearing_in_the_compiler"
    ]
    assert audit["no_fixture_or_expectation_or_label_vocabulary_appears"], audit[
        "forbidden_vocabulary_appearing_in_the_compiler"
    ]
    assert audit["every_declared_kind_has_a_generated_rule"]
    assert sorted(audit["preservation_rule_keys"]) == [
        "constant",
        "operation",
        "relation",
    ]


def test_compiling_twice_gives_the_identical_derivation(world_a):
    world = world_a[0]
    first = phase_s.compile_world(world)
    second = phase_s.compile_world(world)
    assert first.group_order == second.group_order
    assert first.quotient == second.quotient
    assert first.morphisms == second.morphisms


# --- the certificate, once the freeze has been written ---------------------


def test_the_certificate_matches_the_executable():
    if not phase_s.CERTIFICATE_PATH.exists():
        pytest.skip("Phase S certificate not written yet; run --freeze first")
    payload = phase_s.phase_s_certificate(phase_s.build_fixtures())
    assert phase_s.CERTIFICATE_PATH.read_bytes() == phase_s._canonical(payload)


def test_the_certificate_passes_every_mechanical_check():
    if not phase_s.CERTIFICATE_PATH.exists():
        pytest.skip("Phase S certificate not written yet")
    payload = json.loads(phase_s.CERTIFICATE_PATH.read_text())
    assert payload["all_mechanical_checks_pass"]
    assert all(payload["mechanical_checks"].values())
    assert payload["mechanical_check_count"] >= 30
    assert payload["phase"] == "S"


def test_the_certificate_names_what_it_does_not_establish():
    if not phase_s.CERTIFICATE_PATH.exists():
        pytest.skip("Phase S certificate not written yet")
    payload = json.loads(phase_s.CERTIFICATE_PATH.read_text())
    open_items = " ".join(payload["what_this_freeze_does_not_establish"]).lower()
    assert "labelled-data burden" in open_items
    assert "cheaper derivation" in open_items
    assert "representative" in open_items
    assert payload["compiler"]["what_the_compiler_is_not"]
