"""Exact Issue 017.20 Type-Schema-to-preparation compiler audit.

This research module is intentionally outside the stable ``decision_model`` API.
It freezes and audits one abstract Fano incidence source, tests the strongest
conditional fixed-test construction, and fails closed where the declared authority
supplies neither a source-to-preparation map nor a source-authorized test map.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

from decision_model import Test, resolve_test
from decision_model._audit_01717 import DIMENSION, NO_EFFECT, YES_EFFECT

AUDIT_SCHEMA: Final = "gpt-01720-input-constitution-audit/v2"
SOURCE_SCHEMA: Final = "gpt-01720-fano-incidence-source/v1"
SOURCE_SHA256: Final = (
    "20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c"
)
BASE_COMMIT: Final = "853f0371f675d911a2988f63c3eaeb5d2901bf6e"
GPT_REVISION: Final = (
    "e9ccdfdb1b4cdc6e342197d13b68c4d4c493997b495f4e83c5cceb553943bb60"
)
THEORY_REVISION: Final = (
    "b14aeb1bb9e07662e0e6fdde697f728cae3457324c0e2abd0cfd42975b8aac9a"
)
SOURCE_PATH: Final = Path(__file__).with_name("source_fixture.json")
ARTIFACT_PATH: Final = Path(__file__).with_name("input_constitution_audit.json")
SOURCE_REPO_PATH: Final = (
    "issues/017-generalized-born-rule/017.20-Code-attachments/source_fixture.json"
)
ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)

Datum = tuple[str, str]
Ray = tuple[Fraction, ...]


class MissingDeclaredRelation(RuntimeError):
    """Raised when an input schema does not type its opaque relation rows."""


@dataclass(frozen=True, slots=True)
class SourceSchema:
    """The executable portion of the frozen source Type Schema."""

    points: tuple[str, ...]
    lines: tuple[str, ...]
    incidence: frozenset[Datum] | None
    decision_labels: tuple[str, str]
    opaque_rows: tuple[Datum, ...] = ()

    @property
    def inputs(self) -> tuple[Datum, ...]:
        return tuple(itertools.product(self.points, self.lines))

    def semantic_class(self, datum: Datum) -> str:
        point, line = datum
        if point not in self.points or line not in self.lines:
            raise ValueError(f"datum {datum!r} is outside Point x Line")
        if self.incidence is None:
            raise MissingDeclaredRelation(
                "IncidenceDecision is undefined without a declared incident relation"
            )
        return "incident" if datum in self.incidence else "nonincident"


@dataclass(frozen=True, slots=True)
class Automorphism:
    """One sort-preserving automorphism of the frozen incidence structure."""

    point_images: tuple[str, ...]
    line_images: tuple[str, ...]

    def apply(self, datum: Datum, source: SourceSchema) -> Datum:
        point_map = dict(zip(source.points, self.point_images, strict=True))
        line_map = dict(zip(source.lines, self.line_images, strict=True))
        return point_map[datum[0]], line_map[datum[1]]


@dataclass(frozen=True, slots=True)
class RationalRay:
    """One exact nonzero rational witness in the real preparation carrier."""

    identifier: str
    coordinates: Ray

    def __post_init__(self) -> None:
        if len(self.coordinates) != DIMENSION:
            raise ValueError(f"preparation rays must have dimension {DIMENSION}")
        if not all(isinstance(value, Fraction) for value in self.coordinates):
            raise TypeError("preparation coordinates must be exact Fractions")
        if self.squared_norm <= 0:
            raise ValueError("preparation rays must be nonzero")

    @property
    def squared_norm(self) -> Fraction:
        return sum((value * value for value in self.coordinates), ZERO)


@dataclass(frozen=True, slots=True)
class CompilerRepresentatives:
    """A constitutive ray choice for the two source relation classes."""

    identifier: str
    incident: Ray
    nonincident: Ray

    def for_class(self, semantic_class: str) -> Ray:
        if semantic_class == "incident":
            return self.incident
        if semantic_class == "nonincident":
            return self.nonincident
        raise KeyError(semantic_class)


def _basis_ray(index: int) -> Ray:
    if index < 0 or index >= DIMENSION:
        raise ValueError("basis-ray index is outside the preparation carrier")
    return tuple(ONE if position == index else ZERO for position in range(DIMENSION))


REPRESENTATIVE_A: Final = CompilerRepresentatives(
    "representative-a",
    incident=_basis_ray(0),
    nonincident=_basis_ray(2),
)
REPRESENTATIVE_B: Final = CompilerRepresentatives(
    "representative-b",
    incident=_basis_ray(1),
    nonincident=_basis_ray(3),
)
REPRESENTATIVES: Final = (REPRESENTATIVE_A, REPRESENTATIVE_B)

# This is a conditional conformance witness only. The source schema and current OT
# authority do not select this 017.18 projector test or its label-to-slot binding.
CONDITIONAL_FIXED_TEST: Final = Test(
    (("incident", YES_EFFECT), ("nonincident", NO_EFFECT))
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(data)


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(key) or _contains_float(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(item) for item in value)
    return False


def load_source(path: Path = SOURCE_PATH) -> tuple[SourceSchema, dict[str, object]]:
    """Load the byte-pinned source and reject any post-freeze change."""

    source_bytes = path.read_bytes()
    observed_sha256 = _sha256(source_bytes)
    if observed_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            f"source fixture changed after freeze: {observed_sha256} != {SOURCE_SHA256}"
        )
    payload = json.loads(source_bytes)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("unexpected source fixture schema")

    sorts = payload["sorts"]
    relations = payload["relations"]
    decision = payload["decision_type"]
    source = SourceSchema(
        points=tuple(sorts["point"]),
        lines=tuple(sorts["line"]),
        incidence=frozenset(tuple(row) for row in relations["incident"]),
        decision_labels=tuple(decision["labels"]),
    )
    return source, payload


def audit_source(
    source: SourceSchema, source_document: dict[str, object]
) -> dict[str, object]:
    """Verify the full source document and finite Fano laws without literal gates."""

    required_sections = {
        "schema",
        "domain",
        "sorts",
        "relations",
        "admitted_inputs",
        "equivalence",
        "automorphisms",
        "decision_type",
        "compiler_visible_semantics",
        "compiler_forbidden_semantics",
        "preexisting_source_authority",
    }
    incidence = source.incidence
    if incidence is None:
        raise RuntimeError("the primary fixture must declare incidence")

    point_degrees = Counter(point for point, _ in incidence)
    line_degrees = Counter(line for _, line in incidence)
    lines_by_point_pair: Counter[tuple[str, str]] = Counter()
    points_by_line_pair: Counter[tuple[str, str]] = Counter()
    for line in source.lines:
        members = sorted(point for point, candidate in incidence if candidate == line)
        for pair in itertools.combinations(members, 2):
            lines_by_point_pair[pair] += 1
    for point in source.points:
        members = sorted(line for candidate, line in incidence if candidate == point)
        for pair in itertools.combinations(members, 2):
            points_by_line_pair[pair] += 1

    inputs = source.inputs
    class_counts = Counter(source.semantic_class(datum) for datum in inputs)
    declared_orbits = source_document["equivalence"]["expected_orbits"]
    declared_orbit_counts = {
        row["semantic_class"]: row["cardinality"] for row in declared_orbits
    }
    forbidden = set(source_document["compiler_forbidden_semantics"])
    visible = set(source_document["compiler_visible_semantics"])
    authority = source_document["preexisting_source_authority"]
    decision = source_document["decision_type"]
    admitted = source_document["admitted_inputs"]
    automorphisms = source_document["automorphisms"]

    gates = {
        "complete_machine_readable_type_schema": required_sections
        <= set(source_document),
        "at_least_two_meaningful_components": len(source.points) == len(source.lines) == 7,
        "nontrivial_schema_relation": len(incidence) == 21,
        "semantically_nonequivalent_inputs": class_counts
        == {"incident": 21, "nonincident": 28},
        "raw_encoding_equivalence_or_automorphism": bool(
            source_document["equivalence"]["raw_encoding_equivalence"]
        )
        and automorphisms["expected_group_order"] == 168,
        "exact_finite_data": not _contains_float(source_document),
        "fixed_decision_type": source.decision_labels == ("incident", "nonincident")
        and decision["fixed_across_inputs"] is True,
        "declared_cardinality_matches_generator": admitted["cardinality"]
        == len(inputs),
        "declared_orbit_census_matches_relation": declared_orbit_counts
        == dict(class_counts),
        "visible_semantics_declared": "the declared point-line incidence relation"
        in visible,
        "forbidden_target_semantics_declared": {
            "target probability",
            "Born expectation value",
            "OT preparation or ray coordinate",
            "effect coordinate",
            "answer label stored as an input field",
            "per-instance target-state table",
        }
        <= forbidden,
        "preexisting_abstract_authority_named": bool(authority["banked_gpt_result"])
        and authority["abstract_implementation"] == "topographo/core/f2_groups.py",
    }
    laws = {
        "seven_distinct_points": len(source.points) == len(set(source.points)) == 7,
        "seven_distinct_lines": len(source.lines) == len(set(source.lines)) == 7,
        "three_lines_per_point": set(point_degrees.values()) == {3},
        "three_points_per_line": set(line_degrees.values()) == {3},
        "each_point_pair_on_one_line": len(lines_by_point_pair) == 21
        and set(lines_by_point_pair.values()) == {1},
        "each_line_pair_meets_at_one_point": len(points_by_line_pair) == 21
        and set(points_by_line_pair.values()) == {1},
        "forty_nine_admitted_inputs": len(inputs) == 49,
        "twenty_one_incident_inputs": class_counts
        == {"incident": 21, "nonincident": 28},
        "datum_has_only_point_and_line_fields": all(
            len(datum) == 2
            and datum[0] in source.points
            and datum[1] in source.lines
            for datum in inputs
        ),
    }
    return {
        "fixture_selection": {
            "passes": all(gates.values()),
            "gates": gates,
            "selection_route": "existing banked GPT abstract finite fixture",
            "qualification": (
                "the abstract incidence reduct predates this execution; its Issue-011 "
                "genealogy is FIPS-derived, while these source bytes contain no Event, "
                "probability, effect, or preparation coordinates"
            ),
        },
        "incidence_laws": laws,
        "class_counts": dict(sorted(class_counts.items())),
        "relation_digest": _canonical_digest(sorted([list(row) for row in incidence])),
        "decision_semantics": {
            "global_schema_rule_completely_determines_binary_decision": True,
            "per_instance_answer_field_present": False,
            "predictive_claim_permitted": False,
        },
    }


def enumerate_automorphisms(source: SourceSchema) -> tuple[Automorphism, ...]:
    """Enumerate all 7! point images and their forced line images."""

    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("cannot enumerate incidence automorphisms")
    line_members = {
        line: frozenset(point for point, candidate in incidence if candidate == line)
        for line in source.lines
    }
    line_for_members = {members: line for line, members in line_members.items()}
    found: list[Automorphism] = []
    for point_images in itertools.permutations(source.points):
        point_map = dict(zip(source.points, point_images, strict=True))
        line_images: list[str] = []
        for line in source.lines:
            image_members = frozenset(point_map[point] for point in line_members[line])
            image_line = line_for_members.get(image_members)
            if image_line is None:
                break
            line_images.append(image_line)
        if len(line_images) == len(source.lines) and len(set(line_images)) == len(source.lines):
            found.append(Automorphism(point_images, tuple(line_images)))
    return tuple(found)


def input_orbits(
    source: SourceSchema, automorphisms: tuple[Automorphism, ...]
) -> tuple[tuple[Datum, ...], ...]:
    """Return exact point-line input orbits under all schema automorphisms."""

    remaining = set(source.inputs)
    orbits: list[tuple[Datum, ...]] = []
    while remaining:
        seed = min(remaining)
        orbit = {automorphism.apply(seed, source) for automorphism in automorphisms}
        orbits.append(tuple(sorted(orbit)))
        remaining -= orbit
    return tuple(sorted(orbits, key=lambda orbit: (len(orbit), orbit)))


def canonical_ray(ray: Ray) -> Ray:
    """Canonicalize a nonzero rational projective ray by its first nonzero entry."""

    first = next((value for value in ray if value != 0), None)
    if first is None:
        raise ValueError("zero has no projective class")
    return tuple(value / first for value in ray)


def projectively_equivalent(left: Ray, right: Ray) -> bool:
    return canonical_ray(left) == canonical_ray(right)


def compile_datum(
    datum: Datum,
    source: SourceSchema,
    representatives: CompilerRepresentatives = REPRESENTATIVE_A,
) -> RationalRay:
    """Conditional formula after external test and representative supply.

    This is not admitted as the task-level compiler: the two representatives and
    the test/slot binding are constitutive inputs absent from the source authority.
    """

    semantic_class = source.semantic_class(datum)
    return RationalRay(
        f"{representatives.identifier}:{semantic_class}",
        representatives.for_class(semantic_class),
    )


def _distribution_values(preparation: RationalRay) -> dict[str, Fraction]:
    distribution = resolve_test(preparation, CONDITIONAL_FIXED_TEST)
    values: dict[str, Fraction] = {}
    for label, probability in distribution:
        value = probability.value
        if not isinstance(value, Fraction):
            raise TypeError("the fixed-test audit must remain exact")
        values[label] = value
    return values


def _permute_coordinates(values: Ray, images: tuple[int, ...]) -> Ray:
    out = [ZERO] * len(values)
    for source, target in enumerate(images):
        out[target] = values[source]
    return tuple(out)


def _compiler_dependency_audit() -> dict[str, object]:
    tree = ast.parse(inspect.getsource(compile_datum))
    attributes = sorted({node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)})
    allowed_attributes = {
        "for_class",
        "identifier",
        "semantic_class",
    }
    signature = tuple(inspect.signature(compile_datum).parameters)
    return {
        "observed_attribute_reads": attributes,
        "allowed_attribute_reads": sorted(allowed_attributes),
        "only_allowed_attributes_read": set(attributes) <= allowed_attributes,
        "signature": list(signature),
        "no_per_instance_target_argument": signature
        == ("datum", "source", "representatives"),
        "source_datum_fields": ["point", "line"],
        "no_per_instance_probability_preparation_or_answer_field": True,
        "schema_relation_computes_the_complete_binary_decision": True,
        "interpretation": (
            "this is a conformance construction, not predictive evidence; the "
            "decision follows from globally declared source semantics"
        ),
    }


def audit_conditional_construction(
    source: SourceSchema,
    automorphisms: tuple[Automorphism, ...],
    orbits: tuple[tuple[Datum, ...], ...],
) -> dict[str, object]:
    """Audit the strongest witness after making its constitutive inputs explicit."""

    candidate_records: list[dict[str, object]] = []
    all_valid = True
    all_exact = True
    all_fixed_test_correct = True
    all_equivalent_inputs_equal = True
    all_distinguishable = True

    for representatives in REPRESENTATIVES:
        distributions: Counter[tuple[Fraction, Fraction]] = Counter()
        for datum in source.inputs:
            preparation = compile_datum(datum, source, representatives)
            values = _distribution_values(preparation)
            semantic_class = source.semantic_class(datum)
            expected = (
                {"incident": ONE, "nonincident": ZERO}
                if semantic_class == "incident"
                else {"incident": ZERO, "nonincident": ONE}
            )
            all_valid &= len(preparation.coordinates) == DIMENSION
            all_valid &= preparation.squared_norm > 0
            all_exact &= all(isinstance(value, Fraction) for value in values.values())
            all_fixed_test_correct &= values == expected and sum(values.values(), ZERO) == ONE
            distributions[(values["incident"], values["nonincident"])] += 1

        for automorphism in automorphisms:
            for datum in source.inputs:
                transformed = automorphism.apply(datum, source)
                left = compile_datum(datum, source, representatives).coordinates
                right = compile_datum(transformed, source, representatives).coordinates
                all_equivalent_inputs_equal &= projectively_equivalent(left, right)

        incidence = source.incidence
        if incidence is None:
            raise MissingDeclaredRelation("primary source lost incidence")
        incident = compile_datum(next(iter(incidence)), source, representatives)
        nonincident_datum = next(datum for datum in source.inputs if datum not in incidence)
        nonincident = compile_datum(nonincident_datum, source, representatives)
        distinguishable = not projectively_equivalent(
            incident.coordinates, nonincident.coordinates
        )
        all_distinguishable &= distinguishable
        candidate_records.append(
            {
                "id": representatives.identifier,
                "formula": "C(point,line) = r_incident iff incident(point,line), else r_nonincident",
                "incident_ray": _sparse_ray(representatives.incident),
                "nonincident_ray": _sparse_ray(representatives.nonincident),
                "valid_preparations": True,
                "schema_equivalence_invariant": True,
                "distinguishes_semantic_classes": distinguishable,
                "fixed_test_distribution_census": [
                    {
                        "distribution": [
                            _fraction_text(distribution[0]),
                            _fraction_text(distribution[1]),
                        ],
                        "inputs": count,
                    }
                    for distribution, count in sorted(distributions.items())
                ],
            }
        )

    rivals_projectively_distinct = all(
        not projectively_equivalent(
            left.for_class(semantic_class), right.for_class(semantic_class)
        )
        for left, right in itertools.combinations(REPRESENTATIVES, 2)
        for semantic_class in source.decision_labels
    )
    rival_predictions_equal = all(
        _distribution_values(compile_datum(datum, source, REPRESENTATIVE_A))
        == _distribution_values(compile_datum(datum, source, REPRESENTATIVE_B))
        for datum in source.inputs
    )

    coordinate_symmetry = tuple(1 if index == 0 else 0 if index == 1 else 3 if index == 2 else 2 if index == 3 else index for index in range(DIMENSION))
    symmetry_preserves_test = (
        _permute_coordinates(YES_EFFECT.diagonal, coordinate_symmetry)
        == YES_EFFECT.diagonal
        and _permute_coordinates(NO_EFFECT.diagonal, coordinate_symmetry)
        == NO_EFFECT.diagonal
    )
    symmetry_maps_rivals = (
        _permute_coordinates(REPRESENTATIVE_A.incident, coordinate_symmetry)
        == REPRESENTATIVE_B.incident
        and _permute_coordinates(REPRESENTATIVE_A.nonincident, coordinate_symmetry)
        == REPRESENTATIVE_B.nonincident
    )
    dependencies = _compiler_dependency_audit()

    gates = {
        "A_valid_preparation_conditional": all_valid,
        "B_semantic_equivalence_conditional": all_equivalent_inputs_equal,
        "C_equivariance": None,
        "C_equivariance_status": (
            "missing bridge: no certified action of the 168 source automorphisms "
            "on the full Theory-27 preparation carrier is supplied"
        ),
        "D_distinguishability_conditional": all_distinguishable,
        "E_one_input_independent_test_conditional": all_fixed_test_correct,
        "E_source_authorized_test": False,
        "F_no_direct_per_instance_target_leakage": dependencies[
            "only_allowed_attributes_read"
        ]
        and dependencies["no_per_instance_target_argument"]
        and dependencies["no_per_instance_probability_preparation_or_answer_field"],
        "F_schema_rule_already_determines_decision": True,
        "exact_rational_evaluation": all_exact,
        "learned_parameter_count": 0,
        "task_positive_admitted": False,
    }

    return {
        "status": "conditional witness; rejected as a task-level compiler",
        "conditions_not_derived": [
            "selection of the 017.18 diagonal projector test",
            "binding IncidenceDecision labels to the two test slots",
            "one preparation ray for each source relation class",
        ],
        "compiler_family": {
            "type": "C_{r_I,r_N}: Input_Sigma -> P(R^16)",
            "factorization": "Input_Sigma -> {incident, nonincident} -> P(R^16)",
            "orbit_classes": [source.semantic_class(orbit[0]) for orbit in orbits],
            "orbit_sizes": [len(orbit) for orbit in orbits],
            "learned_parameter_count": 0,
            "per_instance_lookup": False,
            "representatives_are_constitutive_not_declared": True,
        },
        "conditional_fixed_test": {
            "labels": list(CONDITIONAL_FIXED_TEST.labels),
            "same_immutable_test_for_all_inputs": True,
            "source_authorized_for_IncidenceDecision": False,
            "effect_source": "decision_model._audit_01717 fixed 017.18 evaluator witness",
            "incident_effect": _effect_record(YES_EFFECT.diagonal),
            "nonincident_effect": _effect_record(NO_EFFECT.diagonal),
            "formal_test_sum_is_unit": all(
                yes + no == ONE
                for yes, no in zip(YES_EFFECT.diagonal, NO_EFFECT.diagonal, strict=True)
            ),
            "slot_bijections_before_semantic_binding": math.factorial(2),
        },
        "representative_compilers": candidate_records,
        "gates": gates,
        "dependency_audit": dependencies,
        "rival_search": {
            "representatives_exhibited": len(REPRESENTATIVES),
            "projectively_distinct": rivals_projectively_distinct,
            "identical_conditional_fixed_test_predictions": rival_predictions_equal,
            "test_preserving_coordinate_symmetry": {
                "image_by_source_index": list(coordinate_symmetry),
                "preserves_both_effects": symmetry_preserves_test,
                "maps_representative_a_to_b": symmetry_maps_rivals,
            },
            "certified_as_full_OT_representation_gauge": False,
            "conclusion": (
                "the exhibited rivals reject C3 but do not decide C1 versus C2: "
                "they lie in one explicit fixed-test symmetry orbit, and current "
                "authority does not identify that test stabilizer with full OT gauge"
            ),
        },
        "conditional_residue": {
            "conditions": "after fixing the imported rank-(2,14) test and its slot binding",
            "ambient_real_family": {
                "family": "RP^1 x RP^13",
                "real_dimension": 14,
                "cardinality": "continuum",
            },
            "executable_exact_rational_subset": {
                "family": "P(Q^2) x P(Q^14)",
                "cardinality": "countably infinite",
            },
            "per_instance_correct_fixed_test_dimension": 21 * 1 + 28 * 13,
            "schema_invariant_correct_fixed_test_dimension": 1 + 13,
            "conditional_dimension_removed": (21 * 1 + 28 * 13) - (1 + 13),
        },
    }


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _sparse_ray(ray: Ray) -> list[dict[str, object]]:
    return [
        {"index": index, "value": _fraction_text(value)}
        for index, value in enumerate(ray)
        if value != 0
    ]


def _effect_record(diagonal: tuple[Fraction, ...]) -> dict[str, object]:
    return {
        "dimension": len(diagonal),
        "rank": sum(value != 0 for value in diagonal),
        "nonzero_indices": [index for index, value in enumerate(diagonal) if value != 0],
        "positive": all(value >= 0 for value in diagonal),
        "bounded_by_unit": all(value <= 1 for value in diagonal),
        "projector": all(value * value == value for value in diagonal),
    }


def make_relation_erased_control(source: SourceSchema) -> SourceSchema:
    """Preserve raw rows as opaque bytes while removing their relation type."""

    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("primary source already lacks incidence")
    return SourceSchema(
        points=source.points,
        lines=source.lines,
        incidence=None,
        decision_labels=source.decision_labels,
        opaque_rows=tuple(sorted(incidence)),
    )


def audit_control(source: SourceSchema) -> dict[str, object]:
    """Apply the production compiler to a matched relation-unavailable schema."""

    control = make_relation_erased_control(source)
    failure = None
    try:
        compile_datum(control.inputs[0], control, REPRESENTATIVE_A)
    except MissingDeclaredRelation as exc:
        failure = str(exc)

    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("primary source lost incidence")
    structured_order = 168
    type_only_order = math.factorial(len(source.points)) * math.factorial(len(source.lines))
    seed = control.inputs[0]
    type_only_orbit = {
        (point_image, line_image)
        for point_image in control.points
        for line_image in control.lines
    }
    return {
        "name": "relation-erased matched control",
        "same_compiler_entry_point": "compile_datum",
        "preserved": {
            "points_identical": control.points == source.points,
            "lines_identical": control.lines == source.lines,
            "admitted_inputs_identical": control.inputs == source.inputs,
            "opaque_rows_identical": set(control.opaque_rows) == set(incidence),
            "opaque_rows_sha256": _canonical_digest(
                [list(row) for row in control.opaque_rows]
            ),
        },
        "removed": "the declaration that the 21 opaque rows mean point-line incidence",
        "declared_automorphism_order": {
            "structured": structured_order,
            "relation_erased": type_only_order,
            "derivation": "7! independent point permutations times 7! independent line permutations",
            "expansion_factor": type_only_order // structured_order,
        },
        "type_only_orbit_witness": {
            "seed": list(seed),
            "reachable_pairs": len(type_only_orbit),
            "equals_all_admitted_inputs": type_only_orbit == set(control.inputs),
        },
        "semantic_input_orbits_after_removal": 1,
        "incidence_decision_defined": False,
        "compiler_failed_closed": failure is not None,
        "failure": failure,
        "interpretation": (
            "the relation is load-bearing for the two-class source quotient; "
            "retaining its raw rows without a declared meaning does not license use"
        ),
    }


def authority_audit() -> dict[str, object]:
    """Record which admissible authorities do and do not supply the bridges."""

    theory = f"quilt+s3://protology#package=occurrence/theory@{THEORY_REVISION}"
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{GPT_REVISION}"
    return {
        "preparation_carrier": "P(V), instantiated as P(R^16) by the exact 017.18 witness",
        "state_evaluation": "omega_[x](e) = <x,iota(e)x>/<x,x>",
        "authorities": [
            {
                "uri": f"{theory}&path=27-the-lift-theorem-and-probability-law.md",
                "finding": "takes [x] as supplied and leaves domain/physical preparation -> [x] open",
            },
            {
                "uri": f"{theory}&path=41-effect-census-test-admissibility-and-sce-sharp-shadow.md",
                "finding": "Fano/FIPS/SCE source geometry gains no formal target type without an explicit bridge",
            },
            {
                "uri": f"{theory}&path=43-pauli-fano-quantum-representation-boundary.md",
                "finding": (
                    "Encoding I requires noncanonical U and phi; Encoding II either "
                    "starts from a forbidden Event vector or lands in CP^7, not P(R^16)"
                ),
            },
            {
                "uri": f"{gpt}&path=issues/017-jev-pivot/017.18-Kiro-Born-first-Decision-Model-Zero-cross-carrier-audit-result.md",
                "finding": (
                    "supplies an exact fixed-test evaluator witness, but its source is "
                    "already p and it does not interpret IncidenceDecision"
                ),
            },
        ],
        "missing_typed_bridges": [
            "source Fano incidence -> distinguished Theory-27 preparation ray",
            "source automorphism -> certified action on the full preparation carrier",
            "IncidenceDecision -> distinguished formal OT test",
        ],
        "first_missing_arrow": "source semantics -> distinguished linear/projective preparation data",
        "scope": "No claim is made that no bridge can exist outside the declared evidence fence.",
    }


def build_payload() -> dict[str, object]:
    source, source_document = load_source()
    source_result = audit_source(source, source_document)
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)
    conditional = audit_conditional_construction(source, automorphisms, orbits)
    control = audit_control(source)

    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("primary source lost incidence")
    automorphism_laws = {
        "group_order_is_168": len(automorphisms) == 168,
        "every_automorphism_preserves_incidence": all(
            (datum in incidence)
            == (automorphism.apply(datum, source) in incidence)
            for automorphism in automorphisms
            for datum in source.inputs
        ),
        "two_input_orbits": sorted(len(orbit) for orbit in orbits) == [21, 28],
        "orbits_match_semantic_classes": all(
            len({source.semantic_class(datum) for datum in orbit}) == 1 for orbit in orbits
        ),
    }

    gates = conditional["gates"]
    mechanical_checks = {
        **source_result["fixture_selection"]["gates"],
        **source_result["incidence_laws"],
        **automorphism_laws,
        "source_byte_hash_matches_local_freeze": _sha256(SOURCE_PATH.read_bytes())
        == SOURCE_SHA256,
        "conditional_preparations_valid": gates["A_valid_preparation_conditional"],
        "conditional_semantic_equivalence": gates[
            "B_semantic_equivalence_conditional"
        ],
        "conditional_distinguishability": gates[
            "D_distinguishability_conditional"
        ],
        "conditional_fixed_test_exact": gates[
            "E_one_input_independent_test_conditional"
        ],
        "no_direct_per_instance_target_leakage": gates[
            "F_no_direct_per_instance_target_leakage"
        ],
        "conditional_evaluation_exact": gates["exact_rational_evaluation"],
        "rivals_projectively_distinct": conditional["rival_search"][
            "projectively_distinct"
        ],
        "rivals_have_same_conditional_predictions": conditional["rival_search"][
            "identical_conditional_fixed_test_predictions"
        ],
        "rival_symmetry_preserves_test": conditional["rival_search"][
            "test_preserving_coordinate_symmetry"
        ]["preserves_both_effects"],
        "rival_symmetry_maps_a_to_b": conditional["rival_search"][
            "test_preserving_coordinate_symmetry"
        ]["maps_representative_a_to_b"],
        "same_compiler_fails_on_relation_erased_control": control[
            "compiler_failed_closed"
        ],
        "control_preserves_raw_carrier_and_rows": all(
            control["preserved"][key] is True
            for key in (
                "points_identical",
                "lines_identical",
                "admitted_inputs_identical",
                "opaque_rows_identical",
            )
        ),
        "control_type_only_action_is_transitive": control["type_only_orbit_witness"][
            "equals_all_admitted_inputs"
        ],
    }
    if not all(mechanical_checks.values()):
        broken = sorted(name for name, passed in mechanical_checks.items() if not passed)
        raise RuntimeError(f"mechanical audit checks failed: {broken}")

    authority = source_document["preexisting_source_authority"]
    return {
        "schema": AUDIT_SCHEMA,
        "task": "occurrence/gpt Issue 017.20",
        "implementation_base_commit": BASE_COMMIT,
        "source_freeze": {
            "schema": SOURCE_SCHEMA,
            "path": SOURCE_REPO_PATH,
            "sha256": SOURCE_SHA256,
            "preexisting_incidence_reduct": True,
            "preexisting_banked_authority": authority["banked_gpt_result"],
            "exact_decision_bearing_schema_has_separate_prior_immutable_pin": False,
            "local_execution_order": "source bytes were written and hashed before conditional compiler code",
            "independent_preregistration_grade": "not established",
            "target_coordinates_in_semantic_payload": False,
        },
        "source": source_result,
        "source_automorphisms": {
            "search_space": math.factorial(7),
            "group_order": len(automorphisms),
            "input_orbits": [
                {
                    "semantic_class": source.semantic_class(orbit[0]),
                    "size": len(orbit),
                    "witnesses": [list(datum) for datum in orbit[:2]],
                }
                for orbit in orbits
            ],
            "laws": automorphism_laws,
        },
        "authority_audit": authority_audit(),
        "conditional_construction": conditional,
        "task_canonicality_verdict": {
            "code": "Cempty",
            "symbol": "C∅",
            "label": "unavailable under the evidence fence",
            "reason": (
                "no source-authorized map selects a Theory-27 preparation or an "
                "IncidenceDecision formal test; the executable family begins only "
                "after those constitutive choices are supplied"
            ),
            "conditional_classification": (
                "C1 versus C2 unresolved after supplied-test conditioning; the "
                "explicit rivals share a fixed-test symmetry orbit that is not "
                "certified as the full OT representation gauge"
            ),
            "C0_boundary": (
                "choosing concrete basis rays without the missing bridges is an "
                "arbitrary embedding witness, not a scientific positive"
            ),
            "strongest_justified_task_status": True,
        },
        "declared_derived_constitutive_ledger": {
            "declared": [
                "two opaque seven-element sorts Point and Line",
                "21 point-line incidences",
                "49 admitted Point x Line inputs",
                "incidence-preserving raw-presentation equivalences",
                "fixed binary IncidenceDecision semantics",
            ],
            "derived": [
                "Fano incidence laws and automorphism group order 168",
                "two relation classes/orbits of sizes 21 and 28",
                "a conditional factorization through incident/nonincident",
                "failure of that factorization when incidence meaning is erased",
            ],
            "still_constitutive": [
                "selection of a formal OT test for IncidenceDecision",
                "one of 2 label-to-slot bijections after a binary test is supplied",
                "selection of preparation representatives",
                "a target action or equivalence corresponding to source automorphisms",
            ],
            "full_task_residue": {
                "status": "not typed sharply enough by current authority to quantify",
                "reason": (
                    "the admissible family of source-authorized IncidenceDecision "
                    "tests and its relation to preparation gauge are both absent"
                ),
            },
            "conditional_after_imported_test_and_slot_binding": conditional[
                "conditional_residue"
            ],
        },
        "control": control,
        "fixed_test_pipeline": {
            "type_correct_after_constitutive_test_supply": True,
            "source_authorized": False,
            "task_positive_admitted": False,
            "pipeline": (
                "Point x Line -> conditional representative -> imported fixed "
                "IncidenceDecision-labeled test -> exact distribution"
            ),
            "distribution_by_semantic_class": {
                "incident": {"incident": "1", "nonincident": "0"},
                "nonincident": {"incident": "0", "nonincident": "1"},
            },
            "interpretation": (
                "exact evaluator conformance only; deterministic correctness follows "
                "from the complete incidence rule declared in the schema"
            ),
            "predictive_accuracy_optimized": False,
            "training_performed": False,
        },
        "training_handoff": {
            "justified_now": False,
            "reason": (
                "the only quantified ray residue is conditional on an unauthorized "
                "test and is invisible to that sole test; no typed training signal "
                "selects it"
            ),
            "fixture_scale_binary_decision": "symbolically determined by the Type Schema",
            "future_condition": (
                "supply an independently authorized source-to-preparation principle "
                "or richer test family before charging any residue to Training"
            ),
        },
        "reproducibility": {
            "runtime": "Python >=3.11; decision-model 1.1.0; standard library otherwise",
            "commands": [
                "PYTHONPATH=decision-model/src uv run --frozen python issues/017-generalized-born-rule/017.20-Code-attachments/audit_input_constitution.py --check",
                "PYTHONPATH=decision-model/src uv run --frozen pytest issues/017-generalized-born-rule/017.20-Code-attachments/test_audit_input_constitution.py",
            ],
            "network_required": False,
            "learned_parameters": 0,
            "optimizer_or_training_imported": False,
            "interact_imported": False,
        },
        "mechanical_checks": mechanical_checks,
        "all_mechanical_checks_pass": True,
        "task_positive_admission": False,
        "disposition": (
            "STRONG NEGATIVE: Cempty under the evidence fence; the exact conditional "
            "fixed-test witness does not supply the missing constitution bridges"
        ),
    }


def canonical_audit_bytes() -> bytes:
    return (json.dumps(build_payload(), indent=2, sort_keys=True) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifact byte-for-byte instead of printing it",
    )
    args = parser.parse_args(argv)
    expected = canonical_audit_bytes()
    if args.check:
        if not ARTIFACT_PATH.exists():
            print(f"missing artifact: {ARTIFACT_PATH}", file=sys.stderr)
            return 1
        observed = ARTIFACT_PATH.read_bytes()
        if observed != expected:
            print(
                f"artifact mismatch: observed {_sha256(observed)}, "
                f"expected {_sha256(expected)}",
                file=sys.stderr,
            )
            return 1
        print(f"017.20 audit verified: {_sha256(expected)}")
        return 0
    sys.stdout.buffer.write(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
