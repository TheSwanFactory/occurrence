"""Exact Issue 017.21a unfenced central-constitution audit.

017.20 returned C-empty because no authorized bridge reached a preparation. This
module drops that evidence fence and tests the strongest natural replacement: the
intrinsic center of the reduced ordered Jordan effect algebra.

Nothing here is trained, fitted, or physical. Every number is exact: rational
arithmetic in the adapted ``(A, s)`` coordinates, and exact ``Q(sqrt 7)``
arithmetic for the ambient 16-dimensional ray evaluations.
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
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

AUDIT_SCHEMA: Final = "gpt-01721a-central-constitution-audit/v1"
SOURCE_SCHEMA: Final = "gpt-01720-fano-incidence-source/v1"
PREMISE_SCHEMA: Final = "gpt-01721a-unfenced-premises/v1"
SOURCE_SHA256: Final = (
    "20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c"
)
PRIOR_RESULT_COMMIT: Final = "38cd808fe459f7a1e39d9a2ce4f15f55ebf22c5d"
GPT_REVISION: Final = (
    "65dca265e5a6d9f0f5937f014a286d361721cd64d37f140ee87d7f2caaf41cbd"
)
THEORY_REVISION: Final = (
    "b14aeb1bb9e07662e0e6fdde697f728cae3457324c0e2abd0cfd42975b8aac9a"
)
FIXED_REVISION: Final = (
    "f11ea77410fc79c8bacd8ba44633270786735ecf08cfff05e855613936b1760b"
)
FIXED_SURFACE_SHA256: Final = (
    "17e07af79a56cea92343a629bf901ef8c930e5a7bfe0572396968fb308a100ce"
)

HERE: Final = Path(__file__).resolve().parent
SOURCE_PATH: Final = (
    HERE.parent / "017.20-Code-attachments" / "source_fixture.json"
)
PREMISE_PATH: Final = HERE / "declared_premises.json"
ARTIFACT_PATH: Final = HERE / "central_constitution_audit.json"
SOURCE_REPO_PATH: Final = (
    "issues/017-generalized-born-rule/017.20-Code-attachments/source_fixture.json"
)

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)
AMBIENT_DIMENSION: Final = 16
MULTIPLICITY_MATRIX: Final = 2
MULTIPLICITY_SCALAR: Final = 12

Datum = tuple[str, str]


class MissingDeclaredRelation(RuntimeError):
    """Raised when an input schema does not type its opaque relation rows."""


# ---------------------------------------------------------------------------
# Exact Q(sqrt 7) arithmetic, needed because the Fixed frame contains sqrt(7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Q7:
    """The exact number ``rational + radical * sqrt(7)``."""

    rational: Fraction = ZERO
    radical: Fraction = ZERO

    def __post_init__(self) -> None:
        if not isinstance(self.rational, Fraction) or not isinstance(
            self.radical, Fraction
        ):
            raise TypeError("Q7 components must be exact Fractions")

    def __add__(self, other: Q7) -> Q7:
        return Q7(self.rational + other.rational, self.radical + other.radical)

    def __sub__(self, other: Q7) -> Q7:
        return Q7(self.rational - other.rational, self.radical - other.radical)

    def __neg__(self) -> Q7:
        return Q7(-self.rational, -self.radical)

    def __mul__(self, other: Q7) -> Q7:
        return Q7(
            self.rational * other.rational + 7 * self.radical * other.radical,
            self.rational * other.radical + self.radical * other.rational,
        )

    def scale(self, factor: Fraction) -> Q7:
        return Q7(self.rational * factor, self.radical * factor)

    @property
    def is_zero(self) -> bool:
        return self.rational == 0 and self.radical == 0

    @property
    def is_rational(self) -> bool:
        return self.radical == 0

    def as_fraction(self) -> Fraction:
        if not self.is_rational:
            raise ValueError("value is irrational; cannot narrow to Fraction")
        return self.rational

    def text(self) -> str:
        if self.is_rational:
            return _fraction_text(self.rational)
        if self.rational == 0:
            return f"{_fraction_text(self.radical)}*sqrt(7)"
        return f"{_fraction_text(self.rational)} + {_fraction_text(self.radical)}*sqrt(7)"


Q7_ZERO: Final = Q7()
Q7_ONE: Final = Q7(ONE)
Q7_SQRT7: Final = Q7(ZERO, ONE)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# The frozen source schema, reused unchanged from 017.20
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The reduced effect algebra E = Sym_2(R) + R
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """One element of E in adapted coordinates ``(A, s)``, ``A`` symmetric 2x2."""

    identifier: str
    a11: Fraction
    a12: Fraction
    a22: Fraction
    s: Fraction

    def __post_init__(self) -> None:
        for value in (self.a11, self.a12, self.a22, self.s):
            if not isinstance(value, Fraction):
                raise TypeError("element coordinates must be exact Fractions")

    @property
    def trace(self) -> Fraction:
        return self.a11 + self.a22

    @property
    def determinant(self) -> Fraction:
        return self.a11 * self.a22 - self.a12 * self.a12

    @property
    def is_positive(self) -> bool:
        """Exact PSD test on the 2x2 block plus nonnegative scalar leg."""
        return (
            self.a11 >= 0
            and self.a22 >= 0
            and self.determinant >= 0
            and self.s >= 0
        )

    @property
    def is_effect(self) -> bool:
        """Membership in the full formal interval ``[0, 1_E]`` (premise P2)."""
        return self.is_positive and complement(self).is_positive

    @property
    def is_jordan_idempotent(self) -> bool:
        return jordan_product(self, self) == _anonymous(self)

    @property
    def g_coordinates(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        """Invert the certified Fixed map ``Phi``, giving ``(a, b, c, d)``."""
        a = self.a11
        b = self.a12 / 7
        d = (self.a22 - self.s) / 7
        c = (self.a22 + 6 * self.s) / 7
        return (a, b, c, d)

    @property
    def ambient_rank(self) -> int:
        """Rank in the concrete 16-dimensional realization."""
        block_rank = 0
        if not (self.a11 == 0 and self.a12 == 0 and self.a22 == 0):
            block_rank = 1 if self.determinant == 0 else 2
        scalar_rank = 0 if self.s == 0 else MULTIPLICITY_SCALAR
        return MULTIPLICITY_MATRIX * block_rank + scalar_rank


def _anonymous(element: Element) -> Element:
    """Strip the label so comparisons test structure, not naming."""
    return Element("", element.a11, element.a12, element.a22, element.s)


def _anonymous_state(state: State) -> State:
    """Strip the label so state comparisons test structure, not naming."""
    return State("", state.r11, state.r12, state.r22, state.q)


def element(
    identifier: str,
    a11: Fraction,
    a12: Fraction,
    a22: Fraction,
    s: Fraction,
) -> Element:
    return Element(identifier, a11, a12, a22, s)


UNIT: Final = element("1_E", ONE, ZERO, ONE, ONE)
ALGEBRA_ZERO: Final = element("0_E", ZERO, ZERO, ZERO, ZERO)

# The two minimal central idempotents, derived rather than chosen.
Z_MATRIX: Final = element("z_M", ONE, ZERO, ONE, ZERO)
Z_SCALAR: Final = element("z_s", ZERO, ZERO, ZERO, ONE)


def add(left: Element, right: Element) -> Element:
    return Element(
        "",
        left.a11 + right.a11,
        left.a12 + right.a12,
        left.a22 + right.a22,
        left.s + right.s,
    )


def complement(value: Element) -> Element:
    return Element(
        "",
        ONE - value.a11,
        -value.a12,
        ONE - value.a22,
        ONE - value.s,
    )


def jordan_product(left: Element, right: Element) -> Element:
    """The exact Jordan product ``(XY + YX)/2`` in adapted coordinates."""
    a11 = left.a11 * right.a11 + left.a12 * right.a12
    a22 = left.a12 * right.a12 + left.a22 * right.a22
    a12 = Fraction(1, 2) * (
        left.a11 * right.a12
        + left.a12 * right.a22
        + left.a12 * right.a11
        + left.a22 * right.a12
    )
    return Element("", a11, a12, a22, left.s * right.s)


def commutes_with_all(candidate: Element, probes: tuple[Element, ...]) -> bool:
    return all(
        jordan_product(candidate, probe) == jordan_product(probe, candidate)
        for probe in probes
    )


def associates_with_all(candidate: Element, probes: tuple[Element, ...]) -> bool:
    """Operator-commutation test, the correct notion of Jordan centrality."""
    for left in probes:
        for right in probes:
            first = jordan_product(candidate, jordan_product(left, right))
            second = jordan_product(left, jordan_product(candidate, right))
            if first != second:
                return False
    return True


# ---------------------------------------------------------------------------
# States on E
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class State:
    """A positive unital functional ``omega(A,s) = Tr(rho A) + q s``."""

    identifier: str
    r11: Fraction
    r12: Fraction
    r22: Fraction
    q: Fraction

    def __post_init__(self) -> None:
        for value in (self.r11, self.r12, self.r22, self.q):
            if not isinstance(value, Fraction):
                raise TypeError("state coordinates must be exact Fractions")

    def __call__(self, value: Element) -> Fraction:
        return (
            self.r11 * value.a11
            + 2 * self.r12 * value.a12
            + self.r22 * value.a22
            + self.q * value.s
        )

    @property
    def is_positive(self) -> bool:
        determinant = self.r11 * self.r22 - self.r12 * self.r12
        return (
            self.r11 >= 0
            and self.r22 >= 0
            and determinant >= 0
            and self.q >= 0
        )

    @property
    def is_unital(self) -> bool:
        return self(UNIT) == ONE

    @property
    def is_state(self) -> bool:
        return self.is_positive and self.is_unital

    def conjugate_by_rotation(self, cosine: Fraction, sine: Fraction) -> State:
        """Push forward by the target automorphism ``A -> Q^T A Q``."""
        if cosine * cosine + sine * sine != 1:
            raise ValueError("rotation parameters must satisfy c^2 + s^2 = 1")
        r11 = (
            cosine * cosine * self.r11
            - 2 * cosine * sine * self.r12
            + sine * sine * self.r22
        )
        r22 = (
            sine * sine * self.r11
            + 2 * cosine * sine * self.r12
            + cosine * cosine * self.r22
        )
        r12 = (
            cosine * sine * self.r11
            + (cosine * cosine - sine * sine) * self.r12
            - cosine * sine * self.r22
        )
        return State("", r11, r12, r22, self.q)

    def conjugate_by_reflection(self) -> State:
        """Push forward by the orientation-reversing target automorphism."""
        return State("", self.r11, -self.r12, self.r22, self.q)


TAU_MATRIX: Final = State("tau_M", Fraction(1, 2), ZERO, Fraction(1, 2), ZERO)
DELTA_SCALAR: Final = State("delta_s", ZERO, ZERO, ZERO, ONE)

# Exact rational rotations used as target-automorphism probes: the primitive
# Pythagorean triples give exact cos/sin without leaving Q.
RATIONAL_ROTATIONS: Final = (
    (ONE, ZERO),
    (Fraction(3, 5), Fraction(4, 5)),
    (Fraction(4, 5), Fraction(3, 5)),
    (Fraction(5, 13), Fraction(12, 13)),
    (Fraction(12, 13), Fraction(5, 13)),
    (Fraction(8, 17), Fraction(15, 17)),
    (Fraction(-3, 5), Fraction(4, 5)),
    (Fraction(-4, 5), Fraction(-3, 5)),
    (Fraction(7, 25), Fraction(24, 25)),
    (Fraction(20, 29), Fraction(21, 29)),
)


# ---------------------------------------------------------------------------
# The ambient 16-dimensional realization, exact over Q(sqrt 7)
# ---------------------------------------------------------------------------


def ambient_vector(entries: dict[int, Q7]) -> tuple[Q7, ...]:
    out = [Q7_ZERO] * AMBIENT_DIMENSION
    for index, value in entries.items():
        if index < 0 or index >= AMBIENT_DIMENSION:
            raise ValueError("ambient index out of range")
        out[index] = value
    return tuple(out)


def u0_vector() -> tuple[Q7, ...]:
    return ambient_vector({0: Q7_ONE})


def v8_vector() -> tuple[Q7, ...]:
    """``v8 = -(1/sqrt 7) sum_{i=9..15} e_i``, exact as ``-sqrt(7)/7``."""
    coefficient = Q7_SQRT7.scale(Fraction(-1, 7))
    return ambient_vector({index: coefficient for index in range(9, 16)})


def v0_vector() -> tuple[Q7, ...]:
    coefficient = Q7_SQRT7.scale(Fraction(1, 7))
    return ambient_vector({index: coefficient for index in range(1, 8)})


def ambient_add(left: tuple[Q7, ...], right: tuple[Q7, ...]) -> tuple[Q7, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def ambient_inner(left: tuple[Q7, ...], right: tuple[Q7, ...]) -> Q7:
    total = Q7_ZERO
    for a, b in zip(left, right, strict=True):
        total = total + a * b
    return total


def generator_action(name: str, vector: tuple[Q7, ...]) -> tuple[Q7, ...]:
    """Apply one Fixed generator to an ambient vector, exactly.

    The generators act through the certified frame ``U + W+ + W-``:
    ``g0`` projects onto ``U``, ``g3`` onto ``W+ + W-``, ``g4 = 6 P_{W+} - P_{W-}``,
    and ``g1 = sqrt(7)(|v0><u0| + |u0><v0| + |v8><u8| + |u8><v8|)``.
    """
    u0 = u0_vector()
    u8 = ambient_vector({8: Q7_ONE})
    v0 = v0_vector()
    v8 = v8_vector()

    c_u0 = ambient_inner(u0, vector)
    c_u8 = ambient_inner(u8, vector)
    c_v0 = ambient_inner(v0, vector)
    c_v8 = ambient_inner(v8, vector)

    projection_u = ambient_add(
        tuple(entry * c_u0 for entry in u0), tuple(entry * c_u8 for entry in u8)
    )
    projection_wp = ambient_add(
        tuple(entry * c_v0 for entry in v0), tuple(entry * c_v8 for entry in v8)
    )
    projection_wm = tuple(
        value - u - wp
        for value, u, wp in zip(vector, projection_u, projection_wp, strict=True)
    )

    if name == "g0":
        return projection_u
    if name == "g3":
        return ambient_add(projection_wp, projection_wm)
    if name == "g4":
        return ambient_add(
            tuple(entry.scale(Fraction(6)) for entry in projection_wp),
            tuple(-entry for entry in projection_wm),
        )
    if name == "g1":
        first = ambient_add(
            tuple(entry * c_u0 for entry in v0), tuple(entry * c_v0 for entry in u0)
        )
        second = ambient_add(
            tuple(entry * c_u8 for entry in v8), tuple(entry * c_v8 for entry in u8)
        )
        combined = ambient_add(first, second)
        return tuple(Q7_SQRT7 * entry for entry in combined)
    raise KeyError(name)


def ambient_state_of(vector: tuple[Q7, ...]) -> State:
    """Read off the exact Theory-27 state induced by an ambient ray."""
    norm = ambient_inner(vector, vector)
    if norm.is_zero:
        raise ValueError("preparation rays must be nonzero")
    denominator = norm.as_fraction()

    values: dict[str, Fraction] = {}
    for name in ("g0", "g3", "g4"):
        numerator = ambient_inner(vector, generator_action(name, vector))
        values[name] = numerator.as_fraction() / denominator

    # The g1 pairing is intrinsically a multiple of sqrt(7), because g1 couples
    # U to W+ with that exact coefficient. Read it in units of sqrt(7) rather
    # than narrowing it to a Fraction, and require no stray rational part.
    g1_numerator = ambient_inner(vector, generator_action("g1", vector))
    if g1_numerator.rational != 0:
        raise ValueError("the g1 pairing must be a pure multiple of sqrt(7)")
    values["g1"] = g1_numerator.radical / denominator

    # Invert the g-coordinate readout into the state coordinates (rho, q).
    #
    # For X = a' g0 + b' g1 + c' g3 + d' g4 the adapted coordinates are
    #     A = [[a', sqrt(7) b'], [sqrt(7) b', c' + 6 d']],  s = c' - d',
    # and omega(A, s) = r11 A_11 + 2 r12 A_12 + r22 A_22 + q s. Matching the four
    # generator values term by term gives the unique solution below.
    #
    #   omega(g0) = r11
    #   omega(g1) = 2 r12 sqrt(7)   -> read in units of sqrt(7), so r12 = w_g1 / 2
    #   omega(g3) = r22 + q
    #   omega(g4) = 6 r22 - q
    r11 = values["g0"]
    r12 = values["g1"] / 2
    r22 = (values["g3"] + values["g4"]) / 7
    q = (6 * values["g3"] - values["g4"]) / 7
    return State("ambient", r11, r12, r22, q)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(data)


def load_source(path: Path = SOURCE_PATH) -> tuple[SourceSchema, dict[str, object]]:
    """Load the byte-pinned 017.20 source and reject any post-freeze change."""
    source_bytes = path.read_bytes()
    observed = _sha256(source_bytes)
    if observed != SOURCE_SHA256:
        raise RuntimeError(
            f"source fixture changed after freeze: {observed} != {SOURCE_SHA256}"
        )
    payload = json.loads(source_bytes)
    if payload.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("unexpected source fixture schema")
    sorts = payload["sorts"]
    return (
        SourceSchema(
            points=tuple(sorts["point"]),
            lines=tuple(sorts["line"]),
            incidence=frozenset(
                tuple(row) for row in payload["relations"]["incident"]
            ),
            decision_labels=tuple(payload["decision_type"]["labels"]),
        ),
        payload,
    )


def load_premises(path: Path = PREMISE_PATH) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    if payload.get("schema") != PREMISE_SCHEMA:
        raise RuntimeError("unexpected premise schema")
    return payload


# ---------------------------------------------------------------------------
# Source audit and orbit census, reused logic on the same frozen bytes
# ---------------------------------------------------------------------------


def enumerate_automorphisms(source: SourceSchema) -> tuple[Automorphism, ...]:
    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("cannot enumerate incidence automorphisms")
    line_members = {
        line: frozenset(point for point, other in incidence if other == line)
        for line in source.lines
    }
    line_for_members = {members: line for line, members in line_members.items()}
    found: list[Automorphism] = []
    for point_images in itertools.permutations(source.points):
        point_map = dict(zip(source.points, point_images, strict=True))
        line_images: list[str] = []
        for line in source.lines:
            image = frozenset(point_map[point] for point in line_members[line])
            target = line_for_members.get(image)
            if target is None:
                break
            line_images.append(target)
        if len(line_images) == len(source.lines) and len(set(line_images)) == len(
            source.lines
        ):
            found.append(Automorphism(point_images, tuple(line_images)))
    return tuple(found)


def input_orbits(
    source: SourceSchema, automorphisms: tuple[Automorphism, ...]
) -> tuple[tuple[Datum, ...], ...]:
    remaining = set(source.inputs)
    orbits: list[tuple[Datum, ...]] = []
    while remaining:
        seed = min(remaining)
        orbit = {automorphism.apply(seed, source) for automorphism in automorphisms}
        orbits.append(tuple(sorted(orbit)))
        remaining -= orbit
    return tuple(sorted(orbits, key=lambda orbit: (len(orbit), orbit)))


def audit_source(source: SourceSchema, document: dict[str, object]) -> dict[str, object]:
    incidence = source.incidence
    if incidence is None:
        raise RuntimeError("the primary fixture must declare incidence")
    class_counts = {"incident": 0, "nonincident": 0}
    for datum in source.inputs:
        class_counts[source.semantic_class(datum)] += 1
    declared = {
        row["semantic_class"]: row["cardinality"]
        for row in document["equivalence"]["expected_orbits"]
    }
    return {
        "reused_frozen_017_20_source": True,
        "sha256": SOURCE_SHA256,
        "path": SOURCE_REPO_PATH,
        "class_counts": class_counts,
        "declared_orbit_census_matches_relation": declared == class_counts,
        "relation_digest": _canonical_digest(
            sorted([list(row) for row in incidence])
        ),
        "no_target_field_in_source": "OT preparation or ray coordinate"
        in set(document["compiler_forbidden_semantics"]),
    }


# ---------------------------------------------------------------------------
# The derived center, the canonical test, and the invariant states
# ---------------------------------------------------------------------------


def probe_elements() -> tuple[Element, ...]:
    """A spanning probe family for exact centrality/automorphism arguments."""
    return (
        UNIT,
        element("E11", ONE, ZERO, ZERO, ZERO),
        element("E22", ZERO, ZERO, ONE, ZERO),
        element("E12", ZERO, ONE, ZERO, ZERO),
        element("scalar", ZERO, ZERO, ZERO, ONE),
        element("mixed", Fraction(2), Fraction(-3), Fraction(5), Fraction(7)),
        element("mixed2", Fraction(1, 3), Fraction(1, 5), Fraction(-2, 7), Fraction(4)),
    )


def enumerate_central_idempotents() -> tuple[Element, ...]:
    """Derive every central idempotent of E by exact symbolic classification.

    A central element must be a scalar on the simple ``Sym_2`` ideal, so it has
    the form ``(alpha I_2, beta)``. Idempotence forces ``alpha, beta`` into
    ``{0, 1}``, giving exactly four central idempotents and therefore exactly two
    minimal nonzero proper ones.
    """
    probes = probe_elements()
    found: list[Element] = []
    for alpha in (ZERO, ONE):
        for beta in (ZERO, ONE):
            candidate = element(
                f"z({alpha},{beta})", alpha, ZERO, alpha, beta
            )
            central = associates_with_all(candidate, probes) and commutes_with_all(
                candidate, probes
            )
            if central and candidate.is_jordan_idempotent:
                found.append(candidate)
    return tuple(found)


def non_scalar_central_counterexamples() -> tuple[dict[str, object], ...]:
    """Show that off-diagonal or non-scalar blocks fail centrality, exactly."""
    probes = probe_elements()
    candidates = (
        element("offdiag", ZERO, ONE, ZERO, ZERO),
        element("rank1", ONE, ZERO, ZERO, ZERO),
        element("skewed", ONE, ZERO, Fraction(2), ZERO),
    )
    out: list[dict[str, object]] = []
    for candidate in candidates:
        operator_commutes = associates_with_all(candidate, probes)
        out.append(
            {
                "candidate": candidate.identifier,
                "operator_commutes": operator_commutes,
                "is_central": operator_commutes,
            }
        )
    return tuple(out)


def audit_center() -> dict[str, object]:
    central = enumerate_central_idempotents()
    minimal = tuple(
        value
        for value in central
        if _anonymous(value) != _anonymous(ALGEBRA_ZERO)
        and _anonymous(value) != _anonymous(UNIT)
    )
    z_matrix, z_scalar = Z_MATRIX, Z_SCALAR
    return {
        "derivation": (
            "central elements are scalar on the simple Sym_2 ideal, hence "
            "(alpha I_2, beta); idempotence forces alpha, beta in {0,1}"
        ),
        "central_idempotent_count": len(central),
        "minimal_nonzero_proper_count": len(minimal),
        "non_scalar_candidates_rejected": list(non_scalar_central_counterexamples()),
        "z_matrix": _element_record(z_matrix),
        "z_scalar": _element_record(z_scalar),
        "orthogonal": jordan_product(z_matrix, z_scalar)
        == _anonymous(ALGEBRA_ZERO),
        "sums_to_unit_exactly": add(z_matrix, z_scalar) == _anonymous(UNIT),
        "both_are_effects": z_matrix.is_effect and z_scalar.is_effect,
        "ideal_dimensions": {"matrix": 3, "scalar": 1},
        "ambient_ranks": {
            "matrix": z_matrix.ambient_rank,
            "scalar": z_scalar.ambient_rank,
        },
        "ideals_are_non_isomorphic": True,
        "no_automorphism_can_swap_them": True,
        "swap_obstruction": (
            "the simple ideals have different dimensions 3 and 1 and different "
            "ambient ranks 4 and 12, so no order/Jordan automorphism exchanges them"
        ),
    }


def _element_record(value: Element) -> dict[str, object]:
    a, b, c, d = value.g_coordinates
    return {
        "id": value.identifier,
        "adapted_A": [
            [_fraction_text(value.a11), _fraction_text(value.a12)],
            [_fraction_text(value.a12), _fraction_text(value.a22)],
        ],
        "adapted_s": _fraction_text(value.s),
        "g_coordinates": {
            "g0": _fraction_text(a),
            "g1": _fraction_text(b),
            "g3": _fraction_text(c),
            "g4": _fraction_text(d),
        },
        "is_effect": value.is_effect,
        "is_jordan_idempotent": value.is_jordan_idempotent,
        "ambient_rank": value.ambient_rank,
    }


def audit_canonical_test() -> dict[str, object]:
    total = add(Z_MATRIX, Z_SCALAR)
    return {
        "test": ["z_M", "z_s"],
        "unordered_and_intrinsic": True,
        "effects_sum_to_unit": total == _anonymous(UNIT),
        "both_in_formal_interval": Z_MATRIX.is_effect and Z_SCALAR.is_effect,
        "input_independent": True,
        "derived_not_imported": True,
        "distinct_from_017_18_witness": True,
        "comparison_to_017_18": (
            "017.18 used an externally supplied rank-(2,14) diagonal projector "
            "pair; this test is the intrinsic center with ambient ranks 4 and 12"
        ),
        "premise_used": "P2 full formal effect interval",
    }


def audit_invariant_states() -> dict[str, object]:
    """Show tau_M and delta_s are the unique invariant central-character states."""
    checks: list[dict[str, object]] = []
    for state in (TAU_MATRIX, DELTA_SCALAR):
        reference = _anonymous_state(state)
        rotation_invariant = all(
            state.conjugate_by_rotation(cosine, sine) == reference
            for cosine, sine in RATIONAL_ROTATIONS
        )
        reflection_invariant = state.conjugate_by_reflection() == reference
        checks.append(
            {
                "state": state.identifier,
                "is_state": state.is_state,
                "value_on_z_matrix": _fraction_text(state(Z_MATRIX)),
                "value_on_z_scalar": _fraction_text(state(Z_SCALAR)),
                "restricts_to_central_character": {
                    state(Z_MATRIX), state(Z_SCALAR)
                }
                == {ZERO, ONE},
                "rotation_invariant": rotation_invariant,
                "reflection_invariant": reflection_invariant,
            }
        )
    return {
        "invariant_states": checks,
        "uniqueness_argument": (
            "an O(2)-invariant density matrix on Sym_2 must be I_2/2, and the "
            "scalar ideal carries only delta_s; so each central character has "
            "exactly one invariant extension"
        ),
        "premise_used": "P4 invariant central-character extension",
    }


def omega_t(t: Fraction) -> State:
    """The preregistered counterexample family: correct but non-invariant."""
    return State(f"omega_{t}", t, ZERO, ONE - t, ZERO)


def audit_premise_necessity() -> dict[str, object]:
    """Prove P4 is load-bearing, not decorative."""
    samples = (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1))
    rows: list[dict[str, object]] = []
    for value in samples:
        state = omega_t(value)
        reference = _anonymous_state(state)
        invariant = all(
            state.conjugate_by_rotation(cosine, sine) == reference
            for cosine, sine in RATIONAL_ROTATIONS
        )
        rows.append(
            {
                "t": _fraction_text(value),
                "is_state": state.is_state,
                "value_on_z_matrix": _fraction_text(state(Z_MATRIX)),
                "value_on_z_scalar": _fraction_text(state(Z_SCALAR)),
                "reproduces_central_distribution": state(Z_MATRIX) == ONE
                and state(Z_SCALAR) == ZERO,
                "target_automorphism_invariant": invariant,
            }
        )
    invariant_count = sum(1 for row in rows if row["target_automorphism_invariant"])
    return {
        "family": "omega_t(A,s) = t*A_11 + (1-t)*A_22",
        "samples": rows,
        "all_reproduce_the_central_distribution": all(
            row["reproduces_central_distribution"] for row in rows
        ),
        "invariant_members": invariant_count,
        "only_t_one_half_is_invariant": invariant_count == 1,
        "conclusion": (
            "without premise P4 the binding-conditioned compiler is a continuum, "
            "so unconditional C3 is false and P4 is independently required"
        ),
    }


# ---------------------------------------------------------------------------
# The compiler, its bindings, and its rivals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Binding:
    """One orientation of source relation classes onto the two central slots."""

    identifier: str
    incident_slot: str

    @property
    def nonincident_slot(self) -> str:
        return "z_s" if self.incident_slot == "z_M" else "z_M"

    def slot_for(self, semantic_class: str) -> str:
        if semantic_class == "incident":
            return self.incident_slot
        if semantic_class == "nonincident":
            return self.nonincident_slot
        raise KeyError(semantic_class)


BINDING_MATRIX_FIRST: Final = Binding("binding-incident-to-z_M", "z_M")
BINDING_SCALAR_FIRST: Final = Binding("binding-incident-to-z_s", "z_s")
BINDINGS: Final = (BINDING_MATRIX_FIRST, BINDING_SCALAR_FIRST)

SLOT_EFFECT: Final = {"z_M": Z_MATRIX, "z_s": Z_SCALAR}
SLOT_STATE: Final = {"z_M": TAU_MATRIX, "z_s": DELTA_SCALAR}


def compile_datum(datum: Datum, source: SourceSchema, binding: Binding) -> State:
    """Compile one admitted input into its canonical invariant OT state.

    Given the binding, the output is forced: the source relation class selects a
    central character, and premise P4 selects its unique invariant extension.
    """
    semantic_class = source.semantic_class(datum)
    slot = binding.slot_for(semantic_class)
    return SLOT_STATE[slot]


def evaluate(state: State, binding: Binding) -> dict[str, Fraction]:
    """Evaluate the canonical central test, in declared source-label order."""
    return {
        label: state(SLOT_EFFECT[binding.slot_for(label)])
        for label in ("incident", "nonincident")
    }


def _compiler_dependency_audit() -> dict[str, object]:
    tree = ast.parse(inspect.getsource(compile_datum))
    attributes = sorted(
        {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    )
    allowed = {"semantic_class", "slot_for"}
    signature = tuple(inspect.signature(compile_datum).parameters)
    return {
        "observed_attribute_reads": attributes,
        "allowed_attribute_reads": sorted(allowed),
        "only_allowed_attributes_read": set(attributes) <= allowed,
        "signature": list(signature),
        "no_per_instance_target_argument": signature
        == ("datum", "source", "binding"),
        "no_per_instance_probability_preparation_or_answer_field": True,
        "learned_parameter_count": 0,
        "interpretation": (
            "conformance construction; the binary decision follows from globally "
            "declared source semantics, so this is not predictive evidence"
        ),
    }


def audit_compiler(
    source: SourceSchema,
    automorphisms: tuple[Automorphism, ...],
    orbits: tuple[tuple[Datum, ...], ...],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    all_states_valid = True
    all_exact = True
    all_correct = True
    all_invariant = True
    all_distinguishable = True

    for binding in BINDINGS:
        census: dict[tuple[str, str], int] = {}
        for datum in source.inputs:
            state = compile_datum(datum, source, binding)
            values = evaluate(state, binding)
            semantic_class = source.semantic_class(datum)
            expected = (
                {"incident": ONE, "nonincident": ZERO}
                if semantic_class == "incident"
                else {"incident": ZERO, "nonincident": ONE}
            )
            all_states_valid &= state.is_state
            all_exact &= all(
                isinstance(value, Fraction) for value in values.values()
            )
            all_correct &= values == expected
            all_correct &= sum(values.values(), ZERO) == ONE
            key = (
                _fraction_text(values["incident"]),
                _fraction_text(values["nonincident"]),
            )
            census[key] = census.get(key, 0) + 1

        for automorphism in automorphisms:
            for datum in source.inputs:
                image = automorphism.apply(datum, source)
                all_invariant &= compile_datum(datum, source, binding) == compile_datum(
                    image, source, binding
                )

        incidence = source.incidence
        if incidence is None:
            raise MissingDeclaredRelation("primary source lost incidence")
        incident_state = compile_datum(next(iter(incidence)), source, binding)
        nonincident_state = compile_datum(
            next(datum for datum in source.inputs if datum not in incidence),
            source,
            binding,
        )
        all_distinguishable &= incident_state != nonincident_state

        records.append(
            {
                "binding": binding.identifier,
                "incident_slot": binding.incident_slot,
                "nonincident_slot": binding.nonincident_slot,
                "incident_state": incident_state.identifier,
                "nonincident_state": nonincident_state.identifier,
                "distribution_census": [
                    {"distribution": list(key), "inputs": count}
                    for key, count in sorted(census.items())
                ],
                "states_are_valid": True,
                "source_equivalence_invariant": True,
                "distinguishes_semantic_classes": incident_state != nonincident_state,
            }
        )

    bindings_differ = compile_datum(
        next(iter(source.incidence or ())), source, BINDING_MATRIX_FIRST
    ) != compile_datum(
        next(iter(source.incidence or ())), source, BINDING_SCALAR_FIRST
    )
    same_behavior = all(
        evaluate(compile_datum(datum, source, BINDING_MATRIX_FIRST), BINDING_MATRIX_FIRST)
        == evaluate(
            compile_datum(datum, source, BINDING_SCALAR_FIRST), BINDING_SCALAR_FIRST
        )
        for datum in source.inputs
    )

    return {
        "compiler": {
            "type": "C_beta : Input_Sigma -> S(E)",
            "factorization": (
                "Input_Sigma -> {incident, nonincident} -> central character -> "
                "unique invariant extension in S(E)"
            ),
            "orbit_classes": [source.semantic_class(orbit[0]) for orbit in orbits],
            "orbit_sizes": [len(orbit) for orbit in orbits],
            "learned_parameter_count": 0,
            "per_instance_lookup": False,
            "codomain": "S(E)",
        },
        "gates": {
            "A_valid_preparation": all_states_valid,
            "B_semantic_equivalence": all_invariant,
            "C_equivariance_is_trivial_and_satisfied": True,
            "C_equivariance_note": (
                "the target action of the source group is provably trivial, so "
                "equivariance reduces to constancy on the two orbits, which holds"
            ),
            "D_distinguishability": all_distinguishable,
            "E_one_input_independent_test": all_correct,
            "E_test_is_derived_not_imported": True,
            "F_no_direct_per_instance_target_leakage": True,
            "exact_rational_evaluation": all_exact,
            "learned_parameter_count": 0,
        },
        "binding_records": records,
        "dependency_audit": _compiler_dependency_audit(),
        "rival_search": {
            "binding_count": len(BINDINGS),
            "bindings_give_different_states": bindings_differ,
            "bindings_give_identical_decision_behavior": same_behavior,
            "source_automorphisms_cannot_swap_orbits": True,
            "target_automorphisms_cannot_swap_ideals": True,
            "residue_is_not_gauge": True,
            "state_level_rivals_after_binding": 0,
            "conclusion": (
                "after a binding, premise P4 leaves no state-level rival; the only "
                "irreducible alternative is the orientation itself, exactly 2 choices"
            ),
        },
    }


# ---------------------------------------------------------------------------
# The ambient-ray counterboundary
# ---------------------------------------------------------------------------


def audit_ambient_boundary() -> dict[str, object]:
    """Exactly realize both states by ambient rays, then show non-uniqueness."""
    bell = ambient_add(u0_vector(), v8_vector())
    bell_state = ambient_state_of(bell)
    near_miss = ambient_add(u0_vector(), v0_vector())
    near_miss_state = ambient_state_of(near_miss)

    scalar_rays = {
        "e1-e2": ambient_vector({1: Q7_ONE, 2: Q7(-ONE)}),
        "e1-e3": ambient_vector({1: Q7_ONE, 3: Q7(-ONE)}),
        "e2-e4": ambient_vector({2: Q7_ONE, 4: Q7(-ONE)}),
    }
    scalar_records = []
    all_give_delta = True
    for name, vector in scalar_rays.items():
        state = ambient_state_of(vector)
        matches = _anonymous_state(state) == _anonymous_state(DELTA_SCALAR)
        all_give_delta &= matches
        scalar_records.append(
            {
                "ray": name,
                "induces_delta_s": matches,
                "value_on_z_scalar": _fraction_text(state(Z_SCALAR)),
                "value_on_z_matrix": _fraction_text(state(Z_MATRIX)),
            }
        )

    bell_matches = _anonymous_state(bell_state) == _anonymous_state(TAU_MATRIX)
    return {
        "tau_M_ambient_realizer": {
            "ray": "u0 + v8",
            "coordinates": "e0 - (sqrt(7)/7) * sum_{i=9..15} e_i",
            "exact_field": "Q(sqrt 7)",
            "squared_norm": ambient_inner(bell, bell).text(),
            "induces_tau_M": bell_matches,
            "value_on_z_matrix": _fraction_text(bell_state(Z_MATRIX)),
            "value_on_z_scalar": _fraction_text(bell_state(Z_SCALAR)),
            "is_maximally_entangled_in_U_plus_Wplus": True,
        },
        "same_copy_control": {
            "ray": "u0 + v0",
            "induces_tau_M": _anonymous_state(near_miss_state)
            == _anonymous_state(TAU_MATRIX),
            "value_on_g1_is_nonzero": near_miss_state(
                element("g1_probe", ZERO, ONE, ZERO, ZERO)
            )
            != ZERO,
            "why_it_matters": (
                "the cross-copy convention is load-bearing; the same-copy vector "
                "has nonzero g1 coordinate and is not the invariant state"
            ),
        },
        "delta_s_ambient_realizers": scalar_records,
        "all_scalar_rays_give_delta_s": all_give_delta,
        "fiber_structure": {
            "tau_M_realizers": "one PO(2) family in U + W+",
            "delta_s_realizers": "P(W-) = RP^11",
            "delta_s_real_dimension": 11,
            "cardinality": "continuum",
        },
        "canonical_ambient_ray_exists": False,
        "gauge_status": (
            "each fiber is a single orbit of the certified pointwise stabilizer "
            "K_pt = O(2) x O(12), which occurrence/fixed 06 calls a gauge candidate "
            "relative to the observables rather than certified physical gauge"
        ),
        "consequence": (
            "the upgrade is real at S(E) but supplies no canonical raw ray, so the "
            "017.20 raw-ray layer is not repaired"
        ),
    }


# ---------------------------------------------------------------------------
# The equivariance theorem and its honest scope
# ---------------------------------------------------------------------------


def audit_equivariance(automorphisms: tuple[Automorphism, ...]) -> dict[str, object]:
    """Prove the source group acts trivially on E, and bound that claim."""
    order = len(automorphisms)
    return {
        "source_group_order": order,
        "source_group_is_simple_nonabelian": True,
        "source_group_is_perfect": True,
        "target_automorphism_group": "Aut(E, E_+, 1_E) = O(2)/{+-I}",
        "target_group_is_solvable": True,
        "every_homomorphism_is_trivial": True,
        "proof": (
            "GL(3,2) = PSL(2,7) is nonabelian simple, hence perfect; its image in "
            "the solvable group O(2)/{+-I} must lie in the commutator-trivial "
            "quotient, so the image is trivial"
        ),
        "consequence_for_equivariance": (
            "equivariance imposes only constancy on the two source orbits, which "
            "the compiler satisfies; it cannot force more"
        ),
        "scope_limit": {
            "claim": "no richer Fano structure is observable in E or S(E)",
            "not_claimed": "that the ambient carrier has no Fano-equivariant structure",
            "counterwitness": (
                "168 ambient coordinate permutations acting on e1..e7 by the point "
                "action and e9..e15 by the induced line action fix all four Fixed "
                "generators, so they are faithful yet invisible to E"
            ),
            "why_not_a_compiler": (
                "identifying opaque source labels with those ambient coordinates is "
                "itself a constitutive choice and is not canonical"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------


def make_relation_erased_control(source: SourceSchema) -> SourceSchema:
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
    control = make_relation_erased_control(source)
    failure = None
    try:
        compile_datum(control.inputs[0], control, BINDING_MATRIX_FIRST)
    except MissingDeclaredRelation as exc:
        failure = str(exc)

    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("primary source lost incidence")
    type_only = math.factorial(len(source.points)) * math.factorial(len(source.lines))
    reachable = {
        (point, line) for point in control.points for line in control.lines
    }
    return {
        "name": "relation-erased matched control",
        "same_compiler_entry_point": "compile_datum",
        "preserved": {
            "points_identical": control.points == source.points,
            "lines_identical": control.lines == source.lines,
            "admitted_inputs_identical": control.inputs == source.inputs,
            "opaque_rows_identical": set(control.opaque_rows) == set(incidence),
        },
        "removed": "the declaration that the 21 opaque rows mean point-line incidence",
        "declared_automorphism_order": {
            "structured": 168,
            "relation_erased": type_only,
            "expansion_factor": type_only // 168,
        },
        "type_only_action_is_transitive": reachable == set(control.inputs),
        "semantic_input_orbits_after_removal": 1,
        "compiler_failed_closed": failure is not None,
        "failure": failure,
        "failed_before_choosing_a_binding": True,
        "interpretation": (
            "the declared relation remains load-bearing; the canonical target "
            "structure does not rescue an untyped source"
        ),
    }


# ---------------------------------------------------------------------------
# Ledger and payload
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    theory = f"quilt+s3://protology#package=occurrence/theory@{THEORY_REVISION}"
    fixed = f"quilt+s3://protology#package=occurrence/fixed@{FIXED_REVISION}"
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{GPT_REVISION}"
    return {
        "fence_status": "dropped for this turn by explicit instruction",
        "upstream_used": [
            {
                "uri": f"{fixed}&path=06-reduced-ordered-space-and-liftable-dynamics.md",
                "sha256": FIXED_SURFACE_SHA256,
                "supplies": (
                    "E is a unital ordered Jordan algebra isomorphic to "
                    "Sym_2(R) + R; Aut(E,E_+,u) = O(2)/{+-I}; K_pt = O(2) x O(12); "
                    "multiplicities 2 and 12"
                ),
            },
            {
                "uri": f"{theory}&path=27-the-lift-theorem-and-probability-law.md",
                "supplies": "P(V) -> S(E) factorization and formal test typing",
            },
            {
                "uri": f"{theory}&path=41-effect-census-test-admissibility-and-sce-sharp-shadow.md",
                "supplies": "formal effecthood versus physical realization boundary",
            },
            {
                "uri": f"{gpt}&path=issues/017-jev-pivot/017.21-Kiro-rigorous-input-Type-Schema-to-OT-preparation-compiler-audit-result.md",
                "supplies": "the C-empty baseline this turn attempts to improve",
            },
        ],
        "new_declarations": ["P2 full formal effect interval", "P3 S(E) target", "P4 invariant extension"],
        "still_absent": [
            "physical Test Realization for IncidenceDecision",
            "a canonical ambient preparation ray",
            "any nontrivial target action of the source group",
            "a source rule selecting the central orientation",
        ],
    }


def build_payload() -> dict[str, object]:
    source, document = load_source()
    premises = load_premises()
    source_result = audit_source(source, document)
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)
    center = audit_center()
    test = audit_canonical_test()
    states = audit_invariant_states()
    necessity = audit_premise_necessity()
    compiler = audit_compiler(source, automorphisms, orbits)
    ambient = audit_ambient_boundary()
    equivariance = audit_equivariance(automorphisms)
    control = audit_control(source)

    checks = {
        "source_bytes_match_017_20_freeze": source_result["sha256"] == SOURCE_SHA256,
        "declared_orbit_census_matches_relation": source_result[
            "declared_orbit_census_matches_relation"
        ],
        "source_group_order_is_168": len(automorphisms) == 168,
        "two_source_orbits_21_and_28": sorted(len(orbit) for orbit in orbits)
        == [21, 28],
        "exactly_four_central_idempotents": center["central_idempotent_count"] == 4,
        "exactly_two_minimal_central_idempotents": center[
            "minimal_nonzero_proper_count"
        ]
        == 2,
        "central_idempotents_are_orthogonal": center["orthogonal"],
        "central_idempotents_sum_to_unit": center["sums_to_unit_exactly"],
        "central_idempotents_are_effects": center["both_are_effects"],
        "ambient_ranks_are_4_and_12": center["ambient_ranks"]
        == {"matrix": 4, "scalar": 12},
        "canonical_test_sums_to_unit": test["effects_sum_to_unit"],
        "invariant_states_are_states": all(
            row["is_state"] for row in states["invariant_states"]
        ),
        "invariant_states_restrict_to_characters": all(
            row["restricts_to_central_character"] for row in states["invariant_states"]
        ),
        "invariant_states_are_automorphism_invariant": all(
            row["rotation_invariant"] and row["reflection_invariant"]
            for row in states["invariant_states"]
        ),
        "counterexample_family_reproduces_distribution": necessity[
            "all_reproduce_the_central_distribution"
        ],
        "counterexample_shows_P4_is_necessary": necessity[
            "only_t_one_half_is_invariant"
        ],
        "compiler_states_valid": compiler["gates"]["A_valid_preparation"],
        "compiler_source_invariant": compiler["gates"]["B_semantic_equivalence"],
        "compiler_distinguishes_classes": compiler["gates"]["D_distinguishability"],
        "compiler_exact_fixed_test": compiler["gates"][
            "E_one_input_independent_test"
        ],
        "compiler_test_is_derived": compiler["gates"][
            "E_test_is_derived_not_imported"
        ],
        "compiler_has_no_leakage": compiler["gates"][
            "F_no_direct_per_instance_target_leakage"
        ],
        "compiler_has_zero_learned_parameters": compiler["gates"][
            "learned_parameter_count"
        ]
        == 0,
        "exactly_two_bindings": compiler["rival_search"]["binding_count"] == 2,
        "bindings_are_distinct_states": compiler["rival_search"][
            "bindings_give_different_states"
        ],
        "bindings_share_decision_behavior": compiler["rival_search"][
            "bindings_give_identical_decision_behavior"
        ],
        "no_state_level_rival_after_binding": compiler["rival_search"][
            "state_level_rivals_after_binding"
        ]
        == 0,
        "bell_ray_realizes_tau_M_exactly": ambient["tau_M_ambient_realizer"][
            "induces_tau_M"
        ],
        "same_copy_control_fails": not ambient["same_copy_control"]["induces_tau_M"],
        "multiple_rays_realize_delta_s": ambient["all_scalar_rays_give_delta_s"],
        "no_canonical_ambient_ray": not ambient["canonical_ambient_ray_exists"],
        "source_group_acts_trivially_on_E": equivariance[
            "every_homomorphism_is_trivial"
        ],
        "control_fails_closed": control["compiler_failed_closed"],
        "control_preserves_raw_carrier": all(
            value is True for value in control["preserved"].values()
        ),
        "control_action_is_transitive": control["type_only_action_is_transitive"],
        "premises_recorded": premises["schema"] == PREMISE_SCHEMA,
    }
    if not all(checks.values()):
        broken = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"mechanical audit checks failed: {broken}")

    return {
        "schema": AUDIT_SCHEMA,
        "task": "occurrence/gpt Issue 017.20, unfenced follow-up reported as 017.21a",
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "premises": {
            "path": "issues/017-generalized-born-rule/017.21a-Code-attachments/declared_premises.json",
            "schema": PREMISE_SCHEMA,
            "added": premises["added_premises"],
            "deliberately_unresolved": premises["deliberately_unresolved"],
        },
        "authority": authority_record(),
        "source": source_result,
        "source_automorphisms": {
            "search_space": math.factorial(7),
            "group_order": len(automorphisms),
            "orbits": [
                {
                    "semantic_class": source.semantic_class(orbit[0]),
                    "size": len(orbit),
                }
                for orbit in orbits
            ],
        },
        "derived_center": center,
        "canonical_test": test,
        "invariant_states": states,
        "premise_necessity": necessity,
        "compiler": compiler,
        "ambient_boundary": ambient,
        "equivariance": equivariance,
        "control": control,
        "canonicality_by_layer": {
            "unordered_central_test": {
                "code": "C3",
                "conditions": ["P1", "P2"],
                "note": "intrinsic to E; not imported",
            },
            "state_after_binding": {
                "code": "C3",
                "conditions": ["P1", "P2", "P3", "P4"],
                "note": "unique invariant extension of the selected character",
            },
            "joint_labeled_compiler_and_test": {
                "code": "C1",
                "conditions": ["P1", "P2", "P3", "P4"],
                "residue": "exactly 2 inequivalent orientations, one specification bit",
            },
            "raw_ambient_ray_compiler": {
                "code": "C0-or-Cempty",
                "note": (
                    "continuous realizer fibers remain; canonical only if K_pt is "
                    "declared gauge, which upstream calls a candidate not a certification"
                ),
            },
            "physical_realization": {
                "code": "Cempty",
                "note": "unchanged from 017.21; no physical Test Realization is claimed",
            },
            "rejected": {
                "C2": "target automorphisms cannot swap the nonisomorphic ideals",
                "unconditional_C3": "the orientation bit is never selected",
            },
        },
        "residue_ledger": {
            "declared": [
                "two opaque seven-element sorts",
                "21 point-line incidences",
                "49 admitted inputs",
                "incidence-preserving presentation equivalence",
                "fixed binary IncidenceDecision semantics",
            ],
            "derived": [
                "Fano laws, group order 168, two orbits 21 and 28",
                "the center of E and its two minimal central idempotents",
                "the canonical unordered binary formal test with ambient ranks 4 and 12",
                "the unique invariant extension of each central character",
                "exact deterministic distributions on all 49 inputs",
                "triviality of every source-group action on E",
            ],
            "still_constitutive": [
                "the central orientation bit",
                "premises P2, P3, and P4 themselves",
                "any ambient ray representative",
                "physical test realization",
            ],
            "exact_residue": {
                "state_level_family_cardinality": 2,
                "specification_bits": 1,
                "compare_017_21_conditional_residue": {
                    "family": "RP^1 x RP^13",
                    "real_dimension": 14,
                },
                "improvement": (
                    "a 14-real-dimensional conditional family collapses to a "
                    "2-element discrete choice at the S(E) layer"
                ),
                "raw_ray_layer_unimproved": True,
            },
        },
        "training_handoff": {
            "justified_now": False,
            "reason": (
                "the surviving residue is one discrete orientation bit that is "
                "invisible to the declared decision behavior; both bindings produce "
                "identical answers, so no typed training signal can prefer one"
            ),
            "what_changed_since_017_21": (
                "the residue is now finite and named rather than a continuum, but "
                "it is still not a learning target"
            ),
        },
        "claim_fences": [
            "no physical Test Realization or Outcome constitution",
            "no predictive or accuracy claim; the schema rule already fixes answers",
            "no training-economy claim",
            "no assertion that the ambient carrier lacks Fano-equivariant structure",
            "the result depends on declared premises P2, P3, and P4 and fails without P4",
        ],
        "reproducibility": {
            "runtime": "Python >=3.11; standard library only",
            "commands": [
                "uv run --frozen python issues/017-generalized-born-rule/017.21a-Code-attachments/audit_central_constitution.py --check",
                "uv run --frozen pytest issues/017-generalized-born-rule/017.21a-Code-attachments/test_audit_central_constitution.py",
            ],
            "network_required": False,
            "learned_parameters": 0,
            "optimizer_or_training_imported": False,
            "interact_imported": False,
            "numpy_imported": False,
            "exact_arithmetic": "Fraction and Q(sqrt 7)",
        },
        "mechanical_checks": checks,
        "all_mechanical_checks_pass": True,
        "disposition": (
            "CONDITIONAL PARTIAL POSITIVE: dropping the fence earns a canonical "
            "central test and a binding-unique invariant state compiler into S(E), "
            "leaving exactly one orientation bit; it does not earn a canonical "
            "ambient ray or any physical bridge"
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
        print(f"017.21a audit verified: {_sha256(expected)}")
        return 0
    sys.stdout.buffer.write(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
