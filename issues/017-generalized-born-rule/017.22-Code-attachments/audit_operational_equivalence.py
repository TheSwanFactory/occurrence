"""Exact Issue 017.22 operational TDM equivalence and S(E) state-boundary audit.

017.21a earned a canonical central test and a binding-unique state compiler into
``S(E)``, leaving one orientation bit and no canonical ambient ray. This module
decides what that means operationally:

* which 017.21a premises were already certified Theory rather than declarations;
* what public object a typed decision model's realization actually induces;
* the exact observational equivalences at the state and whole-realization layers;
* whether the orientation bit and the ambient-ray fibers are observable at the
  declared typed boundary, without mislabeling either as certified gauge;
* whether ``S(E)`` is the correct one-shot operational state carrier;
* and whether the ``Aut(E, E_+, 1_E)``-invariant states are exactly one
  one-parameter family.

Nothing here is trained, fitted, or physical. There is no optimizer, no learned
parameter, no NumPy, and no Interact import. Every number is exact: ``Fraction``
arithmetic in the adapted ``(A, s)`` coordinates and exact ``Q(sqrt 7)``
arithmetic for the ambient 16-dimensional ray evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

AUDIT_SCHEMA: Final = "gpt-01722-operational-equivalence-audit/v1"
TABLE_SCHEMA: Final = "gpt-01722-orientation-behavior-table/v1"
SOURCE_SCHEMA: Final = "gpt-01720-fano-incidence-source/v1"
PRIOR_PREMISE_SCHEMA: Final = "gpt-01721a-unfenced-premises/v1"
PRIOR_AUDIT_SCHEMA: Final = "gpt-01721a-central-constitution-audit/v1"

SOURCE_SHA256: Final = (
    "20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c"
)
PRIOR_PREMISES_SHA256: Final = (
    "3cb2b3f4decba679dffa05ebe72ded9a7b9645b7df8bb4c1e2bd87cd4198e921"
)
PRIOR_AUDIT_SHA256: Final = (
    "e9581c25db9f82dbbba6bb1fe800e3d3b94c92d20fc394ab8adb48ec0d1ff59c"
)

PRIOR_RESULT_COMMIT: Final = "349e4db9d46be431e9f5f40808f594ae108ab6a8"
TASK_GPT_REVISION: Final = (
    "19a52d803acd6a9f19e7f4aee04d398633028179aad20d916541884cb91774f3"
)
PRIMARY_GPT_AUTHORITY: Final = (
    "70617692cbd4f5dea992955307a2afa76d3a18d0d3da9bfc26a5477fb8cc7604"
)
THEORY_REVISION_CONSULTED: Final = (
    "27dda92bc42ea84dc7ff384deac86b6a44ada3370bd2a638aaabcd91f7621e09"
)
THEORY_REVISION_CITED_BY_017_21A: Final = (
    "b14aeb1bb9e07662e0e6fdde697f728cae3457324c0e2abd0cfd42975b8aac9a"
)
FIXED_REVISION: Final = (
    "f11ea77410fc79c8bacd8ba44633270786735ecf08cfff05e855613936b1760b"
)
FIXED_SURFACE_SHA256: Final = (
    "17e07af79a56cea92343a629bf901ef8c930e5a7bfe0572396968fb308a100ce"
)

HERE: Final = Path(__file__).resolve().parent
SOURCE_PATH: Final = HERE.parent / "017.20-Code-attachments" / "source_fixture.json"
PRIOR_PREMISES_PATH: Final = (
    HERE.parent / "017.21a-Code-attachments" / "declared_premises.json"
)
PRIOR_AUDIT_PATH: Final = (
    HERE.parent / "017.21a-Code-attachments" / "central_constitution_audit.json"
)
ARTIFACT_PATH: Final = HERE / "operational_equivalence_audit.json"
TABLE_PATH: Final = HERE / "orientation_behavior_table.json"
TABLE_REPO_PATH: Final = (
    "issues/017-generalized-born-rule/017.22-Code-attachments/"
    "orientation_behavior_table.json"
)

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)
AMBIENT_DIMENSION: Final = 16
SYM_AMBIENT_DIMENSION: Final = AMBIENT_DIMENSION * (AMBIENT_DIMENSION + 1) // 2
EFFECT_ALGEBRA_DIMENSION: Final = 4
MULTIPLICITY_SCALAR: Final = 12

Datum = tuple[str, str]


class MissingDeclaredRelation(RuntimeError):
    """Raised when an input schema does not type its opaque relation rows."""


# ---------------------------------------------------------------------------
# Exact Q(sqrt 7) arithmetic, because the certified Fixed frame contains sqrt(7)
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
        return (
            f"{_fraction_text(self.rational)} + "
            f"{_fraction_text(self.radical)}*sqrt(7)"
        )


Q7_ZERO: Final = Q7()
Q7_ONE: Final = Q7(ONE)
Q7_SQRT7: Final = Q7(ZERO, ONE)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# The frozen source Type Schema, reused byte-pinned from 017.20
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
        """Exact PSD test on the 2x2 block plus a nonnegative scalar leg."""
        return (
            self.a11 >= 0 and self.a22 >= 0 and self.determinant >= 0 and self.s >= 0
        )

    @property
    def is_effect(self) -> bool:
        """Membership in the certified formal interval ``Eff(E) = [0, 1_E]``."""
        return self.is_positive and complement(self).is_positive

    @property
    def is_central(self) -> bool:
        """Centrality in E: the 2x2 block must act as a scalar."""
        return self.a12 == 0 and self.a11 == self.a22

    @property
    def coordinates(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.a11, self.a12, self.a22, self.s)

    @property
    def ambient_rank(self) -> int:
        """Rank in the concrete 16-dimensional realization."""
        block_rank = 0
        if not (self.a11 == 0 and self.a12 == 0 and self.a22 == 0):
            block_rank = 1 if self.determinant == 0 else 2
        scalar_rank = 0 if self.s == 0 else MULTIPLICITY_SCALAR
        return 2 * block_rank + scalar_rank


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
Z_MATRIX: Final = element("z_M", ONE, ZERO, ONE, ZERO)
Z_SCALAR: Final = element("z_s", ZERO, ZERO, ZERO, ONE)


def complement(value: Element) -> Element:
    return Element(
        "",
        ONE - value.a11,
        -value.a12,
        ONE - value.a22,
        ONE - value.s,
    )


def subtract(left: Element, right: Element) -> Element:
    return Element(
        "",
        left.a11 - right.a11,
        left.a12 - right.a12,
        left.a22 - right.a22,
        left.s - right.s,
    )


def _anonymous(value: Element) -> Element:
    return Element("", value.a11, value.a12, value.a22, value.s)


def projection(identifier: str, cosine: Fraction, sine: Fraction) -> Element:
    """The rank-one projection onto the line spanned by ``(cosine, sine)``."""
    if cosine * cosine + sine * sine != 1:
        raise ValueError("projection parameters must satisfy c^2 + s^2 = 1")
    return element(
        identifier,
        cosine * cosine,
        cosine * sine,
        sine * sine,
        ZERO,
    )


# ---------------------------------------------------------------------------
# States on E
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class State:
    """A functional ``omega(A, s) = Tr(rho A) + q s`` in adapted coordinates."""

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
    def coordinates(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.r11, self.r12, self.r22, self.q)

    @property
    def is_positive(self) -> bool:
        determinant = self.r11 * self.r22 - self.r12 * self.r12
        return self.r11 >= 0 and self.r22 >= 0 and determinant >= 0 and self.q >= 0

    @property
    def is_unital(self) -> bool:
        return self(UNIT) == ONE

    @property
    def is_state(self) -> bool:
        return self.is_positive and self.is_unital

    @property
    def central_parameter(self) -> Fraction:
        """The central observable ``lambda = omega(z_M)``."""
        return self(Z_MATRIX)


def _anonymous_state(state: State) -> State:
    return State("", state.r11, state.r12, state.r22, state.q)


def state_from_vector(
    identifier: str, vector: tuple[Fraction, Fraction, Fraction, Fraction]
) -> State:
    return State(identifier, *vector)


TAU_MATRIX: Final = State("tau_M", Fraction(1, 2), ZERO, Fraction(1, 2), ZERO)
DELTA_SCALAR: Final = State("delta_s", ZERO, ZERO, ZERO, ONE)


def omega_lambda(value: Fraction) -> State:
    """The candidate invariant state ``omega_lambda(A,s) = l Tr(A)/2 + (1-l) s``."""
    half = value / 2
    return State(f"omega_{_fraction_text(value)}", half, ZERO, half, ONE - value)


# Exact rational rotations: primitive Pythagorean triples give exact cos/sin
# without leaving Q, so the whole automorphism sweep stays exact.
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

Vector4 = tuple[Fraction, Fraction, Fraction, Fraction]


def rotate_vector(vector: Vector4, cosine: Fraction, sine: Fraction) -> Vector4:
    """Push a state coordinate vector forward by ``A -> Q^T A Q``."""
    if cosine * cosine + sine * sine != 1:
        raise ValueError("rotation parameters must satisfy c^2 + s^2 = 1")
    r11, r12, r22, q = vector
    return (
        cosine * cosine * r11 - 2 * cosine * sine * r12 + sine * sine * r22,
        cosine * sine * r11 + (cosine * cosine - sine * sine) * r12
        - cosine * sine * r22,
        sine * sine * r11 + 2 * cosine * sine * r12 + cosine * cosine * r22,
        q,
    )


def reflect_vector(vector: Vector4) -> Vector4:
    """Push forward by the orientation-reversing target automorphism."""
    r11, r12, r22, q = vector
    return (r11, -r12, r22, q)


TARGET_AUTOMORPHISM_PROBES: Final = tuple(
    [("rotation", cosine, sine) for cosine, sine in RATIONAL_ROTATIONS]
    + [("reflection", ONE, ZERO)]
)


def push_forward(vector: Vector4, probe: tuple[str, Fraction, Fraction]) -> Vector4:
    kind, cosine, sine = probe
    if kind == "rotation":
        return rotate_vector(vector, cosine, sine)
    if kind == "reflection":
        return reflect_vector(vector)
    raise KeyError(kind)


def push_state(state: State, probe: tuple[str, Fraction, Fraction]) -> State:
    return state_from_vector("", push_forward(state.coordinates, probe))


# ---------------------------------------------------------------------------
# Exact rational linear algebra, used for the invariant-state derivation
# ---------------------------------------------------------------------------


def row_reduce(rows: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """Exact reduced row echelon form plus the list of pivot columns."""
    matrix = [list(row) for row in rows]
    pivots: list[int] = []
    row_index = 0
    width = len(matrix[0]) if matrix else 0
    for column in range(width):
        selected = None
        for candidate in range(row_index, len(matrix)):
            if matrix[candidate][column] != 0:
                selected = candidate
                break
        if selected is None:
            continue
        matrix[row_index], matrix[selected] = matrix[selected], matrix[row_index]
        pivot = matrix[row_index][column]
        matrix[row_index] = [entry / pivot for entry in matrix[row_index]]
        for other in range(len(matrix)):
            if other == row_index:
                continue
            factor = matrix[other][column]
            if factor != 0:
                matrix[other] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(
                        matrix[other], matrix[row_index], strict=True
                    )
                ]
        pivots.append(column)
        row_index += 1
        if row_index == len(matrix):
            break
    return matrix, pivots


def null_space(rows: list[list[Fraction]]) -> tuple[int, list[Vector4]]:
    """Exact null-space basis of a rational matrix with four columns."""
    reduced, pivots = row_reduce(rows)
    width = len(rows[0])
    free = [column for column in range(width) if column not in pivots]
    basis: list[Vector4] = []
    for column in free:
        vector = [ZERO] * width
        vector[column] = ONE
        for pivot_row, pivot_column in enumerate(pivots):
            vector[pivot_column] = -reduced[pivot_row][column]
        basis.append(tuple(vector))  # type: ignore[arg-type]
    return len(pivots), basis


def matrix_vector(rows: list[list[Fraction]], vector: Vector4) -> list[Fraction]:
    return [
        sum((entry * value for entry, value in zip(row, vector, strict=True)), ZERO)
        for row in rows
    ]


def determinant_4x4(rows: list[list[Fraction]]) -> Fraction:
    """Exact determinant by cofactor expansion; the matrices here are tiny."""
    size = len(rows)
    if size == 1:
        return rows[0][0]
    total = ZERO
    for column in range(size):
        minor = [
            [entry for index, entry in enumerate(row) if index != column]
            for row in rows[1:]
        ]
        sign = ONE if column % 2 == 0 else -ONE
        total += sign * rows[0][column] * determinant_4x4(minor)
    return total


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


def u8_vector() -> tuple[Q7, ...]:
    return ambient_vector({8: Q7_ONE})


def v0_vector() -> tuple[Q7, ...]:
    coefficient = Q7_SQRT7.scale(Fraction(1, 7))
    return ambient_vector({index: coefficient for index in range(1, 8)})


def v8_vector() -> tuple[Q7, ...]:
    coefficient = Q7_SQRT7.scale(Fraction(-1, 7))
    return ambient_vector({index: coefficient for index in range(9, 16)})


def ambient_add(left: tuple[Q7, ...], right: tuple[Q7, ...]) -> tuple[Q7, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def ambient_scale(vector: tuple[Q7, ...], factor: Fraction) -> tuple[Q7, ...]:
    return tuple(entry.scale(factor) for entry in vector)


def ambient_inner(left: tuple[Q7, ...], right: tuple[Q7, ...]) -> Q7:
    total = Q7_ZERO
    for a, b in zip(left, right, strict=True):
        total = total + a * b
    return total


def generator_action(name: str, vector: tuple[Q7, ...]) -> tuple[Q7, ...]:
    """Apply one certified Fixed generator to an ambient vector, exactly.

    The generators act through the certified frame ``U + W+ + W-``: ``g0``
    projects onto ``U``, ``g3`` onto ``W+ + W-``, ``g4 = 6 P_{W+} - P_{W-}``, and
    ``g1 = sqrt(7) (|v0><u0| + |u0><v0| + |v8><u8| + |u8><v8|)``.
    """
    u0 = u0_vector()
    u8 = u8_vector()
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
    """Read off the exact Theory-27 state induced by an ambient preparation ray."""
    norm = ambient_inner(vector, vector)
    if norm.is_zero:
        raise ValueError("preparation rays must be nonzero")
    denominator = norm.as_fraction()

    values: dict[str, Fraction] = {}
    for name in ("g0", "g3", "g4"):
        numerator = ambient_inner(vector, generator_action(name, vector))
        values[name] = numerator.as_fraction() / denominator

    g1_numerator = ambient_inner(vector, generator_action("g1", vector))
    if g1_numerator.rational != 0:
        raise ValueError("the g1 pairing must be a pure multiple of sqrt(7)")
    values["g1"] = g1_numerator.radical / denominator

    r11 = values["g0"]
    r12 = values["g1"] / 2
    r22 = (values["g3"] + values["g4"]) / 7
    q = (6 * values["g3"] - values["g4"]) / 7
    return State("ambient", r11, r12, r22, q)


def ambient_proportional(left: tuple[Q7, ...], right: tuple[Q7, ...]) -> bool:
    """Exact projective equality test in ``P(V)``."""
    pivot = None
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if not a.is_zero or not b.is_zero:
            pivot = index
            break
    if pivot is None:
        raise ValueError("both vectors are zero")
    a, b = left[pivot], right[pivot]
    if a.is_zero or b.is_zero:
        return False
    # left * b == right * a coordinatewise iff the rays coincide.
    return all(x * b == y * a for x, y in zip(left, right, strict=True))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source(path: Path = SOURCE_PATH) -> tuple[SourceSchema, dict[str, object]]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"source fixture changed: expected {SOURCE_SHA256}, observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != SOURCE_SCHEMA:
        raise RuntimeError("unexpected source schema")
    sorts = document["sorts"]
    labels = document["decision_type"]["labels"]
    schema = SourceSchema(
        points=tuple(sorts["point"]),
        lines=tuple(sorts["line"]),
        incidence=frozenset(
            (row[0], row[1]) for row in document["relations"]["incident"]
        ),
        decision_labels=(labels[0], labels[1]),
    )
    return schema, document


def load_prior_premises(path: Path = PRIOR_PREMISES_PATH) -> dict[str, object]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != PRIOR_PREMISES_SHA256:
        raise RuntimeError(
            f"017.21a premises changed: expected {PRIOR_PREMISES_SHA256}, "
            f"observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != PRIOR_PREMISE_SCHEMA:
        raise RuntimeError("unexpected 017.21a premise schema")
    return document


def load_prior_audit(path: Path = PRIOR_AUDIT_PATH) -> dict[str, object]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != PRIOR_AUDIT_SHA256:
        raise RuntimeError(
            f"017.21a audit changed: expected {PRIOR_AUDIT_SHA256}, observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != PRIOR_AUDIT_SCHEMA:
        raise RuntimeError("unexpected 017.21a audit schema")
    return document


def enumerate_automorphisms(source: SourceSchema) -> tuple[Automorphism, ...]:
    """Recover the incidence automorphisms by exhausting point relabelings."""
    incidence = source.incidence
    if incidence is None:
        raise MissingDeclaredRelation("cannot enumerate without a declared relation")
    lines_by_points = {
        line: frozenset(point for point, other in incidence if other == line)
        for line in source.lines
    }
    found: list[Automorphism] = []
    for permutation in itertools.permutations(source.points):
        point_map = dict(zip(source.points, permutation, strict=True))
        line_images: dict[str, str] = {}
        consistent = True
        for line, members in lines_by_points.items():
            image = frozenset(point_map[point] for point in members)
            match = [
                other for other, others in lines_by_points.items() if others == image
            ]
            if len(match) != 1:
                consistent = False
                break
            line_images[line] = match[0]
        if not consistent:
            continue
        if len(set(line_images.values())) != len(source.lines):
            continue
        found.append(
            Automorphism(
                point_images=permutation,
                line_images=tuple(line_images[line] for line in source.lines),
            )
        )
    return tuple(found)


def input_orbits(
    source: SourceSchema, automorphisms: tuple[Automorphism, ...]
) -> tuple[tuple[Datum, ...], ...]:
    remaining = list(source.inputs)
    orbits: list[tuple[Datum, ...]] = []
    while remaining:
        seed = remaining[0]
        orbit = {
            automorphism.apply(seed, source) for automorphism in automorphisms
        }
        orbits.append(tuple(sorted(orbit)))
        remaining = [datum for datum in remaining if datum not in orbit]
    return tuple(orbits)


# ---------------------------------------------------------------------------
# Probe families: exact states and exact effects used throughout the audit
# ---------------------------------------------------------------------------

LAMBDA_GRID: Final = tuple(Fraction(index, 12) for index in range(13))

PYTHAGOREAN_DIRECTIONS: Final = (
    ("e1", ONE, ZERO),
    ("e2", ZERO, ONE),
    ("3-4-5", Fraction(3, 5), Fraction(4, 5)),
    ("4-3-5", Fraction(4, 5), Fraction(3, 5)),
    ("5-12-13", Fraction(5, 13), Fraction(12, 13)),
    ("12-5-13", Fraction(12, 13), Fraction(5, 13)),
    ("8-15-17", Fraction(8, 17), Fraction(15, 17)),
    ("7-24-25", Fraction(7, 25), Fraction(24, 25)),
    ("20-21-29", Fraction(20, 29), Fraction(21, 29)),
)


def probe_states() -> tuple[State, ...]:
    """Exact states spanning the interesting corners of ``S(E)``."""
    out: list[State] = [TAU_MATRIX, DELTA_SCALAR]
    for value in (ZERO, Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), ONE):
        out.append(omega_lambda(value))
    half = Fraction(1, 2)
    quarter = Fraction(1, 4)
    out.extend(
        [
            State("rho_e1", ONE, ZERO, ZERO, ZERO),
            State("rho_e2", ZERO, ZERO, ONE, ZERO),
            State("rho_e1_half_scalar", half, ZERO, ZERO, half),
            State("rho_diag_quarter", quarter, ZERO, quarter, half),
            State("rho_rank1_plus", quarter, quarter, quarter, half),
            State("rho_rank1_minus", quarter, -quarter, quarter, half),
            State("rho_skew_third", Fraction(1, 3), ZERO, Fraction(1, 6), half),
            State("omega_t_one_third", Fraction(1, 3), ZERO, Fraction(2, 3), ZERO),
            State("omega_t_one_twelfth", Fraction(1, 12), ZERO, Fraction(11, 12), ZERO),
        ]
    )
    for name, cosine, sine in PYTHAGOREAN_DIRECTIONS:
        block = projection(f"P_{name}", cosine, sine)
        out.append(State(f"rho_{name}", block.a11, block.a12, block.a22, ZERO))
    return tuple(out)


def probe_effects() -> tuple[Element, ...]:
    """Exact formal effects used as separating and rival test probes."""
    half = Fraction(1, 2)
    out: list[Element] = [
        ALGEBRA_ZERO,
        UNIT,
        Z_MATRIX,
        Z_SCALAR,
        element("E11", ONE, ZERO, ZERO, ZERO),
        element("E22", ZERO, ZERO, ONE, ZERO),
        element("G_plus", half, half, half, ZERO),
        element("G_minus", half, -half, half, ZERO),
        element("half_unit", half, ZERO, half, half),
        element("E11_plus_scalar", ONE, ZERO, ZERO, ONE),
        element("unsharp_quarter", Fraction(1, 4), ZERO, Fraction(3, 4), half),
    ]
    for name, cosine, sine in PYTHAGOREAN_DIRECTIONS:
        out.append(projection(f"P_{name}", cosine, sine))
    return tuple(out)


SEPARATING_BASIS: Final = (
    Z_SCALAR,
    Z_MATRIX,
    element("E11", ONE, ZERO, ZERO, ZERO),
    element("G_plus", Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), ZERO),
)


def rational_state_grid(denominator: int) -> tuple[State, ...]:
    """Every exact state whose coordinates lie on a fixed rational grid."""
    values = [Fraction(index, denominator) for index in range(denominator + 1)]
    signed = values + [-value for value in values if value != 0]
    out: list[State] = []
    for r11 in values:
        for r22 in values:
            for r12 in signed:
                q = ONE - r11 - r22
                candidate = State("grid", r11, r12, r22, q)
                if candidate.is_state:
                    out.append(candidate)
    return tuple(out)


# ---------------------------------------------------------------------------
# 1. The declared effect family E_Sigma
# ---------------------------------------------------------------------------


def audit_effect_family(source: SourceSchema) -> dict[str, object]:
    """Define and census ``E_Sigma``, the effects reachable through Sigma."""
    declared = (Z_MATRIX, Z_SCALAR)
    closure = (ALGEBRA_ZERO, Z_MATRIX, Z_SCALAR, UNIT)

    basis_rows = [list(effect.coordinates) for effect in SEPARATING_BASIS]
    basis_determinant = determinant_4x4(basis_rows)

    declared_rank, _ = null_space([list(effect.coordinates) for effect in declared])
    span_rows = [list(effect.coordinates) for effect in declared]
    reduced, pivots = row_reduce(span_rows)
    del reduced, declared_rank

    return {
        "definition": (
            "E_Sigma is the set of formal effects that some declared test of the "
            "Type Schema actually evaluates: E_Sigma = image(T) for the declared "
            "test realization T, closed under coarse-graining of its slots"
        ),
        "authority_for_the_notation": (
            "occurrence/gpt top-level 15 section 8 already writes E_Sigma subset "
            "Eff(E) for the admissible effect/test family induced by a Type Schema"
        ),
        "declared_operations": [
            {
                "operation": "IncidenceDecision",
                "answers": list(source.decision_labels),
                "answer_count": len(source.decision_labels),
                "admitted_inputs": len(source.inputs),
            }
        ],
        "E_Sigma": [effect.identifier for effect in declared],
        "E_Sigma_cardinality": len(declared),
        "E_Sigma_is_orientation_independent": True,
        "E_Sigma_orientation_note": (
            "the orientation bit permutes which answer label maps to which central "
            "effect; the set E_Sigma is unchanged, so the observational relation "
            "defined below does not depend on the orientation"
        ),
        "coarse_grained_closure": [effect.identifier for effect in closure],
        "closure_cardinality": len(closure),
        "all_members_are_effects": all(effect.is_effect for effect in declared),
        "all_members_are_central": all(effect.is_central for effect in declared),
        "members_sum_to_unit": _anonymous(
            element(
                "",
                Z_MATRIX.a11 + Z_SCALAR.a11,
                Z_MATRIX.a12 + Z_SCALAR.a12,
                Z_MATRIX.a22 + Z_SCALAR.a22,
                Z_MATRIX.s + Z_SCALAR.s,
            )
        )
        == _anonymous(UNIT),
        "span_dimension": len(pivots),
        "algebra_dimension": EFFECT_ALGEBRA_DIMENSION,
        "span_is_proper": len(pivots) < EFFECT_ALGEBRA_DIMENSION,
        "eff_E_spans_the_algebra": {
            "witness": [effect.identifier for effect in SEPARATING_BASIS],
            "all_witnesses_are_effects": all(
                effect.is_effect for effect in SEPARATING_BASIS
            ),
            "determinant": _fraction_text(basis_determinant),
            "is_a_basis": basis_determinant != 0,
            "consequence": (
                "Eff(E) contains a linear basis of E, so the full formal effect "
                "family separates states and S(E) admits no further quotient "
                "without losing a formal probability"
            ),
        },
    }


# ---------------------------------------------------------------------------
# 2. State-level observational equivalence
# ---------------------------------------------------------------------------


def sigma_equivalent(left: State, right: State, family: tuple[Element, ...]) -> bool:
    return all(left(effect) == right(effect) for effect in family)


def audit_state_equivalence() -> dict[str, object]:
    """Audit the relation ``omega ~_Sigma omega'`` and characterize its quotient."""
    declared = (Z_MATRIX, Z_SCALAR)
    closure = (ALGEBRA_ZERO, Z_MATRIX, Z_SCALAR, UNIT)
    states = probe_states()
    effects = probe_effects()

    all_are_states = all(state.is_state for state in states)

    reflexive = all(sigma_equivalent(state, state, declared) for state in states)
    symmetric = all(
        sigma_equivalent(left, right, declared)
        == sigma_equivalent(right, left, declared)
        for left in states
        for right in states
    )
    transitive = True
    for left in states:
        for middle in states:
            for right in states:
                if sigma_equivalent(left, middle, declared) and sigma_equivalent(
                    middle, right, declared
                ):
                    transitive &= sigma_equivalent(left, right, declared)

    lambda_characterizes = all(
        sigma_equivalent(left, right, declared)
        == (left.central_parameter == right.central_parameter)
        for left in states
        for right in states
    )
    closure_agrees = all(
        sigma_equivalent(left, right, declared)
        == sigma_equivalent(left, right, closure)
        for left in states
        for right in states
    )

    classes: dict[str, list[str]] = {}
    for state in states:
        key = _fraction_text(state.central_parameter)
        classes.setdefault(key, []).append(state.identifier)

    # A fat fiber: three distinct states with the same central observable.
    fiber_states = (
        State("rho_e1_half_scalar", Fraction(1, 2), ZERO, ZERO, Fraction(1, 2)),
        State("rho_diag_quarter", Fraction(1, 4), ZERO, Fraction(1, 4), Fraction(1, 2)),
        State(
            "rho_rank1_plus",
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 2),
        ),
    )
    fiber_pairwise_distinct = all(
        _anonymous_state(left) != _anonymous_state(right)
        for index, left in enumerate(fiber_states)
        for right in fiber_states[index + 1 :]
    )
    fiber_all_equivalent = all(
        sigma_equivalent(fiber_states[0], other, declared) for other in fiber_states
    )
    fiber_separating: list[dict[str, object]] = []
    for index, left in enumerate(fiber_states):
        for right in fiber_states[index + 1 :]:
            witness = next(
                (
                    effect.identifier
                    for effect in effects
                    if left(effect) != right(effect)
                ),
                None,
            )
            fiber_separating.append(
                {
                    "pair": [left.identifier, right.identifier],
                    "separating_effect_outside_E_Sigma": witness,
                    "separated": witness is not None,
                }
            )

    # The lambda = 0 fiber is a single point, because positivity collapses it.
    grid = rational_state_grid(6)
    lambda_zero = [state for state in grid if state.central_parameter == 0]
    lambda_zero_is_singleton = all(
        _anonymous_state(state) == _anonymous_state(DELTA_SCALAR)
        for state in lambda_zero
    )

    surjective = all(
        omega_lambda(value).is_state
        and omega_lambda(value).central_parameter == value
        for value in LAMBDA_GRID
    )

    # Eff(E)-separation, checked on every distinct probe pair.
    separated_pairs = 0
    total_pairs = 0
    for index, left in enumerate(states):
        for right in states[index + 1 :]:
            if _anonymous_state(left) == _anonymous_state(right):
                continue
            total_pairs += 1
            if any(left(effect) != right(effect) for effect in effects):
                separated_pairs += 1

    return {
        "definition": (
            "omega ~_Sigma omega' iff omega(e) = omega'(e) for every e in E_Sigma"
        ),
        "probe_state_count": len(states),
        "all_probe_states_are_states": all_are_states,
        "is_an_equivalence_relation": {
            "reflexive": reflexive,
            "symmetric": symmetric,
            "transitive": transitive,
            "why": (
                "the relation is the kernel of the evaluation map "
                "omega -> (omega(e))_{e in E_Sigma}, so it is automatically an "
                "equivalence relation; the checks above confirm it mechanically"
            ),
        },
        "quotient_invariant": {
            "name": "lambda",
            "formula": "lambda = omega(z_M), and omega(z_s) = 1 - lambda",
            "why_one_number_suffices": (
                "z_M + z_s = 1_E and every state is unital, so the second declared "
                "value is determined by the first"
            ),
            "lambda_characterizes_the_relation": lambda_characterizes,
            "coarse_grained_closure_adds_nothing": closure_agrees,
            "range_is_the_unit_interval": surjective,
            "quotient": "S(E) / ~_Sigma is in exact bijection with [0,1]",
        },
        "observed_classes": {
            key: sorted(members) for key, members in sorted(classes.items())
        },
        "observed_class_count": len(classes),
        "fat_fiber_witness": {
            "lambda": "1/2",
            "states": [state.identifier for state in fiber_states],
            "pairwise_distinct": fiber_pairwise_distinct,
            "all_Sigma_equivalent": fiber_all_equivalent,
            "separating_effects_outside_E_Sigma": fiber_separating,
            "consequence": (
                "for the declared test, ~_Sigma is strictly coarser than equality "
                "of states: a two-real-dimensional set of states is observationally "
                "identical at each lambda > 0"
            ),
        },
        "degenerate_fiber": {
            "lambda": "0",
            "grid_denominator": 6,
            "states_on_grid_with_lambda_zero": len(lambda_zero),
            "fiber_is_the_single_state_delta_s": lambda_zero_is_singleton,
            "proof": (
                "omega(z_M) = r11 + r22 = 0 with r11, r22 >= 0 forces r11 = r22 = 0, "
                "and positive semidefiniteness then forces r12^2 <= r11 r22 = 0, so "
                "rho = 0 and unitality gives q = 1"
            ),
        },
        "special_cases": {
            "E_Sigma_equals_Eff_E": {
                "relation_becomes_equality": separated_pairs == total_pairs,
                "distinct_probe_pairs": total_pairs,
                "separated_probe_pairs": separated_pairs,
                "proof": (
                    "Eff(E) contains a linear basis of E, so agreement on all of "
                    "Eff(E) forces equality of the functionals"
                ),
            },
            "one_fixed_finite_test": (
                "many distinct states are observationally identical; the quotient is "
                "exactly [0,1] for the 017.21a central test"
            ),
            "the_017_21a_central_test": {
                "tau_M_lambda": _fraction_text(TAU_MATRIX.central_parameter),
                "delta_s_lambda": _fraction_text(DELTA_SCALAR.central_parameter),
                "the_two_compiled_states_are_not_Sigma_equivalent": not
                sigma_equivalent(TAU_MATRIX, DELTA_SCALAR, declared),
            },
        },
    }


# ---------------------------------------------------------------------------
# 3. The invariant state family
# ---------------------------------------------------------------------------


def audit_invariant_family() -> dict[str, object]:
    """Derive, by exact linear algebra, all Aut(E,E_+,1_E)-invariant states."""
    rows: list[list[Fraction]] = []
    identity = [
        (ONE, ZERO, ZERO, ZERO),
        (ZERO, ONE, ZERO, ZERO),
        (ZERO, ZERO, ONE, ZERO),
        (ZERO, ZERO, ZERO, ONE),
    ]
    for probe in TARGET_AUTOMORPHISM_PROBES:
        columns = [push_forward(basis, probe) for basis in identity]
        for row_index in range(4):
            row = [columns[column][row_index] for column in range(4)]
            row[row_index] -= ONE
            rows.append(row)

    rank, basis = null_space(rows)
    trace_direction: Vector4 = (ONE, ZERO, ONE, ZERO)
    scalar_direction: Vector4 = (ZERO, ZERO, ZERO, ONE)
    trace_in_kernel = all(value == 0 for value in matrix_vector(rows, trace_direction))
    scalar_in_kernel = all(
        value == 0 for value in matrix_vector(rows, scalar_direction)
    )

    family_records: list[dict[str, object]] = []
    all_invariant = True
    all_states = True
    all_match_lambda = True
    for value in LAMBDA_GRID:
        candidate = omega_lambda(value)
        invariant = all(
            _anonymous_state(push_state(candidate, probe))
            == _anonymous_state(candidate)
            for probe in TARGET_AUTOMORPHISM_PROBES
        )
        all_invariant &= invariant
        all_states &= candidate.is_state
        all_match_lambda &= candidate.central_parameter == value
        family_records.append(
            {
                "lambda": _fraction_text(value),
                "is_state": candidate.is_state,
                "is_invariant": invariant,
                "value_on_z_M": _fraction_text(candidate(Z_MATRIX)),
                "value_on_z_s": _fraction_text(candidate(Z_SCALAR)),
            }
        )

    outside = []
    for value in (Fraction(-1, 3), Fraction(4, 3)):
        candidate = omega_lambda(value)
        outside.append(
            {
                "lambda": _fraction_text(value),
                "is_state": candidate.is_state,
                "violated": "positivity of rho" if value < 0 else "positivity of q",
            }
        )

    # Every non-invariant probe state must fail at least one probe automorphism.
    non_invariant_detected = 0
    non_invariant_total = 0
    for state in probe_states():
        is_family = state.r12 == 0 and state.r11 == state.r22
        if is_family:
            continue
        non_invariant_total += 1
        if any(
            _anonymous_state(push_state(state, probe)) != _anonymous_state(state)
            for probe in TARGET_AUTOMORPHISM_PROBES
        ):
            non_invariant_detected += 1

    # Each ~_Sigma class contains exactly one invariant state.
    cross_section = all(
        omega_lambda(value).central_parameter == value for value in LAMBDA_GRID
    ) and all(
        _anonymous_state(omega_lambda(left)) != _anonymous_state(omega_lambda(right))
        for left in LAMBDA_GRID
        for right in LAMBDA_GRID
        if left != right
    )

    return {
        "target_automorphism_group": "Aut(E, E_+, 1_E) = O(2)/{+-I}",
        "action": "(A, s) -> (Q^T A Q, s); the scalar ideal is fixed pointwise",
        "probe_count": len(TARGET_AUTOMORPHISM_PROBES),
        "constraint_matrix": {
            "rows": len(rows),
            "columns": 4,
            "rank": rank,
            "null_space_dimension": len(basis),
            "null_space_basis": [
                [_fraction_text(entry) for entry in vector] for vector in basis
            ],
            "trace_direction_in_kernel": trace_in_kernel,
            "scalar_direction_in_kernel": scalar_in_kernel,
            "kernel_is_spanned_by_those_two": rank == 2
            and len(basis) == 2
            and trace_in_kernel
            and scalar_in_kernel,
        },
        "derivation": (
            "invariance under the whole rotation subgroup kills the spin-2 part of "
            "rho, forcing r12 = 0 and r11 = r22; writing rho = (lambda/2) I_2 and "
            "imposing unitality gives q = 1 - lambda"
        ),
        "family": "omega_lambda(A, s) = lambda * Tr(A)/2 + (1 - lambda) * s",
        "family_records": family_records,
        "all_family_members_are_states": all_states,
        "all_family_members_are_invariant": all_invariant,
        "lambda_is_the_value_on_z_M": all_match_lambda,
        "outside_the_interval": outside,
        "positivity_bounds_lambda_to_unit_interval": all(
            row["is_state"] is False for row in outside
        ),
        "non_invariant_probe_states": non_invariant_total,
        "non_invariant_probe_states_detected": non_invariant_detected,
        "endpoints": {
            "lambda_1": "tau_M(A, s) = Tr(A)/2",
            "lambda_0": "delta_s(A, s) = s",
        },
        "is_a_cross_section_of_the_observational_quotient": cross_section,
        "cross_section_consequence": (
            "lambda -> omega_lambda is an exact bijection from S(E)/~_Sigma onto the "
            "invariant states, so premise P4 selects one canonical representative "
            "per observational class rather than constraining behavior"
        ),
        "verdict": (
            "the Aut(E,E_+,1_E)-invariant normalized states are exactly the "
            "one-parameter family omega_lambda with lambda in [0,1]"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Whole-realization semantics and observational equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Realization:
    """One complete labeled realization ``R = (C, T)`` of the declared contract.

    ``C`` assigns a state to each declared source relation class and ``T``
    interprets each declared answer label as a formal effect. Both halves are
    internal; only the induced labeled distribution is public.
    """

    identifier: str
    compiler: tuple[tuple[str, State], ...]
    test: tuple[tuple[str, Element], ...]
    description: str = ""

    def state_for(self, semantic_class: str) -> State:
        for key, value in self.compiler:
            if key == semantic_class:
                return value
        raise KeyError(semantic_class)

    def effect_for(self, label: str) -> Element:
        for key, value in self.test:
            if key == label:
                return value
        raise KeyError(label)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.test)

    @property
    def test_is_normalized(self) -> bool:
        total = ALGEBRA_ZERO
        for _, effect in self.test:
            total = Element(
                "",
                total.a11 + effect.a11,
                total.a12 + effect.a12,
                total.a22 + effect.a22,
                total.s + effect.s,
            )
        return _anonymous(total) == _anonymous(UNIT)

    @property
    def test_slots_are_effects(self) -> bool:
        return all(effect.is_effect for _, effect in self.test)

    @property
    def states_are_states(self) -> bool:
        return all(state.is_state for _, state in self.compiler)

    @property
    def uses_only_central_effects(self) -> bool:
        return all(effect.is_central for _, effect in self.test)

    @property
    def uses_only_invariant_states(self) -> bool:
        return all(
            state.r12 == 0 and state.r11 == state.r22 for _, state in self.compiler
        )


def public_family(
    realization: Realization, source: SourceSchema
) -> dict[Datum, dict[str, Fraction]]:
    """The induced public object ``F_R : Input_Sigma -> Delta(A_d)``."""
    out: dict[Datum, dict[str, Fraction]] = {}
    for datum in source.inputs:
        state = realization.state_for(source.semantic_class(datum))
        out[datum] = {
            label: state(realization.effect_for(label))
            for label in realization.labels
        }
    return out


def is_distribution(row: dict[str, Fraction]) -> bool:
    return all(value >= 0 for value in row.values()) and sum(
        row.values(), ZERO
    ) == ONE


ORIENTATION_ONE: Final = Realization(
    identifier="orientation-incident-to-z_M",
    compiler=(("incident", TAU_MATRIX), ("nonincident", DELTA_SCALAR)),
    test=(("incident", Z_MATRIX), ("nonincident", Z_SCALAR)),
    description="017.21a binding-incident-to-z_M",
)
ORIENTATION_TWO: Final = Realization(
    identifier="orientation-incident-to-z_s",
    compiler=(("incident", DELTA_SCALAR), ("nonincident", TAU_MATRIX)),
    test=(("incident", Z_SCALAR), ("nonincident", Z_MATRIX)),
    description="017.21a binding-incident-to-z_s",
)
ORIENTATIONS: Final = (ORIENTATION_ONE, ORIENTATION_TWO)


def sharp_rival_realizations() -> tuple[Realization, ...]:
    """Exact non-invariant realizations that reproduce the same public family.

    For any rank-one projection ``P`` the pair ``(P, 1_E - P)`` is a formal test
    and the states ``omega_P`` and either ``delta_s`` or ``omega_{P_perp}``
    reproduce the declared deterministic behavior exactly.
    """
    out: list[Realization] = []
    for name, cosine, sine in PYTHAGOREAN_DIRECTIONS:
        block = projection(f"P_{name}", cosine, sine)
        perp = projection(f"P_{name}_perp", -sine, cosine)
        slot_incident = block
        slot_nonincident = complement(block)
        state_block = State(f"omega_P_{name}", block.a11, block.a12, block.a22, ZERO)
        state_perp = State(
            f"omega_Pperp_{name}", perp.a11, perp.a12, perp.a22, ZERO
        )
        out.append(
            Realization(
                identifier=f"rival-sharp-scalar-{name}",
                compiler=(("incident", state_block), ("nonincident", DELTA_SCALAR)),
                test=(
                    ("incident", slot_incident),
                    ("nonincident", slot_nonincident),
                ),
                description=(
                    "rank-one projection test with the matching pure state and the "
                    "scalar character"
                ),
            )
        )
        out.append(
            Realization(
                identifier=f"rival-sharp-pair-{name}",
                compiler=(("incident", state_block), ("nonincident", state_perp)),
                test=(
                    ("incident", slot_incident),
                    ("nonincident", slot_nonincident),
                ),
                description=(
                    "rank-one projection test with both orthogonal pure states; no "
                    "central effect and no invariant state appears"
                ),
            )
        )
    return tuple(out)


def _family_key(
    family: dict[Datum, dict[str, Fraction]], labels: tuple[str, ...]
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple(
        (
            datum[0],
            datum[1],
            tuple(_fraction_text(family[datum][label]) for label in labels),
        )
        for datum in sorted(family)
    )


def audit_realization_equivalence(source: SourceSchema) -> dict[str, object]:
    """Audit ``R ~^TDM_Sigma R'`` on the finite fixture."""
    realizations = list(ORIENTATIONS) + list(sharp_rival_realizations())
    families = {
        realization.identifier: public_family(realization, source)
        for realization in realizations
    }
    labels = source.decision_labels

    records: list[dict[str, object]] = []
    all_valid = True
    all_distributions = True
    reproduce_target = True
    for realization in realizations:
        family = families[realization.identifier]
        valid = (
            realization.test_is_normalized
            and realization.test_slots_are_effects
            and realization.states_are_states
        )
        all_valid &= valid
        distributions = all(is_distribution(row) for row in family.values())
        all_distributions &= distributions
        matches = all(
            family[datum]
            == (
                {labels[0]: ONE, labels[1]: ZERO}
                if source.semantic_class(datum) == labels[0]
                else {labels[0]: ZERO, labels[1]: ONE}
            )
            for datum in source.inputs
        )
        reproduce_target &= matches
        records.append(
            {
                "realization": realization.identifier,
                "description": realization.description,
                "incident_state": realization.state_for("incident").identifier,
                "nonincident_state": realization.state_for("nonincident").identifier,
                "incident_slot": realization.effect_for("incident").identifier,
                "nonincident_slot": realization.effect_for("nonincident").identifier,
                "test_is_normalized": realization.test_is_normalized,
                "test_slots_are_effects": realization.test_slots_are_effects,
                "states_are_states": realization.states_are_states,
                "uses_only_central_effects": realization.uses_only_central_effects,
                "uses_only_invariant_states": realization.uses_only_invariant_states,
                "every_output_is_a_distribution": distributions,
                "reproduces_the_declared_behavior": matches,
            }
        )

    keys = {
        identifier: _family_key(family, labels)
        for identifier, family in families.items()
    }
    distinct_families = len(set(keys.values()))
    orientations_agree = keys[ORIENTATION_ONE.identifier] == keys[
        ORIENTATION_TWO.identifier
    ]

    identifiers = [realization.identifier for realization in realizations]
    reflexive = all(keys[name] == keys[name] for name in identifiers)
    symmetric = all(
        (keys[left] == keys[right]) == (keys[right] == keys[left])
        for left in identifiers
        for right in identifiers
    )
    transitive = all(
        not (keys[left] == keys[middle] and keys[middle] == keys[right])
        or keys[left] == keys[right]
        for left in identifiers
        for middle in identifiers
        for right in identifiers
    )

    internal_pairs = {
        (
            realization.state_for("incident").coordinates,
            realization.state_for("nonincident").coordinates,
            realization.effect_for("incident").coordinates,
        )
        for realization in realizations
    }

    return {
        "definition": (
            "R ~^TDM_Sigma R' iff F_R = F_R' as functions on every admitted input "
            "and every declared operation"
        ),
        "why_it_is_an_equivalence_relation": (
            "it is the kernel of R -> F_R, so reflexivity, symmetry and "
            "transitivity are automatic; verified mechanically below"
        ),
        "checked_realizations": len(realizations),
        "records": records,
        "all_realizations_are_well_formed": all_valid,
        "all_outputs_are_distributions": all_distributions,
        "all_reproduce_the_declared_behavior": reproduce_target,
        "distinct_public_families": distinct_families,
        "single_equivalence_class": distinct_families == 1,
        "distinct_internal_realizations": len(internal_pairs),
        "orientations_are_TDM_equivalent": orientations_agree,
        "relation_checks": {
            "reflexive": reflexive,
            "symmetric": symmetric,
            "transitive": transitive,
        },
        "coordinated_change_witness": {
            "claim": (
                "the relation allows coordinated internal changes of state and test "
                "representation whenever the labeled distribution is unchanged"
            ),
            "orientation_pair": (
                "orientation-incident-to-z_M and orientation-incident-to-z_s change "
                "both the compiled state and the answer-to-effect labeling, and the "
                "public family is byte-identical"
            ),
            "sharp_rival_pair": (
                "the rank-one projection rivals change the test away from the center "
                "and the states away from the invariant family, and the public "
                "family is still byte-identical"
            ),
            "class_is_at_least_a_continuum": True,
            "continuum_reason": (
                "the rank-one projection rivals are indexed by a direction in the "
                "real projective line, so the equivalence class of the declared "
                "behavior contains at least an RP^1 family of realizations"
            ),
        },
    }


# ---------------------------------------------------------------------------
# 5. What P4 actually forces once the public behavior is fixed
# ---------------------------------------------------------------------------


def effect_grid() -> tuple[Element, ...]:
    """Every formal effect whose adapted coordinates lie on a quarter grid."""
    values = [Fraction(index, 4) for index in range(5)]
    signed = sorted(set(values + [-value for value in values]))
    out: list[Element] = []
    for a11 in values:
        for a22 in values:
            for a12 in signed:
                for s in values:
                    candidate = element("", a11, a12, a22, s)
                    if candidate.is_effect:
                        out.append(candidate)
    return tuple(out)


def audit_p4_forcing() -> dict[str, object]:
    """Enumerate every invariant-state realization of the declared behavior.

    The declared behavior requires one effect ``e`` and two invariant states with
    ``omega_{lambda_I}(e) = 1`` and ``omega_{lambda_N}(e) = 0``. Because an
    invariant state only sees ``(Tr A / 2, s)``, this pins ``e`` to a central
    idempotent and the two parameters to the interval endpoints.
    """
    grid = effect_grid()
    solutions: list[dict[str, object]] = []
    for effect in grid:
        for lambda_incident in LAMBDA_GRID:
            if omega_lambda(lambda_incident)(effect) != ONE:
                continue
            for lambda_nonincident in LAMBDA_GRID:
                if omega_lambda(lambda_nonincident)(effect) != ZERO:
                    continue
                solutions.append(
                    {
                        "incident_slot_coordinates": [
                            _fraction_text(value) for value in effect.coordinates
                        ],
                        "incident_slot_is_z_M": _anonymous(effect)
                        == _anonymous(Z_MATRIX),
                        "incident_slot_is_z_s": _anonymous(effect)
                        == _anonymous(Z_SCALAR),
                        "incident_slot_is_central": effect.is_central,
                        "lambda_incident": _fraction_text(lambda_incident),
                        "lambda_nonincident": _fraction_text(lambda_nonincident),
                    }
                )
    matched = {
        (
            str(row["lambda_incident"]),
            str(row["lambda_nonincident"]),
            bool(row["incident_slot_is_z_M"]),
            bool(row["incident_slot_is_z_s"]),
        )
        for row in solutions
    }
    expected = {("1", "0", True, False), ("0", "1", False, True)}
    non_central = sum(1 for effect in grid if not effect.is_central)
    off_diagonal = sum(1 for effect in grid if effect.a12 != 0)
    return {
        "search_space": {
            "effect_grid_size": len(grid),
            "non_central_effects_in_grid": non_central,
            "effects_with_nonzero_off_diagonal": off_diagonal,
            "lambda_grid_size": len(LAMBDA_GRID),
            "lambda_pairs_scanned_per_effect": len(LAMBDA_GRID) ** 2,
            "grid_denominators": {"effects": 4, "lambda": 12},
            "search_is_not_vacuous": non_central > 0 and off_diagonal > 0,
        },
        "solution_count": len(solutions),
        "solutions": solutions,
        "exactly_two_solutions": len(solutions) == 2,
        "solutions_are_the_two_orientations": matched == expected,
        "centrality_is_derived_not_assumed": all(
            bool(row["incident_slot_is_central"]) for row in solutions
        ),
        "closed_form_proof": (
            "omega_lambda(e) = lambda * t + (1 - lambda) * s with t = Tr(A_e)/2 in "
            "[0,1] and s = s_e in [0,1]. A convex combination equals 1 only if the "
            "active coordinate equals 1, and equals 0 only if the active coordinate "
            "equals 0. Requiring both for the same e forces lambda_I and lambda_N "
            "into {0,1} with opposite choices; t = 1 with 0 <= A <= I_2 forces "
            "A = I_2 and then positivity of the complement forces s = 0, giving "
            "e = z_M; symmetrically the other branch gives e = z_s"
        ),
        "consequence": (
            "premise P4 plus the declared deterministic behavior forces the test to "
            "be exactly the intrinsic central test and the compiled states to be the "
            "two central characters; 017.21a assumed centrality of the test, and it "
            "is in fact derivable"
        ),
    }


# ---------------------------------------------------------------------------
# 6. The machine-readable orientation behavior table
# ---------------------------------------------------------------------------


def build_table(
    source: SourceSchema, orbits: tuple[tuple[Datum, ...], ...]
) -> dict[str, object]:
    orbit_index = {
        datum: index for index, orbit in enumerate(orbits) for datum in orbit
    }
    labels = source.decision_labels
    rows: list[dict[str, object]] = []
    census: dict[str, dict[str, int]] = {}
    for realization in ORIENTATIONS:
        family = public_family(realization, source)
        counts: dict[str, int] = {}
        for datum in source.inputs:
            semantic_class = source.semantic_class(datum)
            state = realization.state_for(semantic_class)
            values = family[datum]
            distribution = tuple(_fraction_text(values[label]) for label in labels)
            counts["|".join(distribution)] = (
                counts.get("|".join(distribution), 0) + 1
            )
            rows.append(
                {
                    "orientation": realization.identifier,
                    "point": datum[0],
                    "line": datum[1],
                    "semantic_class": semantic_class,
                    "source_orbit": orbit_index[datum],
                    "compiled_state": state.identifier,
                    "state_coordinates": {
                        "r11": _fraction_text(state.r11),
                        "r12": _fraction_text(state.r12),
                        "r22": _fraction_text(state.r22),
                        "q": _fraction_text(state.q),
                    },
                    "answer_to_effect": {
                        label: realization.effect_for(label).identifier
                        for label in labels
                    },
                    "public_distribution": {
                        label: _fraction_text(values[label]) for label in labels
                    },
                    "total": _fraction_text(sum(values.values(), ZERO)),
                }
            )
        census[realization.identifier] = dict(sorted(counts.items()))

    first = [row for row in rows if row["orientation"] == ORIENTATIONS[0].identifier]
    second = [row for row in rows if row["orientation"] == ORIENTATIONS[1].identifier]
    public_columns_agree = all(
        left["public_distribution"] == right["public_distribution"]
        and left["point"] == right["point"]
        and left["line"] == right["line"]
        for left, right in zip(first, second, strict=True)
    )
    internal_columns_differ = all(
        left["compiled_state"] != right["compiled_state"]
        for left, right in zip(first, second, strict=True)
    )

    return {
        "schema": TABLE_SCHEMA,
        "task": "occurrence/gpt Issue 017.22 required artifact 3",
        "source_sha256": SOURCE_SHA256,
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "operation": "IncidenceDecision",
        "answer_labels": list(labels),
        "orientations": [realization.identifier for realization in ORIENTATIONS],
        "admitted_inputs": len(source.inputs),
        "row_count": len(rows),
        "distribution_census": census,
        "public_columns_agree_across_orientations": public_columns_agree,
        "internal_state_columns_differ_across_orientations": internal_columns_differ,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 7. Reclassifying the 017.21a orientation bit
# ---------------------------------------------------------------------------


def push_element(
    value: Element, probe: tuple[str, Fraction, Fraction]
) -> Element:
    """Target-automorphism action on an element; the scalar leg is fixed."""
    moved = push_forward(value.coordinates, probe)
    return Element("", moved[0], moved[1], moved[2], moved[3])


def audit_orientation_classification(source: SourceSchema) -> dict[str, object]:
    """Classify the surviving 017.21a specification bit exactly."""
    declared = (Z_MATRIX, Z_SCALAR)
    first = public_family(ORIENTATION_ONE, source)
    second = public_family(ORIENTATION_TWO, source)

    central_effects_are_fixed = True
    central_effects_swapped = False
    states_swapped = False
    for probe in TARGET_AUTOMORPHISM_PROBES:
        moved_matrix = push_element(Z_MATRIX, probe)
        moved_scalar = push_element(Z_SCALAR, probe)
        central_effects_are_fixed &= _anonymous(moved_matrix) == _anonymous(
            Z_MATRIX
        ) and _anonymous(moved_scalar) == _anonymous(Z_SCALAR)
        central_effects_swapped |= _anonymous(moved_matrix) == _anonymous(Z_SCALAR)
        states_swapped |= _anonymous_state(
            push_state(TAU_MATRIX, probe)
        ) == _anonymous_state(DELTA_SCALAR)

    return {
        "the_bit": (
            "which declared answer label binds to which minimal central idempotent"
        ),
        "cardinality": len(ORIENTATIONS),
        "specification_bits": 1,
        "questions": {
            "distinct_in_S_E": _anonymous_state(TAU_MATRIX)
            != _anonymous_state(DELTA_SCALAR),
            "distinct_as_labeled_internal_OT_realizations": (
                ORIENTATION_ONE.compiler != ORIENTATION_TWO.compiler
                and ORIENTATION_ONE.test != ORIENTATION_TWO.test
            ),
            "distinct_under_TDM_observational_equivalence": first != second,
            "distinguishable_by_any_Sigma_admitted_observation": any(
                first[datum] != second[datum] for datum in source.inputs
            ),
            "states_are_Sigma_equivalent": sigma_equivalent(
                TAU_MATRIX, DELTA_SCALAR, declared
            ),
        },
        "gauge_test": {
            "target_automorphism_probes": len(TARGET_AUTOMORPHISM_PROBES),
            "central_effects_are_pointwise_fixed": central_effects_are_fixed,
            "some_automorphism_swaps_the_central_effects": central_effects_swapped,
            "some_automorphism_swaps_the_two_states": states_swapped,
            "ideal_dimensions": {"z_M": 3, "z_s": 1},
            "structural_reason": (
                "an automorphism permutes the minimal ideals, and the two ideals have "
                "different dimensions 3 and 1, so no automorphism can exchange them"
            ),
            "is_certified_representation_gauge": False,
        },
        "classification": "unobservable internal realization ambiguity under Sigma",
        "classification_definition": (
            "two complete labeled realizations R and R' are a Sigma-unobservable "
            "labeled-realization ambiguity when F_R = F_R' while the internal pair "
            "(compiled state, answer-to-effect labeling) differs and no certified "
            "symmetry of the target datum carries one to the other"
        ),
        "explicitly_not": {
            "observable_constitutive_residue": (
                "rejected: the two realizations agree on every admitted input"
            ),
            "certified_representation_gauge": (
                "rejected: no element of Aut(E,E_+,1_E) relates them, so calling it "
                "gauge would import a symmetry that does not exist"
            ),
            "physical_gauge": (
                "rejected: no physical Test Realization is certified at all"
            ),
        },
        "sharper_statement": (
            "the orientation bit is not a distinguished two-element residue at the "
            "observational layer; it is the two-point slice that premise P4 and "
            "centrality cut out of an equivalence class containing at least an RP^1 "
            "family of Sigma-equivalent realizations"
        ),
        "training_status": "non-training residue under the present Type Schema",
    }


# ---------------------------------------------------------------------------
# 8. The ambient-ray boundary
# ---------------------------------------------------------------------------


def frame_combination(
    alpha: Fraction, beta: Fraction, gamma: Fraction, delta: Fraction
) -> tuple[Q7, ...]:
    vector = ambient_scale(u0_vector(), alpha)
    vector = ambient_add(vector, ambient_scale(u8_vector(), beta))
    vector = ambient_add(vector, ambient_scale(v0_vector(), gamma))
    return ambient_add(vector, ambient_scale(v8_vector(), delta))


def audit_ambient_boundary() -> dict[str, object]:
    """Compare P(V) with S(E) at the formal, evaluative, and identity layers."""
    coefficients = (Fraction(-1), ZERO, ONE)
    tau_realizers: list[tuple[str, tuple[Q7, ...]]] = []
    for alpha, beta, gamma, delta in itertools.product(coefficients, repeat=4):
        if alpha == 0 and beta == 0 and gamma == 0 and delta == 0:
            continue
        vector = frame_combination(alpha, beta, gamma, delta)
        if ambient_inner(vector, vector).is_zero:
            continue
        try:
            state = ambient_state_of(vector)
        except ValueError:
            continue
        if _anonymous_state(state) == _anonymous_state(TAU_MATRIX):
            name = (
                f"{_fraction_text(alpha)}*u0 + {_fraction_text(beta)}*u8 + "
                f"{_fraction_text(gamma)}*v0 + {_fraction_text(delta)}*v8"
            )
            tau_realizers.append((name, vector))

    distinct_tau: list[tuple[str, tuple[Q7, ...]]] = []
    for name, vector in tau_realizers:
        if not any(
            ambient_proportional(vector, other) for _, other in distinct_tau
        ):
            distinct_tau.append((name, vector))

    scalar_rays = {
        "e1-e2": ambient_vector({1: Q7_ONE, 2: Q7(-ONE)}),
        "e1-e3": ambient_vector({1: Q7_ONE, 3: Q7(-ONE)}),
        "e2-e4": ambient_vector({2: Q7_ONE, 4: Q7(-ONE)}),
        "e9-e10": ambient_vector({9: Q7_ONE, 10: Q7(-ONE)}),
        "e1+e2-2e3": ambient_vector(
            {1: Q7_ONE, 2: Q7_ONE, 3: Q7(Fraction(-2))}
        ),
    }
    delta_records: list[dict[str, object]] = []
    all_delta = True
    for name, vector in scalar_rays.items():
        state = ambient_state_of(vector)
        matches = _anonymous_state(state) == _anonymous_state(DELTA_SCALAR)
        all_delta &= matches
        delta_records.append({"ray": name, "induces_delta_s": matches})
    names = list(scalar_rays)
    delta_pairwise_distinct = all(
        not ambient_proportional(scalar_rays[left], scalar_rays[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )

    near_miss = frame_combination(ONE, ZERO, ONE, ZERO)
    near_miss_state = ambient_state_of(near_miss)

    effects = probe_effects()
    agree_on_all_effects = all(
        ambient_state_of(scalar_rays[name])(effect) == DELTA_SCALAR(effect)
        for name in names
        for effect in effects
    )

    return {
        "map": "q : P(V) -> S(E), [x] -> omega_hat_[x], certified by Theory 27",
        "level_1_formal_preparation": {
            "question": (
                "are distinct rays in one fiber genuinely distinct supplied "
                "preparations under current Theory?"
            ),
            "answer": "yes",
            "why": (
                "Theory 27 takes the ray [x] in P(V) as the supplied preparation "
                "input, so distinct rays are distinct inputs to the theorem even "
                "when they induce the same state"
            ),
            "tau_M_realizer_count_on_the_sign_grid": len(tau_realizers),
            "projectively_distinct_tau_M_realizers": len(distinct_tau),
            "tau_M_realizer_names": [name for name, _ in distinct_tau],
            "delta_s_realizers": delta_records,
            "all_sampled_scalar_rays_induce_delta_s": all_delta,
            "delta_s_realizers_are_projectively_distinct": delta_pairwise_distinct,
            "same_copy_near_miss": {
                "ray": "u0 + v0",
                "induces_tau_M": _anonymous_state(near_miss_state)
                == _anonymous_state(TAU_MATRIX),
                "note": (
                    "the cross-copy convention is load-bearing; the same-copy vector "
                    "is not the invariant state"
                ),
            },
        },
        "level_2_formal_evaluation": {
            "question": (
                "can any declared TDM test distinguish two rays with the same image "
                "in S(E)?"
            ),
            "answer": "no, and the statement is a theorem, not an observation",
            "theorem": (
                "if q([x]) = q([y]) then for every formal test T = {e_i} with e_i in "
                "Eff(E) and sum e_i = 1_E, P(i | [x], T) = omega_hat_[x](e_i) = "
                "omega_hat_[y](e_i) = P(i | [y], T); the declared IncidenceDecision "
                "test is one such T, so no admitted observation separates the fiber"
            ),
            "authority": (
                "Theory 27 section 2 factorization P(V) -> S(E) followed by "
                "S(E) x Eff(E) -> [0,1]"
            ),
            "mechanically_checked_on_probe_effects": agree_on_all_effects,
            "probe_effect_count": len(effects),
        },
        "level_3_instance_identity": {
            "question": (
                "does 017.10 or top-level 15 require a TDM implementation to preserve "
                "which ambient ray realizes omega?"
            ),
            "answer": "no such requirement exists, and no prohibition exists either",
            "017_10_says": (
                "a TDM instance is Type Schema + Training; the Type Schema is the "
                "public type boundary and internals may differ radically between "
                "implementations"
            ),
            "top_level_15_says": (
                "the Type Schema is part of what object was trained, and Training "
                "should constitute a state representation over the declared effect "
                "family E_Sigma"
            ),
            "conclusion": (
                "ambient-ray identity is below the declared public boundary and is "
                "simply undefined by the current TDM vocabulary; this is a "
                "terminology gap, not a derived identity rule"
            ),
        },
        "forgotten_information": {
            "ambient_dimension": AMBIENT_DIMENSION,
            "projective_dimension": AMBIENT_DIMENSION - 1,
            "state_moments_retained": EFFECT_ALGEBRA_DIMENSION,
            "state_free_coordinates_after_normalization": EFFECT_ALGEBRA_DIMENSION - 1,
            "sym_ambient_dimension": SYM_AMBIENT_DIMENSION,
            "directions_of_Sym_V_invisible_to_E": SYM_AMBIENT_DIMENSION
            - EFFECT_ALGEBRA_DIMENSION,
            "generic_fiber_real_dimension_at_least": (AMBIENT_DIMENSION - 1)
            - (EFFECT_ALGEBRA_DIMENSION - 1),
            "delta_s_fiber": "P(W-) = RP^11, real dimension 11",
            "tau_M_fiber": "one PO(2) family in U + W+",
            "status_of_the_dimension_count": (
                "arithmetic consequence of the rank of the moment map; the discrete "
                "witnesses above are exact, the dimension statement is a count"
            ),
            "observable_through_the_declared_one_shot_interface": False,
        },
        "verdict": (
            "ambient-ray nonuniqueness is a lower-level realization ambiguity that "
            "is irrelevant to one-shot evaluation, and simultaneously an unresolved "
            "instance-identity question; it is not a blocker for TDM constitution"
        ),
    }


# ---------------------------------------------------------------------------
# 9. Is S(E) the right operational carrier?
# ---------------------------------------------------------------------------


def audit_state_boundary(effect_family: dict[str, object]) -> dict[str, object]:
    separation = effect_family["eff_E_spans_the_algebra"]
    assert isinstance(separation, dict)
    return {
        "proposed_boundary": "operational TDM state = omega in S(E)",
        "one_shot_evaluation_factors_through_S_E": True,
        "one_shot_authority": (
            "Theory 27 section 2 states the factorization explicitly, so this is "
            "certified structure and not a new premise"
        ),
        "all_formal_effect_probabilities_preserved": True,
        "S_E_is_minimal_for_the_full_formal_family": bool(
            separation["is_a_basis"]
        ),
        "minimality_argument": (
            "Eff(E) contains a linear basis of E, so omega -> (omega(e))_{e in "
            "Eff(E)} is injective; any strictly coarser carrier loses a formal "
            "probability, hence S(E) is the smallest carrier through which every "
            "formal one-shot evaluation factors"
        ),
        "sufficient_for_the_declared_Sigma": (
            "S(E)/~_Sigma = [0,1] already suffices for the audited Type Schema, so "
            "S(E) is finer than the present declaration needs"
        ),
        "three_layer_answer": {
            "declared_Sigma_sufficient_carrier": "[0,1] via lambda = omega(z_M)",
            "canonical_formal_one_shot_carrier": "S(E)",
            "strictly_redundant_for_one_shot_evaluation": "P(V)",
        },
        "what_is_forgotten_from_the_ray": (
            "every component of the rank-one form associated with [x] that is "
            "orthogonal to the four-dimensional image of iota"
        ),
        "forgotten_information_is_observable_under_Sigma": False,
        "beyond_one_shot_evaluation": {
            "recurrent_dynamics": "not settled here; may require more than S(E)",
            "Interact": "explicitly out of scope and not used to repair anything",
            "physical_preparation_realization": (
                "Theory 27 sections 5 and 7 leave the physical map to [x] open, so "
                "the ray layer cannot be discarded as meaningless"
            ),
            "enlarged_effect_families": (
                "Theory 27 section 7 warns that an enriched retained datum may "
                "refine the admissible equivalence class, which would require "
                "rechecking this boundary"
            ),
            "evaluate_is_not_enact": True,
        },
        "verdict": (
            "S(E) is the correct and minimal operational state carrier for one-shot "
            "formal TDM evaluation; it is not claimed sufficient for autonomous "
            "dynamics, enactment, or physical preparation"
        ),
    }


# ---------------------------------------------------------------------------
# 10. The corrected 017.21a premise ledger
# ---------------------------------------------------------------------------


def audit_premise_ledger(premises: dict[str, object]) -> dict[str, object]:
    """Re-grade P2, P3 and P4 against what current Theory actually certifies."""
    added = premises["added_premises"]
    assert isinstance(added, list)
    by_id = {entry["id"]: entry for entry in added}  # type: ignore[index]

    prior_status = {
        key: str(entry["status"]) for key, entry in by_id.items()  # type: ignore[index]
    }
    claimed_new = {
        key: "declared" in value and "017.21a" in value
        for key, value in prior_status.items()
    }

    corrections = [
        {
            "id": "P1",
            "name": "reduced ordered Jordan effect algebra",
            "017_21a_status": prior_status["P1"],
            "corrected_status": "certified OT structure",
            "unchanged": True,
            "authority": (
                f"occurrence/fixed@{FIXED_REVISION}"
                "&path=06-reduced-ordered-space-and-liftable-dynamics.md"
            ),
        },
        {
            "id": "P2",
            "name": "full formal effect interval",
            "017_21a_status": prior_status["P2"],
            "corrected_status": "certified OT structure, not a new declaration",
            "unchanged": False,
            "why": (
                "Theory 27 section 1 states 'Its formal effect interval is Eff(E) = "
                "[0, 1_E]' and section 2 defines Test_formal(E) as every finite "
                "family of effects summing to 1_E; Theory 41 restates Eff(E) = "
                "{e in E : 0 <= e <= 1_E}. The interval is certified Theory, so "
                "017.21a should not have listed it as its own premise"
            ),
            "what_remains_correctly_fenced": (
                "formal effecthood is still not physical Test Realization; that "
                "fence is Theory 41's, and 017.21a kept it"
            ),
            "effect_on_the_017_21a_verdict": (
                "the conditional result is strictly stronger than advertised: one of "
                "its three declarations was already authority"
            ),
        },
        {
            "id": "P3",
            "name": "operational state target",
            "017_21a_status": prior_status["P3"],
            "corrected_status": (
                "choice of abstraction boundary, mathematically free given "
                "certified structure, and now justified rather than declared"
            ),
            "unchanged": False,
            "why": (
                "the mathematics is certified: Theory 27 section 2 gives both "
                "P(V) -> S(E) and S(E) x Eff(E) -> [0,1]. What 017.21a added was not "
                "new mathematics but a decision about where the compiler's codomain "
                "sits. That decision is now earned: one-shot evaluation provably "
                "factors through S(E), and S(E) is minimal because Eff(E) separates "
                "states"
            ),
            "residual_declarative_content": (
                "identifying two ambient rays with the same induced state as the same "
                "compiler output is an identity convention at the realization layer; "
                "current TDM vocabulary neither requires nor forbids it"
            ),
        },
        {
            "id": "P4",
            "name": "invariant central-character extension",
            "017_21a_status": prior_status["P4"],
            "corrected_status": (
                "genuinely new principle, but a representative-selection principle "
                "rather than a behavioral constraint"
            ),
            "unchanged": False,
            "is_it_forced_by_canonicality": False,
            "why_not_forced": (
                "017.20's canonicality ladder asks for invariance under declared "
                "source equivalence and equivariance under source automorphisms. "
                "017.21a itself proves the source group acts trivially on E, so "
                "source-side naturality imposes nothing at all on the target. P4 "
                "instead demands a fixed point of the target automorphism group, "
                "which is a different and additional requirement"
            ),
            "is_it_stronger_than_the_stated_canonicality_requirement": True,
            "equivalent_honest_formulation": (
                "no-unsupplied-structure: the compiled state must be definable from "
                "the selected central character alone, without choosing a frame in "
                "Sym_2(R) that the Type Schema never supplied. The invariant states "
                "are exactly the states with no such extra frame content, so this "
                "formulation and P4 pick out the same family"
            ),
            "is_it_unnecessary_after_the_operational_quotient": (
                "unnecessary for public behavior, necessary for canonical "
                "representative selection"
            ),
            "exact_role": (
                "lambda -> omega_lambda is a bijection from S(E)/~_Sigma onto the "
                "invariant states, so P4 chooses one canonical representative per "
                "observational class. It also does real work that 017.21a did not "
                "claim: together with the declared deterministic behavior it forces "
                "the declared test to be the intrinsic central test"
            ),
            "is_it_still_load_bearing_for_017_21a_uniqueness": True,
            "load_bearing_note": (
                "without P4 the realization class is at least an RP^1 family, so "
                "017.21a's omega_t counterexample stands and unconditional C3 remains "
                "correctly rejected"
            ),
        },
    ]

    return {
        "prior_premise_artifact": {
            "path": (
                "issues/017-generalized-born-rule/017.21a-Code-attachments/"
                "declared_premises.json"
            ),
            "sha256": PRIOR_PREMISES_SHA256,
            "schema": PRIOR_PREMISE_SCHEMA,
        },
        "017_21a_claimed_these_as_its_own_declarations": {
            key: value for key, value in claimed_new.items()
        },
        "corrections": corrections,
        "summary": {
            "already_certified_OT_structure": ["P1", "P2"],
            "derived_and_now_justified_abstraction_boundary": ["P3"],
            "new_mathematical_choice": [],
            "new_canonicity_or_naturality_principle": ["P4"],
            "unnecessary_after_the_operational_quotient": [],
            "unnecessary_for_public_behavior_only": ["P4"],
        },
        "language_correction": (
            "the 'new declaration' label must be dropped for P2 and softened for P3; "
            "only P4 is a new principle, and it is a representative-selection "
            "principle, not a probability-fixing axiom"
        ),
    }


# ---------------------------------------------------------------------------
# 11. The public semantic object
# ---------------------------------------------------------------------------


def audit_public_semantics(source: SourceSchema) -> dict[str, object]:
    labels = source.decision_labels
    family = public_family(ORIENTATION_ONE, source)
    total = all(datum in family for datum in source.inputs)
    typed = all(is_distribution(row) for row in family.values())
    exact = all(
        isinstance(value, Fraction)
        for row in family.values()
        for value in row.values()
    )
    return {
        "type": "F_{R,d} : Input_{Sigma,d} -> Delta(A_d)",
        "packaging": (
            "for several declared operations, package the maps as one typed family "
            "F_R = (F_{R,d})_{d in Ops_Sigma}; the audited Sigma declares one "
            "operation, so the family has a single component"
        ),
        "declared_operations": 1,
        "answer_labels": list(labels),
        "admitted_inputs": len(source.inputs),
        "is_total": total,
        "every_value_is_a_distribution": typed,
        "every_value_is_exact_rational": exact,
        "what_it_is": (
            "the Sigma-semantics of one realization: the complete observable "
            "projection of an internal realization onto the declared typed boundary"
        ),
        "what_it_is_not": {
            "the_semantics_of_a_TDM_contract": (
                "no: the contract is the Type Schema itself, which constrains which "
                "families are admissible but does not single one out"
            ),
            "the_identity_of_a_TDM_instance": (
                "no: 017.10 makes instance identity Type Schema + Training, and "
                "distinct artifacts can induce one F"
            ),
            "only_one_observable_projection_of_a_richer_instance": (
                "yes: this is the correct reading, and it is why behavioral "
                "equivalence must not be equated with artifact identity"
            ),
        },
        "conformance_note": (
            "the declared schema rule already fixes every answer, so F carries no "
            "predictive content here; it is a conformance object"
        ),
    }


# ---------------------------------------------------------------------------
# 12. Contract, observation, and instance identity
# ---------------------------------------------------------------------------


def audit_identity_layers(source: SourceSchema) -> dict[str, object]:
    first = public_family(ORIENTATION_ONE, source)
    second = public_family(ORIENTATION_TWO, source)
    return {
        "contract_equivalence": {
            "definition": "two models satisfy the same Type Schema",
            "holds_for_the_two_orientations": True,
            "why": "both realize the identical declared Sigma without modification",
        },
        "observational_equivalence": {
            "definition": (
                "two realizations induce the same typed distribution on every "
                "admitted input and declared operation"
            ),
            "holds_for_the_two_orientations": first == second,
        },
        "instance_identity": {
            "definition_in_force": "TDM instance = Type Schema + Training = TSAT",
            "authority": "017.10 sections 7 and 11; top-level 15 section 9",
            "status_for_the_two_orientations": "undefined by current vocabulary",
            "why": (
                "both orientations have the identical Type Schema and identical "
                "Training, namely none: zero learned parameters. TSAT therefore "
                "assigns them the same identity while they remain distinct "
                "artifacts with distinct internal content"
            ),
        },
        "terminology_defect": {
            "exact_defect": (
                "TSAT does not determine a TDM instance when Training is empty. For "
                "zero-training compilations, the constitutive choices made by the "
                "compiler are neither Type Schema nor Training, so the identity "
                "decomposition has no slot for them"
            ),
            "why_it_matters_now": (
                "017.21a produced exactly such an artifact: a fully compiled model "
                "whose only remaining content is a constitutive choice invisible to "
                "both halves of TSAT"
            ),
            "options_not_chosen_here": [
                "widen Training to include zero-parameter constitutive choices",
                "define TDM instance identity as an equivalence class under "
                "observational equivalence",
                "add a third identity component for Constitution",
            ],
            "recommendation": (
                "open a separate terminology issue; do not redefine TSAT inside an "
                "audit result"
            ),
            "recommended_successor_issue": (
                "TDM instance identity for zero-training constitutions: does TSAT "
                "need a Constitution component?"
            ),
        },
        "do_not_collapse": (
            "contract equivalence, observational equivalence, and instance identity "
            "are three different relations; this audit establishes the first two for "
            "the orientation pair and leaves the third open by authority"
        ),
    }


# ---------------------------------------------------------------------------
# 13. Residue, training relevance, and successor feasibility
# ---------------------------------------------------------------------------


def parametric_realization(
    lambda_incident: Fraction, lambda_nonincident: Fraction, orientation: int
) -> Realization:
    """The candidate successor family: one invariant state per source orbit."""
    if orientation not in (1, 2):
        raise ValueError("orientation must be 1 or 2")
    incident_slot = Z_MATRIX if orientation == 1 else Z_SCALAR
    nonincident_slot = Z_SCALAR if orientation == 1 else Z_MATRIX
    return Realization(
        identifier=(
            f"theta-{_fraction_text(lambda_incident)}-"
            f"{_fraction_text(lambda_nonincident)}-orientation-{orientation}"
        ),
        compiler=(
            ("incident", omega_lambda(lambda_incident)),
            ("nonincident", omega_lambda(lambda_nonincident)),
        ),
        test=(("incident", incident_slot), ("nonincident", nonincident_slot)),
        description="successor feasibility probe; no training is performed",
    )


def audit_successor_feasibility(source: SourceSchema) -> dict[str, object]:
    """Check identifiability and orientation absorption for theta = (l_I, l_N)."""
    grid = tuple(Fraction(index, 4) for index in range(5))
    families: dict[tuple[str, str], tuple[object, ...]] = {}
    orientation_absorbed = True
    observable_directly = True
    all_valid = True
    for lambda_incident in grid:
        for lambda_nonincident in grid:
            first = parametric_realization(lambda_incident, lambda_nonincident, 1)
            mirrored = parametric_realization(
                ONE - lambda_incident, ONE - lambda_nonincident, 2
            )
            first_family = public_family(first, source)
            mirrored_family = public_family(mirrored, source)
            orientation_absorbed &= first_family == mirrored_family
            all_valid &= (
                first.test_is_normalized
                and first.test_slots_are_effects
                and first.states_are_states
                and all(is_distribution(row) for row in first_family.values())
            )
            incidence = source.incidence
            assert incidence is not None
            witness = next(iter(sorted(incidence)))
            observable_directly &= (
                first_family[witness]["incident"] == lambda_incident
            )
            families[
                (_fraction_text(lambda_incident), _fraction_text(lambda_nonincident))
            ] = _family_key(first_family, source.decision_labels)

    identifiable = len(set(families.values())) == len(families)

    return {
        "candidate_residual": "theta = (lambda_I, lambda_N) in [0,1]^2",
        "candidate_family": "omega_lambda(A, s) = lambda Tr(A)/2 + (1 - lambda) s",
        "parameter_count": 2,
        "grid_probed": [_fraction_text(value) for value in grid],
        "probed_theta_count": len(families),
        "all_probes_are_well_formed_realizations": all_valid,
        "theta_is_identifiable_from_the_declared_surface": identifiable,
        "theta_is_directly_observable": observable_directly,
        "observability_argument": (
            "lambda_I = omega_{lambda_I}(z_M) is literally the declared probability "
            "of the answer 'incident' on an incident input, so a nonzero residual is "
            "read off the public surface rather than inferred"
        ),
        "orientation_bit_is_absorbed": orientation_absorbed,
        "absorption_statement": (
            "for every theta, orientation 1 at theta and orientation 2 at "
            "(1 - lambda_I, 1 - lambda_N) induce the identical public family, so the "
            "017.21a orientation bit becomes an exact involution of the parameter "
            "space rather than an extra degree of freedom; a trained theta absorbs it"
        ),
        "what_the_successor_Type_Schema_must_declare": [
            "the Point and Line sorts",
            "the Fano incidence structure",
            "the presentation equivalence and its automorphism group",
            "the induced input orbit quotient",
            "the answer type IncidenceDecision",
        ],
        "what_it_must_not_declare": [
            "the decision rule",
            "any per-orbit probability",
            "any preparation, effect, or ray coordinate",
        ],
        "hard_precondition": (
            "the current fixture declares the deterministic rule, which pins theta "
            "to an endpoint and leaves nothing to learn. The successor is only a real "
            "experiment if the answers supplied to Training are not derivable from "
            "Sigma, so the source must declare structure while the answer statistics "
            "come from data"
        ),
        "derived_versus_learned_split": {
            "derived_by_Sigma": (
                "the two-cell orbit partition of the 49 inputs, the automorphism "
                "group of order 168, the central test, and the invariant family"
            ),
            "learned_by_Training": "exactly two scalars in [0,1]",
            "baseline_burden": (
                "a matched untyped learner must recover the orbit partition from the "
                "same source information before it can estimate two numbers"
            ),
        },
        "is_the_fixture_mathematically_earned": True,
        "earned_qualification": (
            "earned as a mathematical handoff: the residual is observable, "
            "identifiable, finite dimensional, and orientation free. Not yet earned "
            "as a training-economy claim, which requires the matched benchmark"
        ),
        "not_executed_here": {
            "training_data_generated": False,
            "optimizer_used": False,
            "benchmark_run": False,
            "model_comparison_run": False,
        },
    }


def audit_residue_ledger() -> dict[str, object]:
    return {
        "public_TDM_behavior": {
            "residue": "none",
            "cardinality": 1,
            "why": (
                "the declared schema rule fixes every answer, so exactly one public "
                "family F is admissible for the audited Sigma"
            ),
            "observable_under_Sigma": True,
            "training_eligible": False,
            "training_note": "nothing is left to learn at this fixture scale",
        },
        "abstract_OT_realization": {
            "residue": (
                "the observational equivalence class of labeled realizations "
                "(compiled state, answer-to-effect labeling)"
            ),
            "cardinality": "at least a continuum",
            "exhibited_subfamily": "RP^1 of rank-one projection realizations",
            "cardinality_under_P4_and_centrality": 2,
            "specification_bits_under_P4": 1,
            "observable_under_Sigma": False,
            "is_certified_gauge": False,
            "training_eligible": False,
        },
        "ambient_ray_realization": {
            "residue": "the fiber of q : P(V) -> S(E) over the compiled state",
            "delta_s_fiber": "RP^11, real dimension 11",
            "tau_M_fiber": "one PO(2) family",
            "generic_fiber_real_dimension_at_least": 12,
            "observable_under_Sigma": False,
            "is_certified_gauge": False,
            "gauge_candidate_note": (
                "occurrence/fixed 06 calls the pointwise stabilizer "
                "K_pt = O(2) x O(12) a gauge candidate relative to the observables, "
                "not certified gauge"
            ),
            "training_eligible": False,
        },
        "physical_realization": {
            "residue": "the entire physical Test Realization and Outcome bridge",
            "grade": "C-empty",
            "observable_under_Sigma": False,
            "training_eligible": False,
            "unchanged_since": "017.21",
        },
        "comparison_with_017_21a": {
            "017_21a_headline": "one discrete orientation bit at the S(E) layer",
            "017_22_correction": (
                "the bit is real only relative to P4 and centrality; the "
                "observational class it sits inside is at least one dimensional, and "
                "in the successor's enlarged family the bit is absorbed by an exact "
                "parameter involution"
            ),
        },
    }


def audit_training_relevance() -> dict[str, object]:
    return {
        "admission_test": (
            "a residual may be handed to Training only if it is observable through a "
            "declared TDM surface or an explicitly named future target, and is not "
            "already derivable or quotientable under the accepted semantics"
        ),
        "candidates": [
            {
                "degree_of_freedom": "the central orientation bit",
                "observable_under_Sigma": False,
                "derivable": False,
                "quotientable": True,
                "verdict": "non-training residue",
                "why": (
                    "both orientations produce identical declared behavior, so no "
                    "typed loss can prefer one; in the successor family it is "
                    "absorbed by theta -> 1 - theta"
                ),
            },
            {
                "degree_of_freedom": "the ambient preparation ray within a fiber",
                "observable_under_Sigma": False,
                "derivable": False,
                "quotientable": True,
                "verdict": "non-training residue",
                "why": "one-shot evaluation provably factors through S(E)",
            },
            {
                "degree_of_freedom": (
                    "the non-invariant directions of a compiled state, that is the "
                    "fiber of ~_Sigma"
                ),
                "observable_under_Sigma": False,
                "derivable": False,
                "quotientable": True,
                "verdict": "non-training residue under the present Sigma",
                "why": (
                    "the declared test observes only lambda; these directions become "
                    "observable only if Sigma declares a richer E_Sigma"
                ),
            },
            {
                "degree_of_freedom": "the central observable lambda per source orbit",
                "observable_under_Sigma": True,
                "derivable": True,
                "quotientable": False,
                "verdict": (
                    "the only legitimate Training target, and only once Sigma stops "
                    "declaring the decision rule"
                ),
                "why": (
                    "lambda = omega(z_M) is a declared answer probability, so it is "
                    "observable and identifiable; under the current fixture it is "
                    "derivable from the declared rule and therefore not yet a target"
                ),
            },
        ],
        "current_fixture_has_a_nonempty_training_problem": False,
        "successor_fixture_would_have_one": True,
    }


# ---------------------------------------------------------------------------
# 14. Control: the declared relation is still load-bearing
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
    failed_closed = False
    message = ""
    try:
        public_family(ORIENTATION_ONE, control)
    except MissingDeclaredRelation as error:
        failed_closed = True
        message = str(error)
    return {
        "what_was_removed": "the semantic meaning of the 21 declared incidence rows",
        "what_was_preserved": {
            "both_typed_sorts": control.points == source.points
            and control.lines == source.lines,
            "all_49_admitted_inputs": len(control.inputs) == len(source.inputs),
            "all_21_rows_as_opaque_payload": len(control.opaque_rows) == 21,
            "the_declared_answer_labels": control.decision_labels
            == source.decision_labels,
        },
        "public_family_fails_closed": failed_closed,
        "failure_message": message,
        "consequence": (
            "the public semantic object F_R cannot even be formed without the "
            "declared relation, so E_Sigma and both equivalence relations inherit "
            "that dependency"
        ),
    }


# ---------------------------------------------------------------------------
# 15. Authority and payload
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    theory = f"quilt+s3://protology#package=occurrence/theory@{THEORY_REVISION_CONSULTED}"
    fixed = f"quilt+s3://protology#package=occurrence/fixed@{FIXED_REVISION}"
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.22-Task-operational-TDM-equivalence-and-S-E-state-boundary-audit.md"
        ),
        "primary_gpt_authority_named_by_the_task": PRIMARY_GPT_AUTHORITY,
        "theory_revision_consulted": THEORY_REVISION_CONSULTED,
        "theory_revision_cited_by_017_21a": THEORY_REVISION_CITED_BY_017_21A,
        "upstream_used": [
            {
                "uri": f"{theory}&path=27-the-lift-theorem-and-probability-law.md",
                "supplies": (
                    "Eff(E) = [0,1_E]; Test_formal(E); the factorization "
                    "P(V) -> S(E) followed by S(E) x Eff(E) -> [0,1]; the explicit "
                    "fence that physical preparation and measurement realization "
                    "remain open"
                ),
            },
            {
                "uri": (
                    f"{theory}&path=41-effect-census-test-admissibility-"
                    "and-sce-sharp-shadow.md"
                ),
                "supplies": (
                    "formal effect membership Eff(E) = {e in E : 0 <= e <= 1_E}, "
                    "intrinsic finite common-test extendability, and the formal "
                    "versus physical boundary"
                ),
            },
            {
                "uri": (
                    f"{fixed}&path=06-reduced-ordered-space-and-liftable-dynamics.md"
                ),
                "sha256": FIXED_SURFACE_SHA256,
                "supplies": (
                    "E is a unital ordered Jordan algebra isomorphic to "
                    "Sym_2(R) + R; Aut(E,E_+,1_E) = O(2)/{+-I}; K_pt = O(2) x O(12) "
                    "as a gauge candidate; multiplicities 2 and 12"
                ),
            },
            {
                "uri": f"{gpt}&path=15-type-the-model-not-merely-the-decision.md",
                "supplies": (
                    "the notation E_Sigma subset Eff(E) for a schema-induced effect "
                    "family, and the reading that Training constitutes a state over "
                    "that family; also that Type Schema is part of model identity"
                ),
            },
            {
                "uri": (
                    f"{gpt}&path=issues/017-jev-pivot/"
                    "017.10-GPT-typed-decision-model-type-schema-and-TSAT.md"
                ),
                "supplies": (
                    "TDM, Type Schema, Training, and TSAT; the public typed boundary "
                    "and the statement that internals may differ between "
                    "implementations"
                ),
            },
            {
                "uri": (
                    f"{gpt}&path=issues/017-jev-pivot/"
                    "017.21a-Kiro-unfenced-central-constitution-follow-up.md"
                ),
                "supplies": "the result being reclassified",
            },
        ],
        "local_pins": [
            {
                "path": (
                    "issues/017-generalized-born-rule/017.20-Code-attachments/"
                    "source_fixture.json"
                ),
                "sha256": SOURCE_SHA256,
            },
            {
                "path": (
                    "issues/017-generalized-born-rule/017.21a-Code-attachments/"
                    "declared_premises.json"
                ),
                "sha256": PRIOR_PREMISES_SHA256,
            },
            {
                "path": (
                    "issues/017-generalized-born-rule/017.21a-Code-attachments/"
                    "central_constitution_audit.json"
                ),
                "sha256": PRIOR_AUDIT_SHA256,
            },
        ],
        "still_absent": [
            "a physical Test Realization for IncidenceDecision",
            "a canonical ambient preparation ray",
            "a TDM instance-identity rule for zero-training constitutions",
            "a source rule selecting the central orientation",
        ],
        "no_new_premise_introduced": True,
    }


def canonical_table_bytes() -> bytes:
    source, _ = load_source()
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)
    table = build_table(source, orbits)
    return (json.dumps(table, indent=2, sort_keys=True) + "\n").encode()


def build_payload() -> dict[str, object]:
    source, _ = load_source()
    premises = load_prior_premises()
    prior_audit = load_prior_audit()
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)

    effects = audit_effect_family(source)
    state_equivalence = audit_state_equivalence()
    invariant = audit_invariant_family()
    realization = audit_realization_equivalence(source)
    forcing = audit_p4_forcing()
    orientation = audit_orientation_classification(source)
    ambient = audit_ambient_boundary()
    boundary = audit_state_boundary(effects)
    ledger = audit_premise_ledger(premises)
    semantics = audit_public_semantics(source)
    identity = audit_identity_layers(source)
    successor = audit_successor_feasibility(source)
    residue = audit_residue_ledger()
    training = audit_training_relevance()
    control = audit_control(source)

    table = json.loads(canonical_table_bytes())
    table_digest = _sha256(canonical_table_bytes())

    prior_orientation_count = prior_audit["compiler"]["rival_search"][  # type: ignore[index]
        "binding_count"
    ]

    checks = {
        "source_bytes_match_017_20_freeze": True,
        "prior_premises_bytes_match_017_21a": True,
        "prior_audit_bytes_match_017_21a": True,
        "prior_audit_reported_two_bindings": prior_orientation_count == 2,
        "source_group_order_is_168": len(automorphisms) == 168,
        "two_source_orbits_21_and_28": sorted(len(orbit) for orbit in orbits)
        == [21, 28],
        "E_Sigma_has_two_members": effects["E_Sigma_cardinality"] == 2,
        "E_Sigma_members_are_effects": effects["all_members_are_effects"],
        "E_Sigma_members_are_central": effects["all_members_are_central"],
        "E_Sigma_members_sum_to_unit": effects["members_sum_to_unit"],
        "E_Sigma_span_is_proper": effects["span_is_proper"],
        "eff_E_contains_a_basis": effects["eff_E_spans_the_algebra"]["is_a_basis"],  # type: ignore[index]
        "all_probe_states_are_states": state_equivalence[
            "all_probe_states_are_states"
        ],
        "sigma_relation_is_an_equivalence": all(
            state_equivalence["is_an_equivalence_relation"][key]  # type: ignore[index]
            for key in ("reflexive", "symmetric", "transitive")
        ),
        "sigma_relation_is_exactly_equal_lambda": state_equivalence[
            "quotient_invariant"
        ]["lambda_characterizes_the_relation"],  # type: ignore[index]
        "coarse_graining_does_not_refine_the_quotient": state_equivalence[
            "quotient_invariant"
        ]["coarse_grained_closure_adds_nothing"],  # type: ignore[index]
        "quotient_is_the_unit_interval": state_equivalence["quotient_invariant"][
            "range_is_the_unit_interval"
        ],  # type: ignore[index]
        "fat_fiber_is_observationally_identical": state_equivalence[
            "fat_fiber_witness"
        ]["all_Sigma_equivalent"],  # type: ignore[index]
        "fat_fiber_members_are_distinct_states": state_equivalence[
            "fat_fiber_witness"
        ]["pairwise_distinct"],  # type: ignore[index]
        "fat_fiber_is_separated_outside_E_Sigma": all(
            row["separated"]  # type: ignore[index]
            for row in state_equivalence["fat_fiber_witness"][  # type: ignore[index]
                "separating_effects_outside_E_Sigma"
            ]
        ),
        "lambda_zero_fiber_is_a_single_state": state_equivalence[
            "degenerate_fiber"
        ]["fiber_is_the_single_state_delta_s"],  # type: ignore[index]
        "eff_E_separation_makes_the_relation_equality": state_equivalence[
            "special_cases"
        ]["E_Sigma_equals_Eff_E"]["relation_becomes_equality"],  # type: ignore[index]
        "central_test_separates_the_two_compiled_states": state_equivalence[
            "special_cases"
        ]["the_017_21a_central_test"][  # type: ignore[index]
            "the_two_compiled_states_are_not_Sigma_equivalent"
        ],
        "invariance_constraint_rank_is_two": invariant["constraint_matrix"]["rank"]  # type: ignore[index]
        == 2,
        "invariant_kernel_is_trace_plus_scalar": invariant["constraint_matrix"][
            "kernel_is_spanned_by_those_two"
        ],  # type: ignore[index]
        "omega_lambda_are_all_states": invariant["all_family_members_are_states"],
        "omega_lambda_are_all_invariant": invariant[
            "all_family_members_are_invariant"
        ],
        "lambda_equals_value_on_z_M": invariant["lambda_is_the_value_on_z_M"],
        "outside_unit_interval_fails_positivity": invariant[
            "positivity_bounds_lambda_to_unit_interval"
        ],
        "every_non_invariant_probe_is_detected": invariant[
            "non_invariant_probe_states"
        ]
        == invariant["non_invariant_probe_states_detected"],
        "invariant_family_is_a_cross_section": invariant[
            "is_a_cross_section_of_the_observational_quotient"
        ],
        "all_realizations_well_formed": realization[
            "all_realizations_are_well_formed"
        ],
        "all_realizations_reproduce_declared_behavior": realization[
            "all_reproduce_the_declared_behavior"
        ],
        "all_realizations_form_one_class": realization["single_equivalence_class"],
        "orientations_are_TDM_equivalent": realization[
            "orientations_are_TDM_equivalent"
        ],
        "realization_relation_is_an_equivalence": all(
            realization["relation_checks"][key]  # type: ignore[index]
            for key in ("reflexive", "symmetric", "transitive")
        ),
        "class_contains_more_than_the_two_orientations": realization[
            "distinct_internal_realizations"
        ]
        > 2,
        "P4_search_is_not_vacuous": forcing["search_space"][  # type: ignore[index]
            "search_is_not_vacuous"
        ],
        "P4_forces_exactly_two_labeled_realizations": forcing[
            "exactly_two_solutions"
        ],
        "P4_solutions_are_the_two_orientations": forcing[
            "solutions_are_the_two_orientations"
        ],
        "P4_derives_centrality_of_the_test": forcing[
            "centrality_is_derived_not_assumed"
        ],
        "orientation_is_unobservable": not orientation["questions"][  # type: ignore[index]
            "distinguishable_by_any_Sigma_admitted_observation"
        ],
        "orientation_states_are_distinct_in_S_E": orientation["questions"][  # type: ignore[index]
            "distinct_in_S_E"
        ],
        "orientation_states_are_not_Sigma_equivalent": not orientation[
            "questions"
        ]["states_are_Sigma_equivalent"],  # type: ignore[index]
        "no_automorphism_swaps_the_central_effects": not orientation["gauge_test"][  # type: ignore[index]
            "some_automorphism_swaps_the_central_effects"
        ],
        "central_effects_are_pointwise_fixed": orientation["gauge_test"][  # type: ignore[index]
            "central_effects_are_pointwise_fixed"
        ],
        "orientation_is_not_certified_gauge": not orientation["gauge_test"][  # type: ignore[index]
            "is_certified_representation_gauge"
        ],
        "multiple_rays_realize_tau_M": ambient["level_1_formal_preparation"][  # type: ignore[index]
            "projectively_distinct_tau_M_realizers"
        ]
        > 1,
        "multiple_rays_realize_delta_s": ambient["level_1_formal_preparation"][  # type: ignore[index]
            "all_sampled_scalar_rays_induce_delta_s"
        ],
        "delta_s_rays_are_projectively_distinct": ambient[
            "level_1_formal_preparation"
        ]["delta_s_realizers_are_projectively_distinct"],  # type: ignore[index]
        "same_copy_near_miss_fails": not ambient["level_1_formal_preparation"][  # type: ignore[index]
            "same_copy_near_miss"
        ]["induces_tau_M"],
        "fiber_is_invisible_on_every_probe_effect": ambient[
            "level_2_formal_evaluation"
        ]["mechanically_checked_on_probe_effects"],  # type: ignore[index]
        "ray_information_is_not_observable_under_Sigma": not ambient[
            "forgotten_information"
        ]["observable_through_the_declared_one_shot_interface"],  # type: ignore[index]
        "S_E_is_minimal_for_the_formal_family": boundary[
            "S_E_is_minimal_for_the_full_formal_family"
        ],
        "public_family_is_total": semantics["is_total"],
        "public_family_is_typed": semantics["every_value_is_a_distribution"],
        "public_family_is_exact": semantics["every_value_is_exact_rational"],
        "017_21a_did_label_P2_and_P3_as_declarations": all(
            ledger["017_21a_claimed_these_as_its_own_declarations"][key]  # type: ignore[index]
            for key in ("P2", "P3", "P4")
        ),
        "orientations_share_the_contract": identity["contract_equivalence"][  # type: ignore[index]
            "holds_for_the_two_orientations"
        ],
        "orientations_share_the_observable_semantics": identity[
            "observational_equivalence"
        ]["holds_for_the_two_orientations"],  # type: ignore[index]
        "theta_is_identifiable": successor[
            "theta_is_identifiable_from_the_declared_surface"
        ],
        "theta_is_directly_observable": successor["theta_is_directly_observable"],
        "orientation_bit_is_absorbed_by_theta": successor[
            "orientation_bit_is_absorbed"
        ],
        "successor_probes_are_well_formed": successor[
            "all_probes_are_well_formed_realizations"
        ],
        "no_training_executed": all(
            value is False for value in successor["not_executed_here"].values()  # type: ignore[union-attr]
        ),
        "control_fails_closed": control["public_family_fails_closed"],
        "control_preserves_the_raw_carrier": all(
            value is True for value in control["what_was_preserved"].values()  # type: ignore[union-attr]
        ),
        "table_row_count_is_98": table["row_count"] == 2 * len(source.inputs),
        "table_public_columns_agree": table[
            "public_columns_agree_across_orientations"
        ],
        "table_internal_columns_differ": table[
            "internal_state_columns_differ_across_orientations"
        ],
    }
    if not all(checks.values()):
        broken = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"mechanical audit checks failed: {broken}")

    return {
        "schema": AUDIT_SCHEMA,
        "task": (
            "occurrence/gpt Issue 017.22 operational TDM equivalence and the S(E) "
            "state boundary"
        ),
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "authority": authority_record(),
        "source": {
            "schema": SOURCE_SCHEMA,
            "sha256": SOURCE_SHA256,
            "admitted_inputs": len(source.inputs),
            "automorphism_group_order": len(automorphisms),
            "orbits": [
                {
                    "semantic_class": source.semantic_class(orbit[0]),
                    "size": len(orbit),
                }
                for orbit in orbits
            ],
        },
        "A_premise_correction": ledger,
        "B_public_semantics": semantics,
        "C_state_equivalence": {
            "effect_family": effects,
            "relation": state_equivalence,
        },
        "D_realization_equivalence": {
            "relation": realization,
            "P4_forcing": forcing,
            "orientation_classification": orientation,
        },
        "E_ambient_rays": ambient,
        "F_state_boundary": boundary,
        "G_invariant_family": invariant,
        "H_residue_ledger": residue,
        "I_training_relevance": training,
        "J_successor": successor,
        "control": audit_control(source),
        "artifacts": {
            "orientation_behavior_table": {
                "path": TABLE_REPO_PATH,
                "schema": TABLE_SCHEMA,
                "sha256": table_digest,
                "rows": table["row_count"],
            }
        },
        "reproducibility": {
            "runtime": "Python >=3.11; standard library only",
            "commands": [
                "uv run --frozen python issues/017-generalized-born-rule/"
                "017.22-Code-attachments/audit_operational_equivalence.py --check",
                "uv run --frozen pytest issues/017-generalized-born-rule/"
                "017.22-Code-attachments/test_audit_operational_equivalence.py",
            ],
            "network_required": False,
            "learned_parameters": 0,
            "optimizer_or_training_imported": False,
            "interact_imported": False,
            "numpy_imported": False,
            "exact_arithmetic": "Fraction and Q(sqrt 7)",
        },
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": True,
        "claim_fences": [
            "no physical Test Realization or Outcome constitution is claimed",
            "no predictive, accuracy, or training-economy claim is made",
            "operational equivalence is not called gauge; no certified symmetry "
            "relates the two orientations",
            "public behavioral equivalence is not equated with TSAT instance identity",
            "nothing is trained, and no data or benchmark is generated",
            "the S(E) boundary is claimed for one-shot evaluation only; Evaluate is "
            "not Enact",
        ],
        "disposition": (
            "STRONG POSITIVE ON THE STATE BOUNDARY WITH AN OPEN INSTANCE-IDENTITY "
            "DEFECT: one-shot TDM evaluation factors through S(E) by certified "
            "Theory and S(E) is minimal because Eff(E) separates states; both "
            "observational equivalences are formalized and verified; the 017.21a "
            "orientation bit and the ambient-ray fibers are provably unobservable at "
            "the declared boundary and are provably not certified gauge; the "
            "Aut(E,E_+,1_E)-invariant states are exactly omega_lambda for lambda in "
            "[0,1]; P2 was already certified Theory and P4 is a representative-"
            "selection principle that also forces the central test; the two-parameter "
            "Fano successor is mathematically earned, conditional on a Type Schema "
            "that declares structure without declaring the decision rule"
        ),
    }


def canonical_audit_bytes() -> bytes:
    return (json.dumps(build_payload(), indent=2, sort_keys=True) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify both committed artifacts byte-for-byte instead of printing",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write both canonical artifacts to disk",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="print the orientation behavior table instead of the audit",
    )
    args = parser.parse_args(argv)

    audit = canonical_audit_bytes()
    table = canonical_table_bytes()

    if args.check:
        failures = []
        for path, expected in ((ARTIFACT_PATH, audit), (TABLE_PATH, table)):
            if not path.exists():
                failures.append(f"missing artifact: {path}")
                continue
            observed = path.read_bytes()
            if observed != expected:
                failures.append(
                    f"artifact mismatch for {path.name}: observed "
                    f"{_sha256(observed)}, expected {_sha256(expected)}"
                )
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1
        print(
            "017.22 audit verified: "
            f"{_sha256(audit)} table {_sha256(table)}"
        )
        return 0

    if args.write:
        ARTIFACT_PATH.write_bytes(audit)
        TABLE_PATH.write_bytes(table)
        print(f"wrote {ARTIFACT_PATH.name} {_sha256(audit)}")
        print(f"wrote {TABLE_PATH.name} {_sha256(table)}")
        return 0

    sys.stdout.buffer.write(table if args.table else audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
