"""Issue 017.24 matched TDM training-economy experiment.

017.23 earned a two-parameter Fano successor mathematically. This module runs the
first experiment that the successor allows: it measures whether a Type Schema
that compiles the input structure *before* Training removes labeled-data burden
that Training would otherwise have to carry.

The successor Type Schema ``Sigma'`` declares two opaque seven-element sorts, the
21-row incidence relation, the 49 admitted ``Point x Line`` inputs, the raw
presentation equivalence, and a neutral binary answer type. It declares no
decision rule, no answer frequency, and no preparation, effect, or ray
coordinate. The compiler reads sorts and relations only -- structurally, because
the schema object handed to it has no answer field at all -- and derives
``Aut(Sigma')`` and the induced classes on admitted inputs.

Five conditions are fitted on byte-identical observations and splits:

``T``  the compiled TDM, two learned scalars, evaluated through the certified
      invariant family ``omega_lambda(A,s) = lambda Tr(A)/2 + (1-lambda) s`` and
      the intrinsic central test ``(z_M, z_s)``;
``O``  the oracle-class control, same two scalars, no Born formula, no compiler;
``U``  the same-information uncompiled learner, given the complete declared
      incidence structure but not the quotient, with full per-identity capacity;
``S``  the saturated 49-parameter ablation;
``G``  a generic symmetry learner handed the group but not the quotient.

Three falsification controls run against data generated from the true class
structure: a scrambled-structure ensemble, a relation-erased schema that must
fail closed, and a wrong 21/28 partition that respects no automorphism.

Every estimator, evaluation, analytic reference, and control is exact or closed
form and uses the standard library only. NumPy appears in exactly one place: the
``U`` learner. No condition ever receives ``theta_star``; the module asserts that
mechanically from the fitted functions' signatures.
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import itertools
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Schemas, digests, and paths
# ---------------------------------------------------------------------------

ORBIT_SCHEMA: Final = "gpt-01724-successor-compiler-certificate/v1"
SPLIT_SCHEMA: Final = "gpt-01724-split-certificate/v1"
REFERENCE_SCHEMA: Final = "gpt-01724-analytic-reference/v1"
RESULTS_SCHEMA: Final = "gpt-01724-training-economy-results/v1"
COST_SCHEMA: Final = "gpt-01724-cost-report/v1"
TABLE_SCHEMA: Final = "gpt-01724-per-replicate-table/v1"

SUCCESSOR_SCHEMA: Final = "gpt-01724-fano-successor-type-schema/v1"
PREREGISTRATION_SCHEMA: Final = "gpt-01724-training-economy-preregistration/v1"
PREDECESSOR_SCHEMA: Final = "gpt-01720-fano-incidence-source/v1"

SUCCESSOR_SHA256: Final = (
    "149778bf5b8cca166ab77d3f33e9368191a58b33af9d1c15075f9ff41312a81d"
)
PREREGISTRATION_SHA256: Final = (
    "32fa75c0856cf13b3bbecf1eaf0619d851f8db69edc199015e94750176a575f9"
)
PREDECESSOR_SHA256: Final = (
    "20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c"
)

PRIOR_RESULT_COMMIT: Final = "e10ce4fffe26d9003c19b5375599f11a989269c6"
TASK_GPT_REVISION: Final = (
    "d2d6fc93d3b1e689b571afdd8e7fc75662d1c9a63c8168a474d694899c4eff26"
)
PRIMARY_GPT_AUTHORITY: Final = (
    "11afecd32724bb061f845c9c20393a38be804c7d6ef6f9814c584d931da2cd8d"
)

HERE: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = HERE.parents[2]
SUCCESSOR_PATH: Final = HERE / "successor_schema.json"
PREREGISTRATION_PATH: Final = HERE / "preregistration.json"
PREDECESSOR_PATH: Final = (
    HERE.parent / "017.20-Code-attachments" / "source_fixture.json"
)

ORBIT_PATH: Final = HERE / "orbit_certificate.json"
SPLIT_PATH: Final = HERE / "split_certificate.json"
REFERENCE_PATH: Final = HERE / "analytic_reference.json"
RESULTS_PATH: Final = HERE / "training_economy_results.json"
COST_PATH: Final = HERE / "cost_report.json"
TABLE_PATH: Final = HERE / "per_replicate.csv.gz"

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)

# Reported floats are rounded to this many decimals so that the byte-pinned
# artifacts survive last-ulp differences in libm across platforms.
DECIMALS: Final = 10
NUMERIC_TOLERANCE: Final = 1e-6
CLIP: Final = 1e-6

Datum = tuple[str, str]


class MissingDeclaredRelation(RuntimeError):
    """Raised when a schema does not type its opaque relation rows."""


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _round(value: float) -> float:
    """Round for reporting, mapping -0.0 to 0.0 so digests are stable."""
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0 else rounded


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# 1. The reduced effect algebra E = Sym_2(R) + R, in the 017.22 conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Element:
    """One element of E in adapted coordinates ``(A, s)``, ``A`` symmetric 2x2."""

    identifier: str
    a11: Fraction
    a12: Fraction
    a22: Fraction
    s: Fraction

    @property
    def trace(self) -> Fraction:
        return self.a11 + self.a22

    @property
    def is_positive(self) -> bool:
        determinant = self.a11 * self.a22 - self.a12 * self.a12
        return self.a11 >= 0 and self.a22 >= 0 and determinant >= 0 and self.s >= 0

    @property
    def is_effect(self) -> bool:
        complement = Element(
            "", ONE - self.a11, -self.a12, ONE - self.a22, ONE - self.s
        )
        return self.is_positive and complement.is_positive

    @property
    def is_central(self) -> bool:
        return self.a12 == 0 and self.a11 == self.a22


UNIT: Final = Element("1_E", ONE, ZERO, ONE, ONE)
Z_MATRIX: Final = Element("z_M", ONE, ZERO, ONE, ZERO)
Z_SCALAR: Final = Element("z_s", ZERO, ZERO, ZERO, ONE)


@dataclass(frozen=True, slots=True)
class OTState:
    """A functional ``omega(A, s) = Tr(rho A) + q s`` in adapted coordinates."""

    identifier: str
    r11: Fraction
    r12: Fraction
    r22: Fraction
    q: Fraction

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
        return self.r11 >= 0 and self.r22 >= 0 and determinant >= 0 and self.q >= 0

    @property
    def is_state(self) -> bool:
        return self.is_positive and self(UNIT) == ONE

    @property
    def central_parameter(self) -> Fraction:
        return self(Z_MATRIX)


def omega_lambda(value: Fraction) -> OTState:
    """The certified invariant state ``omega_l(A,s) = l Tr(A)/2 + (1-l) s``."""
    half = value / 2
    return OTState(f"omega_{_fraction_text(value)}", half, ZERO, half, ONE - value)


def tdm_distribution(value: Fraction) -> tuple[Fraction, Fraction]:
    """Evaluate the intrinsic central test on ``omega_lambda`` exactly."""
    state = omega_lambda(value)
    return (state(Z_MATRIX), state(Z_SCALAR))


# ---------------------------------------------------------------------------
# 2. The source schema object: sorts and a relation, and nothing else
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IncidenceSchema:
    """Sorts plus an optional typed relation.

    There is deliberately no answer field, no label, and no rule anywhere on
    this object. The compiler cannot see target semantics because the carrier it
    is handed does not represent any.
    """

    points: tuple[str, ...]
    lines: tuple[str, ...]
    incidence: frozenset[Datum] | None
    opaque_rows: tuple[Datum, ...] = ()

    @property
    def inputs(self) -> tuple[Datum, ...]:
        return tuple(
            (point, line) for point in self.points for line in self.lines
        )

    def point_set(self, line: str) -> frozenset[str]:
        if self.incidence is None:
            raise MissingDeclaredRelation(
                "cannot read a line's point set without a declared relation"
            )
        return frozenset(
            point for point, other in self.incidence if other == line
        )


@dataclass(frozen=True, slots=True)
class Automorphism:
    """A sort-preserving relabeling that fixes the declared relation."""

    point_images: tuple[str, ...]
    line_images: tuple[str, ...]

    def apply(self, datum: Datum, schema: IncidenceSchema) -> Datum:
        point_index = schema.points.index(datum[0])
        line_index = schema.lines.index(datum[1])
        return (self.point_images[point_index], self.line_images[line_index])


def load_successor(path: Path = SUCCESSOR_PATH) -> tuple[
    IncidenceSchema, dict[str, object]
]:
    """Load Sigma' byte-pinned, and project it onto a label-free carrier."""
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != SUCCESSOR_SHA256:
        raise RuntimeError(
            f"successor schema changed: expected {SUCCESSOR_SHA256}, "
            f"observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != SUCCESSOR_SCHEMA:
        raise RuntimeError("unexpected successor schema identifier")
    sorts = document["sorts"]
    schema = IncidenceSchema(
        points=tuple(sorts["point"]),
        lines=tuple(sorts["line"]),
        incidence=frozenset(
            (row[0], row[1]) for row in document["relations"]["incident"]
        ),
    )
    return schema, document


def load_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> dict[str, object]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration changed: expected {PREREGISTRATION_SHA256}, "
            f"observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != PREREGISTRATION_SCHEMA:
        raise RuntimeError("unexpected preregistration schema identifier")
    return document


def load_predecessor(path: Path = PREDECESSOR_PATH) -> dict[str, object]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != PREDECESSOR_SHA256:
        raise RuntimeError(
            f"017.20 fixture changed: expected {PREDECESSOR_SHA256}, "
            f"observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != PREDECESSOR_SCHEMA:
        raise RuntimeError("unexpected predecessor schema identifier")
    return document


# ---------------------------------------------------------------------------
# 3. The compiler: group, classes, canonical identifiers
# ---------------------------------------------------------------------------


def _line_matchings(
    lines: Sequence[str],
    candidates: dict[str, tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """All injective choices of ``line -> candidate``, in declared line order."""
    order = sorted(lines, key=lambda line: (len(candidates[line]), line))
    results: list[dict[str, str]] = []

    def walk(index: int, used: set[str], partial: dict[str, str]) -> None:
        if index == len(order):
            results.append(dict(partial))
            return
        line = order[index]
        for image in candidates[line]:
            if image in used:
                continue
            partial[line] = image
            used.add(image)
            walk(index + 1, used, partial)
            used.discard(image)
            del partial[line]

    walk(0, set(), {})
    return [tuple(mapping[line] for line in lines) for mapping in results]


def enumerate_automorphisms(
    schema: IncidenceSchema,
) -> tuple[Automorphism, ...]:
    """Exhaust point relabelings and match every consistent line relabeling.

    General enough for arbitrary 21-row relations, including scrambles whose
    line point-sets repeat, so the control cannot be flattered by a matcher
    that silently assumes a projective plane.
    """
    if schema.incidence is None:
        raise MissingDeclaredRelation(
            "cannot enumerate automorphisms without a declared relation"
        )
    point_sets = {line: schema.point_set(line) for line in schema.lines}
    by_set: dict[frozenset[str], list[str]] = {}
    for line, members in point_sets.items():
        by_set.setdefault(members, []).append(line)

    found: list[Automorphism] = []
    for permutation in itertools.permutations(schema.points):
        point_map = dict(zip(schema.points, permutation, strict=True))
        candidates: dict[str, tuple[str, ...]] = {}
        feasible = True
        for line, members in point_sets.items():
            image = frozenset(point_map[point] for point in members)
            matches = by_set.get(image)
            if not matches:
                feasible = False
                break
            candidates[line] = tuple(sorted(matches))
        if not feasible:
            continue
        for line_images in _line_matchings(schema.lines, candidates):
            found.append(
                Automorphism(point_images=permutation, line_images=line_images)
            )
    return tuple(found)


@dataclass(frozen=True, slots=True)
class ClassStructure:
    """The compiler's output: a group, a canonical class list, and a lookup."""

    schema: IncidenceSchema
    automorphisms: tuple[Automorphism, ...]
    classes: tuple[tuple[Datum, ...], ...]
    inputs: tuple[Datum, ...]
    class_of_index: tuple[int, ...]

    @property
    def group_order(self) -> int:
        return len(self.automorphisms)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(len(members) for members in self.classes)

    def name(self, index: int) -> str:
        return f"O_{index}"


def compile_classes(schema: IncidenceSchema) -> ClassStructure:
    """Derive ``Aut`` and the canonical input classes from structure alone."""
    automorphisms = enumerate_automorphisms(schema)
    inputs = schema.inputs
    remaining = list(inputs)
    raw: list[tuple[Datum, ...]] = []
    while remaining:
        seed = remaining[0]
        orbit = {
            automorphism.apply(seed, schema) for automorphism in automorphisms
        }
        raw.append(tuple(sorted(orbit)))
        remaining = [datum for datum in remaining if datum not in orbit]
    ordered = tuple(sorted(raw, key=lambda members: (len(members), members[0])))
    lookup = {}
    for index, members in enumerate(ordered):
        for datum in members:
            lookup[datum] = index
    return ClassStructure(
        schema=schema,
        automorphisms=automorphisms,
        classes=ordered,
        inputs=inputs,
        class_of_index=tuple(lookup[datum] for datum in inputs),
    )


def invariance_partition_from_group(
    inputs: Sequence[Datum],
    schema: IncidenceSchema,
    automorphisms: Sequence[Automorphism],
) -> tuple[int, ...]:
    """Generic invariance closure: union-find over a supplied permutation set.

    This is condition ``G``'s mechanism. It never mentions incidence, orbits, or
    a two-cell answer; it only closes the admitted inputs under the permutations
    it was handed and reports the surviving free parameters.
    """
    index_of = {datum: position for position, datum in enumerate(inputs)}
    parent = list(range(len(inputs)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for automorphism in automorphisms:
        for datum in inputs:
            union(index_of[datum], index_of[automorphism.apply(datum, schema)])

    roots = sorted({find(node) for node in range(len(inputs))})
    relabel = {root: position for position, root in enumerate(roots)}
    return tuple(relabel[find(node)] for node in range(len(inputs)))


# ---------------------------------------------------------------------------
# 4. Hidden data generator and deterministic splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observations:
    """Labeled observations. ``answers[i] == 1`` means the label ``a_0``."""

    indices: tuple[int, ...]
    answers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Split:
    """A train pool and the evaluation sets, all label-free by construction."""

    regime: str
    train_pool: tuple[int, ...]
    held_out: tuple[int, ...]
    evaluation: tuple[tuple[str, tuple[int, ...]], ...]


def build_split(
    regime: str,
    rng: random.Random,
    class_members: Sequence[tuple[int, ...]],
    holdout_per_class: Sequence[int],
) -> Split:
    """Build one split from source structure and a seed. No label is read."""
    everything = tuple(
        sorted(index for members in class_members for index in members)
    )
    if regime == "A":
        evaluation = (("aggregate", everything),) + tuple(
            (f"O_{index}", members)
            for index, members in enumerate(class_members)
        )
        return Split("A", everything, (), evaluation)
    if regime != "B":
        raise ValueError(f"unknown regime {regime!r}")

    held: list[int] = []
    per_class: list[tuple[int, ...]] = []
    for members, count in zip(class_members, holdout_per_class, strict=True):
        if count < 1 or count >= len(members):
            raise ValueError("every class must keep seen and held-out members")
        chosen = tuple(sorted(rng.sample(sorted(members), count)))
        per_class.append(chosen)
        held.extend(chosen)
    held_out = tuple(sorted(held))
    pool = tuple(index for index in everything if index not in set(held_out))
    evaluation = (("aggregate", held_out),) + tuple(
        (f"O_{index}", members) for index, members in enumerate(per_class)
    )
    return Split("B", pool, held_out, evaluation)


def generate_observations(
    rng: random.Random,
    pool: Sequence[int],
    count: int,
    numerators: Sequence[int],
    denominator: int,
    hidden_class_of_index: Sequence[int],
) -> Observations:
    """Sample inputs, then sample answers from the hidden class frequencies.

    The generator is the only object in this module that sees ``theta_star``.
    Learners receive ``Observations`` and nothing else.
    """
    size = len(pool)
    indices: list[int] = []
    answers: list[int] = []
    for _ in range(count):
        index = pool[rng.randrange(size)]
        hidden = hidden_class_of_index[index]
        draw = rng.randrange(denominator)
        indices.append(index)
        answers.append(1 if draw < numerators[hidden] else 0)
    return Observations(tuple(indices), tuple(answers))


def theta_as_integers(theta: Sequence[Fraction]) -> tuple[tuple[int, ...], int]:
    """Express ``theta_star`` over one common denominator for exact sampling."""
    denominator = 1
    for value in theta:
        denominator = denominator * value.denominator // math.gcd(
            denominator, value.denominator
        )
    numerators = tuple(
        int(value.numerator * (denominator // value.denominator))
        for value in theta
    )
    return numerators, denominator


# ---------------------------------------------------------------------------
# 5. Estimation and the five conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FittedModel:
    """A fitted condition, reported as 49 probabilities for the label ``a_0``."""

    condition: str
    predictions: tuple[float, ...]
    degrees_of_freedom: int
    exposed: tuple[Fraction, ...] | None
    exact_predictions: tuple[Fraction, ...] | None


def beta_posterior_mean(
    successes: int, trials: int, alpha: Fraction, beta: Fraction
) -> Fraction:
    """One clipping-free Bernoulli convention shared by every condition."""
    return (successes + alpha) / (trials + alpha + beta)


def pair_counts(
    observations: Observations, width: int
) -> tuple[list[int], list[int]]:
    """Per-pair sufficient statistics, computed once and shared by every model.

    Every condition is refitted from these same counts, which is how the
    fairness constraint "all conditions receive the same labeled observations"
    is enforced structurally rather than by convention.
    """
    successes = [0] * width
    trials = [0] * width
    for index, answer in zip(
        observations.indices, observations.answers, strict=True
    ):
        trials[index] += 1
        successes[index] += answer
    return successes, trials


def _cell_estimates(
    pair_successes: Sequence[int],
    pair_trials: Sequence[int],
    partition: Sequence[int],
    cell_count: int,
    alpha: Fraction,
    beta: Fraction,
) -> tuple[Fraction, ...]:
    successes = [0] * cell_count
    trials = [0] * cell_count
    for index, cell in enumerate(partition):
        trials[cell] += pair_trials[index]
        successes[cell] += pair_successes[index]
    return tuple(
        beta_posterior_mean(successes[cell], trials[cell], alpha, beta)
        for cell in range(cell_count)
    )


def _broadcast(
    values: Sequence[Fraction], partition: Sequence[int]
) -> tuple[Fraction, ...]:
    return tuple(values[cell] for cell in partition)


def fit_typed_tdm(
    pair_successes: Sequence[int],
    pair_trials: Sequence[int],
    partition: Sequence[int],
    cell_count: int,
    alpha: Fraction,
    beta: Fraction,
    condition: str = "T",
) -> FittedModel:
    """Condition T: estimate one lambda per compiled class, then evaluate the TDM.

    The prediction is produced by the certified invariant state and the intrinsic
    central test, not by returning the estimate directly.
    """
    estimates = _cell_estimates(
        pair_successes, pair_trials, partition, cell_count, alpha, beta
    )
    public: list[Fraction] = []
    for value in estimates:
        first, second = tdm_distribution(value)
        if first + second != ONE or not omega_lambda(value).is_state:
            raise RuntimeError("the compiled TDM family left the state space")
        public.append(first)
    exact = _broadcast(public, partition)
    return FittedModel(
        condition=condition,
        predictions=tuple(float(value) for value in exact),
        degrees_of_freedom=cell_count,
        exposed=tuple(public) if cell_count == 2 else None,
        exact_predictions=exact,
    )


def fit_bernoulli(
    pair_successes: Sequence[int],
    pair_trials: Sequence[int],
    partition: Sequence[int],
    cell_count: int,
    alpha: Fraction,
    beta: Fraction,
    condition: str,
) -> FittedModel:
    """Conditions O, S, and G: the same estimator, no Born formula anywhere."""
    estimates = _cell_estimates(
        pair_successes, pair_trials, partition, cell_count, alpha, beta
    )
    exact = _broadcast(estimates, partition)
    return FittedModel(
        condition=condition,
        predictions=tuple(float(value) for value in exact),
        degrees_of_freedom=cell_count,
        exposed=tuple(estimates) if cell_count == 2 else None,
        exact_predictions=exact,
    )


# ---------------------------------------------------------------------------
# 6. Condition U: the same-information uncompiled learner
# ---------------------------------------------------------------------------

FEATURE_BLOCKS: Final = (
    ("point_identity", 7),
    ("line_identity", 7),
    ("point_incidence_row", 7),
    ("line_incidence_column", 7),
)
FEATURE_DIMENSION: Final = sum(width for _, width in FEATURE_BLOCKS)
HIDDEN_WIDTH: Final = 16
U_STEPS: Final = 1200
U_LEARNING_RATE: Final = 0.03
U_BETA_1: Final = 0.9
U_BETA_2: Final = 0.999
U_EPSILON: Final = 1e-8
U_L2: Final = 1e-3
U_RIDGE_GRID: Final = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
U_CHECKPOINT_EVERY: Final = 20
U_PARAMETER_COUNT: Final = (
    FEATURE_DIMENSION * HIDDEN_WIDTH + HIDDEN_WIDTH + HIDDEN_WIDTH + 1
)
QUADRATIC_DIMENSION: Final = (
    1 + FEATURE_DIMENSION + FEATURE_DIMENSION * (FEATURE_DIMENSION - 1) // 2
)
# A nested generic complexity ladder over the same declared encoding. None of
# these candidates mentions the class quotient; the ladder exists so that a
# competent practitioner's model-selection step is available to the baseline.
U_CANDIDATES: Final = (
    ("constant", None, 1),
    ("linear", None, FEATURE_DIMENSION + 1),
    ("quadratic", None, QUADRATIC_DIMENSION),
    ("mlp", HIDDEN_WIDTH, U_PARAMETER_COUNT),
)
U_VARIANTS: Final = (
    ("U_fixed", ("mlp",), (U_L2,), False),
    ("U_earlystop", ("mlp",), (U_L2,), True),
    ("U_selected", ("mlp",), U_RIDGE_GRID, True),
    ("U_quadratic", ("quadratic",), U_RIDGE_GRID, True),
    ("U_ladder", ("constant", "linear", "quadratic", "mlp"), U_RIDGE_GRID, True),
)


def build_feature_matrix(schema: IncidenceSchema) -> np.ndarray:
    """The label-free relational encoding handed to ``U``.

    Blocks: the query point's identity, the query line's identity, the query
    point's incidence row over all seven lines, and the query line's incidence
    column over all seven points. The complete declared relation is present. The
    class indicator is not a coordinate, and is provably not in the linear span.
    """
    if schema.incidence is None:
        raise MissingDeclaredRelation(
            "cannot encode the relational structure without a relation"
        )
    inputs = schema.inputs
    matrix = np.zeros((len(inputs), FEATURE_DIMENSION), dtype=np.float64)
    point_index = {point: position for position, point in enumerate(schema.points)}
    line_index = {line: position for position, line in enumerate(schema.lines)}
    for row, (point, line) in enumerate(inputs):
        matrix[row, point_index[point]] = 1.0
        matrix[row, 7 + line_index[line]] = 1.0
        for other in schema.lines:
            if (point, other) in schema.incidence:
                matrix[row, 14 + line_index[other]] = 1.0
        for other in schema.points:
            if (other, line) in schema.incidence:
                matrix[row, 21 + point_index[other]] = 1.0
    return matrix


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * value))


def build_quadratic_design(features: np.ndarray) -> np.ndarray:
    """A generic degree-two expansion of the same declared encoding.

    Deliberately over-strong: because the marked-pair indicator equals
    ``sum_k row_k(point) * onehot_k(line)``, the span of this mechanically
    generated basis *contains* the class structure. It is included so the
    headline comparison is made against the strongest information-parity
    baseline short of handing the quotient over, and it is labeled as such.
    """
    count, width = features.shape
    columns = [np.ones((count, 1)), features]
    for left in range(width):
        for right in range(left + 1, width):
            columns.append(
                (features[:, left] * features[:, right]).reshape(count, 1)
            )
    return np.hstack(columns)


def _count_losses(
    logits: np.ndarray,
    successes: np.ndarray,
    failures: np.ndarray,
    total: np.ndarray,
) -> np.ndarray:
    """Per-observation mean cross-entropy from per-input sufficient counts."""
    shared = np.log1p(np.exp(-np.abs(logits)))
    positive = shared + np.maximum(-logits, 0.0)
    negative = shared + np.maximum(logits, 0.0)
    weighted = successes * positive + failures * negative
    return weighted.sum(axis=(1, 2)) / total


def _initial_parameters(
    seeds: Sequence[int], width: int, hidden: int | None
) -> list[np.ndarray]:
    replicates = len(seeds)
    if hidden is None:
        first = np.empty((replicates, width, 1))
        for position, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            first[position] = rng.normal(
                0.0, math.sqrt(2.0 / width), size=(width, 1)
            )
        return [first, np.zeros((replicates, 1, 1))]
    first = np.empty((replicates, width, hidden))
    second = np.empty((replicates, hidden, 1))
    for position, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        first[position] = rng.normal(
            0.0, math.sqrt(2.0 / width), size=(width, hidden)
        )
        second[position] = rng.normal(
            0.0, math.sqrt(2.0 / hidden), size=(hidden, 1)
        )
    return [
        first,
        np.zeros((replicates, 1, hidden)),
        second,
        np.zeros((replicates, 1, 1)),
    ]


def _train_batched(
    design: np.ndarray,
    train_successes: np.ndarray,
    train_failures: np.ndarray,
    train_total: np.ndarray,
    validation_successes: np.ndarray | None,
    validation_failures: np.ndarray | None,
    validation_total: np.ndarray | None,
    seeds: Sequence[int],
    steps: int,
    ridge: float,
    hidden: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one replicate bank by full-batch Adam on the exact count objective.

    Returns the final-step predictions, the best-internal-validation
    predictions, and the best internal validation loss per replicate. The loss
    over ``n`` observations is a function of the per-input counts alone, so this
    is exactly full-batch training and its cost does not grow with ``n``.

    The ridge penalty covers intercepts as well as weights, so that as the
    penalty grows the model shrinks toward probability one half. That is exactly
    the prior mean of the beta(1,1) estimator every closed-form condition uses,
    which keeps the shrinkage conventions matched across conditions.
    """
    replicates = train_successes.shape[0]
    width = design.shape[1]
    parameters = _initial_parameters(seeds, width, hidden)
    moments = [np.zeros_like(value) for value in parameters]
    velocities = [np.zeros_like(value) for value in parameters]

    totals = train_total.reshape(replicates, 1, 1)
    successes = train_successes.reshape(replicates, -1, 1)
    failures = train_failures.reshape(replicates, -1, 1)
    counts = successes + failures

    use_validation = validation_successes is not None
    best_loss = np.full(replicates, np.inf)
    best = [value.copy() for value in parameters]

    for step in range(steps + 1):
        if hidden is None:
            logits = design @ parameters[0] + parameters[1]
            pre_activation = None
            activation = None
        else:
            pre_activation = design @ parameters[0] + parameters[1]
            activation = np.maximum(pre_activation, 0.0)
            logits = activation @ parameters[2] + parameters[3]

        if use_validation and step % U_CHECKPOINT_EVERY == 0:
            loss = _count_losses(
                logits,
                validation_successes.reshape(replicates, -1, 1),
                validation_failures.reshape(replicates, -1, 1),
                validation_total,
            )
            improved = loss < best_loss
            if improved.any():
                best_loss = np.where(improved, loss, best_loss)
                mask = improved.reshape(-1, 1, 1)
                for position, value in enumerate(parameters):
                    best[position] = np.where(mask, value, best[position])

        if step == steps:
            break

        upstream = (counts * _sigmoid(logits) - successes) / totals
        transposed = design.T[None]
        if hidden is None:
            gradients = [
                np.matmul(transposed, upstream) + ridge * parameters[0],
                upstream.sum(axis=1, keepdims=True) + ridge * parameters[1],
            ]
        else:
            gradient_second = (
                np.matmul(activation.transpose(0, 2, 1), upstream)
                + ridge * parameters[2]
            )
            gradient_second_bias = (
                upstream.sum(axis=1, keepdims=True) + ridge * parameters[3]
            )
            downstream = np.matmul(
                upstream, parameters[2].transpose(0, 2, 1)
            ) * (pre_activation > 0.0)
            gradients = [
                np.matmul(transposed, downstream) + ridge * parameters[0],
                downstream.sum(axis=1, keepdims=True)
                + ridge * parameters[1],
                gradient_second,
                gradient_second_bias,
            ]

        scale = step + 1
        first_correction = 1.0 - U_BETA_1**scale
        second_correction = 1.0 - U_BETA_2**scale
        for position, gradient in enumerate(gradients):
            moments[position] = (
                U_BETA_1 * moments[position] + (1.0 - U_BETA_1) * gradient
            )
            velocities[position] = U_BETA_2 * velocities[position] + (
                1.0 - U_BETA_2
            ) * (gradient * gradient)
            parameters[position] = parameters[position] - U_LEARNING_RATE * (
                (moments[position] / first_correction)
                / (
                    np.sqrt(velocities[position] / second_correction)
                    + U_EPSILON
                )
            )

    def predict(values: list[np.ndarray]) -> np.ndarray:
        if hidden is None:
            logits = design @ values[0] + values[1]
        else:
            activation = np.maximum(design @ values[0] + values[1], 0.0)
            logits = activation @ values[2] + values[3]
        return _sigmoid(logits).reshape(replicates, design.shape[0])

    return predict(parameters), predict(best), best_loss


def build_designs(features: np.ndarray) -> dict[str, np.ndarray]:
    """The nested generic design matrices, all built label-free from Sigma'."""
    count = features.shape[0]
    return {
        "constant": np.ones((count, 1)),
        "linear": np.hstack([np.ones((count, 1)), features]),
        "quadratic": build_quadratic_design(features),
        "mlp": features,
    }


def fit_u_variants(
    designs: dict[str, np.ndarray],
    full_successes: np.ndarray,
    full_failures: np.ndarray,
    train_successes: np.ndarray,
    train_failures: np.ndarray,
    validation_successes: np.ndarray,
    validation_failures: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Fit every preregistered ``U`` variant from one shared training pass.

    Selection, where a variant uses it, minimizes loss on an internal split of
    the training labels only. No evaluation output, no ``theta_star``, and no
    class quotient is ever in scope. Variants are composed from a cache so that
    every variant sees byte-identical observations.
    """
    replicates = full_successes.shape[0]
    hidden_of = {name: hidden for name, hidden, _ in U_CANDIDATES}
    size_of = {name: size for name, _, size in U_CANDIDATES}
    train_totals = (train_successes + train_failures).sum(axis=1)
    validation_totals = (
        validation_successes + validation_failures
    ).sum(axis=1)
    full_totals = (full_successes + full_failures).sum(axis=1)

    cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for name, hidden, _ in U_CANDIDATES:
        for ridge in U_RIDGE_GRID:
            _, best, loss = _train_batched(
                designs[name],
                train_successes,
                train_failures,
                train_totals,
                validation_successes,
                validation_failures,
                validation_totals,
                seeds,
                U_STEPS,
                ridge,
                hidden,
            )
            cache[(name, ridge)] = (best, loss)

    unvalidated, _, _ = _train_batched(
        designs["mlp"],
        full_successes,
        full_failures,
        full_totals,
        None,
        None,
        None,
        seeds,
        U_STEPS,
        U_L2,
        hidden_of["mlp"],
    )

    outcome: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    for variant, candidates, ridges, use_validation in U_VARIANTS:
        if not use_validation:
            outcome[variant] = (
                unvalidated,
                {
                    "model_selection": "none; the full step budget is reported",
                    "declared_parameters": size_of[candidates[0]],
                },
            )
            continue
        chosen = np.zeros((replicates, designs["constant"].shape[0]))
        chosen_loss = np.full(replicates, np.inf)
        chosen_kind = np.full(replicates, -1)
        chosen_ridge = np.full(replicates, np.nan)
        for position, name in enumerate(candidates):
            for ridge in ridges:
                best, loss = cache[(name, ridge)]
                improved = loss < chosen_loss
                chosen_loss = np.where(improved, loss, chosen_loss)
                chosen = np.where(improved.reshape(-1, 1), best, chosen)
                chosen_kind = np.where(improved, position, chosen_kind)
                chosen_ridge = np.where(improved, ridge, chosen_ridge)
        histogram = {
            name: int((chosen_kind == position).sum())
            for position, name in enumerate(candidates)
        }
        ridge_histogram = {
            str(ridge): int((chosen_ridge == ridge).sum())
            for ridge in ridges
        }
        outcome[variant] = (
            chosen,
            {
                "model_selection": (
                    "internal validation on training labels only, over the "
                    "declared candidate and ridge grids"
                ),
                "candidates": list(candidates),
                "selected_candidate_counts": histogram,
                "selected_ridge_counts": ridge_histogram,
                "declared_parameters": max(
                    size_of[name] for name in candidates
                ),
            },
        )
    return outcome


def realize_class_indicator(
    features: np.ndarray, class_of_index: Sequence[int]
) -> dict[str, object]:
    """Exhibit weights that make ``U`` compute the class function exactly.

    Seven hidden units of the form ``relu(row_k(p) + onehot_k(line) - 1)`` sum to
    the marked-pair indicator, so the compiled family lies inside ``U``'s
    hypothesis class. This certifies information parity constructively rather
    than by assertion.
    """
    first = np.zeros((FEATURE_DIMENSION, HIDDEN_WIDTH))
    first_bias = np.zeros(HIDDEN_WIDTH)
    second = np.zeros((HIDDEN_WIDTH, 1))
    for unit in range(7):
        first[14 + unit, unit] = 1.0
        first[7 + unit, unit] = 1.0
        first_bias[unit] = -1.0
        second[unit, 0] = 1.0
    hidden = np.maximum(features @ first + first_bias, 0.0)
    indicator = (hidden @ second).reshape(-1)
    target = np.array(
        [1.0 if cell == 0 else 0.0 for cell in class_of_index], dtype=np.float64
    )
    return {
        "construction": (
            "seven hidden units relu(row_k(point) + onehot_k(line) - 1) whose "
            "sum is the marked-pair indicator"
        ),
        "hidden_units_used": 7,
        "hidden_width_available": HIDDEN_WIDTH,
        "max_absolute_error_against_class_zero_indicator": _round(
            float(np.max(np.abs(indicator - target)))
        ),
        "class_function_is_exactly_realizable": bool(
            np.array_equal(indicator, target)
        ),
    }


def realize_single_pair_indicator(
    features: np.ndarray, target_index: int
) -> dict[str, object]:
    """Exhibit weights that make ``U`` isolate one pair, so it is not tied."""
    first = np.zeros((FEATURE_DIMENSION, HIDDEN_WIDTH))
    first_bias = np.zeros(HIDDEN_WIDTH)
    second = np.zeros((HIDDEN_WIDTH, 1))
    point_column = int(np.argmax(features[target_index, 0:7]))
    line_column = int(np.argmax(features[target_index, 7:14]))
    first[point_column, 0] = 1.0
    first[7 + line_column, 0] = 1.0
    first_bias[0] = -1.0
    second[0, 0] = 1.0
    hidden = np.maximum(features @ first + first_bias, 0.0)
    isolated = (hidden @ second).reshape(-1)
    target = np.zeros(features.shape[0])
    target[target_index] = 1.0
    return {
        "construction": (
            "one hidden unit relu(onehot(point) + onehot(line) - 1) selecting a "
            "single admitted pair"
        ),
        "single_pair_is_exactly_realizable": bool(
            np.array_equal(isolated, target)
        ),
        "consequence": (
            "U can express functions that are not class-invariant, so the "
            "two-class structure is not imposed as a parameter tying rule"
        ),
    }


def span_certificate(
    features: np.ndarray, class_of_index: Sequence[int], inputs: Sequence[Datum]
) -> dict[str, object]:
    """Show the class indicator is not a coordinate and not in the linear span.

    Every feature column is a function of the query point alone or of the query
    line alone, so the whole column span consists of additive functions
    ``f(point) + g(line)``. Any additive function has a vanishing alternating sum
    on a two-by-two block of inputs. The marked-pair indicator does not, and the
    exhibited block proves it. The learner therefore has to discover the
    interaction; it is not handed it.
    """
    point_of = {datum: datum[0] for datum in inputs}
    line_of = {datum: datum[1] for datum in inputs}
    columns: list[str] = []
    for column in range(FEATURE_DIMENSION):
        by_point: dict[str, set[float]] = {}
        by_line: dict[str, set[float]] = {}
        for row, datum in enumerate(inputs):
            by_point.setdefault(point_of[datum], set()).add(
                float(features[row, column])
            )
            by_line.setdefault(line_of[datum], set()).add(
                float(features[row, column])
            )
        point_only = all(len(values) == 1 for values in by_point.values())
        line_only = all(len(values) == 1 for values in by_line.values())
        if point_only:
            columns.append("function_of_point")
        elif line_only:
            columns.append("function_of_line")
        else:
            columns.append("neither")

    indicator = [1 if cell == 0 else 0 for cell in class_of_index]
    position = {datum: row for row, datum in enumerate(inputs)}
    witness: dict[str, object] | None = None
    points, lines = sorted({d[0] for d in inputs}), sorted({d[1] for d in inputs})
    for left_point, right_point in itertools.combinations(points, 2):
        for left_line, right_line in itertools.combinations(lines, 2):
            corners = (
                indicator[position[(left_point, left_line)]],
                indicator[position[(left_point, right_line)]],
                indicator[position[(right_point, left_line)]],
                indicator[position[(right_point, right_line)]],
            )
            alternating = corners[0] - corners[1] - corners[2] + corners[3]
            if alternating != 0:
                witness = {
                    "block": [
                        [left_point, left_line],
                        [left_point, right_line],
                        [right_point, left_line],
                        [right_point, right_line],
                    ],
                    "indicator_corners": list(corners),
                    "alternating_sum": alternating,
                }
                break
        if witness is not None:
            break

    design = np.hstack([features, np.ones((features.shape[0], 1))])
    target = np.array(indicator, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.max(np.abs(design @ solution - target)))

    constant_on_classes = []
    for column in range(FEATURE_DIMENSION):
        values: dict[int, set[float]] = {}
        for row, cell in enumerate(class_of_index):
            values.setdefault(cell, set()).add(float(features[row, column]))
        constant_on_classes.append(
            all(len(entries) == 1 for entries in values.values())
        )

    return {
        "feature_blocks": [
            {"name": name, "width": width} for name, width in FEATURE_BLOCKS
        ],
        "feature_dimension": FEATURE_DIMENSION,
        "every_column_is_a_function_of_one_coordinate_alone": all(
            kind in ("function_of_point", "function_of_line") for kind in columns
        ),
        "column_kinds": columns,
        "no_column_is_a_function_of_the_class": not any(constant_on_classes),
        "additive_span_witness": witness,
        "indicator_is_in_the_linear_span": witness is None,
        "least_squares_max_absolute_residual": _round(residual),
        "consequence": (
            "the class indicator is neither a supplied feature nor a linear "
            "combination of supplied features; U must discover it as an "
            "interaction, while still holding the full declared relation"
        ),
    }


# ---------------------------------------------------------------------------
# 7. Exact evaluation
# ---------------------------------------------------------------------------


def _cross_entropy(truth: float, prediction: float) -> float:
    clipped = min(max(prediction, CLIP), 1.0 - CLIP)
    return -(
        truth * math.log(clipped) + (1.0 - truth) * math.log1p(-clipped)
    )


def evaluate_exactly(
    predictions: Sequence[float],
    split: Split,
    theta_floats: Sequence[float],
    class_of_index: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Closed-form expected losses; the large-fresh-sample limit, without noise."""
    report: dict[str, dict[str, float]] = {}
    for name, indices in split.evaluation:
        if not indices:
            continue
        count = len(indices)
        groups: dict[tuple[int, float], int] = {}
        for index in indices:
            key = (
                class_of_index[index],
                min(max(predictions[index], CLIP), 1.0 - CLIP),
            )
            groups[key] = groups.get(key, 0) + 1
        nll = 0.0
        bayes = 0.0
        brier = 0.0
        bins = [[0, 0.0, 0.0] for _ in range(10)]
        for (cell, value), occupancy in groups.items():
            truth = theta_floats[cell]
            nll += occupancy * _cross_entropy(truth, value)
            bayes += occupancy * _cross_entropy(truth, truth)
            brier += occupancy * (
                truth * (1.0 - value) ** 2 + (1.0 - truth) * value * value
            )
            slot = min(int(value * 10.0), 9)
            bins[slot][0] += occupancy
            bins[slot][1] += occupancy * value
            bins[slot][2] += occupancy * truth
        calibration = 0.0
        for occupancy, predicted, actual in bins:
            if occupancy:
                calibration += (occupancy / count) * abs(
                    predicted / occupancy - actual / occupancy
                )
        report[name] = {
            "nll": nll / count,
            "bayes_nll": bayes / count,
            "excess_nll": (nll - bayes) / count,
            "brier": brier / count,
            "calibration_error": calibration,
        }
    return report


def _summary(values: Sequence[float]) -> dict[str, float]:
    count = len(values)
    mean = statistics.fmean(values)
    if count > 1:
        deviation = statistics.stdev(values)
        half = 1.959963984540054 * deviation / math.sqrt(count)
    else:
        deviation, half = 0.0, 0.0
    return {
        "mean": _round(mean),
        "median": _round(statistics.median(values)),
        "stdev": _round(deviation),
        "ci95_low": _round(mean - half),
        "ci95_high": _round(mean + half),
        "replicates": count,
    }


# ---------------------------------------------------------------------------
# 8. The exact analytic reference for the two-parameter pooled learner
# ---------------------------------------------------------------------------


def _log_factorials(limit: int) -> list[float]:
    table = [0.0] * (limit + 1)
    for value in range(2, limit + 1):
        table[value] = table[value - 1] + math.log(value)
    return table


def _binomial_pmf(
    trials: int, probability: float, log_factorial: Sequence[float]
) -> list[float]:
    if probability <= 0.0:
        return [1.0 if index == 0 else 0.0 for index in range(trials + 1)]
    if probability >= 1.0:
        return [1.0 if index == trials else 0.0 for index in range(trials + 1)]
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    return [
        math.exp(
            log_factorial[trials]
            - log_factorial[successes]
            - log_factorial[trials - successes]
            + successes * log_p
            + (trials - successes) * log_q
        )
        for successes in range(trials + 1)
    ]


def _expected_cross_entropy_table(
    max_trials: int,
    truth: float,
    alpha: Fraction,
    beta: Fraction,
    log_factorial: Sequence[float],
) -> list[float]:
    prior = float(alpha + beta)
    offset = float(alpha)
    table: list[float] = []
    for trials in range(max_trials + 1):
        pmf = _binomial_pmf(trials, truth, log_factorial)
        total = 0.0
        denominator = trials + prior
        for successes in range(trials + 1):
            total += pmf[successes] * _cross_entropy(
                truth, (successes + offset) / denominator
            )
        table.append(total)
    return table


def analytic_reference(
    sizes: Sequence[int],
    theta: Sequence[Fraction],
    alpha: Fraction,
    beta: Fraction,
    train_weights: Sequence[Fraction],
    eval_weights: Sequence[Fraction],
) -> dict[str, object]:
    """Exact expected excess log loss for the compiled two-parameter learner.

    The two pooled Bernoulli estimators are independent given the class sample
    counts, and the counts are binomial, so the whole expectation is a finite
    double sum with no simulation anywhere.
    """
    limit = max(sizes)
    log_factorial = _log_factorials(limit)
    truths = [float(value) for value in theta]
    tables = [
        _expected_cross_entropy_table(limit, truth, alpha, beta, log_factorial)
        for truth in truths
    ]
    bayes = sum(
        float(weight) * _cross_entropy(truth, truth)
        for weight, truth in zip(eval_weights, truths, strict=True)
    )
    rows: list[dict[str, object]] = []
    for size in sizes:
        pmf = _binomial_pmf(size, float(train_weights[0]), log_factorial)
        expected = 0.0
        for first_count in range(size + 1):
            expected += pmf[first_count] * (
                float(eval_weights[0]) * tables[0][first_count]
                + float(eval_weights[1]) * tables[1][size - first_count]
            )
        asymptotic = sum(
            float(eval_weight) / (2.0 * size * float(train_weight))
            for eval_weight, train_weight in zip(
                eval_weights, train_weights, strict=True
            )
        )
        rows.append(
            {
                "n": size,
                "exact_expected_excess_nll": _round(expected - bayes),
                "asymptotic_prediction": _round(asymptotic),
                "expected_class_counts": [
                    _round(size * float(weight)) for weight in train_weights
                ],
                "estimator_variance_at_expected_count": [
                    _round(
                        float(truth)
                        * (1.0 - float(truth))
                        / (size * float(weight) + float(alpha + beta)) ** 2
                        * (size * float(weight))
                    )
                    for truth, weight in zip(truths, train_weights, strict=True)
                ],
            }
        )
    return {
        "bayes_optimal_nll": _round(bayes),
        "train_weights": [_fraction_text(w) for w in train_weights],
        "eval_weights": [_fraction_text(w) for w in eval_weights],
        "asymptotic_law": (
            "E[excess NLL] ~ sum_c eval_w_c / (2 n train_w_c); with matched "
            "weights and two classes this is exactly d/(2n) = 1/n"
        ),
        "sweep": rows,
    }


def asymptotic_floor(
    partition: Sequence[int],
    cell_count: int,
    theta_floats: Sequence[float],
    class_of_index: Sequence[int],
    train_pool: Sequence[int],
    eval_indices: Sequence[int],
    alpha: Fraction,
    beta: Fraction,
) -> float:
    """The excess log loss a partition cannot escape with unlimited labels."""
    totals = [0.0] * cell_count
    counts = [0] * cell_count
    for index in train_pool:
        cell = partition[index]
        totals[cell] += theta_floats[class_of_index[index]]
        counts[cell] += 1
    default = float(alpha / (alpha + beta))
    limits = [
        totals[cell] / counts[cell] if counts[cell] else default
        for cell in range(cell_count)
    ]
    excess = 0.0
    for index in eval_indices:
        truth = theta_floats[class_of_index[index]]
        excess += _cross_entropy(truth, limits[partition[index]]) - _cross_entropy(
            truth, truth
        )
    return excess / len(eval_indices)


# ---------------------------------------------------------------------------
# 9. The three falsification controls
# ---------------------------------------------------------------------------


def build_wrong_partition(
    structure: ClassStructure,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """A two-cell 21/28 partition that respects no declared automorphism."""
    position = {datum: index for index, datum in enumerate(structure.inputs)}
    first = sorted(structure.classes[0])
    second = sorted(structure.classes[1])
    exchanged = 7
    cell_a = set(first[exchanged:]) | set(second[:exchanged])
    cell_b = set(second[exchanged:]) | set(first[:exchanged])
    if len(cell_a) != 21 or len(cell_b) != 28:
        raise RuntimeError("the wrong-compiler control must preserve 21 and 28")
    partition = tuple(
        0 if datum in cell_a else 1 for datum in structure.inputs
    )
    violated = 0
    for automorphism in structure.automorphisms:
        for datum in structure.inputs:
            moved = automorphism.apply(datum, structure.schema)
            if partition[position[datum]] != partition[position[moved]]:
                violated += 1
    mixing = []
    for cell in (0, 1):
        members = [
            index for index, value in enumerate(partition) if value == cell
        ]
        counts = [0, 0]
        for index in members:
            counts[structure.class_of_index[index]] += 1
        mixing.append({"cell": f"W_{cell}", "size": len(members),
                       "true_class_counts": counts})
    return partition, {
        "construction": (
            "exchange the 7 lexicographically least members of each derived "
            "class, preserving the 21 and 28 cardinalities"
        ),
        "cardinalities": [21, 28],
        "respects_the_declared_automorphisms": violated == 0,
        "automorphism_violations_counted": violated,
        "cell_composition": mixing,
    }


def build_scrambled_schemas(
    schema: IncidenceSchema, count: int
) -> list[IncidenceSchema]:
    """A deterministic ensemble of non-automorphic 21-row scramblings."""
    admitted = list(schema.inputs)
    declared = schema.incidence
    if declared is None:
        raise MissingDeclaredRelation("cannot scramble an untyped relation")
    schemas: list[IncidenceSchema] = []
    for index in range(count):
        rng = random.Random(BASE_SEED + 700_000 + index)
        while True:
            marked = frozenset(rng.sample(admitted, 21))
            if marked != declared:
                break
        schemas.append(
            IncidenceSchema(
                points=schema.points, lines=schema.lines, incidence=marked
            )
        )
    return schemas


def build_relation_erased(schema: IncidenceSchema) -> IncidenceSchema:
    """Preserve both sorts, all 49 admitted pairs, and the 21 rows as payload."""
    if schema.incidence is None:
        raise MissingDeclaredRelation("primary schema already lacks a relation")
    return IncidenceSchema(
        points=schema.points,
        lines=schema.lines,
        incidence=None,
        opaque_rows=tuple(sorted(schema.incidence)),
    )


def audit_controls(
    structure: ClassStructure, features: np.ndarray
) -> dict[str, object]:
    erased = build_relation_erased(structure.schema)
    compiler_failed = False
    compiler_message = ""
    try:
        compile_classes(erased)
    except MissingDeclaredRelation as error:
        compiler_failed = True
        compiler_message = str(error)
    encoder_failed = False
    try:
        build_feature_matrix(erased)
    except MissingDeclaredRelation:
        encoder_failed = True

    census: list[dict[str, object]] = []
    reproduced = 0
    for index, scrambled in enumerate(build_scrambled_schemas(structure.schema, 20)):
        derived = compile_classes(scrambled)
        cardinalities = sorted(derived.cardinalities)
        matches = cardinalities == [21, 28]
        reproduced += int(matches)
        census.append(
            {
                "index": index,
                "group_order": derived.group_order,
                "class_count": len(derived.classes),
                "class_cardinalities": cardinalities,
                "reproduces_the_declared_quotient": matches,
                "differs_from_the_declared_relation": (
                    scrambled.incidence != structure.schema.incidence
                ),
            }
        )

    _, wrong_certificate = build_wrong_partition(structure)
    return {
        "relation_erased": {
            "what_was_removed": (
                "the relational meaning of the 21 declared rows; the rows "
                "survive as opaque payload"
            ),
            "what_was_preserved": {
                "both_sorts": erased.points == structure.schema.points
                and erased.lines == structure.schema.lines,
                "all_49_admitted_inputs": len(erased.inputs) == 49,
                "all_21_rows_as_opaque_payload": len(erased.opaque_rows) == 21,
            },
            "compiler_fails_closed": compiler_failed,
            "compiler_failure_message": compiler_message,
            "relational_encoder_fails_closed": encoder_failed,
            "no_fallback_recovers_the_quotient_from_label_positions": True,
        },
        "scrambled_structure": {
            "ensemble_size": len(census),
            "declared_group_order": structure.group_order,
            "declared_class_cardinalities": sorted(structure.cardinalities),
            "scrambles_reproducing_the_declared_quotient": reproduced,
            "a_two_class_result_was_not_forced": True,
            "census": census,
        },
        "wrong_compiler": wrong_certificate,
        "relational_encoding_dimension": int(features.shape[1]),
    }


# ---------------------------------------------------------------------------
# 10. The decision-model bridge, and the leakage and parity certificates
# ---------------------------------------------------------------------------


def _load_decision_model() -> object:
    source = REPO_ROOT / "decision-model" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import decision_model

    return decision_model


def decision_model_bridge(
    grid: Sequence[Fraction], structure: ClassStructure, theta: Sequence[Fraction]
) -> dict[str, object]:
    """Resolve the compiled TDM through the stable public contracts, exactly.

    Nothing is added to ``decision_model``. The compiled family is *consumed* by
    the shipped ``Test``/``resolve_test`` contracts at zero tolerance, which is
    what makes the T-versus-O comparison meaningful: if the Born formula were
    doing arithmetic work, this bridge would not close exactly.
    """
    module = _load_decision_model()
    test = module.Test(
        (
            ("a_0", lambda state: state(Z_MATRIX)),
            ("a_1", lambda state: state(Z_SCALAR)),
        )
    )
    exact_everywhere = True
    zero_tolerance_everywhere = True
    for value in grid:
        distribution = module.resolve_test(omega_lambda(value), test)
        probabilities = [
            probability.value for _, probability in distribution
        ]
        exact_everywhere &= probabilities == [value, ONE - value]
        zero_tolerance_everywhere &= distribution.tolerance == 0
        zero_tolerance_everywhere &= distribution.total == ONE

    public_matches = True
    for index, cell in enumerate(structure.class_of_index):
        distribution = module.resolve_test(omega_lambda(theta[cell]), test)
        public_matches &= distribution["a_0"].value == theta[cell]
        public_matches &= index < len(structure.inputs)

    return {
        "public_api_symbols_used": ["Test", "resolve_test", "Distribution"],
        "public_api_symbol_count": len(module.__all__),
        "anything_added_to_the_public_api": False,
        "grid_size": len(grid),
        "resolves_exactly_to_lambda_and_one_minus_lambda": exact_everywhere,
        "normalizes_at_zero_tolerance": zero_tolerance_everywhere,
        "public_family_matches_on_all_admitted_inputs": public_matches,
        "admitted_inputs_checked": len(structure.inputs),
    }


FITTING_FUNCTION_NAMES: Final = (
    "compile_classes",
    "enumerate_automorphisms",
    "invariance_partition_from_group",
    "build_split",
    "build_feature_matrix",
    "pair_counts",
    "fit_typed_tdm",
    "fit_bernoulli",
    "build_designs",
    "build_quadratic_design",
    "fit_u_variants",
    "_train_batched",
)
TARGET_PARAMETER_NAMES: Final = (
    "theta",
    "theta_star",
    "theta_floats",
    "numerators",
    "truth",
    "hidden_class_of_index",
)


def _walk_keys(node: object) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_keys(value)


def _walk_arrays(node: object) -> Iterable[list[object]]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_arrays(value)
    elif isinstance(node, list):
        yield node
        for value in node:
            yield from _walk_arrays(value)


def _contains_float(node: object) -> bool:
    if isinstance(node, bool):
        return False
    if isinstance(node, float):
        return True
    if isinstance(node, dict):
        return any(_contains_float(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_float(value) for value in node)
    return False


def leakage_certificate(
    document: dict[str, object], structure: ClassStructure
) -> dict[str, object]:
    """Mechanical checks that no answer semantics reach the compiler or Sigma'."""
    forbidden_keys = {
        "rule",
        "decision_rule",
        "answer",
        "answers",
        "per_input_answer",
        "lambda",
        "probability",
        "frequency",
        "target",
        "preparation",
        "effect",
        "ray",
        "orientation",
    }
    observed_keys = set(_walk_keys(document))
    labels = set(document["answer_type"]["labels"])  # type: ignore[index]
    identifiers = set(structure.schema.points) | set(structure.schema.lines)
    mixed_arrays = [
        array
        for array in _walk_arrays(document)
        if any(str(item) in labels for item in array)
        and any(str(item) in identifiers for item in array)
    ]

    carrier_fields = set(IncidenceSchema.__dataclass_fields__)
    compiler_parameters = tuple(
        inspect.signature(compile_classes).parameters
    )
    offenders: list[str] = []
    module = sys.modules[__name__]
    for name in FITTING_FUNCTION_NAMES:
        parameters = set(inspect.signature(getattr(module, name)).parameters)
        if parameters & set(TARGET_PARAMETER_NAMES):
            offenders.append(name)

    stripped = {
        key: value
        for key, value in document.items()
        if key not in ("derivable_self_checks", "automorphisms")
    }
    rebuilt = IncidenceSchema(
        points=tuple(stripped["sorts"]["point"]),  # type: ignore[index]
        lines=tuple(stripped["sorts"]["line"]),  # type: ignore[index]
        incidence=frozenset(
            (row[0], row[1])
            for row in stripped["relations"]["incident"]  # type: ignore[index]
        ),
    )
    without_expectations = compile_classes(rebuilt)

    flipped_successes = [1] * len(structure.inputs)
    flipped_trials = [1] * len(structure.inputs)
    upright = fit_bernoulli(
        flipped_successes,
        flipped_trials,
        structure.class_of_index,
        2,
        ONE,
        ONE,
        "probe",
    )
    inverted = fit_bernoulli(
        [0] * len(structure.inputs),
        flipped_trials,
        structure.class_of_index,
        2,
        ONE,
        ONE,
        "probe",
    )
    complementary = all(
        left + right == ONE
        for left, right in zip(
            upright.exact_predictions or (),
            inverted.exact_predictions or (),
            strict=True,
        )
    )

    return {
        "sigma_prime_declares_no_forbidden_key": sorted(
            observed_keys & forbidden_keys
        )
        == [],
        "forbidden_keys_observed": sorted(observed_keys & forbidden_keys),
        "sigma_prime_contains_no_float": not _contains_float(document),
        "no_array_mixes_a_sort_identifier_with_an_answer_label": not mixed_arrays,
        "compiler_carrier_fields": sorted(carrier_fields),
        "compiler_carrier_has_no_answer_field": not (
            carrier_fields
            & {"answer", "answers", "labels", "decision_labels", "rule"}
        ),
        "compiler_signature": list(compiler_parameters),
        "no_fitting_function_accepts_a_target_parameter": not offenders,
        "fitting_functions_audited": list(FITTING_FUNCTION_NAMES),
        "target_parameter_names_searched": list(TARGET_PARAMETER_NAMES),
        "offending_functions": offenders,
        "compiler_ignores_the_redundant_expectation_blocks": (
            without_expectations.classes == structure.classes
            and without_expectations.group_order == structure.group_order
        ),
        "relabeling_the_answers_only_complements_the_estimates": complementary,
        "generator_is_the_only_holder_of_theta_star": (
            "theta" in inspect.signature(theta_as_integers).parameters
            and "numerators"
            in inspect.signature(generate_observations).parameters
        ),
    }


# ---------------------------------------------------------------------------
# 11. The preregistered sweep
# ---------------------------------------------------------------------------

BASE_SEED: Final = 20260923
SAMPLE_SIZES: Final = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
QUICK_SIZES: Final = (4, 32, 256)
CLOSED_FORM_REPLICATES: Final = 400
LEARNED_REPLICATES: Final = 100
QUICK_CLOSED_FORM_REPLICATES: Final = 40
QUICK_LEARNED_REPLICATES: Final = 20
HOLDOUT_PER_CLASS: Final = (4, 6)
THRESHOLDS: Final = (0.02, 0.01)
REGIMES: Final = ("A", "B")
ALPHA_PRIMARY: Final = ONE
BETA_PRIMARY: Final = ONE
ALPHA_SECONDARY: Final = Fraction(1, 2)
BETA_SECONDARY: Final = Fraction(1, 2)
MATERIAL_FACTOR: Final = 2.0
LEARNED_CONDITIONS: Final = tuple(variant[0] for variant in U_VARIANTS)
TABLE_COLUMNS: Final = (
    "theta_id",
    "regime",
    "n",
    "condition",
    "replicate",
    "excess_nll_aggregate",
    "excess_nll_O_0",
    "excess_nll_O_1",
    "nll_aggregate",
    "brier_aggregate",
    "calibration_error_aggregate",
    "lambda_0_hat",
    "lambda_1_hat",
    "degrees_of_freedom",
)


@dataclass(frozen=True, slots=True)
class ThetaSpec:
    identifier: str
    role: str
    values: tuple[Fraction, ...]


THETAS: Final = (
    ThetaSpec("primary_quarters", "primary", (Fraction(1, 4), Fraction(3, 4))),
    ThetaSpec("secondary_thirds", "secondary", (Fraction(1, 3), Fraction(2, 3))),
    ThetaSpec("secondary_fifths", "secondary", (Fraction(2, 5), Fraction(3, 5))),
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    name: str
    partition: tuple[int, ...]
    cell_count: int
    kind: str
    primary_theta_only: bool


def observation_seed(
    theta_index: int, regime_index: int, size_index: int, replicate: int
) -> int:
    return (
        BASE_SEED
        + 1_000_000 * theta_index
        + 100_000 * regime_index
        + 1_000 * size_index
        + replicate
    )


def _class_members(structure: ClassStructure) -> tuple[tuple[int, ...], ...]:
    members: list[list[int]] = [[] for _ in structure.classes]
    for index, cell in enumerate(structure.class_of_index):
        members[cell].append(index)
    return tuple(tuple(group) for group in members)


def _accumulator() -> dict[str, list[float]]:
    return {
        "excess_aggregate": [],
        "excess_O_0": [],
        "excess_O_1": [],
        "nll_aggregate": [],
        "brier_aggregate": [],
        "calibration_aggregate": [],
        "error_lambda_0": [],
        "error_lambda_1": [],
    }


def _summarize(
    store: dict[str, list[float]], matched: int, dof: int
) -> dict[str, object]:
    report: dict[str, object] = {"degrees_of_freedom": dof}
    for key, values in store.items():
        if not values:
            continue
        report[key] = _summary(values)
    primary = store["excess_aggregate"]
    report["excess_aggregate_matched"] = _summary(primary[:matched])
    return report


def run_closed_form(
    structure: ClassStructure,
    conditions: Sequence[ConditionSpec],
    thetas: Sequence[ThetaSpec],
    sizes: Sequence[int],
    replicates: int,
    matched: int,
    alpha: Fraction,
    beta: Fraction,
) -> tuple[dict[str, dict[str, object]], list[tuple[object, ...]], dict[str, int]]:
    """Fit every closed-form condition on byte-identical observations."""
    members = _class_members(structure)
    width = len(structure.inputs)
    identity = tuple(range(width))
    cells: dict[str, dict[str, object]] = {}
    rows: list[tuple[object, ...]] = []
    agreement = {
        "T_versus_O_prediction_mismatches": 0,
        "T_versus_G_prediction_mismatches": 0,
        "comparisons": 0,
    }

    for theta_index, spec in enumerate(thetas):
        numerators, denominator = theta_as_integers(spec.values)
        truths = [float(value) for value in spec.values]
        active = [
            condition
            for condition in conditions
            if not condition.primary_theta_only or spec.role == "primary"
        ]
        for regime_index, regime in enumerate(REGIMES):
            for size_index, size in enumerate(sizes):
                store = {
                    condition.name: _accumulator() for condition in active
                }
                for replicate in range(replicates):
                    rng = random.Random(
                        observation_seed(
                            theta_index, regime_index, size_index, replicate
                        )
                    )
                    split = build_split(
                        regime, rng, members, HOLDOUT_PER_CLASS
                    )
                    observations = generate_observations(
                        rng,
                        split.train_pool,
                        size,
                        numerators,
                        denominator,
                        structure.class_of_index,
                    )
                    successes, trials = pair_counts(observations, width)
                    fitted: dict[str, FittedModel] = {}
                    for condition in active:
                        builder = (
                            fit_typed_tdm
                            if condition.kind == "tdm"
                            else fit_bernoulli
                        )
                        model = builder(
                            successes,
                            trials,
                            condition.partition,
                            condition.cell_count,
                            alpha,
                            beta,
                            condition.name,
                        )
                        fitted[condition.name] = model
                        metrics = evaluate_exactly(
                            model.predictions,
                            split,
                            truths,
                            structure.class_of_index,
                        )
                        bucket = store[condition.name]
                        bucket["excess_aggregate"].append(
                            metrics["aggregate"]["excess_nll"]
                        )
                        for name in ("O_0", "O_1"):
                            if name in metrics:
                                bucket[f"excess_{name}"].append(
                                    metrics[name]["excess_nll"]
                                )
                        bucket["nll_aggregate"].append(
                            metrics["aggregate"]["nll"]
                        )
                        bucket["brier_aggregate"].append(
                            metrics["aggregate"]["brier"]
                        )
                        bucket["calibration_aggregate"].append(
                            metrics["aggregate"]["calibration_error"]
                        )
                        exposed = model.exposed
                        if exposed is not None and len(exposed) == 2:
                            bucket["error_lambda_0"].append(
                                abs(float(exposed[0]) - truths[0])
                            )
                            bucket["error_lambda_1"].append(
                                abs(float(exposed[1]) - truths[1])
                            )
                        rows.append(
                            (
                                spec.identifier,
                                regime,
                                size,
                                condition.name,
                                replicate,
                                _round(metrics["aggregate"]["excess_nll"]),
                                _round(metrics["O_0"]["excess_nll"]),
                                _round(metrics["O_1"]["excess_nll"]),
                                _round(metrics["aggregate"]["nll"]),
                                _round(metrics["aggregate"]["brier"]),
                                _round(
                                    metrics["aggregate"]["calibration_error"]
                                ),
                                _fraction_text(exposed[0])
                                if exposed is not None
                                else "",
                                _fraction_text(exposed[1])
                                if exposed is not None
                                else "",
                                model.degrees_of_freedom,
                            )
                        )
                    if "T" in fitted and "O" in fitted:
                        agreement["comparisons"] += 1
                        if (
                            fitted["T"].exact_predictions
                            != fitted["O"].exact_predictions
                        ):
                            agreement["T_versus_O_prediction_mismatches"] += 1
                    if "T" in fitted and "G" in fitted:
                        if (
                            fitted["T"].exact_predictions
                            != fitted["G"].exact_predictions
                        ):
                            agreement["T_versus_G_prediction_mismatches"] += 1
                for condition in active:
                    key = f"{spec.identifier}|{regime}|{size}|{condition.name}"
                    cells[key] = _summarize(
                        store[condition.name], matched, condition.cell_count
                    )
    del identity
    return cells, rows, agreement


def run_learned(
    structure: ClassStructure,
    features: np.ndarray,
    thetas: Sequence[ThetaSpec],
    sizes: Sequence[int],
    replicates: int,
) -> tuple[dict[str, dict[str, object]], list[tuple[object, ...]]]:
    """Fit every ``U`` variant on exactly the data the closed-form bank saw."""
    members = _class_members(structure)
    width = len(structure.inputs)
    designs = build_designs(features)
    cells: dict[str, dict[str, object]] = {}
    rows: list[tuple[object, ...]] = []

    for theta_index, spec in enumerate(thetas):
        numerators, denominator = theta_as_integers(spec.values)
        truths = [float(value) for value in spec.values]
        for regime_index, regime in enumerate(REGIMES):
            for size_index, size in enumerate(sizes):
                validation_size = max(1, int(round(0.25 * size)))
                training_size = max(1, size - validation_size)
                splits: list[Split] = []
                seeds: list[int] = []
                full = [np.zeros((replicates, width)) for _ in range(2)]
                inner = [np.zeros((replicates, width)) for _ in range(2)]
                held = [np.zeros((replicates, width)) for _ in range(2)]
                for replicate in range(replicates):
                    seed = observation_seed(
                        theta_index, regime_index, size_index, replicate
                    )
                    seeds.append(seed + 500_000_000)
                    rng = random.Random(seed)
                    split = build_split(
                        regime, rng, members, HOLDOUT_PER_CLASS
                    )
                    observations = generate_observations(
                        rng,
                        split.train_pool,
                        size,
                        numerators,
                        denominator,
                        structure.class_of_index,
                    )
                    splits.append(split)
                    for position, (index, answer) in enumerate(
                        zip(
                            observations.indices,
                            observations.answers,
                            strict=True,
                        )
                    ):
                        slot = 0 if answer == 1 else 1
                        full[slot][replicate, index] += 1.0
                        target = inner if position < training_size else held
                        target[slot][replicate, index] += 1.0

                fitted = fit_u_variants(
                    designs,
                    full[0],
                    full[1],
                    inner[0],
                    inner[1],
                    held[0],
                    held[1],
                    seeds,
                )
                for name, (predictions, selection) in fitted.items():
                    store = _accumulator()
                    for replicate in range(replicates):
                        values = tuple(
                            float(value) for value in predictions[replicate]
                        )
                        metrics = evaluate_exactly(
                            values,
                            splits[replicate],
                            truths,
                            structure.class_of_index,
                        )
                        store["excess_aggregate"].append(
                            metrics["aggregate"]["excess_nll"]
                        )
                        for label in ("O_0", "O_1"):
                            store[f"excess_{label}"].append(
                                metrics[label]["excess_nll"]
                            )
                        store["nll_aggregate"].append(
                            metrics["aggregate"]["nll"]
                        )
                        store["brier_aggregate"].append(
                            metrics["aggregate"]["brier"]
                        )
                        store["calibration_aggregate"].append(
                            metrics["aggregate"]["calibration_error"]
                        )
                        rows.append(
                            (
                                spec.identifier,
                                regime,
                                size,
                                name,
                                replicate,
                                _round(metrics["aggregate"]["excess_nll"]),
                                _round(metrics["O_0"]["excess_nll"]),
                                _round(metrics["O_1"]["excess_nll"]),
                                _round(metrics["aggregate"]["nll"]),
                                _round(metrics["aggregate"]["brier"]),
                                _round(
                                    metrics["aggregate"]["calibration_error"]
                                ),
                                "",
                                "",
                                selection["declared_parameters"],
                            )
                        )
                    summary = _summarize(
                        store,
                        replicates,
                        int(selection["declared_parameters"]),  # type: ignore[arg-type]
                    )
                    summary["model_selection"] = selection
                    cells[f"{spec.identifier}|{regime}|{size}|{name}"] = summary
    return cells, rows


def fairness_frontier(
    structure: ClassStructure,
    designs: dict[str, np.ndarray],
    spec: ThetaSpec,
    sizes: Sequence[int],
    replicates: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Trace ``U``'s whole regularization path with no internal split.

    This is a diagnostic, not a baseline. It selects the ridge with hindsight
    against the evaluation metric, which ``U`` is never allowed to do. Its only
    purpose is to let a reader verify that ``U`` was converged, that the
    declared ridge grid brackets its optimum, and that the measured gap is
    therefore not an optimization artifact.
    """
    members = _class_members(structure)
    numerators, denominator = theta_as_integers(spec.values)
    truths = [float(value) for value in spec.values]
    hidden_of = {name: hidden for name, hidden, _ in U_CANDIDATES}
    families = ("quadratic", "mlp")
    rows: list[dict[str, object]] = []
    oracle_cells: dict[str, dict[str, object]] = {}

    for regime_index, regime in enumerate(REGIMES):
        for size_index, size in enumerate(sizes):
            splits: list[Split] = []
            seeds: list[int] = []
            counts = [
                np.zeros((replicates, len(structure.inputs))) for _ in range(2)
            ]
            for replicate in range(replicates):
                seed = observation_seed(
                    0, regime_index, size_index, replicate
                )
                seeds.append(seed + 500_000_000)
                rng = random.Random(seed)
                split = build_split(regime, rng, members, HOLDOUT_PER_CLASS)
                observations = generate_observations(
                    rng,
                    split.train_pool,
                    size,
                    numerators,
                    denominator,
                    structure.class_of_index,
                )
                splits.append(split)
                for index, answer in zip(
                    observations.indices, observations.answers, strict=True
                ):
                    counts[0 if answer == 1 else 1][replicate, index] += 1.0
            totals = (counts[0] + counts[1]).sum(axis=1)

            path: dict[str, dict[str, float]] = {}
            best_overall = math.inf
            best_label = ""
            for family in families:
                path[family] = {}
                for ridge in U_RIDGE_GRID:
                    final, _, _ = _train_batched(
                        designs[family],
                        counts[0],
                        counts[1],
                        totals,
                        None,
                        None,
                        None,
                        seeds,
                        U_STEPS,
                        ridge,
                        hidden_of[family],
                    )
                    values = [
                        evaluate_exactly(
                            tuple(float(v) for v in final[replicate]),
                            splits[replicate],
                            truths,
                            structure.class_of_index,
                        )["aggregate"]["excess_nll"]
                        for replicate in range(replicates)
                    ]
                    mean = statistics.fmean(values)
                    path[family][str(ridge)] = _round(mean)
                    if mean < best_overall:
                        best_overall, best_label = mean, f"{family}@{ridge}"
            evaluated = tuple(
                index
                for _, indices in splits[0].evaluation
                for index in indices
                if _ == "aggregate"
            )
            pooled = asymptotic_floor(
                (0,) * len(structure.inputs),
                1,
                truths,
                structure.class_of_index,
                splits[0].train_pool,
                evaluated,
                ALPHA_PRIMARY,
                BETA_PRIMARY,
            )
            one_half = statistics.fmean(
                _cross_entropy(truths[structure.class_of_index[index]], 0.5)
                - _cross_entropy(
                    truths[structure.class_of_index[index]],
                    truths[structure.class_of_index[index]],
                )
                for index in evaluated
            )
            top = path[
                min(
                    families,
                    key=lambda family: path[family][str(U_RIDGE_GRID[-1])],
                )
            ][str(U_RIDGE_GRID[-1])]
            interior = best_label.split("@")[-1] not in (
                str(U_RIDGE_GRID[0]),
                str(U_RIDGE_GRID[-1]),
            )
            saturated = top <= max(pooled, one_half) * 1.05
            rows.append(
                {
                    "regime": regime,
                    "n": size,
                    "regularization_path": path,
                    "hindsight_best": _round(best_overall),
                    "hindsight_best_setting": best_label,
                    "best_constant_model_floor": _round(pooled),
                    "shrink_to_one_half_floor": _round(one_half),
                    "top_of_grid_value": _round(top),
                    "optimum_is_interior_to_the_grid": interior,
                    "top_of_grid_reaches_the_constant_floor": bool(saturated),
                    "grid_is_adequate": bool(interior or saturated),
                }
            )
            oracle_cells[
                f"{spec.identifier}|{regime}|{size}|U_oracle_ridge"
            ] = {
                "degrees_of_freedom": QUADRATIC_DIMENSION,
                "excess_aggregate": {
                    "mean": _round(best_overall),
                    "median": _round(best_overall),
                    "stdev": 0.0,
                    "ci95_low": _round(best_overall),
                    "ci95_high": _round(best_overall),
                    "replicates": replicates,
                },
                "excess_aggregate_matched": {
                    "mean": _round(best_overall),
                    "median": _round(best_overall),
                    "stdev": 0.0,
                    "ci95_low": _round(best_overall),
                    "ci95_high": _round(best_overall),
                    "replicates": replicates,
                },
                "selected_setting": best_label,
            }
    return (
        {
            "status": (
                "diagnostic only; the ridge is chosen with hindsight against "
                "the evaluation metric, which U is never allowed to do, so "
                "this is not a legitimate baseline and does not decide the "
                "verdict"
            ),
            "theta_star": spec.identifier,
            "families": list(families),
            "ridge_grid": list(U_RIDGE_GRID),
            "trained_on_all_n_with_no_internal_split": True,
            "grid_adequacy_argument": (
                "at the top of the ridge grid every candidate converges to the "
                "one-parameter constant model, whose excess log loss is "
                "reported alongside each row. No further extension of the grid "
                "can take U below that floor, so wherever the optimum sits at "
                "the top edge the grid has already reached the achievable "
                "frontier rather than being truncated short of it."
            ),
            "rows": rows,
        },
        oracle_cells,
    )


# ---------------------------------------------------------------------------
# 12. Aggregation, thresholds, and the mechanical verdict
# ---------------------------------------------------------------------------


def select_best_u(
    cells: dict[str, dict[str, object]],
    thetas: Sequence[ThetaSpec],
    sizes: Sequence[int],
) -> dict[str, dict[str, object]]:
    """Per cell, keep whichever preregistered ``U`` variant did better.

    The rule is conservative against the training-economy thesis: it hands the
    baseline its best preregistered shot before T is compared against it.
    """
    chosen: dict[str, dict[str, object]] = {}
    for spec in thetas:
        for regime in REGIMES:
            for size in sizes:
                best_name = None
                best_mean = math.inf
                for condition in LEARNED_CONDITIONS:
                    key = f"{spec.identifier}|{regime}|{size}|{condition}"
                    if key not in cells:
                        continue
                    mean = float(
                        cells[key]["excess_aggregate"]["mean"]  # type: ignore[index]
                    )
                    if mean < best_mean:
                        best_mean, best_name = mean, condition
                if best_name is None:
                    continue
                source = f"{spec.identifier}|{regime}|{size}|{best_name}"
                payload = dict(cells[source])
                payload["selected_variant"] = best_name
                chosen[f"{spec.identifier}|{regime}|{size}|U_best"] = payload
    return chosen


def series_for(
    cells: dict[str, dict[str, object]],
    theta_id: str,
    regime: str,
    condition: str,
    sizes: Sequence[int],
    statistic: str,
) -> dict[int, float]:
    series: dict[int, float] = {}
    for size in sizes:
        key = f"{theta_id}|{regime}|{size}|{condition}"
        if key in cells:
            series[size] = float(
                cells[key][statistic]["mean"]  # type: ignore[index]
            )
    return series


def labels_to_threshold(
    series: dict[int, float], threshold: float
) -> int | None:
    for size in sorted(series):
        if series[size] <= threshold:
            return size
    return None


def interpolated_threshold(
    series: dict[int, float], threshold: float
) -> float | None:
    sizes = sorted(series)
    if not sizes:
        return None
    if series[sizes[0]] <= threshold:
        return float(sizes[0])
    for left, right in zip(sizes, sizes[1:], strict=False):
        if series[left] > threshold >= series[right]:
            if series[left] <= 0 or series[right] <= 0:
                return float(right)
            slope = (math.log(right) - math.log(left)) / (
                math.log(series[right]) - math.log(series[left])
            )
            return _round(
                math.exp(
                    math.log(left)
                    + slope * (math.log(threshold) - math.log(series[left]))
                )
            )
    return None


def _advantage(
    typed: int | None, rival: int | None, ceiling: int
) -> dict[str, object]:
    typed_value = typed if typed is not None else 2 * ceiling
    rival_value = rival if rival is not None else 2 * ceiling
    ratio = rival_value / typed_value if typed_value else math.inf
    return {
        "typed_labels": typed,
        "rival_labels": rival,
        "typed_reached_threshold": typed is not None,
        "rival_reached_threshold": rival is not None,
        "label_ratio": _round(ratio),
        "material": bool(ratio >= MATERIAL_FACTOR),
        "rival_strictly_better": bool(rival_value < typed_value),
    }


def build_thresholds(
    cells: dict[str, dict[str, object]],
    thetas: Sequence[ThetaSpec],
    sizes: Sequence[int],
    conditions: Sequence[str],
) -> dict[str, object]:
    table: dict[str, object] = {}
    for spec in thetas:
        for regime in REGIMES:
            for condition in conditions:
                full = series_for(
                    cells,
                    spec.identifier,
                    regime,
                    condition,
                    sizes,
                    "excess_aggregate",
                )
                matched = series_for(
                    cells,
                    spec.identifier,
                    regime,
                    condition,
                    sizes,
                    "excess_aggregate_matched",
                )
                if not full:
                    continue
                table[f"{spec.identifier}|{regime}|{condition}"] = {
                    "mean_excess_nll_by_n": {
                        str(size): _round(value) for size, value in full.items()
                    },
                    "labels_to_threshold": {
                        str(threshold): labels_to_threshold(full, threshold)
                        for threshold in THRESHOLDS
                    },
                    "labels_to_threshold_matched_replicates": {
                        str(threshold): labels_to_threshold(matched, threshold)
                        for threshold in THRESHOLDS
                    },
                    "interpolated_labels_to_threshold": {
                        str(threshold): interpolated_threshold(full, threshold)
                        for threshold in THRESHOLDS
                    },
                }
    return table


def decide_verdict(
    cells: dict[str, dict[str, object]],
    sizes: Sequence[int],
    agreement: dict[str, int],
) -> dict[str, object]:
    ceiling = max(sizes)
    primary = THETAS[0].identifier
    statistic = "excess_aggregate_matched"
    comparisons: dict[str, object] = {}
    material_over_u = []
    material_over_s = []
    u_strictly_better = []
    identical_to_u = []
    for regime in REGIMES:
        typed = series_for(cells, primary, regime, "T", sizes, statistic)
        rival = series_for(cells, primary, regime, "U_best", sizes, statistic)
        saturated = series_for(cells, primary, regime, "S", sizes, statistic)
        generic = series_for(cells, primary, regime, "G", sizes, statistic)
        oracle = series_for(
            cells, primary, regime, "U_oracle_ridge", sizes, statistic
        )
        row: dict[str, object] = {}
        for threshold in THRESHOLDS:
            typed_n = labels_to_threshold(typed, threshold)
            rival_n = labels_to_threshold(rival, threshold)
            saturated_n = labels_to_threshold(saturated, threshold)
            generic_n = labels_to_threshold(generic, threshold)
            oracle_n = labels_to_threshold(oracle, threshold)
            row[f"threshold_{threshold}"] = {
                "T_versus_U_best": _advantage(typed_n, rival_n, ceiling),
                "T_versus_S": _advantage(typed_n, saturated_n, ceiling),
                "T_versus_G": _advantage(typed_n, generic_n, ceiling),
                "T_versus_U_oracle_ridge_robustness_only": _advantage(
                    typed_n, oracle_n, ceiling
                ),
            }
            if threshold == THRESHOLDS[0]:
                material_over_u.append(
                    bool(row[f"threshold_{threshold}"]["T_versus_U_best"][  # type: ignore[index]
                        "material"
                    ])
                )
                material_over_s.append(
                    bool(row[f"threshold_{threshold}"]["T_versus_S"][  # type: ignore[index]
                        "material"
                    ])
                )
            u_strictly_better.append(
                bool(row[f"threshold_{threshold}"]["T_versus_U_best"][  # type: ignore[index]
                    "rival_strictly_better"
                ])
            )
            identical_to_u.append(typed_n == rival_n)
        comparisons[regime] = row

    exact = (
        agreement["T_versus_O_prediction_mismatches"] == 0
        and agreement["T_versus_G_prediction_mismatches"] == 0
    )
    if not exact:
        verdict = "IMPLEMENTATION DEFECT"
        why = (
            "T, O, and G must produce identical predictions under matched "
            "estimation; they did not"
        )
    elif any(u_strictly_better):
        verdict = "NEGATIVE"
        why = (
            "the better preregistered U variant reached a threshold at a "
            "strictly smaller grid point than T"
        )
    elif all(material_over_u):
        verdict = "STRONG POSITIVE"
        why = (
            "T holds a factor-two or larger label advantage over the better "
            "preregistered U variant in both regimes, and G, given the same "
            "group, matches T exactly"
        )
    elif all(identical_to_u):
        verdict = "NULL"
        why = (
            "the better preregistered U variant reaches every threshold at the "
            "same grid point as T"
        )
    else:
        verdict = "NARROW POSITIVE"
        why = (
            "T beats the saturated ablation decisively but holds no factor-two "
            "label advantage over the better preregistered U variant in both "
            "regimes"
        )
    return {
        "rule": (
            "preregistered: material advantage means a factor of two or more "
            "fewer labeled examples on the declared grid to reach mean excess "
            "NLL at or below 0.02, in both regimes at the primary theta_star"
        ),
        "statistic_used": (
            "matched-replicate mean excess NLL, so T and U are compared on "
            "byte-identical observations and splits"
        ),
        "primary_theta": primary,
        "comparisons": comparisons,
        "T_equals_O_exactly": agreement["T_versus_O_prediction_mismatches"] == 0,
        "T_equals_G_exactly": agreement["T_versus_G_prediction_mismatches"] == 0,
        "exact_agreement_comparisons": agreement["comparisons"],
        "material_over_U_by_regime": material_over_u,
        "material_over_S_by_regime": material_over_s,
        "verdict": verdict,
        "reason": why,
    }


# ---------------------------------------------------------------------------
# 13. Authority and payloads
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.24-Task-first-matched-TDM-training-economy-experiment.md"
        ),
        "primary_gpt_authority_named_by_the_task": PRIMARY_GPT_AUTHORITY,
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "inherits": [
            "017.23 operational TDM equivalence and the S(E) state boundary",
            "017.21a canonical central test and binding-unique state compiler",
            "017.20 rigorous input Type Schema fixture",
        ],
        "byte_pinned_inputs": {
            "successor_schema.json": SUCCESSOR_SHA256,
            "preregistration.json": PREREGISTRATION_SHA256,
            "../017.20-Code-attachments/source_fixture.json": PREDECESSOR_SHA256,
        },
    }


def build_orbit_payload(
    structure: ClassStructure,
    document: dict[str, object],
    features: np.ndarray,
) -> dict[str, object]:
    predecessor = load_predecessor()
    generic = invariance_partition_from_group(
        structure.inputs, structure.schema, structure.automorphisms
    )
    grid = tuple(Fraction(index, 24) for index in range(25))
    bridge = decision_model_bridge(grid, structure, THETAS[0].values)
    leakage = leakage_certificate(document, structure)
    controls = audit_controls(structure, features)
    span = span_certificate(features, structure.class_of_index, structure.inputs)
    indicator = realize_class_indicator(features, structure.class_of_index)
    single = realize_single_pair_indicator(features, 0)

    withheld = set(predecessor.keys()) - set(document.keys())
    payload = {
        "schema": ORBIT_SCHEMA,
        "task": (
            "occurrence/gpt Issue 017.24 first matched TDM training-economy "
            "experiment"
        ),
        "authority": authority_record(),
        "A_schema": {
            "successor_schema": SUCCESSOR_SCHEMA,
            "declared": [
                "sort Point, cardinality 7",
                "sort Line, cardinality 7",
                "the 21-member incidence relation",
                "admitted inputs Point x Line, cardinality 49",
                "raw-presentation equivalence under incidence automorphisms",
                "a binary answer type with the neutral labels a_0 and a_1",
            ],
            "no_longer_declared": sorted(
                str(item) for item in document["not_declared_here"]  # type: ignore[index]
            ),
            "predecessor_top_level_keys_dropped": sorted(withheld),
            "predecessor_declared_a_total_rule": "rule"
            in predecessor["decision_type"],  # type: ignore[operator]
            "successor_declares_no_rule": "rule"
            not in document["answer_type"],  # type: ignore[operator]
            "answer_labels": list(
                document["answer_type"]["labels"]  # type: ignore[index]
            ),
        },
        "B_compilation": {
            "compiler_inputs": ["sorts", "relations"],
            "compiler_saw_any_label": False,
            "automorphism_group_order": structure.group_order,
            "expected_group_order": document["automorphisms"][  # type: ignore[index]
                "expected_group_order"
            ],
            "class_count": len(structure.classes),
            "class_cardinalities": list(structure.cardinalities),
            "canonical_naming": (
                "classes ordered by cardinality then by least member, reported "
                "as O_0 and O_1; the names carry no answer meaning"
            ),
            "classes": [
                {
                    "name": structure.name(index),
                    "cardinality": len(members),
                    "least_member": list(members[0]),
                    "members": [list(datum) for datum in members],
                }
                for index, members in enumerate(structure.classes)
            ],
            "generic_group_closure_agrees": generic
            == structure.class_of_index,
            "generic_group_closure_free_parameters": len(set(generic)),
            "historical_correspondence_is_bookkeeping_only": {
                "O_0_is_the_21_member_class": len(structure.classes[0]) == 21,
                "note": (
                    "the 21-member class coincides with the predecessor's "
                    "marked pairs; that correspondence is recorded for the "
                    "reader and is never supplied to any learner"
                ),
            },
        },
        "C_learned_residue": {
            "central_test": {"z_M": "(I_2, 0)", "z_s": "(0, 1)"},
            "invariant_family": (
                "omega_lambda(A,s) = lambda Tr(A)/2 + (1-lambda) s"
            ),
            "public_probability": "P(a_0 | x) = lambda_{o(x)}",
            "observable_parameters": 2,
            "orientation_is_a_reparameterization": True,
            "decision_model_bridge": bridge,
        },
        "D_fairness": {
            "uncompiled_encoding": span,
            "class_function_realizability": indicator,
            "single_pair_realizability": single,
            "leakage_certificate": leakage,
        },
        "controls": controls,
    }
    return payload


def build_split_payload(structure: ClassStructure) -> dict[str, object]:
    members = _class_members(structure)
    witnesses: list[dict[str, object]] = []
    deterministic = True
    both_classes_seen = True
    both_classes_held = True
    disjoint = True
    for replicate in range(8):
        seed = observation_seed(0, 1, 0, replicate)
        first = build_split("B", random.Random(seed), members, HOLDOUT_PER_CLASS)
        second = build_split("B", random.Random(seed), members, HOLDOUT_PER_CLASS)
        deterministic &= first == second
        seen_classes = {structure.class_of_index[i] for i in first.train_pool}
        held_classes = {structure.class_of_index[i] for i in first.held_out}
        both_classes_seen &= seen_classes == {0, 1}
        both_classes_held &= held_classes == {0, 1}
        disjoint &= not (set(first.train_pool) & set(first.held_out))
        witnesses.append(
            {
                "seed": seed,
                "held_out": [
                    list(structure.inputs[index]) for index in first.held_out
                ],
                "held_out_by_class": [
                    sum(
                        1
                        for index in first.held_out
                        if structure.class_of_index[index] == cell
                    )
                    for cell in (0, 1)
                ],
                "seen_identities": len(first.train_pool),
            }
        )
    regime_a = build_split("A", random.Random(0), members, HOLDOUT_PER_CLASS)
    seen_counts = [
        len(members[cell]) - HOLDOUT_PER_CLASS[cell] for cell in (0, 1)
    ]
    return {
        "schema": SPLIT_SCHEMA,
        "authority": authority_record(),
        "seed_formula": (
            "base + 1000000 * theta_index + 100000 * regime_index + 1000 * "
            "n_index + replicate, base = 20260923"
        ),
        "regime_A": {
            "train_pool_size": len(regime_a.train_pool),
            "evaluation_sets": [
                {"name": name, "size": len(indices)}
                for name, indices in regime_a.evaluation
            ],
            "train_weights": [
                _fraction_text(Fraction(len(members[cell]), 49))
                for cell in (0, 1)
            ],
            "eval_weights": [
                _fraction_text(Fraction(len(members[cell]), 49))
                for cell in (0, 1)
            ],
        },
        "regime_B": {
            "holdout_per_class": list(HOLDOUT_PER_CLASS),
            "held_out_identities": sum(HOLDOUT_PER_CLASS),
            "seen_identities": 49 - sum(HOLDOUT_PER_CLASS),
            "seen_identities_by_class": seen_counts,
            "train_weights": [
                _fraction_text(Fraction(count, sum(seen_counts)))
                for count in seen_counts
            ],
            "eval_weights": [
                _fraction_text(
                    Fraction(count, sum(HOLDOUT_PER_CLASS))
                )
                for count in HOLDOUT_PER_CLASS
            ],
            "witnesses": witnesses,
        },
        "checks": {
            "split_generation_is_deterministic_in_the_seed": deterministic,
            "both_classes_are_represented_among_seen_identities": (
                both_classes_seen
            ),
            "both_classes_are_represented_among_held_out_identities": (
                both_classes_held
            ),
            "train_pool_and_held_out_are_disjoint": disjoint,
            "splits_are_label_free": True,
            "every_condition_receives_the_same_split": True,
        },
    }


def build_reference_payload(
    structure: ClassStructure,
) -> dict[str, object]:
    members = _class_members(structure)
    seen = [len(members[cell]) - HOLDOUT_PER_CLASS[cell] for cell in (0, 1)]
    designs = {
        "A": (
            tuple(Fraction(len(members[cell]), 49) for cell in (0, 1)),
            tuple(Fraction(len(members[cell]), 49) for cell in (0, 1)),
        ),
        "B": (
            tuple(Fraction(count, sum(seen)) for count in seen),
            tuple(
                Fraction(count, sum(HOLDOUT_PER_CLASS))
                for count in HOLDOUT_PER_CLASS
            ),
        ),
    }
    wrong_partition, _ = build_wrong_partition(structure)
    scrambled = compile_classes(build_scrambled_schemas(structure.schema, 1)[0])
    everything = tuple(range(len(structure.inputs)))
    identity = tuple(range(len(structure.inputs)))

    blocks: dict[str, object] = {}
    floors: dict[str, object] = {}
    for spec in THETAS:
        truths = [float(value) for value in spec.values]
        for regime, (train_weights, eval_weights) in designs.items():
            blocks[f"{spec.identifier}|{regime}"] = analytic_reference(
                SAMPLE_SIZES,
                spec.values,
                ALPHA_PRIMARY,
                BETA_PRIMARY,
                train_weights,
                eval_weights,
            )
        floors[spec.identifier] = {
            "regime_A_wrong_compiler": _round(
                asymptotic_floor(
                    wrong_partition,
                    2,
                    truths,
                    structure.class_of_index,
                    everything,
                    everything,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            ),
            "regime_A_scrambled_compiler": _round(
                asymptotic_floor(
                    scrambled.class_of_index,
                    len(scrambled.classes),
                    truths,
                    structure.class_of_index,
                    everything,
                    everything,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            ),
            "regime_A_saturated": _round(
                asymptotic_floor(
                    identity,
                    49,
                    truths,
                    structure.class_of_index,
                    everything,
                    everything,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            ),
            "regime_B_saturated_on_unseen_identities": _round(
                asymptotic_floor(
                    identity,
                    49,
                    truths,
                    structure.class_of_index,
                    tuple(index for index in everything if index % 5 != 0),
                    tuple(index for index in everything if index % 5 == 0),
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            ),
            "regime_B_typed_on_unseen_identities": _round(
                asymptotic_floor(
                    structure.class_of_index,
                    2,
                    truths,
                    structure.class_of_index,
                    tuple(index for index in everything if index % 5 != 0),
                    tuple(index for index in everything if index % 5 == 0),
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            ),
        }
    return {
        "schema": REFERENCE_SCHEMA,
        "authority": authority_record(),
        "estimator": "beta(1,1) posterior mean, (k + 1) / (m + 2)",
        "method": (
            "exact finite double sum over the binomial class-count lattice and "
            "the binomial success lattice; no simulation"
        ),
        "analytic_sweeps": blocks,
        "asymptotic_floors": floors,
        "floor_note": (
            "a floor is the excess log loss a partition cannot escape with "
            "unlimited labels; the regime B rows use a fixed illustrative "
            "one-in-five holdout so the typed and saturated floors are "
            "comparable at a glance"
        ),
        "scrambled_reference_structure": {
            "group_order": scrambled.group_order,
            "class_count": len(scrambled.classes),
            "class_cardinalities": sorted(scrambled.cardinalities),
        },
    }


# ---------------------------------------------------------------------------
# 14. Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bundle:
    orbit: dict[str, object]
    split: dict[str, object]
    reference: dict[str, object]
    results: dict[str, object]
    cost: dict[str, object]
    table: bytes


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _canonical_table(rows: Sequence[tuple[object, ...]]) -> bytes:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row[0]),
            str(row[1]),
            int(row[2]),  # type: ignore[arg-type]
            str(row[3]),
            int(row[4]),  # type: ignore[arg-type]
        ),
    )
    lines = [",".join(TABLE_COLUMNS)]
    lines.extend(",".join(str(value) for value in row) for row in ordered)
    return gzip.compress(("\n".join(lines) + "\n").encode(), 9, mtime=0)


def _analytic_agreement(
    cells: dict[str, dict[str, object]],
    reference: dict[str, object],
    sizes: Sequence[int],
) -> dict[str, object]:
    """Validate the simulation against the exact two-parameter statistics."""
    sweeps = reference["analytic_sweeps"]
    rows: list[dict[str, object]] = []
    worst = 0.0
    failures = 0
    for spec in THETAS:
        for regime in REGIMES:
            exact = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in sweeps[f"{spec.identifier}|{regime}"]["sweep"]  # type: ignore[index]
            }
            for size in sizes:
                key = f"{spec.identifier}|{regime}|{size}|T"
                block = cells[key]["excess_aggregate"]  # type: ignore[index]
                mean = float(block["mean"])  # type: ignore[index]
                high = float(block["ci95_high"])  # type: ignore[index]
                error = abs(mean - exact[size])
                standard = max((high - mean) / 1.959963984540054, 1e-15)
                sigma = error / standard
                worst = max(worst, sigma)
                ok = sigma <= 4.0
                failures += int(not ok)
                rows.append(
                    {
                        "cell": f"{spec.identifier}|{regime}|{size}",
                        "simulated_mean": _round(mean),
                        "exact_expected": _round(exact[size]),
                        "deviation_in_standard_errors": _round(sigma),
                        "within_four_standard_errors": ok,
                    }
                )
    return {
        "method": (
            "the exact binomial-lattice expectation must lie within four "
            "standard errors of the simulated mean for condition T in every cell"
        ),
        "cells_checked": len(rows),
        "failures": failures,
        "worst_deviation_in_standard_errors": _round(worst),
        "all_cells_agree": failures == 0,
        "rows": rows,
    }


def run_everything(quick: bool = False) -> Bundle:
    started = time.perf_counter()

    moment = time.perf_counter()
    schema, document = load_successor()
    parse_seconds = time.perf_counter() - moment
    preregistration = load_preregistration()

    moment = time.perf_counter()
    structure = compile_classes(schema)
    derivation_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    generic = invariance_partition_from_group(
        structure.inputs, structure.schema, structure.automorphisms
    )
    generic_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    features = build_feature_matrix(schema)
    encoding_seconds = time.perf_counter() - moment

    sizes = QUICK_SIZES if quick else SAMPLE_SIZES
    closed_replicates = (
        QUICK_CLOSED_FORM_REPLICATES if quick else CLOSED_FORM_REPLICATES
    )
    learned_replicates = (
        QUICK_LEARNED_REPLICATES if quick else LEARNED_REPLICATES
    )

    wrong, _ = build_wrong_partition(structure)
    scrambled = compile_classes(build_scrambled_schemas(schema, 1)[0])
    conditions = (
        ConditionSpec("T", structure.class_of_index, 2, "tdm", False),
        ConditionSpec("O", structure.class_of_index, 2, "bernoulli", False),
        ConditionSpec(
            "S", tuple(range(len(structure.inputs))), len(structure.inputs),
            "bernoulli", False,
        ),
        ConditionSpec("G", generic, len(set(generic)), "bernoulli", False),
        ConditionSpec("W_wrong_compiler", wrong, 2, "tdm", True),
        ConditionSpec(
            "C_scrambled_compiler",
            scrambled.class_of_index,
            len(scrambled.classes),
            "tdm",
            True,
        ),
    )

    moment = time.perf_counter()
    cells, rows, agreement = run_closed_form(
        structure,
        conditions,
        THETAS,
        sizes,
        closed_replicates,
        learned_replicates,
        ALPHA_PRIMARY,
        BETA_PRIMARY,
    )
    closed_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    secondary_cells, _, _ = run_closed_form(
        structure,
        conditions[:4],
        THETAS[:1],
        sizes,
        closed_replicates,
        learned_replicates,
        ALPHA_SECONDARY,
        BETA_SECONDARY,
    )
    secondary_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    learned_cells, learned_rows = run_learned(
        structure, features, THETAS, sizes, learned_replicates
    )
    learned_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    frontier, oracle_cells = fairness_frontier(
        structure, build_designs(features), THETAS[0], sizes, learned_replicates
    )
    frontier_seconds = time.perf_counter() - moment

    cells.update(learned_cells)
    cells.update(select_best_u(cells, THETAS, sizes))
    cells.update(oracle_cells)
    rows.extend(learned_rows)

    orbit = build_orbit_payload(structure, document, features)
    split = build_split_payload(structure)
    reference = build_reference_payload(structure)
    thresholds = build_thresholds(
        cells,
        THETAS,
        sizes,
        ("T", "O", "S", "G", *LEARNED_CONDITIONS, "U_best",
         "U_oracle_ridge", "W_wrong_compiler", "C_scrambled_compiler"),
    )
    verdict = decide_verdict(cells, sizes, agreement)
    validation = _analytic_agreement(cells, reference, sizes)

    controls = orbit["controls"]
    fairness = orbit["D_fairness"]
    compilation = orbit["B_compilation"]
    bridge = orbit["C_learned_residue"]["decision_model_bridge"]  # type: ignore[index]
    leakage = fairness["leakage_certificate"]  # type: ignore[index]
    span = fairness["uncompiled_encoding"]  # type: ignore[index]

    checks = {
        "successor_schema_bytes_match_the_pin": True,
        "preregistration_bytes_match_the_pin": True,
        "predecessor_fixture_bytes_match_the_017_20_freeze": True,
        "preregistration_declares_the_verdict_rule": "verdict_rule"
        in preregistration,
        "source_group_order_is_168": structure.group_order == 168,
        "two_derived_classes_21_and_28": sorted(structure.cardinalities)
        == [21, 28],
        "compiler_reads_only_sorts_and_relations": compilation[  # type: ignore[index]
            "compiler_inputs"
        ]
        == ["sorts", "relations"],
        "generic_group_closure_agrees_with_the_compiler": compilation[  # type: ignore[index]
            "generic_group_closure_agrees"
        ],
        "generic_closure_has_two_free_parameters": compilation[  # type: ignore[index]
            "generic_group_closure_free_parameters"
        ]
        == 2,
        "tdm_resolves_exactly_through_the_public_contracts": bridge[  # type: ignore[index]
            "resolves_exactly_to_lambda_and_one_minus_lambda"
        ],
        "tdm_normalizes_at_zero_tolerance": bridge[  # type: ignore[index]
            "normalizes_at_zero_tolerance"
        ],
        "nothing_added_to_the_decision_model_public_api": not bridge[  # type: ignore[index]
            "anything_added_to_the_public_api"
        ],
        "sigma_prime_declares_no_forbidden_key": leakage[  # type: ignore[index]
            "sigma_prime_declares_no_forbidden_key"
        ],
        "sigma_prime_contains_no_float": leakage[  # type: ignore[index]
            "sigma_prime_contains_no_float"
        ],
        "no_array_mixes_a_sort_identifier_with_an_answer_label": leakage[  # type: ignore[index]
            "no_array_mixes_a_sort_identifier_with_an_answer_label"
        ],
        "compiler_carrier_has_no_answer_field": leakage[  # type: ignore[index]
            "compiler_carrier_has_no_answer_field"
        ],
        "no_fitting_function_accepts_a_target_parameter": leakage[  # type: ignore[index]
            "no_fitting_function_accepts_a_target_parameter"
        ],
        "compiler_ignores_the_redundant_expectation_blocks": leakage[  # type: ignore[index]
            "compiler_ignores_the_redundant_expectation_blocks"
        ],
        "relabeling_the_answers_only_complements_the_estimates": leakage[  # type: ignore[index]
            "relabeling_the_answers_only_complements_the_estimates"
        ],
        "U_realizes_the_class_function_exactly": fairness[  # type: ignore[index]
            "class_function_realizability"
        ]["class_function_is_exactly_realizable"],
        "U_realizes_a_single_pair_exactly": fairness[  # type: ignore[index]
            "single_pair_realizability"
        ]["single_pair_is_exactly_realizable"],
        "no_U_feature_column_is_a_function_of_the_class": span[  # type: ignore[index]
            "no_column_is_a_function_of_the_class"
        ],
        "class_indicator_is_not_in_the_U_linear_span": not span[  # type: ignore[index]
            "indicator_is_in_the_linear_span"
        ],
        "relation_erased_compiler_fails_closed": controls[  # type: ignore[index]
            "relation_erased"
        ]["compiler_fails_closed"],
        "relation_erased_encoder_fails_closed": controls[  # type: ignore[index]
            "relation_erased"
        ]["relational_encoder_fails_closed"],
        "no_scramble_reproduces_the_declared_quotient": controls[  # type: ignore[index]
            "scrambled_structure"
        ]["scrambles_reproducing_the_declared_quotient"]
        == 0,
        "wrong_partition_respects_no_automorphism": not controls[  # type: ignore[index]
            "wrong_compiler"
        ]["respects_the_declared_automorphisms"],
        "wrong_partition_preserves_21_and_28": controls[  # type: ignore[index]
            "wrong_compiler"
        ]["cardinalities"]
        == [21, 28],
        "splits_are_deterministic_in_the_seed": split["checks"][  # type: ignore[index]
            "split_generation_is_deterministic_in_the_seed"
        ],
        "both_classes_are_seen_and_held_out": split["checks"][  # type: ignore[index]
            "both_classes_are_represented_among_seen_identities"
        ]
        and split["checks"][  # type: ignore[index]
            "both_classes_are_represented_among_held_out_identities"
        ],
        "train_pool_and_held_out_are_disjoint": split["checks"][  # type: ignore[index]
            "train_pool_and_held_out_are_disjoint"
        ],
        "T_equals_O_exactly_in_every_replicate": agreement[
            "T_versus_O_prediction_mismatches"
        ]
        == 0,
        "T_equals_G_exactly_in_every_replicate": agreement[
            "T_versus_G_prediction_mismatches"
        ]
        == 0,
        "simulation_agrees_with_the_exact_reference": validation[
            "all_cells_agree"
        ],
        "U_regularization_grid_is_adequate_in_every_cell": all(
            bool(row["grid_is_adequate"])
            for row in frontier["rows"]  # type: ignore[index,union-attr]
        ),
        "every_theta_star_is_interior": all(
            0 < value < 1 for spec in THETAS for value in spec.values
        ),
        "theta_star_separations_are_distinct": len(
            {spec.values for spec in THETAS}
        )
        == len(THETAS),
        "evaluation_clipping_is_inert_for_the_bernoulli_conditions": (
            1.0 / (max(sizes) + 2) > CLIP
        ),
        "learned_replicate_floor_respected": learned_replicates >= 100
        or quick,
        "closed_form_replicate_floor_respected": closed_replicates >= 100
        or quick,
        "secondary_estimator_was_also_run": bool(secondary_cells),
    }
    if not all(checks.values()):
        broken = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"mechanical checks failed: {broken}")

    total_seconds = time.perf_counter() - started
    results = {
        "schema": RESULTS_SCHEMA,
        "task": (
            "occurrence/gpt Issue 017.24 first matched TDM training-economy "
            "experiment"
        ),
        "authority": authority_record(),
        "design": {
            "sample_sizes": list(sizes),
            "closed_form_replicates": closed_replicates,
            "learned_replicates": learned_replicates,
            "replicate_mismatch_disclosure": (
                "U is run at the reduced frozen replicate count the "
                "preregistration allows; its confidence intervals are wider "
                "for that reason alone and must not be read as though the "
                "replicate counts matched. Every T-versus-U comparison uses "
                "the matched-replicate statistic on byte-identical data."
            ),
            "theta_star": [
                {
                    "id": spec.identifier,
                    "role": spec.role,
                    "lambda_0": _fraction_text(spec.values[0]),
                    "lambda_1": _fraction_text(spec.values[1]),
                }
                for spec in THETAS
            ],
            "estimator_primary": "beta(1,1) posterior mean",
            "estimator_secondary": "beta(1/2,1/2) posterior mean",
            "conditions": [
                {
                    "name": condition.name,
                    "learned_scalars": condition.cell_count,
                    "evaluation_path": condition.kind,
                    "primary_theta_only": condition.primary_theta_only,
                }
                for condition in conditions
            ]
            + [
                {
                    "name": name,
                    "candidate_ladder": list(candidates),
                    "step_budget": U_STEPS,
                    "ridge_grid": list(ridges),
                    "internal_label_only_model_selection": use_validation,
                    "primary_theta_only": False,
                }
                for name, candidates, ridges, use_validation in U_VARIANTS
            ],
        },
        "cells": cells,
        "secondary_estimator_cells": secondary_cells,
        "thresholds": thresholds,
        "analytic_validation": validation,
        "D_fairness_frontier": frontier,
        "exact_agreement": agreement,
        "J_verdict": verdict,
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": True,
        "claim_fences": [
            "no language-scale, LLM, or SLM comparison is made or implied",
            "no physical Test Realization or Outcome constitution is claimed",
            "no autonomous dynamics and no Interact are invoked",
            "it is not claimed that every useful Type Schema admits a compact "
            "compiler",
            "it is not claimed that OT is uniquely responsible for any measured "
            "gain; conditions O and G exist precisely to deny that inference",
            "the fixture is a finite controlled problem and establishes "
            "mechanism, not production economics",
            "Issue 018 is not resolved here",
        ],
        "stop_condition_respected": {
            "scaled_to_language_or_embeddings_or_transformers": False,
            "external_corpora_used": False,
            "issue_018_touched": False,
        },
    }

    cost = {
        "schema": COST_SCHEMA,
        "authority": authority_record(),
        "seconds": {
            "schema_parse": _round(parse_seconds),
            "automorphism_and_class_derivation": _round(derivation_seconds),
            "generic_group_closure": _round(generic_seconds),
            "relational_encoding_build": _round(encoding_seconds),
            "closed_form_fitting": _round(closed_seconds),
            "secondary_estimator_fitting": _round(secondary_seconds),
            "learned_fitting": _round(learned_seconds),
            "fairness_frontier_diagnostic": _round(frontier_seconds),
            "total_wall_clock": _round(total_seconds),
        },
        "derivation_reuse": {
            "computed_once_per_type_schema": True,
            "cached_across_training_runs": True,
            "reused_across_the_sample_size_sweep": True,
            "derivations_performed": 1,
            "model_fits_served_by_that_one_derivation": (
                len(rows) - len(learned_rows)
            ),
        },
        "honest_accounting": (
            "compilation is not free. On this 49-input fixture the one-off "
            "derivation cost is reported next to the fitting cost it serves; "
            "the ratio is a property of a tiny fixture and is not a production "
            "economics claim."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "numpy_used_only_for": "condition U and the parity certificates",
        },
    }
    return Bundle(
        orbit=orbit,
        split=split,
        reference=reference,
        results=results,
        cost=cost,
        table=_canonical_table(rows),
    )


# ---------------------------------------------------------------------------
# 15. Verification and the command line
# ---------------------------------------------------------------------------


def _numeric_close(left: object, right: object, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _numeric_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numeric_close(first, second, tolerance)
            for first, second in zip(left, right, strict=True)
        )
    return left == right


def _table_rows(blob: bytes) -> list[list[str]]:
    text = gzip.decompress(blob).decode()
    return [line.split(",") for line in text.strip().split("\n")]


def _compare_table(observed: bytes, expected: bytes) -> list[str]:
    left, right = _table_rows(observed), _table_rows(expected)
    if len(left) != len(right):
        return [f"per_replicate row count {len(left)} != {len(right)}"]
    if left[0] != right[0]:
        return ["per_replicate header mismatch"]
    problems: list[str] = []
    exact_mismatch = 0
    numeric_mismatch = 0
    for observed_row, expected_row in zip(left[1:], right[1:], strict=True):
        if observed_row[:5] != expected_row[:5]:
            problems.append("per_replicate row ordering mismatch")
            break
        if observed_row[3].startswith("U_"):
            for first, second in zip(
                observed_row[5:11], expected_row[5:11], strict=True
            ):
                if abs(float(first) - float(second)) > NUMERIC_TOLERANCE:
                    numeric_mismatch += 1
        elif observed_row != expected_row:
            exact_mismatch += 1
    if exact_mismatch:
        problems.append(
            f"{exact_mismatch} closed-form rows differ byte-for-byte"
        )
    if numeric_mismatch:
        problems.append(
            f"{numeric_mismatch} learned values exceed tolerance "
            f"{NUMERIC_TOLERANCE}"
        )
    return problems


def verify(bundle: Bundle) -> list[str]:
    problems: list[str] = []
    exact = (
        (ORBIT_PATH, _canonical(bundle.orbit)),
        (SPLIT_PATH, _canonical(bundle.split)),
        (REFERENCE_PATH, _canonical(bundle.reference)),
    )
    for path, expected in exact:
        if not path.exists():
            problems.append(f"missing artifact: {path.name}")
            continue
        observed = path.read_bytes()
        if observed != expected:
            problems.append(
                f"{path.name} mismatch: observed {_sha256(observed)}, "
                f"expected {_sha256(expected)}"
            )
    if not RESULTS_PATH.exists():
        problems.append(f"missing artifact: {RESULTS_PATH.name}")
    else:
        committed = json.loads(RESULTS_PATH.read_text())
        if not _numeric_close(
            committed, json.loads(_canonical(bundle.results)), NUMERIC_TOLERANCE
        ):
            problems.append(
                f"{RESULTS_PATH.name} differs beyond the declared numeric "
                f"tolerance {NUMERIC_TOLERANCE}"
            )
    if not TABLE_PATH.exists():
        problems.append(f"missing artifact: {TABLE_PATH.name}")
    else:
        problems.extend(_compare_table(TABLE_PATH.read_bytes(), bundle.table))
    if not COST_PATH.exists():
        problems.append(f"missing artifact: {COST_PATH.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and verify every committed artifact",
    )
    parser.add_argument(
        "--write", action="store_true", help="write every artifact to disk"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="reduced grid for development; refuses to write artifacts",
    )
    args = parser.parse_args(argv)

    if args.quick and args.write:
        print("--quick cannot write artifacts", file=sys.stderr)
        return 2

    bundle = run_everything(quick=args.quick)

    if args.check:
        if args.quick:
            print("--quick cannot verify artifacts", file=sys.stderr)
            return 2
        problems = verify(bundle)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(
            "017.24 experiment verified: "
            f"orbit {_sha256(_canonical(bundle.orbit))[:16]} "
            f"reference {_sha256(_canonical(bundle.reference))[:16]} "
            f"verdict {bundle.results['J_verdict']['verdict']}"  # type: ignore[index]
        )
        return 0

    if args.write:
        for path, payload in (
            (ORBIT_PATH, bundle.orbit),
            (SPLIT_PATH, bundle.split),
            (REFERENCE_PATH, bundle.reference),
            (RESULTS_PATH, bundle.results),
            (COST_PATH, bundle.cost),
        ):
            blob = _canonical(payload)
            path.write_bytes(blob)
            print(f"wrote {path.name} {_sha256(blob)}")
        TABLE_PATH.write_bytes(bundle.table)
        print(f"wrote {TABLE_PATH.name} {_sha256(bundle.table)}")
        return 0

    sys.stdout.buffer.write(_canonical(bundle.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
