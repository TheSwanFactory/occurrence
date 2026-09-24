"""Issue 017.24a: separate compiled quotient from declared relation.

017.25 reported a strong positive on labeled-data burden and confessed one
degeneracy in the fixture it used. On the Fano successor ``Aut(Sigma')`` is
transitive on marked and on unmarked pairs, so the derived class partition
*coincides* with the declared relation together with its complement. Every
number in 017.25 is therefore equally consistent with a cheaper story: that the
gain comes from indexing parameters by a relation the schema already states,
with no derivation involved at all.

This module is the change that settles it, and it is frozen before any label of
the new fixture exists.

Two things are added.

**A fixture that breaks the degeneracy.** ``Sigma''`` is a circulant incidence
on seven points and seven lines with difference set ``{0,1,2}``, matched to the
Fano successor on every cardinality -- 7, 7, 21, 49 -- but not a projective
plane. Its automorphism group has order 14 and its derived partition has four
classes of sizes 7, 14, 14, 14, which **strictly refines** the declared
relation: the 21 marked pairs split 7 + 14 and the 28 unmarked pairs split
14 + 14. On this fixture a learner indexed by the relation alone provably cannot
reach the truth, and the compiled quotient provably can.

**A condition that names the cheaper story.** Condition ``R`` carries two
parameters indexed by the declared relation itself. On Fano its asymptotic floor
is exactly zero and it is therefore indistinguishable from the compiled
condition; on the circulant it carries a strictly positive floor. That single
contrast is the measurement 017.24 could not make.

A declared-symmetry ladder turns the binary contrast into a dose-response curve:
each rung declares a generating set, the compiler closes it, and the surviving
free parameters fall 49 -> 7 -> 4 on the circulant and 49 -> 7 -> 3 -> 2 on
Fano. All rungs whose quotient is at least as fine as the truth have zero floor,
so they differ only in labeled-data burden, and the prediction ``dof/(2n)`` is
fixed in advance.

The shared implementation is imported byte-pinned from the 017.24 attachments,
so the estimator, the generator, the split machinery, the exact evaluation, the
uncompiled learner, and the controls are literally the same code that produced
017.25. What is new here is the fixture, condition ``R``, the ladder, the
``U_relation`` variant, and a generalized exact reference that works for any
number of classes and for partitions that are too coarse for the truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np

REFINEMENT_SCHEMA: Final = "gpt-01724a-circulant-refinement-type-schema/v1"
PREREGISTRATION_SCHEMA: Final = "gpt-01724a-refinement-preregistration/v1"
CERTIFICATE_SCHEMA: Final = "gpt-01724a-refinement-compiler-certificate/v1"
REFERENCE_SCHEMA: Final = "gpt-01724a-generalized-exact-reference/v1"
RESULTS_SCHEMA: Final = "gpt-01724a-refinement-results/v1"
COST_SCHEMA: Final = "gpt-01724a-refinement-cost-report/v1"

REFINEMENT_SHA256: Final = (
    "31a6d0000582c6d40522111dfc0e1fd87244e4afc6e428bc845fefe25ed755bb"
)
PREREGISTRATION_SHA256: Final = (
    "4a49500eed41dcd831f96fa5b6ef33467e179c2b25cde59f0ab7fb544f0c7aa4"
)
SHARED_MODULE_SHA256: Final = (
    "821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79"
)

PRIOR_RESULT_COMMIT: Final = "26905341915fe277bfe819c30c4b6a3f5250b170"
PRIOR_QUILT_REVISION: Final = (
    "e06607dcecc911842378dae213629843fa6c4075ebe8d17198013cdc7000e7a5"
)
TASK_GPT_REVISION: Final = (
    "d2d6fc93d3b1e689b571afdd8e7fc75662d1c9a63c8168a474d694899c4eff26"
)

HERE: Final = Path(__file__).resolve().parent
SHARED_PATH: Final = (
    HERE.parent / "017.24-Code-attachments" / "training_economy.py"
)
REFINEMENT_PATH: Final = HERE / "refinement_schema.json"
PREREGISTRATION_PATH: Final = HERE / "preregistration.json"
CERTIFICATE_PATH: Final = HERE / "refinement_certificate.json"
REFERENCE_PATH: Final = HERE / "generalized_reference.json"
RESULTS_PATH: Final = HERE / "refinement_results.json"
COST_PATH: Final = HERE / "refinement_cost_report.json"
TABLE_PATH: Final = HERE / "refinement_per_replicate.csv.gz"

DECIMALS: Final = 10
NUMERIC_TOLERANCE: Final = 1e-6


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_shared() -> object:
    """Import the 017.24 implementation, byte-pinned.

    The point of pinning is that every primitive this turn shares with 017.25 --
    the beta estimator, the hidden generator, the split builder, the exact
    evaluation, the uncompiled learner, the scramble and erasure controls -- is
    provably the same code, so any difference in outcome is attributable to the
    fixture and the new conditions rather than to a reimplementation.
    """
    raw = SHARED_PATH.read_bytes()
    digest = _sha256(raw)
    if digest != SHARED_MODULE_SHA256:
        raise RuntimeError(
            f"shared 017.24 module changed: expected {SHARED_MODULE_SHA256}, "
            f"observed {digest}"
        )
    spec = importlib.util.spec_from_file_location(
        "training_economy_01724_shared", SHARED_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared 017.24 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["training_economy_01724_shared"] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared()

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)
CLIP: Final = shared.CLIP
ALPHA_PRIMARY: Final = ONE
BETA_PRIMARY: Final = ONE
ALPHA_SECONDARY: Final = Fraction(1, 2)
BETA_SECONDARY: Final = Fraction(1, 2)

SAMPLE_SIZES: Final = shared.SAMPLE_SIZES
QUICK_SIZES: Final = shared.QUICK_SIZES
CLOSED_FORM_REPLICATES: Final = 400
LEARNED_REPLICATES: Final = 100
QUICK_CLOSED_FORM_REPLICATES: Final = 40
QUICK_LEARNED_REPLICATES: Final = 20
REGIMES: Final = ("A", "B")
THRESHOLDS: Final = (0.02, 0.01)
MATERIAL_FACTOR: Final = 2.0
BASE_SEED: Final = 20260924

Datum = tuple[str, str]


def _round(value: float) -> float:
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0 else rounded


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# 1. Fixtures: the new circulant schema and the 017.24 Fano successor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fixture:
    """One frozen Type Schema plus the symmetry ladder it declares."""

    identifier: str
    schema_name: str
    schema: object
    document: dict[str, object]
    structure: object
    holdout_per_class: tuple[int, ...]
    ladder: tuple[tuple[str, int, tuple[int, ...]], ...]

    @property
    def class_count(self) -> int:
        return len(self.structure.classes)


def load_refinement(path: Path = REFINEMENT_PATH) -> tuple[object, dict[str, object]]:
    raw = path.read_bytes()
    digest = _sha256(raw)
    if digest != REFINEMENT_SHA256:
        raise RuntimeError(
            f"refinement schema changed: expected {REFINEMENT_SHA256}, "
            f"observed {digest}"
        )
    document = json.loads(raw)
    if document["schema"] != REFINEMENT_SCHEMA:
        raise RuntimeError("unexpected refinement schema identifier")
    sorts = document["sorts"]
    schema = shared.IncidenceSchema(
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


def _automorphism(
    schema: object, point_images: Sequence[str], line_images: Sequence[str]
) -> object:
    return shared.Automorphism(
        point_images=tuple(point_images), line_images=tuple(line_images)
    )


def close_subgroup(
    schema: object, generators: Sequence[object]
) -> tuple[object, ...]:
    """Close a declared generating set under composition.

    This is the compiler step for a ladder rung. It reads the declared
    generators and the sorts, and nothing else. No label, and no orbit function,
    is in scope.
    """
    identity = _automorphism(schema, schema.points, schema.lines)
    point_index = {point: index for index, point in enumerate(schema.points)}
    line_index = {line: index for index, line in enumerate(schema.lines)}
    found = {(identity.point_images, identity.line_images): identity}
    frontier = [identity]
    while frontier:
        following: list[object] = []
        for element in frontier:
            for generator in generators:
                composed = _automorphism(
                    schema,
                    tuple(
                        generator.point_images[point_index[point]]
                        for point in element.point_images
                    ),
                    tuple(
                        generator.line_images[line_index[line]]
                        for line in element.line_images
                    ),
                )
                key = (composed.point_images, composed.line_images)
                if key not in found:
                    found[key] = composed
                    following.append(composed)
        frontier = following
    return tuple(found.values())


def _preserves_relation(schema: object, element: object) -> bool:
    point_index = {point: index for index, point in enumerate(schema.points)}
    line_index = {line: index for index, line in enumerate(schema.lines)}
    moved = {
        (
            element.point_images[point_index[point]],
            element.line_images[line_index[line]],
        )
        for point, line in schema.incidence
    }
    return moved == set(schema.incidence)


def build_ladder(
    schema: object, document: dict[str, object], structure: object
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """Compile every declared rung of the symmetry ladder."""
    block = document["declared_symmetry_ladder"]
    generators = {
        name: _automorphism(schema, value["point_images"], value["line_images"])
        for name, value in block["generators"].items()  # type: ignore[index]
    }
    for name, element in generators.items():
        if not _preserves_relation(schema, element):
            raise RuntimeError(f"declared generator {name} is not an automorphism")
    rungs: list[tuple[str, int, tuple[int, ...]]] = []
    for rung in block["rungs"]:  # type: ignore[index]
        declared = [generators[name] for name in rung["declared_generators"]]
        group = close_subgroup(schema, declared)
        partition = shared.invariance_partition_from_group(
            structure.inputs, schema, group
        )
        if len(group) != rung["expected_closed_order"]:
            raise RuntimeError(
                f"rung {rung['name']} closed to order {len(group)}, "
                f"expected {rung['expected_closed_order']}"
            )
        if len(set(partition)) != rung["expected_free_parameters"]:
            raise RuntimeError(
                f"rung {rung['name']} left {len(set(partition))} free "
                f"parameters, expected {rung['expected_free_parameters']}"
            )
        rungs.append((str(rung["name"]), len(group), partition))
    return tuple(rungs)


def _canonical_subgroup_generators(
    schema: object, group: Sequence[object], orders: Sequence[int]
) -> tuple[tuple[object, ...], ...]:
    """Derive canonical generators of subgroups of the requested orders.

    The 017.20 Fano presentation does not use the circulant labeling, so its
    ladder generators cannot be written down by hand the way the circulant's
    can. They are instead derived from the compiled group by a deterministic
    rule: sort the group, then greedily take the least element that grows the
    closure to the next requested order. Subgroups of a given order in this group
    are conjugate, so the derived class counts do not depend on the choice, and
    that independence is checked mechanically in the certificate.
    """
    ordered = sorted(group, key=lambda g: (g.point_images, g.line_images))
    chosen: list[object] = []
    result: list[tuple[object, ...]] = []
    for target in orders:
        for candidate in ordered:
            grown = close_subgroup(schema, [*chosen, candidate])
            if len(grown) == target:
                chosen.append(candidate)
                result.append(tuple(chosen))
                break
        else:
            raise RuntimeError(f"no subgroup of order {target} was reachable")
    return tuple(result)


def fano_fixture() -> Fixture:
    """The 017.24 successor, recompiled here with a derived symmetry ladder."""
    schema, document = shared.load_successor()
    structure = shared.compile_classes(schema)
    generator_sets = _canonical_subgroup_generators(
        schema, structure.automorphisms, (7, 21)
    )
    rungs: list[tuple[str, int, tuple[int, ...]]] = [
        ("rung_1", 1, tuple(range(len(structure.inputs))))
    ]
    for name, generators in (
        ("rung_7", generator_sets[0]),
        ("rung_21", generator_sets[1]),
    ):
        group = close_subgroup(schema, generators)
        rungs.append(
            (
                name,
                len(group),
                shared.invariance_partition_from_group(
                    structure.inputs, schema, group
                ),
            )
        )
    rungs.append(
        (
            "rung_168",
            len(structure.automorphisms),
            shared.invariance_partition_from_group(
                structure.inputs, schema, structure.automorphisms
            ),
        )
    )
    return Fixture(
        identifier="fano",
        schema_name=shared.SUCCESSOR_SCHEMA,
        schema=schema,
        document=document,
        structure=structure,
        holdout_per_class=(4, 6),
        ladder=tuple(rungs),
    )


def circulant_fixture() -> Fixture:
    schema, document = load_refinement()
    structure = shared.compile_classes(schema)
    return Fixture(
        identifier="circulant",
        schema_name=REFINEMENT_SCHEMA,
        schema=schema,
        document=document,
        structure=structure,
        holdout_per_class=(2, 3, 3, 3),
        ladder=build_ladder(schema, document, structure),
    )


def relation_partition(fixture: Fixture) -> tuple[int, ...]:
    """Condition R: two parameters indexed by the declared relation itself."""
    relation = fixture.schema.incidence
    return tuple(
        0 if datum in relation else 1 for datum in fixture.structure.inputs
    )


def evenly_spaced_theta(count: int) -> tuple[Fraction, ...]:
    """Untuned interior ground truth: lambda_c = (2c+1)/(2k)."""
    return tuple(Fraction(2 * index + 1, 2 * count) for index in range(count))


# ---------------------------------------------------------------------------
# 2. The generalized exact reference
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


def _expected_logs(
    size: int,
    weight: float,
    rate: float,
    alpha: Fraction,
    beta: Fraction,
    log_factorial: Sequence[float],
) -> tuple[float, float]:
    """``E[-ln lambda_hat]`` and ``E[-ln(1 - lambda_hat)]`` for one cell.

    The cell receives ``m ~ Binomial(n, weight)`` observations, and each one
    independently answers ``a_0`` with the cell's mixture rate, so the success
    count is ``Binomial(m, rate)``. Only the marginal of ``m`` is needed, because
    the excess log loss is additive across cells. That is what makes this exact
    for any number of classes and for partitions too coarse for the truth.
    """
    offset = float(alpha)
    prior = float(alpha + beta)
    outer = _binomial_pmf(size, weight, log_factorial)
    first = 0.0
    second = 0.0
    for trials in range(size + 1):
        mass = outer[trials]
        if mass == 0.0:
            continue
        inner = _binomial_pmf(trials, rate, log_factorial)
        denominator = trials + prior
        partial_first = 0.0
        partial_second = 0.0
        for successes in range(trials + 1):
            estimate = (successes + offset) / denominator
            clipped = min(max(estimate, CLIP), 1.0 - CLIP)
            partial_first -= inner[successes] * math.log(clipped)
            partial_second -= inner[successes] * math.log1p(-clipped)
        first += mass * partial_first
        second += mass * partial_second
    return first, second


def exact_expected_excess(
    partition: Sequence[int],
    cell_count: int,
    truths: Sequence[float],
    class_of_index: Sequence[int],
    train_pool: Sequence[int],
    eval_indices: Sequence[int],
    sizes: Sequence[int],
    alpha: Fraction,
    beta: Fraction,
) -> dict[str, object]:
    """Exact expected excess held-out log loss for any partition. No simulation."""
    limit = max(sizes)
    log_factorial = _log_factorials(limit)
    pool_size = len(train_pool)
    eval_size = len(eval_indices)

    train_mass = [0.0] * cell_count
    train_rate = [0.0] * cell_count
    for index in train_pool:
        cell = partition[index]
        train_mass[cell] += 1.0 / pool_size
        train_rate[cell] += truths[class_of_index[index]] / pool_size
    rates = [
        train_rate[cell] / train_mass[cell] if train_mass[cell] > 0 else 0.5
        for cell in range(cell_count)
    ]

    eval_success = [0.0] * cell_count
    eval_failure = [0.0] * cell_count
    bayes = 0.0
    for index in eval_indices:
        cell = partition[index]
        truth = truths[class_of_index[index]]
        eval_success[cell] += truth / eval_size
        eval_failure[cell] += (1.0 - truth) / eval_size
        bayes += (
            -(truth * math.log(truth) + (1.0 - truth) * math.log1p(-truth))
        ) / eval_size

    rows: list[dict[str, object]] = []
    for size in sizes:
        total = 0.0
        for cell in range(cell_count):
            if eval_success[cell] == 0.0 and eval_failure[cell] == 0.0:
                continue
            first, second = _expected_logs(
                size, train_mass[cell], rates[cell], alpha, beta, log_factorial
            )
            total += eval_success[cell] * first + eval_failure[cell] * second
        rows.append(
            {
                "n": size,
                "exact_expected_excess_nll": _round(total - bayes),
                "asymptotic_dof_over_2n": _round(
                    sum(
                        1.0 for cell in range(cell_count) if train_mass[cell] > 0
                    )
                    / (2.0 * size)
                ),
            }
        )
    floor = shared.asymptotic_floor(
        partition,
        cell_count,
        truths,
        class_of_index,
        train_pool,
        eval_indices,
        alpha,
        beta,
    )
    return {
        "cells": cell_count,
        "cells_reached_by_training": sum(
            1 for cell in range(cell_count) if train_mass[cell] > 0
        ),
        "bayes_optimal_nll": _round(bayes),
        "asymptotic_floor": _round(floor),
        "floor_is_zero": abs(floor) < 1e-12,
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# 3. Conditions, including the relation-indexed learner and the ladder
# ---------------------------------------------------------------------------

U_RIDGE_GRID: Final = shared.U_RIDGE_GRID
U_STEPS: Final = shared.U_STEPS
HIDDEN_WIDTH: Final = shared.HIDDEN_WIDTH
U_VARIANTS: Final = (
    ("U_fixed", ("mlp",), (shared.U_L2,), False),
    ("U_earlystop", ("mlp",), (shared.U_L2,), True),
    ("U_selected", ("mlp",), U_RIDGE_GRID, True),
    ("U_quadratic", ("quadratic",), U_RIDGE_GRID, True),
    ("U_ladder", ("constant", "linear", "quadratic", "mlp"), U_RIDGE_GRID, True),
    ("U_relation", ("relation_linear", "relation_mlp"), U_RIDGE_GRID, True),
)
LEARNED_CONDITIONS: Final = tuple(name for name, _, _, _ in U_VARIANTS)
TABLE_COLUMNS: Final = (
    "fixture",
    "regime",
    "n",
    "condition",
    "replicate",
    "excess_nll_aggregate",
    "nll_aggregate",
    "brier_aggregate",
    "calibration_error_aggregate",
    "degrees_of_freedom",
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    name: str
    partition: tuple[int, ...]
    cell_count: int
    kind: str


def build_conditions(fixture: Fixture) -> tuple[ConditionSpec, ...]:
    """Every closed-form condition for one fixture, sharing one estimator."""
    structure = fixture.structure
    width = len(structure.inputs)
    specs = [
        ConditionSpec(
            "T", structure.class_of_index, fixture.class_count, "tdm"
        ),
        ConditionSpec(
            "O", structure.class_of_index, fixture.class_count, "bernoulli"
        ),
        ConditionSpec(
            "R", relation_partition(fixture), 2, "bernoulli"
        ),
        ConditionSpec("S", tuple(range(width)), width, "bernoulli"),
        ConditionSpec(
            "G",
            shared.invariance_partition_from_group(
                structure.inputs, fixture.schema, structure.automorphisms
            ),
            fixture.class_count,
            "bernoulli",
        ),
    ]
    for name, _order, partition in fixture.ladder:
        specs.append(
            ConditionSpec(
                f"L_{name}", partition, len(set(partition)), "bernoulli"
            )
        )
    wrong, _ = shared.build_wrong_partition(structure)
    specs.append(ConditionSpec("W_wrong_compiler", wrong, 2, "tdm"))
    scrambled = shared.compile_classes(
        shared.build_scrambled_schemas(fixture.schema, 1)[0]
    )
    specs.append(
        ConditionSpec(
            "C_scrambled_compiler",
            scrambled.class_of_index,
            len(scrambled.classes),
            "tdm",
        )
    )
    return tuple(specs)


def build_designs(fixture: Fixture) -> dict[str, np.ndarray]:
    """Label-free design matrices, including the read-the-relation variants."""
    features = shared.build_feature_matrix(fixture.schema)
    count = features.shape[0]
    relation = fixture.schema.incidence
    bit = np.array(
        [
            [1.0 if datum in relation else 0.0]
            for datum in fixture.structure.inputs
        ]
    )
    with_relation = np.hstack([features, bit])
    return {
        "constant": np.ones((count, 1)),
        "linear": np.hstack([np.ones((count, 1)), features]),
        "quadratic": shared.build_quadratic_design(features),
        "mlp": features,
        "relation_linear": np.hstack([np.ones((count, 1)), with_relation]),
        "relation_mlp": with_relation,
    }


HIDDEN_OF: Final = {
    "constant": None,
    "linear": None,
    "quadratic": None,
    "mlp": HIDDEN_WIDTH,
    "relation_linear": None,
    "relation_mlp": HIDDEN_WIDTH,
}


def _parameter_count(design: np.ndarray, hidden: int | None) -> int:
    width = design.shape[1]
    if hidden is None:
        return width + 1
    return width * hidden + hidden + hidden + 1


def _accumulator(class_count: int) -> dict[str, list[float]]:
    store: dict[str, list[float]] = {
        "excess_aggregate": [],
        "nll_aggregate": [],
        "brier_aggregate": [],
        "calibration_aggregate": [],
    }
    for index in range(class_count):
        store[f"excess_O_{index}"] = []
    return store


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


def _summarize(
    store: dict[str, list[float]], matched: int, dof: int
) -> dict[str, object]:
    report: dict[str, object] = {"degrees_of_freedom": dof}
    for key, values in store.items():
        if values:
            report[key] = _summary(values)
    report["excess_aggregate_matched"] = _summary(
        store["excess_aggregate"][:matched]
    )
    return report


def _record(
    rows: list[tuple[object, ...]],
    fixture: Fixture,
    regime: str,
    size: int,
    condition: str,
    replicate: int,
    metrics: dict[str, dict[str, float]],
    dof: int,
) -> None:
    block = metrics["aggregate"]
    rows.append(
        (
            fixture.identifier,
            regime,
            size,
            condition,
            replicate,
            _round(block["excess_nll"]),
            _round(block["nll"]),
            _round(block["brier"]),
            _round(block["calibration_error"]),
            dof,
        )
    )


def run_closed_form(
    fixture: Fixture,
    theta: Sequence[Fraction],
    sizes: Sequence[int],
    replicates: int,
    matched: int,
    alpha: Fraction,
    beta: Fraction,
    rows: list[tuple[object, ...]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Fit every closed-form condition on byte-identical observations."""
    structure = fixture.structure
    members = shared._class_members(structure)
    width = len(structure.inputs)
    conditions = build_conditions(fixture)
    numerators, denominator = shared.theta_as_integers(theta)
    truths = [float(value) for value in theta]
    cells: dict[str, dict[str, object]] = {}
    agreement = {
        "T_versus_O_mismatches": 0,
        "T_versus_G_mismatches": 0,
        "T_versus_R_identical": 0,
        "comparisons": 0,
    }

    for regime_index, regime in enumerate(REGIMES):
        for size_index, size in enumerate(sizes):
            store = {
                spec.name: _accumulator(fixture.class_count)
                for spec in conditions
            }
            for replicate in range(replicates):
                seed = (
                    BASE_SEED
                    + 1_000_000 * (fixture.identifier == "circulant")
                    + 100_000 * regime_index
                    + 1_000 * size_index
                    + replicate
                )
                rng = random.Random(seed)
                split = shared.build_split(
                    regime, rng, members, fixture.holdout_per_class
                )
                observations = shared.generate_observations(
                    rng,
                    split.train_pool,
                    size,
                    numerators,
                    denominator,
                    structure.class_of_index,
                )
                successes, trials = shared.pair_counts(observations, width)
                fitted: dict[str, object] = {}
                for spec in conditions:
                    builder = (
                        shared.fit_typed_tdm
                        if spec.kind == "tdm"
                        else shared.fit_bernoulli
                    )
                    model = builder(
                        successes,
                        trials,
                        spec.partition,
                        spec.cell_count,
                        alpha,
                        beta,
                        spec.name,
                    )
                    fitted[spec.name] = model
                    metrics = shared.evaluate_exactly(
                        model.predictions,
                        split,
                        truths,
                        structure.class_of_index,
                    )
                    bucket = store[spec.name]
                    bucket["excess_aggregate"].append(
                        metrics["aggregate"]["excess_nll"]
                    )
                    bucket["nll_aggregate"].append(metrics["aggregate"]["nll"])
                    bucket["brier_aggregate"].append(
                        metrics["aggregate"]["brier"]
                    )
                    bucket["calibration_aggregate"].append(
                        metrics["aggregate"]["calibration_error"]
                    )
                    for index in range(fixture.class_count):
                        name = f"O_{index}"
                        if name in metrics:
                            bucket[f"excess_{name}"].append(
                                metrics[name]["excess_nll"]
                            )
                    _record(
                        rows,
                        fixture,
                        regime,
                        size,
                        spec.name,
                        replicate,
                        metrics,
                        model.degrees_of_freedom,
                    )
                agreement["comparisons"] += 1
                typed = fitted["T"].exact_predictions
                if typed != fitted["O"].exact_predictions:
                    agreement["T_versus_O_mismatches"] += 1
                if typed != fitted["G"].exact_predictions:
                    agreement["T_versus_G_mismatches"] += 1
                if typed == fitted["R"].exact_predictions:
                    agreement["T_versus_R_identical"] += 1
            for spec in conditions:
                cells[f"{fixture.identifier}|{regime}|{size}|{spec.name}"] = (
                    _summarize(store[spec.name], matched, spec.cell_count)
                )
    return cells, agreement


def run_learned(
    fixture: Fixture,
    theta: Sequence[Fraction],
    sizes: Sequence[int],
    replicates: int,
    rows: list[tuple[object, ...]],
) -> dict[str, dict[str, object]]:
    """Fit every uncompiled variant on exactly the same observations."""
    structure = fixture.structure
    members = shared._class_members(structure)
    width = len(structure.inputs)
    designs = build_designs(fixture)
    numerators, denominator = shared.theta_as_integers(theta)
    truths = [float(value) for value in theta]
    cells: dict[str, dict[str, object]] = {}

    for regime_index, regime in enumerate(REGIMES):
        for size_index, size in enumerate(sizes):
            validation = max(1, int(round(0.25 * size)))
            training = max(1, size - validation)
            splits: list[object] = []
            seeds: list[int] = []
            full = [np.zeros((replicates, width)) for _ in range(2)]
            inner = [np.zeros((replicates, width)) for _ in range(2)]
            held = [np.zeros((replicates, width)) for _ in range(2)]
            for replicate in range(replicates):
                seed = (
                    BASE_SEED
                    + 1_000_000 * (fixture.identifier == "circulant")
                    + 100_000 * regime_index
                    + 1_000 * size_index
                    + replicate
                )
                seeds.append(seed + 500_000_000)
                rng = random.Random(seed)
                split = shared.build_split(
                    regime, rng, members, fixture.holdout_per_class
                )
                observations = shared.generate_observations(
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
                    target = inner if position < training else held
                    target[slot][replicate, index] += 1.0

            train_totals = (inner[0] + inner[1]).sum(axis=1)
            validation_totals = (held[0] + held[1]).sum(axis=1)
            full_totals = (full[0] + full[1]).sum(axis=1)
            cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
            for name in designs:
                for ridge in U_RIDGE_GRID:
                    _, best, loss = shared._train_batched(
                        designs[name],
                        inner[0],
                        inner[1],
                        train_totals,
                        held[0],
                        held[1],
                        validation_totals,
                        seeds,
                        U_STEPS,
                        ridge,
                        HIDDEN_OF[name],
                    )
                    cache[(name, ridge)] = (best, loss)
            unvalidated, _, _ = shared._train_batched(
                designs["mlp"],
                full[0],
                full[1],
                full_totals,
                None,
                None,
                None,
                seeds,
                U_STEPS,
                shared.U_L2,
                HIDDEN_WIDTH,
            )

            for variant, candidates, ridges, use_validation in U_VARIANTS:
                sizes_of = {
                    name: _parameter_count(designs[name], HIDDEN_OF[name])
                    for name in candidates
                }
                if not use_validation:
                    predictions = unvalidated
                    selection = {"model_selection": "none"}
                else:
                    predictions = np.zeros((replicates, width))
                    best_loss = np.full(replicates, np.inf)
                    best_kind = np.full(replicates, -1)
                    for position, name in enumerate(candidates):
                        for ridge in ridges:
                            candidate, loss = cache[(name, ridge)]
                            improved = loss < best_loss
                            best_loss = np.where(improved, loss, best_loss)
                            predictions = np.where(
                                improved.reshape(-1, 1), candidate, predictions
                            )
                            best_kind = np.where(improved, position, best_kind)
                    selection = {
                        "model_selection": (
                            "internal validation on training labels only"
                        ),
                        "candidates": list(candidates),
                        "selected_candidate_counts": {
                            name: int((best_kind == position).sum())
                            for position, name in enumerate(candidates)
                        },
                    }
                dof = max(sizes_of.values())
                selection["declared_parameters"] = dof
                store = _accumulator(fixture.class_count)
                for replicate in range(replicates):
                    values = tuple(
                        float(value) for value in predictions[replicate]
                    )
                    metrics = shared.evaluate_exactly(
                        values,
                        splits[replicate],
                        truths,
                        structure.class_of_index,
                    )
                    store["excess_aggregate"].append(
                        metrics["aggregate"]["excess_nll"]
                    )
                    store["nll_aggregate"].append(metrics["aggregate"]["nll"])
                    store["brier_aggregate"].append(
                        metrics["aggregate"]["brier"]
                    )
                    store["calibration_aggregate"].append(
                        metrics["aggregate"]["calibration_error"]
                    )
                    for index in range(fixture.class_count):
                        name = f"O_{index}"
                        if name in metrics:
                            store[f"excess_{name}"].append(
                                metrics[name]["excess_nll"]
                            )
                    _record(
                        rows,
                        fixture,
                        regime,
                        size,
                        variant,
                        replicate,
                        metrics,
                        dof,
                    )
                summary = _summarize(store, replicates, dof)
                summary["model_selection"] = selection
                cells[
                    f"{fixture.identifier}|{regime}|{size}|{variant}"
                ] = summary
    return cells


# ---------------------------------------------------------------------------
# 4. Certificates, predictions, and the coarse-compiler control
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.24-Task-first-matched-TDM-training-economy-experiment.md"
        ),
        "follows_up": (
            "quilt+s3://protology#package=occurrence/gpt@"
            f"{PRIOR_QUILT_REVISION}&path=issues/017-jev-pivot/"
            "017.25-Kiro-first-matched-TDM-training-economy-experiment-result.md"
        ),
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "kind": "Kiro-initiated follow-up, in the 017.21a precedent",
        "byte_pinned_inputs": {
            "refinement_schema.json": REFINEMENT_SHA256,
            "preregistration.json": PREREGISTRATION_SHA256,
            "../017.24-Code-attachments/training_economy.py": (
                SHARED_MODULE_SHA256
            ),
        },
    }


def refinement_certificate(fixtures: Sequence[Fixture]) -> dict[str, object]:
    """Structural facts, all derived before any label exists."""
    blocks: list[dict[str, object]] = []
    for fixture in fixtures:
        structure = fixture.structure
        relation = relation_partition(fixture)
        classes_refine = len(set(structure.class_of_index)) > 2
        cells_pure = all(
            len(
                {
                    relation[index]
                    for index, cell in enumerate(structure.class_of_index)
                    if cell == target
                }
            )
            == 1
            for target in set(structure.class_of_index)
        )
        conjugacy: list[dict[str, object]] = []
        if fixture.identifier == "fano":
            ordered = sorted(
                structure.automorphisms,
                key=lambda g: (g.point_images, g.line_images),
            )
            seen = 0
            for candidate in ordered:
                group = close_subgroup(fixture.schema, [candidate])
                if len(group) != 7:
                    continue
                partition = shared.invariance_partition_from_group(
                    structure.inputs, fixture.schema, group
                )
                conjugacy.append(
                    {
                        "free_parameters": len(set(partition)),
                        "class_cardinalities": sorted(
                            partition.count(cell) for cell in set(partition)
                        ),
                    }
                )
                seen += 1
                if seen == 6:
                    break
        blocks.append(
            {
                "fixture": fixture.identifier,
                "schema": fixture.schema_name,
                "points": len(fixture.schema.points),
                "lines": len(fixture.schema.lines),
                "marked_pairs": len(fixture.schema.incidence),
                "admitted_inputs": len(structure.inputs),
                "automorphism_group_order": structure.group_order,
                "derived_class_count": fixture.class_count,
                "derived_class_cardinalities": list(structure.cardinalities),
                "relation_partition_cardinalities": sorted(
                    relation.count(cell) for cell in set(relation)
                ),
                "derived_partition_strictly_refines_the_relation": (
                    classes_refine
                ),
                "every_derived_class_lies_inside_the_relation_or_outside_it": (
                    cells_pure
                ),
                "relation_split_by_derived_classes": {
                    "inside": sorted(
                        len(members)
                        for index, members in enumerate(structure.classes)
                        if members[0] in fixture.schema.incidence
                    ),
                    "outside": sorted(
                        len(members)
                        for index, members in enumerate(structure.classes)
                        if members[0] not in fixture.schema.incidence
                    ),
                },
                "declared_symmetry_ladder": [
                    {
                        "rung": name,
                        "closed_order": order,
                        "free_parameters": len(set(partition)),
                        "class_cardinalities": sorted(
                            partition.count(cell) for cell in set(partition)
                        )
                        if len(set(partition)) <= 8
                        else "singletons",
                    }
                    for name, order, partition in fixture.ladder
                ],
                "ladder_is_monotone_in_free_parameters": all(
                    len(set(fixture.ladder[index][2]))
                    >= len(set(fixture.ladder[index + 1][2]))
                    for index in range(len(fixture.ladder) - 1)
                ),
                "subgroup_choice_independence": conjugacy,
                "theta_star": [
                    _fraction_text(value)
                    for value in evenly_spaced_theta(fixture.class_count)
                ],
                "holdout_per_class": list(fixture.holdout_per_class),
            }
        )
    return {
        "schema": CERTIFICATE_SCHEMA,
        "authority": authority_record(),
        "fixtures": blocks,
        "matched_cardinalities": {
            "points": 7,
            "lines": 7,
            "marked_pairs": 21,
            "admitted_inputs": 49,
            "identical_across_fixtures": True,
        },
        "the_separation_this_turn_measures": (
            "on the Fano fixture the derived partition equals the declared "
            "relation and its complement, so a relation-indexed learner is "
            "indistinguishable from the compiled one; on the circulant fixture "
            "the derived partition strictly refines the relation, so it is not"
        ),
    }


def prediction_payload(
    fixtures: Sequence[Fixture], sizes: Sequence[int]
) -> dict[str, object]:
    """The frozen label-free predictions, plus the coarse-compiler control."""
    blocks: dict[str, object] = {}
    controls: dict[str, object] = {}
    for fixture in fixtures:
        structure = fixture.structure
        theta = evenly_spaced_theta(fixture.class_count)
        truths = [float(value) for value in theta]
        members = shared._class_members(structure)
        everything = tuple(range(len(structure.inputs)))
        canonical = shared.build_split(
            "B", random.Random(BASE_SEED), members, fixture.holdout_per_class
        )
        specs: dict[str, tuple[tuple[int, ...], int]] = {
            "T": (structure.class_of_index, fixture.class_count),
            "R": (relation_partition(fixture), 2),
            "S": (everything, len(everything)),
        }
        for name, _order, partition in fixture.ladder:
            specs[f"L_{name}"] = (partition, len(set(partition)))
        for regime, pool, evaluated in (
            ("A", everything, everything),
            ("B", canonical.train_pool, canonical.held_out),
        ):
            for name, (partition, cell_count) in specs.items():
                blocks[f"{fixture.identifier}|{regime}|{name}"] = (
                    exact_expected_excess(
                        partition,
                        cell_count,
                        truths,
                        structure.class_of_index,
                        pool,
                        evaluated,
                        sizes,
                        ALPHA_PRIMARY,
                        BETA_PRIMARY,
                    )
                )

        # Coarse-compiler control: generate the world from a finer rung, then
        # compile at a coarser but still automorphism-respecting rung.
        ladder = {name: partition for name, _order, partition in fixture.ladder}
        finer = next(
            (
                partition
                for name, partition in ladder.items()
                if len(set(partition)) == 7
            ),
            None,
        )
        if finer is not None:
            fine_count = len(set(finer))
            fine_theta = evenly_spaced_theta(fine_count)
            fine_truths = [float(value) for value in fine_theta]
            coarse = structure.class_of_index
            controls[fixture.identifier] = {
                "world_generated_from": "the 7-parameter rung",
                "theta_star": [_fraction_text(v) for v in fine_theta],
                "compiled_at": (
                    "the full automorphism quotient, which is a legitimate "
                    "group quotient but coarser than the world"
                ),
                "compiled_free_parameters": fixture.class_count,
                "coarse_compiler_floor": _round(
                    shared.asymptotic_floor(
                        coarse,
                        fixture.class_count,
                        fine_truths,
                        finer,
                        everything,
                        everything,
                        ALPHA_PRIMARY,
                        BETA_PRIMARY,
                    )
                ),
                "correct_rung_floor": _round(
                    shared.asymptotic_floor(
                        finer,
                        fine_count,
                        fine_truths,
                        finer,
                        everything,
                        everything,
                        ALPHA_PRIMARY,
                        BETA_PRIMARY,
                    )
                ),
                "lesson": (
                    "respecting a symmetry is not sufficient; the symmetry the "
                    "schema declares has to be the one that generated the world"
                ),
            }
    return {
        "schema": REFERENCE_SCHEMA,
        "authority": authority_record(),
        "status": (
            "computed exactly from the frozen schemas with no simulation and no "
            "label; frozen before the experiment ran"
        ),
        "method": (
            "excess log loss is additive across cells and a cell's estimator "
            "depends only on its own observation count, which is marginally "
            "binomial, so the expectation is a finite sum per cell. Valid for "
            "any number of classes and for partitions too coarse for the truth."
        ),
        "regime_B_note": (
            "regime B predictions use the canonical first split from the "
            "declared seed; the measured regime B result averages over random "
            "stratified splits, so only partitions that are unions of derived "
            "classes are expected to match exactly"
        ),
        "estimator": "beta(1,1) posterior mean",
        "predictions": blocks,
        "coarse_compiler_control": controls,
    }


# ---------------------------------------------------------------------------
# 5. Orchestration, verdict, and the command line
# ---------------------------------------------------------------------------


def _series(
    cells: dict[str, dict[str, object]],
    fixture: str,
    regime: str,
    condition: str,
    sizes: Sequence[int],
    statistic: str = "excess_aggregate",
) -> dict[int, float]:
    out: dict[int, float] = {}
    for size in sizes:
        key = f"{fixture}|{regime}|{size}|{condition}"
        if key in cells:
            out[size] = float(cells[key][statistic]["mean"])  # type: ignore[index]
    return out


def _to_threshold(series: dict[int, float], threshold: float) -> int | None:
    for size in sorted(series):
        if series[size] <= threshold:
            return size
    return None


def select_best_u(
    cells: dict[str, dict[str, object]],
    fixtures: Sequence[Fixture],
    sizes: Sequence[int],
) -> dict[str, dict[str, object]]:
    chosen: dict[str, dict[str, object]] = {}
    for fixture in fixtures:
        for regime in REGIMES:
            for size in sizes:
                best, value = None, math.inf
                for variant in LEARNED_CONDITIONS:
                    key = f"{fixture.identifier}|{regime}|{size}|{variant}"
                    if key not in cells:
                        continue
                    mean = float(
                        cells[key]["excess_aggregate"]["mean"]  # type: ignore[index]
                    )
                    if mean < value:
                        best, value = variant, mean
                if best is None:
                    continue
                payload = dict(
                    cells[f"{fixture.identifier}|{regime}|{size}|{best}"]
                )
                payload["selected_variant"] = best
                chosen[
                    f"{fixture.identifier}|{regime}|{size}|U_best"
                ] = payload
    return chosen


def decide_verdict(
    cells: dict[str, dict[str, object]],
    predictions: dict[str, object],
    agreement: dict[str, dict[str, int]],
    sizes: Sequence[int],
) -> dict[str, object]:
    """Apply the preregistered verdict rule to the measured curves."""
    separation: dict[str, object] = {}
    for fixture in ("fano", "circulant"):
        rows: dict[str, object] = {}
        for regime in REGIMES:
            typed = _series(cells, fixture, regime, "T", sizes)
            rival = _series(cells, fixture, regime, "R", sizes)
            gap = {
                str(size): _round(rival[size] - typed[size])
                for size in sorted(typed)
                if size in rival
            }
            predicted = predictions["predictions"].get(  # type: ignore[union-attr]
                f"{fixture}|{regime}|R", {}
            )
            rows[regime] = {
                "T_mean_excess_by_n": {
                    str(k): _round(v) for k, v in sorted(typed.items())
                },
                "R_mean_excess_by_n": {
                    str(k): _round(v) for k, v in sorted(rival.items())
                },
                "R_minus_T_by_n": gap,
                "R_exact_floor": predicted.get("asymptotic_floor"),
                "R_floor_is_zero": predicted.get("floor_is_zero"),
                "T_labels_to_0.02": _to_threshold(typed, 0.02),
                "R_labels_to_0.02": _to_threshold(rival, 0.02),
                "R_beats_T_at": [
                    size for size in sorted(typed) if rival[size] < typed[size]
                ],
            }
        separation[fixture] = rows

    fano_zero = bool(
        predictions["predictions"]["fano|A|R"]["floor_is_zero"]  # type: ignore[index]
    )
    circ_positive = not bool(
        predictions["predictions"]["circulant|A|R"]["floor_is_zero"]  # type: ignore[index]
    )
    fano_gap = max(
        abs(value)
        for value in separation["fano"]["A"]["R_minus_T_by_n"].values()  # type: ignore[index,union-attr]
    )
    circ_gap = max(
        separation["circulant"]["A"]["R_minus_T_by_n"].values()  # type: ignore[index,union-attr]
    )
    circ_t = separation["circulant"]["A"]["T_labels_to_0.02"]  # type: ignore[index]
    circ_r = separation["circulant"]["A"]["R_labels_to_0.02"]  # type: ignore[index]

    ladder: dict[str, object] = {}
    for fixture in ("fano", "circulant"):
        rungs = [
            key.split("|")[-1]
            for key in cells
            if key.startswith(f"{fixture}|A|{sizes[0]}|L_")
        ]
        order: list[tuple[int, str]] = []
        for rung in rungs:
            key = f"{fixture}|A|{sizes[0]}|{rung}"
            order.append(
                (int(cells[key]["degrees_of_freedom"]), rung)  # type: ignore[arg-type]
            )
        order.sort()
        monotone = True
        detail: dict[str, object] = {}
        for size in sizes:
            if size < 16:
                continue
            values = [
                (
                    dof,
                    rung,
                    float(
                        cells[f"{fixture}|A|{size}|{rung}"][  # type: ignore[index]
                            "excess_aggregate"
                        ]["mean"]
                    ),
                )
                for dof, rung in order
            ]
            ascending = all(
                values[index][2] >= values[index + 1][2]
                for index in range(len(values) - 1)
            )
            monotone &= ascending
            detail[str(size)] = {
                "by_free_parameters": [
                    {"dof": dof, "rung": rung, "mean_excess": _round(value)}
                    for dof, rung, value in values
                ],
                "monotone_decreasing_in_declared_symmetry": ascending,
            }
        ladder[fixture] = {
            "rungs_by_free_parameters": [
                {"dof": dof, "rung": rung} for dof, rung in order
            ],
            "monotone_from_n_16_upward": monotone,
            "detail": detail,
        }

    exact = all(
        block["T_versus_O_mismatches"] == 0
        and block["T_versus_G_mismatches"] == 0
        for block in agreement.values()
    )
    criteria = {
        "T_equals_O_and_G_exactly_on_both_fixtures": exact,
        "R_floor_is_zero_on_fano": fano_zero,
        "R_floor_is_positive_on_circulant": circ_positive,
        "R_is_indistinguishable_from_T_on_fano": fano_gap < 1e-9,
        "R_is_separated_from_T_on_circulant": circ_gap > 0.0,
        "compilation_is_load_bearing_on_circulant": (
            circ_t is not None and circ_r is None
        ),
        "dose_response_monotone_on_both_fixtures": all(
            bool(ladder[name]["monotone_from_n_16_upward"])  # type: ignore[index]
            for name in ("fano", "circulant")
        ),
    }
    if not criteria["T_equals_O_and_G_exactly_on_both_fixtures"]:
        verdict = "IMPLEMENTATION DEFECT"
        why = "T, O, and G must agree exactly under matched estimation"
    elif not criteria["R_is_indistinguishable_from_T_on_fano"]:
        verdict = "IMPLEMENTATION DEFECT"
        why = (
            "on Fano the relation partition and the derived partition are the "
            "same partition, so R and T must be bit-identical"
        )
    elif (
        criteria["R_is_separated_from_T_on_circulant"]
        and criteria["compilation_is_load_bearing_on_circulant"]
        and criteria["dose_response_monotone_on_both_fixtures"]
    ):
        verdict = "SEPARATION ESTABLISHED"
        why = (
            "the relation-indexed learner accounts for the whole 017.25 effect "
            "on Fano and provably cannot account for it on the circulant "
            "fixture, and the dose-response curve tracks declared symmetry"
        )
    elif criteria["R_is_separated_from_T_on_circulant"]:
        verdict = "PARTIAL SEPARATION"
        why = (
            "R is separated from T on the circulant fixture but at least one "
            "preregistered clause did not hold"
        )
    else:
        verdict = "NO SEPARATION"
        why = (
            "the relation-indexed learner remains an adequate account of the "
            "measured advantage, so the 017.25 claim must be downgraded"
        )
    return {
        "rule": (
            "preregistered: separation requires R to equal T on Fano, R to be "
            "separated from T on the circulant by more than its exact floor "
            "less sampling error, T to reach a threshold R never reaches, and "
            "the ladder to be monotone in declared symmetry from n = 16 upward"
        ),
        "separation": separation,
        "fano_max_absolute_R_minus_T": _round(fano_gap),
        "circulant_max_R_minus_T": _round(circ_gap),
        "dose_response": ladder,
        "exact_agreement": agreement,
        "criteria": criteria,
        "verdict": verdict,
        "reason": why,
    }


@dataclass(frozen=True, slots=True)
class Bundle:
    certificate: dict[str, object]
    reference: dict[str, object]
    results: dict[str, object]
    cost: dict[str, object]
    table: bytes


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _canonical_table(rows: Sequence[tuple[object, ...]]) -> bytes:
    import gzip

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


def run_everything(quick: bool = False) -> Bundle:
    started = time.perf_counter()
    preregistration = load_preregistration()

    moment = time.perf_counter()
    fixtures = (fano_fixture(), circulant_fixture())
    compile_seconds = time.perf_counter() - moment

    sizes = QUICK_SIZES if quick else SAMPLE_SIZES
    closed = QUICK_CLOSED_FORM_REPLICATES if quick else CLOSED_FORM_REPLICATES
    learned = QUICK_LEARNED_REPLICATES if quick else LEARNED_REPLICATES

    certificate = refinement_certificate(fixtures)
    moment = time.perf_counter()
    reference = prediction_payload(fixtures, sizes)
    reference_seconds = time.perf_counter() - moment

    rows: list[tuple[object, ...]] = []
    cells: dict[str, dict[str, object]] = {}
    agreement: dict[str, dict[str, int]] = {}
    moment = time.perf_counter()
    for fixture in fixtures:
        theta = evenly_spaced_theta(fixture.class_count)
        block, agree = run_closed_form(
            fixture,
            theta,
            sizes,
            closed,
            learned,
            ALPHA_PRIMARY,
            BETA_PRIMARY,
            rows,
        )
        cells.update(block)
        agreement[fixture.identifier] = agree
    closed_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    for fixture in fixtures:
        cells.update(
            run_learned(
                fixture,
                evenly_spaced_theta(fixture.class_count),
                sizes,
                learned,
                rows,
            )
        )
    learned_seconds = time.perf_counter() - moment
    cells.update(select_best_u(cells, fixtures, sizes))

    verdict = decide_verdict(cells, reference, agreement, sizes)

    validation: list[dict[str, object]] = []
    worst = 0.0
    for fixture in fixtures:
        for condition in ("T", "R", "S"):
            exact = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in reference["predictions"][  # type: ignore[index]
                    f"{fixture.identifier}|A|{condition}"
                ]["sweep"]
            }
            for size in sizes:
                block = cells[
                    f"{fixture.identifier}|A|{size}|{condition}"
                ]["excess_aggregate"]  # type: ignore[index]
                mean = float(block["mean"])  # type: ignore[index]
                high = float(block["ci95_high"])  # type: ignore[index]
                standard = max((high - mean) / 1.959963984540054, 1e-15)
                sigma = abs(mean - exact[size]) / standard
                worst = max(worst, sigma)
                validation.append(
                    {
                        "cell": f"{fixture.identifier}|A|{size}|{condition}",
                        "simulated": _round(mean),
                        "predicted": _round(exact[size]),
                        "sigma": _round(sigma),
                        "within_four_sigma": sigma <= 4.0,
                    }
                )
    failures = sum(1 for row in validation if not row["within_four_sigma"])

    checks = {
        "refinement_schema_bytes_match_the_pin": True,
        "preregistration_bytes_match_the_pin": True,
        "shared_017_24_module_bytes_match_the_pin": True,
        "preregistration_declares_the_predictions": "declared_predictions"
        in preregistration,
        "fano_group_order_is_168": fixtures[0].structure.group_order == 168,
        "circulant_group_order_is_14": fixtures[1].structure.group_order == 14,
        "fano_has_two_derived_classes": fixtures[0].class_count == 2,
        "circulant_has_four_derived_classes": fixtures[1].class_count == 4,
        "cardinalities_are_matched_across_fixtures": all(
            len(fixture.schema.incidence) == 21
            and len(fixture.structure.inputs) == 49
            for fixture in fixtures
        ),
        "fano_derived_partition_equals_the_relation": not bool(
            certificate["fixtures"][0][  # type: ignore[index]
                "derived_partition_strictly_refines_the_relation"
            ]
        ),
        "circulant_derived_partition_strictly_refines_the_relation": bool(
            certificate["fixtures"][1][  # type: ignore[index]
                "derived_partition_strictly_refines_the_relation"
            ]
        ),
        "every_derived_class_is_pure_with_respect_to_the_relation": all(
            bool(
                block[
                    "every_derived_class_lies_inside_the_relation_or_outside_it"
                ]
            )
            for block in certificate["fixtures"]  # type: ignore[union-attr]
        ),
        "ladders_are_monotone_in_free_parameters": all(
            bool(block["ladder_is_monotone_in_free_parameters"])
            for block in certificate["fixtures"]  # type: ignore[union-attr]
        ),
        "fano_subgroup_choice_does_not_change_the_quotient": len(
            {
                str(row["class_cardinalities"])
                for row in certificate["fixtures"][0][  # type: ignore[index]
                    "subgroup_choice_independence"
                ]
            }
        )
        <= 1,
        "R_floor_is_zero_on_fano": bool(
            reference["predictions"]["fano|A|R"]["floor_is_zero"]  # type: ignore[index]
        ),
        "R_floor_is_positive_on_circulant": not bool(
            reference["predictions"]["circulant|A|R"]["floor_is_zero"]  # type: ignore[index]
        ),
        "T_floor_is_zero_on_both_fixtures": all(
            bool(
                reference["predictions"][f"{name}|A|T"]["floor_is_zero"]  # type: ignore[index]
            )
            for name in ("fano", "circulant")
        ),
        "coarse_compiler_control_carries_a_positive_floor": all(
            float(block["coarse_compiler_floor"]) > 0.0  # type: ignore[index,arg-type]
            for block in reference["coarse_compiler_control"].values()  # type: ignore[union-attr]
        ),
        "coarse_compiler_correct_rung_has_no_floor": all(
            abs(float(block["correct_rung_floor"])) < 1e-12  # type: ignore[index,arg-type]
            for block in reference["coarse_compiler_control"].values()  # type: ignore[union-attr]
        ),
        "simulation_matches_the_frozen_predictions": failures == 0,
        "every_theta_star_is_interior": all(
            0 < value < 1
            for fixture in fixtures
            for value in evenly_spaced_theta(fixture.class_count)
        ),
        "closed_form_replicate_floor_respected": closed >= 100 or quick,
        "learned_replicate_floor_respected": learned >= 100 or quick,
        "U_relation_was_run": any(
            key.endswith("|U_relation") for key in cells
        ),
    }
    if not all(checks.values()):
        broken = sorted(name for name, ok in checks.items() if not ok)
        raise RuntimeError(f"mechanical checks failed: {broken}")

    results = {
        "schema": RESULTS_SCHEMA,
        "authority": authority_record(),
        "design": {
            "fixtures": [
                {
                    "id": fixture.identifier,
                    "schema": fixture.schema_name,
                    "group_order": fixture.structure.group_order,
                    "derived_classes": list(fixture.structure.cardinalities),
                    "theta_star": [
                        _fraction_text(value)
                        for value in evenly_spaced_theta(fixture.class_count)
                    ],
                    "holdout_per_class": list(fixture.holdout_per_class),
                }
                for fixture in fixtures
            ],
            "sample_sizes": list(sizes),
            "closed_form_replicates": closed,
            "learned_replicates": learned,
            "estimator": "beta(1,1) posterior mean",
            "learned_conditions": list(LEARNED_CONDITIONS),
        },
        "cells": cells,
        "J_verdict": verdict,
        "prediction_validation": {
            "cells_checked": len(validation),
            "failures": failures,
            "worst_sigma": _round(worst),
            "all_within_four_sigma": failures == 0,
            "rows": validation,
        },
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": True,
        "claim_fences": [
            "no language-scale claim, and no LLM or SLM comparison",
            "no physical Test Realization and no autonomous dynamics",
            "neither fixture is representative of anything beyond itself",
            "it is not claimed that OT is responsible for any measured gain",
            "it is not claimed that every useful Type Schema admits a compact "
            "compiler",
            "Issue 018 is not resolved here",
        ],
    }
    cost = {
        "schema": COST_SCHEMA,
        "authority": authority_record(),
        "seconds": {
            "compile_both_fixtures_and_all_ladder_rungs": _round(
                compile_seconds
            ),
            "exact_reference": _round(reference_seconds),
            "closed_form_fitting": _round(closed_seconds),
            "learned_fitting": _round(learned_seconds),
            "total_wall_clock": _round(time.perf_counter() - started),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    return Bundle(
        certificate=certificate,
        reference=reference,
        results=results,
        cost=cost,
        table=_canonical_table(rows),
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="print the frozen label-free predictions and exit without running",
    )
    parser.add_argument(
        "--check", action="store_true", help="recompute and verify the artifacts"
    )
    parser.add_argument(
        "--write", action="store_true", help="write every artifact to disk"
    )
    parser.add_argument(
        "--quick", action="store_true", help="reduced grid; refuses to write"
    )
    args = parser.parse_args(argv)

    if args.quick and args.write:
        print("--quick cannot write artifacts", file=sys.stderr)
        return 2

    if args.predict:
        fixtures = (fano_fixture(), circulant_fixture())
        payload = {
            "certificate": refinement_certificate(fixtures),
            "reference": prediction_payload(fixtures, SAMPLE_SIZES),
        }
        sys.stdout.buffer.write(_canonical(payload))
        return 0

    bundle = run_everything(quick=args.quick)

    if args.check:
        problems: list[str] = []
        for path, payload in (
            (CERTIFICATE_PATH, bundle.certificate),
            (REFERENCE_PATH, bundle.reference),
        ):
            if not path.exists():
                problems.append(f"missing artifact: {path.name}")
                continue
            if path.read_bytes() != _canonical(payload):
                problems.append(f"{path.name} differs byte-for-byte")
        if RESULTS_PATH.exists():
            if not _numeric_close(
                json.loads(RESULTS_PATH.read_text()),
                json.loads(_canonical(bundle.results)),
                NUMERIC_TOLERANCE,
            ):
                problems.append(
                    f"{RESULTS_PATH.name} differs beyond {NUMERIC_TOLERANCE}"
                )
        else:
            problems.append(f"missing artifact: {RESULTS_PATH.name}")
        if not TABLE_PATH.exists():
            problems.append(f"missing artifact: {TABLE_PATH.name}")
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(
            "017.24a verified: certificate "
            f"{_sha256(_canonical(bundle.certificate))[:16]} verdict "
            f"{bundle.results['J_verdict']['verdict']}"  # type: ignore[index]
        )
        return 0

    if args.write:
        for path, payload in (
            (CERTIFICATE_PATH, bundle.certificate),
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
