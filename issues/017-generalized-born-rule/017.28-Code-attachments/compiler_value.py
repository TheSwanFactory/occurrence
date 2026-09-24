"""Issue 017.28 Phase T: is the stronger compiler's extra structure worth anything?

Phase S found a clean separation and banked it before any target existed. On one
typed world -- a Latin square of order 6, declared as three carriers and one total
binary operation -- the frozen fixed-point refinement puts all 36 admitted inputs
in a single cell, while the exact automorphism action splits them into orbits of
12 and 24. That is a structural fact with an exact witness.

A structural fact is not a reason to pay for a compiler. 017.28 section 9 says so
plainly, and this module answers the question it actually asks:

    does the extra pre-label distinction reduce what Training has to learn,
    and at what sample size does it start paying for itself?

Two losses, both required.

**Approximation loss.** Q_ref merges two Q_aut cells carrying different
probabilities, so the best Q_ref-measurable predictor cannot be right about
either. That floor is computed exactly -- in rationals, then in 40-digit decimal
arithmetic -- from cell weights and theta_star, with no simulation anywhere.

**Estimation loss.** Q_aut carries more parameters than Q_ref, so at small n the
coarser compiler may win by pooling harder. The crossover is measured rather than
assumed, and a result where refinement wins throughout the tested regime would be
reported as the real result it is.

Four things make this the statistical half of a fenced experiment.

**The structure is read, not recomputed from scratch.** The Phase S certificate is
byte-pinned, and condition H reads Q_aut cell membership straight out of it by a
path that never calls a compiler. If the in-process compiler and the banked freeze
disagree in even one replicate, the run refuses to emit a payload.

**theta_star is fixed by a rule, not chosen.** The rule is inherited verbatim from
017.24a: lambda*_c = (2c + 1) / (2k) for k derived cells. With k = 2 that is
(1/4, 3/4). No freedom was exercised after the structure was visible, which is the
only way a separation cannot have been selected to flatter a condition.

**Every condition is refit from the same per-input counts.** Identical labelled
data is enforced structurally, by the byte-pinned 017.24 sufficient-statistic
path, rather than by convention.

**Every prediction is preregistered.** ``preregistration.json`` is written, pinned
by digest, committed, and banked before the sweep runs, and it contains the exact
closed-form value of every regime-A cell this run will measure.

The uncompiled generic learners of 017.25 and 017.27 are deliberately absent.
Section 9 makes them secondary and the main comparison compiler-versus-compiler;
017.27 already measured the compiled-versus-uncompiled axis on two other worlds
and nothing here disturbs it.
"""

from __future__ import annotations

import argparse
import decimal
import gzip
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

PREREGISTRATION_SCHEMA: Final = "gpt-01728-phase-t-preregistration/v1"
REFERENCE_SCHEMA: Final = "gpt-01728-phase-t-exact-reference/v1"
RESULTS_SCHEMA: Final = "gpt-01728-phase-t-results/v1"
COST_SCHEMA: Final = "gpt-01728-phase-t-cost-report/v1"

# The Phase S freeze of this turn, byte-pinned. Every structural quantity used
# here comes through this pin, so it is provably the derivation banked before
# theta_star was evaluated.
PHASE_S_MODULE_SHA256: Final = (
    "d8a57e810676b74fbd07395542afda8682c417082a371f2a138f02d3b13fd525"
)
PHASE_S_CERTIFICATE_SHA256: Final = (
    "5dff9481109cb70b84b0669c37e4a4fc9a235201c6a4e926a6ec9cfcdfb56275"
)
# The 017.24a implementation, byte-pinned, which itself pins 017.24. Every
# estimator, split builder, hidden generator, and exact reference is shared with
# 017.24, 017.25, 017.25a and 017.27 at the level of bytes.
REFINE_MODULE_SHA256: Final = (
    "9fcf033500b33cc9665a58ab04a24e6d7921417a5789cc7039550823eda7085f"
)
SHARED_MODULE_SHA256: Final = (
    "821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79"
)
PREREGISTRATION_SHA256: Final = (
    "cb3915f8daa687c1fe97f9d248f1402bf68de9790b269f3e0474b95b0a09b1cc"
)

TASK_GPT_REVISION: Final = (
    "e5289c582121bda62394321bf73b4b476f9bb23c49eb6efd19044c14366009f1"
)
PHASE_S_COMMIT: Final = "b973940a12be7effc4fd71a43e3d4bd9a4f17eca"
PHASE_S_QUILT_REVISION: Final = "3f0fcafe36df11bd2c80c9c988b584de3078a3c100aecb607d11b6bac526044b"
PRIOR_RESULT_COMMIT: Final = "2817b8c1a1d0e6b10d4ea6ac16d1bbce35df06a8"

HERE: Final = Path(__file__).resolve().parent
PHASE_S_PATH: Final = HERE / "compiler_scope.py"
PHASE_S_CERTIFICATE_PATH: Final = HERE / "phase_s_certificate.json"
REFINE_PATH: Final = (
    HERE.parent / "017.24a-Code-attachments" / "quotient_refinement.py"
)
PREREGISTRATION_PATH: Final = HERE / "preregistration.json"
REFERENCE_PATH: Final = HERE / "phase_t_reference.json"
RESULTS_PATH: Final = HERE / "phase_t_results.json"
COST_PATH: Final = HERE / "phase_t_cost_report.json"
TABLE_PATH: Final = HERE / "phase_t_per_replicate.csv.gz"

NUMERIC_TOLERANCE: Final = 1e-6


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


scope = _load_pinned(PHASE_S_PATH, PHASE_S_MODULE_SHA256, "compiler_scope_01728_pinned")
refine = _load_pinned(
    REFINE_PATH, REFINE_MODULE_SHA256, "quotient_refinement_01724a_pinned"
)
shared = refine.shared
exact = scope.exact

if refine.SHARED_MODULE_SHA256 != SHARED_MODULE_SHA256:
    raise RuntimeError("the 017.24a pin of the 017.24 implementation moved")

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)
CLIP: Final = shared.CLIP
ALPHA_PRIMARY: Final = ONE
BETA_PRIMARY: Final = ONE
ALPHA_SECONDARY: Final = Fraction(1, 2)
BETA_SECONDARY: Final = Fraction(1, 2)

# The inherited 017.24a grid, extended downward by two points. The separation
# leaves A with two parameters against R's one, so the estimation crossover is
# expected at very small n and the inherited grid would not resolve it. The
# extension is declared here and used for every condition equally.
SAMPLE_SIZES: Final = (1, 2, *refine.SAMPLE_SIZES)
QUICK_SIZES: Final = (2, 32, 256)
REPLICATES: Final = 2000
QUICK_REPLICATES: Final = 100
SECONDARY_REPLICATES: Final = 400
REGIMES: Final = ("A", "B")
THRESHOLDS: Final = (0.02, 0.01)
# A base distinct from 017.24 (20260923), 017.24a (20260924) and 017.26
# (20260927), so this turn's observations are independent of every banked one.
BASE_SEED: Final = 20260929

# Held-out identities per Q_aut cell in regime B, by rule rather than by choice.
HOLDOUT_RULE: Final = "hold out max(1, cell size // 4) identities from every Q_aut cell"

DECIMAL_DIGITS: Final = 40

CONDITION_ROLES: Final = {
    "A": "exact automorphism compiler: one observable scalar per Q_aut cell",
    "H": (
        "handed exact oracle: Q_aut cell membership read out of the byte-pinned "
        "Phase S certificate by a path that never calls a compiler"
    ),
    "R": "refinement compiler: one observable scalar per Q_ref cell",
    "S": "saturated ablation: one observable scalar per admitted input",
    "W": (
        "falsification control: a partition with the Q_aut cardinalities that "
        "respects no derived structure-preserving map"
    ),
}

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

_round = refine._round
_fraction_text = refine._fraction_text


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


def load_phase_s_certificate(
    path: Path = PHASE_S_CERTIFICATE_PATH,
) -> dict[str, object]:
    """Read the banked Phase S freeze, byte-pinned."""
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != PHASE_S_CERTIFICATE_SHA256:
        raise RuntimeError(
            f"the Phase S certificate changed: expected "
            f"{PHASE_S_CERTIFICATE_SHA256}, observed {observed}"
        )
    document = json.loads(raw)
    if document["phase"] != "S":
        raise RuntimeError("unexpected phase in the Phase S certificate")
    if not document["all_mechanical_checks_pass"]:
        raise RuntimeError("the Phase S certificate did not pass its own checks")
    return document


def load_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, object]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"the preregistration changed: expected {PREREGISTRATION_SHA256}, "
            f"observed {observed}"
        )
    document = json.loads(raw)
    if document["schema"] != PREREGISTRATION_SCHEMA:
        raise RuntimeError("unexpected preregistration schema identifier")
    return document


# ---------------------------------------------------------------------------
# 1. Condition H: the exact quotient, handed over, avoiding the compiler
# ---------------------------------------------------------------------------


def handed_quotient(block: dict[str, object], world: object) -> tuple[int, ...]:
    """Parse Q_aut out of the banked freeze, extensionally.

    This is what makes the A-against-H comparison mean something. A runs the
    exact compiler in this process. H is handed the cell membership lists the
    Phase S certificate recorded, parses them back into a partition without ever
    calling a compiler, and fits with the identical estimator. If the two
    disagree, either the compiler is not reproducing its own banked output or the
    estimator is not shared, and either way the run stops.
    """
    membership = block["Q_aut_membership"]
    position = {"".join(row): index for index, row in enumerate(world.inputs)}
    assignment = [-1] * len(position)
    for name in sorted(membership, key=lambda key: int(key.split("_")[1])):
        for member in membership[name]:
            assignment[position[member]] = int(name.split("_")[1])
    if min(assignment) < 0:
        raise RuntimeError("the freeze did not cover every admitted input")
    return tuple(assignment)


# ---------------------------------------------------------------------------
# 2. The falsification control
# ---------------------------------------------------------------------------


def build_wrong_partition(
    world: object, quotient: Sequence[int], morphisms: object
) -> tuple[tuple[int, ...], dict[str, object]]:
    """A partition with the exact cardinalities that respects no derived map.

    The 017.24a construction, reimplemented here because the byte-pinned version
    takes a 017.24a fixture object: move the k lexicographically least members of
    each exact cell into the next cell cyclically, with one common k, so every
    cardinality is preserved exactly while invariance is destroyed.
    """
    cells = [
        sorted(index for index, value in enumerate(quotient) if value == cell)
        for cell in sorted(set(quotient))
    ]
    exchanged = min(max(1, len(members) // 3) for members in cells)
    count = len(cells)
    assignment = [-1] * len(quotient)
    for cell, members in enumerate(cells):
        for offset, index in enumerate(members):
            assignment[index] = (cell + 1) % count if offset < exchanged else cell
    partition = tuple(assignment)
    sizes = sorted(partition.count(cell) for cell in range(count))
    if sizes != sorted(len(members) for members in cells):
        raise RuntimeError("the control changed a cardinality")
    violations = 0
    if morphisms is not None:
        index_of = {row: position for position, row in enumerate(world.inputs)}
        for morphism in morphisms:
            for row in world.inputs:
                moved = morphism.send(world, row)
                if partition[index_of[row]] != partition[index_of[moved]]:
                    violations += 1
    return partition, {
        "construction": (
            f"move the {exchanged} lexicographically least members of each exact "
            "cell into the next cell cyclically, preserving every cardinality"
        ),
        "members_exchanged_per_cell": exchanged,
        "cardinalities": sorted(len(members) for members in cells),
        "respects_the_derived_maps": violations == 0,
        "invariance_violations_counted": violations,
    }


# ---------------------------------------------------------------------------
# 3. The target: a rule, evaluated only now
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    name: str
    partition: tuple[int, ...]
    cell_count: int
    role: str


@dataclass(frozen=True, slots=True)
class Target:
    """The designated separation fixture, plus everything Phase T adds to it."""

    identifier: str
    role: str
    world: object
    document: dict[str, object]
    exact_partition: tuple[int, ...]
    refined_partition: tuple[int, ...]
    handed: tuple[int, ...]
    group_order: int
    morphisms: object
    conditions: tuple[ConditionSpec, ...]
    theta: tuple[Fraction, ...]
    holdout: tuple[int, ...]
    wrong_certificate: dict[str, object]

    @property
    def cell_count(self) -> int:
        return len(set(self.exact_partition))

    @property
    def width(self) -> int:
        return len(self.world.inputs)

    @property
    def members(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                index
                for index, value in enumerate(self.exact_partition)
                if value == cell
            )
            for cell in sorted(set(self.exact_partition))
        )


def build_targets() -> tuple[Target, ...]:
    """Assemble every fixture the Phase S freeze designated, and nothing else."""
    certificate = load_phase_s_certificate()
    designated = certificate["E_first_separation"]["designated_for_phase_T"]
    if not designated:
        raise RuntimeError(
            "Phase S designated no fixture, so section 15 forbids inventing "
            "Phase T"
        )
    built: list[Target] = []
    for block in designated:
        world = exact.world_from_document(block["document"], str(block["world"]))
        derived = scope.derive_exact_quotient(world)
        refined = scope.derive_refinement_quotient(world)
        handed = handed_quotient(block, world)
        cell_count = len(set(derived.partition))
        wrong, wrong_certificate = build_wrong_partition(
            world, derived.partition, derived.morphisms
        )
        specs = (
            ConditionSpec("A", derived.partition, cell_count, CONDITION_ROLES["A"]),
            ConditionSpec("H", handed, len(set(handed)), CONDITION_ROLES["H"]),
            ConditionSpec(
                "R", refined.partition, len(set(refined.partition)),
                CONDITION_ROLES["R"],
            ),
            ConditionSpec(
                "S",
                tuple(range(len(world.inputs))),
                len(world.inputs),
                CONDITION_ROLES["S"],
            ),
            ConditionSpec("W", wrong, cell_count, CONDITION_ROLES["W"]),
        )
        members = tuple(
            tuple(
                index
                for index, value in enumerate(derived.partition)
                if value == cell
            )
            for cell in sorted(set(derived.partition))
        )
        built.append(
            Target(
                identifier=str(block["world"]),
                role=str(block["role"]),
                world=world,
                document=block["document"],
                exact_partition=derived.partition,
                refined_partition=refined.partition,
                handed=handed,
                group_order=derived.group_order,
                morphisms=derived.morphisms,
                conditions=specs,
                theta=refine.evenly_spaced_theta(cell_count),
                holdout=tuple(max(1, len(cell) // 4) for cell in members),
                wrong_certificate=wrong_certificate,
            )
        )
    return tuple(built)


# ---------------------------------------------------------------------------
# 4. Approximation loss, exactly
# ---------------------------------------------------------------------------


def _decimal_cross_entropy(
    truth: decimal.Decimal, prediction: decimal.Decimal
) -> decimal.Decimal:
    return -(
        truth * prediction.ln()
        + (decimal.Decimal(1) - truth) * (decimal.Decimal(1) - prediction).ln()
    )


def exact_approximation_floor(
    partition: Sequence[int],
    theta: Sequence[Fraction],
    class_of_index: Sequence[int],
    pool: Sequence[int],
    evaluation: Sequence[int],
) -> dict[str, object]:
    """The irreducible proper-scoring floor of the best measurable predictor.

    Exact in two independent arithmetics. The best predictor measurable with
    respect to ``partition`` assigns each cell the pool-weighted mean of the true
    probabilities inside it, which is a rational number; the floor is then the
    evaluation-weighted mean of the cross-entropy gap. The rational cell limits
    are computed with ``Fraction`` and reported as exact ratios; the logarithms
    are then taken in 40-digit decimal arithmetic, so the reported floor is not a
    64-bit float result. The float value from the byte-pinned 017.24 helper is
    reported alongside and required to agree, which is a cross-check on both.
    """
    with decimal.localcontext() as context:
        context.prec = DECIMAL_DIGITS
        cells = sorted(set(partition))
        totals = {cell: ZERO for cell in cells}
        counts = {cell: 0 for cell in cells}
        for index in pool:
            cell = partition[index]
            totals[cell] += theta[class_of_index[index]]
            counts[cell] += 1
        default = ALPHA_PRIMARY / (ALPHA_PRIMARY + BETA_PRIMARY)
        limits = {
            cell: (totals[cell] / counts[cell] if counts[cell] else default)
            for cell in cells
        }
        floor = decimal.Decimal(0)
        for index in evaluation:
            truth = theta[class_of_index[index]]
            truth_decimal = decimal.Decimal(truth.numerator) / decimal.Decimal(
                truth.denominator
            )
            limit = limits[partition[index]]
            limit_decimal = decimal.Decimal(limit.numerator) / decimal.Decimal(
                limit.denominator
            )
            floor += _decimal_cross_entropy(
                truth_decimal, limit_decimal
            ) - _decimal_cross_entropy(truth_decimal, truth_decimal)
        floor /= decimal.Decimal(len(evaluation))
        helper = shared.asymptotic_floor(
            partition,
            len(cells),
            [float(value) for value in theta],
            class_of_index,
            pool,
            evaluation,
            ALPHA_PRIMARY,
            BETA_PRIMARY,
        )
        return {
            "cells": len(cells),
            "cell_limits": {
                f"c_{cell}": _fraction_text(limits[cell]) for cell in cells
            },
            "cell_sizes_in_the_training_pool": {
                f"c_{cell}": counts[cell] for cell in cells
            },
            "floor_40_digit_decimal": str(+floor),
            "floor_rounded": _round(float(floor)),
            "floor_from_the_byte_pinned_017_24_helper": _round(helper),
            "the_two_arithmetics_agree": abs(float(floor) - helper) < 1e-12,
            "floor_is_zero": abs(float(floor)) < 1e-12,
            "method": (
                "rational cell limits from cell weights and theta_star, then "
                "cross-entropy gaps in 40-digit decimal arithmetic. No "
                "simulation, no float logarithm"
            ),
        }


def _observation_seed(
    target_index: int, regime_index: int, size_index: int, replicate: int
) -> int:
    return (
        BASE_SEED
        + 1_000_000 * target_index
        + 100_000 * regime_index
        + 1_000 * size_index
        + replicate
    )


def exact_reference(
    targets: Sequence[Target], sizes: Sequence[int], replicates: int
) -> dict[str, object]:
    """Every prediction this turn makes, in closed form and before any label.

    Regime A is exact outright: pool and evaluation set are the whole
    admitted-input set, so the generalized 017.24a reference applies directly.
    Regime B depends on which identities a replicate holds out, and a split is
    built from source structure and a seed without reading a label, so the floor
    is averaged over the actual splits the run will use and the full sweep is
    reported for the declared reference split. No label exists in this function.
    """
    predictions: dict[str, object] = {}
    floors: dict[str, object] = {}
    for target_index, target in enumerate(targets):
        truths = [float(value) for value in target.theta]
        whole = tuple(range(target.width))
        for spec in target.conditions:
            predictions[f"{target.identifier}|A|{spec.name}"] = (
                refine.exact_expected_excess(
                    spec.partition,
                    spec.cell_count,
                    truths,
                    target.exact_partition,
                    whole,
                    whole,
                    sizes,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            )
            floors[f"{target.identifier}|A|{spec.name}"] = (
                exact_approximation_floor(
                    spec.partition, target.theta, target.exact_partition, whole, whole
                )
            )
        reference_rng = random.Random(_observation_seed(target_index, 1, 0, 0))
        reference_split = shared.build_split(
            "B", reference_rng, target.members, target.holdout
        )
        for spec in target.conditions:
            predictions[f"{target.identifier}|B|{spec.name}"] = (
                refine.exact_expected_excess(
                    spec.partition,
                    spec.cell_count,
                    truths,
                    target.exact_partition,
                    reference_split.train_pool,
                    reference_split.evaluation[0][1],
                    sizes,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            )
        averaged: dict[str, list[float]] = {
            spec.name: [] for spec in target.conditions
        }
        for replicate in range(replicates):
            rng = random.Random(_observation_seed(target_index, 1, 0, replicate))
            split = shared.build_split("B", rng, target.members, target.holdout)
            for spec in target.conditions:
                averaged[spec.name].append(
                    shared.asymptotic_floor(
                        spec.partition,
                        spec.cell_count,
                        truths,
                        target.exact_partition,
                        split.train_pool,
                        split.evaluation[0][1],
                        ALPHA_PRIMARY,
                        BETA_PRIMARY,
                    )
                )
        floors[f"{target.identifier}|B|over_every_split"] = {
            spec.name: {
                "mean": _round(sum(averaged[spec.name]) / len(averaged[spec.name])),
                "min": _round(min(averaged[spec.name])),
                "max": _round(max(averaged[spec.name])),
                "splits": len(averaged[spec.name]),
            }
            for spec in target.conditions
        }
    return {
        "schema": REFERENCE_SCHEMA,
        "authority": authority_record(),
        "method": (
            "the generalized 017.24a exact reference, imported byte-pinned: "
            "excess log loss is additive across cells and a cell's estimator "
            "depends only on its own observation count, which is marginally "
            "binomial, so the expectation is a finite sum per cell with no "
            "simulation"
        ),
        "no_label_exists_in_this_computation": True,
        "theta_star": {
            target.identifier: [_fraction_text(value) for value in target.theta]
            for target in targets
        },
        "predictions": predictions,
        "approximation_floors": floors,
    }


def target_admissibility(target: Target) -> dict[str, object]:
    """Check the section 9 and section 10 constraints on the hidden world.

    theta_star is fixed by rule, so what has to be verified is that the rule
    happened to satisfy the constraints: every probability interior, not all
    equal, the truth indexed by Q_aut cells, and at least one Q_ref cell merging
    Q_aut cells that carry different probabilities. The last one is the whole
    point of the design and it is checked rather than asserted.
    """
    merged: list[dict[str, object]] = []
    for cell in sorted(set(target.refined_partition)):
        members = [
            index
            for index, value in enumerate(target.refined_partition)
            if value == cell
        ]
        values = sorted(
            {_fraction_text(target.theta[target.exact_partition[index]]) for index in members}
        )
        if len(values) > 1:
            merged.append(
                {
                    "Q_ref_cell": f"R_{cell}",
                    "size": len(members),
                    "distinct_true_probabilities_merged": values,
                }
            )
    return {
        "rule": "lambda*_c = (2c + 1) / (2k) for k exact cells",
        "why_a_rule": (
            "inherited verbatim from 017.24a and evaluated only after the Phase S "
            "freeze was committed and banked, so no separation can have been "
            "selected to flatter a condition"
        ),
        "theta_star": [_fraction_text(value) for value in target.theta],
        "every_probability_is_interior": all(
            ZERO < value < ONE for value in target.theta
        ),
        "not_all_probabilities_are_equal": len(set(target.theta)) > 1,
        "the_truth_is_indexed_by_Q_aut_cells": len(target.theta) == target.cell_count,
        "Q_ref_cells_that_merge_distinct_probabilities": merged,
        "at_least_one_Q_ref_cell_merges_distinct_probabilities": bool(merged),
        "degrees_of_freedom": {
            spec.name: spec.cell_count for spec in target.conditions
        },
        "holdout_rule": HOLDOUT_RULE,
        "holdout_per_cell": list(target.holdout),
    }


# ---------------------------------------------------------------------------
# 5. The sweep
# ---------------------------------------------------------------------------


def _record(
    rows: list[tuple[object, ...]],
    identifier: str,
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
            identifier,
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


def _absorb(
    store: dict[str, list[float]],
    metrics: dict[str, dict[str, float]],
    cell_count: int,
) -> None:
    store["excess_aggregate"].append(metrics["aggregate"]["excess_nll"])
    store["nll_aggregate"].append(metrics["aggregate"]["nll"])
    store["brier_aggregate"].append(metrics["aggregate"]["brier"])
    store["calibration_aggregate"].append(metrics["aggregate"]["calibration_error"])
    for index in range(cell_count):
        name = f"O_{index}"
        if name in metrics:
            store[f"excess_{name}"].append(metrics[name]["excess_nll"])


def run_closed_form(
    target: Target,
    target_index: int,
    sizes: Sequence[int],
    replicates: int,
    alpha: Fraction,
    beta: Fraction,
    rows: list[tuple[object, ...]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Fit every condition on byte-identical observations, from shared counts."""
    truths = [float(value) for value in target.theta]
    numerators, denominator = shared.theta_as_integers(target.theta)
    cells: dict[str, dict[str, object]] = {}
    agreement = {
        "comparisons": 0,
        "A_versus_H_mismatches": 0,
        "A_versus_R_identical": 0,
        "A_versus_W_identical": 0,
    }
    for regime_index, regime in enumerate(REGIMES):
        for size_index, size in enumerate(sizes):
            store = {
                spec.name: refine._accumulator(target.cell_count)
                for spec in target.conditions
            }
            for replicate in range(replicates):
                rng = random.Random(
                    _observation_seed(
                        target_index, regime_index, size_index, replicate
                    )
                )
                split = shared.build_split(
                    regime, rng, target.members, target.holdout
                )
                observations = shared.generate_observations(
                    rng,
                    split.train_pool,
                    size,
                    numerators,
                    denominator,
                    target.exact_partition,
                )
                successes, trials = shared.pair_counts(observations, target.width)
                fitted: dict[str, object] = {}
                for spec in target.conditions:
                    model = shared.fit_bernoulli(
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
                        model.predictions, split, truths, target.exact_partition
                    )
                    _absorb(store[spec.name], metrics, target.cell_count)
                    _record(
                        rows,
                        target.identifier,
                        regime,
                        size,
                        spec.name,
                        replicate,
                        metrics,
                        model.degrees_of_freedom,
                    )
                agreement["comparisons"] += 1
                derived = fitted["A"].exact_predictions
                if derived != fitted["H"].exact_predictions:
                    agreement["A_versus_H_mismatches"] += 1
                if derived == fitted["R"].exact_predictions:
                    agreement["A_versus_R_identical"] += 1
                if derived == fitted["W"].exact_predictions:
                    agreement["A_versus_W_identical"] += 1
            for spec in target.conditions:
                cells[
                    f"{target.identifier}|{regime}|{size}|{spec.name}"
                ] = refine._summarize(
                    store[spec.name], replicates, spec.cell_count
                )
    return cells, agreement


# ---------------------------------------------------------------------------
# 6. The two losses, read off the sweep
# ---------------------------------------------------------------------------


def _series(
    cells: dict[str, dict[str, object]],
    identifier: str,
    regime: str,
    condition: str,
    sizes: Sequence[int],
) -> list[float]:
    return [
        float(
            cells[f"{identifier}|{regime}|{size}|{condition}"][  # type: ignore[index]
                "excess_aggregate"
            ]["mean"]
        )
        for size in sizes
    ]


def _standard_error(block: dict[str, object]) -> float:
    mean = float(block["mean"])  # type: ignore[arg-type]
    high = float(block["ci95_high"])  # type: ignore[arg-type]
    return max((high - mean) / 1.959963984540054, 1e-15)


def _to_threshold(
    sizes: Sequence[int], series: Sequence[float], threshold: float
) -> int | None:
    for size, value in zip(sizes, series, strict=True):
        if value <= threshold:
            return size
    return None


def estimation_crossover(
    cells: dict[str, dict[str, object]],
    identifier: str,
    regime: str,
    sizes: Sequence[int],
) -> dict[str, object]:
    """Where the extra structural distinction starts paying for its variance.

    The crossover is the smallest tested n at which the exact compiler's mean
    excess loss is at or below the refinement compiler's, and stays there for
    every larger tested n. Reporting it that way rather than as a first crossing
    matters: a single lucky cell should not be allowed to declare a crossover.
    Every size is additionally reported with the difference in standard-error
    units, so a reader can see which orderings are actually resolved.
    """
    exact_series = _series(cells, identifier, regime, "A", sizes)
    refined_series = _series(cells, identifier, regime, "R", sizes)
    rows: list[dict[str, object]] = []
    for size, left, right in zip(sizes, exact_series, refined_series, strict=True):
        exact_block = cells[f"{identifier}|{regime}|{size}|A"]["excess_aggregate"]  # type: ignore[index]
        refined_block = cells[f"{identifier}|{regime}|{size}|R"]["excess_aggregate"]  # type: ignore[index]
        spread = math.sqrt(
            _standard_error(exact_block) ** 2 + _standard_error(refined_block) ** 2
        )
        rows.append(
            {
                "n": size,
                "A": _round(left),
                "R": _round(right),
                "difference_A_minus_R": _round(left - right),
                "difference_in_standard_errors": _round((left - right) / spread),
                "A_is_better": left <= right,
                "ordering_is_resolved_at_two_sigma": abs(left - right) >= 2.0 * spread,
            }
        )
    crossover: int | None = None
    for index, size in enumerate(sizes):
        if all(
            exact_series[later] <= refined_series[later]
            for later in range(index, len(sizes))
        ):
            crossover = size
            break
    return {
        "definition": (
            "the smallest tested n at which A's mean excess loss is at or below "
            "R's and remains so for every larger tested n"
        ),
        "crossover_n": crossover,
        "R_wins_at": [int(row["n"]) for row in rows if not row["A_is_better"]],
        "R_wins_resolved_at_two_sigma_at": [
            int(row["n"])
            for row in rows
            if not row["A_is_better"] and row["ordering_is_resolved_at_two_sigma"]
        ],
        "rows": rows,
    }


VERDICTS: Final = (
    "SEPARATION AND VALUE",
    "SEPARATION WITHOUT DEMONSTRATED VALUE",
    "NO SEPARATION IN SCOPE",
    "COMPILER DEFECT",
)


def decide_verdict(
    cells: dict[str, dict[str, object]],
    reference: dict[str, object],
    targets: Sequence[Target],
    agreements: dict[str, dict[str, int]],
    certificate: dict[str, object],
    sizes: Sequence[int],
) -> dict[str, object]:
    """Score the section 16 criteria and choose exactly one of four verdicts."""
    per_fixture: dict[str, object] = {}
    for target in targets:
        floors = reference["approximation_floors"]
        exact_floor = floors[f"{target.identifier}|A|A"]
        refined_floor = floors[f"{target.identifier}|A|R"]
        exact_series = _series(cells, target.identifier, "A", "A", sizes)
        refined_series = _series(cells, target.identifier, "A", "R", sizes)
        per_fixture[target.identifier] = {
            "role": target.role,
            "admitted_inputs": target.width,
            "group_order": target.group_order,
            "Q_aut_cells": target.cell_count,
            "Q_ref_cells": len(set(target.refined_partition)),
            "degrees_of_freedom": {
                spec.name: spec.cell_count for spec in target.conditions
            },
            "A_approximation_floor": exact_floor["floor_40_digit_decimal"],
            "R_approximation_floor": refined_floor["floor_40_digit_decimal"],
            "A_floor_is_zero": exact_floor["floor_is_zero"],
            "R_floor_is_strictly_positive": not refined_floor["floor_is_zero"],
            "crossover_regime_A": estimation_crossover(
                cells, target.identifier, "A", sizes
            ),
            "crossover_regime_B": estimation_crossover(
                cells, target.identifier, "B", sizes
            ),
            "A_reaches_threshold_at": _to_threshold(
                sizes, exact_series, THRESHOLDS[0]
            ),
            "R_reaches_threshold_at": _to_threshold(
                sizes, refined_series, THRESHOLDS[0]
            ),
            "A_excess_at_the_largest_n": _round(exact_series[-1]),
            "R_excess_at_the_largest_n": _round(refined_series[-1]),
            "A_equals_H_in_every_replicate": (
                agreements[target.identifier]["A_versus_H_mismatches"] == 0
            ),
            "A_equals_H_comparisons": agreements[target.identifier]["comparisons"],
            "A_and_R_coincided_in": agreements[target.identifier][
                "A_versus_R_identical"
            ],
            "A_and_W_coincided_in": agreements[target.identifier][
                "A_versus_W_identical"
            ],
            "admissibility": target_admissibility(target),
        }

    separation = certificate["E_first_separation"]["separations_found"] > 0
    ledger = certificate["D_search_ledger"]
    no_orbit_split = not any(row["Q_ref_splits_an_automorphism_orbit"] for row in ledger)
    identical = all(
        agreement["A_versus_H_mismatches"] == 0 for agreement in agreements.values()
    )
    criteria = {
        "the_structural_separation_is_banked_in_phase_S": separation,
        "Q_ref_never_split_an_automorphism_orbit": no_orbit_split,
        "A_is_bit_identical_to_the_handed_exact_oracle": identical,
        "R_carries_a_strictly_positive_approximation_floor": all(
            bool(block["R_floor_is_strictly_positive"])  # type: ignore[index]
            for block in per_fixture.values()
        ),
        "A_carries_a_zero_approximation_floor": all(
            bool(block["A_floor_is_zero"]) for block in per_fixture.values()  # type: ignore[index]
        ),
        "A_overtakes_R_within_the_tested_grid_and_stays_ahead": all(
            block["crossover_regime_A"]["crossover_n"] is not None  # type: ignore[index]
            for block in per_fixture.values()
        ),
        "A_reaches_the_threshold_and_R_never_does": all(
            block["A_reaches_threshold_at"] is not None  # type: ignore[index]
            and block["R_reaches_threshold_at"] is None  # type: ignore[index]
            for block in per_fixture.values()
        ),
        "the_falsification_control_respects_no_derived_map": all(
            not bool(target.wrong_certificate["respects_the_derived_maps"])
            for target in targets
        ),
    }

    if not identical or not no_orbit_split:
        verdict = "COMPILER DEFECT"
        why = (
            "either the in-process exact compiler disagreed with its own banked "
            "freeze, or the refinement split an automorphism orbit. Section 16 "
            "says stop and repair rather than interpret statistics"
        )
    elif not separation:
        verdict = "NO SEPARATION IN SCOPE"
        why = "Phase S found no separating world inside the frozen family"
    elif all(criteria.values()):
        verdict = "SEPARATION AND VALUE"
        why = (
            "a preregistered fixture family yielded Q_ref != Q_aut; the exact "
            "distinctions carry target-relevant information, measured as a "
            "strictly positive irreducible floor for the refinement compiler "
            "against zero for the exact one; and the exact compiler's asymptotic "
            "advantage is observed inside the tested grid and does not reverse"
        )
    else:
        verdict = "SEPARATION WITHOUT DEMONSTRATED VALUE"
        why = (
            "the separation is structural, but at least one preregistered "
            "criterion for the extra structure being worth its statistical and "
            "compilation cost did not hold"
        )

    return {
        "rule": (
            "COMPILER DEFECT if A and H ever disagree or the refinement splits an "
            "orbit; NO SEPARATION IN SCOPE if Phase S found none; SEPARATION AND "
            "VALUE if every criterion below holds; otherwise SEPARATION WITHOUT "
            "DEMONSTRATED VALUE. Frozen in the preregistration before the sweep"
        ),
        "criteria": criteria,
        "per_fixture": per_fixture,
        "verdict": verdict,
        "why": why,
        "interpretation_permitted": [
            "on the tested typed world, fixed-point structural refinement failed "
            "to distinguish admitted inputs separated by the world's exact "
            "automorphism action",
            "those additional pre-label distinctions reduced the approximation "
            "burden of the tested decision model",
        ],
        "interpretation_forbidden": [
            "that exact automorphism computation is generally necessary; on 82 of "
            "87 worlds in the frozen family it was not",
            "that Q_aut is the universal TDM compiler",
            "that a refinement-based learner cannot learn the distinction from "
            "labels; nothing here tests that",
            "any learner-independent sample-complexity lower bound",
            "that OT caused any measured advantage",
            "that these finite worlds represent language-scale input",
            "that types make learning cheaper; this turn compares two compilers "
            "on one world, not typed against untyped",
        ],
    }


# ---------------------------------------------------------------------------
# 7. Preregistration, reference, results
# ---------------------------------------------------------------------------


def authority_record() -> dict[str, object]:
    gpt = f"quilt+s3://protology#package=occurrence/gpt@{TASK_GPT_REVISION}"
    return {
        "task": (
            f"{gpt}&path=issues/017-jev-pivot/"
            "017.28-Task-compiler-scope-refinement-versus-exact-automorphism.md"
        ),
        "phase_s_freeze": (
            "quilt+s3://protology#package=occurrence/gpt@"
            f"{PHASE_S_QUILT_REVISION}&path=issues/017-jev-pivot/"
            "017.28a-Kiro-phase-S-compiler-scope-freeze.md"
        ),
        "phase_s_commit": PHASE_S_COMMIT,
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "byte_pinned_inputs": {
            "compiler_scope.py": PHASE_S_MODULE_SHA256,
            "phase_s_certificate.json": PHASE_S_CERTIFICATE_SHA256,
            "preregistration.json": PREREGISTRATION_SHA256,
            "../017.24a-Code-attachments/quotient_refinement.py": REFINE_MODULE_SHA256,
            "../017.24-Code-attachments/training_economy.py": SHARED_MODULE_SHA256,
        },
    }


def build_preregistration() -> dict[str, object]:
    """Everything frozen before the sweep: theta_star, seeds, thresholds, values.

    Called with ``--preregister`` and committed and banked before any observation
    is generated. It contains the exact closed-form value of every regime-A cell
    the run will measure, so every measured number has a committed prediction to
    be checked against.
    """
    targets = build_targets()
    reference = exact_reference(targets, SAMPLE_SIZES, REPLICATES)
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "phase": "T",
        "authority": {
            key: value
            for key, value in authority_record().items()
            if key != "byte_pinned_inputs"
        },
        "byte_pinned_inputs": {
            "compiler_scope.py": PHASE_S_MODULE_SHA256,
            "phase_s_certificate.json": PHASE_S_CERTIFICATE_SHA256,
            "../017.24a-Code-attachments/quotient_refinement.py": REFINE_MODULE_SHA256,
            "../017.24-Code-attachments/training_economy.py": SHARED_MODULE_SHA256,
        },
        "fixtures": [
            {
                "world": target.identifier,
                "role": target.role,
                "admitted_inputs": target.width,
                "group_order": target.group_order,
                "Q_aut_cell_sizes": scope._cell_sizes(target.exact_partition),
                "Q_ref_cell_sizes": scope._cell_sizes(target.refined_partition),
                "conditions": [
                    {
                        "name": spec.name,
                        "role": spec.role,
                        "parameters": spec.cell_count,
                    }
                    for spec in target.conditions
                ],
                "falsification_control": target.wrong_certificate,
                "admissibility": target_admissibility(target),
            }
            for target in targets
        ],
        "protocol": {
            "sample_sizes": list(SAMPLE_SIZES),
            "replicates": REPLICATES,
            "secondary_prior_replicates": SECONDARY_REPLICATES,
            "regimes": {
                "A": (
                    "train and evaluate under the uniform distribution on every "
                    "admitted input; exact expected loss in closed form"
                ),
                "B": (
                    "hold out whole input identities stratified by Q_aut cell; "
                    "train only on observations from the seen identities; "
                    "evaluate exactly on the unseen ones"
                ),
            },
            "estimator": {
                "primary": "beta(1,1) posterior mean, (k + 1) / (m + 2)",
                "secondary": "beta(1/2,1/2) posterior mean",
                "shared_by": (
                    "every condition, refit from the same per-input sufficient "
                    "counts of the same observations"
                ),
            },
            "base_seed": BASE_SEED,
            "seed_rule": (
                "BASE_SEED + 1000000 * fixture + 100000 * regime + 1000 * size "
                "+ replicate; one stream builds the split and then the "
                "observations, so every condition sees byte-identical data"
            ),
            "thresholds": list(THRESHOLDS),
            "holdout_rule": HOLDOUT_RULE,
            "crossover_definition": (
                "the smallest tested n at which A's mean excess loss is at or "
                "below R's and remains so for every larger tested n"
            ),
        },
        "verdict_rule": {
            "options": list(VERDICTS),
            "criteria": [
                "the structural separation is banked in Phase S",
                "Q_ref never split an automorphism orbit",
                "A is bit-identical to the handed exact oracle",
                "R carries a strictly positive approximation floor",
                "A carries a zero approximation floor",
                "A overtakes R within the tested grid and stays ahead",
                "A reaches the 0.02 threshold and R never does",
                "the falsification control respects no derived map",
            ],
            "selection": (
                "COMPILER DEFECT dominates; then NO SEPARATION IN SCOPE; then "
                "SEPARATION AND VALUE if every criterion holds; otherwise "
                "SEPARATION WITHOUT DEMONSTRATED VALUE"
            ),
        },
        "frozen_predictions": reference,
        "what_would_falsify_the_thesis": (
            "R beating A at every tested sample size, which is possible in "
            "principle because R carries one parameter against A's two and its "
            "floor is only 0.1 nats. If that happened the verdict would be "
            "SEPARATION WITHOUT DEMONSTRATED VALUE and that is the result that "
            "would be reported"
        ),
    }


@dataclass(frozen=True, slots=True)
class Bundle:
    reference: dict[str, object]
    results: dict[str, object]
    cost: dict[str, object]
    table: bytes


def _prediction_validation(
    cells: dict[str, dict[str, object]],
    reference: dict[str, object],
    targets: Sequence[Target],
    sizes: Sequence[int],
) -> dict[str, object]:
    """Every regime A cell must land within four standard errors of its exact value."""
    rows: list[dict[str, object]] = []
    worst = 0.0
    failures = 0
    for target in targets:
        for spec in target.conditions:
            predicted = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in reference["predictions"][  # type: ignore[index]
                    f"{target.identifier}|A|{spec.name}"
                ]["sweep"]
            }
            for size in sizes:
                block = cells[f"{target.identifier}|A|{size}|{spec.name}"][
                    "excess_aggregate"
                ]  # type: ignore[index]
                mean = float(block["mean"])  # type: ignore[arg-type]
                standard = _standard_error(block)
                sigma = abs(mean - predicted[size]) / standard
                worst = max(worst, sigma)
                ok = sigma <= 4.0
                failures += int(not ok)
                rows.append(
                    {
                        "cell": f"{target.identifier}|A|{size}|{spec.name}",
                        "simulated": _round(mean),
                        "predicted": _round(predicted[size]),
                        "sigma": _round(sigma),
                        "within_four_sigma": ok,
                    }
                )
    return {
        "method": (
            "the exact expectation must lie within four standard errors of the "
            "simulated mean for every condition in every regime A cell; the "
            "standard error is recovered from the reported confidence interval so "
            "the check is reproducible from the artifact alone"
        ),
        "cells_checked": len(rows),
        "failures": failures,
        "worst_sigma": _round(worst),
        "all_within_four_sigma": failures == 0,
        "rows": rows,
    }


def run_everything(quick: bool = False) -> Bundle:
    started = time.perf_counter()
    sizes = QUICK_SIZES if quick else SAMPLE_SIZES
    replicates = QUICK_REPLICATES if quick else REPLICATES

    moment = time.perf_counter()
    certificate = load_phase_s_certificate()
    targets = build_targets()
    assemble_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    reference = exact_reference(targets, sizes, replicates)
    reference_seconds = time.perf_counter() - moment

    rows: list[tuple[object, ...]] = []
    cells: dict[str, dict[str, object]] = {}
    agreements: dict[str, dict[str, int]] = {}
    moment = time.perf_counter()
    for target_index, target in enumerate(targets):
        block, agreement = run_closed_form(
            target,
            target_index,
            sizes,
            replicates,
            ALPHA_PRIMARY,
            BETA_PRIMARY,
            rows,
        )
        cells.update(block)
        agreements[target.identifier] = agreement
    sweep_seconds = time.perf_counter() - moment

    validation = _prediction_validation(cells, reference, targets, sizes)
    verdict = decide_verdict(
        cells, reference, targets, agreements, certificate, sizes
    )

    moment = time.perf_counter()
    secondary: dict[str, dict[str, object]] = {}
    for target_index, target in enumerate(targets):
        block, _agreement = run_closed_form(
            target,
            target_index,
            (sizes[0], sizes[len(sizes) // 2], sizes[-1]),
            SECONDARY_REPLICATES if not quick else QUICK_REPLICATES,
            ALPHA_SECONDARY,
            BETA_SECONDARY,
            [],
        )
        secondary.update(block)
    secondary_seconds = time.perf_counter() - moment

    preregistered: dict[str, object] | None = None
    preregistration_agrees: bool | None = None
    if PREREGISTRATION_PATH.exists() and not PREREGISTRATION_SHA256.startswith(
        "PREREGISTRATION"
    ):
        preregistered = load_preregistration()
        preregistration_agrees = refine._numeric_close(
            preregistered["frozen_predictions"]["predictions"],  # type: ignore[index]
            json.loads(_canonical(reference))["predictions"],
            NUMERIC_TOLERANCE,
        ) if not quick else None

    checks: dict[str, bool] = {
        "the_phase_s_module_bytes_match_the_pin": True,
        "the_phase_s_certificate_bytes_match_the_pin": True,
        "the_017_24a_implementation_bytes_match_the_pin": True,
        "the_017_24_implementation_bytes_match_the_pin": True,
        "the_phase_s_certificate_passed_every_check_of_its_own": bool(
            certificate["all_mechanical_checks_pass"]
        ),
        "phase_s_designated_at_least_one_separation_fixture": len(targets) > 0,
        "the_exact_compiler_reproduced_its_banked_quotient": all(
            tuple(target.exact_partition) == tuple(target.handed)
            for target in targets
        ),
        "A_and_H_agreed_in_every_replicate": all(
            agreement["A_versus_H_mismatches"] == 0
            for agreement in agreements.values()
        ),
        "every_regime_A_cell_landed_within_four_sigma": bool(
            validation["all_within_four_sigma"]
        ),
        "the_verdict_is_one_of_the_preregistered_outcomes": verdict["verdict"]
        in VERDICTS,
    }
    for target in targets:
        tag = target.identifier
        admissibility = target_admissibility(target)
        checks[f"{tag}: theta_star is interior and unequal"] = bool(
            admissibility["every_probability_is_interior"]
        ) and bool(admissibility["not_all_probabilities_are_equal"])
        checks[f"{tag}: the truth is indexed by Q_aut cells"] = bool(
            admissibility["the_truth_is_indexed_by_Q_aut_cells"]
        )
        checks[f"{tag}: a Q_ref cell merges Q_aut cells of distinct probability"] = (
            bool(admissibility["at_least_one_Q_ref_cell_merges_distinct_probabilities"])
        )
        checks[f"{tag}: Q_ref is strictly coarser than Q_aut"] = exact._refines(
            target.exact_partition, target.refined_partition
        ) and not exact._refines(target.refined_partition, target.exact_partition)
        checks[f"{tag}: the falsification control respects no derived map"] = not bool(
            target.wrong_certificate["respects_the_derived_maps"]
        )
        checks[f"{tag}: the saturated model carries the most parameters"] = (
            target.width > target.cell_count
        )
        floors = reference["approximation_floors"]
        checks[f"{tag}: the two floor arithmetics agree for every condition"] = all(
            bool(floors[f"{tag}|A|{spec.name}"]["the_two_arithmetics_agree"])
            for spec in target.conditions
        )
        exact_series = _series(cells, tag, "A", "A", sizes)
        checks[f"{tag}: A improves monotonically over the tested grid"] = all(
            later <= earlier + 1e-9
            for earlier, later in zip(exact_series[:-1], exact_series[1:], strict=True)
        )
    if preregistration_agrees is not None:
        checks["the_measured_reference_matches_the_preregistered_one"] = (
            preregistration_agrees
        )
    if not all(checks.values()):
        raise RuntimeError(
            "Phase T mechanical checks failed: "
            f"{sorted(name for name, ok in checks.items() if not ok)}"
        )

    results = {
        "schema": RESULTS_SCHEMA,
        "phase": "T",
        "authority": authority_record(),
        "design": {
            "fixtures": [
                {
                    "world": target.identifier,
                    "role": target.role,
                    "admitted_inputs": target.width,
                    "group_order": target.group_order,
                    "Q_aut_cell_sizes": scope._cell_sizes(target.exact_partition),
                    "Q_ref_cell_sizes": scope._cell_sizes(target.refined_partition),
                    "theta_star": [
                        _fraction_text(value) for value in target.theta
                    ],
                    "holdout_per_cell": list(target.holdout),
                    "conditions": [
                        {
                            "name": spec.name,
                            "role": spec.role,
                            "parameters": spec.cell_count,
                        }
                        for spec in target.conditions
                    ],
                    "admissibility": target_admissibility(target),
                    "falsification_control": target.wrong_certificate,
                }
                for target in targets
            ],
            "sample_sizes": list(sizes),
            "replicates": replicates,
            "regimes": {
                "A": "uniform over every admitted input, train and evaluate",
                "B": "hold out whole identities stratified by Q_aut cell",
            },
            "uncompiled_learners": (
                "deliberately absent. Section 9 makes them secondary and the main "
                "comparison compiler-versus-compiler; 017.27 measured the "
                "compiled-against-uncompiled axis on two other worlds and nothing "
                "here disturbs it"
            ),
        },
        "cells": cells,
        "secondary_prior_cells": secondary,
        "agreement": agreements,
        "approximation_loss": reference["approximation_floors"],
        "prediction_validation": validation,
        "K_verdict": verdict,
        "L_claim_ledger": claim_ledger(targets, certificate, reference),
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": all(checks.values()),
        "fences": [
            "no language-scale claim and no comparison to any language model",
            "no physical Test Realization and no autonomous dynamics",
            "no Interact",
            "OT is not claimed to be responsible for any measured gain",
            "the fixture is not representative of anything beyond itself",
            "Issue 018 is not resolved",
            "nothing was added to the stable decision-model public API",
        ],
    }
    cost = {
        "schema": COST_SCHEMA,
        "phase": "T",
        "authority": authority_record(),
        "seconds": {
            "assemble_and_recompile_the_designated_fixtures": _round(
                assemble_seconds
            ),
            "exact_reference_and_approximation_floors": _round(reference_seconds),
            "closed_form_sweep": _round(sweep_seconds),
            "secondary_prior_sweep": _round(secondary_seconds),
            "total_wall_clock": _round(time.perf_counter() - started),
        },
        "structural_compilation_cost_from_phase_S": (
            "see phase_s_cost_report.json; compilation cost is a Phase S "
            "measurement and is not re-measured here"
        ),
        "fits": len(rows),
        "environment": {"python": sys.version.split()[0]},
    }
    return Bundle(
        reference=reference, results=results, cost=cost, table=_canonical_table(rows)
    )


def claim_ledger(
    targets: Sequence[Target],
    certificate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, object]:
    """The section 20 L ledger, separated by how each item was obtained."""
    ledger = certificate["D_search_ledger"]
    return {
        "declared": [
            "for every world in the frozen family: typed carriers with their "
            "elements, and primitives each carrying a declared kind and signature",
            "for the separation fixture: three carriers of six elements and one "
            "total binary operation of 36 rows",
            "the admitted-input product type",
            "a binary response type with two neutral labels",
        ],
        "derived_by_Q_ref": [
            f"the refinement quotient of all {len(ledger)} searched worlds, in "
            "1 to 4 rounds of polynomial work",
            "on 82 of those worlds the refinement quotient equals the exact "
            "automorphism-orbit quotient",
            "on the separation fixture the refinement quotient is the single cell "
            "of all 36 admitted inputs",
        ],
        "derived_only_by_Q_aut": [
            f"|Aut| = {target.group_order} and {target.cell_count} orbits with "
            f"cardinalities {scope._cell_sizes(target.exact_partition)} on "
            f"{target.identifier}"
            for target in targets
        ]
        + [
            "the exact non-existence of a structure-preserving map between the "
            "witness pair, certified by exhausting the product of symmetric "
            "groups and confirmed by applying every derived element",
            "the orbit distinction on the separation fixture, which no level of "
            "the tested refinement ladder reaches",
        ],
        "learned_from_labels": [
            f"{spec.cell_count} observable scalars for condition {spec.name} on "
            f"{target.identifier}"
            for target in targets
            for spec in target.conditions
        ],
        "implementation_choice": [
            "the frozen refinement construction itself: the slot pool, one "
            "operation closure round, the pair-indexed atomic type, and the "
            "monotone fixpoint criterion",
            "the declared size bounds of the family, set by an off-family "
            "tractability probe",
            "materializing the derived group and closing the inputs by union-find "
            "when the group is small enough, rather than asking the orbit "
            "question pair by pair",
            "the exact-compiler audit rule: first world of every family plus "
            "every designated fixture",
            "extending the inherited sample grid downward by n = 1 and n = 2 so "
            "the crossover is resolvable",
            "beta(1,1) primary and beta(1/2,1/2) secondary",
            "2000 replicates, the 0.02 threshold, and the "
            "stays-ahead definition of the crossover",
            "40-digit decimal arithmetic for the approximation floor",
            "canonical JSON and float rounding for reproducible digests",
        ],
        "still_open": [
            "whether the separation generalizes beyond typed worlds declaring a "
            "total binary operation; every separating world in the frozen family "
            "declares one and no relation-only world separated",
            "what tractable compiler approximates Q_aut without paying exact "
            "global-symmetry cost; the widest separation in the ledger, "
            "composite_product:n=10:m=5:offsets=1 at 1 cell against 13, was left "
            "untested by the frozen protocol",
            "how broad a schema class can be certified for refinement "
            "sufficiency; 82 of 87 is an observation, not a theorem",
            "whether a learner given the refinement quotient and labels can "
            "recover the orbit distinction; nothing here tests that",
            "any learner-independent sample-complexity claim",
            "definability, congruence, bisimulation-like, context, and "
            "task-relative quotients, all untouched",
            "physical realization, Interact, and Issue 018",
        ],
        "theta_star_was_evaluated_only_after_the_phase_S_freeze_was_banked": True,
        "theta_star": reference["theta_star"],
    }


def verify(bundle: Bundle) -> list[str]:
    problems: list[str] = []
    expected = _canonical(bundle.reference)
    if not REFERENCE_PATH.exists():
        problems.append(f"missing artifact: {REFERENCE_PATH.name}")
    elif REFERENCE_PATH.read_bytes() != expected:
        problems.append(
            f"{REFERENCE_PATH.name} mismatch: observed "
            f"{_sha256(REFERENCE_PATH.read_bytes())}, expected {_sha256(expected)}"
        )
    if not RESULTS_PATH.exists():
        problems.append(f"missing artifact: {RESULTS_PATH.name}")
    elif not refine._numeric_close(
        json.loads(RESULTS_PATH.read_text()),
        json.loads(_canonical(bundle.results)),
        NUMERIC_TOLERANCE,
    ):
        problems.append(
            f"{RESULTS_PATH.name} differs beyond the declared numeric tolerance "
            f"{NUMERIC_TOLERANCE}"
        )
    if not TABLE_PATH.exists():
        problems.append(f"missing artifact: {TABLE_PATH.name}")
    elif TABLE_PATH.read_bytes() != bundle.table:
        problems.append(
            f"{TABLE_PATH.name} mismatch: observed "
            f"{_sha256(TABLE_PATH.read_bytes())}, expected {_sha256(bundle.table)}"
        )
    if not COST_PATH.exists():
        problems.append(f"missing artifact: {COST_PATH.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregister",
        action="store_true",
        help="write the preregistration, which contains no observation",
    )
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
        help="reduced grid for development; refuses to write or verify",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a short human summary instead of the payload",
    )
    args = parser.parse_args(argv)

    if args.quick and (args.write or args.check or args.preregister):
        print("--quick cannot write, verify, or preregister", file=sys.stderr)
        return 2

    if args.preregister:
        payload = build_preregistration()
        blob = _canonical(payload)
        PREREGISTRATION_PATH.write_bytes(blob)
        print(f"wrote {PREREGISTRATION_PATH.name} {_sha256(blob)}")
        return 0

    bundle = run_everything(quick=args.quick)

    if args.check:
        problems = verify(bundle)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(
            "017.28 Phase T verified: reference "
            f"{_sha256(_canonical(bundle.reference))[:16]} verdict "
            f"{bundle.results['K_verdict']['verdict']}"  # type: ignore[index]
        )
        return 0

    if args.write:
        for path, payload in (
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

    if args.summary:
        results = bundle.results
        verdict = results["K_verdict"]
        print(f"verdict: {verdict['verdict']}")
        for name, block in verdict["per_fixture"].items():
            print(f"\n{name}  |X|={block['admitted_inputs']} |Aut|={block['group_order']}")
            print(f"  dof: {block['degrees_of_freedom']}")
            print(f"  A floor: {block['A_approximation_floor'][:24]}")
            print(f"  R floor: {block['R_approximation_floor'][:24]}")
            print(f"  crossover (regime A): n = {block['crossover_regime_A']['crossover_n']}")
            print(f"  R wins at: {block['crossover_regime_A']['R_wins_at']}")
            print(
                f"  A reaches 0.02 at n = {block['A_reaches_threshold_at']}; "
                f"R at {block['R_reaches_threshold_at']}"
            )
            print("  n        A         R       A-R    sigma")
            for row in block["crossover_regime_A"]["rows"]:
                print(
                    f"  {row['n']:5d} {row['A']:9.4f} {row['R']:9.4f} "
                    f"{row['difference_A_minus_R']:9.4f} "
                    f"{row['difference_in_standard_errors']:8.2f}"
                )
        print(
            f"\nchecks: {results['mechanical_check_count']} "
            f"all pass: {results['all_mechanical_checks_pass']}"
        )
        print(
            f"prediction validation: worst sigma "
            f"{results['prediction_validation']['worst_sigma']} "
            f"failures {results['prediction_validation']['failures']}"
        )
        return 0

    sys.stdout.buffer.write(_canonical(bundle.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
