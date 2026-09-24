"""Issue 017.26 Phase T: what the derived quotient is worth once labels exist.

Phase S derived, from ordinary typed world descriptions and before any target
existed, that Fixture A has ``|Aut| = 14`` with input cells 7/14/14/14 and
Fixture B has ``|Aut| = 6`` with input cells 3/6/6/6/6. That freeze is committed
and banked. This module is the second half of the 017.26b section 7 split, and it
may only exist after that commit.

Everything statistical is new here and nothing structural is. The Phase S module
is imported byte-pinned, so the group and the quotient this turn measures against
are provably the ones banked before ``theta_star`` was chosen. The estimator, the
hidden generator, the split builder, the exact evaluation, the uncompiled learner,
and the generalized exact reference are imported byte-pinned from the 017.24a
attachments, which in turn pin 017.24, so nothing about the statistics was
reimplemented for this turn either.

Five conditions carry the argument.

``D``  the derived structural compiler. Receives the declared world, runs the
       Phase S compiler, fits one observable probability per derived cell.
``H``  the handed-quotient oracle. Receives ``q_Sigma`` outright, read out of the
       committed Phase S certificate by a code path that never touches the
       compiler, and fits with the identical estimator. 017.26b section 8 requires
       ``predictions_D = predictions_H`` in every replicate.
``R``  the preregistered shortcut learner, carrying one parameter per cell of the
       frozen ``L_k`` partition and nothing derived. This is the explicit
       cheaper-explanation adversary, and section 8 forbids it the automorphism
       group, canonical labelling, or any equivalent global structural quotient.
``U``  the same-information generic learners. They receive a label-free encoding
       of every declared fact and a model-selection ladder, and the span
       certificate shows their quadratic design contains the cell indicator
       exactly, so they are not crippled.
``S``  the saturated model, one parameter per admitted input.

Three controls run against the same observations: the quotient a single primitive
can support on its own, which 017.26b section 10 requires to be insufficient; a
cardinality-matched partition respecting no derived map; and the quotient of a
scrambled world.

The one interpretation this module is not permitted to support is stated in its
own results payload. ``D`` and ``U`` receive the same source information. Any
advantage measures inductive and computational leverage from deterministic
structural compilation under the tested learners. It is not a sample-complexity
lower bound, and it is not evidence that ``U`` lacked information.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np

PREREGISTRATION_SCHEMA: Final = "gpt-01726-phase-t-preregistration/v1"
REFERENCE_SCHEMA: Final = "gpt-01726-phase-t-exact-reference/v1"
RESULTS_SCHEMA: Final = "gpt-01726-phase-t-results/v1"
COST_SCHEMA: Final = "gpt-01726-phase-t-cost-report/v1"

# The Phase S freeze, byte-pinned. Every structural quantity this turn uses comes
# through this pin, so it is provably the derivation banked before theta_star.
PHASE_S_MODULE_SHA256: Final = (
    "e7074a60ed460b7b98d7f234e3af06aec169e5ae526e477c62e1e8799d0f2e3b"
)
PHASE_S_CERTIFICATE_SHA256: Final = (
    "f2e36998f0e5715d785d5dd0a86790fc32604040334c8250d2fd6c845c29709a"
)
# The 017.24a implementation, byte-pinned, which itself pins 017.24.
REFINE_MODULE_SHA256: Final = (
    "9fcf033500b33cc9665a58ab04a24e6d7921417a5789cc7039550823eda7085f"
)
SHARED_MODULE_SHA256: Final = (
    "821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79"
)
PREREGISTRATION_SHA256: Final = (
    "55dc5c7a452d2b6ef4cb7429d4e0111ac54c598b317fc7006d576ba1f52735d1"
)

TASK_GPT_REVISION: Final = (
    "be3d592a9c1a975f52cc292d6bfb6689c8c63d3c8bfadcbe796c427f1399657b"
)
PHASE_S_COMMIT: Final = "08437b1b418c89dc7c2974a172a4ef965406ca65"
PHASE_S_QUILT_REVISION: Final = (
    "57da798c3beee322e425f613c8e61496b3d9ab74655dc1367388b8219a3682af"
)
PRIOR_RESULT_COMMIT: Final = "2cb043a7c92ad16b9e9df784fce82d3d61531b24"

HERE: Final = Path(__file__).resolve().parent
PHASE_S_PATH: Final = HERE / "derived_symmetry.py"
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


phase_s = _load_pinned(
    PHASE_S_PATH, PHASE_S_MODULE_SHA256, "derived_symmetry_01726_pinned"
)
refine = _load_pinned(
    REFINE_PATH, REFINE_MODULE_SHA256, "quotient_refinement_01724a_pinned"
)
shared = refine.shared

if refine.SHARED_MODULE_SHA256 != SHARED_MODULE_SHA256:
    raise RuntimeError("the 017.24a pin of the 017.24 implementation moved")

ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)
CLIP: Final = shared.CLIP
ALPHA_PRIMARY: Final = ONE
BETA_PRIMARY: Final = ONE
ALPHA_SECONDARY: Final = Fraction(1, 2)
BETA_SECONDARY: Final = Fraction(1, 2)

SAMPLE_SIZES: Final = refine.SAMPLE_SIZES
QUICK_SIZES: Final = refine.QUICK_SIZES
CLOSED_FORM_REPLICATES: Final = 400
LEARNED_REPLICATES: Final = 100
QUICK_CLOSED_FORM_REPLICATES: Final = 40
QUICK_LEARNED_REPLICATES: Final = 20
REGIMES: Final = ("A", "B")
THRESHOLDS: Final = (0.02, 0.01)
MATERIAL_FACTOR: Final = 2.0
# A base distinct from 017.24 (20260923) and 017.24a (20260924), so this turn's
# observations are independent of every banked one.
BASE_SEED: Final = 20260927
LEARNER_SEED_OFFSET: Final = 500_000_000

# Held-out identities per derived cell in regime B, one entry per cell in the
# canonical cell order. Every cell must keep both seen and unseen members.
HOLDOUT_PER_CELL: Final = {"A": (2, 3, 3, 3), "B": (1, 2, 2, 2, 2)}

U_RIDGE_GRID: Final = shared.U_RIDGE_GRID
U_STEPS: Final = shared.U_STEPS
HIDDEN_WIDTH: Final = shared.HIDDEN_WIDTH
U_VARIANTS: Final = (
    ("U_fixed", ("mlp",), (shared.U_L2,), False),
    ("U_earlystop", ("mlp",), (shared.U_L2,), True),
    ("U_selected", ("mlp",), U_RIDGE_GRID, True),
    ("U_quadratic", ("quadratic",), U_RIDGE_GRID, True),
    ("U_ladder", ("constant", "linear", "quadratic", "mlp"), U_RIDGE_GRID, True),
    ("U_shortcut", ("shortcut_linear", "shortcut_mlp"), U_RIDGE_GRID, True),
)
LEARNED_CONDITIONS: Final = tuple(name for name, _, _, _ in U_VARIANTS)
HIDDEN_OF: Final = {
    "constant": None,
    "linear": None,
    "quadratic": None,
    "mlp": HIDDEN_WIDTH,
    "shortcut_linear": None,
    "shortcut_mlp": HIDDEN_WIDTH,
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


def load_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> dict[str, object]:
    raw = path.read_bytes()
    observed = _sha256(raw)
    if observed != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration changed: expected {PREREGISTRATION_SHA256}, "
            f"observed {observed}"
        )
    document = json.loads(raw)
    if document["schema"] != PREREGISTRATION_SCHEMA:
        raise RuntimeError("unexpected preregistration schema identifier")
    return document


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
    return document


# ---------------------------------------------------------------------------
# 1. Condition H: the quotient, handed over, by a path that avoids the compiler
# ---------------------------------------------------------------------------


def handed_quotient(
    certificate: dict[str, object], identifier: str, world: object
) -> tuple[int, ...]:
    """Read ``q_Sigma`` out of the banked freeze, extensionally.

    This is what makes the ``D`` against ``H`` comparison mean something. ``D``
    runs the compiler in this process. ``H`` is handed the cell membership lists
    the freeze recorded, parses them back into a partition without ever calling
    the compiler, and fits with the identical estimator. If the two disagree,
    either the compiler is not reproducing its own banked output or the estimator
    is not shared, and 017.26b section 8 says to stop and repair before
    interpreting anything.
    """
    for block in certificate["fixtures"]:  # type: ignore[union-attr]
        if block["fixture"] != identifier:
            continue
        cells = block["C_derived_before_any_label"]["cells"]
        if not isinstance(cells, dict):
            raise RuntimeError("the freeze did not list cell membership")
        position = {"".join(row): index for index, row in enumerate(world.inputs)}
        assignment = [-1] * len(position)
        for name in sorted(cells, key=lambda key: int(key.split("_")[1])):
            cell = int(name.split("_")[1])
            for member in cells[name]:
                assignment[position[member]] = cell
        if min(assignment) < 0:
            raise RuntimeError("the freeze did not cover every admitted input")
        return tuple(assignment)
    raise RuntimeError(f"the freeze has no fixture {identifier!r}")


# ---------------------------------------------------------------------------
# 2. The label-free encoding handed to the uncompiled learners
# ---------------------------------------------------------------------------


def build_feature_matrix(world: object) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Encode every declared fact about an admitted input, without any label.

    Blocks are generated from the declared signatures: the identity of each input
    component inside its carrier; for every relation, every argument position, and
    every type-correct filling of the other positions by input components, the
    indicator vector of that slice; for every relation, the truth value on the
    input components themselves; and for every operation, the identity of its
    value on the input components. The complete declared extension of every
    primitive is therefore present, which is what information parity means here.
    """
    inputs = world.inputs
    width = len(inputs)
    columns: list[np.ndarray] = []
    blocks: list[dict[str, object]] = []

    def add(name: str, matrix: np.ndarray) -> None:
        columns.append(matrix)
        blocks.append({"block": name, "columns": int(matrix.shape[1])})

    for position, carrier_name in enumerate(world.input_signature):
        elements = world.carrier(carrier_name).elements
        index_of = {element: slot for slot, element in enumerate(elements)}
        matrix = np.zeros((width, len(elements)))
        for row, entry in enumerate(inputs):
            matrix[row, index_of[entry[position]]] = 1.0
        add(f"identity_of_x{position}_in_{carrier_name}", matrix)

    components: list[tuple[int, str]] = list(enumerate(world.input_signature))
    for primitive in world.primitives:
        if primitive.kind == "relation":
            for free in range(len(primitive.argument_carriers)):
                others = [
                    [
                        slot
                        for slot, carrier_name in components
                        if carrier_name == primitive.argument_carriers[index]
                    ]
                    for index in range(len(primitive.argument_carriers))
                    if index != free
                ]
                if any(not pool for pool in others):
                    continue
                for chosen in itertools.product(*others):
                    pool = world.carrier(
                        primitive.argument_carriers[free]
                    ).elements
                    matrix = np.zeros((width, len(pool)))
                    for row, entry in enumerate(inputs):
                        fixed = [entry[slot] for slot in chosen]
                        for slot, element in enumerate(pool):
                            candidate = list(fixed)
                            candidate.insert(free, element)
                            if tuple(candidate) in primitive.extension:
                                matrix[row, slot] = 1.0
                    label = ",".join(
                        "free" if index == free else f"x{chosen[index if index < free else index - 1]}"
                        for index in range(len(primitive.argument_carriers))
                    )
                    add(f"slice_of_{primitive.name}({label})", matrix)
            pools = [
                [
                    slot
                    for slot, carrier_name in components
                    if carrier_name == argument
                ]
                for argument in primitive.argument_carriers
            ]
            if all(pools):
                for chosen in itertools.product(*pools):
                    matrix = np.zeros((width, 1))
                    for row, entry in enumerate(inputs):
                        candidate = tuple(entry[slot] for slot in chosen)
                        matrix[row, 0] = (
                            1.0 if candidate in primitive.extension else 0.0
                        )
                    label = ",".join(f"x{slot}" for slot in chosen)
                    add(f"{primitive.name}({label})", matrix)
        elif primitive.kind == "operation":
            pools = [
                [
                    slot
                    for slot, carrier_name in components
                    if carrier_name == argument
                ]
                for argument in primitive.argument_carriers
            ]
            if not all(pools):
                continue
            table = dict(primitive.graph)
            target = world.carrier(primitive.value_carrier).elements
            index_of = {element: slot for slot, element in enumerate(target)}
            for chosen in itertools.product(*pools):
                matrix = np.zeros((width, len(target)))
                for row, entry in enumerate(inputs):
                    value = table[tuple(entry[slot] for slot in chosen)]
                    matrix[row, index_of[value]] = 1.0
                label = ",".join(f"x{slot}" for slot in chosen)
                add(f"value_of_{primitive.name}({label})", matrix)
    return np.hstack(columns), blocks


def build_shortcut_matrix(world: object) -> np.ndarray:
    """The frozen ``L_k`` features, handed to ``U_shortcut`` as numeric columns."""
    _terms, features = phase_s.build_shortcut_language(world)
    width = len(world.inputs)
    matrix = np.zeros((width, max(1, len(features))))
    for column, feature in enumerate(features):
        for row in range(width):
            value = feature.values[row]
            matrix[row, column] = float(value) if not isinstance(
                value, bool
            ) else (1.0 if value else 0.0)
    return matrix


def build_designs(world: object) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    features, blocks = build_feature_matrix(world)
    width = features.shape[0]
    shortcut = np.hstack([features, build_shortcut_matrix(world)])
    return (
        {
            "constant": np.ones((width, 1)),
            "linear": np.hstack([np.ones((width, 1)), features]),
            "quadratic": shared.build_quadratic_design(features),
            "mlp": features,
            "shortcut_linear": np.hstack([np.ones((width, 1)), shortcut]),
            "shortcut_mlp": shortcut,
        },
        blocks,
    )


def _parameter_count(design: np.ndarray, hidden: int | None) -> int:
    columns = design.shape[1]
    if hidden is None:
        return columns + 1
    return columns * hidden + hidden + hidden + 1


def span_certificate(
    design: np.ndarray, quotient: Sequence[int]
) -> dict[str, object]:
    """Show constructively that the learners' design contains the cell function.

    017.26 section 7 says do not cripple ``U``. The strongest available form of
    that is to exhibit, rather than assert, that the cell indicator lies in the
    span of the design the learners are given, so any label-efficiency gap is
    about inductive bias and optimization rather than about reachability.
    """
    rows: list[dict[str, object]] = []
    for cell in sorted(set(quotient)):
        target = np.array(
            [1.0 if value == cell else 0.0 for value in quotient]
        )
        solution, *_rest = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.abs(design @ solution - target).max())
        rows.append({"cell": f"Q_{cell}", "max_abs_residual": _round(residual)})
    single = np.zeros(design.shape[0])
    single[0] = 1.0
    solution, *_rest = np.linalg.lstsq(design, single, rcond=None)
    single_residual = float(np.abs(design @ solution - single).max())
    return {
        "design": "quadratic",
        "columns": int(design.shape[1]),
        "per_cell": rows,
        "every_cell_indicator_lies_in_the_span": all(
            float(row["max_abs_residual"]) < 1e-8 for row in rows
        ),
        "a_single_input_indicator_also_lies_in_the_span": single_residual < 1e-8,
        "single_input_max_abs_residual": _round(single_residual),
        "reading": (
            "the compiled family is inside the learners' hypothesis class, so "
            "any measured gap is inductive and computational and not a question "
            "of what the learner could represent"
        ),
    }


# ---------------------------------------------------------------------------
# 3. The controls, built generically from the derived quotient
# ---------------------------------------------------------------------------


def build_wrong_partition(
    world: object, compiled: object
) -> tuple[tuple[int, ...], dict[str, object]]:
    """A partition with the derived cardinalities that respects no derived map.

    The 017.24a construction, generalized to any number of cells: move the ``k``
    lexicographically least members of each derived cell into the next cell
    cyclically, with one common ``k``, so every cardinality is preserved exactly
    while the partition stops being invariant.
    """
    quotient = compiled.quotient
    cells = [
        sorted(
            index for index, value in enumerate(quotient) if value == cell
        )
        for cell in sorted(set(quotient))
    ]
    exchanged = min(max(1, len(members) // 3) for members in cells)
    count = len(cells)
    assignment = [-1] * len(quotient)
    for cell, members in enumerate(cells):
        for position, index in enumerate(members):
            assignment[index] = (
                (cell + 1) % count if position < exchanged else cell
            )
    partition = tuple(assignment)
    sizes = [partition.count(cell) for cell in range(count)]
    if sizes != [len(members) for members in cells]:
        raise RuntimeError("the wrong-compiler control changed a cardinality")
    violations = 0
    if compiled.morphisms is not None:
        index_of = {row: position for position, row in enumerate(world.inputs)}
        for morphism in compiled.morphisms:
            for row in world.inputs:
                moved = morphism.send(world, row)
                if partition[index_of[row]] != partition[index_of[moved]]:
                    violations += 1
    return partition, {
        "construction": (
            f"move the {exchanged} lexicographically least members of each "
            "derived cell into the next cell cyclically, preserving every "
            "cardinality"
        ),
        "members_exchanged_per_cell": exchanged,
        "cardinalities": [len(members) for members in cells],
        "respects_the_derived_maps": violations == 0,
        "invariance_violations_counted": violations,
    }


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    name: str
    partition: tuple[int, ...]
    cell_count: int
    role: str


@dataclass(frozen=True, slots=True)
class Target:
    """One declared world plus everything Phase T adds to it."""

    identifier: str
    world: object
    document: dict[str, object]
    compiled: object
    handed: tuple[int, ...]
    shortcut: tuple[int, ...]
    conditions: tuple[ConditionSpec, ...]
    theta: tuple[Fraction, ...]
    holdout: tuple[int, ...]
    designs: dict[str, np.ndarray]
    feature_blocks: list[dict[str, object]]
    wrong_certificate: dict[str, object]

    @property
    def cell_count(self) -> int:
        return len(set(self.compiled.quotient))

    @property
    def width(self) -> int:
        return len(self.world.inputs)

    @property
    def members(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                index
                for index, value in enumerate(self.compiled.quotient)
                if value == cell
            )
            for cell in sorted(set(self.compiled.quotient))
        )


def build_targets() -> tuple[Target, ...]:
    """Assemble both fixtures, with theta_star fixed by rule rather than choice."""
    certificate = load_phase_s_certificate()
    built: list[Target] = []
    for loader in (phase_s.load_world_a, phase_s.load_world_b):
        world, document = loader()
        identifier = world.identifier
        compiled = phase_s.compile_world(world)
        handed = handed_quotient(certificate, identifier, world)
        _terms, features = phase_s.build_shortcut_language(world)
        shortcut = phase_s.joint_partition(features, len(world.inputs))
        cell_count = len(set(compiled.quotient))

        specs: list[ConditionSpec] = [
            ConditionSpec("D", compiled.quotient, cell_count, "treatment"),
            ConditionSpec("H", handed, len(set(handed)), "oracle"),
            ConditionSpec(
                "R", shortcut, len(set(shortcut)), "preregistered shortcut"
            ),
            ConditionSpec(
                "S",
                tuple(range(len(world.inputs))),
                len(world.inputs),
                "saturated ablation",
            ),
        ]
        for primitive in world.primitives:
            without = phase_s.compile_world(
                phase_s.world_from_document(
                    phase_s.erase_primitive(document, primitive.name),
                    f"{identifier}:without:{primitive.name}",
                )
            )
            specs.append(
                ConditionSpec(
                    f"P_without_{primitive.name}",
                    without.quotient,
                    len(set(without.quotient)),
                    "single-primitive coarser quotient",
                )
            )
        wrong, wrong_certificate = build_wrong_partition(world, compiled)
        specs.append(
            ConditionSpec("W_wrong_compiler", wrong, cell_count, "control")
        )
        scrambled = phase_s.compile_world(
            phase_s.world_from_document(
                phase_s.scramble_primitive(
                    document, world.primitives[0].name, phase_s.SCRAMBLE_SEED
                ),
                f"{identifier}:scrambled",
            )
        )
        specs.append(
            ConditionSpec(
                "C_scrambled_compiler",
                scrambled.quotient,
                len(set(scrambled.quotient)),
                "control",
            )
        )
        designs, blocks = build_designs(world)
        built.append(
            Target(
                identifier=identifier,
                world=world,
                document=document,
                compiled=compiled,
                handed=handed,
                shortcut=shortcut,
                conditions=tuple(specs),
                theta=refine.evenly_spaced_theta(cell_count),
                holdout=HOLDOUT_PER_CELL[identifier],
                designs=designs,
                feature_blocks=blocks,
                wrong_certificate=wrong_certificate,
            )
        )
    return tuple(built)


# ---------------------------------------------------------------------------
# 4. The exact reference, computed without generating a single label
# ---------------------------------------------------------------------------


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

    Regime A is exact outright: the pool and the evaluation set are the whole
    admitted-input set, so the generalized 017.24a reference applies directly.
    Regime B depends on which identities a replicate holds out, and a split is
    built from source structure and a seed without reading a label, so the
    asymptotic floor is averaged over the actual splits the run will use and the
    full sweep is reported for a declared reference split. No label exists at any
    point in this function.
    """
    everything: dict[str, object] = {}
    for target_index, target in enumerate(targets):
        truths = [float(value) for value in target.theta]
        whole = tuple(range(target.width))
        for spec in target.conditions:
            everything[f"{target.identifier}|A|{spec.name}"] = (
                refine.exact_expected_excess(
                    spec.partition,
                    spec.cell_count,
                    truths,
                    target.compiled.quotient,
                    whole,
                    whole,
                    sizes,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            )
        reference_rng = random.Random(_observation_seed(target_index, 1, 0, 0))
        reference_split = shared.build_split(
            "B", reference_rng, target.members, target.holdout
        )
        for spec in target.conditions:
            everything[f"{target.identifier}|B|{spec.name}"] = (
                refine.exact_expected_excess(
                    spec.partition,
                    spec.cell_count,
                    truths,
                    target.compiled.quotient,
                    reference_split.train_pool,
                    reference_split.evaluation[0][1],
                    sizes,
                    ALPHA_PRIMARY,
                    BETA_PRIMARY,
                )
            )
        floors: dict[str, list[float]] = {
            spec.name: [] for spec in target.conditions
        }
        for replicate in range(replicates):
            rng = random.Random(_observation_seed(target_index, 1, 0, replicate))
            split = shared.build_split(
                "B", rng, target.members, target.holdout
            )
            for spec in target.conditions:
                floors[spec.name].append(
                    shared.asymptotic_floor(
                        spec.partition,
                        spec.cell_count,
                        truths,
                        target.compiled.quotient,
                        split.train_pool,
                        split.evaluation[0][1],
                        ALPHA_PRIMARY,
                        BETA_PRIMARY,
                    )
                )
        everything[f"{target.identifier}|B|floors_over_every_split"] = {
            spec.name: {
                "mean": _round(sum(floors[spec.name]) / len(floors[spec.name])),
                "min": _round(min(floors[spec.name])),
                "max": _round(max(floors[spec.name])),
                "splits": len(floors[spec.name]),
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
        "regime_B_note": (
            "a split is built from source structure and a seed and reads no "
            "label, so the reference split below is the one replicate 0 of "
            "regime B will actually use, and the floors are averaged over every "
            "split the run will use"
        ),
        "theta_star": {
            target.identifier: [
                _fraction_text(value) for value in target.theta
            ]
            for target in targets
        },
        "predictions": everything,
    }


def sufficiency_certificate(target: Target) -> dict[str, object]:
    """Check the 017.26b section 10 constraints on theta_star, mechanically.

    The rule fixing theta_star is declared, not tuned, so the thing that has to
    be verified is that the rule happened to satisfy the constraints: every
    probability interior, not all equal, and every coarser partition that any
    single primitive or the frozen shortcut language can support carrying a
    strictly positive floor.
    """
    truths = [float(value) for value in target.theta]
    whole = tuple(range(target.width))
    rows: list[dict[str, object]] = []
    for spec in target.conditions:
        if spec.role not in (
            "single-primitive coarser quotient",
            "preregistered shortcut",
        ):
            continue
        floor = shared.asymptotic_floor(
            spec.partition,
            spec.cell_count,
            truths,
            target.compiled.quotient,
            whole,
            whole,
            ALPHA_PRIMARY,
            BETA_PRIMARY,
        )
        rows.append(
            {
                "condition": spec.name,
                "role": spec.role,
                "cells": spec.cell_count,
                "coarser_than_the_derived_quotient": phase_s._refines(
                    target.compiled.quotient, spec.partition
                )
                and not phase_s._refines(spec.partition, target.compiled.quotient),
                "asymptotic_floor": _round(floor),
                "floor_is_strictly_positive": floor > 1e-9,
            }
        )
    return {
        "rule": "lambda*_c = (2c + 1) / (2k) for k derived cells",
        "why_a_rule": (
            "fixed by rule rather than chosen, so no separation is selected to "
            "flatter any condition, and chosen only after the Phase S freeze was "
            "committed as 017.26b section 7 requires"
        ),
        "theta_star": [_fraction_text(value) for value in target.theta],
        "every_probability_is_interior": all(
            ZERO < value < ONE for value in target.theta
        ),
        "not_all_probabilities_are_equal": len(set(target.theta)) > 1,
        "insufficient_partitions": rows,
        "every_coarser_declared_partition_carries_a_positive_floor": all(
            bool(row["floor_is_strictly_positive"]) for row in rows
        ),
    }


# ---------------------------------------------------------------------------
# 5. The run
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
    store["calibration_aggregate"].append(
        metrics["aggregate"]["calibration_error"]
    )
    for index in range(cell_count):
        name = f"O_{index}"
        if name in metrics:
            store[f"excess_{name}"].append(metrics[name]["excess_nll"])


def run_closed_form(
    target: Target,
    target_index: int,
    sizes: Sequence[int],
    replicates: int,
    matched: int,
    alpha: Fraction,
    beta: Fraction,
    rows: list[tuple[object, ...]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Fit every closed-form condition on byte-identical observations."""
    truths = [float(value) for value in target.theta]
    numerators, denominator = shared.theta_as_integers(target.theta)
    cells: dict[str, dict[str, object]] = {}
    agreement = {
        "comparisons": 0,
        "D_versus_H_mismatches": 0,
        "D_versus_R_identical": 0,
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
                    target.compiled.quotient,
                )
                successes, trials = shared.pair_counts(
                    observations, target.width
                )
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
                        model.predictions,
                        split,
                        truths,
                        target.compiled.quotient,
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
                derived = fitted["D"].exact_predictions
                if derived != fitted["H"].exact_predictions:
                    agreement["D_versus_H_mismatches"] += 1
                if derived == fitted["R"].exact_predictions:
                    agreement["D_versus_R_identical"] += 1
            for spec in target.conditions:
                cells[
                    f"{target.identifier}|{regime}|{size}|{spec.name}"
                ] = refine._summarize(
                    store[spec.name], matched, spec.cell_count
                )
    return cells, agreement


def run_learned(
    target: Target,
    target_index: int,
    sizes: Sequence[int],
    replicates: int,
    rows: list[tuple[object, ...]],
) -> dict[str, dict[str, object]]:
    """Fit every uncompiled variant on exactly the same observations."""
    truths = [float(value) for value in target.theta]
    numerators, denominator = shared.theta_as_integers(target.theta)
    width = target.width
    designs = target.designs
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
                seed = _observation_seed(
                    target_index, regime_index, size_index, replicate
                )
                seeds.append(seed + LEARNER_SEED_OFFSET)
                rng = random.Random(seed)
                split = shared.build_split(
                    regime, rng, target.members, target.holdout
                )
                observations = shared.generate_observations(
                    rng,
                    split.train_pool,
                    size,
                    numerators,
                    denominator,
                    target.compiled.quotient,
                )
                splits.append(split)
                for position, (index, answer) in enumerate(
                    zip(observations.indices, observations.answers, strict=True)
                ):
                    slot = 0 if answer == 1 else 1
                    full[slot][replicate, index] += 1.0
                    bucket = inner if position < training else held
                    bucket[slot][replicate, index] += 1.0

            train_totals = (inner[0] + inner[1]).sum(axis=1)
            validation_totals = (held[0] + held[1]).sum(axis=1)
            full_totals = (full[0] + full[1]).sum(axis=1)
            cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
            for name in designs:
                for ridge in U_RIDGE_GRID:
                    _final, best, loss = shared._train_batched(
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
            unvalidated, _best, _loss = shared._train_batched(
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
                counts = {
                    name: _parameter_count(designs[name], HIDDEN_OF[name])
                    for name in candidates
                }
                if not use_validation:
                    predictions = unvalidated
                    selection: dict[str, object] = {
                        "model_selection": (
                            "none; the full step budget is reported"
                        )
                    }
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
                            "internal validation on training labels only, over "
                            "the declared candidate and ridge grids"
                        ),
                        "candidates": list(candidates),
                        "selected_candidate_counts": {
                            name: int((best_kind == position).sum())
                            for position, name in enumerate(candidates)
                        },
                    }
                dof = max(counts.values())
                selection["declared_parameters"] = dof
                store = refine._accumulator(target.cell_count)
                for replicate in range(replicates):
                    values = tuple(
                        float(value) for value in predictions[replicate]
                    )
                    metrics = shared.evaluate_exactly(
                        values,
                        splits[replicate],
                        truths,
                        target.compiled.quotient,
                    )
                    _absorb(store, metrics, target.cell_count)
                    _record(
                        rows,
                        target.identifier,
                        regime,
                        size,
                        variant,
                        replicate,
                        metrics,
                        dof,
                    )
                summary = refine._summarize(store, replicates, dof)
                summary["model_selection"] = selection
                cells[
                    f"{target.identifier}|{regime}|{size}|{variant}"
                ] = summary
    return cells


# ---------------------------------------------------------------------------
# 6. Selection, thresholds, and the verdict
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


def _to_threshold(
    sizes: Sequence[int], series: Sequence[float], threshold: float
) -> int | None:
    for size, value in zip(sizes, series, strict=True):
        if value <= threshold:
            return size
    return None


def select_best_uncompiled(
    cells: dict[str, dict[str, object]],
    targets: Sequence[Target],
    sizes: Sequence[int],
) -> None:
    """Add a per-cell best-of-the-uncompiled row, deliberately against the thesis.

    Picking the winning uncompiled variant separately in every cell is more
    generous to the baseline than any single variant could be, and it is the row
    the labelled-data-burden criterion is scored against.
    """
    for target in targets:
        for regime in REGIMES:
            for size in sizes:
                best_name = None
                best_value = None
                for variant in LEARNED_CONDITIONS:
                    key = f"{target.identifier}|{regime}|{size}|{variant}"
                    value = float(
                        cells[key]["excess_aggregate"]["mean"]  # type: ignore[index]
                    )
                    if best_value is None or value < best_value:
                        best_value, best_name = value, variant
                assert best_name is not None
                chosen = dict(
                    cells[f"{target.identifier}|{regime}|{size}|{best_name}"]
                )
                chosen["selected_variant"] = best_name
                cells[f"{target.identifier}|{regime}|{size}|U_best"] = chosen


def decide_verdict(
    cells: dict[str, dict[str, object]],
    targets: Sequence[Target],
    agreements: dict[str, dict[str, int]],
    certificate: dict[str, object],
    sizes: Sequence[int],
) -> dict[str, object]:
    """Score the 017.26b section 11 criteria and choose exactly one verdict."""
    structure = {
        block["fixture"]: block for block in certificate["fixtures"]  # type: ignore[union-attr]
    }
    burden: dict[str, object] = {}
    per_fixture: dict[str, object] = {}
    for target in targets:
        block = structure[target.identifier]
        derived_series = _series(cells, target.identifier, "A", "D", sizes)
        comparators: dict[str, object] = {}
        derived_at = _to_threshold(sizes, derived_series, THRESHOLDS[0])
        for comparator in ("U_best", "S", "R"):
            series = _series(cells, target.identifier, "A", comparator, sizes)
            reached = _to_threshold(sizes, series, THRESHOLDS[0])
            if derived_at is None:
                material = False
            elif reached is None:
                material = True
            else:
                material = reached >= MATERIAL_FACTOR * derived_at
            comparators[comparator] = {
                "n_at_threshold": reached,
                "materially_worse_than_D": material,
                "excess_at_1024": _round(series[-1]),
            }
        burden[target.identifier] = {
            "threshold": THRESHOLDS[0],
            "D_n_at_threshold": derived_at,
            "D_excess_at_1024": _round(derived_series[-1]),
            "comparators": comparators,
            "material_against_at_least_one_uncompiled_comparator": any(
                bool(value["materially_worse_than_D"])  # type: ignore[index]
                for name, value in comparators.items()
                if name in ("U_best", "S")
            ),
        }
        shortcut = block["D_frozen_shortcut_language"]
        non_local = block["D_disclosed_non_local_adversary"]
        per_fixture[target.identifier] = {
            "admitted_inputs": target.width,
            "structure_preserving_group_order": (
                target.compiled.group_order
            ),
            "derived_cells": target.cell_count,
            "derived_cell_cardinalities": target.compiled.cell_sizes,
            "D_and_H_degrees_of_freedom": target.cell_count,
            "R_degrees_of_freedom": len(set(target.shortcut)),
            "S_degrees_of_freedom": target.width,
            "U_parameter_counts": {
                name: _parameter_count(design, HIDDEN_OF[name])
                for name, design in sorted(target.designs.items())
            },
            "declared_primitives": len(target.world.primitives),
            "primitives_are_jointly_load_bearing": bool(
                block["G_joint_load_bearing"][  # type: ignore[index]
                    "primitives_are_jointly_load_bearing"
                ]
            ),
            "frozen_shortcut_language_recovers_the_quotient": bool(
                shortcut["the_whole_language_read_jointly_recovers_the_quotient"]  # type: ignore[index]
            ),
            "disclosed_non_local_adversary_recovers_the_quotient": bool(
                non_local["recovers_the_derived_quotient"]  # type: ignore[index]
            ),
            "every_perturbation_matched_its_frozen_direction": all(
                bool(row["prediction_held"])
                for row in block["F_source_perturbations"]  # type: ignore[union-attr]
            ),
            "D_equals_H_in_every_replicate": (
                agreements[target.identifier]["D_versus_H_mismatches"] == 0
            ),
            "D_equals_H_comparisons": agreements[target.identifier][
                "comparisons"
            ],
            "D_and_R_coincided_in": agreements[target.identifier][
                "D_versus_R_identical"
            ],
        }

    criteria = {
        "no_symmetry_or_equivalence_metadata_is_declared": all(
            bool(
                structure[target.identifier]["B_anti_shortcut_audit"][  # type: ignore[index]
                    "structural_subtree_is_clean"
                ]
            )
            for target in targets
        ),
        "generic_compilation_derived_the_groups_and_quotients_before_theta": (
            bool(certificate["all_mechanical_checks_pass"])
            and all(
                bool(
                    structure[target.identifier][  # type: ignore[index]
                        "E_independence_of_the_derivation"
                    ]["relabelling"]["quotient_is_identical"]
                )
                for target in targets
            )
        ),
        "D_equals_H_statistically": all(
            agreements[target.identifier]["D_versus_H_mismatches"] == 0
            for target in targets
        )
        and all(
            tuple(target.compiled.quotient) == tuple(target.handed)
            for target in targets
        ),
        "the_quotient_is_not_recoverable_by_the_frozen_shortcut_language": all(
            not bool(
                structure[target.identifier]["D_frozen_shortcut_language"][  # type: ignore[index]
                    "the_whole_language_read_jointly_recovers_the_quotient"
                ]
            )
            for target in targets
        ),
        "source_perturbations_behaved_as_structural_preservation_predicts": all(
            all(
                bool(row["prediction_held"])
                for row in structure[target.identifier][  # type: ignore[union-attr]
                    "F_source_perturbations"
                ]
            )
            for target in targets
        ),
        "the_richer_fixture_demonstrates_joint_primitive_load_bearing": bool(
            structure[targets[1].identifier]["G_joint_load_bearing"][  # type: ignore[index]
                "primitives_are_jointly_load_bearing"
            ]
        ),
        "D_shows_materially_lower_labelled_data_burden_on_both_fixtures": all(
            bool(
                burden[target.identifier][  # type: ignore[index]
                    "material_against_at_least_one_uncompiled_comparator"
                ]
            )
            for target in targets
        ),
    }

    structural = {
        name: value
        for name, value in criteria.items()
        if name != "D_shows_materially_lower_labelled_data_burden_on_both_fixtures"
    }
    if not criteria["D_equals_H_statistically"]:
        verdict = "IMPLEMENTATION DEFECT"
        why = (
            "the derived compiler and the handed-quotient oracle did not agree, "
            "so 017.26b section 8 requires repair before anything is inferred"
        )
    elif not criteria[
        "the_quotient_is_not_recoverable_by_the_frozen_shortcut_language"
    ]:
        verdict = "CHEAP-STATISTIC COLLAPSE"
        why = (
            "a member of the frozen shortcut language recovers the relevant "
            "quotient, so the structural derivation is unnecessary on that "
            "fixture"
        )
    elif not all(structural.values()):
        verdict = "NEGATIVE"
        why = (
            "the compiler did not derive the quotient from ordinary world "
            "structure under the declared controls: "
            f"{sorted(name for name, ok in structural.items() if not ok)}"
        )
    elif all(criteria.values()):
        verdict = "STRONG POSITIVE"
        why = (
            "on these finite worlds, ordinary typed source structure was "
            "sufficient for a generic exact compiler to derive useful invariance "
            "before labels, shrinking the statistical family learned from data"
        )
    else:
        verdict = "STRUCTURAL POSITIVE"
        why = (
            "typed world structure was compiled into exact pre-label invariance, "
            "but this experiment does not establish an additional "
            "training-economy advantage over the tested generic learners"
        )
    return {
        "rule": (
            "017.26b section 11. Strong positive requires both fixtures and all "
            "seven criteria. Structural positive is criteria one to six with the "
            "labelled-data-burden criterion failing. Cheap-statistic collapse is "
            "a frozen L_k statistic recovering the quotient. Implementation "
            "defect is checked first and outranks every scientific verdict."
        ),
        "criteria": criteria,
        "per_fixture": per_fixture,
        "labelled_data_burden": burden,
        "verdict": verdict,
        "why": why,
        "interpretation_permitted": (
            "under the preregistered learners, deterministic structural "
            "compilation reached a smaller statistical family before labels, and "
            "the tested uncompiled learners required more labelled examples to "
            "reach the same predictive criterion"
        ),
        "interpretation_forbidden": [
            "that Training requires N labels unless a schema compiler supplies "
            "the quotient",
            "that the schema contains information unavailable to U; D and U "
            "receive the same source information",
            "any learner-independent sample-complexity claim; this turn proves "
            "no lower bound",
            "that TDM compilation is automorphism quotienting; Q_aut is one "
            "preregistered exact structural compiler",
            "that either fixture is representative of anything beyond itself",
        ],
    }


# ---------------------------------------------------------------------------
# 7. Assembly
# ---------------------------------------------------------------------------


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
        "phase_s_freeze": (
            "quilt+s3://protology#package=occurrence/gpt@"
            f"{PHASE_S_QUILT_REVISION}&path=issues/017-jev-pivot/"
            "017.26c-Kiro-phase-S-structure-freeze.md"
        ),
        "phase_s_commit": PHASE_S_COMMIT,
        "prior_result_commit": PRIOR_RESULT_COMMIT,
        "byte_pinned_inputs": {
            "derived_symmetry.py": PHASE_S_MODULE_SHA256,
            "phase_s_certificate.json": PHASE_S_CERTIFICATE_SHA256,
            "preregistration.json": PREREGISTRATION_SHA256,
            "../017.24a-Code-attachments/quotient_refinement.py": (
                REFINE_MODULE_SHA256
            ),
            "../017.24-Code-attachments/training_economy.py": (
                SHARED_MODULE_SHA256
            ),
        },
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
            exact = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in reference["predictions"][  # type: ignore[index]
                    f"{target.identifier}|A|{spec.name}"
                ]["sweep"]
            }
            for size in sizes:
                block = cells[
                    f"{target.identifier}|A|{size}|{spec.name}"
                ]["excess_aggregate"]  # type: ignore[index]
                mean = float(block["mean"])  # type: ignore[index]
                high = float(block["ci95_high"])  # type: ignore[index]
                standard = max((high - mean) / 1.959963984540054, 1e-15)
                sigma = abs(mean - exact[size]) / standard
                worst = max(worst, sigma)
                ok = sigma <= 4.0
                failures += int(not ok)
                rows.append(
                    {
                        "cell": f"{target.identifier}|A|{size}|{spec.name}",
                        "simulated": _round(mean),
                        "predicted": _round(exact[size]),
                        "sigma": _round(sigma),
                        "within_four_sigma": ok,
                    }
                )
    return {
        "method": (
            "the exact expectation must lie within four standard errors of the "
            "simulated mean for every closed-form condition in every regime A "
            "cell; the standard error is recovered from the reported confidence "
            "interval so the check is reproducible from the artifact alone"
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
    closed_replicates = (
        QUICK_CLOSED_FORM_REPLICATES if quick else CLOSED_FORM_REPLICATES
    )
    learned_replicates = (
        QUICK_LEARNED_REPLICATES if quick else LEARNED_REPLICATES
    )

    moment = time.perf_counter()
    certificate = load_phase_s_certificate()
    targets = build_targets()
    compile_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    reference = exact_reference(targets, sizes, closed_replicates)
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
            closed_replicates,
            learned_replicates,
            ALPHA_PRIMARY,
            BETA_PRIMARY,
            rows,
        )
        cells.update(block)
        agreements[target.identifier] = agreement
    closed_seconds = time.perf_counter() - moment

    moment = time.perf_counter()
    for target_index, target in enumerate(targets):
        cells.update(
            run_learned(target, target_index, sizes, learned_replicates, rows)
        )
    learned_seconds = time.perf_counter() - moment

    select_best_uncompiled(cells, targets, sizes)
    validation = _prediction_validation(cells, reference, targets, sizes)
    verdict = decide_verdict(cells, targets, agreements, certificate, sizes)

    secondary: dict[str, object] = {}
    for target_index, target in enumerate(targets):
        block, _agreement = run_closed_form(
            target,
            target_index,
            (sizes[0], sizes[len(sizes) // 2], sizes[-1]),
            max(40, closed_replicates // 10),
            max(40, closed_replicates // 10),
            ALPHA_SECONDARY,
            BETA_SECONDARY,
            [],
        )
        secondary.update(block)

    checks: dict[str, bool] = {
        "the_phase_s_module_bytes_match_the_pin": True,
        "the_phase_s_certificate_bytes_match_the_pin": True,
        "the_017_24a_implementation_bytes_match_the_pin": True,
        "the_017_24_implementation_bytes_match_the_pin": True,
        "the_phase_s_certificate_passed_every_check_of_its_own": bool(
            certificate["all_mechanical_checks_pass"]
        ),
        "the_compiler_reproduced_its_banked_quotient_on_every_fixture": all(
            tuple(target.compiled.quotient) == tuple(target.handed)
            for target in targets
        ),
        "D_and_H_agreed_in_every_replicate": all(
            agreement["D_versus_H_mismatches"] == 0
            for agreement in agreements.values()
        ),
        "every_regime_A_cell_landed_within_four_sigma": bool(
            validation["all_within_four_sigma"]
        ),
        "the_verdict_is_one_of_the_preregistered_outcomes": verdict[
            "verdict"
        ]
        in (
            "STRONG POSITIVE",
            "STRUCTURAL POSITIVE",
            "CHEAP-STATISTIC COLLAPSE",
            "NEGATIVE",
            "IMPLEMENTATION DEFECT",
        ),
    }
    for target in targets:
        tag = target.identifier
        sufficiency = sufficiency_certificate(target)
        span = span_certificate(target.designs["quadratic"], target.compiled.quotient)
        checks[f"fixture_{tag}_theta_star_is_interior_and_unequal"] = bool(
            sufficiency["every_probability_is_interior"]
        ) and bool(sufficiency["not_all_probabilities_are_equal"])
        checks[
            f"fixture_{tag}_every_coarser_declared_partition_carries_a_floor"
        ] = bool(
            sufficiency["every_coarser_declared_partition_carries_a_positive_floor"]
        )
        checks[f"fixture_{tag}_the_learners_design_contains_every_cell"] = bool(
            span["every_cell_indicator_lies_in_the_span"]
        )
        checks[f"fixture_{tag}_the_learners_design_is_not_crippled"] = bool(
            span["a_single_input_indicator_also_lies_in_the_span"]
        )
        checks[
            f"fixture_{tag}_the_wrong_compiler_control_respects_no_derived_map"
        ] = not bool(target.wrong_certificate["respects_the_derived_maps"])
        checks[f"fixture_{tag}_the_saturated_model_carries_the_most_parameters"] = (
            target.width > target.cell_count
        )
        checks[f"fixture_{tag}_D_is_strictly_finer_than_R"] = phase_s._refines(
            target.compiled.quotient, target.shortcut
        ) and not phase_s._refines(target.shortcut, target.compiled.quotient)
        derived = _series(cells, tag, "A", "D", sizes)
        checks[f"fixture_{tag}_D_improves_monotonically_from_16_upward"] = all(
            later <= earlier + 1e-9
            for earlier, later in zip(derived[2:-1], derived[3:], strict=True)
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
                    "fixture": target.identifier,
                    "admitted_inputs": target.width,
                    "structure_preserving_group_order": (
                        target.compiled.group_order
                    ),
                    "derived_cells": target.cell_count,
                    "derived_cell_cardinalities": target.compiled.cell_sizes,
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
                    "feature_blocks": target.feature_blocks,
                    "sufficiency_certificate": sufficiency_certificate(target),
                    "span_certificate": span_certificate(
                        target.designs["quadratic"], target.compiled.quotient
                    ),
                    "wrong_compiler_control": target.wrong_certificate,
                }
                for target in targets
            ],
            "sample_sizes": list(sizes),
            "replicates": {
                "closed_form": closed_replicates,
                "learned": learned_replicates,
            },
            "regimes": {
                "A": (
                    "train and evaluate under the uniform distribution on every "
                    "admitted input; exact expected loss in closed form"
                ),
                "B": (
                    "hold out whole input identities stratified by derived cell; "
                    "train only on observations from the seen identities; "
                    "evaluate exactly on the unseen ones"
                ),
            },
            "estimator": {
                "primary": "beta(1,1) posterior mean, (k + 1) / (m + 2)",
                "secondary": "beta(1/2,1/2) posterior mean",
                "shared_by": "every closed-form condition, refit from the same "
                "per-input counts",
            },
            "learned_conditions": list(LEARNED_CONDITIONS),
        },
        "cells": cells,
        "secondary_prior_cells": secondary,
        "agreement": agreements,
        "prediction_validation": validation,
        "K_verdict": verdict,
        "L_claim_ledger": claim_ledger(targets, certificate),
        "mechanical_checks": checks,
        "mechanical_check_count": len(checks),
        "all_mechanical_checks_pass": all(checks.values()),
        "fences": [
            "no language-scale claim and no comparison to any language model",
            "no physical Test Realization and no autonomous dynamics",
            "no Interact",
            "OT is not claimed to be responsible for any measured gain; 017.25 "
            "and 017.25a established T = O = G for the measured statistical "
            "behaviour and nothing here disturbs it",
            "neither fixture is representative of anything beyond itself",
            "Issue 018 is not resolved",
            "nothing was added to the stable decision-model public API",
        ],
    }
    cost = {
        "schema": COST_SCHEMA,
        "phase": "T",
        "authority": authority_record(),
        "seconds": {
            "compile_and_assemble": _round(compile_seconds),
            "exact_reference": _round(reference_seconds),
            "closed_form_fitting": _round(closed_seconds),
            "learned_fitting": _round(learned_seconds),
            "total_wall_clock": _round(time.perf_counter() - started),
        },
        "reading": (
            "derivation cost is the compile column and it is a fraction of a "
            "second on both worlds, against minutes of fitting. That is a fact "
            "about these worlds and not a general claim: the search is "
            "exhaustive over a product of symmetric groups and is factorial in "
            "carrier cardinality in the worst case."
        ),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    return Bundle(
        reference=reference,
        results=results,
        cost=cost,
        table=_canonical_table(rows),
    )


def claim_ledger(
    targets: Sequence[Target], certificate: dict[str, object]
) -> dict[str, object]:
    """The 017.26b section 12 ledger, separated by how each item was obtained."""
    return {
        "declared": [
            "two typed carriers per world, with their elements",
            "one binary relation on the single-relation world; one binary "
            "relation and one total typed map on the richer world",
            "the admitted-input product type of each world",
            "a binary response type with two neutral labels",
        ],
        "derived_before_any_label": [
            f"|Aut| = {target.compiled.group_order} and "
            f"{target.cell_count} input cells with cardinalities "
            f"{target.compiled.cell_sizes} on fixture {target.identifier}"
            for target in targets
        ]
        + [
            "the frozen shortcut language L_k at depth 2 and its result on both "
            "worlds",
            "the disclosed non-local pair-refinement result on both worlds",
            "the derived quotient of every erased, scrambled, and "
            "symmetry-broken world",
            "the maximality witnesses and the relabelling and key-stripping "
            "independence audits",
        ],
        "learned_from_labels": [
            f"{target.cell_count} observable scalars for D and H on fixture "
            f"{target.identifier}"
            for target in targets
        ]
        + [
            "one observable scalar per L_k cell for R",
            "one observable scalar per admitted input for S",
            "the weights of every uncompiled variant",
        ],
        "implementation_choice": [
            "exhaustive backtracking rather than a canonicalization library",
            "the variable-ordering heuristic, which affects pruning only",
            "asking the orbit question pair by pair rather than materializing "
            "the group",
            "the depth k = 2 bound on the shortcut language",
            "the reduction sizes used for the unpruned brute-force witness",
            "beta(1,1) as the primary estimator and beta(1/2,1/2) as secondary",
            "Adam, the step budget, the ridge grid, and the hidden width for "
            "the uncompiled learners",
            "the 0.02 excess-log-loss threshold and the factor of two that "
            "makes a burden difference material",
            "canonical JSON serialization and float rounding for reproducible "
            "digests",
        ],
        "still_open": [
            "whether Q_aut is the right compiler outside these two worlds",
            "whether another structural quotient is preferable; definability, "
            "context, congruence, bisimulation-like, and task-relative "
            "constructions are untouched",
            "tractability at larger schema sizes; the disclosed pair refinement "
            "already reaches the same partition far more cheaply on both worlds",
            "any learner-independent sample-complexity claim, which would need "
            "a lower bound this turn does not provide",
            "how a real-domain schema would be constructed",
            "any OT-specific contribution to training economy",
            "physical realization and Interact",
            "Issue 018 instance identity",
        ],
        "phase_s_open_items_carried_forward": certificate[
            "what_this_freeze_does_not_establish"
        ],
    }


# ---------------------------------------------------------------------------
# 8. Verification and the command line
# ---------------------------------------------------------------------------


def verify(bundle: Bundle) -> list[str]:
    problems: list[str] = []
    expected = _canonical(bundle.reference)
    if not REFERENCE_PATH.exists():
        problems.append(f"missing artifact: {REFERENCE_PATH.name}")
    elif REFERENCE_PATH.read_bytes() != expected:
        problems.append(
            f"{REFERENCE_PATH.name} mismatch: observed "
            f"{_sha256(REFERENCE_PATH.read_bytes())}, expected "
            f"{_sha256(expected)}"
        )
    if not RESULTS_PATH.exists():
        problems.append(f"missing artifact: {RESULTS_PATH.name}")
    elif not refine._numeric_close(
        json.loads(RESULTS_PATH.read_text()),
        json.loads(_canonical(bundle.results)),
        NUMERIC_TOLERANCE,
    ):
        problems.append(
            f"{RESULTS_PATH.name} differs beyond the declared numeric "
            f"tolerance {NUMERIC_TOLERANCE}"
        )
    for path in (TABLE_PATH, COST_PATH):
        if not path.exists():
            problems.append(f"missing artifact: {path.name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="emit the exact reference without generating a single label",
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
        help="reduced grid for development; refuses to write artifacts",
    )
    args = parser.parse_args(argv)

    if args.quick and (args.write or args.check):
        print("--quick cannot write or verify artifacts", file=sys.stderr)
        return 2

    if args.predict:
        targets = build_targets()
        payload = {
            "reference": exact_reference(
                targets, SAMPLE_SIZES, CLOSED_FORM_REPLICATES
            ),
            "sufficiency": {
                target.identifier: sufficiency_certificate(target)
                for target in targets
            },
        }
        sys.stdout.buffer.write(_canonical(payload))
        return 0

    bundle = run_everything(quick=args.quick)

    if args.check:
        problems = verify(bundle)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        print(
            "017.26 Phase T verified: reference "
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

    sys.stdout.buffer.write(_canonical(bundle.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
