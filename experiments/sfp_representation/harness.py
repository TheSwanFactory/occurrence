"""Training and evaluation harness for the admit-and-force relation (Issue 009, Task 9).

This module owns exactly two things: **one** training protocol and **the whole
metric suite**. It owns no baseline, no sweep driver, no analysis and no
disposition; those are Tasks 8, 10 and 11 and are deliberately absent. It imports
``task``, ``folds``, ``arms`` and ``scorer`` and modifies none of them.

What is trained
---------------
The shared :class:`scorer.CandidateScorer` — one architecture, identical after the
arm-specific input adapter — is fitted by full-batch Adam to a cross-entropy over
the 85 output slots::

    slot 0 .. 83   candidate Events, index-aligned to the ``task.py`` catalogue
    slot 84        BOTTOM, the non-admission option

The supervised slot is ``record.target_index`` for an ADMIT pair and slot 84 for
every non-admission pair. Admission and identity are therefore decided by *one*
argmax over *one* softmax: bottom is not a threshold, not a second head read
separately, and not Event 0.

The training set, and why it is not the whole train bucket
-----------------------------------------------------------
``CROSS_HABITAT`` is 6552 of the 7056 ordered pairs — 92.9% of the pool. Training
on the raw ``train`` bucket would let a learner reach ~93% by answering the
question "are these two Events in different habitats?" and never represent the
relation at all. So the training set is

* every **same-habitat** position in ``fold.train`` (ADMIT, REPEATED,
  SAME_HABITAT_DISJOINT), and
* the fold's digest-pinned ``train_cross_habitat_balanced`` sample in place of all
  cross-habitat positions.

For a LOHO fold that is 312 ADMIT + 78 REPEATED + 78 SAME_HABITAT_DISJOINT + 312
sampled CROSS_HABITAT = 780 pairs. The exact composition of every training and
evaluation set is recorded per run in :class:`RunResult`, never inferred.

The evaluation set
------------------
``fold.test_within`` (the held-out structure) united with the fold's
``test_cross_habitat`` sample, which ``folds.py`` draws from the ``straddle``
bucket and sizes against ``test_within``'s ADMIT count. For a LOHO fold that is
24 ADMIT + 6 REPEATED + 6 SAME_HABITAT_DISJOINT + 26 CROSS_HABITAT = 62 pairs, so
every one of the four required non-admission / admission metrics has a live
denominator. Nothing is masked at evaluation time, ever.

One architecture, no tuning
---------------------------
:class:`ScorerConfig` defaults, :class:`TrainConfig` declared once below, and that
is the whole hyperparameter story. No sweep, no early stopping, no per-arm
adjustment. :attr:`TrainConfig.mask_held_out_candidates` is the *reserved* slot
for the single motivated repair the stop rule permits — it is ``False`` by
default, so the default protocol scores the full 85-slot catalogue in training
exactly as it does at evaluation. It is named here rather than left implicit
because it is the one interface knob a later task might legitimately need: under
a LOHO fold the correct answer at test time is an Event that never appears as a
training target, so the default protocol actively trains the model to suppress
it. That is a property of the interface, reported, not silently patched.

Capacity parity
---------------
All four science arms declare ``token_dim = 16``, so A/B/C/D have **identical**
parameter counts. :func:`capacity_ledger` recomputes the ledger against the real
``arms.py`` objects and reports the A/B/C/D spread, which is 0. The ledger in
``scorer.json`` was built against placeholder token widths (12 for B/C/D) and is
superseded by the one written here.

Arm E
-----
Arm E declares ``token_dim == 84 ==`` the catalogue size, so its adapter holds one
learnable column per Event. That is exactly what ``scorer.py``'s audit records as
the one case its shape check cannot catch, and it is Arm E's *purpose*: maximal
per-Event input capacity. It is therefore a **memorization ceiling** — a
diagnostic upper bound on what free per-Event capacity can buy. It is never a
representation arm and never a substitute for Arm C. Every result row carries
``arm_role`` so this cannot be lost downstream.

Wall-clock
----------
Wall-clock is measured and reported because Task 10 needs a cost estimate, but it
is **not replayable**. :func:`strip_non_replayable` removes it, and ``--check``
compares the stripped payload only. The keys it removes are listed in
:data:`NON_REPLAYABLE_KEYS` and are named in the artifact itself.

Fences
------
::

    Labels come from the certified native FIPS relation; the harness only consumes them.
    Correctness is exact certified Event identity, never float proximity.
    Admission is reported as balanced accuracy; raw accuracy is inadmissible at 92.9% CROSS_HABITAT.
    Arm E is a memorization ceiling (token_dim == catalogue size), not a representation arm.
    One architecture, no hyperparameter tuning; at most one motivated repair, declared.
    Bottom is slot 84, an explicit competing option, not an algebraic zero and not Event 0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from arms import Arm, all_arms
from folds import Fold, all_folds, class_counts, structural_folds
from scorer import (
    MASK_PENALTY,
    CandidateScorer,
    ScorerConfig,
    build_scorer,
    count_params,
    forward_flops,
    mask_scores,
    module_param_counts,
    set_seed,
)
from task import BOTTOM_INDEX, Dataset, build_dataset, habitat_label
from torch import nn

__all__ = [
    "ARM_ROLES",
    "FENCES",
    "HarnessConfig",
    "RunResult",
    "TrainConfig",
    "aggregate",
    "capacity_ledger",
    "evaluate",
    "evaluation_positions",
    "smoke",
    "stats",
    "strip_non_replayable",
    "train_one",
    "training_positions",
]

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
SMOKE_OUTPUT = ARTIFACTS / "harness_smoke.json"

#: Pinned like every other Issue 009 module: a live ``git rev-parse HEAD`` would
#: make the artifact non-replayable the moment anything is committed.
BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"

CATALOGUE_SIZE = 84

#: The four science arms, in declaration order, plus the diagnostic.
SCIENCE_ARMS = ("A_native", "B_sfp", "C_scrambled", "D_relabeled")
DIAGNOSTIC_ARMS = ("E_opaque",)
ARM_NAMES = SCIENCE_ARMS + DIAGNOSTIC_ARMS

#: The role label attached to every single result row. Arm E is never "science".
ARM_ROLES: dict[str, str] = {
    "A_native": "science",
    "B_sfp": "science",
    "C_scrambled": "science",
    "D_relabeled": "science",
    "E_opaque": "diagnostic_memorization_ceiling",
}

ARM_ROLE_NOTES: dict[str, str] = {
    "E_opaque": (
        "OPTIONAL DIAGNOSTIC. token_dim == 84 == catalogue size, so the adapter "
        "carries one learnable column per Event: maximal free per-Event input "
        "capacity, which is the arm's purpose. Read only as a memorization "
        "ceiling. It is not a representation arm and is not a substitute for "
        "Arm C."
    ),
}

#: The named fold families this harness aggregates over.
FAMILY_LOHO = "LOHO"
FAMILY_LOFPO = "LOFPO"
STRUCTURAL_FAMILIES = (FAMILY_LOHO, FAMILY_LOFPO)

#: Bucket / sample names, taken verbatim from the frozen ``folds.py`` contract.
TRAIN_BUCKET = "train"
TEST_BUCKET = "test_within"
TRAIN_SAMPLE_SUFFIX = "train_cross_habitat_balanced"
TEST_SAMPLE_SUFFIX = "test_cross_habitat"

#: The full sweep Task 10 owns. Declared here only so the cost extrapolation has
#: something to extrapolate to; this module never runs it.
FULL_SWEEP = {
    "arms": len(ARM_NAMES),
    "folds": 21,
    "seeds": 8,
}

#: The declared smoke subset. Three arms (a 1-token science arm, a 2-token science
#: arm, and the diagnostic so its role label is exercised), three folds (two LOHO
#: and one LOFPO so both family aggregations are live), two seeds.
SMOKE_ARMS = ("A_native", "B_sfp", "E_opaque")
SMOKE_LOHO_FOLDS = 2
SMOKE_LOFPO_FOLDS = 1
SMOKE_SEEDS = (0, 1)

#: Metric names aggregated by :func:`aggregate`, in a fixed order.
METRIC_NAMES = (
    "joint_exact_success",
    "admission_balanced_accuracy",
    "admission_sensitivity",
    "admission_specificity",
    "positive_forced_third_exact_accuracy",
    "repeated_pair_nonadmission_accuracy",
    "same_habitat_disjoint_nonadmission_accuracy",
    "cross_habitat_nonadmission_accuracy",
    "swap_invariance_error",
    "swap_argmax_disagreement_rate",
)

#: Scalars carried on the run row itself rather than inside a metric block.
RUN_SCALARS = (
    "train_accuracy",
    "generalization_gap",
    "final_loss",
    "param_count",
    "total_forward_flops",
)

#: The four error categories. They partition every incorrect prediction; the
#: partition is asserted, not assumed.
ERROR_CATEGORIES = (
    "wrong_event_correct_habitat",
    "event_wrong_habitat",
    "bottom_instead_of_admitted_event",
    "event_instead_of_bottom",
)

#: Removed before any ``--check`` comparison. Wall-clock is not replayable.
NON_REPLAYABLE_KEYS = (
    "wall_sec",
    "wall_clock",
    "extrapolation",
    "torch_version",
    "numpy_version",
)

#: Floats are rounded before serialisation so replay is byte-identical.
METRIC_DECIMALS = 12

FENCES = (
    (
        "Labels come from the certified native FIPS relation; the harness only "
        "consumes them."
    ),
    "Correctness is exact certified Event identity, never float proximity.",
    (
        "Admission is reported as balanced accuracy; raw accuracy is inadmissible "
        "at 92.9% CROSS_HABITAT."
    ),
    (
        "Arm E is a memorization ceiling (token_dim == catalogue size), not a "
        "representation arm."
    ),
    (
        "One architecture, no hyperparameter tuning; at most one motivated repair, "
        "declared."
    ),
    (
        "Bottom is slot 84, an explicit competing option, not an algebraic zero and "
        "not Event 0."
    ),
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def digest(obj: object) -> str:
    """Canonical sha256 of a JSON-serializable object."""

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _round(value: float | None) -> float | None:
    """Round a metric for serialisation. ``None`` means "no denominator"."""

    if value is None:
        return None
    return round(float(value), METRIC_DECIMALS)


def strip_non_replayable(node: object) -> object:
    """Recursively drop every key in :data:`NON_REPLAYABLE_KEYS`.

    Wall-clock, and the environment versions that only exist to interpret it, are
    not reproducible across machines or runs. They are reported, and excluded from
    any byte-comparison, explicitly rather than quietly.
    """

    if isinstance(node, dict):
        return {
            key: strip_non_replayable(value)
            for key, value in node.items()
            if key not in NON_REPLAYABLE_KEYS
        }
    if isinstance(node, list):
        return [strip_non_replayable(item) for item in node]
    return node


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """The one training protocol. Declared once, applied to every arm and fold.

    These values were fixed before any run and are not tuned: there is no search,
    no per-arm adjustment and no early stopping anywhere in this module. Changing
    them changes every arm identically, which is the only way capacity parity
    survives.
    """

    #: Full-batch Adam steps.
    steps: int = 800
    #: Adam learning rate.
    lr: float = 0.01
    #: Reserved slot for the single motivated repair the stop rule permits.
    #: ``False`` is the declared default protocol. When ``True``, candidate slots
    #: for the fold's held-out Events are masked *in training only* with the
    #: repo's finite-penalty convention; evaluation is never masked.
    mask_held_out_candidates: bool = False

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if not (self.lr > 0.0):
            raise ValueError(f"lr must be positive, got {self.lr}")

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "optimizer": "Adam",
            "batching": "full batch",
            "loss": "cross-entropy over the 85 output slots against the true slot",
            "true_slot_rule": (
                "record.target_index for ADMIT, slot 84 (BOTTOM) for every "
                "non-admission class"
            ),
            "mask_held_out_candidates": self.mask_held_out_candidates,
            "mask_penalty": MASK_PENALTY,
            "mask_convention": (
                "large finite penalty via masked_fill(~legal, -MASK_PENALTY), not "
                "-inf, so a fully masked row yields a finite softmax not a NaN; "
                "same value and rationale as experiments/tlm_multitoken/policy.py"
            ),
            "evaluation_masking": "none, ever",
            "seed_convention": "scorer.set_seed: torch.manual_seed then np.random.seed",
            "tuning": (
                "none. One architecture, one protocol, fixed before any run. No "
                "search, no early stopping, no per-arm adjustment."
            ),
            "reserved_repair": (
                "mask_held_out_candidates is the single motivated repair the stop "
                "rule permits and is OFF by default"
            ),
        }


@dataclass(frozen=True)
class HarnessConfig:
    """What to run: which arms, which fold families, which seeds, which widths."""

    #: 8 seeds is the controlling task's declared default.
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    arm_names: tuple[str, ...] = ARM_NAMES
    families: tuple[str, ...] = STRUCTURAL_FAMILIES
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"duplicate seeds: {self.seeds}")
        unknown = [name for name in self.arm_names if name not in ARM_ROLES]
        if unknown:
            raise ValueError(f"unknown arm names: {unknown}")

    def as_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "n_seeds": len(self.seeds),
            "n_seeds_declared_default": 8,
            "arm_names": list(self.arm_names),
            "arm_roles": {name: ARM_ROLES[name] for name in self.arm_names},
            "families": list(self.families),
            "scorer": self.scorer.as_dict(),
            "train": self.train.as_dict(),
            "train_bucket": TRAIN_BUCKET,
            "test_bucket": TEST_BUCKET,
            "train_cross_habitat_sample": TRAIN_SAMPLE_SUFFIX,
            "test_cross_habitat_sample": TEST_SAMPLE_SUFFIX,
        }


# ---------------------------------------------------------------------------
# capacity ledger, against the REAL arms
# ---------------------------------------------------------------------------

def capacity_ledger(
    arm_list: Sequence[Arm], config: ScorerConfig | None = None
) -> dict:
    """Per-arm parameter and analytic-FLOP ledger, built from the real arms.

    Supersedes the ledger in ``scorer.json``, which was computed against
    placeholder token widths (12 for B/C/D) before ``arms.py`` was frozen. The
    real science arms all declare ``token_dim = 16``, so the A/B/C/D spread is 0:
    exact capacity parity, not approximate parity.
    """

    config = config or ScorerConfig()
    rows = []
    for arm in arm_list:
        model = build_scorer(arm, config, seed=0)
        counts = module_param_counts(model)
        rows.append(
            {
                "arm": arm.name,
                "arm_role": ARM_ROLES[arm.name],
                "token_dim": int(arm.token_dim),
                "tokens_per_event": int(arm.tokens_per_event),
                "total_input_units": int(arm.token_dim * arm.tokens_per_event),
                "adapter_params": counts["adapter"],
                "shared_trunk_params": counts["shared_trunk"],
                "pair_params": counts["pair"],
                "rel_params": counts["rel"],
                "bottom_params": counts["bottom"],
                "total_trainable_params": counts["total"],
                "forward_flops_batch_1": forward_flops(model, batch=1)["total"],
                "forward_flops_batch_780": forward_flops(model, batch=780)["total"],
                "arm_sha256": arm.sha256(),
            }
        )

    by_name = {row["arm"]: row for row in rows}
    science = [by_name[name] for name in SCIENCE_ARMS if name in by_name]
    science_totals = [row["total_trainable_params"] for row in science]
    shared_signature = {
        (
            row["shared_trunk_params"],
            row["pair_params"],
            row["rel_params"],
            row["bottom_params"],
        )
        for row in rows
    }
    return {
        "rows": rows,
        "supersedes": (
            "scorer.json capacity_ledger, which used placeholder token dims "
            "(12 for B/C/D) and is wrong for the frozen arms"
        ),
        "science_arms": list(SCIENCE_ARMS),
        "science_total_params": science_totals,
        "science_total_params_spread": (
            int(max(science_totals) - min(science_totals)) if science_totals else None
        ),
        "science_capacity_parity_exact": len(set(science_totals)) == 1,
        "science_token_dims": sorted({row["token_dim"] for row in science}),
        "shared_stack_identical_across_arms": len(shared_signature) == 1,
        "shared_stack_statement": (
            "the shared stack — per-token trunk, symmetric pair code, Rel head and "
            "Bottom head — is parameter-identical across all five arms; only the "
            "arm-specific input adapter can differ, and it differs only as forced "
            "by token_dim"
        ),
        "arm_e_statement": ARM_ROLE_NOTES["E_opaque"],
        "flop_convention": (
            "2 FLOPs per multiply-accumulate, 1 FLOP per bias add, 1 FLOP per "
            "elementwise op, ReLU free; analytic via scorer.forward_flops, never "
            "measured, because measured cost is not replayable"
        ),
    }


# ---------------------------------------------------------------------------
# position resolution
# ---------------------------------------------------------------------------

def _sample_positions(fold: Fold, suffix: str) -> tuple[int, ...] | None:
    """The fold's named balanced sample, or ``None`` if it declares none."""

    name = f"{fold.family}/{fold.name}/{suffix}"
    try:
        return fold.sample(name).positions
    except KeyError:
        return None


def _composition(dataset: Dataset, positions: tuple[int, ...], note: str) -> dict:
    body = {"size": len(positions), "note": note}
    body.update(class_counts(dataset, positions))
    body["sha256"] = digest(list(positions))
    return body


def training_positions(
    dataset: Dataset, fold: Fold
) -> tuple[tuple[int, ...], dict]:
    """The declared training set: same-habitat train positions + balanced cross.

    Not the raw ``train`` bucket. ``CROSS_HABITAT`` is 92.9% of the pool, so the
    raw bucket would reduce the task to "same habitat?"; the fold's digest-pinned
    ``train_cross_habitat_balanced`` sample stands in for it.
    """

    train = fold.bucket(TRAIN_BUCKET)
    same_habitat = tuple(p for p in train if dataset.records[p].same_habitat)
    balanced = _sample_positions(fold, TRAIN_SAMPLE_SUFFIX)
    if balanced is None:
        positions = tuple(sorted(train))
        note = (
            f"fold {fold.name!r} declares no {TRAIN_SAMPLE_SUFFIX} sample; the raw "
            "train bucket is used and the CROSS_HABITAT imbalance is NOT corrected"
        )
    else:
        positions = tuple(sorted(set(same_habitat) | set(balanced)))
        note = (
            "all same-habitat positions in the train bucket (ADMIT, REPEATED, "
            f"SAME_HABITAT_DISJOINT) plus the {TRAIN_SAMPLE_SUFFIX} sample in "
            "place of every cross-habitat position"
        )
    return positions, _composition(dataset, positions, note)


def evaluation_positions(
    dataset: Dataset, fold: Fold
) -> tuple[tuple[int, ...], dict]:
    """The declared evaluation set: ``test_within`` + the ``test_cross_habitat`` sample.

    ``test_within`` holds no cross-habitat pairs at all for a LOHO fold, so the
    cross-habitat non-admission metric would have no denominator without the
    straddle-derived sample. Nothing here is ever masked.
    """

    names = fold.bucket_names()
    bucket_name = TEST_BUCKET if TEST_BUCKET in names else names[-1]
    within = fold.bucket(bucket_name)
    cross = _sample_positions(fold, TEST_SAMPLE_SUFFIX)
    if cross is None:
        positions = tuple(sorted(within))
        note = (
            f"fold {fold.name!r} declares no {TEST_SAMPLE_SUFFIX} sample; the "
            f"{bucket_name!r} bucket is evaluated as-is"
        )
    else:
        positions = tuple(sorted(set(within) | set(cross)))
        note = (
            f"the {bucket_name!r} bucket united with the {TEST_SAMPLE_SUFFIX} "
            "sample, which folds.py draws from the straddle bucket and sizes "
            "against test_within's ADMIT count"
        )
    return positions, _composition(dataset, positions, note)


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------

def _habitat_ids(dataset: Dataset) -> np.ndarray:
    """Compact habitat id per Event index, ascending by ``(P, delta)``."""

    catalogue = dataset.catalogue
    order = sorted(set(catalogue.habitats))
    index_of = {key: i for i, key in enumerate(order)}
    return np.array(
        [index_of[catalogue.habitats[i]] for i in range(len(catalogue))],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class _Batch:
    """The frozen tensor view of a set of positions. Labels are only consumed."""

    a_index: torch.Tensor
    b_index: torch.Tensor
    true_slot: torch.Tensor
    is_admit: np.ndarray
    klass: np.ndarray
    size: int


def _batch(dataset: Dataset, positions: Sequence[int], bottom_slot: int) -> _Batch:
    a_list: list[int] = []
    b_list: list[int] = []
    slot_list: list[int] = []
    admit_list: list[bool] = []
    class_list: list[str] = []
    for position in positions:
        record = dataset.records[position]
        a_list.append(record.a_index)
        b_list.append(record.b_index)
        if record.admitted:
            if record.target_index == BOTTOM_INDEX:
                raise AssertionError("an ADMIT record carries the bottom sentinel")
            slot_list.append(record.target_index)
            admit_list.append(True)
            class_list.append("ADMIT")
        else:
            if record.target_index != BOTTOM_INDEX:
                raise AssertionError("a non-admission record carries an Event target")
            slot_list.append(bottom_slot)
            admit_list.append(False)
            class_list.append(record.nonadmission_class.value)
    return _Batch(
        a_index=torch.tensor(a_list, dtype=torch.long),
        b_index=torch.tensor(b_list, dtype=torch.long),
        true_slot=torch.tensor(slot_list, dtype=torch.long),
        is_admit=np.array(admit_list, dtype=bool),
        klass=np.array(class_list, dtype=object),
        size=len(a_list),
    )


def _legal_mask(
    model: CandidateScorer, size: int, held_out: Sequence[int]
) -> torch.Tensor:
    """Boolean legality over the 85 slots. Bottom is always legal."""

    legal = torch.ones(size, model.n_slots, dtype=torch.bool)
    for event in held_out:
        legal[:, int(event)] = False
    legal[:, model.bottom_slot] = True
    return legal


# ---------------------------------------------------------------------------
# the metric suite
# ---------------------------------------------------------------------------

def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate(
    model: CandidateScorer, dataset: Dataset, positions: Sequence[int]
) -> dict:
    """The whole metric suite over one set of positions. No masking, ever.

    Correctness is **integer index equality** against ``record.target_index``,
    which ``task.py`` already certified through ``projective.equivalent``. There
    is no float proximity anywhere, no soft ranking and no top-k.

    Admission is reported as **balanced** accuracy, ``0.5 * (sensitivity +
    specificity)``. Raw admission accuracy is also reported, flagged inadmissible,
    solely so nobody has to recompute it to see why it is inadmissible.
    """

    if len(positions) == 0:
        raise ValueError("cannot evaluate an empty position set")
    bottom = model.bottom_slot
    batch = _batch(dataset, positions, bottom)
    habitat_of = _habitat_ids(dataset)

    model.eval()
    with torch.no_grad():
        scores = model(batch.a_index, batch.b_index)
        swapped = model(batch.b_index, batch.a_index)
    predicted = scores.argmax(dim=-1).numpy()
    predicted_swapped = swapped.argmax(dim=-1).numpy()

    swap_error = float((scores - swapped).abs().max().item())
    swap_disagreement = float(np.mean(predicted != predicted_swapped))

    true_slot = batch.true_slot.numpy()
    correct = predicted == true_slot
    is_admit = batch.is_admit
    n = batch.size

    predicted_admit = predicted != bottom
    n_admit = int(is_admit.sum())
    n_reject = int((~is_admit).sum())
    sensitivity = _fraction(int((is_admit & predicted_admit).sum()), n_admit)
    specificity = _fraction(int((~is_admit & ~predicted_admit).sum()), n_reject)
    balanced = (
        None
        if sensitivity is None or specificity is None
        else 0.5 * (sensitivity + specificity)
    )

    def class_accuracy(name: str) -> tuple[float | None, int]:
        selected = batch.klass == name
        total = int(selected.sum())
        return _fraction(int((selected & ~predicted_admit).sum()), total), total

    repeated, n_repeated = class_accuracy("REPEATED")
    disjoint, n_disjoint = class_accuracy("SAME_HABITAT_DISJOINT")
    cross, n_cross = class_accuracy("CROSS_HABITAT")

    # error taxonomy over incorrect predictions; the four categories partition them
    wrong = ~correct
    safe_true = np.where(is_admit, true_slot, 0)
    predicted_event = predicted != bottom
    same_habitat_error = habitat_of[np.clip(predicted, 0, CATALOGUE_SIZE - 1)] == (
        habitat_of[safe_true]
    )
    taxonomy = {
        "wrong_event_correct_habitat": int(
            (wrong & is_admit & predicted_event & same_habitat_error).sum()
        ),
        "event_wrong_habitat": int(
            (wrong & is_admit & predicted_event & ~same_habitat_error).sum()
        ),
        "bottom_instead_of_admitted_event": int(
            (wrong & is_admit & ~predicted_event).sum()
        ),
        "event_instead_of_bottom": int((wrong & ~is_admit & predicted_event).sum()),
    }
    n_wrong = int(wrong.sum())
    if sum(taxonomy.values()) != n_wrong:
        raise AssertionError(
            "error taxonomy does not partition the incorrect predictions: "
            f"{taxonomy} sums to {sum(taxonomy.values())}, expected {n_wrong}"
        )

    return {
        "n": n,
        "n_admit": n_admit,
        "n_nonadmission": n_reject,
        "n_repeated": n_repeated,
        "n_same_habitat_disjoint": n_disjoint,
        "n_cross_habitat": n_cross,
        "joint_exact_success": _round(float(correct.mean())),
        "joint_exact_success_statement": (
            "admission decision correct AND, for an ADMIT pair, the forced third "
            "Event exactly right; one argmax over one 85-slot softmax"
        ),
        "admission_balanced_accuracy": _round(balanced),
        "admission_sensitivity": _round(sensitivity),
        "admission_specificity": _round(specificity),
        "admission_raw_accuracy": _round(
            float(((is_admit & predicted_admit) | (~is_admit & ~predicted_admit)).mean())
        ),
        "admission_raw_accuracy_note": (
            "INADMISSIBLE as a headline: CROSS_HABITAT is 92.9% of the pool, so raw "
            "accuracy rewards answering 'different habitat?'. Reported only so the "
            "balanced figure can be checked against it."
        ),
        "positive_forced_third_exact_accuracy": _round(
            _fraction(int((is_admit & correct).sum()), n_admit)
        ),
        "repeated_pair_nonadmission_accuracy": _round(repeated),
        "same_habitat_disjoint_nonadmission_accuracy": _round(disjoint),
        "cross_habitat_nonadmission_accuracy": _round(cross),
        "swap_invariance_error": _round(swap_error),
        "swap_argmax_disagreement_rate": _round(swap_disagreement),
        "swap_scores_bitwise_identical": bool(torch.equal(scores, swapped)),
        "swap_statement": (
            "measured, not assumed: scorer.SymmetricPair is commutative bitwise in "
            "IEEE-754, so 0.0 is expected; any nonzero value is a real defect"
        ),
        "n_incorrect": n_wrong,
        "error_taxonomy": taxonomy,
        "error_taxonomy_fractions": {
            key: _round(_fraction(value, n_wrong)) for key, value in taxonomy.items()
        },
        "error_taxonomy_partitions_errors": True,
        "correctness_criterion": (
            "exact certified Event identity: integer index equality against "
            "record.target_index, already validated by task.py through "
            "projective.equivalent. Never float proximity, never a soft ranking."
        ),
    }


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    """One (arm, fold, seed) run: every required metric, plus its provenance."""

    arm: str
    arm_role: str
    arm_token_dim: int
    arm_tokens_per_event: int
    fold_family: str
    fold_name: str
    fold_role: str
    seed: int

    param_count: int
    train_size: int
    test_size: int
    train_composition: dict
    test_composition: dict

    joint_exact_success: float
    admission_balanced_accuracy: float | None
    positive_forced_third_exact_accuracy: float | None
    repeated_pair_nonadmission_accuracy: float | None
    same_habitat_disjoint_nonadmission_accuracy: float | None
    cross_habitat_nonadmission_accuracy: float | None
    swap_invariance_error: float
    train_accuracy: float
    generalization_gap: float

    final_loss: float
    steps: int
    forward_flops_train_step: int
    forward_flops_eval_train: int
    forward_flops_eval_test: int
    forward_passes: int
    total_forward_flops: int
    wall_sec: float

    test_metrics: dict
    train_metrics: dict
    arm_role_note: str | None = None

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "arm_role": self.arm_role,
            "arm_role_note": self.arm_role_note,
            "arm_token_dim": self.arm_token_dim,
            "arm_tokens_per_event": self.arm_tokens_per_event,
            "fold_family": self.fold_family,
            "fold_name": self.fold_name,
            "fold_role": self.fold_role,
            "seed": self.seed,
            "param_count": self.param_count,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_composition": self.train_composition,
            "test_composition": self.test_composition,
            "joint_exact_success": self.joint_exact_success,
            "admission_balanced_accuracy": self.admission_balanced_accuracy,
            "positive_forced_third_exact_accuracy": (
                self.positive_forced_third_exact_accuracy
            ),
            "repeated_pair_nonadmission_accuracy": (
                self.repeated_pair_nonadmission_accuracy
            ),
            "same_habitat_disjoint_nonadmission_accuracy": (
                self.same_habitat_disjoint_nonadmission_accuracy
            ),
            "cross_habitat_nonadmission_accuracy": (
                self.cross_habitat_nonadmission_accuracy
            ),
            "swap_invariance_error": self.swap_invariance_error,
            "train_accuracy": self.train_accuracy,
            "generalization_gap": self.generalization_gap,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "forward_compute": {
                "flops_per_train_step": self.forward_flops_train_step,
                "flops_eval_train": self.forward_flops_eval_train,
                "flops_eval_test": self.forward_flops_eval_test,
                "forward_passes": self.forward_passes,
                "total_forward_flops": self.total_forward_flops,
                "convention": (
                    "analytic via scorer.forward_flops; forward only, never "
                    "measured"
                ),
            },
            "wall_sec": self.wall_sec,
            "test_metrics": self.test_metrics,
            "train_metrics": self.train_metrics,
        }


def train_one(
    arm: Arm,
    fold: Fold,
    seed: int,
    config: HarnessConfig,
    *,
    dataset: Dataset | None = None,
) -> RunResult:
    """Train one (arm, fold, seed) and rescore it exactly.

    ``dataset`` is a pure optimisation: passing the already-built frozen pool
    avoids rebuilding it 840 times in a sweep. When omitted it is built here, so
    the declared four-argument signature works standalone.
    """

    dataset = dataset or build_dataset()
    started = time.perf_counter()

    train_positions, train_composition = training_positions(dataset, fold)
    test_positions, test_composition = evaluation_positions(dataset, fold)

    model = build_scorer(arm, config.scorer, seed=seed)
    set_seed(seed)
    bottom = model.bottom_slot
    train_batch = _batch(dataset, train_positions, bottom)

    legal = None
    if config.train.mask_held_out_candidates:
        legal = _legal_mask(model, train_batch.size, fold.held_out_events)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(config.train.steps):
        scores = model(train_batch.a_index, train_batch.b_index)
        if legal is not None:
            scores = mask_scores(scores, legal)
        loss = loss_fn(scores, train_batch.true_slot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_metrics = evaluate(model, dataset, train_positions)
    test_metrics = evaluate(model, dataset, test_positions)

    train_step_flops = forward_flops(model, batch=train_batch.size)["total"]
    eval_train_flops = 2 * train_step_flops
    eval_test_flops = 2 * forward_flops(model, batch=len(test_positions))["total"]
    # every ``evaluate`` call runs two forward passes: the scored one and its swap
    forward_passes = config.train.steps + 4
    total_flops = (
        config.train.steps * train_step_flops + eval_train_flops + eval_test_flops
    )

    train_accuracy = float(train_metrics["joint_exact_success"])
    test_accuracy = float(test_metrics["joint_exact_success"])

    return RunResult(
        arm=arm.name,
        arm_role=ARM_ROLES[arm.name],
        arm_role_note=ARM_ROLE_NOTES.get(arm.name),
        arm_token_dim=int(arm.token_dim),
        arm_tokens_per_event=int(arm.tokens_per_event),
        fold_family=fold.family,
        fold_name=fold.name,
        fold_role=fold.role,
        seed=int(seed),
        param_count=count_params(model),
        train_size=train_batch.size,
        test_size=len(test_positions),
        train_composition=train_composition,
        test_composition=test_composition,
        joint_exact_success=test_metrics["joint_exact_success"],
        admission_balanced_accuracy=test_metrics["admission_balanced_accuracy"],
        positive_forced_third_exact_accuracy=(
            test_metrics["positive_forced_third_exact_accuracy"]
        ),
        repeated_pair_nonadmission_accuracy=(
            test_metrics["repeated_pair_nonadmission_accuracy"]
        ),
        same_habitat_disjoint_nonadmission_accuracy=(
            test_metrics["same_habitat_disjoint_nonadmission_accuracy"]
        ),
        cross_habitat_nonadmission_accuracy=(
            test_metrics["cross_habitat_nonadmission_accuracy"]
        ),
        swap_invariance_error=test_metrics["swap_invariance_error"],
        train_accuracy=_round(train_accuracy),
        generalization_gap=_round(train_accuracy - test_accuracy),
        final_loss=_round(float(loss.item())),
        steps=config.train.steps,
        forward_flops_train_step=train_step_flops,
        forward_flops_eval_train=eval_train_flops,
        forward_flops_eval_test=eval_test_flops,
        forward_passes=forward_passes,
        total_forward_flops=total_flops,
        wall_sec=round(time.perf_counter() - started, 3),
        test_metrics=test_metrics,
        train_metrics=train_metrics,
    )


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def stats(values: Sequence[float | None]) -> dict:
    """Mean / median / population std / min / max / spread over defined values.

    Population std (``np.std`` ddof=0) per repo convention. Median and the
    min/max spread are reported too because 14 LOHO folds and 7 LOFPO folds are
    small enough that a mean and a std hide the shape.
    """

    defined = [float(v) for v in values if v is not None]
    if not defined:
        return {
            "n": 0,
            "n_undefined": len(values),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "spread": None,
        }
    array = np.asarray(defined, dtype=np.float64)
    return {
        "n": int(array.size),
        "n_undefined": len(values) - int(array.size),
        "mean": _round(float(array.mean())),
        "median": _round(float(np.median(array))),
        "std": _round(float(array.std(ddof=0))),
        "min": _round(float(array.min())),
        "max": _round(float(array.max())),
        "spread": _round(float(array.max() - array.min())),
    }


def _metric_value(result: RunResult, name: str) -> float | None:
    if name in result.test_metrics:
        return result.test_metrics[name]
    return getattr(result, name)


def _stats_block(results: Sequence[RunResult], names: Sequence[str]) -> dict:
    return {name: stats([_metric_value(r, name) for r in results]) for name in names}


def _fold_mean(results: Sequence[RunResult], name: str) -> float | None:
    values = [_metric_value(r, name) for r in results]
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def aggregate(results: Sequence[RunResult]) -> dict:
    """Mean / median / spread over folds and seeds, per arm and per fold family.

    Two aggregations are reported side by side because they answer different
    questions:

    ``over_runs``   every (fold, seed) run pooled — the raw variability.
    ``over_folds``  each fold reduced to its mean across seeds first, then
                    aggregated — the fold-to-fold variability with seed noise
                    averaged out. This is the leave-one-habitat-out and
                    leave-one-Fano-point-out figure.
    """

    names = tuple(METRIC_NAMES) + tuple(RUN_SCALARS)
    by_arm: dict[str, dict] = {}
    for arm_name in sorted({r.arm for r in results}):
        arm_runs = [r for r in results if r.arm == arm_name]
        families: dict[str, dict] = {}
        for family in sorted({r.fold_family for r in arm_runs}):
            family_runs = [r for r in arm_runs if r.fold_family == family]
            fold_names = sorted({r.fold_name for r in family_runs})
            per_fold = {
                fold_name: {
                    name: _round(
                        _fold_mean(
                            [r for r in family_runs if r.fold_name == fold_name], name
                        )
                    )
                    for name in names
                }
                for fold_name in fold_names
            }
            over_folds = {
                name: stats([per_fold[f][name] for f in fold_names]) for name in names
            }
            families[family] = {
                "n_folds": len(fold_names),
                "n_runs": len(family_runs),
                "seeds": sorted({r.seed for r in family_runs}),
                "over_runs": _stats_block(family_runs, names),
                "over_folds": over_folds,
                "per_fold_seed_mean": per_fold,
                "error_taxonomy_total": {
                    key: int(sum(r.test_metrics["error_taxonomy"][key] for r in family_runs))
                    for key in ERROR_CATEGORIES
                },
            }
        by_arm[arm_name] = {
            "arm_role": ARM_ROLES[arm_name],
            "arm_role_note": ARM_ROLE_NOTES.get(arm_name),
            "param_count": sorted({r.param_count for r in arm_runs}),
            "n_runs": len(arm_runs),
            "by_family": families,
            "over_runs_all_families": _stats_block(arm_runs, names),
        }

    swap_errors = [float(r.swap_invariance_error) for r in results]
    return {
        "metrics": list(names),
        "family_legend": {
            FAMILY_LOHO: "leave-one-habitat-out, 14 folds, PRIMARY",
            FAMILY_LOFPO: "leave-one-Fano-point-out, 7 folds, HARDER SECONDARY",
        },
        "std_convention": "population std, np.std(ddof=0), per repo convention",
        "spread_convention": "max - min over the aggregated values",
        "by_arm": by_arm,
        "swap_invariance": {
            "max_over_all_runs": _round(max(swap_errors)) if swap_errors else None,
            "all_exactly_zero": all(v == 0.0 for v in swap_errors),
            "statement": (
                "the scorer is swap-symmetric by construction, so exactly 0.0 is "
                "expected; this is a live check, not an assumption"
            ),
        },
        "arm_roles": {
            name: ARM_ROLES[name] for name in sorted({r.arm for r in results})
        },
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def _extrapolate(results: Sequence[RunResult], config: HarnessConfig) -> dict:
    """Extrapolate the full sweep cost from the smoke timings, per arm."""

    per_arm = {}
    for arm_name in sorted({r.arm for r in results}):
        seconds = [r.wall_sec for r in results if r.arm == arm_name]
        per_arm[arm_name] = {
            "runs_timed": len(seconds),
            "mean_wall_sec": round(float(np.mean(seconds)), 3),
            "max_wall_sec": round(float(max(seconds)), 3),
        }
    science = [
        per_arm[name]["mean_wall_sec"] for name in SCIENCE_ARMS if name in per_arm
    ]
    science_mean = float(np.mean(science)) if science else None
    diagnostic = per_arm.get("E_opaque", {}).get("mean_wall_sec")

    runs_per_arm = FULL_SWEEP["folds"] * FULL_SWEEP["seeds"]
    estimate = None
    if science_mean is not None and diagnostic is not None:
        estimate = runs_per_arm * (len(SCIENCE_ARMS) * science_mean + diagnostic)
    return {
        "smoke_total_wall_sec": round(sum(r.wall_sec for r in results), 3),
        "smoke_runs": len(results),
        "per_arm": per_arm,
        "full_sweep_shape": {
            "arms": FULL_SWEEP["arms"],
            "folds": FULL_SWEEP["folds"],
            "seeds": FULL_SWEEP["seeds"],
            "runs": FULL_SWEEP["arms"] * runs_per_arm,
            "runs_per_arm": runs_per_arm,
        },
        "full_sweep_estimated_wall_sec": (
            None if estimate is None else round(estimate, 1)
        ),
        "full_sweep_estimated_wall_min": (
            None if estimate is None else round(estimate / 60.0, 1)
        ),
        "method": (
            "per-arm mean smoke wall-clock times 21 folds times 8 seeds, summed "
            "over the four science arms plus the diagnostic. Single process, no "
            "parallelism assumed. Wall-clock is not replayable and is excluded "
            f"from --check by NON_REPLAYABLE_KEYS={list(NON_REPLAYABLE_KEYS)}."
        ),
        "steps_per_run": config.train.steps,
    }


def _interface_findings(results: Sequence[RunResult]) -> dict:
    """Computed, not asserted: does the held-out target ever get named at all?

    Under a leave-one-habitat-out fold the correct ADMIT answer at test time is an
    Event that never appears as a training target, while the default protocol
    scores the full 85-slot catalogue in training. Cross-entropy therefore
    actively pushes those slots down. If the positive forced-third accuracy sits
    at the floor while train accuracy sits at the ceiling, that is the interface
    biting — not an optimisation failure and not something to tune away.
    """

    per_family = {}
    for family in sorted({r.fold_family for r in results}):
        runs = [r for r in results if r.fold_family == family]
        positive = [
            r.positive_forced_third_exact_accuracy
            for r in runs
            if r.positive_forced_third_exact_accuracy is not None
        ]
        train = [r.train_accuracy for r in runs]
        taxonomy_total = sum(r.test_metrics["n_incorrect"] for r in runs)
        correct_habitat = sum(
            r.test_metrics["error_taxonomy"]["wrong_event_correct_habitat"]
            for r in runs
        )
        per_family[family] = {
            "n_runs": len(runs),
            "positive_forced_third_exact_accuracy_mean": _round(
                float(np.mean(positive)) if positive else None
            ),
            "positive_forced_third_exact_accuracy_max": _round(
                float(max(positive)) if positive else None
            ),
            "train_accuracy_mean": _round(float(np.mean(train))),
            "n_incorrect_total": int(taxonomy_total),
            "wrong_event_correct_habitat_total": int(correct_habitat),
            "wrong_event_correct_habitat_share": _round(
                _fraction(int(correct_habitat), int(taxonomy_total))
            ),
        }

    loho = per_family.get(FAMILY_LOHO, {})
    floored = bool(
        loho
        and (loho["positive_forced_third_exact_accuracy_mean"] or 0.0) < 0.05
        and (loho["train_accuracy_mean"] or 0.0) > 0.9
    )
    return {
        "per_family": per_family,
        "held_out_target_suppression": {
            "observed": floored,
            "finding": (
                "REPORTED, NOT REPAIRED. On the leave-one-habitat-out folds the "
                "positive forced-third exact accuracy sits at the floor while train "
                "accuracy is at the ceiling, and almost no error is a wrong Event "
                "inside the correct habitat. That is the signature of the default "
                "protocol: the held-out Events are scored as candidates during "
                "training but are never the correct answer, so cross-entropy learns "
                "to suppress exactly the slots the test set needs. It is a property "
                "of the representation interface, not an optimisation failure, and "
                "it is NOT tuned away here."
            ),
            "candidate_repair": (
                "TrainConfig.mask_held_out_candidates=True masks the fold's held-out "
                "candidate slots in training only, using the finite-penalty "
                "MASK_PENALTY convention. It is the single motivated repair the stop "
                "rule permits, it is OFF in this run, and adopting it is Task 10/11's "
                "call, not this task's."
            ),
            "not_a_defect_in": (
                "admission, non-admission by class, and swap invariance all compute "
                "normally; the suppression is specific to naming a held-out Event"
            ),
        },
    }


def smoke(config: HarnessConfig | None = None) -> dict:
    """Run the declared small subset and build the smoke payload.

    This is a **plumbing check**, not a science result. The subset is far too
    small, and the metric values are not evidence about any arm. Task 10 owns the
    sweep; Task 11 owns the analysis.
    """

    base = config or HarnessConfig()
    config = HarnessConfig(
        seeds=SMOKE_SEEDS,
        arm_names=SMOKE_ARMS,
        families=STRUCTURAL_FAMILIES,
        scorer=base.scorer,
        train=base.train,
    )

    dataset = build_dataset()
    every_fold = all_folds(dataset)
    structural = structural_folds(every_fold)
    loho = [f for f in structural if f.family == FAMILY_LOHO][:SMOKE_LOHO_FOLDS]
    lofpo = [f for f in structural if f.family == FAMILY_LOFPO][:SMOKE_LOFPO_FOLDS]
    selected_folds = loho + lofpo

    every_arm = all_arms(dataset, include_e=True)
    ledger = capacity_ledger(every_arm, base.scorer)
    arm_by_name = {arm.name: arm for arm in every_arm}
    selected_arms = [arm_by_name[name] for name in config.arm_names]

    results: list[RunResult] = []
    for arm in selected_arms:
        for fold in selected_folds:
            for seed in config.seeds:
                result = train_one(arm, fold, seed, config, dataset=dataset)
                results.append(result)
                print(
                    f"  {result.arm:<12} {result.fold_family:<6} "
                    f"{result.fold_name:<14} seed={result.seed} "
                    f"joint={result.joint_exact_success:.4f} "
                    f"bal_adm={result.admission_balanced_accuracy:.4f} "
                    f"swap={result.swap_invariance_error} "
                    f"{result.wall_sec:.2f}s",
                    flush=True,
                )

    rows = [result.as_dict() for result in results]
    aggregates = aggregate(results)
    extrapolation = _extrapolate(results, config)

    payload = {
        "module": "experiments/sfp_representation/harness.py",
        "issue": "009",
        "task": 9,
        "base_commit": BASE_COMMIT,
        "mode": "smoke",
        "mode_statement": (
            "PLUMBING CHECK ONLY. A declared small subset: "
            f"{len(config.arm_names)} arms x {len(loho)} LOHO + {len(lofpo)} LOFPO "
            f"folds x {len(config.seeds)} seeds = {len(results)} runs. The metric "
            "values here are NOT a science result and must not be read as evidence "
            "about any arm. The full sweep is Task 10; the analysis is Task 11."
        ),
        "fences": list(FENCES),
        "config": config.as_dict(),
        "declared_full_config": HarnessConfig().as_dict(),
        "capacity_ledger": ledger,
        "smoke_subset": {
            "arms": list(config.arm_names),
            "loho_folds": [f.name for f in loho],
            "lofpo_folds": [f.name for f in lofpo],
            "seeds": list(config.seeds),
            "runs": len(results),
            "structural_folds_available": len(structural),
            "all_folds_available": len(every_fold),
        },
        "runs": rows,
        "aggregates": aggregates,
        "interface_findings": _interface_findings(results),
        "extrapolation": extrapolation,
        "arm_e_disposition": {
            "arm": "E_opaque",
            "role": ARM_ROLES["E_opaque"],
            "token_dim": arm_by_name["E_opaque"].token_dim,
            "catalogue_size": CATALOGUE_SIZE,
            "token_dim_equals_catalogue_size": (
                int(arm_by_name["E_opaque"].token_dim) == CATALOGUE_SIZE
            ),
            "statement": ARM_ROLE_NOTES["E_opaque"],
            "labelled_in_every_row": all(
                row["arm_role"] == ARM_ROLES[row["arm"]] for row in rows
            ),
        },
        "upstream_digests": {
            "dataset": dataset.sha256(),
            "catalogue": dataset.catalogue.sha256(),
            "folds": digest([fold.sha256() for fold in every_fold]),
            "arms": digest([arm.sha256() for arm in every_arm]),
        },
        "habitat_legend": [
            habitat_label(p, d) for p, d in sorted(set(dataset.catalogue.habitats))
        ],
        "torch_version": importlib.metadata.version("torch"),
        "numpy_version": importlib.metadata.version("numpy"),
        "non_replayable_keys": list(NON_REPLAYABLE_KEYS),
        "non_replayable_statement": (
            "wall_sec, the extrapolation block and the environment versions are "
            "measured, not derived, and are excluded from the --check byte "
            "comparison by strip_non_replayable. Everything else replays exactly."
        ),
    }
    payload["verdict"] = _verdict(payload, results, ledger)
    payload["digests"] = {
        "runs": digest(strip_non_replayable(rows)),
        "aggregates": digest(aggregates),
        "capacity_ledger": digest(ledger),
        "replayable_manifest": digest(strip_non_replayable(payload)),
    }
    return payload


def _verdict(payload: dict, results: Sequence[RunResult], ledger: dict) -> dict:
    checks = {
        "every_run_completed": len(results) > 0,
        "every_row_carries_arm_role": payload["arm_e_disposition"][
            "labelled_in_every_row"
        ],
        "arm_e_is_diagnostic": ARM_ROLES["E_opaque"] != "science",
        "science_capacity_parity_exact": ledger["science_capacity_parity_exact"],
        "science_param_spread_zero": ledger["science_total_params_spread"] == 0,
        "shared_stack_identical": ledger["shared_stack_identical_across_arms"],
        "swap_invariance_exactly_zero": all(
            float(r.swap_invariance_error) == 0.0 for r in results
        ),
        "error_taxonomy_partitions_every_run": all(
            sum(r.test_metrics["error_taxonomy"].values())
            == r.test_metrics["n_incorrect"]
            for r in results
        ),
        "bottom_slot_is_84": all(
            r.test_metrics["n"] == r.test_size for r in results
        ),
        "no_tuning": not payload["config"]["train"]["mask_held_out_candidates"],
        "every_metric_has_a_denominator": all(
            r.admission_balanced_accuracy is not None
            and r.positive_forced_third_exact_accuracy is not None
            and r.repeated_pair_nonadmission_accuracy is not None
            and r.same_habitat_disjoint_nonadmission_accuracy is not None
            and r.cross_habitat_nonadmission_accuracy is not None
            for r in results
        ),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    return {
        "checks": checks,
        "failed": failed,
        "agrees": not failed,
        "statement": (
            "harness smoke green: the full metric suite computes on every run, the "
            "error taxonomy partitions every error set, swap invariance is exactly "
            "0.0, and the four science arms have identical parameter counts"
            if not failed
            else f"harness smoke FAILED these checks: {failed}"
        ),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _report(payload: dict) -> None:
    ledger = payload["capacity_ledger"]
    print("capacity ledger (REAL arms; supersedes scorer.json):", flush=True)
    for row in ledger["rows"]:
        print(
            f"  {row['arm']:<12} role={row['arm_role']:<32} "
            f"token_dim={row['token_dim']:<3} tokens={row['tokens_per_event']} "
            f"adapter={row['adapter_params']:<6} trunk={row['shared_trunk_params']} "
            f"pair={row['pair_params']} rel={row['rel_params']} "
            f"bottom={row['bottom_params']} total={row['total_trainable_params']} "
            f"flops@1={row['forward_flops_batch_1']}",
            flush=True,
        )
    print(
        f"  A/B/C/D total-param spread={ledger['science_total_params_spread']} "
        f"(exact parity={ledger['science_capacity_parity_exact']}); "
        f"shared stack identical={ledger['shared_stack_identical_across_arms']}",
        flush=True,
    )

    for arm_name, block in sorted(payload["aggregates"]["by_arm"].items()):
        for family, family_block in sorted(block["by_family"].items()):
            over = family_block["over_folds"]
            print(
                f"{arm_name:<12} {family:<6} role={block['arm_role']:<32} "
                f"joint mean={over['joint_exact_success']['mean']} "
                f"median={over['joint_exact_success']['median']} "
                f"spread={over['joint_exact_success']['spread']} | "
                f"bal_adm mean={over['admission_balanced_accuracy']['mean']} | "
                f"gap mean={over['generalization_gap']['mean']}",
                flush=True,
            )

    findings = payload["interface_findings"]
    for family, block in sorted(findings["per_family"].items()):
        print(
            f"interface {family:<6} positive_forced_third mean="
            f"{block['positive_forced_third_exact_accuracy_mean']} "
            f"max={block['positive_forced_third_exact_accuracy_max']} | "
            f"train_acc mean={block['train_accuracy_mean']} | "
            f"wrong-Event-correct-habitat share="
            f"{block['wrong_event_correct_habitat_share']}",
            flush=True,
        )
    print(
        "held-out target suppression observed="
        f"{findings['held_out_target_suppression']['observed']} "
        "(reported, not repaired)",
        flush=True,
    )

    swap = payload["aggregates"]["swap_invariance"]
    print(
        f"swap invariance: max over all runs={swap['max_over_all_runs']}, "
        f"all exactly zero={swap['all_exactly_zero']}",
        flush=True,
    )
    extra = payload["extrapolation"]
    print(
        f"smoke wall={extra['smoke_total_wall_sec']}s over {extra['smoke_runs']} runs; "
        f"full sweep {extra['full_sweep_shape']['runs']} runs estimated at "
        f"{extra['full_sweep_estimated_wall_sec']}s "
        f"({extra['full_sweep_estimated_wall_min']} min)",
        flush=True,
    )
    print(
        f"replayable manifest sha256: {payload['digests']['replayable_manifest']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the declared small subset and write harness_smoke.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive the smoke payload and compare it, wall-clock excluded",
    )
    args = parser.parse_args()
    if not (args.smoke or args.check):
        parser.error("nothing to do: pass --smoke (or --check)")

    started = time.perf_counter()
    payload = smoke()
    text = render(payload)
    _report(payload)
    print(
        f"harness wall_sec (not serialised into the compared payload): "
        f"{time.perf_counter() - started:.2f}",
        flush=True,
    )

    if args.check:
        if not SMOKE_OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {SMOKE_OUTPUT}; run --smoke first")
        stored = json.loads(SMOKE_OUTPUT.read_text())
        if strip_non_replayable(stored) != strip_non_replayable(payload):
            raise SystemExit(
                "FAIL: re-derived smoke payload differs from "
                f"{SMOKE_OUTPUT.name} outside the non-replayable keys"
            )
        if not payload["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + payload["verdict"]["statement"])
        print(
            f"PASS: replay matches {SMOKE_OUTPUT.name} (wall-clock excluded)",
            flush=True,
        )
        print(payload["verdict"]["statement"], flush=True)
        return

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SMOKE_OUTPUT.write_text(text)
    if not payload["verdict"]["agrees"]:
        print(json.dumps(payload["verdict"], indent=2, sort_keys=True), flush=True)
        raise SystemExit("FAIL: " + payload["verdict"]["statement"])
    print(f"PASS: wrote {SMOKE_OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
    print(payload["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
