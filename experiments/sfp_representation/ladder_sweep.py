"""Training and evaluation for the 009.05 ladder: Rung 1, Rung 2, and Rung 3.

One protocol, declared once and applied identically to every arm, fold and seed.
It is deliberately the *same* protocol `009.02` used -- full-batch Adam, 800
steps, `lr = 0.01`, no early stopping, no tuning, no per-arm adjustment -- because
the causal question `009.03` raised is about the output interface and nothing
else. The only permitted change is the one the type of the output object forces.

Rung 3 is not a separate training run. `ladder_resolver` proved that resolution
is total and injective on the saturated SFP alphabet, so the resolver is applied
to every Rung-2 prediction inside the same evaluation pass, and both numbers are
reported side by side:

    exact_structured_address_accuracy    before resolution
    positive_forced_third_exact_accuracy after resolution

The second name is taken verbatim from `009.02` so the comparison to the failed
84-way catalogue interface is a comparison of the same quantity, not of two
similar-sounding ones.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from arms import all_arms
from folds import all_folds, structural_folds
from ladder_baselines import rung1_marginal
from ladder_heads import (
    LadderConfig,
    build_port_scorer,
    build_structured_scorer,
    capacity_ledger,
    port_loss,
    structured_loss,
)
from ladder_resolver import build_resolvers
from ladder_task import (
    ARM_OUTPUT_CONVENTION,
    PP_CLASSES,
    SCIENCE_ARMS,
    code_tables,
    output_table,
    rung1_class,
    rung1_positions,
    rung1_target,
    rung2_positions,
    rung2_target,
)
from scorer import ScorerConfig, count_params, set_seed
from task import Dataset, build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_sweep.json"
SMOKE_OUTPUT = ARTIFACTS / "ladder_sweep_smoke.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

#: Removed before any ``--check`` comparison. Wall-clock is not replayable.
NON_REPLAYABLE_KEYS = (
    "wall_sec",
    "sweep_wall_sec",
    "started_at",
    "host",
    "torch_version",
    "numpy_version",
)

METRIC_DECIMALS = 12

RUNG1_METRICS = (
    "pp3_exact_accuracy",
    "train_accuracy",
    "generalization_gap",
    "swap_invariance_error",
    "swap_argmax_disagreement_rate",
)

RUNG2_METRICS = (
    "positive_forced_third_exact_accuracy",
    "exact_structured_address_accuracy",
    "joint_exact_success",
    "admission_balanced_accuracy",
    "admission_sensitivity",
    "admission_specificity",
    "field_S_accuracy",
    "field_FFF_accuracy",
    "field_PP_accuracy",
    "field_pp_accuracy",
    "nontrivial_fields_accuracy",
    "repeated_pair_nonadmission_accuracy",
    "same_habitat_disjoint_nonadmission_accuracy",
    "cross_habitat_nonadmission_accuracy",
    "train_accuracy",
    "generalization_gap",
    "swap_invariance_error",
)

FENCES = (
    "One protocol, fixed before any run; no tuning and no per-arm adjustment.",
    "Correctness is exact certified Event identity, never float proximity.",
    "Evaluation is never masked.",
    (
        "positive_forced_third_exact_accuracy is the SAME quantity 009.02 reported, "
        "so the two runs are directly comparable."
    ),
    (
        "The resolver is applied after a structured address exists and may not "
        "repair it."
    ),
    (
        "Copied fields S and FFF are reported separately and excluded from "
        "nontrivial_fields_accuracy, so they cannot inflate an aggregate."
    ),
)

TRIVIAL_FIELDS = ("S", "FFF")
NONTRIVIAL_FIELDS = ("PP", "pp")

COPIED_FIELD_NOTE = (
    "The certified relation copies S and FFF from the input pair to the "
    "consequence on all 336 admitted pairs, for every arm. Those two heads are "
    "therefore near-trivial by construction and are reported separately. "
    "nontrivial_fields_accuracy covers only PP and pp -- the two fields the "
    "relation actually computes -- which is what 009.05 section 5.5 means by "
    "refusing to let copied fields conceal failure of pp."
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), METRIC_DECIMALS)


def strip_non_replayable(node: object) -> object:
    """Recursively drop every key in :data:`NON_REPLAYABLE_KEYS`."""

    if isinstance(node, dict):
        return {
            key: strip_non_replayable(value)
            for key, value in node.items()
            if key not in NON_REPLAYABLE_KEYS
        }
    if isinstance(node, list):
        return [strip_non_replayable(item) for item in node]
    return node


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """The one training protocol, identical to ``harness.TrainConfig``'s values."""

    steps: int = 800
    lr: float = 0.01

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if not self.lr > 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "optimizer": "Adam",
            "batching": "full batch",
            "evaluation_masking": "none, ever",
            "tuning": (
                "none. Identical to the 009.02 protocol so that the output object "
                "is the only difference."
            ),
            "early_stopping": "none",
            "repairs_used": (
                "none. 009.05 section 9 permits at most one predeclared repair per "
                "rung; no rung needed one."
            ),
        }


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...] = SEEDS
    arm_names: tuple[str, ...] = SCIENCE_ARMS
    families: tuple[str, ...] = ("LOHO", "LOFPO")
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"duplicate seeds: {self.seeds}")
        unknown = [name for name in self.arm_names if name not in SCIENCE_ARMS]
        if unknown:
            raise ValueError(f"unknown or non-science arm names: {unknown}")

    def as_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "n_seeds": len(self.seeds),
            "arm_names": list(self.arm_names),
            "arm_output_convention": {
                name: ARM_OUTPUT_CONVENTION[name] for name in self.arm_names
            },
            "families": list(self.families),
            "scorer": self.scorer.as_dict(),
            "ladder": self.ladder.as_dict(),
            "train": self.train.as_dict(),
        }


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung1Batch:
    a_index: torch.Tensor
    b_index: torch.Tensor
    target: torch.Tensor
    size: int


def rung1_batch(dataset: Dataset, table, positions) -> Rung1Batch:
    a_list, b_list, target_list = [], [], []
    for position in positions:
        record = dataset.records[position]
        a_list.append(record.a_index)
        b_list.append(record.b_index)
        target_list.append(rung1_class(rung1_target(record, table)))
    return Rung1Batch(
        a_index=torch.tensor(a_list, dtype=torch.long),
        b_index=torch.tensor(b_list, dtype=torch.long),
        target=torch.tensor(target_list, dtype=torch.long),
        size=len(a_list),
    )


@dataclass(frozen=True)
class Rung2Batch:
    a_index: torch.Tensor
    b_index: torch.Tensor
    is_admit: torch.Tensor
    field_targets: dict[str, torch.Tensor]
    pp_valid: torch.Tensor
    event_index: np.ndarray
    klass: np.ndarray
    input_s: np.ndarray
    input_fff: np.ndarray
    size: int


def rung2_batch(dataset: Dataset, table, positions) -> Rung2Batch:
    a_list, b_list, admit_list = [], [], []
    s_list, fff_list, pp_list, pp_class_list = [], [], [], []
    valid_rows, event_list, class_list = [], [], []
    input_s, input_fff = [], []
    for position in positions:
        record = dataset.records[position]
        target = rung2_target(record, table)
        a_list.append(record.a_index)
        b_list.append(record.b_index)
        admit_list.append(not target.is_bottom)
        event_list.append(target.event_index)
        class_list.append(
            "ADMIT" if record.admitted else record.nonadmission_class.value
        )
        # The input pair's own identity fields, for the declared Rung-2 repair.
        # Read from the arm's code table, never from the certified target.
        input_s.append(table[record.a_index].s)
        input_fff.append(table[record.a_index].fff)
        if target.is_bottom:
            # Placeholders. The loss masks them out and the metrics never read
            # them: a non-admitted pair has no certified address, and inventing
            # one would conflate BOTTOM with an algebraic zero.
            s_list.append(0)
            fff_list.append(0)
            pp_list.append(0)
            pp_class_list.append(0)
            valid_rows.append([False, False, False, False])
        else:
            s_list.append(target.s)
            fff_list.append(target.fff - 1)
            pp_list.append(target.pp_set[0])
            pp_class_list.append(PP_CLASSES.index(target.d))
            valid_rows.append([q in target.pp_set for q in range(4)])
    return Rung2Batch(
        a_index=torch.tensor(a_list, dtype=torch.long),
        b_index=torch.tensor(b_list, dtype=torch.long),
        is_admit=torch.tensor(admit_list, dtype=torch.bool),
        field_targets={
            "S": torch.tensor(s_list, dtype=torch.long),
            "FFF": torch.tensor(fff_list, dtype=torch.long),
            "PP": torch.tensor(pp_list, dtype=torch.long),
            "pp": torch.tensor(pp_class_list, dtype=torch.long),
        },
        pp_valid=torch.tensor(valid_rows, dtype=torch.bool),
        event_index=np.array(event_list, dtype=np.int64),
        klass=np.array(class_list, dtype=object),
        input_s=np.array(input_s, dtype=np.int64),
        input_fff=np.array(input_fff, dtype=np.int64),
        size=len(a_list),
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_rung1(model, batch: Rung1Batch) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(batch.a_index, batch.b_index)
        swapped = model(batch.b_index, batch.a_index)
    predicted = logits.argmax(dim=-1)
    correct = (predicted == batch.target).numpy()
    per_class = {}
    for index, port in enumerate(PP_CLASSES):
        selected = (batch.target == index).numpy()
        per_class[f"{port:02b}"] = _round(
            fraction(int((selected & correct).sum()), int(selected.sum()))
        )
    return {
        "n": batch.size,
        "pp3_exact_accuracy": _round(float(correct.mean())),
        "per_port_accuracy": per_class,
        "swap_invariance_error": _round(
            float((logits - swapped).abs().max().item())
        ),
        "swap_argmax_disagreement_rate": _round(
            float((predicted != swapped.argmax(dim=-1)).float().mean().item())
        ),
        "swap_scores_bitwise_identical": bool(torch.equal(logits, swapped)),
        "correctness_criterion": (
            "integer equality against the local port of the certified forced-third "
            "Event; never float proximity"
        ),
    }


def evaluate_rung2(model, batch: Rung2Batch, resolver, table) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(batch.a_index, batch.b_index)
        swapped = model(batch.b_index, batch.a_index)

    admit_predicted = (out["gate"] > 0.0).numpy()
    is_admit = batch.is_admit.numpy()
    # Under the declared Rung-2 repair the identity fields are copied from the
    # input pair instead of predicted. The copy is recorded per field so no
    # reader can mistake a copied 1.0 for a learned one.
    copied = tuple(name for name in ("S", "FFF") if name not in out)
    s_hat = (
        out["S"].argmax(dim=-1).numpy() if "S" in out else batch.input_s
    )
    fff_hat = (
        out["FFF"].argmax(dim=-1).numpy() + 1 if "FFF" in out else batch.input_fff
    )
    pp_slot_hat = out["PP"].argmax(dim=-1).numpy()
    d_hat = np.array([PP_CLASSES[i] for i in out["pp"].argmax(dim=-1).numpy()])

    valid = batch.pp_valid.numpy()
    n_admit = int(is_admit.sum())
    n_reject = int((~is_admit).sum())

    field_hits = {
        "S": int((is_admit & (s_hat == batch.field_targets["S"].numpy())).sum()),
        "FFF": int(
            (is_admit & (fff_hat == batch.field_targets["FFF"].numpy() + 1)).sum()
        ),
        "PP": int(
            sum(
                1
                for i in range(batch.size)
                if is_admit[i] and valid[i][pp_slot_hat[i]]
            )
        ),
        "pp": int(
            (
                is_admit
                & (
                    d_hat
                    == np.array(
                        [PP_CLASSES[i] for i in batch.field_targets["pp"].numpy()]
                    )
                )
            ).sum()
        ),
    }

    # Exact structured address, then the fixed resolver. Both recorded.
    address_correct = np.zeros(batch.size, dtype=bool)
    resolved_correct = np.zeros(batch.size, dtype=bool)
    resolved_index = np.full(batch.size, -3, dtype=np.int64)
    for i in range(batch.size):
        if not is_admit[i]:
            continue
        fields = (int(s_hat[i]), int(fff_hat[i]), int(pp_slot_hat[i]), int(d_hat[i]))
        address_correct[i] = (
            fields[0] == int(batch.field_targets["S"][i])
            and fields[1] == int(batch.field_targets["FFF"][i]) + 1
            and valid[i][fields[2]]
            and fields[3] == PP_CLASSES[int(batch.field_targets["pp"][i])]
        )
        got = resolver.resolve(*fields)
        resolved_index[i] = got
        resolved_correct[i] = got == batch.event_index[i]

    # Joint success: the admission decision right AND, when admitted, the
    # resolved Event exactly right. Same shape as 009.02's joint metric.
    joint = np.where(
        is_admit,
        admit_predicted & resolved_correct,
        ~admit_predicted,
    )

    sensitivity = fraction(int((is_admit & admit_predicted).sum()), n_admit)
    specificity = fraction(int((~is_admit & ~admit_predicted).sum()), n_reject)

    def class_accuracy(name: str) -> tuple[float | None, int]:
        selected = batch.klass == name
        total = int(selected.sum())
        return fraction(int((selected & ~admit_predicted).sum()), total), total

    repeated, n_repeated = class_accuracy("REPEATED")
    disjoint, n_disjoint = class_accuracy("SAME_HABITAT_DISJOINT")
    cross, n_cross = class_accuracy("CROSS_HABITAT")

    nontrivial = int(
        sum(
            1
            for i in range(batch.size)
            if is_admit[i]
            and valid[i][pp_slot_hat[i]]
            and d_hat[i] == PP_CLASSES[int(batch.field_targets["pp"][i])]
        )
    )

    swap_error = max(
        float((out[key] - swapped[key]).abs().max().item()) for key in out
    )

    return {
        "n": batch.size,
        "n_admit": n_admit,
        "n_nonadmission": n_reject,
        "n_repeated": n_repeated,
        "n_same_habitat_disjoint": n_disjoint,
        "n_cross_habitat": n_cross,
        "admission_sensitivity": _round(sensitivity),
        "admission_specificity": _round(specificity),
        "admission_balanced_accuracy": _round(
            None
            if sensitivity is None or specificity is None
            else 0.5 * (sensitivity + specificity)
        ),
        "field_S_accuracy": _round(fraction(field_hits["S"], n_admit)),
        "field_FFF_accuracy": _round(fraction(field_hits["FFF"], n_admit)),
        "field_PP_accuracy": _round(fraction(field_hits["PP"], n_admit)),
        "field_pp_accuracy": _round(fraction(field_hits["pp"], n_admit)),
        "copied_fields": list(copied),
        "learned_fields": sorted(key for key in out if key != "gate"),
        "nontrivial_fields_accuracy": _round(fraction(nontrivial, n_admit)),
        "exact_structured_address_accuracy": _round(
            fraction(int(address_correct.sum()), n_admit)
        ),
        "positive_forced_third_exact_accuracy": _round(
            fraction(int(resolved_correct.sum()), n_admit)
        ),
        "resolution_drop": _round(
            fraction(
                int(address_correct.sum()) - int(resolved_correct.sum()), n_admit
            )
        ),
        "joint_exact_success": _round(float(joint.mean())),
        "repeated_pair_nonadmission_accuracy": _round(repeated),
        "same_habitat_disjoint_nonadmission_accuracy": _round(disjoint),
        "cross_habitat_nonadmission_accuracy": _round(cross),
        "unresolved_predictions": int((resolved_index == -2).sum()),
        "swap_invariance_error": _round(swap_error),
        "swap_scores_bitwise_identical": all(
            bool(torch.equal(out[key], swapped[key])) for key in out
        ),
        "copied_field_note": COPIED_FIELD_NOTE,
        "correctness_criterion": (
            "exact certified Event identity: integer index equality against "
            "record.target_index after the fixed resolver, which task.py already "
            "validated through projective.equivalent"
        ),
    }


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def train_rung1(arm, fold, seed, config: SweepConfig, dataset, table) -> dict:
    started = time.perf_counter()
    train_positions, test_positions = rung1_positions(dataset, fold)
    model = build_port_scorer(arm, config.scorer, config.ladder, seed=seed)
    set_seed(seed)

    train_batch = rung1_batch(dataset, table, train_positions)
    test_batch = rung1_batch(dataset, table, test_positions)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(config.train.steps):
        logits = model(train_batch.a_index, train_batch.b_index)
        loss = port_loss(logits, train_batch.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_metrics = evaluate_rung1(model, train_batch)
    test_metrics = evaluate_rung1(model, test_batch)
    marginal = rung1_marginal(dataset, table, train_positions)

    return {
        "rung": 1,
        "arm": arm.name,
        "fold_family": fold.family,
        "fold_name": fold.name,
        "seed": int(seed),
        "param_count": count_params(model),
        "train_size": train_batch.size,
        "test_size": test_batch.size,
        "pp3_exact_accuracy": test_metrics["pp3_exact_accuracy"],
        "train_accuracy": train_metrics["pp3_exact_accuracy"],
        "generalization_gap": _round(
            train_metrics["pp3_exact_accuracy"] - test_metrics["pp3_exact_accuracy"]
        ),
        "swap_invariance_error": test_metrics["swap_invariance_error"],
        "swap_argmax_disagreement_rate": test_metrics[
            "swap_argmax_disagreement_rate"
        ],
        "train_marginal_port": marginal,
        "final_loss": _round(float(loss.item())),
        "steps": config.train.steps,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "wall_sec": round(time.perf_counter() - started, 3),
    }


def train_rung2(arm, fold, seed, config: SweepConfig, dataset, table, resolver) -> dict:
    started = time.perf_counter()
    train_positions, test_positions = rung2_positions(dataset, fold)
    model = build_structured_scorer(arm, config.scorer, config.ladder, seed=seed)
    set_seed(seed)

    train_batch = rung2_batch(dataset, table, train_positions)
    test_batch = rung2_batch(dataset, table, test_positions)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    terms: dict[str, torch.Tensor] = {}
    for _ in range(config.train.steps):
        out = model(train_batch.a_index, train_batch.b_index)
        terms = structured_loss(
            out, train_batch.is_admit, train_batch.field_targets, train_batch.pp_valid
        )
        optimizer.zero_grad()
        terms["total"].backward()
        optimizer.step()

    train_metrics = evaluate_rung2(model, train_batch, resolver, table)
    test_metrics = evaluate_rung2(model, test_batch, resolver, table)

    row = {
        "rung": 2,
        "arm": arm.name,
        "fold_family": fold.family,
        "fold_name": fold.name,
        "seed": int(seed),
        "param_count": count_params(model),
        "train_size": train_batch.size,
        "test_size": test_batch.size,
        "train_accuracy": train_metrics["joint_exact_success"],
        "generalization_gap": _round(
            train_metrics["joint_exact_success"] - test_metrics["joint_exact_success"]
        ),
        "final_loss": _round(float(terms["total"].item())),
        "loss_terms": {
            key: _round(float(value.item()))
            for key, value in sorted(terms.items())
        },
        "steps": config.train.steps,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "wall_sec": round(time.perf_counter() - started, 3),
    }
    for name in RUNG2_METRICS:
        if name in test_metrics:
            row[name] = test_metrics[name]
    row["train_accuracy"] = train_metrics["joint_exact_success"]
    row["generalization_gap"] = _round(
        train_metrics["joint_exact_success"] - test_metrics["joint_exact_success"]
    )
    return row


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def stats(values) -> dict:
    defined = [float(v) for v in values if v is not None]
    if not defined:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    array = np.asarray(defined, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": _round(float(array.mean())),
        "median": _round(float(np.median(array))),
        "std": _round(float(array.std(ddof=0))),
        "min": _round(float(array.min())),
        "max": _round(float(array.max())),
    }


def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['arm']}|{row['fold_family']}", []).append(row)
    return {
        key: {
            "n_runs": len(group),
            "n_folds": len({row["fold_name"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}),
            **{name: stats([row.get(name) for row in group]) for name in metrics},
        }
        for key, group in sorted(grouped.items())
    }


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def run(config: SweepConfig, *, verbose: bool = True) -> dict:
    started = time.perf_counter()
    dataset = build_dataset()
    tables = code_tables(dataset)
    resolvers = build_resolvers(dataset)
    arms = {arm.name: arm for arm in all_arms(dataset, include_e=False)}
    folds = [
        fold
        for fold in structural_folds(all_folds(dataset))
        if fold.family in config.families
    ]

    rung1_rows: list[dict] = []
    rung2_rows: list[dict] = []
    total = len(config.arm_names) * len(folds) * len(config.seeds)
    done = 0
    for arm_name in config.arm_names:
        arm = arms[arm_name]
        table = output_table(arm_name, tables)
        resolver = resolvers[arm_name]
        for fold in folds:
            for seed in config.seeds:
                rung1_rows.append(
                    train_rung1(arm, fold, seed, config, dataset, table)
                )
                rung2_rows.append(
                    train_rung2(arm, fold, seed, config, dataset, table, resolver)
                )
                done += 1
                if verbose and done % 25 == 0:
                    print(
                        f"  {done}/{total} cells "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )

    return {
        "config": config.as_dict(),
        "n_cells": total,
        "n_rung1_runs": len(rung1_rows),
        "n_rung2_runs": len(rung2_rows),
        "folds": [[fold.family, fold.name] for fold in folds],
        "capacity_ledger": capacity_ledger(config.ladder, config.scorer),
        "rung1": {
            "runs": rung1_rows,
            "aggregates": aggregate(rung1_rows, RUNG1_METRICS),
            "runs_digest": digest(
                [strip_non_replayable(row) for row in rung1_rows]
            ),
        },
        "rung2": {
            "runs": rung2_rows,
            "aggregates": aggregate(rung2_rows, RUNG2_METRICS),
            "runs_digest": digest(
                [strip_non_replayable(row) for row in rung2_rows]
            ),
        },
        "sweep_wall_sec": round(time.perf_counter() - started, 3),
    }


def configurations() -> dict[str, SweepConfig]:
    """The two declared blocks: the default protocol, and the one Rung-2 repair.

    ``primary`` learns all four address fields. ``repair`` copies the two
    identity-carrying fields the certified relation itself copies and learns only
    the two it computes. Nothing else differs -- same arms, same folds, same
    seeds, same steps, same learning rate.
    """

    base = SweepConfig()
    return {
        "primary": base,
        "repair": replace(
            base, ladder=replace(base.ladder, copy_identity_fields=True)
        ),
    }


def smoke_config() -> SweepConfig:
    """A small declared subset: two arms, one family, two seeds."""

    return SweepConfig(
        seeds=(0, 1),
        arm_names=("B_sfp", "C_scrambled"),
        families=("LOHO",),
    )


def _block_laws(body: dict) -> dict:
    swap_clean = all(
        row["swap_invariance_error"] == 0.0 for row in body["rung1"]["runs"]
    ) and all(row["swap_invariance_error"] == 0.0 for row in body["rung2"]["runs"])
    return {
        "every_cell_ran": body["n_rung1_runs"]
        == body["n_rung2_runs"]
        == body["n_cells"],
        "swap_invariance_exact_everywhere": swap_clean,
        "no_unresolved_prediction_anywhere": all(
            row["test_metrics"]["unresolved_predictions"] == 0
            for row in body["rung2"]["runs"]
        ),
        "resolution_never_drops": all(
            row["test_metrics"]["resolution_drop"] == 0.0
            for row in body["rung2"]["runs"]
        ),
        "capacity_matched_across_science_arms": body["capacity_ledger"][
            "capacity_matched_across_all_science_arms"
        ],
    }


def _headline(body: dict) -> dict:
    r1 = body["rung1"]["aggregates"]
    r2 = body["rung2"]["aggregates"]
    return {
        "rung1": {
            key: value["pp3_exact_accuracy"]["mean"] for key, value in r1.items()
        },
        "rung2_forced_third": {
            key: value["positive_forced_third_exact_accuracy"]["mean"]
            for key, value in r2.items()
        },
        "rung2_exact_address": {
            key: value["exact_structured_address_accuracy"]["mean"]
            for key, value in r2.items()
        },
        "rung2_nontrivial_fields": {
            key: value["nontrivial_fields_accuracy"]["mean"]
            for key, value in r2.items()
        },
        "rung2_field_pp": {
            key: value["field_pp_accuracy"]["mean"] for key, value in r2.items()
        },
    }


def audit(configs: dict[str, SweepConfig] | None = None, *, verbose: bool = True) -> dict:
    configs = configs or configurations()
    blocks = {}
    laws = {}
    for name, config in configs.items():
        if verbose:
            print(f"=== block {name} ===", flush=True)
        body = run(config, verbose=verbose)
        blocks[name] = {**body, "headline": _headline(body)}
        for law, ok in _block_laws(body).items():
            laws[f"{name}/{law}"] = ok

    broken = sorted(name for name, ok in laws.items() if not ok)
    return {
        "module": "ladder_sweep",
        "purpose": (
            "the Rung-1, Rung-2 and Rung-3 runs under one protocol identical to "
            "009.02's, so the output object is the only difference"
        ),
        "executes": "009.05 sections 4, 5, 6 and 9",
        "fences": list(FENCES),
        "copied_field_note": COPIED_FIELD_NOTE,
        "trivial_fields": list(TRIVIAL_FIELDS),
        "nontrivial_fields": list(NONTRIVIAL_FIELDS),
        "blocks": blocks,
        "block_names": sorted(blocks),
        "repair_budget": (
            "009.05 section 9 permits at most one predeclared repair per rung. Rung 1 "
            "used none. Rung 2 uses exactly one: copy_identity_fields, declared in "
            "ladder_heads.LadderConfig and motivated by the diagnosed closed-set "
            "habitat re-identification failure visible in the primary block's "
            "field_FFF_accuracy."
        ),
        "provenance": {
            "base_commit": BASE_COMMIT,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "host": platform.platform(),
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - every declared cell ran in both blocks, swap invariance is "
                "exactly 0 in every run, no prediction failed to resolve, and "
                "resolution never cost a single correct address"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }


def _report(result: dict) -> None:
    for name in result["block_names"]:
        block = result["blocks"][name]
        print(f"--- block {name} ({block['n_cells']} cells) ---", flush=True)
        print("  Rung 1 pp3 exact accuracy (mean):", flush=True)
        for key, value in sorted(block["headline"]["rung1"].items()):
            print(f"    {key:<24} {value}", flush=True)
        print(
            "  Rung 2 forced third exact accuracy (mean, after resolution):",
            flush=True,
        )
        for key, value in sorted(block["headline"]["rung2_forced_third"].items()):
            print(f"    {key:<24} {value}", flush=True)
        print("  Rung 2 pp field accuracy (mean):", flush=True)
        for key, value in sorted(block["headline"]["rung2_field_pp"].items()):
            print(f"    {key:<24} {value}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    configs = (
        {"smoke": smoke_config()}
        if args.smoke
        else configurations()
    )
    target = SMOKE_OUTPUT if args.smoke else OUTPUT
    result = audit(configs, verbose=not args.check)
    text = render(result)
    _report(result)

    if args.check:
        if not target.exists():
            raise SystemExit(f"FAIL: missing {target}; run without --check first")
        expected = strip_non_replayable(json.loads(target.read_text()))
        observed = strip_non_replayable(json.loads(text))
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived sweep is not identical to {target.name} "
                "(wall-clock keys excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {target.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {target.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
