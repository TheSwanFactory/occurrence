"""009.07 sweep: Rung-1 chamber localization and Rung-2 recomposition.

One block, one protocol, no tuning::

    Rung 1   predict only the certified shared chamber q*      4 candidate scores
    Rung 2   BOTTOM gate + query-relative PP + learned pp,
             with S and FFF copied, through the fixed exact
             nonlearned 009.06 resolver

``009.07`` section 9 forbids an architecture search and there is none: one primary
candidate-scoring family, the frozen ``ScorerConfig``, the frozen ``TrainConfig``
values, the frozen splits, the frozen arms, the frozen seeds.

The Rung-2 metric suite is ``ladder_sweep.evaluate_rung2`` **called directly**, not
reimplemented. ``QueryRelativeScorer`` emits exactly the output dictionary that
function already consumes -- a gate, a ``(B, 4)`` ``PP`` tensor and a ``(B, 3)`` ``pp``
tensor, with ``S`` and ``FFF`` absent and therefore copied -- so
``positive_forced_third_exact_accuracy``, ``field_PP_accuracy``,
``field_pp_accuracy``, ``resolution_drop`` and the rest are the **same quantities**
``009.06`` reported, computed by the same code. That is what makes ``0.7016`` and
``0.7158`` legitimate comparison points.

Rung 1 adds two metrics, and the distinction between them is the one thing a reader
must not skip::

    qstar_exact_accuracy   argmax == the single certified shared chamber q*
    pp_set_accuracy        argmax in the target Event's two presentations

``009.06``'s ``0.7158`` was scored under the SET criterion. ``q*`` is a strictly
narrower question, so only ``pp_set_accuracy`` is comparable to it; both are reported
at both rungs and the disposition names which it used.
"""

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from arms import all_arms
from folds import all_folds, structural_folds
from ladder_resolver import build_resolvers
from ladder_sweep import (
    NON_REPLAYABLE_KEYS,
    RUNG2_METRICS,
    evaluate_rung2,
    rung2_batch,
    stats,
    strip_non_replayable,
)
from ladder_task import (
    ARM_OUTPUT_CONVENTION,
    PP_CLASSES,
    SCIENCE_ARMS,
    code_tables,
    output_table,
    rung2_positions,
    rung2_target,
)
from locator_baselines import chamber_marginal
from locator_heads import (
    LocatorConfig,
    build_chamber_locator,
    build_query_relative_scorer,
    capacity_ledger,
    chamber_loss,
    recomposition_loss,
)
from locator_task import (
    CHAMBER_VALUES,
    PRIOR_009_06,
    certified_chamber_of_block,
    chamber_actions,
    qstar_positions,
    qstar_target,
)
from scorer import ScorerConfig, count_params, set_seed
from sfp import SfpCodec, bits2
from task import Dataset, build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_locator_artifacts"
OUTPUT = ARTIFACTS / "locator_sweep.json"
SMOKE_OUTPUT = ARTIFACTS / "locator_sweep_smoke.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

#: The frozen ``009.02``/``009.06`` seeds.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

METRIC_DECIMALS = 12

RUNG1_METRICS = (
    "qstar_exact_accuracy",
    "pp_set_accuracy",
    "train_accuracy",
    "generalization_gap",
    "swap_invariance_error",
    "swap_argmax_disagreement_rate",
)

#: Rung-2 metrics beyond ``ladder_sweep.RUNG2_METRICS``: the point criterion on
#: ``q*``, and the two oracle-assisted localizers ``009.07`` section 5.2 requires.
RUNG2_EXTRA_METRICS = (
    "qstar_exact_accuracy",
    "learned_pp_exact_pp_forced_third_accuracy",
    "exact_pp_learned_pp_forced_third_accuracy",
)

FENCES = (
    "One protocol, fixed before any run; no tuning and no per-arm adjustment.",
    "Correctness is exact certified Event identity, never float proximity.",
    "Evaluation is never masked.",
    (
        "positive_forced_third_exact_accuracy is computed by ladder_sweep."
        "evaluate_rung2 itself, so it is the SAME quantity 009.06 reported."
    ),
    (
        "qstar_exact_accuracy is a POINT criterion and is NOT comparable to 009.06's "
        "set-criterion field_PP_accuracy; pp_set_accuracy is the comparable one."
    ),
    (
        "The headline Rung-2 number uses learned PP and learned pp. The "
        "oracle-assisted cells are localizers and are never quoted as a system."
    ),
    (
        "The resolver is applied after a structured address exists and may not repair "
        "it."
    ),
    (
        "Copied fields S and FFF are recorded per run so no reader can mistake a "
        "copied 1.0 for a learned one."
    ),
    "Rung 1 is admitted-only, so admission is not a target there and is not quoted.",
    "BOTTOM stays distinct from algebraic zero and from UNRESOLVED.",
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), METRIC_DECIMALS)


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """The one training protocol. ``ladder_sweep.TrainConfig``'s values, unchanged."""

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
                "none. Identical to the 009.02 and 009.06 protocol so that the PP "
                "output constitution is the only difference."
            ),
            "early_stopping": "none",
            "repairs_used": (
                "none. 009.07 section 9 permits at most one repair for a mechanically "
                "diagnosed pathology; see the audit's repair_status."
            ),
        }


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...] = SEEDS
    arm_names: tuple[str, ...] = SCIENCE_ARMS
    families: tuple[str, ...] = ("LOHO", "LOFPO")
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    locator: LocatorConfig = field(default_factory=LocatorConfig)
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
            "locator": self.locator.as_dict(),
            "train": self.train.as_dict(),
        }


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rung1Batch:
    a_index: torch.Tensor
    b_index: torch.Tensor
    target: torch.Tensor          # the certified q*, as a chamber value 0..3
    pp_valid: torch.Tensor        # (N, 4) the target Event's two presentations
    size: int


def rung1_batch(
    dataset: Dataset, table, positions, chamber_of_block, action
) -> Rung1Batch:
    """Rung-1 batch: the point ``q*`` target, plus the set mask for the relaxed metric.

    ``pp_valid`` is built from ``ladder_task.rung2_target``'s ``pp_set`` -- the same
    object ``009.06`` scored its ``PP`` head against -- so ``pp_set_accuracy`` is
    measured against exactly ``009.06``'s criterion and the comparison is like for
    like.
    """

    a_list, b_list, target_list, valid_rows = [], [], [], []
    for position in positions:
        record = dataset.records[position]
        a_list.append(record.a_index)
        b_list.append(record.b_index)
        target_list.append(qstar_target(record, chamber_of_block, action))
        pp_set = rung2_target(record, table).pp_set
        valid_rows.append([q in pp_set for q in CHAMBER_VALUES])
    return Rung1Batch(
        a_index=torch.tensor(a_list, dtype=torch.long),
        b_index=torch.tensor(b_list, dtype=torch.long),
        target=torch.tensor(target_list, dtype=torch.long),
        pp_valid=torch.tensor(valid_rows, dtype=torch.bool),
        size=len(a_list),
    )


def chamber_targets(dataset: Dataset, positions, chamber_of_block, action) -> np.ndarray:
    """The certified ``q*`` for a Rung-2 position list; ``-1`` on a BOTTOM row.

    A non-admitted pair has no certified shared chamber. The placeholder is masked
    out of the loss and never read by a metric, exactly as
    ``ladder_sweep.rung2_batch`` handles its own placeholders.
    """

    out = []
    for position in positions:
        record = dataset.records[position]
        out.append(
            qstar_target(record, chamber_of_block, action) if record.admitted else -1
        )
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_rung1(model, batch: Rung1Batch) -> dict:
    """Both criteria, plus the swap check, on one Rung-1 batch."""

    model.eval()
    with torch.no_grad():
        scores = model(batch.a_index, batch.b_index)
        swapped = model(batch.b_index, batch.a_index)
    predicted = scores.argmax(dim=-1)
    exact = (predicted == batch.target).numpy()
    valid = batch.pp_valid.numpy()
    in_set = np.array(
        [bool(valid[i][int(predicted[i])]) for i in range(batch.size)]
    )
    per_chamber = {}
    for q in CHAMBER_VALUES:
        selected = (batch.target == q).numpy()
        per_chamber[bits2(q)] = _round(
            fraction(int((selected & exact).sum()), int(selected.sum()))
        )
    return {
        "n": batch.size,
        "qstar_exact_accuracy": _round(float(exact.mean())),
        "pp_set_accuracy": _round(float(in_set.mean())),
        "per_chamber_accuracy": per_chamber,
        "swap_invariance_error": _round(
            float((scores - swapped).abs().max().item())
        ),
        "swap_argmax_disagreement_rate": _round(
            float((predicted != swapped.argmax(dim=-1)).float().mean().item())
        ),
        "swap_scores_bitwise_identical": bool(torch.equal(scores, swapped)),
        "correctness_criterion": (
            "qstar_exact_accuracy is integer equality against the certified shared "
            "chamber of the certified block; pp_set_accuracy relaxes it to the target "
            "Event's two block-incidence presentations, which is 009.06's criterion. "
            "Never float proximity."
        ),
    }


def localization_diagnostics(
    model, batch, resolver, chamber_target: np.ndarray
) -> dict:
    """``009.07`` section 5.2: which field the residual Rung-2 error lives in.

    Three cells, of which only the third is a system::

        learned PP + exact pp    isolates PP error
        exact PP + learned pp    isolates pp error
        learned PP + learned pp  the main result, reported by evaluate_rung2

    The two oracle-assisted cells substitute the certified field for the learned one
    and are ceilings/localizers, never deployable learned systems. They are computed
    here with the same fixed resolver and the same exact-Event-identity criterion so
    they sit on the same scale as the headline number.
    """

    model.eval()
    with torch.no_grad():
        out = model(batch.a_index, batch.b_index)
    is_admit = batch.is_admit.numpy()
    n_admit = int(is_admit.sum())
    s_hat = batch.input_s
    fff_hat = batch.input_fff
    pp_slot_hat = out["PP"].argmax(dim=-1).numpy()
    d_hat = np.array([PP_CLASSES[i] for i in out["pp"].argmax(dim=-1).numpy()])
    d_exact = np.array(
        [PP_CLASSES[i] for i in batch.field_targets["pp"].numpy()]
    )

    def resolved(pp_values: np.ndarray, d_values: np.ndarray) -> float | None:
        hits = 0
        for i in range(batch.size):
            if not is_admit[i]:
                continue
            got = resolver.resolve(
                int(s_hat[i]), int(fff_hat[i]), int(pp_values[i]), int(d_values[i])
            )
            hits += got == batch.event_index[i]
        return fraction(hits, n_admit)

    qstar_hits = int(
        sum(
            1
            for i in range(batch.size)
            if is_admit[i] and pp_slot_hat[i] == chamber_target[i]
        )
    )
    return {
        "qstar_exact_accuracy": _round(fraction(qstar_hits, n_admit)),
        "learned_pp_exact_pp_forced_third_accuracy": _round(
            resolved(pp_slot_hat, d_exact)
        ),
        "exact_pp_learned_pp_forced_third_accuracy": _round(
            resolved(chamber_target.clip(min=0), d_hat)
        ),
        "oracle_cell_status": (
            "the two oracle-assisted cells substitute a certified field for a learned "
            "one. They localize error and bound it; they are not learned systems and "
            "are never quoted as the result."
        ),
    }


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def train_rung1(
    arm, codes, fold, seed, config: SweepConfig, dataset, table, chamber_of_block, action
) -> dict:
    started = time.perf_counter()
    train_positions, test_positions = qstar_positions(dataset, fold)
    model = build_chamber_locator(arm, codes, config.scorer, config.locator, seed=seed)
    set_seed(seed)

    train_batch = rung1_batch(
        dataset, table, train_positions, chamber_of_block, action
    )
    test_batch = rung1_batch(dataset, table, test_positions, chamber_of_block, action)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(config.train.steps):
        scores = model(train_batch.a_index, train_batch.b_index)
        loss = chamber_loss(scores, train_batch.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_metrics = evaluate_rung1(model, train_batch)
    test_metrics = evaluate_rung1(model, test_batch)
    majority = chamber_marginal(dataset, chamber_of_block, action, train_positions)

    return {
        "rung": 1,
        "arm": arm.name,
        "fold_family": fold.family,
        "fold_name": fold.name,
        "seed": int(seed),
        "param_count": count_params(model),
        "train_size": train_batch.size,
        "test_size": test_batch.size,
        "qstar_exact_accuracy": test_metrics["qstar_exact_accuracy"],
        "pp_set_accuracy": test_metrics["pp_set_accuracy"],
        "train_accuracy": train_metrics["qstar_exact_accuracy"],
        "generalization_gap": _round(
            train_metrics["qstar_exact_accuracy"]
            - test_metrics["qstar_exact_accuracy"]
        ),
        "swap_invariance_error": test_metrics["swap_invariance_error"],
        "swap_argmax_disagreement_rate": test_metrics[
            "swap_argmax_disagreement_rate"
        ],
        "train_majority_chamber": int(majority),
        "final_loss": _round(float(loss.item())),
        "steps": config.train.steps,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "wall_sec": round(time.perf_counter() - started, 3),
    }


def train_rung2(
    arm,
    codes,
    fold,
    seed,
    config: SweepConfig,
    dataset,
    table,
    resolver,
    chamber_of_block,
    action,
) -> dict:
    started = time.perf_counter()
    train_positions, test_positions = rung2_positions(dataset, fold)
    model = build_query_relative_scorer(
        arm, codes, config.scorer, config.locator, seed=seed
    )
    set_seed(seed)

    train_batch = rung2_batch(dataset, table, train_positions)
    test_batch = rung2_batch(dataset, table, test_positions)
    train_chambers = chamber_targets(
        dataset, train_positions, chamber_of_block, action
    )
    test_chambers = chamber_targets(dataset, test_positions, chamber_of_block, action)
    train_chamber_tensor = torch.tensor(train_chambers.clip(min=0), dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    terms: dict[str, torch.Tensor] = {}
    for _ in range(config.train.steps):
        out = model(train_batch.a_index, train_batch.b_index)
        terms = recomposition_loss(
            out,
            train_batch.is_admit,
            train_chamber_tensor,
            train_batch.field_targets["pp"],
        )
        optimizer.zero_grad()
        terms["total"].backward()
        optimizer.step()

    # The metric suite is ladder_sweep's own function, called directly.
    train_metrics = evaluate_rung2(model, train_batch, resolver, table)
    test_metrics = evaluate_rung2(model, test_batch, resolver, table)
    train_metrics.update(
        localization_diagnostics(model, train_batch, resolver, train_chambers)
    )
    test_metrics.update(
        localization_diagnostics(model, test_batch, resolver, test_chambers)
    )

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
        "forced_third_generalization_gap": _round(
            train_metrics["positive_forced_third_exact_accuracy"]
            - test_metrics["positive_forced_third_exact_accuracy"]
        ),
        "pp_generalization_gap": _round(
            train_metrics["field_PP_accuracy"] - test_metrics["field_PP_accuracy"]
        ),
        "final_loss": _round(float(terms["total"].item())),
        "loss_terms": {
            key: _round(float(value.item())) for key, value in sorted(terms.items())
        },
        "steps": config.train.steps,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "wall_sec": round(time.perf_counter() - started, 3),
    }
    for name in RUNG2_METRICS + RUNG2_EXTRA_METRICS:
        if name in test_metrics:
            row[name] = test_metrics[name]
    row["pp_set_accuracy"] = test_metrics["field_PP_accuracy"]
    return row


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

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
    codec = SfpCodec()
    chamber_of_block = certified_chamber_of_block(dataset, codec)
    actions = chamber_actions()
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
        codes = table
        resolver = resolvers[arm_name]
        action = actions[arm_name]
        for fold in folds:
            for seed in config.seeds:
                rung1_rows.append(
                    train_rung1(
                        arm,
                        codes,
                        fold,
                        seed,
                        config,
                        dataset,
                        table,
                        chamber_of_block,
                        action,
                    )
                )
                rung2_rows.append(
                    train_rung2(
                        arm,
                        codes,
                        fold,
                        seed,
                        config,
                        dataset,
                        table,
                        resolver,
                        chamber_of_block,
                        action,
                    )
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
        "capacity_ledger": capacity_ledger(tables, config.locator, config.scorer),
        "rung1": {
            "runs": rung1_rows,
            "aggregates": aggregate(rung1_rows, RUNG1_METRICS),
            "runs_digest": digest(
                [strip_non_replayable(row) for row in rung1_rows]
            ),
        },
        "rung2": {
            "runs": rung2_rows,
            "aggregates": aggregate(
                rung2_rows,
                RUNG2_METRICS
                + RUNG2_EXTRA_METRICS
                + (
                    "pp_set_accuracy",
                    "forced_third_generalization_gap",
                    "pp_generalization_gap",
                ),
            ),
            "runs_digest": digest(
                [strip_non_replayable(row) for row in rung2_rows]
            ),
        },
        "sweep_wall_sec": round(time.perf_counter() - started, 3),
    }


def configurations() -> dict[str, SweepConfig]:
    """The one declared block. ``009.07`` section 9 forbids a search."""

    return {"primary": SweepConfig()}


def smoke_config() -> SweepConfig:
    """A small declared subset: two arms, one family, two seeds."""

    return SweepConfig(
        seeds=(0, 1), arm_names=("B_sfp", "C_scrambled"), families=("LOHO",)
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
        "copied_fields_are_S_and_FFF_everywhere": all(
            row["test_metrics"]["copied_fields"] == ["S", "FFF"]
            for row in body["rung2"]["runs"]
        ),
        "learned_fields_are_PP_and_pp_everywhere": all(
            row["test_metrics"]["learned_fields"] == ["PP", "pp"]
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
        "rung1_qstar_exact": {
            key: value["qstar_exact_accuracy"]["mean"] for key, value in r1.items()
        },
        "rung1_pp_set": {
            key: value["pp_set_accuracy"]["mean"] for key, value in r1.items()
        },
        "rung2_forced_third": {
            key: value["positive_forced_third_exact_accuracy"]["mean"]
            for key, value in r2.items()
        },
        "rung2_field_PP_set": {
            key: value["field_PP_accuracy"]["mean"] for key, value in r2.items()
        },
        "rung2_qstar_exact": {
            key: value["qstar_exact_accuracy"]["mean"] for key, value in r2.items()
        },
        "rung2_field_pp": {
            key: value["field_pp_accuracy"]["mean"] for key, value in r2.items()
        },
        "rung2_learned_PP_exact_pp": {
            key: value["learned_pp_exact_pp_forced_third_accuracy"]["mean"]
            for key, value in r2.items()
        },
        "rung2_exact_PP_learned_pp": {
            key: value["exact_pp_learned_pp_forced_third_accuracy"]["mean"]
            for key, value in r2.items()
        },
    }


def audit(configs: dict[str, SweepConfig] | None = None, *, verbose: bool = True) -> dict:
    configs = configs or configurations()
    blocks = {}
    for name, config in configs.items():
        if verbose:
            print(f"--- block {name} ---", flush=True)
        body = run(config, verbose=verbose)
        body["laws"] = _block_laws(body)
        body["headline"] = _headline(body)
        blocks[name] = body

    laws = {
        name: body["laws"] for name, body in blocks.items()
    }
    agrees = all(
        all(value for value in body.values()) for body in laws.values()
    )
    return {
        "module": "locator_sweep",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "block_names": sorted(blocks),
        "blocks": blocks,
        "laws": laws,
        "prior_009_06": dict(PRIOR_009_06),
        "metric_provenance": (
            "every Rung-2 metric in RUNG2_METRICS is produced by "
            "ladder_sweep.evaluate_rung2, imported and called directly, on batches "
            "built by ladder_sweep.rung2_batch. The two turns therefore report the "
            "same quantities computed by the same code."
        ),
        "criterion_disclosure": (
            "qstar_exact_accuracy is a POINT criterion on the certified shared "
            "chamber. field_PP_accuracy and pp_set_accuracy are the SET criterion "
            "009.06 used. Only the set criterion is comparable to 0.7158, and the "
            "point criterion is the stricter question 009.07 actually poses."
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "every declared cell ran, swap invariance is exact everywhere, no "
                "prediction was unresolved, the fixed resolver never dropped an "
                "address, S and FFF were copied and PP and pp learned in every run, "
                "and capacity is matched across the science arms"
                if agrees
                else "at least one block law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    for name in result["block_names"]:
        block = result["blocks"][name]
        head = block["headline"]
        print(f"--- block {name} ({block['n_cells']} cells) ---", flush=True)
        for label, key in (
            ("Rung 1 q* exact", "rung1_qstar_exact"),
            ("Rung 1 PP set   ", "rung1_pp_set"),
            ("Rung 2 forced 3 ", "rung2_forced_third"),
            ("Rung 2 PP set   ", "rung2_field_PP_set"),
            ("Rung 2 q* exact ", "rung2_qstar_exact"),
            ("Rung 2 pp       ", "rung2_field_pp"),
        ):
            print(f"  {label}:", flush=True)
            for cell, value in sorted(head[key].items()):
                print(f"    {cell:<24} {value}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    configs = {"smoke": smoke_config()} if args.smoke else configurations()
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


# Referenced for the artifact's replay contract; imported so a drift in
# ladder_sweep's non-replayable key list breaks here rather than silently.
assert "wall_sec" in NON_REPLAYABLE_KEYS
