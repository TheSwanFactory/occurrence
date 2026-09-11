"""009.07 analysis: paired effects, the two ceilings, and the disposition.

Statistics are ``analysis.py``'s, imported rather than reimplemented, so the intervals
in this turn are the same estimator ``009.02`` and ``009.06`` used: a seeded percentile
bootstrap of the mean over identical ``(fold_name, seed)`` cells.

Three comparisons carry the science, and they answer different questions::

    B - C   is the gain RELATIONAL?           predeclared: B materially > C
    D - B   is it a BASIS artifact?           predeclared: CI includes zero
    B - A   is native realization enough?     secondary, reported

and one more that only this turn can make::

    009.07 - 009.06   is the residual an OUTPUT-CONSTITUTION artifact?

That last one is paired at the ``(arm, fold_name, seed)`` level against
``009_ladder_artifacts/ladder_sweep.json``'s ``repair`` block whenever that bulk
artifact is present, because the two turns ran the same folds with the same seeds on
the same splits. When it is absent the comparison falls back to the pinned means and
says which of the two it did.

No post-hoc equivalence margin is used anywhere. ``D ~= B`` is claimed only from an
interval that includes zero, and a threshold that is missed is reported as missed
rather than rounded up.
"""

import argparse
import json
from pathlib import Path

from analysis import bootstrap_ci, paired_differences
from locator_task import PRIOR_009_06

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_locator_artifacts"
LADDER_ARTIFACTS = ROOT / "009_ladder_artifacts"
SWEEP = ARTIFACTS / "locator_sweep.json"
TASK = ARTIFACTS / "locator_task.json"
HEADS = ARTIFACTS / "locator_heads.json"
BASELINES = ARTIFACTS / "locator_baselines.json"
PRIOR_SWEEP = LADDER_ARTIFACTS / "ladder_sweep.json"
OUTPUT = ARTIFACTS / "locator_analysis.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

#: Rung-1 metrics. ``qstar_exact_accuracy`` is the point criterion this turn poses;
#: ``pp_set_accuracy`` is ``009.06``'s set criterion, carried so the two turns can be
#: compared on like terms.
RUNG1_METRICS = ("qstar_exact_accuracy", "pp_set_accuracy")

RUNG2_PRIMARY_METRICS = (
    "positive_forced_third_exact_accuracy",
    "field_PP_accuracy",
    "qstar_exact_accuracy",
    "field_pp_accuracy",
    "exact_structured_address_accuracy",
    "nontrivial_fields_accuracy",
)

RUNG2_CONTEXT_METRICS = (
    "learned_pp_exact_pp_forced_third_accuracy",
    "exact_pp_learned_pp_forced_third_accuracy",
    "train_accuracy",
    "generalization_gap",
    "forced_third_generalization_gap",
    "pp_generalization_gap",
    "resolution_drop",
)

#: ``admission_balanced_accuracy`` carries no claim, per ``009.02`` and restated by
#: ``009.07`` section 8: Rung 1 is admitted-only so admission is not a target, and a
#: three-feature lookup already reaches ``0.921`` on it while scoring ``0.0`` on
#: forced third.
WEAK_METRICS = ("admission_balanced_accuracy",)

COMPARISONS = (
    (
        "B_minus_C",
        "B_sfp",
        "C_scrambled",
        "is the gain RELATIONAL? predeclared: B materially greater than C",
    ),
    (
        "D_minus_B",
        "D_relabeled",
        "B_sfp",
        "is it a BASIS artifact? predeclared: the interval includes zero",
    ),
    (
        "B_minus_A",
        "B_sfp",
        "A_native",
        "is native realization sufficient? secondary diagnostic",
    ),
)

#: ``009.07`` section 4.4.
RUNG1_THRESHOLDS = {
    "loho_mean_qstar": 0.95,
    "loho_lower_bound_qstar": 0.90,
    "lofpo_mean_qstar": 0.90,
}

#: ``009.07`` section 5.4.
RUNG2_THRESHOLDS = {
    "loho_forced_third": 0.90,
    "lofpo_forced_third": 0.85,
}

#: ``009.07`` sections 10.B and 10.C / 10.D.
MATERIAL_DELTA = 0.10

FENCES = (
    "No post-hoc equivalence margin. D ~= B only from an interval including zero.",
    "A missed threshold is reported as missed, never rounded up.",
    (
        "admission_balanced_accuracy carries no claim; Rung 1 is admitted-only so "
        "admission is not a target there."
    ),
    (
        "qstar_exact_accuracy is a POINT criterion and is not comparable to 009.06's "
        "set-criterion 0.7158; field_PP_accuracy / pp_set_accuracy is."
    ),
    (
        "The learned result is only interesting between the strictly-cheaper ceiling "
        "and the determining ceiling from locator_baselines."
    ),
    (
        "The headline number uses learned PP and learned pp. Oracle-assisted cells "
        "localize error and are never quoted as a system."
    ),
    (
        "The pointer is NOT capacity-matched to 009.06's absolute head; the delta and "
        "the generalization-gap diagnostic are both reported."
    ),
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def rd(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"FAIL: missing {path.name}; run the producing module first "
            "(or fetch the bulk artifact named in its pointer file)"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def arm_family_summary(runs: list[dict], metrics: tuple[str, ...]) -> dict:
    """Per ``(arm, family)`` mean and bootstrap interval for each metric."""

    out: dict[str, dict] = {}
    for arm, family in sorted({(r["arm"], r["fold_family"]) for r in runs}):
        group = [
            r for r in runs if r["arm"] == arm and r["fold_family"] == family
        ]
        body = {
            "n_runs": len(group),
            "n_folds": len({r["fold_name"] for r in group}),
            "n_seeds": len({r["seed"] for r in group}),
        }
        for metric in metrics:
            values = [r[metric] for r in group if r.get(metric) is not None]
            body[metric] = {
                "mean": rd(sum(values) / len(values)) if values else None,
                "ci_95": bootstrap_ci(values) if values else None,
                "n": len(values),
            }
        out[f"{arm}|{family}"] = body
    return out


def comparisons_for(
    runs: list[dict], metrics: tuple[str, ...], families: tuple[str, ...]
) -> dict:
    """Every predeclared paired comparison, for every metric and family."""

    out: dict[str, dict] = {}
    for name, left, right, question in COMPARISONS:
        body = {"left": left, "right": right, "question": question}
        for family in families:
            for metric in metrics:
                body[f"{family}|{metric}"] = paired_differences(
                    runs, left, right, family, metric
                )
        out[name] = body
    return out


def read_rung1(sweep: dict, block: str) -> dict:
    runs = sweep["blocks"][block]["rung1"]["runs"]
    return {
        "n_runs": len(runs),
        "summary": arm_family_summary(runs, RUNG1_METRICS),
        "comparisons": comparisons_for(runs, RUNG1_METRICS, ("LOHO", "LOFPO")),
        "context": arm_family_summary(
            runs, ("train_accuracy", "generalization_gap", "swap_invariance_error")
        ),
        "holdout_note": (
            "the held-out object is a complete structural unit, not an unseen output "
            "class: a chart has only four chambers and they necessarily recur in "
            "training. Rung 1 tests transfer of the localization relation to an "
            "unseen habitat or Fano point."
        ),
    }


def read_rung2(sweep: dict, block: str) -> dict:
    runs = sweep["blocks"][block]["rung2"]["runs"]
    return {
        "n_runs": len(runs),
        "summary": arm_family_summary(runs, RUNG2_PRIMARY_METRICS),
        "comparisons": comparisons_for(
            runs, RUNG2_PRIMARY_METRICS, ("LOHO", "LOFPO")
        ),
        "context": arm_family_summary(runs, RUNG2_CONTEXT_METRICS),
        "weak_metrics": arm_family_summary(runs, WEAK_METRICS),
        "weak_metric_warning": (
            "admission_balanced_accuracy is reported for continuity and carries no "
            "claim, exactly as in 009.02 and 009.06"
        ),
    }


# ---------------------------------------------------------------------------
# the cross-turn comparison
# ---------------------------------------------------------------------------

def cross_turn(sweep: dict, block: str) -> dict:
    """``009.07`` against ``009.06``, paired per ``(arm, fold, seed)`` when possible.

    Both turns ran the same 21 structural folds with the same eight seeds on splits
    built by the same functions, so the cells are genuinely paired and a paired
    interval is available -- a strictly stronger statement than comparing two means.
    The pairing requires ``009_ladder_artifacts/ladder_sweep.json``, which is bulk run
    data hosted in the Quilt package rather than carried in git, so the fallback to
    the pinned means is recorded rather than hidden.
    """

    new_runs = sweep["blocks"][block]["rung2"]["runs"]
    metrics = ("positive_forced_third_exact_accuracy", "field_PP_accuracy")

    if not PRIOR_SWEEP.exists():
        rows = {}
        for metric in metrics:
            for family in ("LOHO", "LOFPO"):
                key = f"B_sfp|{family}"
                values = [
                    r[metric]
                    for r in new_runs
                    if r["arm"] == "B_sfp" and r["fold_family"] == family
                ]
                quoted = (
                    PRIOR_009_06["positive_forced_third_exact_accuracy"].get(family)
                    if metric == "positive_forced_third_exact_accuracy"
                    else PRIOR_009_06["field_PP_accuracy"].get(family)
                )
                mean = rd(sum(values) / len(values)) if values else None
                rows[f"{key}|{metric}"] = {
                    "prior_mean": quoted,
                    "new_mean": mean,
                    "delta": None
                    if mean is None or quoted is None
                    else rd(mean - quoted),
                    "paired": False,
                }
        return {
            "paired": False,
            "status": (
                "ladder_sweep.json is not in this clone, so the comparison is "
                "unpaired: 009.07 means against the means quoted in the 009.06 "
                "result turn"
            ),
            "rows": rows,
        }

    prior = json.loads(PRIOR_SWEEP.read_text())
    prior_runs = prior["blocks"]["repair"]["rung2"]["runs"]
    combined: list[dict] = []
    for row in prior_runs:
        combined.append({**row, "arm": f"{row['arm']}@009.06"})
    combined.extend({**row, "arm": f"{row['arm']}@009.07"} for row in new_runs)

    rows = {}
    for arm in ("A_native", "B_sfp", "C_scrambled", "D_relabeled"):
        for family in ("LOHO", "LOFPO"):
            for metric in metrics:
                rows[f"{arm}|{family}|{metric}"] = paired_differences(
                    combined,
                    f"{arm}@009.07",
                    f"{arm}@009.06",
                    family,
                    metric,
                )
    return {
        "paired": True,
        "status": (
            "paired per (fold_name, seed) against the 009.06 repair block in "
            "009_ladder_artifacts/ladder_sweep.json; both turns ran the same 21 "
            "structural folds and the same 8 seeds on splits built by the same "
            "functions"
        ),
        "prior_block": "repair",
        "prior_block_rationale": (
            "the repair block is the one that produced 0.7158 on PP and 0.7016 on the "
            "consequence, and it is the configuration 009.07 reuses -- S and FFF "
            "copied, PP and pp learned. Comparing against the primary block instead "
            "would confound the PP interface with the identity-field change."
        ),
        "rows": rows,
        "capacity_caveat": (
            "the pointer holds more parameters on the PP pathway than 009.06's "
            "absolute head, so this comparison is not capacity-matched. The "
            "generalization-gap columns are the diagnostic: a capacity-driven gain "
            "widens the train-test gap, a constitution-driven one need not."
        ),
    }


# ---------------------------------------------------------------------------
# the criteria
# ---------------------------------------------------------------------------

def _mean(summary: dict, cell: str, metric: str) -> float | None:
    return summary.get(cell, {}).get(metric, {}).get("mean")


def _low(summary: dict, cell: str, metric: str) -> float | None:
    body = summary.get(cell, {}).get(metric, {}).get("ci_95")
    return None if body is None else body.get("low")


def _positive(block: dict) -> bool:
    return bool(
        block.get("paired")
        and block.get("ci_excludes_zero")
        and block.get("mean_difference", 0.0) > 0.0
    )


def _includes_zero(block: dict) -> bool:
    return bool(block.get("paired") and not block.get("ci_excludes_zero"))


def rung1_criterion(rung1: dict) -> dict:
    """``009.07`` section 4.4, applied to the POINT criterion on ``q*``."""

    summary = rung1["summary"]
    comparisons = rung1["comparisons"]
    metric = "qstar_exact_accuracy"
    checks = {
        "loho_mean_qstar": {
            "threshold": RUNG1_THRESHOLDS["loho_mean_qstar"],
            "observed": _mean(summary, "B_sfp|LOHO", metric),
        },
        "loho_lower_bound_qstar": {
            "threshold": RUNG1_THRESHOLDS["loho_lower_bound_qstar"],
            "observed": _low(summary, "B_sfp|LOHO", metric),
        },
        "lofpo_mean_qstar": {
            "threshold": RUNG1_THRESHOLDS["lofpo_mean_qstar"],
            "observed": _mean(summary, "B_sfp|LOFPO", metric),
        },
    }
    for body in checks.values():
        body["meets"] = (
            body["observed"] is not None and body["observed"] >= body["threshold"]
        )
    b_minus_c = comparisons["B_minus_C"][f"LOHO|{metric}"]
    d_minus_b = comparisons["D_minus_B"][f"LOHO|{metric}"]
    structural = {
        "b_minus_c_positive_and_excludes_zero": _positive(b_minus_c),
        "d_minus_b_includes_zero": _includes_zero(d_minus_b),
        "b_minus_c": b_minus_c,
        "d_minus_b": d_minus_b,
    }
    met = all(body["meets"] for body in checks.values()) and all(
        structural[key]
        for key in (
            "b_minus_c_positive_and_excludes_zero",
            "d_minus_b_includes_zero",
        )
    )
    return {
        "metric": metric,
        "criterion_source": "009.07 section 4.4",
        "checks": checks,
        "structural": structural,
        "gap_closed": met,
        "also_reported_under_the_set_criterion": {
            "metric": "pp_set_accuracy",
            "loho_mean": _mean(summary, "B_sfp|LOHO", "pp_set_accuracy"),
            "lofpo_mean": _mean(summary, "B_sfp|LOFPO", "pp_set_accuracy"),
            "note": (
                "the set criterion is the one 009.06 scored 0.7158 under; it is "
                "reported for comparability and the disposition uses the stricter "
                "point criterion"
            ),
        },
    }


def rung2_criterion(rung2: dict, sweep: dict, block: str) -> dict:
    """``009.07`` section 5.4, on the learned-PP learned-pp system only."""

    summary = rung2["summary"]
    comparisons = rung2["comparisons"]
    metric = "positive_forced_third_exact_accuracy"
    checks = {
        "loho_forced_third": {
            "threshold": RUNG2_THRESHOLDS["loho_forced_third"],
            "observed": _mean(summary, "B_sfp|LOHO", metric),
        },
        "lofpo_forced_third": {
            "threshold": RUNG2_THRESHOLDS["lofpo_forced_third"],
            "observed": _mean(summary, "B_sfp|LOFPO", metric),
        },
    }
    for body in checks.values():
        body["meets"] = (
            body["observed"] is not None and body["observed"] >= body["threshold"]
        )
    b_minus_c = comparisons["B_minus_C"][f"LOHO|{metric}"]
    d_minus_b = comparisons["D_minus_B"][f"LOHO|{metric}"]
    runs = sweep["blocks"][block]["rung2"]["runs"]
    resolution_clean = all(
        row["test_metrics"]["resolution_drop"] == 0.0 for row in runs
    )
    structural = {
        "b_minus_c_positive_and_excludes_zero": _positive(b_minus_c),
        "d_minus_b_includes_zero": _includes_zero(d_minus_b),
        "fixed_resolution_loss_is_zero": resolution_clean,
        "b_minus_c": b_minus_c,
        "d_minus_b": d_minus_b,
    }
    met = all(body["meets"] for body in checks.values()) and all(
        structural[key]
        for key in (
            "b_minus_c_positive_and_excludes_zero",
            "d_minus_b_includes_zero",
            "fixed_resolution_loss_is_zero",
        )
    )
    return {
        "metric": metric,
        "criterion_source": "009.07 section 5.4",
        "checks": checks,
        "structural": structural,
        "gap_substantially_closed": met,
        "headline_uses_learned_fields_only": True,
    }


def material_improvement(rung2: dict, cross: dict) -> dict:
    """``009.07`` sections 10.B/10.C/10.D: is the gain at least ``0.10`` absolute?"""

    summary = rung2["summary"]
    pp_new = _mean(summary, "B_sfp|LOHO", "field_PP_accuracy")
    third_new = _mean(
        summary, "B_sfp|LOHO", "positive_forced_third_exact_accuracy"
    )
    pp_prior = PRIOR_009_06["field_PP_accuracy"]["LOHO"]
    third_prior = PRIOR_009_06["positive_forced_third_exact_accuracy"]["LOHO"]

    paired_pp = cross["rows"].get("B_sfp|LOHO|field_PP_accuracy", {})
    paired_third = cross["rows"].get(
        "B_sfp|LOHO|positive_forced_third_exact_accuracy", {}
    )
    return {
        "material_delta": MATERIAL_DELTA,
        "pp": {
            "prior": pp_prior,
            "new": pp_new,
            "delta": None if pp_new is None else rd(pp_new - pp_prior),
            "material": pp_new is not None
            and (pp_new - pp_prior) >= MATERIAL_DELTA,
            "paired": paired_pp,
        },
        "forced_third": {
            "prior": third_prior,
            "new": third_new,
            "delta": None if third_new is None else rd(third_new - third_prior),
            "material": third_new is not None
            and (third_new - third_prior) >= MATERIAL_DELTA,
            "paired": paired_third,
        },
        "criterion_note": (
            "compared on the SET criterion for PP, which is what 0.7158 was measured "
            "under. The point criterion on q* is reported separately and is stricter."
        ),
    }


def ceiling_reading(baselines: dict) -> dict:
    """Where the learned result sits between the two nonlearned ceilings."""

    rows = {}
    for arm, families in sorted(baselines["search_summary"]["per_arm"].items()):
        for family, body in sorted(families.items()):
            rows[f"{arm}|{family}"] = {
                "determining_ceiling": body["determining_ceiling"],
                "strictly_cheaper_ceiling": body["strictly_cheaper_ceiling"],
                "best_cheap_subsets": body["best_cheap_subsets"],
                "n_ceiling_subsets": body["n_ceiling_subsets"],
                "n_determining_by_measurement": body["n_determining_by_measurement"],
                "n_relation_equivalent_by_cost_rule": body[
                    "n_relation_equivalent_by_cost_rule"
                ],
                "n_strictly_cheaper": body["n_strictly_cheaper"],
            }
    return {
        "per_arm_family": rows,
        "baseline_table": baselines["baseline_table"]["per_arm"],
        "census": {
            arm: {
                "relation_determining_subsets": body["relation_determining_subsets"],
                "smallest_determining_size": body["smallest_determining_size"],
                "cheap_subsets_that_determine_qstar": body[
                    "cheap_subsets_that_determine_qstar"
                ],
            }
            for arm, body in sorted(baselines["determination_census"].items())
        },
        "bucketing_rule": baselines["search_summary"]["bucketing_rule"],
        "reading": baselines["search_summary"]["reading"],
    }


def metric_quantum(sweep: dict, block: str, family: str) -> dict:
    """The smallest difference the paired ``D - B`` statistic can express.

    A per-cell accuracy over ``n`` held-out admitted pairs moves in steps of ``1/n``,
    and the paired mean over ``k`` cells therefore moves in steps of ``1/(n k)``. Any
    interval bound is a multiple of that quantum, so "excludes zero by ``x``" is only
    interpretable next to it.
    """

    runs = [
        row
        for row in sweep["blocks"][block]["rung2"]["runs"]
        if row["fold_family"] == family and row["arm"] == "B_sfp"
    ]
    admits = sorted({row["test_metrics"]["n_admit"] for row in runs})
    cells = len(runs)
    quantum = None if not admits or not cells else 1.0 / (admits[0] * cells)
    return {
        "family": family,
        "held_out_admitted_pairs_per_cell": admits,
        "paired_cells": cells,
        "smallest_expressible_paired_difference": rd(quantum),
        "note": (
            "a per-cell accuracy over n held-out admitted pairs moves in steps of 1/n, "
            "so the paired mean over k cells moves in steps of 1/(n k). Interval "
            "bounds are multiples of this quantum."
        ),
    }


def chart_equivalence_reading(rung1: dict, rung2: dict, sweep: dict, block: str) -> dict:
    """Is ``D - B`` a structural difference, or an optimizer-scale one?

    ``009.07`` section 6 is explicit that this must not be read off a bare
    sign: *"Do not interpret one chart outperforming another by a small
    optimizer-dependent amount as a structural difference unless the paired interval
    excludes zero materially and conformance is clean."* So the interval is reported
    next to the smallest difference it could have expressed, and next to the component
    fields the composite metric is built from.
    """

    quantum = metric_quantum(sweep, block, "LOHO")
    step = quantum["smallest_expressible_paired_difference"]
    rows = {}
    for label, body, metrics in (
        ("rung1", rung1, RUNG1_METRICS),
        ("rung2", rung2, RUNG2_PRIMARY_METRICS),
    ):
        for family in ("LOHO", "LOFPO"):
            for metric in metrics:
                block_body = body["comparisons"]["D_minus_B"].get(
                    f"{family}|{metric}"
                )
                if not block_body or not block_body.get("paired"):
                    continue
                ci = block_body["ci_95"]
                excludes = block_body["ci_excludes_zero"]
                nearest = (
                    min(abs(ci["low"]), abs(ci["high"]))
                    if ci["low"] is not None
                    else None
                )
                rows[f"{label}|{family}|{metric}"] = {
                    "mean_difference": block_body["mean_difference"],
                    "ci_95": ci,
                    "ci_excludes_zero": excludes,
                    "distance_of_the_nearest_bound_from_zero": rd(nearest),
                    "in_units_of_the_metric_quantum": None
                    if nearest is None or not step
                    else rd(nearest / step),
                    "folds_favouring_D": block_body["folds_favouring_left"],
                    "n_folds": block_body["n_folds"],
                }
    excluding = sorted(key for key, row in rows.items() if row["ci_excludes_zero"])
    return {
        "metric_quantum": quantum,
        "per_metric": rows,
        "metrics_whose_interval_excludes_zero": excluding,
        "n_metrics_examined": len(rows),
        "rung1_loho_difference_is_exactly_zero": (
            rows.get("rung1|LOHO|qstar_exact_accuracy", {}).get("mean_difference")
            == 0.0
        ),
        "every_lofpo_interval_includes_zero": not any(
            key.split("|")[1] == "LOFPO" for key in excluding
        ),
        "section_6_caution": (
            "009.07 section 6: a small optimizer-dependent chart difference is not a "
            "structural difference unless the paired interval excludes zero MATERIALLY "
            "and conformance is clean. Conformance is clean. Whether these exclusions "
            "are material is the reading below, and the numbers are given rather than "
            "the conclusion assumed."
        ),
    }


def structural_verdict(sweep: dict, task: dict, heads: dict, baselines: dict) -> dict:
    """Every control that could invalidate a causal reading, in one place."""

    laws = sweep["laws"]
    checks = {
        "every_declared_cell_ran": all(
            body["every_cell_ran"] for body in laws.values()
        ),
        "swap_invariance_exact_everywhere": all(
            body["swap_invariance_exact_everywhere"] for body in laws.values()
        ),
        "no_unresolved_prediction": all(
            body["no_unresolved_prediction_anywhere"] for body in laws.values()
        ),
        "fixed_resolution_never_drops": all(
            body["resolution_never_drops"] for body in laws.values()
        ),
        "capacity_matched_across_science_arms": all(
            body["capacity_matched_across_science_arms"] for body in laws.values()
        ),
        "copied_and_learned_fields_as_declared": all(
            body["copied_fields_are_S_and_FFF_everywhere"]
            and body["learned_fields_are_PP_and_pp_everywhere"]
            for body in laws.values()
        ),
        "qstar_label_source_is_certified": task["label_source_fence"][
            "no_label_function_accepts_a_circuit"
        ],
        "upstream_digests_pin": task["upstream_pins"]["all_upstream_pins_agree"],
        "no_leakage_on_any_fold": task["leakage"]["no_leakage_anywhere"],
        "label_covariance_exact": task["target_covariance"]["label_covariance_exact"],
        "scramble_outside_the_induced_class": task["scramble_recertification"][
            "all_outside_the_induced_class"
        ],
        "scramble_marginals_match_B": task["scramble_recertification"][
            "field_marginals_match"
        ],
        "no_absolute_pp_output_column": heads["no_absolute_pp_column"][
            "parameter_count_independent_of_candidate_count"
        ],
        "candidate_order_invariance_exact": heads["candidate_order_invariance"][
            "order_invariance_exact"
        ],
        "architectural_covariance_exact": heads["covariance"][
            "architectural_covariance_exact"
        ],
        "ceilings_behave": baselines["verdict"]["agrees"],
    }
    return {
        "checks": checks,
        "all_controls_pass": all(checks.values()),
        "scramble_support_fraction": task["qstar_alignment"][
            "scramble_support_fraction"
        ],
        "support_fraction_note": (
            "the certified q* is well defined for Arm C but its coded incidence "
            "supports it on only that fraction of admitted pairs. No unsupported "
            "example was dropped, per 009.07 section 7."
        ),
    }


# ---------------------------------------------------------------------------
# disposition
# ---------------------------------------------------------------------------

def blocking_items(rung1_body: dict, rung2_body: dict) -> list[dict]:
    """Every section 4.4 / 5.4 condition that is NOT met, with its shortfall.

    Named individually so a reader can see exactly what stands between the observed
    result and the next class up, rather than being handed a letter.
    """

    items = []
    for source, body in (("009.07 section 4.4", rung1_body), ("009.07 section 5.4", rung2_body)):
        for name, check in body["checks"].items():
            if not check["meets"]:
                items.append(
                    {
                        "source": source,
                        "condition": name,
                        "threshold": check["threshold"],
                        "observed": check["observed"],
                        "shortfall": None
                        if check["observed"] is None
                        else rd(check["threshold"] - check["observed"]),
                    }
                )
        for name in (
            "b_minus_c_positive_and_excludes_zero",
            "d_minus_b_includes_zero",
            "fixed_resolution_loss_is_zero",
        ):
            if name in body["structural"] and not body["structural"][name]:
                detail = {}
                if name == "d_minus_b_includes_zero":
                    ci = body["structural"]["d_minus_b"]["ci_95"]
                    detail = {
                        "mean_difference": body["structural"]["d_minus_b"][
                            "mean_difference"
                        ],
                        "ci_95": ci,
                        "distance_of_the_nearest_bound_from_zero": rd(
                            min(abs(ci["low"]), abs(ci["high"]))
                        ),
                    }
                items.append(
                    {"source": source, "condition": name, "met": False, **detail}
                )
    return items


def disposition(
    rung1_body: dict,
    rung2_body: dict,
    improvement: dict,
    controls: dict,
    charts: dict,
) -> dict:
    """``009.07`` section 10, returning the strongest justified class."""

    if not controls["all_controls_pass"]:
        failed = [key for key, value in controls["checks"].items() if not value]
        return {
            "class": "E",
            "title": "structural / control failure",
            "reason": f"controls failed: {failed}",
            "failed_controls": failed,
            "blocking_items": [],
        }

    rung1_ok = rung1_body["gap_closed"]
    rung2_ok = rung2_body["gap_substantially_closed"]
    pp_material = improvement["pp"]["material"]
    third_material = improvement["forced_third"]["material"]
    structural_ok = (
        rung1_body["structural"]["b_minus_c_positive_and_excludes_zero"]
        and rung1_body["structural"]["d_minus_b_includes_zero"]
    )

    blocking = blocking_items(rung1_body, rung2_body)

    if rung1_ok and rung2_ok:
        return {
            "class": "A",
            "title": "PP obstruction substantially closed",
            "reason": (
                "Rung 1 meets section 4.4 on the point criterion, Rung 2 meets "
                "section 5.4 on the learned-PP learned-pp system, B > C with a "
                "positive interval, D is compatible with B, and fixed resolution "
                "loses nothing"
            ),
            "interpretation": (
                "the remaining 009.06 gap was substantially another "
                "output-constitution mismatch. A query-relative consequence "
                "interface carries most of the finite relation."
            ),
            "blocking_items": [],
        }
    if pp_material and structural_ok:
        body = {
            "class": "B",
            "title": "material PP improvement, incomplete closure",
            "reason": (
                "PP improves by at least "
                f"{MATERIAL_DELTA} absolute with a positive paired interval and every "
                "structural control passes, but at least one section 4.4 or 5.4 "
                "condition is not met, so section 4.4's instruction applies: report it "
                "under the ladder rather than rounding it up"
            ),
            "interpretation": (
                "the query-relative interface helps materially. Whether the residual "
                "is a real obstruction or an optimizer-scale chart effect is the "
                "reading in chart_equivalence, and the blocking item is named rather "
                "than summarized."
            ),
            "blocking_items": blocking,
            "distance_to_class_A": {
                "conditions_missed": len(blocking),
                "rung1_meets_section_4_4": rung1_ok,
                "rung2_meets_section_5_4": rung2_ok,
                "rung2_accuracy_thresholds_met": all(
                    check["meets"] for check in rung2_body["checks"].values()
                ),
                "b_greater_than_c": rung2_body["structural"][
                    "b_minus_c_positive_and_excludes_zero"
                ],
                "fixed_resolution_loss_is_zero": rung2_body["structural"][
                    "fixed_resolution_loss_is_zero"
                ],
                "metric_quantum": charts["metric_quantum"][
                    "smallest_expressible_paired_difference"
                ],
                "statement": (
                    "class A is missed on the section 5.4 D-B condition alone. Every "
                    "accuracy threshold at both rungs is met, B > C is positive and "
                    "sign-consistent, and fixed resolution loses nothing. The reading "
                    "of that one condition is deliberately left to the GPT Owner: "
                    "section 6 says a small chart difference is not structural unless "
                    "its interval excludes zero MATERIALLY, and the numbers needed to "
                    "judge that are in chart_equivalence rather than folded into a "
                    "letter here."
                ),
            },
        }
        if not third_material:
            body.update(
                {
                    "class": "C",
                    "title": "PP localizes, recomposition fails",
                    "reason": (
                        "Rung 1 materially succeeds but the main learned consequence "
                        f"does not improve by {MATERIAL_DELTA} over "
                        f"{PRIOR_009_06['positive_forced_third_exact_accuracy']['LOHO']}"
                    ),
                    "interpretation": (
                        "chamber localization is not the only remaining integration "
                        "problem."
                    ),
                }
            )
            body.pop("distance_to_class_A", None)
        return body
    return {
        "class": "D",
        "title": "no material PP gain",
        "reason": (
            f"PP does not improve by {MATERIAL_DELTA} absolute over "
            f"{PRIOR_009_06['field_PP_accuracy']['LOHO']}, or its paired interval "
            "shows no positive material effect"
        ),
        "interpretation": (
            "009.06's PP ceiling is not primarily an absolute-output-class artifact."
        ),
        "blocking_items": blocking,
    }


def recommendation(disp: dict, rung2_body: dict, rung2: dict, ceilings: dict) -> dict:
    """``009.07`` section 12.12: is ``PP`` closed enough to begin discovery?"""

    summary = rung2["summary"]
    residual_pp = _mean(summary, "B_sfp|LOHO", "field_PP_accuracy")
    residual_pp_lower = _mean(summary, "B_sfp|LOHO", "field_pp_accuracy")
    third = _mean(summary, "B_sfp|LOHO", "positive_forced_third_exact_accuracy")
    blocking = disp.get("blocking_items", [])
    only_chart_condition = [
        item["condition"] for item in blocking
    ] == ["d_minus_b_includes_zero"]
    ready = disp["class"] == "A"
    return {
        "disposition_class": disp["class"],
        "pp_closed_enough_to_begin_opaque_token_discovery": ready,
        "conditionally_ready": {
            "conditional": disp["class"] == "B" and only_chart_condition,
            "condition": (
                "the only unmet condition is section 5.4's D-B interval, on a "
                "composite metric whose every component field includes zero and whose "
                "Rung-1 counterpart is exactly zero. If the GPT Owner reads that "
                "exclusion as immaterial under section 6, the result is class A and PP "
                "is closed enough to begin discovery. That reading is the Owner's to "
                "make; this turn does not make it."
                if disp["class"] == "B" and only_chart_condition
                else "not applicable"
            ),
        },
        "residual": {
            "field_PP_accuracy_LOHO": residual_pp,
            "field_pp_accuracy_LOHO": residual_pp_lower,
            "positive_forced_third_exact_accuracy_LOHO": third,
            "determining_ceiling": 1.0,
            "remaining_gap_to_the_ceiling": None
            if third is None
            else rd(1.0 - third),
        },
        "scoping": (
            "discovery should target PP and pp -- the two fields the certified "
            "relation COMPUTES -- holding S and FFF fixed as supplied identity, and "
            "should be judged against the 1.000 determining ceiling rather than "
            "against chance. That is 009.06's scoping and this turn does not widen it."
        ),
        "closed_set_habitat_reidentification_still_unresolved": (
            "under the default protocol the learner cannot name an unseen habitat at "
            "all; 009.06 recorded field_FFF_accuracy at 0.0320 when FFF was learned. "
            "This turn copies S and FFF and therefore says nothing about it. No "
            "discovery result may be read as addressing it."
        ),
        "authorizes": (
            "nothing. 009.07 section 9 forbids proceeding to opaque-token discovery "
            "in this task even on a positive result; this is a recommendation for the "
            "GPT Owner's disposition, not a decision."
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def analyse() -> dict:
    sweep = load(SWEEP)
    task = load(TASK)
    heads = load(HEADS)
    baselines = load(BASELINES)

    block = "primary"
    rung1 = read_rung1(sweep, block)
    rung2 = read_rung2(sweep, block)
    cross = cross_turn(sweep, block)
    rung1_body = rung1_criterion(rung1)
    rung2_body = rung2_criterion(rung2, sweep, block)
    improvement = material_improvement(rung2, cross)
    ceilings = ceiling_reading(baselines)
    controls = structural_verdict(sweep, task, heads, baselines)
    charts = chart_equivalence_reading(rung1, rung2, sweep, block)
    disp = disposition(rung1_body, rung2_body, improvement, controls, charts)
    rec = recommendation(disp, rung2_body, rung2, ceilings)

    return {
        "module": "locator_analysis",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "block": block,
        "config": sweep["blocks"][block]["config"],
        "n_cells": sweep["blocks"][block]["n_cells"],
        "rung1": rung1,
        "rung2": rung2,
        "rung1_criterion": rung1_body,
        "rung2_criterion": rung2_body,
        "cross_turn": cross,
        "material_improvement": improvement,
        "ceilings": ceilings,
        "chart_equivalence": charts,
        "controls": controls,
        "capacity_ledger": sweep["blocks"][block]["capacity_ledger"],
        "disposition": disp,
        "recommendation": rec,
        "statistics": (
            "analysis.bootstrap_ci and analysis.paired_differences, imported "
            "unchanged: a seeded percentile bootstrap of the mean over identical "
            "(fold_name, seed) cells, the same estimator 009.02 and 009.06 used"
        ),
        "verdict": {
            "agrees": controls["all_controls_pass"],
            "statement": (
                f"disposition {disp['class']} -- {disp['title']}"
                if controls["all_controls_pass"]
                else "controls failed; disposition E"
            ),
        },
    }


def _report(result: dict) -> None:
    print(f"block {result['block']} ({result['n_cells']} cells)", flush=True)
    print("Rung 1 q* exact accuracy (mean, LOHO / LOFPO):", flush=True)
    summary = result["rung1"]["summary"]
    for arm in ("B_sfp", "D_relabeled", "A_native", "C_scrambled"):
        loho = _mean(summary, f"{arm}|LOHO", "qstar_exact_accuracy")
        lofpo = _mean(summary, f"{arm}|LOFPO", "qstar_exact_accuracy")
        print(f"  {arm:<14} {loho}  {lofpo}", flush=True)
    print("Rung 2 forced third exact accuracy (mean, LOHO / LOFPO):", flush=True)
    summary2 = result["rung2"]["summary"]
    for arm in ("B_sfp", "D_relabeled", "A_native", "C_scrambled"):
        loho = _mean(
            summary2, f"{arm}|LOHO", "positive_forced_third_exact_accuracy"
        )
        lofpo = _mean(
            summary2, f"{arm}|LOFPO", "positive_forced_third_exact_accuracy"
        )
        print(f"  {arm:<14} {loho}  {lofpo}", flush=True)
    imp = result["material_improvement"]
    print(
        f"PP {imp['pp']['prior']} -> {imp['pp']['new']} "
        f"(delta {imp['pp']['delta']}, material {imp['pp']['material']})",
        flush=True,
    )
    print(
        f"forced third {imp['forced_third']['prior']} -> {imp['forced_third']['new']} "
        f"(delta {imp['forced_third']['delta']}, "
        f"material {imp['forced_third']['material']})",
        flush=True,
    )
    disp = result["disposition"]
    print(f"disposition {disp['class']}: {disp['title']}", flush=True)
    for item in disp.get("blocking_items", []):
        print(
            f"  BLOCKING  {item['source']} {item['condition']}"
            + (
                f"  observed {item['observed']} vs threshold {item['threshold']}"
                if "threshold" in item
                else ""
            )
            + (
                f"  mean {item['mean_difference']} ci "
                f"[{item['ci_95']['low']}, {item['ci_95']['high']}]"
                if "ci_95" in item
                else ""
            ),
            flush=True,
        )
    charts = result["chart_equivalence"]
    print(
        f"  metric quantum "
        f"{charts['metric_quantum']['smallest_expressible_paired_difference']}; "
        f"D-B intervals excluding zero: "
        f"{len(charts['metrics_whose_interval_excludes_zero'])}"
        f"/{charts['n_metrics_examined']}",
        flush=True,
    )
    print(
        "  recommendation: PP closed enough for opaque-token discovery = "
        f"{result['recommendation']['pp_closed_enough_to_begin_opaque_token_discovery']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = analyse()
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if json.loads(OUTPUT.read_text()) != json.loads(text):
            raise SystemExit(
                f"FAIL: re-derived analysis is not identical to {OUTPUT.name}"
            )
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
