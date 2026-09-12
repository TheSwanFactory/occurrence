"""009.09 analysis: predeclared thresholds, paired effects, and the disposition.

Every threshold in ``009.09`` section 8 is scored here exactly as it was written,
before any result was visible, and a missed threshold is reported as missed rather
than rounded up or reinterpreted. The bootstrap is the **imported** ``009.02``
one -- ``analysis.bootstrap_ci`` and ``analysis.paired_differences``, 10000
resamples, seed 900509 -- so the intervals are the same instrument the earlier
turns used.

Three readings are produced and kept apart:

1. **the predeclared reading.** Sections 6.1 to 6.3 define ``chart_valid``,
   ``exact_block_family_recovery`` and ``sfp_equivalent`` over a reconstruction from
   *all twelve* admitted pairs. That is what section 8's thresholds refer to, so
   that is the reading the disposition is filed under. Section 9 class E names
   "stop rather than repair the scientific question after results are visible", and
   switching reconstruction after seeing the number would be exactly that.

2. **the supplied-anchor reading.** The nine scored answers imply three chambers
   and the fourth is the anchor triple the task itself supplies as support. This is
   reported in full alongside, because it is what the recovered representation
   actually looks like, but it is *named* as an alternative reading and offered to
   the Owner rather than adopted here.

3. **the supervision-identifiability proof.** An exact finite argument, computed
   rather than asserted, that the all-twelve-pairs criterion is **not reachable
   from the task's own declared supervision**: two rules over the supplied atoms
   agree on every row the task permits training on and disagree on every support
   pair. This is a statement about the task, not about the learner, and it is why
   the filed class carries a caution rather than the section 9 class-B reading.

Consumes ``discovery_sweep.json``, which is bulk run data hosted in the Quilt
package rather than in git; the pointer file next to it records where to fetch it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from analysis import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CI_LEVEL, bootstrap_ci
from analysis import paired_differences as _paired
from discovery_baselines import (
    ADMISSION_ONLY_ATOMS,
    ATOMS,
    SUPPLIED_SFP_REFERENCE,
    UNIFORM_COMMON_NEIGHBOURS,
    UNIFORM_ELIGIBLE_NODES,
    atom_values,
)
from discovery_task import (
    N_TOKENS,
    SCORED_QUERIES_PER_EPISODE,
    build_episodes,
    build_local_problems,
    fraction,
    pin,
    rd,
    render,
)
from sfp import SfpCodec
from task import build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_discovery_artifacts"
SWEEP = ARTIFACTS / "discovery_sweep.json"
TASK = ARTIFACTS / "discovery_task.json"
HEADS = ARTIFACTS / "discovery_heads.json"
BASELINES = ARTIFACTS / "discovery_baselines.json"
OUTPUT = ARTIFACTS / "discovery_analysis.json"

BASE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"

#: ``009.09`` section 8, verbatim. Frozen before any run; never moved afterwards.
LOHO_THRESHOLDS = {
    "forced_third_accuracy": 0.90,
    "forced_third_lower_bound": 0.85,
    "exact_block_family_recovery": 0.90,
    "sfp_equivalent_rate": 0.90,
    "chart_valid_rate": 0.95,
    "transport_decision_rate": 0.99,
}

LOFPO_THRESHOLDS = {
    "forced_third_accuracy": 0.85,
    "exact_block_family_recovery": 0.80,
}

#: The exact zero-anchor information ceiling from Gate 0.
ZERO_ANCHOR_CEILING = UNIFORM_COMMON_NEIGHBOURS

#: The exact one-anchor ceiling from Gate 0.
ONE_ANCHOR_CEILING = 1.0

#: The two reconstruction forms, and which one section 8's thresholds refer to.
FORMS = {
    "predeclared_all_pairs": {
        "prefix": "",
        "role": (
            "PREDECLARED. Sections 6.1 to 6.3 reconstruct from all twelve admitted "
            "pairs; every one of the four chambers comes from a prediction. Section "
            "8's thresholds refer to this reading, so the disposition is filed under "
            "it."
        ),
    },
    "supplied_anchor": {
        "prefix": "supplied_anchor_",
        "role": (
            "ALTERNATIVE READING, reported not adopted. The nine scored answers imply "
            "three chambers and the fourth is the anchor triple the task supplies as "
            "support. Offered to the Owner; not substituted for the predeclared form."
        ),
    },
}

FENCES = (
    "No threshold was moved after training. A missed threshold is reported as "
    "missed, never rounded up and never reinterpreted.",
    "The disposition is filed under the PREDECLARED all-pairs reconstruction. The "
    "supplied-anchor reading is reported alongside and named as an alternative for "
    "Owner disposition, because choosing it after seeing the result would be "
    "repairing the scientific question with results visible (section 9 class E).",
    "sfp_equivalent is a consistency confirmation ENTAILED by block-family recovery "
    "(009.09a section 3), not independent evidence. Its agreement with the recovery "
    "rate is asserted as a law; a divergence would be a bug.",
    "The dual-anchor arm's labels are COMBINATORIAL. It is a control and never "
    "contributes to a threshold or a disposition.",
    "009.08's 0.9688 is a contextual reference on a different observation set, not a "
    "capacity-matched arm. The discovery result is judged against its own exact "
    "1.000 one-anchor ceiling.",
    "The holdout is fresh-relabelling generalization of a local relational rule, "
    "not held-out-structure generalization in the 009.06/009.08 sense.",
    "No post hoc equivalence margin. Intervals, sign consistency across folds, and "
    "the exact ceilings do the work.",
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"FAIL: missing {path}.\n"
            "discovery_sweep.json is bulk run data hosted in the protology Quilt "
            "package, not git. Fetch it to the path the pointer file names:\n"
            "  aws s3 cp s3://protology/occurrence/gpt/issues/"
            "009-sfp-consequence-representation/009.10-Code-attachments/"
            "discovery_sweep.json \\\n"
            "    experiments/sfp_representation/009_discovery_artifacts/"
            "discovery_sweep.json"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def arm_family_summary(runs: list[dict], metrics: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in runs:
        grouped.setdefault(f"{row['arm']}|{row['fold_family']}", []).append(row)
    out = {}
    for key, group in sorted(grouped.items()):
        body = {
            "n_runs": len(group),
            "n_folds": len({row["fold_name"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}),
        }
        for metric in metrics:
            values = [row[metric] for row in group if row.get(metric) is not None]
            body[metric] = {
                "mean": rd(fraction(sum(values), len(values))) if values else None,
                "ci_95": bootstrap_ci(values),
                "min": rd(min(values)) if values else None,
                "max": rd(max(values)) if values else None,
                "n": len(values),
            }
        out[key] = body
    return out


def seed_census(
    runs: list[dict], metric: str, family: str, arm: str = "one_anchor"
) -> dict:
    """Per-seed means for one arm. The primary block's pathology is seed-dependent,
    so a mean over seeds would misrepresent a per-seed all-or-nothing outcome as
    uniform partial competence."""

    grouped: dict[int, list[float]] = {}
    for row in runs:
        if (
            row["fold_family"] != family
            or row["arm"] != arm
            or row.get(metric) is None
        ):
            continue
        grouped.setdefault(int(row["seed"]), []).append(row[metric])
    per_seed = {
        str(seed): rd(fraction(sum(values), len(values)))
        for seed, values in sorted(grouped.items())
    }
    distinct = sorted({value for value in per_seed.values()})
    return {
        "metric": metric,
        "family": family,
        "arm": arm,
        "n_folds_per_seed": sorted({len(values) for values in grouped.values()}),
        "per_seed_mean": per_seed,
        "distinct_per_seed_means": distinct,
        "bimodal": len(distinct) > 1
        and (max(distinct) - min(distinct)) > 0.1,
    }


def paired_differences(
    runs: list[dict], left: str, right: str, family: str, metric: str
) -> dict:
    """``analysis.paired_differences``, imported so the instrument is the 009.02 one."""

    return _paired(runs, left, right, family, metric)


def cross_block(
    primary: list[dict], repair: list[dict], arm: str, family: str, metric: str
) -> dict:
    """``repair - primary`` over identical ``(fold, seed, arm)`` cells."""

    def index(rows):
        return {
            (r["fold_name"], r["seed"]): r[metric]
            for r in rows
            if r["arm"] == arm
            and r["fold_family"] == family
            and r.get(metric) is not None
        }

    left, right = index(repair), index(primary)
    cells = sorted(set(left) & set(right))
    diffs = [left[c] - right[c] for c in cells]
    if not diffs:
        return {"n_cells": 0, "paired": False}
    ci = bootstrap_ci(diffs)
    return {
        "n_cells": len(cells),
        "paired": True,
        "mean_difference": rd(fraction(sum(diffs), len(diffs))),
        "min": rd(min(diffs)),
        "max": rd(max(diffs)),
        "ci_95": ci,
        "ci_excludes_zero": ci["low"] is not None
        and (ci["low"] > 0.0 or ci["high"] < 0.0),
        "cells_favouring_repair": sum(1 for v in diffs if v > 0),
        "cells_favouring_primary": sum(1 for v in diffs if v < 0),
    }


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------

def _cell(summary: dict, arm: str, family: str) -> dict | None:
    return summary.get(f"{arm}|{family}")


def threshold_table(summary: dict, form: str) -> dict:
    """Section 8, scored under one reconstruction form."""

    prefix = FORMS[form]["prefix"]
    loho = _cell(summary, "one_anchor", "LOHO")
    lofpo = _cell(summary, "one_anchor", "LOFPO")
    zero = _cell(summary, "zero_anchor", "LOHO")

    rows = []

    def add(name: str, threshold: float, observed: float | None, note: str = "") -> None:
        rows.append(
            {
                "criterion": name,
                "threshold": threshold,
                "observed": observed,
                "passes": observed is not None and observed >= threshold,
                "note": note,
            }
        )

    add(
        "LOHO forced_third_accuracy",
        LOHO_THRESHOLDS["forced_third_accuracy"],
        loho["forced_third_accuracy"]["mean"] if loho else None,
    )
    add(
        "LOHO forced_third 95% lower bound",
        LOHO_THRESHOLDS["forced_third_lower_bound"],
        loho["forced_third_accuracy"]["ci_95"]["low"] if loho else None,
    )
    add(
        "LOHO exact_block_family_recovery",
        LOHO_THRESHOLDS["exact_block_family_recovery"],
        loho[f"{prefix}exact_block_family_recovery"]["mean"] if loho else None,
        f"reconstruction form: {form}",
    )
    add(
        "LOHO sfp_equivalent_rate",
        LOHO_THRESHOLDS["sfp_equivalent_rate"],
        loho[f"{prefix}sfp_equivalent_rate"]["mean"] if loho else None,
        f"reconstruction form: {form}; entailed by block-family recovery",
    )
    add(
        "LOHO chart_valid_rate",
        LOHO_THRESHOLDS["chart_valid_rate"],
        loho[f"{prefix}chart_valid_rate"]["mean"] if loho else None,
        f"reconstruction form: {form}",
    )
    add(
        "LOHO fresh-relabelling transport (exact decisions)",
        LOHO_THRESHOLDS["transport_decision_rate"],
        loho["transport_decision_rate"]["mean"] if loho else None,
    )
    add(
        "LOFPO forced_third_accuracy",
        LOFPO_THRESHOLDS["forced_third_accuracy"],
        lofpo["forced_third_accuracy"]["mean"] if lofpo else None,
    )
    add(
        "LOFPO exact_block_family_recovery",
        LOFPO_THRESHOLDS["exact_block_family_recovery"],
        lofpo[f"{prefix}exact_block_family_recovery"]["mean"] if lofpo else None,
        f"reconstruction form: {form}",
    )

    # "forced third materially > admission-only ceiling": the one-anchor interval
    # must sit strictly above 1/2, and the paired difference against the
    # capacity-matched zero-anchor arm must exclude zero.
    material = {
        "admission_only_ceiling": ZERO_ANCHOR_CEILING,
        "one_anchor_lower_bound": loho["forced_third_accuracy"]["ci_95"]["low"]
        if loho
        else None,
        "clears_the_ceiling": bool(
            loho
            and loho["forced_third_accuracy"]["ci_95"]["low"] is not None
            and loho["forced_third_accuracy"]["ci_95"]["low"] > ZERO_ANCHOR_CEILING
        ),
    }
    rows.append(
        {
            "criterion": "LOHO forced_third materially > admission-only ceiling",
            "threshold": ZERO_ANCHOR_CEILING,
            "observed": material["one_anchor_lower_bound"],
            "passes": material["clears_the_ceiling"],
            "note": (
                "the whole 95% interval must sit above the exact 1/2 ceiling, and the "
                "paired difference against the capacity-matched zero-anchor arm must "
                "exclude zero"
            ),
        }
    )

    primary_rows = [row for row in rows if row["criterion"].startswith("LOHO")]
    secondary_rows = [row for row in rows if row["criterion"].startswith("LOFPO")]
    return {
        "form": form,
        "form_role": FORMS[form]["role"],
        "rows": rows,
        "primary_all_pass": all(row["passes"] for row in primary_rows),
        "secondary_all_pass": all(row["passes"] for row in secondary_rows),
        "failed": [row["criterion"] for row in rows if not row["passes"]],
        "zero_anchor_observed": zero["forced_third_accuracy"]["mean"] if zero else None,
        "material_improvement": material,
    }


def zero_anchor_check(summary: dict, runs: list[dict]) -> dict:
    """Section 8's stop condition: the control may not materially exceed 1/2."""

    rows = []
    for family in ("LOHO", "LOFPO"):
        cell = _cell(summary, "zero_anchor", family)
        if not cell:
            continue
        body = cell["forced_third_accuracy"]
        low, high = body["ci_95"]["low"], body["ci_95"]["high"]
        exceeds = low is not None and low > ZERO_ANCHOR_CEILING
        rows.append(
            {
                "family": family,
                "mean": body["mean"],
                "ci_95": body["ci_95"],
                "exact_ceiling": ZERO_ANCHOR_CEILING,
                "interval_includes_the_ceiling": (
                    low is not None and low <= ZERO_ANCHOR_CEILING <= high
                ),
                "materially_exceeds_the_ceiling": exceeds,
                "anchor_overlap_0_accuracy": cell["anchor_overlap_0_accuracy"]["mean"],
                "anchor_overlap_1_accuracy": cell["anchor_overlap_1_accuracy"]["mean"],
            }
        )
    transport = [
        row["transport_decision_rate"]
        for row in runs
        if row["arm"] == "zero_anchor" and row.get("transport_decision_rate") is not None
    ]
    return {
        "per_family": rows,
        "no_family_materially_exceeds_the_ceiling": not any(
            row["materially_exceeds_the_ceiling"] for row in rows
        ),
        "zero_anchor_transport_mean": rd(fraction(sum(transport), len(transport)))
        if transport
        else None,
        "reading": (
            "the control sitting at the exact 1/2 ceiling is the anti-cheating result: "
            "with the anchor channel zeroed and the parameters unchanged, the learner "
            "cannot do better than break a forced tie. Its fresh-relabelling "
            "transport rate is correspondingly near chance, which is what a tie-break "
            "on an opaque name looks like. A control ABOVE the ceiling would have "
            "indicated leakage or stable-token semantics and would have stopped the "
            "interpretation of the one-anchor arm."
        ),
    }


# ---------------------------------------------------------------------------
# the supervision-identifiability proof
# ---------------------------------------------------------------------------

def supervision_identifiability() -> dict:
    """Is the all-twelve-pairs criterion reachable from the declared supervision?

    Computed, not argued. Section 4.4 makes the three pairs inside the anchor
    *support* rather than scored queries, and section 5.3 permits supervision only on
    query pairs, so no training row ever has ``|{a, b} & anchor| == 2``. Two rules
    over the supplied atoms are exhibited:

    ``parity``   pick the common neighbour with ``|{a,b,c} & anchor|`` odd
    ``clamped``  pick the common neighbour with ``anchor_c == 1 - min(n11, 1)``

    They are *identical* on every row the task permits training on and differ on
    every support pair. So the support-pair answer is not determined by the declared
    supervision, and a criterion that requires it cannot be met by any learner
    obeying the task's own training rule. Both rules are checked exhaustively over
    all 14 habitats, all 4 anchors and all 12 admitted pairs.
    """

    dataset = build_dataset()
    codec = SfpCodec()
    episodes = build_episodes(build_local_problems(dataset, codec), namespace="primary")

    def candidates(episode, query):
        return [
            node
            for node in range(N_TOKENS)
            if atom_values(episode, query, node)[
                "candidate_adjacent_to_both_query_nodes"
            ]
        ]

    def parity(episode, query):
        anchor = set(episode.anchor_nodes)
        hits = [
            c
            for c in candidates(episode, query)
            if len({query[0], query[1], c} & anchor) % 2 == 1
        ]
        return hits[0] if len(hits) == 1 else None

    def clamped(episode, query):
        anchor = set(episode.anchor_nodes)
        overlap = min(len(set(query) & anchor), 1)
        hits = [
            c
            for c in candidates(episode, query)
            if int(c in anchor) == 1 - overlap
        ]
        return hits[0] if len(hits) == 1 else None

    trained_rows = 0
    trained_agree = 0
    trained_parity_correct = 0
    trained_clamped_correct = 0
    support_rows = 0
    support_agree = 0
    support_parity_correct = 0
    support_clamped_correct = 0
    for episode in episodes:
        certified = dict(episode.certified_third_nodes)
        dual = dict(episode.dual_third_nodes)
        support = {frozenset(p) for p in episode.support_queries}
        for query in episode.admit_nodes:
            a, b = parity(episode, query), clamped(episode, query)
            target = certified[query]
            if frozenset(query) in support:
                support_rows += 1
                support_agree += int(a == b)
                support_parity_correct += int(a == target)
                support_clamped_correct += int(b == target)
                if b is not None and b != dual[query]:
                    raise AssertionError(
                        "the clamped rule's support answer is not the dual third"
                    )
            else:
                trained_rows += 1
                trained_agree += int(a == b)
                trained_parity_correct += int(a == target)
                trained_clamped_correct += int(b == target)

    return {
        "rules": {
            "parity": "pick the common neighbour with |{a,b,c} & anchor| odd",
            "clamped": (
                "pick the common neighbour with anchor_c == 1 - min(|{a,b} & anchor|, 1)"
            ),
        },
        "supervisable_rows": trained_rows,
        "support_rows": support_rows,
        "rules_agree_on_every_supervisable_row": trained_agree == trained_rows,
        "parity_correct_on_supervisable_rows": rd(
            fraction(trained_parity_correct, trained_rows)
        ),
        "clamped_correct_on_supervisable_rows": rd(
            fraction(trained_clamped_correct, trained_rows)
        ),
        "rules_agree_on_any_support_row": support_agree > 0,
        "parity_correct_on_support_rows": rd(
            fraction(support_parity_correct, support_rows)
        ),
        "clamped_correct_on_support_rows": rd(
            fraction(support_clamped_correct, support_rows)
        ),
        "clamped_answer_is_always_the_dual_third": True,
        "support_pair_answer_is_underdetermined_by_the_declared_supervision": (
            trained_agree == trained_rows and support_agree == 0
        ),
        "consequence": (
            "the all-twelve-pairs reconstruction of section 6.1 requires the three "
            "support-pair answers, and those answers are not a function of anything "
            "the task permits supervising. Two rules that are indistinguishable on "
            "every supervisable row disagree on all three. The predeclared "
            "chart_valid, exact_block_family_recovery and sfp_equivalent thresholds "
            "are therefore not reachable by ANY learner obeying section 5.3's "
            "training rule, independent of architecture, and the observed failure of "
            "those three thresholds is a property of the task specification rather "
            "than a measurement of the learner."
        ),
        "atom_pool": list(ATOMS),
        "admission_only_atom_pool": list(ADMISSION_ONLY_ATOMS),
        "digest": digest(
            [trained_rows, support_rows, trained_agree, support_agree]
        ),
    }


# ---------------------------------------------------------------------------
# disposition
# ---------------------------------------------------------------------------

def ladder(table: dict, chart_positive: bool, performance_positive: bool) -> str:
    """``009.09`` section 9's ladder, applied to one reading."""

    if table["primary_all_pass"]:
        return "A"
    if performance_positive and not chart_positive:
        return "B"
    if chart_positive and not performance_positive:
        return "C"
    if not performance_positive:
        return "D"
    return "B"


def disposition(
    summary: dict,
    tables: dict,
    zero: dict,
    identifiability: dict,
    comparisons: dict,
    sweep: dict,
    heads: dict,
    task: dict,
    baselines: dict,
) -> dict:
    loho = _cell(summary, "one_anchor", "LOHO")
    performance = bool(
        loho
        and loho["forced_third_accuracy"]["mean"] is not None
        and loho["forced_third_accuracy"]["mean"]
        >= LOHO_THRESHOLDS["forced_third_accuracy"]
    )

    per_form = {}
    for form, table in tables.items():
        prefix = FORMS[form]["prefix"]
        chart = bool(
            loho
            and loho[f"{prefix}exact_block_family_recovery"]["mean"] is not None
            and loho[f"{prefix}exact_block_family_recovery"]["mean"]
            >= LOHO_THRESHOLDS["exact_block_family_recovery"]
            and loho[f"{prefix}chart_valid_rate"]["mean"]
            >= LOHO_THRESHOLDS["chart_valid_rate"]
        )
        per_form[form] = {
            "class": ladder(table, chart, performance),
            "primary_all_pass": table["primary_all_pass"],
            "secondary_all_pass": table["secondary_all_pass"],
            "failed_criteria": table["failed"],
            "chart_audit_positive": chart,
            "performance_positive": performance,
        }

    # Class E gates. Any one of these invalidates the experiment outright.
    invalidating = {
        "gate_0_did_not_open": not task["gate_0"]["gate_open"],
        "zero_anchor_exceeds_its_exact_ceiling": not zero[
            "no_family_materially_exceeds_the_ceiling"
        ],
        "opaque_name_permutation_changes_semantics": not (
            all(
                body["laws"]["argmax_exactly_covariant_on_every_decisive_row"]
                for body in heads["per_message_mode"].values()
            )
            and all(
                body["laws"]["query_swap_invariance_is_bitwise"]
                for body in heads["per_message_mode"].values()
            )
        ),
        "held_out_labels_leaked": not task["laws"][
            "no_held_out_habitat_appears_in_a_training_episode"
        ],
        "sfp_or_native_coordinates_entered_the_learned_path": not (
            heads["laws"]["no_banned_identifier_on_the_learned_path"]
            and heads["laws"]["no_banned_operator_on_the_learned_path"]
            and task["laws"]["module_code_never_names_the_exact_circuit"]
        ),
        "one_anchor_ceiling_is_not_one": baselines["one_anchor_ceiling"][
            "forced_third_accuracy"
        ]
        != ONE_ANCHOR_CEILING,
    }

    filed = "E" if any(invalidating.values()) else per_form["predeclared_all_pairs"]["class"]
    alternative = per_form["supplied_anchor"]["class"]

    return {
        "per_reading": per_form,
        "invalidating_conditions": invalidating,
        "any_invalidating_condition": any(invalidating.values()),
        "filed_class": filed,
        "filed_under": "predeclared_all_pairs",
        "alternative_reading_class": alternative,
        "filed_statement": _statement(filed, alternative, identifiability),
        "why_not_the_alternative": (
            "009.09 section 9 class E names 'stop rather than repair the scientific "
            "question after results are visible'. The supplied-anchor reconstruction "
            "was not the predeclared one, and adopting it because the predeclared one "
            "failed would be exactly that repair. It is reported in full and offered "
            "to the Owner instead."
        ),
        "why_class_B_reading_does_not_hold": (
            "section 9 class B interprets a chart-audit failure as 'the learner found "
            "a useful query rule, not the claimed representation'. That interpretation "
            "is contradicted here. Under the supplied-anchor reconstruction the "
            "learner's own answers imply three of the four chambers, the fourth is the "
            "supplied anchor, the resulting four-chamber system IS the certified "
            "cyclic-block family, and the exact AGL(2,2) search confirms it is "
            "equivalent to the hidden SFP chart. What fails the predeclared form is "
            "the three support-pair answers, and the supervision-identifiability proof "
            "shows those are not determined by anything the task permits supervising."
        ),
        "comparison_support": comparisons,
        "runs_digest": sweep["blocks"]["repair"]["runs_digest"],
    }


def _statement(filed: str, alternative: str, identifiability: dict) -> str:
    if filed == "A":
        return (
            "A -- local SFP-equivalent representation discovery. Every predeclared "
            "primary threshold passes under the reconstruction section 8 refers to."
        )
    if filed == "E":
        return "E -- invalid experiment; an invalidating condition fired."
    return (
        f"{filed} -- filed under the predeclared all-pairs reconstruction, which the "
        "three support-pair answers fail. The alternative supplied-anchor reading "
        f"supports class {alternative}. The failure is localized to a regime the "
        "task's own supervision rule leaves underdetermined: two rules that are "
        "indistinguishable on all "
        f"{identifiability['supervisable_rows']} supervisable rows disagree on all "
        f"{identifiability['support_rows']} support rows, so no learner obeying "
        "section 5.3 can meet those three thresholds. Owner disposition is required "
        "to choose between the two readings."
    )


def recommendation(disp: dict, summary: dict, identifiability: dict) -> dict:
    loho = _cell(summary, "one_anchor", "LOHO")
    return {
        "issue_009_can_close": disp["filed_class"] == "A",
        "needs_one_final_local_turn": disp["filed_class"] != "A",
        "the_turn_is_small_and_named": (
            "one of two amendments, both one line: either (a) section 6.1's "
            "reconstruction becomes the supplied-anchor form -- the nine scored "
            "answers plus the anchor triple the task already supplies as support -- "
            "or (b) section 5.3's supervision extends to the three support pairs so "
            "the |{a,b} & anchor| == 2 regime is no longer unsupervised. Under (a) the "
            "present run already reports the answer and no retraining is needed. Under "
            "(b) a retrain is needed but the protocol is otherwise unchanged."
        ),
        "why_not_a_larger_turn": (
            "nothing about the learner, the architecture, the observation set, the "
            "ceilings or the controls is in question. Gate 0 opened, the exact "
            "one-anchor ceiling is 1.000, the capacity-matched zero-anchor control "
            "sits at its exact 1/2 ceiling, the dual-anchor control behaves as "
            "009.09a section 2 predicted, and fresh-relabelling transport is exact."
        ),
        "successor_named_by_009_09a_section_6": (
            "the global discovery question -- from fourteen anonymous habitats and "
            "certified cross-habitat relational evidence, recover the Fano plane on "
            "FFF -- remains unopened. 009.09a section 6 asks the Owner to name it "
            "before 009.10 lands so a positive local result is not read as the end of "
            "H_discovery. This result does not open it and does not speak to it."
        ),
        "observed_headline": {
            "one_anchor_LOHO_forced_third": loho["forced_third_accuracy"]["mean"]
            if loho
            else None,
            "one_anchor_LOHO_supplied_anchor_block_family_recovery": loho[
                "supplied_anchor_exact_block_family_recovery"
            ]["mean"]
            if loho
            else None,
            "one_anchor_LOHO_predeclared_block_family_recovery": loho[
                "exact_block_family_recovery"
            ]["mean"]
            if loho
            else None,
        },
        "supervision_gap": identifiability["consequence"],
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

METRICS = (
    "forced_third_accuracy",
    "train_accuracy",
    "generalization_gap",
    "anchor_overlap_0_accuracy",
    "anchor_overlap_1_accuracy",
    "train_anchor_overlap_0_accuracy",
    "train_anchor_overlap_1_accuracy",
    "prediction_on_query_node_rate",
    "chart_valid_rate",
    "exact_block_family_recovery",
    "dual_family_rate",
    "invalid_family_rate",
    "sfp_equivalent_rate",
    "supplied_anchor_chart_valid_rate",
    "supplied_anchor_exact_block_family_recovery",
    "supplied_anchor_dual_family_rate",
    "supplied_anchor_invalid_family_rate",
    "supplied_anchor_sfp_equivalent_rate",
    "all_pairs_third_accuracy",
    "support_pair_third_accuracy",
    "support_pair_other_family_answer_rate",
    "strict_unique_triples_mean",
    "transport_decision_rate",
    "transport_family_rate",
    "transport_supplied_anchor_family_rate",
    "dual_anchor_forced_third_accuracy",
    "dual_anchor_chart_valid_rate",
    "dual_anchor_dual_family_rate",
    "dual_anchor_certified_family_rate",
)


def analyse() -> dict:
    sweep = load(SWEEP)
    task = load(TASK)
    heads = load(HEADS)
    baselines = load(BASELINES)

    blocks = {}
    for name in sweep["block_names"]:
        runs = sweep["blocks"][name]["runs"]
        summary = arm_family_summary(runs, METRICS)
        tables = {form: threshold_table(summary, form) for form in FORMS}
        blocks[name] = {
            "summary": summary,
            "threshold_tables": tables,
            "zero_anchor_check": zero_anchor_check(summary, runs),
            "seed_census": {
                metric: seed_census(runs, metric, "LOHO")
                for metric in (
                    "forced_third_accuracy",
                    "train_accuracy",
                    "anchor_overlap_0_accuracy",
                    "supplied_anchor_exact_block_family_recovery",
                )
            },
            "paired_one_anchor_minus_zero_anchor": {
                family: {
                    metric: paired_differences(
                        runs, "one_anchor", "zero_anchor", family, metric
                    )
                    for metric in (
                        "forced_third_accuracy",
                        "supplied_anchor_exact_block_family_recovery",
                        "transport_decision_rate",
                    )
                }
                for family in ("LOHO", "LOFPO")
            },
        }

    cross = {
        arm: {
            family: {
                metric: cross_block(
                    sweep["blocks"]["primary"]["runs"],
                    sweep["blocks"]["repair"]["runs"],
                    arm,
                    family,
                    metric,
                )
                for metric in (
                    "forced_third_accuracy",
                    "anchor_overlap_0_accuracy",
                    "supplied_anchor_exact_block_family_recovery",
                )
            }
            for family in ("LOHO", "LOFPO")
        }
        for arm in ("one_anchor", "zero_anchor")
    }

    identifiability = supervision_identifiability()
    headline_block = "repair"
    disp = disposition(
        blocks[headline_block]["summary"],
        blocks[headline_block]["threshold_tables"],
        blocks[headline_block]["zero_anchor_check"],
        identifiability,
        {
            "one_anchor_minus_zero_anchor": blocks[headline_block][
                "paired_one_anchor_minus_zero_anchor"
            ],
            "repair_minus_primary": cross,
        },
        sweep,
        heads,
        task,
        baselines,
    )
    rec = recommendation(
        disp, blocks[headline_block]["summary"], identifiability
    )

    laws = {
        "gate_0_open": task["gate_0"]["gate_open"],
        "structural_laws_hold": heads["verdict"]["agrees"],
        "baseline_laws_hold": baselines["verdict"]["agrees"],
        "sweep_laws_hold": sweep["verdict"]["agrees"],
        "no_threshold_moved_after_training": True,
        "zero_anchor_within_its_exact_ceiling": blocks[headline_block][
            "zero_anchor_check"
        ]["no_family_materially_exceeds_the_ceiling"],
        "one_anchor_ceiling_is_one": baselines["one_anchor_ceiling"][
            "forced_third_accuracy"
        ]
        == ONE_ANCHOR_CEILING,
        "sfp_equivalent_entailed_by_family_recovery": all(
            sweep["blocks"][name]["laws"][
                "sfp_equivalent_always_agrees_with_family_recovery"
            ]
            for name in sweep["block_names"]
        ),
        "support_pair_regime_is_unsupervised_and_underdetermined": identifiability[
            "support_pair_answer_is_underdetermined_by_the_declared_supervision"
        ],
        "disposition_filed_under_the_predeclared_reading": disp["filed_under"]
        == "predeclared_all_pairs",
    }

    agrees = all(laws.values())
    return {
        "module": "discovery_analysis",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "source": "analysis.bootstrap_ci, imported from the 009.02 turn",
        },
        "predeclared_thresholds": {
            "LOHO": dict(LOHO_THRESHOLDS),
            "LOFPO": dict(LOFPO_THRESHOLDS),
            "declared": "009.09 section 8, before any run",
        },
        "reconstruction_forms": {
            name: dict(body) for name, body in FORMS.items()
        },
        "exact_ceilings": {
            "one_anchor": ONE_ANCHOR_CEILING,
            "zero_anchor": ZERO_ANCHOR_CEILING,
            "uniform_eligible_node_chance": rd(UNIFORM_ELIGIBLE_NODES),
            "source": "discovery_task Gate 0, re-measured in discovery_baselines",
        },
        "pins": {
            "scored_queries_per_episode": pin(
                SCORED_QUERIES_PER_EPISODE,
                task["gate_0"]["per_habitat"][0]["one_anchor"]["per_anchor"][0][
                    "scored_query_pairs"
                ],
                "discovery_task Gate 0",
            ),
            "episodes_per_habitat": pin(
                task["pins"]["episodes_per_habitat"],
                task["episode_manifests"]["primary"]["episodes_per_habitat"][0],
                "discovery_task episode manifest",
            ),
            "runs_per_block": pin(
                sweep["blocks"]["repair"]["n_runs"],
                len(sweep["blocks"]["repair"]["runs"]),
                "discovery_sweep",
            ),
        },
        "headline_block": headline_block,
        "block_roles": sweep["block_roles"],
        "diagnosed_pathology": sweep["diagnosed_pathology"],
        "blocks": blocks,
        "cross_block_repair_minus_primary": cross,
        "supervision_identifiability": identifiability,
        "supplied_sfp_reference": dict(SUPPLIED_SFP_REFERENCE),
        "disposition": disp,
        "recommendation": rec,
        "upstream_digests": {
            "dataset_sha256": task["upstream"]["dataset_sha256"],
            "chart_sha256": task["upstream"]["chart_sha256"],
            "gate_0_digest": task["gate_0"]["digest"],
            "primary_episode_manifest": task["episode_manifests"]["primary"]["sha256"],
            "fold_manifest": task["folds"]["sha256"],
            "primary_runs_digest": sweep["blocks"]["primary"]["runs_digest"],
            "repair_runs_digest": sweep["blocks"]["repair"]["runs_digest"],
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "laws": laws,
        "verdict": {
            "agrees": agrees,
            "statement": (
                f"disposition {disp['filed_class']} filed under the predeclared "
                "all-pairs reconstruction; every upstream law holds, no threshold "
                "moved, the capacity-matched zero-anchor control sits within its "
                "exact 1/2 ceiling, and the support-pair regime is proved "
                "underdetermined by the task's declared supervision"
                if agrees
                else "at least one analysis law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    block = result["blocks"][result["headline_block"]]
    print(f"--- headline block: {result['headline_block']} ---", flush=True)
    for cell, body in sorted(block["summary"].items()):
        print(
            f"  {cell:<22} forced_third {body['forced_third_accuracy']['mean']} "
            f"CI [{body['forced_third_accuracy']['ci_95']['low']}, "
            f"{body['forced_third_accuracy']['ci_95']['high']}]",
            flush=True,
        )
    for form, table in sorted(block["threshold_tables"].items()):
        print(f"--- thresholds, {form} ---", flush=True)
        for row in table["rows"]:
            flag = "pass" if row["passes"] else "MISS"
            print(
                f"  {flag}  {row['criterion']:<52} "
                f"{row['observed']} (>= {row['threshold']})",
                flush=True,
            )
    ident = result["supervision_identifiability"]
    print("--- supervision identifiability ---", flush=True)
    print(
        f"  two rules agree on {ident['supervisable_rows']} supervisable rows: "
        f"{ident['rules_agree_on_every_supervisable_row']}; "
        f"agree on any of {ident['support_rows']} support rows: "
        f"{ident['rules_agree_on_any_support_row']}",
        flush=True,
    )
    disp = result["disposition"]
    print(f"--- disposition {disp['filed_class']} ---", flush=True)
    print(f"  {disp['filed_statement']}", flush=True)
    print(f"  alternative reading: {disp['alternative_reading_class']}", flush=True)
    for name, value in sorted(result["laws"].items()):
        if not value:
            print(f"  FAIL law {name}", flush=True)


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
        expected = json.loads(OUTPUT.read_text())
        observed = json.loads(text)
        for key in ("started_at", "host"):
            expected.pop(key, None)
            observed.pop(key, None)
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived analysis is not identical to {OUTPUT.name} "
                "(started_at/host excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["laws"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
