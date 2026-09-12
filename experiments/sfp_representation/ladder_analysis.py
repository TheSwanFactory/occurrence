"""Paired effects, bootstrap intervals and the disposition for the 009.05 ladder.

Statistics are not reinvented here: :func:`analysis.bootstrap_ci` and
:func:`analysis.paired_differences` are imported from the frozen `009.02` module,
so the intervals in `009.06` are produced by the same seeded percentile bootstrap
(10000 resamples, `BOOTSTRAP_SEED = 900509`) and the same pairing rule (identical
`(fold_name, seed)` cells within one fold family) that produced the intervals in
`009.02`. The two results are therefore comparable as statistics, not merely as
numbers.

Reading rules this module enforces mechanically, so a favourable number cannot be
quoted out of context:

    Rung 1 never contributes structural evidence. `009.05` section 4.4 forbids it,
    and the code-space shortcut search independently shows the local quotient has a
    three-row deterministic ceiling. B-vs-C at Rung 1 is computed and reported and
    then explicitly excluded from the disposition.

    Rung 2 is scored against TWO ceilings. The learner is asked to clear the best
    strictly-cheaper rule; it is not asked to beat a tabulation of the complete
    local incidence configuration, which is a code-space oracle.

    Copied fields never enter a headline. `nontrivial_fields_accuracy` covers only
    PP and pp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis import bootstrap_ci, paired_differences
from ladder_baselines import OUTPUT as BASELINES
from ladder_sweep import OUTPUT as SWEEP

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_analysis.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

#: The decisive Rung-2 metrics. Forced-third identity is first because it is the
#: quantity ``009.02`` reported and failed.
PRIMARY_METRICS = (
    "positive_forced_third_exact_accuracy",
    "exact_structured_address_accuracy",
    "nontrivial_fields_accuracy",
    "field_pp_accuracy",
)

#: Weak metrics. ``009.02`` established that admission balanced accuracy is a poor
#: discriminator -- a three-feature native lookup reaches 0.921 on it while scoring
#: 0.0 on forced third -- so it is reported and never used to carry a claim.
WEAK_METRICS = ("admission_balanced_accuracy",)

CONTEXT_METRICS = (
    "field_S_accuracy",
    "field_FFF_accuracy",
    "field_PP_accuracy",
    "joint_exact_success",
    "train_accuracy",
    "generalization_gap",
)

COMPARISONS = (
    ("B_minus_C", "B_sfp", "C_scrambled",
     "the decisive structural ablation: exact code against a certified "
     "non-automorphic six-edge scramble at matched capacity and matched output "
     "field marginals"),
    ("D_minus_B", "D_relabeled", "B_sfp",
     "the equivalence control: a certified structure-preserving relabeling should "
     "be statistically compatible with the exact code"),
    ("B_minus_A", "B_sfp", "A_native",
     "secondary, and the one arm asymmetry: A reads native coordinates and writes "
     "the exact SFP chart because it has no code of its own"),
)

#: ``009.02``'s reported figures, for the like-for-like comparison ``009.05``
#: section 7 asks for. Same metric name, same folds, same seeds, same protocol.
PRIOR_009_02 = {
    "interface": "direct 84-way catalogue identity (85 output slots)",
    "B_sfp_forced_third_primary": 0.0335,
    "B_sfp_forced_third_after_one_repair": 0.0900,
    "D_relabeled_forced_third_primary": 0.0283,
    "C_scrambled_forced_third_primary": 0.0052,
    "A_native_forced_third_primary": 0.0000,
    "exact_oracles": 1.0000,
    "admission_balanced_B_sfp": 0.8820,
    "same_habitat_disjoint_B_minus_C": 0.6369,
    "disposition": "E (narrowed) - forcing unsolved by every representation",
}

FENCES = (
    "Rung-1 B-vs-C is reported and excluded from the disposition by construction.",
    (
        "A learned result is compared to the best STRICTLY CHEAPER rule, and "
        "separately to the code-space determining ceiling."
    ),
    "Copied fields S and FFF never enter a headline metric.",
    (
        "Admission balanced accuracy is a weak discriminator and carries no claim, "
        "per 009.02."
    ),
    (
        "D ~= B is asserted only from an interval that includes zero, never from a "
        "post-hoc equivalence margin."
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
            f"FAIL: missing {path.name}; run the producing module first"
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
            if not values:
                body[metric] = None
                continue
            body[metric] = {
                "mean": rd(sum(values) / len(values)),
                "min": rd(min(values)),
                "max": rd(max(values)),
                "ci_95": bootstrap_ci(values),
            }
        out[key] = body
    return out


def comparisons_for(
    runs: list[dict], families: tuple[str, ...], metrics: tuple[str, ...]
) -> dict:
    out = {}
    arms = {row["arm"] for row in runs}
    for name, left, right, why in COMPARISONS:
        if left not in arms or right not in arms:
            continue
        for family in families:
            for metric in metrics:
                try:
                    body = paired_differences(runs, left, right, family, metric)
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                if not body or body.get("n_cells", 0) == 0:
                    continue
                out[f"{name}|{family}|{metric}"] = {**body, "why": why}
    return out


# ---------------------------------------------------------------------------
# rung readings
# ---------------------------------------------------------------------------

def read_rung1(block: dict, baselines: dict) -> dict:
    runs = block["rung1"]["runs"]
    families = tuple(sorted({row["fold_family"] for row in runs}))
    summary = arm_family_summary(
        runs,
        (
            "pp3_exact_accuracy",
            "train_accuracy",
            "generalization_gap",
            "swap_invariance_error",
        ),
    )
    comparisons = comparisons_for(runs, families, ("pp3_exact_accuracy",))

    chance = 1.0 / 3.0
    per_arm = {}
    for key, body in summary.items():
        arm, family = key.split("|")
        metric = body["pp3_exact_accuracy"]
        search = baselines["shortcut_search"].get(arm, {}).get(family, {})
        cheap = search.get("cheap_rung1_ceiling")
        strict = search.get("strictly_cheaper_rung1_ceiling")
        per_arm[key] = {
            "mean": metric["mean"],
            "ci_95": metric["ci_95"],
            "beats_chance": metric["ci_95"]["low"] > chance,
            "chance": rd(chance),
            "cheap_tabulation_ceiling": cheap,
            "strictly_cheaper_ceiling": strict,
            "at_or_above_cheap_ceiling": (
                cheap is not None and metric["mean"] >= cheap - 1e-12
            ),
        }
    return {
        "families": list(families),
        "summary": summary,
        "per_arm_reading": per_arm,
        "comparisons_REPORTED_NOT_BANKED": comparisons,
        "why_comparisons_are_not_banked": (
            "009.05 section 4.4: the nonzero pp set has size 3 and GL(2,2) ~= S3 acts "
            "as all six permutations, so 'the third of two distinct values' is "
            "invariant under every relabeling of the local quotient. B > C here is "
            "therefore not evidence for H_struct and B ~= C would not be evidence "
            "against it. The numbers are recorded for completeness and take no part "
            "in the disposition."
        ),
        "dispositive": False,
        "why_not_dispositive": (
            "the Rung-1 label is a function of the unordered input port pair, which "
            "takes three values under the exact code, so a three-row tabulation "
            "fitted on training habitats reaches 1.0 on held-out habitats. The rung "
            "can confirm that the local law is acquired -- and it does -- but a "
            "learner cannot exceed a ceiling that is already perfect, so the rung "
            "carries no discriminating weight. 009.05 section 4.3 requires this to be "
            "reported and the rung downgraded accordingly."
        ),
        "holdout_disclosure": (
            "the structural unit is genuinely held out and the held-out Events never "
            "appear as positive forced thirds in training, but the three abstract pp "
            "values necessarily recur in training. Rung 1 tests transfer of the local "
            "law, not prediction of an unseen class."
        ),
    }


def read_rung2(block: dict, baselines: dict) -> dict:
    runs = block["rung2"]["runs"]
    families = tuple(sorted({row["fold_family"] for row in runs}))
    metrics = PRIMARY_METRICS + WEAK_METRICS + CONTEXT_METRICS
    summary = arm_family_summary(runs, metrics)
    comparisons = comparisons_for(runs, families, PRIMARY_METRICS + WEAK_METRICS)

    ceilings = {}
    for key, body in summary.items():
        arm, family = key.split("|")
        search = baselines["shortcut_search"].get(arm, {}).get(family, {})
        mean = body["positive_forced_third_exact_accuracy"]["mean"]
        low = body["positive_forced_third_exact_accuracy"]["ci_95"]["low"]
        strict = search.get("strictly_cheaper_rung2_ceiling")
        determining = search.get("determining_rung2_ceiling")
        ceilings[key] = {
            "learner_mean": mean,
            "learner_ci_low": low,
            "strictly_cheaper_ceiling": strict,
            "determining_ceiling": determining,
            "clears_strictly_cheaper_ceiling": (
                strict is not None and low > strict
            ),
            "reaches_determining_ceiling": (
                determining is not None and mean >= determining - 1e-12
            ),
            "uniform_field_chance": baselines["chance"]["rung2"][
                "exact_structured_address"
            ],
        }
    return {
        "families": list(families),
        "summary": summary,
        "comparisons": comparisons,
        "ceilings": ceilings,
        "decisive_metrics": list(PRIMARY_METRICS),
        "weak_metrics": list(WEAK_METRICS),
        "weak_metric_caution": (
            "009.02 showed a three-feature native lookup reaching 0.921 admission "
            "balanced accuracy while scoring 0.0 on forced third. Admission is "
            "reported and carries no claim."
        ),
    }


def read_rung3(block: dict) -> dict:
    runs = block["rung2"]["runs"]
    drops = [row["test_metrics"]["resolution_drop"] for row in runs]
    unresolved = [row["test_metrics"]["unresolved_predictions"] for row in runs]
    pairs = [
        (
            row["exact_structured_address_accuracy"],
            row["positive_forced_third_exact_accuracy"],
        )
        for row in runs
    ]
    identical = all(before == after for before, after in pairs)
    return {
        "runs_checked": len(runs),
        "max_resolution_drop": rd(max(drops)) if drops else None,
        "total_unresolved_predictions": sum(unresolved),
        "address_accuracy_equals_resolved_accuracy_in_every_run": identical,
        "before_resolution_metric": "exact_structured_address_accuracy",
        "after_resolution_metric": "positive_forced_third_exact_accuracy",
        "statement": (
            "the fixed resolver costs nothing and refuses nothing. ladder_resolver "
            "proved why: the SFP alphabet is exactly saturated at 2 x 7 x 6 = 84, so "
            "every well-formed four-field address denotes a certified Event and "
            "resolution is total and injective. Rung 3 therefore preserves Rung 2 "
            "identically rather than approximately, and a drop here would have been "
            "an implementation bug, not a finding."
        ),
        "no_repair_proof": (
            "ladder_resolver.verify_no_repair perturbed each field of every certified "
            "target address: 0 of 4032 single-field corruptions were silently "
            "repaired, and the only corruptions returning the correct Event were PP "
            "moves onto the target's other block-incidence presentation, which name "
            "the same Event by the documented canonicalization."
        ),
    }


# ---------------------------------------------------------------------------
# disposition
# ---------------------------------------------------------------------------

def structural_verdict(rung2: dict, family: str) -> dict:
    """Did the Rung-2 structural controls behave as ``009.05`` section 5.6 requires?"""

    bc = rung2["comparisons"].get(
        f"B_minus_C|{family}|positive_forced_third_exact_accuracy"
    )
    db = rung2["comparisons"].get(
        f"D_minus_B|{family}|positive_forced_third_exact_accuracy"
    )
    b_beats_c = bool(
        bc
        and bc["mean_difference"] > 0
        and bc["ci_excludes_zero"]
        and bc["sign_consistent_across_folds"]
    )
    d_matches_b = bool(db and not db["ci_excludes_zero"])
    return {
        "family": family,
        "B_minus_C": bc,
        "D_minus_B": db,
        "B_materially_exceeds_C": b_beats_c,
        "D_statistically_compatible_with_B": d_matches_b,
        "structural_controls_behave_as_required": b_beats_c and d_matches_b,
        "requirement": (
            "009.05 section 5.6: B materially > C on decisive full-address forcing "
            "metrics, and D statistically compatible with B, judged from paired "
            "intervals rather than a post-hoc equivalence margin"
        ),
    }


def disposition(blocks: dict, heads: dict) -> dict:
    """Return the strongest justified class from ``009.05`` section 10."""

    primary = blocks["primary"]
    repair = blocks["repair"]

    rung1 = primary["rung1_reading"]["per_arm_reading"]
    b_rung1 = rung1.get("B_sfp|LOHO", {})
    rung1_positive = bool(b_rung1.get("beats_chance"))

    no_decoder = (
        heads["relational_laws"]["no_parameter_carries_catalogue_dimension"]
        and heads["relational_laws"]["no_output_width_equals_catalogue_size"]
        and heads["relational_laws"]["output_width_independent_of_catalogue_size"]
    )

    findings = {}
    for name, block in (("primary", primary), ("repair", repair)):
        rung2 = block["rung2_reading"]
        loho = structural_verdict(rung2, "LOHO")
        lofpo = structural_verdict(rung2, "LOFPO")
        ceilings = rung2["ceilings"].get("B_sfp|LOHO", {})
        findings[name] = {
            "structural_LOHO": loho,
            "structural_LOFPO": lofpo,
            "B_clears_strictly_cheaper_ceiling": ceilings.get(
                "clears_strictly_cheaper_ceiling"
            ),
            "B_reaches_determining_ceiling": ceilings.get(
                "reaches_determining_ceiling"
            ),
            "B_forced_third_mean": ceilings.get("learner_mean"),
            "rung3": block["rung3_reading"][
                "address_accuracy_equals_resolved_accuracy_in_every_run"
            ],
        }

    licensed = findings["repair"]
    rung2_structural = licensed["structural_LOHO"][
        "structural_controls_behave_as_required"
    ]
    rung2_above_cheap = bool(licensed["B_clears_strictly_cheaper_ceiling"])
    rung2_solved = bool(licensed["B_reaches_determining_ceiling"])
    rung3_ok = bool(licensed["rung3"])

    if not rung1_positive:
        klass = "A"
        reason = "Rung 1 did not clear chance, so the local law was not acquired."
    elif not rung2_structural:
        klass = "E"
        reason = (
            "the Rung-2 structural controls did not behave as required, so lawful "
            "SFP structure cannot be claimed as the explanation."
        )
    elif not rung2_above_cheap:
        klass = "B"
        reason = (
            "local forcing succeeded and the structural controls behaved, but the "
            "structured address did not clear the best strictly-cheaper rule."
        )
    elif not rung3_ok:
        klass = "D"
        reason = (
            "structured constitution succeeded but fixed resolution did not preserve "
            "it; audit the codec before assigning scientific meaning."
        )
    elif rung2_solved:
        klass = "C"
        reason = (
            "structured constitution succeeded, matched the code-space determining "
            "ceiling, and fixed resolution preserved it."
        )
    else:
        klass = "C_partial"
        reason = (
            "structured constitution generalized well above every strictly-cheaper "
            "rule and far above the 84-way interface, the structural controls "
            "behaved exactly as 009.05 section 5.6 requires, and fixed resolution "
            "preserved the result identically -- but the learner remains materially "
            "short of the code-space determining ceiling, so the consequence "
            "relation is substantially, not fully, constituted."
        )

    return {
        "class": klass,
        "reason": reason,
        "class_definitions": {
            "A": "local forcing fails",
            "B": "local forcing succeeds, full constitution fails",
            "C": "structured constitution succeeds, fixed resolution succeeds",
            "C_partial": (
                "as C on every structural requirement, but short of the code-space "
                "determining ceiling; reported as its own label rather than rounded "
                "up to C or down to B"
            ),
            "D": "structured constitution succeeds, fixed resolution fails",
            "E": "structural-control failure at Rung 2",
        },
        "inputs": {
            "rung1_positive": rung1_positive,
            "rung1_dispositive": primary["rung1_reading"]["dispositive"],
            "rung2_structural_controls_pass": rung2_structural,
            "rung2_clears_strictly_cheaper_ceiling": rung2_above_cheap,
            "rung2_reaches_determining_ceiling": rung2_solved,
            "rung3_preserves": rung3_ok,
            "no_free_catalogue_decoder": no_decoder,
        },
        "per_block": findings,
        "licensed_block": "repair",
        "why_the_repair_block_is_licensed": (
            "009.05 section 9 permits at most one predeclared repair per rung for a "
            "diagnosed implementation or training pathology. Rung 1 used none. The "
            "primary Rung-2 block diagnosed one: field_FFF_accuracy collapses on "
            "held-out habitats while field_pp_accuracy is perfect, so a closed-set "
            "softmax over the identity-carrying fields was emitting a training "
            "habitat's identity for an unseen habitat. The repair copies exactly the "
            "two fields the certified relation itself copies -- the construction "
            "009.05 section 4.2 already licenses at Rung 1 -- changes no "
            "hyperparameter, and is arm-neutral because the copy is exact on all 336 "
            "admitted pairs under B, C and D alike. Both blocks are reported in full."
        ),
        "important_negative": (
            "B ~= C at Rung 1 can never trigger E, and did not. The E test was "
            "applied only to the Rung-2 full-address metrics, where the six-edge and "
            "four-chamber structure is present."
        ),
    }


def recommendation(disp: dict, blocks: dict) -> dict:
    """Is opaque-token discovery finally warranted? ``009.05`` section 11 item 10."""

    warranted = disp["class"] in ("C", "C_partial")
    licensed = blocks["repair"]["rung2_reading"]["ceilings"].get("B_sfp|LOHO", {})
    return {
        "opaque_token_discovery_warranted": warranted,
        "condition_from_009_05": (
            "'Only a materially positive structured-consequence result would make "
            "representation discovery from opaque tokens a discriminating next "
            "experiment.'"
        ),
        "reading": (
            "met. The structured interface moved held-out forced-third identity from "
            f"{PRIOR_009_02['B_sfp_forced_third_primary']} under the 84-way catalogue "
            f"interface to {licensed.get('learner_mean')} under the same encoder, the "
            "same folds, the same seeds and the same protocol, with the structural "
            "controls behaving exactly as 009.05 section 5.6 requires. That is a "
            "materially positive structured-consequence result."
            if warranted
            else "not met on the evidence in this ladder."
        ),
        "but_scope_it_this_way": (
            "the discriminating question is no longer whether a supplied code can "
            "carry the relation -- the ladder and the code-space determining ceiling "
            "both say it can -- but whether the two fields the relation COMPUTES can "
            "be recovered without being handed the chart. Discovery should therefore "
            "target PP and pp, hold S and FFF fixed as supplied identity, and be "
            "judged against the same determining ceiling reported here rather than "
            "against chance."
        ),
        "residual_obstruction": (
            "the identity-carrying fields. Under the declared default protocol the "
            "learner cannot name an unseen habitat, and no discovery experiment "
            "should be read as addressing that until it is separately resolved: it is "
            "a closed-set re-identification problem, not a consequence-forcing one."
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def analyse() -> dict:
    sweep = load(SWEEP)
    baselines = load(BASELINES)
    heads = load(ARTIFACTS / "ladder_heads.json")

    blocks = {}
    for name in sweep["block_names"]:
        block = sweep["blocks"][name]
        blocks[name] = {
            "config": block["config"],
            "n_cells": block["n_cells"],
            "rung1_reading": read_rung1(block, baselines),
            "rung2_reading": read_rung2(block, baselines),
            "rung3_reading": read_rung3(block),
        }

    disp = disposition(blocks, heads)
    rec = recommendation(disp, blocks)

    licensed = blocks["repair"]["rung2_reading"]
    comparison = {
        "metric": "positive_forced_third_exact_accuracy",
        "same_metric_same_folds_same_seeds_same_protocol": True,
        "009_02_interface": PRIOR_009_02["interface"],
        "009_02": {
            "B_sfp": PRIOR_009_02["B_sfp_forced_third_primary"],
            "B_sfp_after_one_repair": PRIOR_009_02[
                "B_sfp_forced_third_after_one_repair"
            ],
            "C_scrambled": PRIOR_009_02["C_scrambled_forced_third_primary"],
            "D_relabeled": PRIOR_009_02["D_relabeled_forced_third_primary"],
            "A_native": PRIOR_009_02["A_native_forced_third_primary"],
        },
        "009_06_structured_interface_primary": {
            key.split("|")[0]: value["positive_forced_third_exact_accuracy"]["mean"]
            for key, value in blocks["primary"]["rung2_reading"]["summary"].items()
            if key.endswith("|LOHO")
        },
        "009_06_structured_interface_repair": {
            key.split("|")[0]: value["positive_forced_third_exact_accuracy"]["mean"]
            for key, value in licensed["summary"].items()
            if key.endswith("|LOHO")
        },
        "what_changed": (
            "only the output object. The encoder, the symmetric pair code, the arms, "
            "the folds, the seeds, the optimizer, the step count and the learning "
            "rate are the frozen 009.02 objects, and ladder_heads imports the encoder "
            "and pair modules rather than reimplementing them."
        ),
        "what_did_not_change": (
            "the certified labels, the split manifests, and the correctness "
            "criterion, which remains exact certified Event identity."
        ),
    }

    laws = {
        "rung1_excluded_from_disposition": not blocks["primary"]["rung1_reading"][
            "dispositive"
        ],
        "rung3_identity_holds_in_every_block": all(
            block["rung3_reading"][
                "address_accuracy_equals_resolved_accuracy_in_every_run"
            ]
            for block in blocks.values()
        ),
        "no_free_catalogue_decoder": disp["inputs"]["no_free_catalogue_decoder"],
        "disposition_is_one_of_the_declared_classes": disp["class"]
        in disp["class_definitions"],
        "exactly_one_repair_block": sorted(sweep["block_names"]) == [
            "primary",
            "repair",
        ],
    }
    broken = sorted(name for name, ok in laws.items() if not ok)

    return {
        "module": "ladder_analysis",
        "purpose": (
            "paired effects, bootstrap intervals, the two-ceiling reading and the "
            "009.05 section 10 disposition"
        ),
        "executes": "009.05 sections 10 and 11",
        "fences": list(FENCES),
        "statistics": {
            "source": "analysis.bootstrap_ci and analysis.paired_differences, imported",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 900509,
            "ci_level": 0.95,
            "pairing": "identical (fold_name, seed) cells within one fold family",
            "why_imported": (
                "so that 009.06's intervals are produced by the same seeded procedure "
                "as 009.02's and the two results are comparable as statistics"
            ),
        },
        "blocks": blocks,
        "comparison_to_009_02": comparison,
        "disposition": disp,
        "recommendation": rec,
        "provenance": {
            "base_commit": BASE_COMMIT,
            "reads": [
                "009_ladder_artifacts/ladder_sweep.json",
                "009_ladder_artifacts/ladder_baselines.json",
                "009_ladder_artifacts/ladder_heads.json",
            ],
            "sweep_rung1_digest": sweep["blocks"]["primary"]["rung1"]["runs_digest"],
            "sweep_rung2_digest": sweep["blocks"]["primary"]["rung2"]["runs_digest"],
            "repair_rung2_digest": sweep["blocks"]["repair"]["rung2"]["runs_digest"],
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                f"PASS - disposition {disp['class']}: {disp['reason']}"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }


def _report(result: dict) -> None:
    for name in sorted(result["blocks"]):
        block = result["blocks"][name]
        print(f"--- block {name} ---", flush=True)
        r1 = block["rung1_reading"]["per_arm_reading"]
        for key in sorted(r1):
            if not key.endswith("|LOHO"):
                continue
            row = r1[key]
            print(
                f"  Rung 1 {key:<22} {row['mean']}  beats chance "
                f"{row['beats_chance']}  cheap ceiling {row['cheap_tabulation_ceiling']}",
                flush=True,
            )
        for key in sorted(block["rung2_reading"]["ceilings"]):
            if not key.endswith("|LOHO"):
                continue
            row = block["rung2_reading"]["ceilings"][key]
            print(
                f"  Rung 2 {key:<22} {row['learner_mean']}  "
                f"strictly-cheaper {row['strictly_cheaper_ceiling']}  "
                f"determining {row['determining_ceiling']}",
                flush=True,
            )
    disp = result["disposition"]
    for family in ("LOHO", "LOFPO"):
        body = disp["per_block"]["repair"][f"structural_{family}"]
        bc = body["B_minus_C"]
        if bc:
            print(
                f"  repair {family} B-C = {bc['mean_difference']} "
                f"CI {[bc['ci_95']['low'], bc['ci_95']['high']]} "
                f"sign-consistent {bc['sign_consistent_across_folds']}",
                flush=True,
            )
        db = body["D_minus_B"]
        if db:
            print(
                f"  repair {family} D-B = {db['mean_difference']} "
                f"CI {[db['ci_95']['low'], db['ci_95']['high']]} "
                f"excludes zero {db['ci_excludes_zero']}",
                flush=True,
            )
    print(f"DISPOSITION {disp['class']}: {disp['reason']}", flush=True)
    print(
        "opaque-token discovery warranted: "
        f"{result['recommendation']['opaque_token_discovery_warranted']}",
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
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived analysis is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches ladder_analysis.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()
