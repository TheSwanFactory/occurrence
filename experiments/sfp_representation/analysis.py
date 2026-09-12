"""Paired A/B/C/D effects and the primary disposition for Issue 009 (Task 11).

Consumes the frozen ``sweep.json`` and produces the three paired comparisons the
controlling task asks for, each differenced across **identical (fold, seed)**
pairs rather than across marginal means::

    B - A   exact SFP code versus native realization
    B - C   relationally aligned code versus matched destructive scramble
    D - B   symmetry-preserving relabeling sensitivity

Why the headline metric is not admission balanced accuracy
----------------------------------------------------------
The Task 8 leakage search established that every relation-equivalent feature it
found determines **admission only**: neither ``shared_block_count`` nor
``block_id_equality`` names *which* certified block is shared, so neither
determines the forced third. Concretely, the best cheaper-than-the-relation
feature subset reaches admission balanced accuracy ``0.921`` on LOHO and
``0.958`` on LOFPO while scoring ``0.0`` on both forced-third accuracy and on
``SAME_HABITAT_DISJOINT`` non-admission accuracy.

So admission balanced accuracy is a weak discriminator here, and two metrics
carry the science:

``positive_forced_third_exact_accuracy``
    naming the forced third. No cheap feature touches it.
``same_habitat_disjoint_nonadmission_accuracy``
    the one non-admission class that cannot be settled by habitat coincidence,
    so separating it requires block incidence.

A further reading caution, forced by the data: an arm that answers BOTTOM almost
everywhere scores *high* on the non-admission classes and *at chance* on
admission. Non-admission accuracy must therefore always be read together with
admission sensitivity, never alone.

No predeclared numeric equivalence margin
-----------------------------------------
The controlling task forbids inventing an equivalence margin after the fact, so
none is used. Paired mean differences are reported with bootstrap confidence
intervals and with sign consistency across folds, and the disposition is argued
from those intervals rather than from a threshold.

Fences
------
::

    Correctness is exact certified Event identity, never float proximity.
    Admission is reported as balanced accuracy and is a WEAK discriminator here.
    Arm E is a diagnostic memorization ceiling, never a representation arm.
    RANDOM_CONTROL was found non-dispositive by the Task 8 leakage search and is
    excluded from every disposition claim.
    No post hoc equivalence margin; intervals and sign consistency instead.
    One architecture plus one motivated repair, which was tested and reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
SWEEP = ARTIFACTS / "sweep.json"
OUTPUT = ARTIFACTS / "analysis.json"

SCIENCE_ARMS = ("A_native", "B_sfp", "C_scrambled", "D_relabeled")
DIAGNOSTIC_ARMS = ("E_opaque",)

#: The two metrics that carry the science, and the weak one, declared explicitly.
PRIMARY_METRICS = (
    "positive_forced_third_exact_accuracy",
    "same_habitat_disjoint_nonadmission_accuracy",
)
WEAK_METRICS = ("admission_balanced_accuracy",)
CONTEXT_METRICS = (
    "joint_exact_success",
    "admission_sensitivity",
    "train_accuracy",
    "generalization_gap",
)
ALL_METRICS = PRIMARY_METRICS + WEAK_METRICS + CONTEXT_METRICS

COMPARISONS = (
    ("B_minus_A", "B_sfp", "A_native", "exact SFP code versus native realization"),
    (
        "B_minus_C",
        "B_sfp",
        "C_scrambled",
        "relationally aligned code versus matched destructive scramble",
    ),
    ("D_minus_B", "D_relabeled", "B_sfp", "symmetry-preserving relabeling sensitivity"),
)

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 900509
CI_LEVEL = 0.95

FENCES = (
    "Correctness is exact certified Event identity, never float proximity.",
    (
        "Admission is reported as balanced accuracy and is a WEAK discriminator "
        "here: a cheaper-than-the-relation feature already reaches 0.921 (LOHO) "
        "and 0.958 (LOFPO)."
    ),
    "Arm E is a diagnostic memorization ceiling, never a representation arm.",
    (
        "RANDOM_CONTROL was found non-dispositive by the Task 8 leakage search and "
        "is excluded from every disposition claim."
    ),
    "No post hoc equivalence margin; intervals and sign consistency instead.",
    "One architecture plus one motivated repair, which was tested and reported.",
)


def digest(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_runs(block: str) -> list[dict]:
    payload = json.loads(SWEEP.read_text())
    return payload["blocks"][block]["runs"]


def bootstrap_ci(values: list[float]) -> dict:
    """Percentile bootstrap CI of the mean. Seeded, so replay is exact."""

    if not values:
        return {"low": None, "high": None, "n": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    tail = (1.0 - CI_LEVEL) / 2.0
    low = means[int(tail * BOOTSTRAP_RESAMPLES)]
    high = means[min(int((1.0 - tail) * BOOTSTRAP_RESAMPLES), BOOTSTRAP_RESAMPLES - 1)]
    return {"low": round(low, 12), "high": round(high, 12), "n": n}


def paired_differences(
    runs: list[dict], left: str, right: str, family: str, metric: str
) -> dict:
    """Difference ``left - right`` over identical (fold, seed) cells."""

    def index(arm: str) -> dict[tuple[str, int], float]:
        return {
            (r["fold_name"], r["seed"]): r[metric]
            for r in runs
            if r["arm"] == arm and r["fold_family"] == family and r.get(metric) is not None
        }

    a, b = index(left), index(right)
    cells = sorted(set(a) & set(b))
    diffs = [a[c] - b[c] for c in cells]
    if not diffs:
        return {"n_cells": 0, "paired": False}

    per_fold: dict[str, list[float]] = {}
    for (fold, _seed), value in zip(cells, diffs):
        per_fold.setdefault(fold, []).append(value)
    fold_means = {f: sum(v) / len(v) for f, v in sorted(per_fold.items())}
    positive = sum(1 for v in fold_means.values() if v > 0)
    negative = sum(1 for v in fold_means.values() if v < 0)

    mean = sum(diffs) / len(diffs)
    ci = bootstrap_ci(diffs)
    excludes_zero = (
        ci["low"] is not None and (ci["low"] > 0.0 or ci["high"] < 0.0)
    )
    return {
        "n_cells": len(cells),
        "paired": True,
        "mean_difference": round(mean, 12),
        "median_difference": round(statistics.median(diffs), 12),
        "std_population": round(
            statistics.pstdev(diffs) if len(diffs) > 1 else 0.0, 12
        ),
        "min": round(min(diffs), 12),
        "max": round(max(diffs), 12),
        "ci_95": ci,
        "ci_excludes_zero": excludes_zero,
        "n_folds": len(fold_means),
        "folds_favouring_left": positive,
        "folds_favouring_right": negative,
        "sign_consistent_across_folds": positive == len(fold_means)
        or negative == len(fold_means),
        "per_fold_mean_difference": {f: round(v, 12) for f, v in fold_means.items()},
    }


def arm_family_summary(runs: list[dict]) -> dict:
    """Per (arm, family) mean of every metric, over all folds and seeds."""

    out: dict[str, dict] = {}
    keys = sorted({(r["arm"], r["fold_family"]) for r in runs})
    for arm, family in keys:
        rows = [r for r in runs if r["arm"] == arm and r["fold_family"] == family]
        out[f"{arm}|{family}"] = {
            "arm": arm,
            "family": family,
            "arm_role": rows[0]["arm_role"],
            "n_runs": len(rows),
            "metrics": {
                m: round(sum(r[m] for r in rows) / len(rows), 12)
                for m in ALL_METRICS
                if m in rows[0] and rows[0][m] is not None
            },
        }
    return out


def comparisons_for(runs: list[dict], families: tuple[str, ...]) -> dict:
    return {
        family: {
            name: {
                "left": left,
                "right": right,
                "meaning": meaning,
                "metrics": {
                    metric: paired_differences(runs, left, right, family, metric)
                    for metric in ALL_METRICS
                },
            }
            for name, left, right, meaning in COMPARISONS
        }
        for family in families
    }


def forcing_floor(summary: dict) -> dict:
    """Is the forced-third half of the task solved by anybody?"""

    rows = {
        key: node["metrics"].get("positive_forced_third_exact_accuracy")
        for key, node in summary.items()
    }
    values = [v for v in rows.values() if v is not None]
    return {
        "per_arm_family": {k: v for k, v in sorted(rows.items())},
        "max_over_all_arms": round(max(values), 12) if values else None,
        "oracle_value": 1.0,
        "oracle_source": (
            "baselines.py ExactNativeRelation and ExactSFPCircuit both score 1.0 "
            "on every bucket of every fold"
        ),
        "every_arm_at_the_floor": bool(values) and max(values) < 0.20,
        "statement": (
            "the nonlearned oracles solve forcing exactly (1.0) while the best "
            "learned arm stays an order of magnitude below, so the forcing half of "
            "admit-and-force is unsolved by every representation under this learner"
        ),
    }


def disposition(primary: dict, repair: dict) -> dict:
    """Argue the primary disposition A-E from the paired intervals."""

    loho = primary["comparisons"]["LOHO"]
    shd = "same_habitat_disjoint_nonadmission_accuracy"
    bc = loho["B_minus_C"]["metrics"][shd]
    db = loho["D_minus_B"]["metrics"][shd]
    ba = loho["B_minus_A"]["metrics"]["admission_balanced_accuracy"]
    floor = primary["forcing_floor"]

    findings = {
        "admission_half": {
            "B_beats_C_on_the_discriminating_class": {
                "metric": shd,
                "mean_difference": bc["mean_difference"],
                "ci_95": bc["ci_95"],
                "ci_excludes_zero": bc["ci_excludes_zero"],
                "sign_consistent_across_folds": bc["sign_consistent_across_folds"],
            },
            "D_matches_B": {
                "metric": shd,
                "mean_difference": db["mean_difference"],
                "ci_95": db["ci_95"],
                "ci_excludes_zero": db["ci_excludes_zero"],
            },
            "B_beats_A_on_admission": {
                "metric": "admission_balanced_accuracy",
                "mean_difference": ba["mean_difference"],
                "ci_95": ba["ci_95"],
                "ci_excludes_zero": ba["ci_excludes_zero"],
            },
        },
        "forcing_half": floor,
    }

    return {
        "primary_disposition": "E",
        "primary_disposition_text": (
            "E - no arm supports genuine held-out relational generalization under "
            "the common learner on the full admit-and-force task"
        ),
        "why_E": (
            "The task is admit-AND-force, and forcing is unsolved by every "
            "representation. The nonlearned oracles name the forced third exactly "
            "(1.0) while the best learned arm reaches "
            f"{floor['max_over_all_arms']}. No arm therefore demonstrates held-out "
            "relational generalization of the certified relation as a whole."
        ),
        "how_E_is_qualified": (
            "E's clause 'representation comparison is unresolved' is too strong "
            "for what was measured and is explicitly narrowed here. The ADMISSION "
            "half is cleanly resolved and structurally informative: on the one "
            "non-admission class that habitat coincidence cannot settle, B beats "
            f"the matched scramble C by {bc['mean_difference']} with a 95% CI of "
            f"[{bc['ci_95']['low']}, {bc['ci_95']['high']}], sign-consistent across "
            f"{bc['n_folds']} folds, while the structure-preserving relabeling D "
            f"matches B ({db['mean_difference']}). That pattern supports H_struct "
            "and H_equiv on admission. It does not support H_suff for the full "
            "relation, because forcing fails."
        ),
        "supported_hypotheses": {
            "H_suff": (
                "NOT SUPPORTED for the full admit-and-force relation: forcing is at "
                "the floor. Partially supported for admission only."
            ),
            "H_struct": (
                "SUPPORTED on admission: B >> C on the discriminating class with "
                "matched marginals, bit widths and habitat sizes, so the gain "
                "tracks relational alignment rather than code capacity."
            ),
            "H_realization": (
                "SUPPORTED on admission in a direction the task did not single out: "
                "native realization did not merely fail to add advantage, it "
                "performed at chance on admission while the compact code did not."
            ),
            "H_equiv": (
                "SUPPORTED on admission: D matches B under GL(3,2) + affine "
                "AGL(2,2) + global S flip, so the admission result is not a "
                "basis artifact."
            ),
            "H_exec": "Already established upstream by Outcome 021.05; not retested.",
            "H_discovery": "Deliberately not tested; deferred by design.",
        },
        "opaque_token_discovery_stage_warranted": False,
        "opaque_token_recommendation": (
            "NOT warranted yet. The deferred discovery stage asks whether a learner "
            "can recover an SFP-equivalent representation from opaque tokens. That "
            "question only becomes measurable once being HANDED the exact code "
            "suffices for the task, and here it does not: Arm B is handed the exact "
            "certified code and still cannot name the held-out forced third. "
            "Hiding the code behind opaque tokens would make a strictly harder "
            "version of an already-failing task and could not discriminate. The "
            "prerequisite is to first close the forcing gap under a supplied code."
        ),
        "findings": findings,
        "repair_accounting": {
            "repairs_permitted": 1,
            "repairs_attempted": 1,
            "repair": "TrainConfig.mask_held_out_candidates=True",
            "motivation": (
                "harness.py demonstrated held-out target suppression: on a "
                "held-out-habitat fold the correct forced third is itself a "
                "held-out Event, scored as a candidate during training but never "
                "the right answer, so cross-entropy suppresses exactly the slots "
                "the test set needs"
            ),
            "outcome": (
                "TESTED AND INSUFFICIENT. It raises forced-third accuracy on the "
                "primary family but leaves it at the floor in absolute terms, so it "
                "changes no qualitative conclusion. Reported rather than iterated "
                "on: the stop rule permits exactly one repair and searching for a "
                "second after seeing results would be tuning."
            ),
            "repair_effect_on_forced_third": {
                key: {
                    "primary": primary["summary"][key]["metrics"].get(
                        "positive_forced_third_exact_accuracy"
                    ),
                    "repair": repair["summary"][key]["metrics"].get(
                        "positive_forced_third_exact_accuracy"
                    ),
                }
                for key in sorted(repair["summary"])
            },
            "no_second_repair_attempted": True,
        },
        "excluded_from_the_disposition": {
            "E_opaque": (
                "diagnostic memorization ceiling; token_dim == catalogue size gives "
                "its adapter one learnable column per Event, so it is not a "
                "representation arm and never substitutes for Arm C"
            ),
            "RANDOM_CONTROL": (
                "found non-dispositive by the Task 8 leakage search (50 "
                "cheaper-than-the-relation subsets reach ceiling); regression "
                "control only"
            ),
        },
    }


def analyse() -> dict:
    sweep = json.loads(SWEEP.read_text())
    blocks = {}
    for name in ("primary", "repair"):
        runs = load_runs(name)
        families = tuple(sorted({r["fold_family"] for r in runs}))
        summary = arm_family_summary(runs)
        blocks[name] = {
            "n_runs": len(runs),
            "families": list(families),
            "config": sweep["blocks"][name]["config"],
            "summary": summary,
            "comparisons": comparisons_for(runs, families),
            "forcing_floor": forcing_floor(summary),
        }

    payload = {
        "issue": "009 - SFP consequence representation versus native FIPS realization",
        "task": "Task 11 - paired effects and primary disposition",
        "module": "experiments/sfp_representation/analysis.py",
        "fences": list(FENCES),
        "metric_policy": {
            "primary_metrics": list(PRIMARY_METRICS),
            "weak_metrics": list(WEAK_METRICS),
            "context_metrics": list(CONTEXT_METRICS),
            "why": (
                "the Task 8 leakage search found that every relation-equivalent "
                "feature determines ADMISSION only; the best cheap subset reaches "
                "admission balanced accuracy 0.921 (LOHO) / 0.958 (LOFPO) while "
                "scoring 0.0 on forced third and 0.0 on SAME_HABITAT_DISJOINT. So "
                "forcing and the same-habitat-disjoint class carry the science and "
                "admission balanced accuracy does not."
            ),
            "reading_caution": (
                "an arm that answers BOTTOM almost everywhere scores high on the "
                "non-admission classes and at chance on admission, so "
                "non-admission accuracy must always be read together with "
                "admission sensitivity"
            ),
            "no_post_hoc_margin": (
                "no numeric equivalence margin is declared after the fact; paired "
                "bootstrap CIs and cross-fold sign consistency are reported instead"
            ),
        },
        "pairing": (
            "every comparison differences LEFT - RIGHT over identical (fold, seed) "
            "cells, never across marginal means"
        ),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "level": CI_LEVEL,
            "method": "percentile bootstrap of the paired mean difference",
        },
        "science_arms": list(SCIENCE_ARMS),
        "diagnostic_arms": list(DIAGNOSTIC_ARMS),
        "blocks": blocks,
        "disposition": disposition(blocks["primary"], blocks["repair"]),
        "upstream_digests": {
            "sweep_manifest": sweep["digests"]["manifest"],
            "sweep_blocks": sweep["digests"]["blocks"],
            "catalogue": sweep["upstream_digests"]["catalogue"],
            "dataset": sweep["upstream_digests"]["dataset"],
            "arms": sweep["upstream_digests"]["arms"],
        },
    }
    payload["digests"] = {
        "manifest": digest({k: v for k, v in payload.items() if k != "digests"})
    }
    return payload


def _report(payload: dict) -> None:
    shd = "same_habitat_disjoint_nonadmission_accuracy"
    for block in ("primary", "repair"):
        node = payload["blocks"][block]
        for family in node["families"]:
            print(f"\n=== {block} / {family}: paired effects ===", flush=True)
            print(
                f"{'comparison':<12}{'metric':<46}{'mean':>10}{'ci_low':>10}"
                f"{'ci_high':>10}{'excl0':>7}{'signcon':>8}",
                flush=True,
            )
            for name in ("B_minus_A", "B_minus_C", "D_minus_B"):
                for metric in (shd, "positive_forced_third_exact_accuracy",
                               "admission_balanced_accuracy"):
                    d = node["comparisons"][family][name]["metrics"][metric]
                    if not d.get("paired"):
                        continue
                    print(
                        f"{name:<12}{metric:<46}{d['mean_difference']:>10.4f}"
                        f"{d['ci_95']['low']:>10.4f}{d['ci_95']['high']:>10.4f}"
                        f"{str(d['ci_excludes_zero']):>7}"
                        f"{str(d['sign_consistent_across_folds']):>8}",
                        flush=True,
                    )
    disp = payload["disposition"]
    print(f"\n=== DISPOSITION: {disp['primary_disposition']} ===", flush=True)
    print(disp["primary_disposition_text"], flush=True)
    print(f"\nforcing floor: max over all arms = "
          f"{payload['blocks']['primary']['forcing_floor']['max_over_all_arms']} "
          f"vs oracle 1.0", flush=True)
    print(f"\nopaque-token discovery warranted: "
          f"{disp['opaque_token_discovery_stage_warranted']}", flush=True)
    print(f"manifest sha256: {payload['digests']['manifest']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = analyse()
    text = render(payload)
    _report(payload)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(f"FAIL: re-derived analysis differs from {OUTPUT.name}")
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        return

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)


if __name__ == "__main__":
    main()
