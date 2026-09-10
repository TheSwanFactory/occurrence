#!/usr/bin/env python3
"""`008.07` Ladder D: does learned strict composition survive recovered denotations?

Answers one question:

> Does the positive `008.05` Ladder-A result survive when the strict `E^3 R`
> learner receives **train-only recovered and then frozen** Event denotations
> instead of true supplied denotations?

The scientific task is frozen exactly as `008.06` section 7 requires — `E^3 R`,
strict `Occ + Cyc + Sand`, `r = e4`, twenty planar programs, the `008.02` sign-bit
target, the `StructuralPolicy`, the `motif` / `motif_parity` splits, all
deterministic baselines, no result decoder. The single interface change is the
token layer that makes "recovered denotation" well-typed at all; see
``recovery.py``.

What runs, in order:

1. the frozen sign-balanced token vocabulary and the true-denotation token pool;
2. the `008.06` Gate 2 program-space viability check on the token task;
3. the `008.06` Gate 3 target-signature holdout check, including the train-fitted
   sign-pattern lookup baseline on every reported split;
4. **per split**, train-only endpoint-inverse denotation recovery, plus a
   mechanical leak audit — one recovery per split, because a map fitted on one
   split's training half would leak into another split's held-out half;
5. the `008.07` section 6 recovery diagnostics;
6. paired true-vs-recovered `StructuralPolicy` runs over the same policy seeds;
7. deterministic baselines on the recovered cells;
8. the section 10 disposition and section 11 failure classification.

Recovery is the expensive part: one streamed pass over ``84^3 x 20 = 11854080``
exact endpoints per split, about nine CPU-minutes each. ``--recovery-cache``
stores the recovered maps so the policy side can be re-run cheaply.

Usage::

    PYTHONPATH=. python experiments/tlm_multitoken/run_ladder_d_00807.py \
        --seeds 0 1 2 3 4 5 6 7 \
        --out experiments/tlm_multitoken/00807_artifacts

Fence: strict constructors on the exact rational path; recovery from train-side
observations only; exact projective native endpoint scoring; no learned result
decoder; no physical Event supply, no physical program-tree selection, no modular
grokking, no language-model claim, no OT-native probability-head claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.tlm_multitoken import policy, recovery, task

#: `008.05` / `008.06` materiality bar, unchanged.
MATERIAL_MARGIN = 0.10

#: The discriminative cells. `008.07` section 3: these withhold whole values of
#: the target-determining sign-pattern signature, so they pass Gate 3.
SCIENTIFIC_SPLITS = ("motif", "motif_parity")

#: Retained only as a cheap regression control, explicitly non-dispositive
#: because ``SignPatternLookupTrainFitted`` scores 1.000 on it.
CONTROL_SPLITS = ("random",)

FENCE = (
    "strict Occ/Cyc/Sand on the exact rational path; denotations recovered from "
    "train-side observations only and frozen before held-out evaluation; exact "
    "projective native endpoint scoring; no learned result decoder; no physical "
    "Event supply, no physical program-tree selection, no modular grokking, no "
    "language-model advantage, no OT-native probability-head claim, no ambient "
    "generalized Mul as native OT"
)


def sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cell(runs: list[policy.PolicyRun], pool: task.Pool, split: task.Split) -> dict:
    out = {
        "architecture": runs[0].architecture,
        "split": runs[0].split,
        "param_count": runs[0].param_count,
        "n_train": runs[0].train_score.n,
        "n_test": runs[0].test_score.n,
        "train_exact_native_success": policy.mean_std(
            [r.train_score.exact_native_success for r in runs]
        ),
        "test_exact_native_success": policy.mean_std(
            [r.test_score.exact_native_success for r in runs]
        ),
        "test_tree_selection_accuracy": policy.mean_std(
            [r.test_score.tree_selection_accuracy for r in runs]
        ),
        "test_endpoint_match_via_different_tree": policy.mean_std(
            [r.test_score.endpoint_match_via_different_tree for r in runs]
        ),
        "test_selected_defined_share": policy.mean_std(
            [r.test_score.selected_defined_share for r in runs]
        ),
        "test_endpoint_accuracy_given_selected_defined": policy.mean_std(
            [r.test_score.endpoint_accuracy_given_selected_defined for r in runs]
        ),
        "generalization_gap": policy.mean_std([r.generalization_gap for r in runs]),
        "wall_sec": policy.mean_std([r.wall_sec for r in runs]),
        "seeds": [r.seed for r in runs],
        "per_seed_test": {
            str(r.seed): r.test_score.exact_native_success for r in runs
        },
    }
    if split.kind == "compositional_heldout_motif":
        out["per_heldout_pattern"] = _per_pattern(pool, split, runs)
    return out


def _per_pattern(
    pool: task.Pool, split: task.Split, runs: list[policy.PolicyRun]
) -> dict:
    out: dict[str, dict] = {}
    for pattern in sorted({pool.records[i].sign_pattern for i in split.test}):
        indices = [i for i in split.test if pool.records[i].sign_pattern == pattern]
        per_seed = [
            sum(
                1
                for i in indices
                if pool.records[i].matches(
                    run.test_picks_by_input.get(pool.records[i].event_indices)
                )
            )
            / (len(indices) or 1)
            for run in runs
        ]
        out[f"{pattern:03b}"] = {
            "target_program": pool.program_names[
                pool.records[indices[0]].target_program
            ],
            "n_test": len(indices),
            "exact_native_success": policy.mean_std(per_seed),
        }
    return out


def _load_cache(path: Path | None, split_name: str) -> recovery.Recovery | None:
    if path is None or not path.exists():
        return None
    blob = json.loads(path.read_text())
    entry = blob.get(split_name)
    if entry is None:
        return None
    return recovery.Recovery(
        true_idx=tuple(entry["true_idx"]),
        recovered_idx=tuple(entry["recovered_idx"]),
        votes=tuple(tuple(row) for row in entry["votes"]),
        tied_tokens=tuple(entry["tied_tokens"]),
        zero_vote_tokens=tuple(entry["zero_vote_tokens"]),
        n_observed_endpoints=entry["n_observed_endpoints"],
        n_train_records=entry["n_train_records"],
        stream_evaluations=entry["stream_evaluations"],
        wall_sec=entry["wall_sec"],
    )


def _load_sweep(path: Path | None, split_name: str) -> list[dict] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text()).get(f"{split_name}::sweep")


def _save_sweep(path: Path | None, split_name: str, sweep: list[dict]) -> None:
    if path is None:
        return
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob[f"{split_name}::sweep"] = sweep
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")


def _save_cache(path: Path | None, split_name: str, rec: recovery.Recovery) -> None:
    if path is None:
        return
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob[split_name] = {
        "true_idx": list(rec.true_idx),
        "recovered_idx": list(rec.recovered_idx),
        "votes": [list(row) for row in rec.votes],
        "tied_tokens": list(rec.tied_tokens),
        "zero_vote_tokens": list(rec.zero_vote_tokens),
        "n_observed_endpoints": rec.n_observed_endpoints,
        "n_train_records": rec.n_train_records,
        "stream_evaluations": rec.stream_evaluations,
        "wall_sec": rec.wall_sec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "00807_artifacts"
    )
    parser.add_argument("--recovery-cache", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.recovery_cache or (args.out / "recovery_cache.json")

    started = time.time()
    vocab_config = recovery.VocabConfig()
    train_config = policy.TrainConfig(steps=args.steps, seeds=tuple(args.seeds))

    # --- 1. the frozen token layer and the true-denotation arm --------------
    true_idx = recovery.build_vocabulary(vocab_config)
    pool = recovery.materialize_token_pool(true_idx)
    print(f"token vocabulary: {true_idx}", flush=True)
    print(f"true-denotation pool: {len(pool.records)} triples, "
          f"{len(pool.scored)} scored", flush=True)

    # --- 2. Gate 2: program-space viability --------------------------------
    gate = task.viability_gate(pool)
    print("Gate 2 (program-space viability): "
          f"passed={gate.passed} ceiling={gate.availability_ceiling:.4f} "
          f"headroom={gate.learned_headroom:.4f}", flush=True)
    if not gate.passed:
        (args.out / "viability_gate_failure.json").write_text(
            json.dumps(gate.as_dict(), indent=2) + "\n"
        )
        print("VIABILITY GATE FAILED — refusing to train")
        return 2

    splits = task.build_splits(
        pool, include=(*SCIENTIFIC_SPLITS, *CONTROL_SPLITS)
    )
    by_name = {s.name: s for s in splits}
    phi = policy.build_phi(pool)

    # --- 3. Gate 3: target-signature holdout ------------------------------
    baselines_true = {s.name: task.score_baselines(pool, s) for s in splits}
    gate3 = {}
    for s in splits:
        lookup = next(
            b
            for b in baselines_true[s.name]
            if b.name == "SignPatternLookupTrainFitted" and b.half == "test"
        )
        withholds = s.kind == "compositional_heldout_motif"
        gate3[s.name] = {
            "withholds_whole_signature_values": withholds,
            "sign_pattern_lookup_test": lookup.exact_native_success,
            "discriminative": withholds and lookup.exact_native_success < 0.5,
            "role": "scientific" if s.name in SCIENTIFIC_SPLITS else "control",
        }
    print("Gate 3 (target-signature holdout): "
          + ", ".join(
              f"{k}={'discriminative' if v['discriminative'] else 'CONTROL'} "
              f"(lookup {v['sign_pattern_lookup_test']:.4f})"
              for k, v in gate3.items()
          ), flush=True)

    # --- 4/5. recovery, per split, plus the leak audit --------------------
    recoveries: dict[str, recovery.Recovery] = {}
    recovered: dict[str, recovery.RecoveredPools] = {}
    leak_audits: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    sweeps: dict[str, list[dict] | None] = {}
    for name in SCIENTIFIC_SPLITS:
        split = by_name[name]
        cached = _load_cache(cache, name)
        if cached is not None:
            print(f"recovery[{name}]: loaded from cache", flush=True)
            rec = cached
        else:
            print(f"recovery[{name}]: streaming 84^3 x 20 exact endpoints "
                  "(about nine minutes) ...", flush=True)
            rec = recovery.recover_by_endpoint_votes(
                pool, split, true_idx, config=vocab_config
            )
            _save_cache(cache, name, rec)
        recoveries[name] = rec
        leak_audits[name] = recovery.audit_recovery_leak(pool, split, true_idx, rec)
        if not leak_audits[name]["leak_free"]:
            (args.out / "recovery_leak_failure.json").write_text(
                json.dumps(leak_audits[name], indent=2) + "\n"
            )
            print(f"RECOVERY LEAK AUDIT FAILED for {name} — refusing to score")
            return 3
        sweeps[name] = _load_sweep(cache, name)
        if sweeps[name] is None:
            print(f"recovery sweep[{name}]: one shared stream over nested "
                  "training subsamples ...", flush=True)
            sweeps[name] = recovery.recover_observation_sweep(
                pool, split, true_idx, config=vocab_config
            )
            _save_sweep(cache, name, sweeps[name])
        for rung in sweeps[name]:
            print(f"    obs={rung['n_train_observations']:5d} "
                  f"endpoints={rung['n_distinct_endpoints']:5d} "
                  f"catalogue={rung['catalogue_match_count']:2d}/16 "
                  f"ties={rung['n_tied_tokens']:2d} "
                  f"min_margin={rung['min_vote_margin']}", flush=True)
        recovered[name] = recovery.recovered_pool(pool, true_idx, rec)
        diagnostics[name] = recovery.recovery_diagnostics(
            pool, recovered[name], rec, splits=(split,)
        )
        rep = diagnostics[name]["representation_recovery"]
        print(
            f"recovery[{name}]: catalogue {rep['catalogue_match_count']}"
            f"/{rec.n_tokens} exact, sign bit preserved "
            f"{rep['sign_bit_preserved_count']}/{rec.n_tokens}, "
            f"ties {rep['n_tied_tokens']}, leak_free=True",
            flush=True,
        )

    # --- 6. paired true-vs-recovered policy runs --------------------------
    cells_true: dict[str, dict] = {}
    cells_rec: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    ceilings: dict[str, dict] = {}
    all_runs: list[policy.PolicyRun] = []

    for name in (*SCIENTIFIC_SPLITS, *CONTROL_SPLITS):
        split = by_name[name]
        true_runs = [
            policy.train_and_score(
                pool, split, architecture="structural", seed=seed,
                config=train_config, phi=phi,
            )
            for seed in train_config.seeds
        ]
        all_runs.extend(true_runs)
        cells_true[name] = _cell(true_runs, pool, split)
        line = (f"  true       {name:13s} "
                f"test={cells_true[name]['test_exact_native_success']['mean']:.4f}"
                f" +-{cells_true[name]['test_exact_native_success']['std']:.4f}")

        if name in recovered:
            rec_pool = recovered[name].pool
            rec_runs = [
                policy.train_and_score(
                    rec_pool, split, architecture="structural", seed=seed,
                    config=train_config, phi=phi,
                )
                for seed in train_config.seeds
            ]
            all_runs.extend(rec_runs)
            cells_rec[name] = _cell(rec_runs, rec_pool, split)
            deltas = [
                r.test_score.exact_native_success - t.test_score.exact_native_success
                for t, r in zip(true_runs, rec_runs, strict=True)
            ]
            paired[name] = {
                "seeds": list(train_config.seeds),
                "true_test": [t.test_score.exact_native_success for t in true_runs],
                "recovered_test": [r.test_score.exact_native_success for r in rec_runs],
                "delta_per_seed": deltas,
                "delta": policy.mean_std(deltas),
                "median_true": float(
                    np.median([t.test_score.exact_native_success for t in true_runs])
                ),
                "median_recovered": float(
                    np.median([r.test_score.exact_native_success for r in rec_runs])
                ),
            }
            ceilings[name] = {
                "recovered_reachable_ceiling_test": (
                    recovery.recovered_reachable_ceiling(rec_pool, split.test)
                ),
                "recovered_reachable_ceiling_train": (
                    recovery.recovered_reachable_ceiling(rec_pool, split.train)
                ),
                "note": (
                    "share of held-out inputs where SOME legal recovered program "
                    "reaches the ground-truth endpoint; an upper bound on the "
                    "recovered arm independent of any policy decision"
                ),
            }
            line += (f" | recovered="
                     f"{cells_rec[name]['test_exact_native_success']['mean']:.4f}"
                     f" +-{cells_rec[name]['test_exact_native_success']['std']:.4f}"
                     f" | delta={paired[name]['delta']['mean']:+.4f}"
                     f" | reachable ceiling="
                     f"{ceilings[name]['recovered_reachable_ceiling_test']:.4f}")
        print(line, flush=True)

    # --- 7. deterministic baselines on the recovered cells ---------------
    baselines_rec = {
        name: task.score_baselines(recovered[name].pool, by_name[name])
        for name in SCIENTIFIC_SPLITS
    }
    bars_true = {
        name: task.best_deterministic(scores)
        for name, scores in baselines_true.items()
    }
    bars_rec = {
        name: task.best_deterministic(scores)
        for name, scores in baselines_rec.items()
    }
    for name in SCIENTIFIC_SPLITS:
        print(f"  baselines  {name:13s} true bar={bars_true[name].exact_native_success:.4f}"
              f" ({bars_true[name].name}) | recovered bar="
              f"{bars_rec[name].exact_native_success:.4f} ({bars_rec[name].name})",
              flush=True)

    # --- 8. disposition ---------------------------------------------------
    disposition = _disposition(
        cells_rec, bars_rec, paired, ceilings, diagnostics, sweeps
    )

    report = {
        "task": "008.07",
        "result": "008.08",
        "question": (
            "does the positive 008.05 Ladder-A result survive when the strict "
            "E^3 R learner receives train-only recovered and frozen Event "
            "denotations instead of true supplied denotations?"
        ),
        "interface_change": {
            "what": (
                "008.05 fed the policy the exact Event rays, so nothing was "
                "recoverable. Ladder D adds a frozen sign-balanced token layer: "
                "the exposed input is three tokens over a fixed vocabulary whose "
                "denotations must be recovered from training observations."
            ),
            "kept_fixed": [
                "E^3 R", "Occ+Cyc+Sand strict calculus", "r = e4",
                "20 planar programs", "008.02 sign-bit target",
                "StructuralPolicy with fixed phi", "motif + motif_parity splits",
                "all deterministic baselines", "exact native endpoint scoring",
                "no result decoder",
            ],
            "why_necessary": (
                "recovered denotations are not well-typed unless denotations are "
                "hidden behind a symbol layer; this is the minimal change that "
                "makes Ladder D meaningful and it is reported rather than implied"
            ),
        },
        "vocabulary": {**vocab_config.as_dict(), "true_idx": list(true_idx)},
        "gates": {
            "gate_1_target_independence": {
                "target_is_availability_rule": False,
                "target_inspects_model_scores": False,
                "target_uses_a_decoder": False,
                "target_relation": (
                    "008.02 sign-bit latent bracketing grammar, unchanged"
                ),
            },
            "gate_2_program_space_viability": gate.as_dict(),
            "gate_3_target_signature_holdout": gate3,
        },
        "implementation": {
            "arithmetic": "topographo.ssd.exact rational sedenions",
            "recovery": (
                "017 Rec_ray endpoint-inverse voting, streamed over 84^3 x 20 "
                "exact endpoints, train-only, first-max tie convention"
            ),
            "device": "cpu",
            "host": platform.node(),
            "deps": {"torch": torch.__version__, "numpy": np.__version__},
            "train_config": train_config.as_dict(),
        },
        "splits": {
            s.name: {
                "kind": s.kind,
                "rationale": s.rationale,
                "role": "scientific" if s.name in SCIENTIFIC_SPLITS else "control",
                "n_train": len(s.train),
                "n_test": len(s.test),
                "digest_sha256": task.split_digest(pool, s),
                **s.metadata,
            }
            for s in splits
        },
        "recovery": {
            name: {
                **rec.as_dict(),
                "leak_audit": leak_audits[name],
                "diagnostics": diagnostics[name],
                "observation_sweep": sweeps.get(name),
                "observation_sweep_note": (
                    "diagnostic on the recovery interface, not a repair and not a "
                    "second arm: the same voting recovery fitted to nested "
                    "subsamples of the training observations, sharing one streamed "
                    "pass. It exists because full-sample recovery is exact, which "
                    "would otherwise leave the Ladder-D boundary unmeasured."
                ),
            }
            for name, rec in recoveries.items()
        },
        "learned_true_denotations": cells_true,
        "learned_recovered_denotations": cells_rec,
        "paired_true_vs_recovered": paired,
        "recovered_reachable_ceilings": ceilings,
        "baselines_true_denotations": {
            name: [b.as_dict() for b in scores]
            for name, scores in baselines_true.items()
        },
        "baselines_recovered_denotations": {
            name: [b.as_dict() for b in scores]
            for name, scores in baselines_rec.items()
        },
        "best_deployable_baseline_test": {
            "true": {
                name: [b.name, b.exact_native_success]
                for name, b in bars_true.items()
            },
            "recovered": {
                name: [b.name, b.exact_native_success]
                for name, b in bars_rec.items()
            },
        },
        "seed_results": [r.as_dict() for r in all_runs],
        "disposition": disposition,
        "stop_rule": {
            "recovered_denotation_passes": 1,
            "repairs": 0,
            "repair": None,
            "flat_policy_reopened": False,
            "note": (
                "008.07 section 8 allows one recovered-denotation pass plus at "
                "most one repair specific to the denotation/recovery interface. "
                "One pass was run. Whether a repair was warranted is recorded in "
                "the disposition."
            ),
        },
        "fence": FENCE,
        "wall_sec_total": time.time() - started,
    }
    report["manifest_sha256"] = sha256_json(
        {
            "vocabulary": report["vocabulary"],
            "splits": {k: v["digest_sha256"] for k, v in report["splits"].items()},
            "gate_2": gate.as_dict(),
            "recovered_idx": {
                name: list(rec.recovered_idx) for name, rec in recoveries.items()
            },
        }
    )

    tiny = {
        "manifest_sha256": report["manifest_sha256"],
        "catalogue_recovery": {
            name: diagnostics[name]["representation_recovery"]["catalogue_match_count"]
            for name in SCIENTIFIC_SPLITS
        },
        "sign_bit_preserved": {
            name: diagnostics[name]["representation_recovery"][
                "sign_bit_preserved_count"
            ]
            for name in SCIENTIFIC_SPLITS
        },
        "true_test": {
            name: round(c["test_exact_native_success"]["mean"], 4)
            for name, c in cells_true.items()
        },
        "recovered_test": {
            name: round(c["test_exact_native_success"]["mean"], 4)
            for name, c in cells_rec.items()
        },
        "recovery_penalty": {
            name: round(p["delta"]["mean"], 4) for name, p in paired.items()
        },
        "recovered_bar": {
            name: [b.name, round(b.exact_native_success, 4)]
            for name, b in bars_rec.items()
        },
        "recovered_reachable_ceiling": {
            name: round(c["recovered_reachable_ceiling_test"], 4)
            for name, c in ceilings.items()
        },
        "verdict": disposition["verdict"],
        "positive": disposition["positive"],
        "degenerate_as_a_robustness_test": disposition[
            "degenerate_as_a_robustness_test"
        ],
        "recovery_observation_sweep": disposition["recovery_observation_sweep"],
        "failure_classification": disposition["failure_classification"],
        "issue_008_closure_recommendation": disposition[
            "issue_008_closure_recommendation"
        ],
    }

    (args.out / "ladder_d_00807_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    (args.out / "ladder_d_00807_report_tiny.json").write_text(
        json.dumps(tiny, indent=2) + "\n"
    )
    (args.out / "split_metadata.json").write_text(
        json.dumps(
            {"vocabulary": report["vocabulary"], "splits": report["splits"]}, indent=2
        )
        + "\n"
    )
    (args.out / "recovery_diagnostics.json").write_text(
        json.dumps(report["recovery"], indent=2, default=str) + "\n"
    )
    print(json.dumps(tiny, indent=2))
    print(f"wrote {args.out}")
    print(f"elapsed {time.time() - started:.1f}s")
    return 0


def _disposition(
    cells_rec: dict[str, dict],
    bars_rec: dict[str, task.BaselineScore],
    paired: dict[str, dict],
    ceilings: dict[str, dict],
    diagnostics: dict[str, dict],
    sweeps: dict[str, list[dict] | None] | None = None,
) -> dict:
    """`008.07` section 10 success criterion and section 11 classification.

    Also decides whether the *test* was informative, which is separate from
    whether it passed. If recovery is exact, the recovered arm is bit-identical
    to the true arm and a zero recovery penalty says nothing about robustness to
    imperfect recovery. Reporting only "positive, penalty 0.0000" would overstate
    the evidence, so degeneracy is detected and surfaced.
    """
    per_split = {}
    for name, cell in cells_rec.items():
        bar = bars_rec[name]
        values = cell["test_exact_native_success"]["values"]
        beating = [
            v for v in values if v - bar.exact_native_success >= MATERIAL_MARGIN
        ]
        per_split[name] = {
            "recovered_test_mean": cell["test_exact_native_success"]["mean"],
            "recovered_test_median": float(np.median(values)),
            "recovered_test_per_seed": values,
            "best_deployable_baseline": bar.name,
            "best_deployable_baseline_test": bar.exact_native_success,
            "margin": cell["test_exact_native_success"]["mean"]
            - bar.exact_native_success,
            "seeds_beating_materially": len(beating),
            "material_on_more_than_one_seed": len(beating) > 1,
            "recovery_penalty": paired[name]["delta"]["mean"],
            "recovered_reachable_ceiling": ceilings[name][
                "recovered_reachable_ceiling_test"
            ],
        }

    winners = [
        name
        for name, v in per_split.items()
        if v["margin"] >= MATERIAL_MARGIN and v["material_on_more_than_one_seed"]
    ]
    positive = bool(winners)

    # Section 11 classification, driven by the diagnostics rather than by taste.
    classification = None
    if not positive:
        worst = min(per_split, key=lambda k: per_split[k]["margin"])
        rep = diagnostics[worst]["representation_recovery"]
        ceiling = per_split[worst]["recovered_reachable_ceiling"]
        if ceiling < MATERIAL_MARGIN + per_split[worst]["best_deployable_baseline_test"]:
            classification = "denotation recovery quality"
        elif rep["n_tied_tokens"] or rep["order_sensitive_tokens"]:
            classification = "recovery identifiability / tie sensitivity"
        elif rep["sign_bit_preserved_rate"] < 1.0:
            classification = "recovered-domain mismatch"
        else:
            classification = "policy optimization variance"

    if positive:
        best = max(winners, key=lambda k: per_split[k]["margin"])
        verdict = (
            f"POSITIVE on {', '.join(sorted(winners))}: recovered-denotation "
            f"StructuralPolicy reaches "
            f"{per_split[best]['recovered_test_mean']:.4f} against "
            f"{per_split[best]['best_deployable_baseline']} at "
            f"{per_split[best]['best_deployable_baseline_test']:.4f} "
            f"(margin {per_split[best]['margin']:+.4f}), with a recovery penalty "
            f"of {per_split[best]['recovery_penalty']:+.4f} against the paired "
            "true-denotation arm"
        )
        closure = (
            "Issue 008 is ready for closure on the composition question: the "
            "learned advantage survives a nonleaky recovery boundary. Any "
            "follow-on should be narrower than a new ladder."
        )
    else:
        worst = min(per_split, key=lambda k: per_split[k]["margin"])
        verdict = (
            f"NEGATIVE: best recovered margin {per_split[worst]['margin']:+.4f}; "
            f"dominant boundary is {classification}"
        )
        closure = (
            "Issue 008 should not close on Ladder D alone. One narrower follow-on "
            f"targeting {classification} is warranted."
        )

    # Was the recovery boundary actually stressed?
    exact_everywhere = all(
        d["representation_recovery"]["catalogue_match_rate"] == 1.0
        and d["representation_recovery"]["n_tied_tokens"] == 0
        for d in diagnostics.values()
    )
    zero_penalty = all(
        all(delta == 0.0 for delta in p["delta_per_seed"]) for p in paired.values()
    )
    degenerate = exact_everywhere and zero_penalty
    if degenerate:
        verdict += (
            " — but DEGENERATE as a robustness test: recovery is exact on every "
            "token with no ties, so the recovered arm is bit-identical to the "
            "true arm and the zero penalty is an identity, not a measurement of "
            "robustness to imperfect recovery"
        )
        closure = (
            "Issue 008 may close on the composition question: the learned "
            "advantage is unaffected by moving to a train-only recovered "
            "denotation map. But Ladder D did not test robustness to imperfect "
            "recovery, because at this cell the exact native endpoint "
            "over-determines the denotations. One narrower follow-on should "
            "degrade the recovery interface itself — fewer observations, partial "
            "or noisy endpoint reporting — rather than run another ladder."
        )

    sweep_summary = {}
    for name, sweep in (sweeps or {}).items():
        if not sweep:
            continue
        perfect = [r for r in sweep if r["catalogue_match_count"] == 16]
        sweep_summary[name] = {
            "smallest_observation_count_with_exact_recovery": (
                min(r["n_train_observations"] for r in perfect) if perfect else None
            ),
            "largest_observation_count_without_exact_recovery": (
                max(
                    (
                        r["n_train_observations"]
                        for r in sweep
                        if r["catalogue_match_count"] < 16
                    ),
                    default=None,
                )
            ),
            "curve": [
                {
                    "n_train_observations": r["n_train_observations"],
                    "n_distinct_endpoints": r["n_distinct_endpoints"],
                    "catalogue_match_count": r["catalogue_match_count"],
                    "n_tied_tokens": r["n_tied_tokens"],
                    "min_vote_margin": r["min_vote_margin"],
                }
                for r in sweep
            ],
        }

    return {
        "material_margin": MATERIAL_MARGIN,
        "per_split": per_split,
        "positive": positive,
        "verdict": verdict,
        "failure_classification": classification,
        "issue_008_closure_recommendation": closure,
        "recovery_was_exact_on_every_token": exact_everywhere,
        "recovery_penalty_identically_zero": zero_penalty,
        "degenerate_as_a_robustness_test": degenerate,
        "recovery_observation_sweep": sweep_summary,
        "interpretation_rule": (
            "008.06 section 8: Ladder D is not successful merely by beating "
            "chance. It must materially exceed the best deployable deterministic "
            "baseline on the same recovered cell, on more than one seed, and the "
            "recovery penalty against the paired true-denotation arm is reported "
            "as a substantive limitation even when the bar is cleared."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
