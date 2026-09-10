#!/usr/bin/env python3
"""Recovery penalty as a function of recovery quality — an added `008.07` diagnostic.

**This probe was not commissioned by `008.07`.** It is reported as an addition,
with its reason stated, rather than folded into the Ladder-D arms.

`008.07` sections 5, 9 and 10 all ask for the recovery penalty

```text
Delta_s = recovered-denotation success - true-denotation success
```

and `008.06` section 8 makes it central to interpreting Ladder D. But at this cell
recovery is *exact*: all sixteen tokens are recovered with no ties and a minimum
vote margin around 9000, so the recovered pool is bit-identical to the true pool
and ``Delta_s = 0`` on every seed as an **identity**, not a measurement. The
commissioned number exists and is uninformative.

The observation sweep in ``recovery.recover_observation_sweep`` already produces
genuinely imperfect recovered maps as a by-product, at no extra cost: fitting the
same voting recovery to nested subsamples of the *same training half* yields maps
at 7/16 and 14/16 catalogue accuracy. Running the frozen `StructuralPolicy`
against those maps turns the vacuous penalty into a curve.

Nothing about the task, the target, the splits or the policy changes. Every map
here is fitted on training observations only — they are strict subsets of the
observations the primary arm used, so the `008.07` section 4 leak boundary is
inherited rather than re-argued. Scoring is the same exact projective native
endpoint semantics.

Usage::

    PYTHONPATH=. python experiments/tlm_multitoken/probe_degraded_recovery.py \
        --out experiments/tlm_multitoken/00807_artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.tlm_multitoken import policy, recovery, task

FENCE = (
    "added diagnostic, not a commissioned 008.07 arm; every recovered map is "
    "fitted on a subset of the same training observations, so the leak boundary "
    "is inherited; exact projective native endpoint scoring throughout"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--split", default="motif")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "00807_artifacts"
    )
    args = parser.parse_args()
    cache = args.out / "recovery_cache.json"
    if not cache.exists():
        print(f"no recovery cache at {cache}; run run_ladder_d_00807.py first")
        return 2

    started = time.time()
    train_config = policy.TrainConfig(steps=args.steps, seeds=tuple(args.seeds))
    vocab_config = recovery.VocabConfig()
    true_idx = recovery.build_vocabulary(vocab_config)
    pool = recovery.materialize_token_pool(true_idx)
    splits = {
        s.name: s
        for s in task.build_splits(pool, include=("motif", "motif_parity", "random"))
    }
    split = splits[args.split]
    phi = policy.build_phi(pool)

    sweep = json.loads(cache.read_text())[f"{args.split}::sweep"]
    bar = task.best_deterministic(task.score_baselines(pool, split))

    # The paired true-denotation reference, same seeds.
    true_runs = [
        policy.train_and_score(
            pool, split, architecture="structural", seed=seed,
            config=train_config, phi=phi,
        )
        for seed in train_config.seeds
    ]
    true_values = [r.test_score.exact_native_success for r in true_runs]
    print(f"true denotations      16/16  test={np.mean(true_values):.4f}", flush=True)

    rungs = []
    for entry in sweep:
        rec = recovery.Recovery(
            true_idx=tuple(true_idx),
            recovered_idx=tuple(entry["recovered_idx"]),
            votes=(),
            tied_tokens=tuple(entry["tied_tokens"]),
            zero_vote_tokens=(),
            n_observed_endpoints=entry["n_distinct_endpoints"],
            n_train_records=entry["n_train_observations"],
            stream_evaluations=0,
            wall_sec=0.0,
        )
        rebuilt = recovery.recovered_pool(pool, true_idx, rec)
        rec_pool = rebuilt.pool
        runs = [
            policy.train_and_score(
                rec_pool, split, architecture="structural", seed=seed,
                config=train_config, phi=phi,
            )
            for seed in train_config.seeds
        ]
        values = [r.test_score.exact_native_success for r in runs]
        deltas = [d - t for d, t in zip(values, true_values, strict=True)]
        ceiling = recovery.recovered_reachable_ceiling(rec_pool, split.test)
        n = len(rebuilt.pool.records)
        rung = {
            "n_train_observations": entry["n_train_observations"],
            "n_distinct_endpoints": entry["n_distinct_endpoints"],
            "catalogue_match_count": entry["catalogue_match_count"],
            "n_tied_tokens": entry["n_tied_tokens"],
            "sign_bit_preserved_count": sum(
                1
                for a, b in zip(true_idx, entry["recovered_idx"], strict=True)
                if _sign(a) == _sign(b)
            ),
            "recovered_reachable_ceiling_test": ceiling,
            "legal_set_changed_share": sum(rebuilt.legal_set_changed) / n,
            "target_became_undefined_share": sum(rebuilt.target_became_undefined) / n,
            "target_endpoint_changed_share": sum(rebuilt.target_endpoint_changed) / n,
            "recovered_test": policy.mean_std(values),
            "delta_vs_true": policy.mean_std(deltas),
            "beats_deterministic_bar_materially": (
                float(np.mean(values)) - bar.exact_native_success >= 0.10
            ),
            "seeds_beating_bar_materially": sum(
                1 for v in values if v - bar.exact_native_success >= 0.10
            ),
        }
        rungs.append(rung)
        print(
            f"recovered {rung['catalogue_match_count']:2d}/16 "
            f"(obs={rung['n_train_observations']:5d})  "
            f"test={rung['recovered_test']['mean']:.4f}  "
            f"delta={rung['delta_vs_true']['mean']:+.4f}  "
            f"ceiling={ceiling:.4f}  "
            f"endpoint_changed={rung['target_endpoint_changed_share']:.4f}  "
            f"beats_bar={rung['beats_deterministic_bar_materially']}",
            flush=True,
        )

    survives = [r for r in rungs if r["beats_deterministic_bar_materially"]]
    breaks = [r for r in rungs if not r["beats_deterministic_bar_materially"]]
    report = {
        "probe": "degraded recovery penalty curve",
        "commissioned_by_008_07": False,
        "why": (
            "008.07 sections 5/9/10 require the recovery penalty, but at this cell "
            "recovery is exact, so the commissioned penalty is identically zero as "
            "an identity rather than a measurement. This probe measures it against "
            "genuinely imperfect maps produced by the observation sweep."
        ),
        "split": args.split,
        "seeds": list(train_config.seeds),
        "true_denotation_test": policy.mean_std(true_values),
        "best_deployable_deterministic_baseline": {
            "name": bar.name,
            "test": bar.exact_native_success,
        },
        "rungs": rungs,
        "lowest_catalogue_accuracy_still_beating_the_bar": (
            min(r["catalogue_match_count"] for r in survives) if survives else None
        ),
        "highest_catalogue_accuracy_failing_the_bar": (
            max(r["catalogue_match_count"] for r in breaks) if breaks else None
        ),
        "fence": FENCE,
        "wall_sec": time.time() - started,
    }
    path = args.out / "degraded_recovery_probe.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "lowest_catalogue_accuracy_still_beating_the_bar",
        "highest_catalogue_accuracy_failing_the_bar",
    )}, indent=2))
    print(f"wrote {path}")
    return 0


def _sign(catalogue_index: int) -> int:
    from experiments.tlm_multitoken.probe_program_space import sign_bit
    from topographo.ssd import fips_basic

    return sign_bit(fips_basic.EVENTS[catalogue_index])


if __name__ == "__main__":
    raise SystemExit(main())
