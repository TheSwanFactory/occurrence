"""Controlled follow-up to the historical Test-14 D3 comparison.

This script answers two questions separately:

1. Is seed 30's 0.13240-vs-0.13761 gap unusual under the exact historical
   continuum/endpoint protocol?
2. Is there a detectable finite-time e0-vs-e1 initialization effect when the
   orientation and event streams are held fixed?

The historical verdict purported to make retained/sampled role exchange
detectable, which concerns Theorem 3.10(6). Proposition 4.2's separate
left/right-handedness claim is tested pointwise in
occurrence_i_rigor_gauge_check.py. This script diagnoses the historical
experiment rather than proving either theorem.
"""
import argparse
import numpy as np

from occurrence_i_rigor_gauge_check_survival import C, n


def sample_continuum(rng, count):
    """Verbatim continuum-crack sampler from the historical script."""
    a = rng.standard_normal((count, 8))
    a[:, 0] = 0
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.standard_normal((count, 8))
    b[:, 0] = 0
    b -= np.sum(b * a, axis=1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.concatenate([a, b], axis=1) / np.sqrt(2)


def left_step(states, events):
    event_products = np.einsum("ni,ijk->njk", events, C)
    return np.einsum("njk,nj->nk", event_products, states)


def right_step(states, events):
    return np.einsum("ni,nj,ijk->nk", states, events, C, optimize=True)


def normalize(states):
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def spine_share(states):
    return np.mean(states[:, 0] ** 2 + states[:, 8] ** 2)


def historical_run(seed, trajectories=1000, steps=30):
    """Reproduce the historical sequential-RNG, continuum endpoint design."""
    rng = np.random.default_rng(seed)
    standard = np.zeros((trajectories, n))
    standard[:, 0] = 1
    for _ in range(steps):
        standard = normalize(left_step(standard, sample_continuum(rng, trajectories)))

    swapped = np.zeros((trajectories, n))
    swapped[:, 1] = 1
    for _ in range(steps):
        swapped = normalize(right_step(swapped, sample_continuum(rng, trajectories)))
    return spine_share(standard), spine_share(swapped)


def paired_initialization_run(seed, trajectories=1000, steps=30):
    """Compare e0 and e1 starts using identical events in one orientation."""
    rng = np.random.default_rng(seed)
    states = np.zeros((2 * trajectories, n))
    states[:trajectories, 0] = 1
    states[trajectories:, 1] = 1
    for _ in range(steps):
        events = sample_continuum(rng, trajectories)
        paired_events = np.concatenate([events, events])
        states = normalize(left_step(states, paired_events))
    return spine_share(states[:trajectories]), spine_share(states[trajectories:])


def summarize(historical_reps=500, initialization_reps=200):
    historical = np.array([historical_run(seed) for seed in range(historical_reps)])
    differences = historical[:, 0] - historical[:, 1]
    observed = historical_run(30)
    observed_difference = observed[0] - observed[1]
    z_score = (observed_difference - differences.mean()) / differences.std(ddof=1)
    exceedances = np.sum(np.abs(differences) >= abs(observed_difference))

    initialization = np.array(
        [paired_initialization_run(seed) for seed in range(initialization_reps)]
    )
    initialization_effects = initialization[:, 0] - initialization[:, 1]
    initialization_se = initialization_effects.std(ddof=1) / np.sqrt(initialization_reps)

    print(f"seed 30: standard={observed[0]:.5f}, swapped={observed[1]:.5f}")
    print(
        f"historical-protocol diff: mean={differences.mean():+.6f}, "
        f"sd={differences.std(ddof=1):.6f}, seed-30 z={z_score:+.2f}, "
        f"exceedances={exceedances}/{historical_reps}"
    )
    print(
        f"paired e0-e1 effect: mean={initialization_effects.mean():+.6f}, "
        f"95% CI=[{initialization_effects.mean() - 1.96 * initialization_se:+.6f}, "
        f"{initialization_effects.mean() + 1.96 * initialization_se:+.6f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-reps", type=int, default=500)
    parser.add_argument("--initialization-reps", type=int, default=200)
    args = parser.parse_args()
    summarize(args.historical_reps, args.initialization_reps)
