"""
Follow-up to Sec 6.3 of occurrence_i_rigor.md: is the measured ~2-2.4e-4/step
trajectory death rate actually consistent with the paper's stated ~1.0e-3/step
once sample size, run length, or death-tolerance convention are accounted for?

Tests three specific hypotheses, each cheap to falsify:
  (1) N-dependence: does a small-N point estimate drift toward 1e-3, i.e. was
      the paper's figure plausibly a noisy small-sample reading of the same
      underlying ~2e-4 rate? (This is how the historical D3 role-exchange
      statistic ultimately resolved -- worth checking the same hypothesis.)
  (2) Tolerance-convention dependence: is the measured rate sensitive to the
      norm threshold used to declare a trajectory "dead"?
  (3) Time-horizon dependence: does the rate keep climbing at longer T, i.e.
      might 1e-3 be a late-time/near-stationary value beyond what a short
      run would see?

Result (see occurrence_i_rigor.md Sec 6.3 for the write-up): none of the
three hold up. The mortality hazard is stable at ~2.0-2.6e-4 across N in
[200, 20000], T in [30, 800], and death_tol across ten orders of magnitude.
These tests vary conditions within the survival-conditioned protocol; the
later resolution showed that the paper's ~1.0e-3 value measures rejected
annihilating proposals under event resampling, a different observable.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occurrence_i_rigor_gauge_check_survival import n, L_mats, run_survival


def run_time_resolved(N, T, seed, death_tol=1e-9):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    alive = np.ones(N, dtype=bool)
    deaths_per_step = []
    n_alive_per_step = []
    for t in range(T):
        z_idx = rng.integers(0, 84, size=N)
        Mt = L_mats[z_idx]
        Xnew = np.einsum('nij,nj->ni', Mt, X)
        nrm = np.linalg.norm(Xnew, axis=1)
        newly_dead = alive & (nrm < death_tol)
        n_alive_per_step.append(alive.sum())
        deaths_per_step.append(newly_dead.sum())
        alive = alive & (nrm >= death_tol)
        safe_nrm = np.where(nrm > 0, nrm, 1.0)
        X = np.where(alive[:, None], Xnew / safe_nrm[:, None], X)
    return np.array(deaths_per_step), np.array(n_alive_per_step)


if __name__ == "__main__":
    print("=== Hypothesis 1: N-sweep at T=30, single seed each ===")
    print("    (does the point estimate drift toward 1e-3 at low N?)")
    for N in (200, 500, 1000, 2000, 4000, 8000, 20000):
        m, se, na, dr, n_deaths, at_risk_steps = run_survival(
            False, N=N, T=30, burn_in=0, seed=7, death_tol=1e-9
        )
        dr_err = np.sqrt(max(n_deaths, 1)) / at_risk_steps
        print(f"  N={N:6d}: death_rate={dr:.4e} +/- {dr_err:.1e}  (n_deaths={n_deaths})")
    print("  -> no drift toward 1e-3; noisy but centered near 2-4e-4 throughout.\n")

    print("=== Hypothesis 2: sensitivity to the death-tolerance threshold ===")
    print("    (N=8000, T=200, seed=1)")
    for tol in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        m, se, na, dr, _, _ = run_survival(False, N=8000, T=200, burn_in=0, seed=1, death_tol=tol)
        print(f"  death_tol={tol:.0e}:  death_rate={dr:.4e}  (alive at end: {na}/8000)")
    print("  -> flat across 10 orders of magnitude in the threshold; deaths are a")
    print("     hard collapse to exact 0.0, not a graded approach.\n")

    print("=== Hypothesis 3: does the rate keep climbing at longer T? ===")
    print("    (N=20000, T=800, seed=1, 50-step chunks)")
    deaths, alive_counts = run_time_resolved(N=20000, T=800, seed=1)
    chunk = 50
    for start in range(0, 800, chunk):
        d = deaths[start:start + chunk].sum()
        a = alive_counts[start:start + chunk].sum()
        rate = d / a if a > 0 else float("nan")
        print(f"  steps {start:3d}-{start + chunk - 1:3d}: rate={rate:.3e}")
    print(f"  overall rate, all 800 steps: {deaths.sum() / alive_counts.sum():.3e}")
    print("  -> plateaus around 2.2-2.6e-4 well before t=800; no late-time climb toward 1e-3.")
