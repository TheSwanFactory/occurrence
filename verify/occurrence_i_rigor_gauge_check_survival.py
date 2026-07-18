"""
Self-contained (no external file dependency) vectorized rerun of the
standard-vs-swapped spine-share comparison, with survival-conditioning
(discard a trajectory once its norm underflows below a tolerance, per the
paper's own Sec 5.6 handling of "trajectories die (exact annihilation)").
Companion to occurrence_i_rigor_gauge_check.py; this version trades the
naive O(dim) inner loop for precomputed left/right multiplication matrices
so it can run thousands of trajectories for hundreds of steps in seconds.
"""
import numpy as np


def cd_double(mult_table):
    n = mult_table.shape[0]
    dim = 2 * n
    C = np.zeros((dim, dim, dim))
    conj_sign = np.array([1.0] + [-1.0] * (n - 1))

    def basis_pair(idx):
        if idx < n:
            a = np.zeros(n); a[idx] = 1.0; b = np.zeros(n)
        else:
            a = np.zeros(n); b = np.zeros(n); b[idx - n] = 1.0
        return a, b

    def mult_base(u, v):
        out = np.zeros(n)
        for i in range(n):
            if u[i] == 0:
                continue
            for j in range(n):
                if v[j] == 0:
                    continue
                out += u[i] * v[j] * mult_table[i, j]
        return out

    def conj_base(u):
        return u * conj_sign

    for I in range(dim):
        a1, b1 = basis_pair(I)
        for J in range(dim):
            a2, b2 = basis_pair(J)
            real_part = mult_base(a1, a2) - mult_base(conj_base(b2), b1)
            imag_part = mult_base(b2, a1) + mult_base(b1, conj_base(a2))
            for k in range(n):
                if abs(real_part[k]) > 1e-15:
                    C[I, J, k] += real_part[k]
                if abs(imag_part[k]) > 1e-15:
                    C[I, J, n + k] += imag_part[k]
    return C


C = np.ones((1, 1, 1))
for _ in range(4):
    C = cd_double(C)
n = C.shape[0]


def multiply(x, y, C):
    out = np.zeros(n)
    for i in range(n):
        if x[i] == 0:
            continue
        out += x[i] * (y @ C[i])
    return out


crack = []
for i in range(1, 8):
    for j in range(1, 8):
        for s in (1.0, -1.0):
            v = np.zeros(n); v[i] = 1.0 / np.sqrt(2); v[8 + j] = s / np.sqrt(2)
            Lv = np.zeros((n, n))
            for k in range(n):
                ek = np.zeros(n); ek[k] = 1.0
                Lv[:, k] = multiply(v, ek, C)
            if np.linalg.matrix_rank(Lv, tol=1e-9) < n:
                crack.append(v)
crack = np.array(crack)
assert len(crack) == 84

# precompute L_z and R_z matrices for all 84 crack points once
L_mats = np.zeros((84, n, n))
R_mats = np.zeros((84, n, n))
for idx, v in enumerate(crack):
    for k in range(n):
        ek = np.zeros(n); ek[k] = 1.0
        L_mats[idx, :, k] = multiply(v, ek, C)   # column k of L_v: v * e_k
        R_mats[idx, :, k] = multiply(ek, v, C)   # column k of R_v: e_k * v


def spine_share_batch(X):
    return X[:, 0] ** 2 + X[:, 8] ** 2


def run_survival(retained_on_left, N, T, burn_in, seed, death_tol=1e-9):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    alive = np.ones(N, dtype=bool)
    Mats = R_mats if retained_on_left else L_mats
    shares = []
    share_sums = np.zeros(N)
    share_counts = np.zeros(N, dtype=int)
    deaths_per_step = []
    at_risk_steps = 0
    for t in range(T):
        at_risk_steps += alive.sum()
        z_idx = rng.integers(0, 84, size=N)
        Mt = Mats[z_idx]                       # (N, n, n)
        Xnew = np.einsum('nij,nj->ni', Mt, X)  # batched matvec
        nrm = np.linalg.norm(Xnew, axis=1)
        newly_dead = alive & (nrm < death_tol)
        deaths_per_step.append(newly_dead.sum())
        alive = alive & (nrm >= death_tol)
        safe_nrm = np.where(nrm > 0, nrm, 1.0)
        X = np.where(alive[:, None], Xnew / safe_nrm[:, None], X)  # freeze dead trajectories
        if t >= burn_in and alive.sum() > 0:
            step_shares = spine_share_batch(X)
            shares.append(step_shares[alive])
            share_sums[alive] += step_shares[alive]
            share_counts[alive] += 1
    n_alive_final = alive.sum()
    mortality_hazard = sum(deaths_per_step) / at_risk_steps
    flat = np.concatenate(shares) if shares else np.array([])
    mean_share = flat.mean()
    # Cluster by trajectory: repeated observations from one trajectory are
    # temporally dependent and must not be counted as independent samples.
    cluster_residuals = share_sums - mean_share * share_counts
    active_clusters = share_counts > 0
    n_clusters = active_clusters.sum()
    cluster_se = (
        np.sqrt(n_clusters / (n_clusters - 1))
        * np.sqrt(np.sum(cluster_residuals[active_clusters] ** 2))
        / share_counts.sum()
        if n_clusters > 1 else float("nan")
    )
    return mean_share, cluster_se, n_alive_final, mortality_hazard, sum(deaths_per_step), at_risk_steps


if __name__ == "__main__":
    print("[Test D] Reported Test-14-D3 scale under the gauge-equivalent discrete design, 5 seeds:")
    for seed in range(1, 6):
        m_std, se_std, na_std, dr_std, _, _ = run_survival(False, N=1000, T=30, burn_in=0, seed=seed)
        m_swp, se_swp, na_swp, dr_swp, _, _ = run_survival(True, N=1000, T=30, burn_in=0, seed=seed + 50)
        diff = m_std - m_swp
        cse = np.sqrt(se_std ** 2 + se_swp ** 2)
        print(f"  seed {seed}: std={m_std:.5f}+/-{se_std:.5f} (alive {na_std}/1000)   "
              f"swp={m_swp:.5f}+/-{se_swp:.5f} (alive {na_swp}/1000)   diff={diff:+.5f} ({diff / cse:+.2f}sig)")
    print("  (historical values 0.13240/0.13761 used a continuum, endpoint-only, uncoupled protocol)")

    print("\n[Test C] Larger N, longer T=400 (safely below underflow regime), burn=100, 3 seeds each:")
    for seed in range(1, 4):
        m_std, se_std, na_std, dr_std, _, _ = run_survival(False, N=8000, T=400, burn_in=100, seed=seed)
        m_swp, se_swp, na_swp, dr_swp, _, _ = run_survival(True, N=8000, T=400, burn_in=100, seed=seed + 100)
        diff = m_std - m_swp
        cse = np.sqrt(se_std ** 2 + se_swp ** 2)
        print(f"  seed {seed}: std={m_std:.6f}+/-{se_std:.6f} (alive {na_std}/8000, death-rate/step={dr_std:.2e})   "
              f"swp={m_swp:.6f}+/-{se_swp:.6f} (alive {na_swp}/8000, death-rate/step={dr_swp:.2e})   diff={diff:+.6f} ({diff / cse:+.2f}sig)")
    print("  (paper's stated per-step death rate on discrete crack: 1.0e-3; cabarius's independent s* = 0.131745+/-0.000041)")
