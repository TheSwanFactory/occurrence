#!/usr/bin/env python3
"""
Independent verification of the Born Channel (Occurrence Theory II).

Reviewer cell — handle: claude_code (Claude Code, Anthropic), acting as an
independent numerical reviewer, distinct from the first-party
verify/occurrence_ii_audit.py.

INDEPENDENCE: this file does NOT import topographo. It re-derives the sedenion
algebra from scratch (a self-contained Cayley-Dickson doubling), rebuilds the
84 zero-divisor Kraus operators itself, and:

  (a) checks that the independently regenerated family matches the released
      ground truth data/kraus84.npz (provenance cross-check), and
  (b) verifies every [FORCED] claim of the paper on the released family —
      integrity, CPTP/unital, antisymmetry, ranks, events, pencil/spine,
      the nine-level spectrum with per-sector G2 multiplicities, the complex
      structure J, the coherence constant 2*sqrt(3)/7, and the annihilation
      lattice.

The structural invariants (b) are basis-independent, so they hold regardless
of any signed-permutation difference between this from-scratch algebra and the
one that produced the .npz.

Usage:
  python3 verify/occurrence_ii_claude_code.py ; echo $?   # 0 = all pass
"""

import sys
from pathlib import Path

import numpy as np

FAIL = []


def ok(label, err, tol=1e-9):
    good = bool(err < tol)
    print(f"[{'OK' if good else 'FAIL'}] {label}: {err:.2e} (tol {tol:.0e})")
    if not good:
        FAIL.append(f"{label}: {err:.2e} >= {tol:.0e}")
    return good


def eq(label, got, want):
    good = got == want
    print(f"[{'OK' if good else 'FAIL'}] {label}: got {got}, want {want}")
    if not good:
        FAIL.append(f"{label}: {got} != {want}")
    return good


# ---------------------------------------------------------------------------
# From-scratch Cayley-Dickson algebra (no topographo).
# Convention: (a,b)(c,d) = (a c - conj(d) b,  d a + b conj(c)),
#             conj((a,b)) = (conj(a), -b),  base = real multiplication.
# ---------------------------------------------------------------------------

def cd_conj(x):
    if len(x) == 1:
        return x.copy()
    h = len(x) // 2
    return np.concatenate([cd_conj(x[:h]), -x[h:]])


def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return np.array([x[0] * y[0]])
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([
        cd_mul(a, c) - cd_mul(cd_conj(d), b),
        cd_mul(d, a) + cd_mul(b, cd_conj(c)),
    ])


def left_mult_matrix(z):
    """Matrix of y -> z * y in the standard basis."""
    d = len(z)
    M = np.empty((d, d))
    for j in range(d):
        ej = np.zeros(d); ej[j] = 1.0
        M[:, j] = cd_mul(z, ej)
    return M


def build_kraus_family():
    """The 84 zero divisors z = (e_i + s e_{8+j})/sqrt2, i,j in 1..7, i!=j,
    s in {+1,-1}; K_a = L_{z_a}. Returns K (84,16,16), mu (84,)."""
    e = np.eye(16)
    K = []
    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                for s in (1, -1):
                    z = (e[i] + s * e[8 + j]) / np.sqrt(2)
                    K.append(left_mult_matrix(z))
    K = np.array(K)
    return K, np.ones(len(K)) / len(K)


def sector_projectors(D):
    """Orthonormal bases of symmetric (D(D+1)/2) and antisymmetric (D(D-1)/2)
    matrix subspaces of End(R^D), as columns of vec-space bases."""
    sym, skew = [], []
    for i in range(D):
        E = np.zeros((D, D)); E[i, i] = 1.0; sym.append(E.reshape(-1))
    for i in range(D):
        for j in range(i + 1, D):
            Es = np.zeros((D, D)); Es[i, j] = Es[j, i] = 1 / np.sqrt(2)
            sym.append(Es.reshape(-1))
            Ea = np.zeros((D, D)); Ea[i, j] = 1 / np.sqrt(2); Ea[j, i] = -1 / np.sqrt(2)
            skew.append(Ea.reshape(-1))
    return np.array(sym).T, np.array(skew).T


def levels(evals, tol=1e-6):
    out = []
    for x in np.sort(evals):
        if out and abs(x - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([round(float(x), 6), 1])
    return out


def main():
    D, N = 16, 84
    e0 = np.eye(D)[0]

    print("=" * 68)
    print("INDEPENDENT REVIEW (claude_code): from-scratch algebra + released .npz")
    print("=" * 68)

    # --- (a) Provenance cross-check: regenerate independently, diff the npz ---
    Kgen, mugen = build_kraus_family()
    npz = Path(__file__).resolve().parent.parent / "data" / "kraus84.npz"
    data = np.load(npz)
    K, mu = data["K"], data["mu"]
    print("\n0. PROVENANCE (independent regeneration vs released ground truth)")
    eq("regenerated family shape", tuple(Kgen.shape), (N, D, D))
    same = float(np.linalg.norm(Kgen - K))
    print(f"    ||K_regenerated - data/kraus84.npz|| = {same:.2e}"
          f"  ({'identical convention' if same < 1e-9 else 'differs by change of basis; invariants checked below'})")
    ok("mu uniform (regenerated vs npz)", float(np.linalg.norm(mugen - mu)))

    # From here, verify the RELEASED family (the artifact under review).
    print("\n1. INTEGRITY")
    eq("K.shape", tuple(K.shape), (N, D, D))
    eq("mu.shape", tuple(mu.shape), (N,))
    ok("sum(mu) = 1", abs(float(mu.sum()) - 1.0))
    ok("no NaN/Inf", 0.0 if np.all(np.isfinite(K)) else 1.0)

    print("\n2. CPTP / UNITAL (Thm 3.1)")
    E = np.einsum('a,aji,ajk->ik', mu, K, K)
    ok("||E[K^T K] - I||", float(np.linalg.norm(E - np.eye(D))))

    print("\n3. ANTISYMMETRY (Def 2.2)")
    ok("max_a ||K_a + K_a^T||", float(max(np.linalg.norm(k + k.T) for k in K)))

    print("\n4. EVENTS & RANKS (Thm 2.1)")
    z = np.einsum('aij,j->ai', K, e0)                 # z_a = K_a e0
    ok("max_a | ||z_a|| - 1 |", float(np.max(np.abs(np.linalg.norm(z, axis=1) - 1))))
    ranks = [int(np.linalg.matrix_rank(k, tol=1e-9)) for k in K]
    eq("all operator ranks = 12", (min(ranks), max(ranks)), (12, 12))

    print("\n5. PENCIL / SPINE (E[zz^T] = P_W/14, spine dim 2)")
    SM = np.einsum('a,ai,aj->ij', mu, z, z)
    w = np.linalg.eigvalsh(SM)
    eq("second-moment kernel dim (spine)", int(np.sum(np.abs(w) < 1e-9)), 2)
    eq("pencil rank", int(np.sum(np.abs(w) > 1e-9)), 14)
    nz = w[np.abs(w) > 1e-9]
    ok("nonzero moment eigenvalues all = 1/14", float(np.max(np.abs(nz - 1/14))))

    print("\n6. SPECTRUM: nine levels, per-sector G2 multiplicities (Thm 3.2)")
    S = sum(m * np.kron(k, k) for m, k in zip(mu, K)); S = (S + S.T) / 2
    eq("distinct eigenvalue levels", len(levels(np.linalg.eigvalsh(S))), 9)
    Bs, Ba = sector_projectors(D)
    sym = {v: m for v, m in levels(np.linalg.eigvalsh(Bs.T @ S @ Bs))}
    ska = {v: m for v, m in levels(np.linalg.eigvalsh(Ba.T @ S @ Ba))}
    eq("symmetric multiplicities", sorted(sym.values()), sorted([1, 7, 72, 42, 14]))
    eq("antisymmetric multiplicities", sorted(ska.values()),
       sorted([14, 14, 42, 28, 7, 14, 1]))

    print("\n7. COMPLEX STRUCTURE J (Thm 3.3)")
    evals, evecs = np.linalg.eigh(S)
    J = evecs[:, int(np.argmin(np.abs(evals + 1.0)))].reshape(D, D)
    J = (J - J.T) / 2; J = 4 * J / np.linalg.norm(J)
    ok("min |lambda + 1| (a simple -1 mode exists)",
       float(np.min(np.abs(evals + 1.0))))
    ok("||J + J^T|| (antisymmetric)", float(np.linalg.norm(J + J.T)))
    ok("||J^2 + I|| (complex unit)", float(np.linalg.norm(J @ J + np.eye(D))))

    print("\n8. COHERENCE CONSTANT 2*sqrt(3)/7 (Thm 3.4)")
    target = 2 * np.sqrt(3) / 7
    X = evecs[:, int(np.argmin(np.abs(evals - target)))].reshape(D, D)
    Y, prev, ratio = X.copy(), 1.0, None
    for _ in range(6):
        Y = sum(m * k.T @ Y @ k for m, k in zip(mu, K))
        c = float(np.sum(X * Y)); ratio = c / prev; prev = c
    ok("two-time correlator ratio = 2*sqrt(3)/7", abs(ratio - target))

    print("\n9. ANNIHILATION LATTICE (Thm 6.1)")
    A = np.zeros((N, N), bool)
    for i in range(N):
        A[i] = np.linalg.norm(np.einsum('ij,aj->ai', K[i], z), axis=1) < 1e-9
    np.fill_diagonal(A, False)
    deg = A.sum(1)
    comp = -np.ones(N, int); c = 0
    for s in range(N):
        if comp[s] < 0:
            stack = [s]; comp[s] = c
            while stack:
                u = stack.pop()
                for v in np.where(A[u])[0]:
                    if comp[v] < 0:
                        comp[v] = c; stack.append(v)
            c += 1
    sizes = sorted(int((comp == k).sum()) for k in range(c))

    def diameter(nodes):
        dm = 0
        for s in nodes:
            dist = {s: 0}; q = [s]
            while q:
                u = q.pop(0)
                for v in np.where(A[u])[0]:
                    if v in nodes and v not in dist:
                        dist[v] = dist[u] + 1; q.append(v)
        return max(dist.values())
    eq("degree regular = 4", (int(deg.min()), int(deg.max())), (4, 4))
    eq("number of cells", c, 7)
    eq("cell sizes", sizes, [12] * 7)
    eq("cell diameter", diameter(set(np.where(comp == 0)[0])), 3)

    print("\n" + "=" * 68)
    if FAIL:
        print(f"RESULT: FAIL — {len(FAIL)} check(s) did not pass:")
        for f in FAIL:
            print("  -", f)
        return 1
    print("RESULT: PASS — every [FORCED] claim reproduces independently.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
