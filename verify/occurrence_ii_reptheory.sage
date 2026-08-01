#!/usr/bin/env sage
# occurrence_ii_reptheory.sage
# Comprehensive EXACT / REPRESENTATION-THEORY verification for Occurrence
# Theory II, using SageMath. This is the exact/symbolic complement to the
# floating-point numpy audit (verify/occurrence_ii_audit.py): it does the
# things numpy cannot — exact linear algebra over Q and honest G2/SU(3)
# representation theory.
#
# Independent of `topographo`. Loads only data/kraus84.npz. Exit 0 iff all
# assertions pass.
#
#   sage verify/occurrence_ii_reptheory.sage ; echo $?
#
# Sections:
#   1. Exact spectrum over Q            (upgrades Theorem 3.2 to theorem-grade)
#   2. G2-module structure of End(R^16) (§10.1: names every eigenspace)
#   3. G2 -> SU(3) branching            (Conjecture C3)
#   4. Design-Theorem invariants        (§10.2 / OT Theorem 3.13)
#
# KEY FINDING (section 2/3): the dim-14 irrational (p) sector is 7 + 7 as a
# G2-module, NOT the adjoint g2. The dimension coincidence 14 = dim g2 is just
# that — a coincidence. The four adjoint (14) copies live in the +/-3/7 and 0
# sectors. So the p-sector branches under SU(3) as 2*(3 + 3bar + 1), with NO
# gluon octet; the octet 8 appears only where an adjoint 14 does (the +/-3/7
# sectors). This bears directly on Conjecture C3.

import os
import hashlib
import numpy as np

FAIL = []
def check(name, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)

_here = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(_here, "..", "data", "kraus84.npz"))
K, mu = data["K"], data["mu"]
check("artifact shapes are K=(84,16,16), mu=(84,)",
      K.shape == (84, 16, 16) and mu.shape == (84,))
check("artifact arrays are finite", np.isfinite(K).all() and np.isfinite(mu).all())
if FAIL:
    raise RuntimeError("artifact integrity checks failed: " + "; ".join(FAIL))
N, D = K.shape[0], K.shape[1]

# ===========================================================================
# 1. EXACT SPECTRUM OVER Q
# The Kraus entries are 0, +/-1/sqrt2, so sqrt2*K is integral and
# S = sum_a mu_a K_a (x) K_a = (1/168) sum_a (sqrt2 K_a)(x)(sqrt2 K_a) lands in Q.
# ===========================================================================
print("=" * 70)
print("1. EXACT SPECTRUM OVER Q  (Theorem 3.2, theorem-grade)")
print("=" * 70)
Kt = np.rint(np.sqrt(2) * K).astype(int)
reconstruction_error = float(np.max(np.abs(np.sqrt(2) * K - Kt)))
check("sqrt2*K reconstructs integral entries", reconstruction_error < 1e-12,
      detail=f"max error {reconstruction_error:.3e}")
check("reconstructed entries lie in {-1,0,1}", set(np.unique(Kt)).issubset({-1, 0, 1}))
weight_error = float(np.max(np.abs(mu - np.full(N, 1 / 84))))
check("released weights are uniformly 1/84", weight_error < 1e-15,
      detail=f"max error {weight_error:.3e}")
if FAIL:
    raise RuntimeError("exact reconstruction prerequisites failed: " + "; ".join(FAIL))
print("  reconstructed Kraus SHA-256:", hashlib.sha256(Kt.tobytes()).hexdigest())
Ssum = sum(np.kron(Kt[a], Kt[a]) for a in range(N))       # integer 256x256
S = Matrix(QQ, Ssum.tolist()) / 168
check("superoperator S is symmetric over Q without repair", S.is_symmetric())

x = polygen(QQ)
# claimed spectrum: 0, +/-1/7, +/-3/7, +/-1, and the irrational pair +/-2sqrt3/7.
# (2sqrt3/7)^2 = 12/49, so the irrational pair are the roots of x^2 - 12/49.
minpoly = x * (x - 1/7) * (x + 1/7) * (x - 3/7) * (x + 3/7) * (x - 1) * (x + 1) * (x^2 - 12/49)
check("m(S) = 0 for the claimed spectrum (spectrum is exactly these roots)",
      minpoly(S) == 0)

I256 = identity_matrix(QQ, 256)
mult = {}
for lam, name in [(0, "0"), (1/7, "1/7"), (-1/7, "-1/7"), (3/7, "3/7"),
                  (-3/7, "-3/7"), (1, "1"), (-1, "-1")]:
    mult[name] = 256 - (S - lam * I256).rank()
irr = 256 - (S * S - (12/49) * I256).rank()               # counts both +/-2sqrt3/7
Q3.<r3> = QuadraticField(3)
SQ3 = S.change_ring(Q3)
irr_pos = 256 - (SQ3 - (2*r3/7) * identity_matrix(Q3, 256)).rank()
irr_neg = 256 - (SQ3 + (2*r3/7) * identity_matrix(Q3, 256)).rank()
print("  exact multiplicities (via rank over Q):")
for k in ["1", "3/7", "1/7", "0", "-1/7", "-3/7", "-1"]:
    print(f"    {k:>12} : {mult[k]}")
print(f"    {'+/-2sqrt3/7':>12} : {irr}  ({irr_pos} positive, {irr_neg} negative)")
check("multiplicities are {1,1,21,21,42,42,100} + 28 (=14+14)",
      sorted(mult[k] for k in ["1", "-1", "3/7", "-3/7", "1/7", "-1/7", "0"]) == [1, 1, 21, 21, 42, 42, 100]
      and irr == 28 and irr_pos == 14 and irr_neg == 14)
check("multiplicities sum to 256",
      sum(mult[k] for k in ["1", "-1", "3/7", "-3/7", "1/7", "-1/7", "0"]) + irr == 256)

# ===========================================================================
# 2. G2-MODULE STRUCTURE OF End(R^16)   (§10.1: name every eigenspace)
# G2 = Aut(O) acts on the sedenions S = O + O.l diagonally: fixes e0, e8;
# acts as the standard rep 7 on e1..e7 and on e9..e15. So R^16 = 2*(1) + 2*(7)
# and End(R^16) decomposes into the irreps {1, 7, 14, 27} only. We build the
# concrete G2 action, then read off each Phi-eigenspace as a G2-module.
# ===========================================================================
print("\n" + "=" * 70)
print("2. NUMERICAL G2-MODULE IDENTIFICATION OF End(R^16)  (§10.1)")
print("=" * 70)

def cd_conj(v):
    n = len(v)
    if n == 1:
        return v.copy()
    h = n // 2
    return np.concatenate([cd_conj(v[:h]), -v[h:]])
def cd_mul(u, v):
    n = len(u)
    if n == 1:
        return np.array([u[0] * v[0]])
    h = n // 2
    a, b, c, d = u[:h], u[h:], v[:h], v[h:]
    return np.concatenate([cd_mul(a, c) - cd_mul(cd_conj(d), b),
                           cd_mul(d, a) + cd_mul(b, cd_conj(c))])
Omul = np.zeros((8, 8, 8))
for i in range(8):
    for j in range(8):
        ei = np.zeros(8); ei[i] = 1; ej = np.zeros(8); ej[j] = 1
        Omul[:, i, j] = cd_mul(ei, ej)

# g2 = Der(O): 8x8 D with D e0 = 0 and D(xy) = (Dx)y + x(Dy)
rows = []
for i in range(8):
    for j in range(8):
        for c in range(8):
            r = np.zeros((8, 8))
            for k in range(8): r[c, k] += Omul[k, i, j]
            for k in range(8): r[k, i] -= Omul[c, k, j]
            for k in range(8): r[k, j] -= Omul[c, i, k]
            rows.append(r.reshape(-1))
for c in range(8):
    r = np.zeros((8, 8)); r[c, 0] = 1; rows.append(r.reshape(-1))
_, sv, vt = np.linalg.svd(np.array(rows))
der_basis = [vt[t].reshape(8, 8) for t in range(int(np.sum(sv > 1e-9)), 64)]
check("dim Der(O) = 14 (= dim g2)", len(der_basis) == 14)

np.random.seed(20260711)
def random_g():
    A = sum(cc * b for cc, b in zip(np.random.randn(len(der_basis)), der_basis))
    g8 = np.array(matrix(RDF, A.tolist()).exp())          # in G2 < SO(8), fixes e0
    g16 = np.zeros((16, 16)); g16[:8, :8] = g8; g16[8:, 8:] = g8
    c7 = np.trace(g8) - 1                                  # char of the 7
    c7s = np.trace(g8 @ g8) - 1                            # char of the 7 at g^2
    # 14 = Lambda^2(7) - 7 ; 27 = Sym^2(7) - 1
    chars = {1: 1.0, 7: c7, 14: (c7**2 - c7s)/2 - c7, 27: (c7**2 + c7s)/2 - 1}
    return g16, chars

gtest, _ = random_g()
autoerr = max(np.linalg.norm(
    cd_mul(gtest[:8, :8] @ np.eye(8)[i], gtest[:8, :8] @ np.eye(8)[j]) - gtest[:8, :8] @ Omul[:, i, j])
    for i in range(8) for j in range(8))
check("exp(g2) elements are genuine octonion automorphisms", autoerr < 1e-9)

Sf = np.array(S, dtype=float)

# sanity check: Phi actually commutes with the concrete G2 action.
# R(g) X = g X g^T, i.e. on vec-space R(g) = g16 (x) g16; a symmetry iff
# [S, g16 (x) g16] = 0. Fast confidence boost that G2 really is a symmetry.
g16chk, _ = random_g()
Rg = np.kron(g16chk, g16chk)
commerr = float(np.max(np.abs(Sf @ Rg - Rg @ Sf)))
check("Phi commutes with the concrete G2 action ([S, g(x)g] = 0)",
      commerr < 1e-9, detail=f"||[S, R(g)]|| = {commerr:.1e}")

w, Vf = np.linalg.eigh(Sf)
groups = {}
for idx, ev in enumerate(w):
    groups.setdefault(round(float(ev), 4), []).append(idx)

gs = [random_g() for _ in range(6)]
irrep_dims = np.array([1, 7, 14, 27])
def decomp_str(m):
    return " + ".join((f"{c}*{d}" if c > 1 else f"{d}") for c, d in zip(m, [1, 7, 14, 27]) if c) or "0"

print("  Phi-eigenspace as a G2-module (irreps 1, 7, 14, 27):")
total = np.zeros(4, dtype=int)
psector = None
for key in sorted(groups, reverse=True):
    cols = groups[key]; B = Vf[:, cols]; dimE = len(cols)
    A = []; rhs = []
    for g16, chars in gs:
        acc = sum(np.sum(B[:, t] * (g16 @ B[:, t].reshape(16, 16) @ g16.T).reshape(-1))
                  for t in range(dimE))
        A.append([chars[1], chars[7], chars[14], chars[27]]); rhs.append(acc)
    A.append(list(irrep_dims)); rhs.append(dimE)
    m, *_ = np.linalg.lstsq(np.array(A), np.array(rhs), rcond=None)
    mi = np.rint(m).astype(int)
    good = (int(mi @ irrep_dims) == dimE) and np.max(np.abs(np.array(A) @ mi - np.array(rhs))) < 1e-4 and np.all(mi >= 0)
    total += mi
    tag = ""
    if abs(abs(key) - 2 * np.sqrt(3) / 7) < 1e-3:
        tag = "  <- irrational (p) sector"
        if key > 0:
            psector = tuple(int(v) for v in mi)
    print(f"    lambda={key:+.4f}  dim {dimE:>3}  =  {decomp_str(mi):<22}{tag}")
    check(f"seeded numerical fit at lambda={key:+.4f} is integral", good)

check("End(R^16) totals = 8*1 + 12*7 + 4*14 + 4*27", tuple(int(v) for v in total) == (8, 12, 4, 4))
check("the p-sector is 7 + 7 (NOT the adjoint 14)", psector == (0, 2, 0, 0),
      detail=f"got {decomp_str(psector)}")

# ===========================================================================
# 3. G2 -> SU(3) BRANCHING   (Conjecture C3)
# ===========================================================================
print("\n" + "=" * 70)
print("3. G2 -> SU(3) BRANCHING  (Conjecture C3)")
print("=" * 70)
G2 = WeylCharacterRing("G2", style="coroots")
A2 = WeylCharacterRing("A2", style="coroots")
rule = branching_rule("G2", "A2", "extended")
seven = G2(1, 0) if G2(1, 0).degree() == 7 else G2(0, 1)
adjoint = G2(0, 1) if G2(0, 1).degree() == 14 else G2(1, 0)

def branch_dims(chi):
    b = chi.branch(A2, rule=rule)
    return sorted([irr.degree() for irr, m in zip(b.monomials(), b.coefficients()) for _ in range(m)])

d7 = branch_dims(seven)
d14 = branch_dims(adjoint)
print(f"  7            -> {d7}   (3 + 3bar + 1)")
print(f"  14 (adjoint) -> {d14}   (8 + 3 + 3bar)")
check("7 branches as 3 + 3bar + 1", d7 == [1, 3, 3])
check("the adjoint 14 branches as 8 + 3 + 3bar (NOT 8 + 6)", d14 == [3, 3, 8])
d_psector = sorted(d7 + d7)                                # the actual p-sector is 7 + 7
print(f"  p-sector  = 7 + 7  -> {d_psector}   (2*(3 + 3bar + 1); NO octet)")
check("p-sector (7+7) branches as 2*(3 + 3bar + 1), containing no octet",
      d_psector == [1, 1, 3, 3, 3, 3])
# where the gauge story now lives: the +/-3/7 sectors are 7 + 14 as G2-modules.
d_37 = sorted(d7 + d14)
print(f"  +/-3/7    = 7 + 14 -> {d_37}   (8 + 2*3 + 2*3bar + 1; CONTAINS the octet)")
check("+/-3/7 sector (7+14) contains exactly one gluon octet 8", d_37.count(8) == 1)

# ===========================================================================
# 4. DESIGN-THEOREM INVARIANTS   (§10.2 / OT Theorem 3.13)
# ===========================================================================
print("\n" + "=" * 70)
print("4. DESIGN-THEOREM INVARIANTS  (§10.2)")
print("=" * 70)
W = seven + seven
symW = W.symmetric_square()
inv = sum(co for mon, co in zip(symW.monomials(), symW.coefficients()) if mon.degree() == 1)
print(f"  dim of G2-invariant symmetric forms on W = 7 + 7 : {inv}")
check("that space is 3-dim (so E[zz^T]=P_W/14 is NOT forced by symmetry alone)", inv == 3)

# ===========================================================================
print("\n" + "=" * 70)
if FAIL:
    print(f"RESULT: FAIL — {len(FAIL)} check(s): " + "; ".join(FAIL))
    import sys
    sys.exit(1)
print("RESULT: PASS — exact spectrum/branching and seeded numerical G2 fits hold.")
print("=" * 70)
