#!/usr/bin/env sage
# Exact settlement-channel spectrum — a runnable proof of Theorem 3.12 (Paper I).
#
# The numpy audit (verify/occurrence_i_audit.py) certifies the channel spectrum
# NUMERICALLY at threshold 1e-12. This script upgrades that [C] certificate to an
# EXACT symbolic result: every basic Kraus operator L_z has entries in {0, ±1},
# so the 84-event channel Phi is an exact rational 256x256 matrix, and its
# characteristic polynomial factors exactly over Q. No tolerance, no threshold.
#
# It is self-contained: it builds the sedenion algebra S from the Cayley-Dickson
# doubling over Q from scratch (no dependency on `topographo` or data/kraus84.npz),
# passes the four mandatory gates of the Verification Protocol (Appendix), then
# proves the Theorem 3.12 spectrum by exact factorization.
#
# Run:
#   sage verify/occurrence_i_spectrum.sage
# or paste into SageMathCell (https://sagecell.sagemath.org), no install needed.
#
# What it proves (all exact over Q):
#   - Gates: composition, antisymmetry of L, quadratic x^2 = -e0, Moufang.
#   - The basic crack is exactly 84 elements (Definition 3.4).
#   - Forced equilibrium (1/84) sum M_z = I           (Theorem 3.6).
#   - The channel spectrum is exactly (1/7).{0, +-1, +-3, +-2sqrt3, +-7} with the
#     Theorem 3.12 multiplicities; 2sqrt3/7 appears as the exact factor x^2-12/49.
#   - trace(Phi) = 0 as a 256x256 matrix                (Section 3.7).
#   - Phi(L_e8) = -L_e8, the undamped parity            (Theorem 3.9b).
#   - 7 and 14 are the two fundamental G2 irreps        (Theorem 3.2, rep-theory).

# ---------------------------------------------------------------------------
# Cayley-Dickson algebra S = A_4 over Q  (gamma = -1 at each doubling).
# ---------------------------------------------------------------------------
def vadd(x, y): return [a + b for a, b in zip(x, y)]
def vsub(x, y): return [a - b for a, b in zip(x, y)]

def cd_conj(x):
    n = len(x)
    if n == 1:
        return [x[0]]
    h = n // 2
    return cd_conj(x[:h]) + [-t for t in x[h:]]

def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return [x[0] * y[0]]
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    # (a,b)(c,d) = (a c - conj(d) b,  d a + b conj(c))    [Baez convention]
    left = vsub(cd_mul(a, c), cd_mul(cd_conj(d), b))
    right = vadd(cd_mul(d, a), cd_mul(b, cd_conj(c)))
    return left + right

DIM = 16
def basis(j):
    v = [QQ(0)] * DIM
    v[j] = QQ(1)
    return v
E = [basis(j) for j in range(DIM)]

def norm2(x):
    return sum(t * t for t in x)

def Lmat(x):
    # Left multiplication: column j is x * e_j.
    return matrix(QQ, [[cd_mul(x, E[j])[i] for j in range(DIM)] for i in range(DIM)])

# ---------------------------------------------------------------------------
# Mandatory gates (Verification Protocol, Appendix). Exact over Q.
# ---------------------------------------------------------------------------
neg_e0 = [(-1 if k == 0 else 0) for k in range(DIM)]

gate_quadratic = all(cd_mul(E[i], E[i]) == neg_e0 for i in range(1, DIM))
gate_antisym = all((Lmat(E[i]) + Lmat(E[i]).T) == 0 for i in range(1, DIM))

# Composition ||x y|| = ||x|| ||y|| on a hull. Test on the octonion hull (idx 0..7)
# with deterministic rational octonions (composition fails on all of S, holds on hulls).
def oct_elt(coeffs):
    v = [QQ(0)] * DIM
    for i, c in enumerate(coeffs):
        v[i] = QQ(c)
    return v
xo = oct_elt([1, 2, -3, 0, 5, 0, -1, 4])
yo = oct_elt([-2, 1, 0, 3, 0, -4, 2, 1])
gate_composition = norm2(cd_mul(xo, yo)) == norm2(xo) * norm2(yo)

# Moufang (ab)(ca) = a((bc)a) on octonion basis (idx 0..7).
def moufang_ok(a, b, c):
    lhs = cd_mul(cd_mul(a, b), cd_mul(c, a))
    rhs = cd_mul(a, cd_mul(cd_mul(b, c), a))
    return lhs == rhs
gate_moufang = all(moufang_ok(E[i], E[j], E[k])
                   for i in range(8) for j in range(8) for k in range(8))

print("=== Mandatory gates (exact over Q) ===")
print("  composition ||xy||=||x||||y|| on octonion hull :", gate_composition)
print("  antisymmetry L_x + L_x^T = 0 for pure basis     :", gate_antisym)
print("  quadratic x^2 = -e0 for pure unit basis         :", gate_quadratic)
print("  Moufang (ab)(ca) = a((bc)a) on octonion basis   :", gate_moufang)
assert gate_composition and gate_antisym and gate_quadratic and gate_moufang, \
    "a mandatory gate failed -- algebra implementation is not trustworthy"

# ---------------------------------------------------------------------------
# Basic crack Sigma: basis-form zero divisors (Definition 3.4).
# z = w / sqrt(2) with w = e_a +- e_b; singular iff det L_w = 0 (scale-invariant).
# ---------------------------------------------------------------------------
crack = []
for a in range(DIM):
    for b in range(a + 1, DIM):
        for s in (1, -1):
            w = vadd(E[a], [s * t for t in E[b]])
            if Lmat(w).det() == 0:
                crack.append(w)
print("\n=== Basic crack ===")
print("  |Sigma| (basis-form zero divisors):", len(crack))
assert len(crack) == 84, f"expected 84 basic zero divisors, got {len(crack)}"

# ---------------------------------------------------------------------------
# Forced equilibrium (Theorem 3.6):  (1/84) sum_z M_z = I,  M_z = L_z^T L_z.
# With z = w/sqrt(2):  M_z = (1/2) L_w^T L_w.
# ---------------------------------------------------------------------------
Msum = matrix(QQ, DIM, DIM, 0)
for w in crack:
    Lw = Lmat(w)
    Msum += Lw.T * Lw
Mavg = Msum / (2 * len(crack))
print("\n=== Forced equilibrium (Theorem 3.6) ===")
print("  (1/84) sum M_z == I :", Mavg == identity_matrix(QQ, DIM))
assert Mavg == identity_matrix(QQ, DIM)

# ---------------------------------------------------------------------------
# Channel superoperator Phi on End(R^16), as an exact 256x256 rational matrix.
# Phi(X) = (1/84) sum_z L_z^T X L_z.  vec(A X B) = (B^T (x) A) vec(X),
# with A = L_z^T, B = L_z, and L_z = L_w/sqrt(2) contributing 1/2 per event.
# ---------------------------------------------------------------------------
Phi = matrix(QQ, 256, 256, 0)
for w in crack:
    LwT = Lmat(w).T
    Phi += LwT.tensor_product(LwT)
Phi = Phi / (2 * len(crack))

print("\n=== Channel spectrum (Theorem 3.12) — exact factorization ===")
cp = Phi.charpoly()
print("  charpoly factors as:")
print("   ", cp.factor())

x = cp.variables()[0]
expected = ((x - 1) * (x + 1)
            * (x - 3/7)^21 * (x + 3/7)^21
            * (x - 1/7)^42 * (x + 1/7)^42
            * x^100
            * (x^2 - 12/49)^14)
print("  matches (1/7).{0,+-1,+-3,+-2sqrt3,+-7} with Thm 3.12 mults :",
      cp == expected)
assert cp == expected, "channel spectrum does not match Theorem 3.12"

# 2sqrt3/7 is real and irrational; recorded as the exact quadratic x^2 - 12/49.
assert (2 * sqrt(3) / 7)^2 == QQ(12) / 49
print("  2sqrt3/7 certified as exact roots of x^2 - 12/49            :", True)

# trace of Phi as a 256x256 matrix is 0 (Section 3.7).
print("  trace(Phi) as 256x256 matrix == 0                          :", Phi.trace() == 0)
assert Phi.trace() == 0

# ---------------------------------------------------------------------------
# Parity (Theorem 3.9b):  Phi(L_e8) = -L_e8  (exact -1 eigenvalue, mult 1).
# ---------------------------------------------------------------------------
Le8 = Lmat(E[8])
vecLe8 = vector(QQ, [Le8[i, j] for j in range(DIM) for i in range(DIM)])  # column-stacked
img = Phi * vecLe8
print("\n=== Parity (Theorem 3.9b) ===")
print("  Phi(L_e8) == -L_e8 :", img == -vecLe8)
assert img == -vecLe8

# ---------------------------------------------------------------------------
# Representation theory (Theorem 3.2): 7 and 14 are the two fundamental G2 irreps,
# consistent with S = 1 (+) 1' (+) (7 (x) 2), dim 1+1+14 = 16.
# ---------------------------------------------------------------------------
print("\n=== G2 rep-theory sanity (Theorem 3.2) ===")
G2 = WeylCharacterRing("G2", style="coroots")
d10, d01 = G2(1, 0).degree(), G2(0, 1).degree()
fund = sorted([d10, d01])
print(f"  G2 fundamental irrep dims: {fund}  (expect [7, 14])")
assert fund == [7, 14]
print("  isotypic dim 1 + 1 + (7 x 2) = 16 :", 1 + 1 + 7 * 2 == 16)

print("\nALL EXACT CHECKS PASSED — Theorem 3.12 spectrum proved symbolically over Q.")
