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
#   - The basic crack is exactly 84 elements            (Definition 3.4).
#   - Master identity M_x = I + T_x; T_x sym & traceless (Theorem 3.5).
#   - Axis pair J=R_e8: J^2=-I, J^T=-J, [L_e8,R_e8]=0    (Theorem 3.9a).
#   - {L_e8, L_z} = 0 for every crack event             (Theorem 3.9b).
#   - Forced equilibrium (1/84) sum M_z = I; E_mu[T_z]=0 (Theorems 3.6, 3.10.5).
#   - Associative envelope = End(R^16) dim 256; Lie
#     algebra = so(16) dim 120, exact ranks over Q       (Theorem 3.8c,d).
#   - Second moment (1/84) sum z z^T = P_W / 14          (Theorem 3.13).
#   - The channel spectrum is exactly (1/7).{0, +-1, +-3, +-2sqrt3, +-7} with the
#     Theorem 3.12 multiplicities; 2sqrt3/7 appears as the exact factor x^2-12/49.
#   - trace(Phi) = 0 as a 256x256 matrix                (Section 3.7).
#   - Phi(L_e8) = -L_e8, the undamped parity            (Theorem 3.9b).
#   - Flicker: Phi(P_e0) = Phi(P_e8) = P_W/14           (Theorem 5.3).
#   - Occupancy balance M_{z e8} = M_z                  (Theorem 5.5).
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

def Rmat(x):
    # Right multiplication: column j is e_j * x.
    return matrix(QQ, [[cd_mul(E[j], x)[i] for j in range(DIM)] for i in range(DIM)])

def colvec(x):
    return matrix(QQ, DIM, 1, x)

def Tmat(x):
    # Alternator T_x = L_{x^2} - L_x^2 (Definition 3.1).
    return Lmat(cd_mul(x, x)) - Lmat(x) * Lmat(x)

def Mmat(x):
    # Metric operator M_x = L_x^T L_x.
    L = Lmat(x)
    return L.T * L

# ---------------------------------------------------------------------------
# Mandatory gates (Verification Protocol, Appendix). Exact over Q.
# ---------------------------------------------------------------------------
neg_e0 = [(-1 if k == 0 else 0) for k in range(DIM)]

# A quadratic identity is determined by its values on basis vectors and pair
# sums (polarization).  Use that spanning family rather than basis-only tests.
pure_sedenion_probes = [E[i] for i in range(1, DIM)] + [
    vadd(E[i], E[j]) for i in range(1, DIM) for j in range(i + 1, DIM)
]
gate_quadratic = all(
    cd_mul(v, v) == [(-norm2(v) if k == 0 else 0) for k in range(DIM)]
    for v in pure_sedenion_probes
)
gate_antisym = all((Lmat(E[i]) + Lmat(E[i]).T) == 0 for i in range(1, DIM))

# Composition is quadratic in each argument.  Basis vectors plus pair sums
# polarize both slots and therefore check the identity for all octonions.
oct_probes = [E[i] for i in range(8)] + [
    vadd(E[i], E[j]) for i in range(8) for j in range(i + 1, 8)
]
gate_composition = all(
    norm2(cd_mul(a, b)) == norm2(a) * norm2(b)
    for a in oct_probes for b in oct_probes
)

# Moufang is quadratic in a and linear in b,c.  Pair sums polarize the repeated
# a slot; basis vectors span the other two slots.
def moufang_ok(a, b, c):
    lhs = cd_mul(cd_mul(a, b), cd_mul(c, a))
    rhs = cd_mul(a, cd_mul(cd_mul(b, c), a))
    return lhs == rhs
gate_moufang = all(moufang_ok(a, E[j], E[k])
                   for a in oct_probes for j in range(8) for k in range(8))

print("=== Mandatory gates (exact over Q) ===")
print("  composition ||xy||=||x||||y|| on octonion hull :", gate_composition)
print("  antisymmetry L_x + L_x^T = 0 for all pure x      :", gate_antisym)
print("  quadratic x^2 = -||x||^2 e0 (polarized)         :", gate_quadratic)
print("  Moufang (ab)(ca) = a((bc)a) (polarized)         :", gate_moufang)
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
# Tier 3 -- exact algebraic identities (replace the "0.0" / "1e-15" certificates).
# ---------------------------------------------------------------------------
I16 = identity_matrix(QQ, DIM)
Le8 = Lmat(E[8])
Re8 = Rmat(E[8])

print("\n=== Exact algebraic identities ===")

# The homogeneous form is M_x = ||x||^2 I + T_x.  Checking the same
# polarization-spanning family proves it for every pure x; unit x gives the
# paper's M_x = I + T_x statement.
master = all(Mmat(v) == norm2(v) * I16 + Tmat(v) for v in pure_sedenion_probes)
t_sym_tr = all(Tmat(v) == Tmat(v).T and Tmat(v).trace() == 0
               for v in pure_sedenion_probes)
print("  master identity M_x = ||x||^2 I + T_x (polarized):", master)
print("  T_x symmetric and traceless (corollary)          :", t_sym_tr)
assert master and t_sym_tr

# Axis pair J = R_e8 (Theorem 3.9a): J^2 = -I, J^T = -J, [L_e8, R_e8] = 0.
axis = (Re8 * Re8 == -I16) and (Re8.T == -Re8) and (Le8 * Re8 == Re8 * Le8)
print("  J^2=-I, J^T=-J, [L_e8,R_e8]=0 (Thm 3.9a)          :", axis)
assert axis

# Clifford anticommutation {L_e8, L_z} = 0 for every crack event (Theorem 3.9b).
clifford = all(Le8 * Lmat(w) + Lmat(w) * Le8 == 0 for w in crack)
print("  {L_e8, L_z} = 0 for all 84 events (Thm 3.9b)      :", clifford)
assert clifford

# ---------------------------------------------------------------------------
# Forced equilibrium (Theorem 3.6):  (1/84) sum_z M_z = I,  M_z = L_z^T L_z.
# With z = w/sqrt(2):  M_z = (1/2) L_w^T L_w.  (Hence E_mu[T_z] = 0, Thm 3.10.5.)
# ---------------------------------------------------------------------------
Msum = matrix(QQ, DIM, DIM, 0)
for w in crack:
    Lw = Lmat(w)
    Msum += Lw.T * Lw
Mavg = Msum / (2 * len(crack))
print("\n=== Forced equilibrium (Theorem 3.6) ===")
print("  (1/84) sum M_z == I :", Mavg == I16)
print("  hence E_mu[T_z] == 0 (Thm 3.10.5 mean) :", (Mavg - I16) == 0)
assert Mavg == I16

# ---------------------------------------------------------------------------
# Tier 4 -- exact structure dimensions via rank over Q (Theorem 3.8c,d).
# numpy computes these as tolerance-dependent numerical ranks; over Q they are
# exact integer dimensions of the generated associative / Lie algebras.
# ---------------------------------------------------------------------------
class Span:
    """Row-echelon accumulator; add(M) is True iff flat(M) is new-independent."""
    def __init__(self):
        self.rows = []   # (pivot_col, normalized row)
        self.mats = []
    def add(self, M):
        v = M.list()
        for p, r in self.rows:
            if v[p] != 0:
                f = v[p]
                v = [v[k] - f * r[k] for k in range(len(v))]
        for p in range(len(v)):
            if v[p] != 0:
                inv = 1 / v[p]
                self.rows.append((p, [c * inv for c in v]))
                self.mats.append(M)
                return True
        return False
    def dim(self):
        return len(self.rows)

gens = [Lmat(E[i]) for i in range(1, DIM)]   # {L_x : x pure basis}

def assoc_envelope_dim(gens):
    S = Span()
    S.add(I16)
    for g in gens:
        S.add(g)
    frontier = list(S.mats)
    while frontier:
        nxt = []
        for f in frontier:
            for g in gens:
                p = g * f
                if S.add(p):
                    nxt.append(p)
        frontier = nxt
    return S.dim()

def lie_closure_dim(gens):
    S = Span()
    for g in gens:
        S.add(g)
    changed = True
    while changed:
        changed = False
        cur = list(S.mats)
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                if S.add(cur[i] * cur[j] - cur[j] * cur[i]):
                    changed = True
    return S.dim()

print("\n=== Structure dimensions (Theorem 3.8c,d) — exact rank over Q ===")
env_dim = assoc_envelope_dim(gens)
lie_dim = lie_closure_dim(gens)
print(f"  associative envelope of {{L_x}} = End(R^16), dim  : {env_dim}  (expect 256)")
print(f"  Lie algebra generated by {{L_x}} = so(16), dim    : {lie_dim}  (expect 120)")
assert env_dim == 256 and lie_dim == 120

# ---------------------------------------------------------------------------
# Second moment (Theorem 3.13):  (1/84) sum_z z z^T = P_W / 14.
# With z = w/sqrt(2):  z z^T = (1/2) w w^T.
# ---------------------------------------------------------------------------
P_e0 = colvec(E[0]) * colvec(E[0]).T
P_e8 = colvec(E[8]) * colvec(E[8]).T
P_W = I16 - P_e0 - P_e8
Zsum = matrix(QQ, DIM, DIM, 0)
for w in crack:
    cv = colvec(w)
    Zsum += cv * cv.T
Zsum = Zsum / (2 * len(crack))
print("\n=== Second moment (Theorem 3.13) ===")
print("  (1/84) sum z z^T == P_W / 14 :", Zsum == P_W / 14)
assert Zsum == P_W / 14

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
def veccol(M):
    return vector(QQ, [M[i, j] for j in range(DIM) for i in range(DIM)])  # column-stacked
def unvec(v):
    return matrix(QQ, DIM, DIM, lambda i, j: v[i + DIM * j])

vecLe8 = veccol(Le8)
print("\n=== Parity (Theorem 3.9b) ===")
print("  Phi(L_e8) == -L_e8 :", Phi * vecLe8 == -vecLe8)
assert Phi * vecLe8 == -vecLe8

# ---------------------------------------------------------------------------
# Flicker (Theorem 5.3):  Phi(P_e0) = Phi(P_e8) = P_W/14 (spine->pencil collapse).
# ---------------------------------------------------------------------------
f0 = unvec(Phi * veccol(P_e0))
f8 = unvec(Phi * veccol(P_e8))
print("\n=== Flicker (Theorem 5.3) ===")
print("  Phi(P_e0) == P_W/14 :", f0 == P_W / 14)
print("  Phi(P_e8) == P_W/14 :", f8 == P_W / 14)
assert f0 == P_W / 14 and f8 == P_W / 14

# ---------------------------------------------------------------------------
# Occupancy balance (Theorem 5.5): the involution z -> z e8 preserves M_z
# pointwise, so e0 and e8 carry equal stationary occupancy.
# ---------------------------------------------------------------------------
occ = all(Mmat(cd_mul(w, E[8])) == Mmat(w) for w in crack)
print("\n=== Occupancy balance (Theorem 5.5) ===")
print("  M_{z e8} == M_z for all 84 events :", occ)
assert occ

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
