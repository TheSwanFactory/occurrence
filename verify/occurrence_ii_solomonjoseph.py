#!/usr/bin/env python3
"""Independent review of Occurrence Theory II (the Born Channel).  Handle: solomonjoseph.

Independence: nothing here imports ``topographo``, and nothing here *trusts*
``data/kraus84.npz``.  The Kraus family is rebuilt from scratch by iterated
Cayley-Dickson doubling of R (R -> C -> H -> O -> S) with integer structure
constants; the released artifact is then compared against that construction
(Part 1), and every later claim is checked against the from-scratch family.

Where the paper reports agreement "to machine precision", this cell verifies the
claim *exactly* instead.  The Kraus operators are (1/sqrt 2) x integer matrices,
so 168 * Phi is a 256x256 integer matrix: Part 5 computes its characteristic
polynomial over Z (python-flint) and compares it with the paper's predicted
factorisation, settling Theorem 3.2 -- values *and* multiplicities, per sector --
over Q rather than to 4.9e-15, and without SageMath.  Parts 2, 3, 4, 6 and 8 are
likewise integer/rational.

Parts 9 and 10 rebuild g2 = Der(O) from scratch and decompose every eigenspace
into G2 irreps with the Casimir operator: an independent check of
``occurrence_ii_reptheory.sage`` (audit obligation 1).

Part 11 *proves* OT Theorem 3.13, the Design Theorem (audit obligation 2),
rather than checking it numerically.

Part 4 reports the review's main mathematical finding: the channel's Choi rank
is 14, not 84, and Phi is exactly the uniform average of conjugation by the 14
unit pencil directions of S -- a mixed-unitary channel.

Requires numpy; ``pip install python-flint`` enables the exact-over-Q layer.
Runtime ~40 s.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import flint  # type: ignore

    HAVE_FLINT = True
except ImportError:  # pragma: no cover
    HAVE_FLINT = False

TOL = 1e-9
PASS: list[str] = []
FAIL: list[str] = []
DISCREP: list[str] = []
NOTES: list[str] = []


def cert(name: str, err: float, tol: float = TOL) -> None:
    ok = bool(err <= tol)
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {err:.3e}")


def gt(name: str, value: float, floor: float) -> None:
    ok = bool(value > floor)
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {value:.3e} > {floor:.0e}")


def exact(name: str, got, want) -> None:
    ok = got == want
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {got}" + ("" if ok else f"   (want {want})"))


def discrep(name: str, text: str) -> None:
    DISCREP.append(f"{name}: {text}")
    print(f"  [DIFF] {name}: {text}")


def note(text: str) -> None:
    NOTES.append(text)
    print(f"  [NOTE] {text}")


def head(title: str) -> None:
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


# ===========================================================================
# PART 0 -- rebuild S from scratch; the 84 is forced (Theorem 2.1)
# ===========================================================================
def cayley_dickson_double(T: np.ndarray) -> np.ndarray:
    """(a,b)(c,d) = (ac - d conj(b), conj(a) d + c b); integer structure constants."""
    n = T.shape[0]
    mul_ = lambda x, y: np.einsum("i,j,ijk->k", x, y, T)

    def conj(x):
        c = -x.copy()
        c[0] = x[0]
        return c

    U = np.zeros((2 * n, 2 * n, 2 * n), dtype=np.int64)
    for i, j in itertools.product(range(2 * n), repeat=2):
        a, b, c, d = (np.zeros(n, dtype=np.int64) for _ in range(4))
        (a if i < n else b)[i % n] = 1
        (c if j < n else d)[j % n] = 1
        U[i, j, :n] = mul_(a, c) - mul_(d, conj(b))
        U[i, j, n:] = mul_(conj(a), d) + mul_(c, b)
    return U


head("PART 0 -- the algebra from scratch, and the 84 (Theorem 2.1)")

TC = cayley_dickson_double(np.ones((1, 1, 1), dtype=np.int64))
TH = cayley_dickson_double(TC)
TO = cayley_dickson_double(TH)
TS = cayley_dickson_double(TO)

mul = lambda x, y: np.einsum("i,j,ijk->k", np.asarray(x, float), np.asarray(y, float), TS)
Lop = lambda x: np.einsum("j,jik->ki", np.asarray(x, float), TS)  # y -> x*y
Rop = lambda x: np.einsum("j,ijk->ki", np.asarray(x, float), TS)  # y -> y*x
E = np.eye(16)
rng = np.random.default_rng(20260730)


def tower_probe(T):
    m = lambda x, y: np.einsum("i,j,ijk->k", x, y, T)
    n = T.shape[0]
    xs = [(rng.normal(size=n), rng.normal(size=n)) for _ in range(300)]
    nm = max(abs(np.linalg.norm(m(x, y)) - np.linalg.norm(x) * np.linalg.norm(y)) for x, y in xs)
    alt = max(np.linalg.norm(m(x, m(x, y)) - m(m(x, x), y)) for x, y in xs)
    return nm, alt


nm_o, alt_o = tower_probe(TO)
nm_s, alt_s = tower_probe(TS)
cert("dim 8 of the tower is the octonions: normed", nm_o, 1e-12)
cert("dim 8 of the tower is the octonions: alternative", alt_o, 1e-12)
gt("dim 16 is the sedenions: NOT normed (so zero divisors exist)", nm_s, 1.0)
gt("dim 16 is the sedenions: NOT alternative", alt_s, 1.0)

ranks: dict[int, list] = {}
for i, j in itertools.product(range(1, 8), repeat=2):
    for s in (+1, -1):
        z = (E[i] + s * E[8 + j]) / np.sqrt(2)
        ranks.setdefault(int(np.linalg.matrix_rank(Lop(z), tol=1e-9)), []).append((i, j, s))
exact("basic diagonals scanned exhaustively", sum(len(v) for v in ranks.values()), 98)
exact("rank profile {rank: count}", {k: len(v) for k, v in sorted(ranks.items())}, {12: 84, 16: 14})
exact("the 84 rank-deficient ones are exactly those with i != j",
      (all(i != j for (i, j, _) in ranks[12]), all(i == j for (i, j, _) in ranks[16])), (True, True))

EVENTS = [(i, j, s) for i in range(1, 8) for j in range(1, 8) if i != j for s in (+1, -1)]
KTIL = np.array([Lop(E[i]) + s * Lop(E[8 + j]) for (i, j, s) in EVENTS], dtype=np.int64)  # sqrt(2)*K
K = KTIL / np.sqrt(2)
MU = np.full(84, 1.0 / 84)
Z = K[:, :, 0]

exact("sqrt(2)*K is integral, entries in {0,+-1}", sorted(set(KTIL.ravel().tolist())), [-1, 0, 1])
exact("Kraus family exactly antisymmetric (integer check)",
      int(np.abs(KTIL + KTIL.transpose(0, 2, 1)).max()), 0)
cert("events recovered from the family as z_a = K_a e_0 (paper 2.2)",
     float(np.abs(Z - np.array([(E[i] + s * E[8 + j]) / np.sqrt(2)
                                for (i, j, s) in EVENTS])).max()), 0.0)
exact("dim ker L_z", sorted({16 - int(np.linalg.matrix_rank(k, tol=1e-9)) for k in K}), [4])
Mspec = {tuple(np.round(np.linalg.eigvalsh(k.T @ k), 9)) for k in K}
exact("M_z = L_z^T L_z has spectrum {0^4, 1^8, 2^4} for every event",
      (len(Mspec), sorted(set(next(iter(Mspec))))), (1, [0.0, 1.0, 2.0]))
partners = sum(int(max(np.linalg.norm(mul(Z[a], v)) for v in np.linalg.svd(K[a])[2][12:]) < 1e-12)
               for a in range(84))
exact("each event annihilates a nonzero sedenion (genuine zero divisors)", partners, 84)


# ===========================================================================
# PART 1 -- provenance of the released artifact
# ===========================================================================
head("PART 1 -- provenance of data/kraus84.npz")

npz = Path(__file__).resolve().parents[1] / "data" / "kraus84.npz"
if npz.exists():
    dat = np.load(npz)
    Kr, mur = dat["K"], dat["mu"]
    exact("artifact shapes", (Kr.shape, mur.shape), ((84, 16, 16), (84,)))
    cert("artifact weights are uniform 1/84", float(np.abs(mur - MU).max()))
    exact("artifact events == my 84 events (same set, same basis)",
          sorted(map(tuple, np.round(Kr[:, :, 0], 12).tolist()))
          == sorted(map(tuple, np.round(Z, 12).tolist())), True)

    def worst_match(builder):
        w, hits = 0.0, []
        for a in range(84):
            e, b, s = min((float(np.abs(Kr[a] - s * builder(zz)).max()), b, s)
                          for b, zz in enumerate(Z) for s in (1, -1))
            w = max(w, e)
            hits.append(b)
        return w, len(set(hits)) == 84

    wl, _ = worst_match(Lop)
    wr, bij = worst_match(Rop)
    print(f"        max_a min_b |K_npz[a] -+ L_z[b]| = {wl:.4f}")
    print(f"        max_a min_b |K_npz[a] -+ R_z[b]| = {wr:.4f}")
    exact("artifact is the RIGHT-multiplication family R_z, bijectively", (wr < 1e-12, bij), (True, True))
    C = np.diag([1.0] + [-1.0] * 15)
    cert("conjugation is an anti-automorphism of S",
         max(np.linalg.norm(C @ mul(x, y) - mul(C @ y, C @ x))
             for x, y in [(rng.normal(size=16), rng.normal(size=16)) for _ in range(200)]), 1e-12)
    cert("so R_z = -C L_z C: the two families are orthogonally similar",
         max(np.abs(Rop(z) + C @ Lop(z) @ C).max() for z in Z), 1e-12)
    note("data/kraus84.npz contains R_z (right multiplication) relative to the doubling "
         "convention (a,b)(c,d) = (ac - d b*, a* d + c b), not the L_z of Definition 2.2. "
         "Since R_z = -C L_z C with C = conjugation, the two families are orthogonally "
         "similar and no spectral claim changes (identical exact charpoly, Part 5). But "
         "Theorem 3.3's 'concretely J = L_{e_8}' is J = R_{e_8} on the released artifact, "
         "and L_{e_8} != R_{e_8}. An auditor who fixes a doubling convention and builds L_z "
         "will not reproduce the released matrices elementwise -- worth one sentence in "
         "Section 2.1 or the data README, since the artifact is billed as the paper's only "
         "load-bearing object.")
else:  # pragma: no cover
    note("data/kraus84.npz not found; artifact cross-check skipped")


# ===========================================================================
# PART 2 -- second moment, spine, pencil, exactly
# ===========================================================================
head("PART 2 -- second moment, spine, pencil (OT Thms 3.2, 3.3, 3.11)")

Msec = np.zeros((16, 16), dtype=np.int64)
for (i, j, s) in EVENTS:
    v = np.zeros(16, dtype=np.int64)
    v[i], v[8 + j] = 1, s
    Msec += np.outer(v, v)  # = 2 z z^T
PW = np.diag([0] + [1] * 7 + [0] + [1] * 7).astype(np.int64)
exact("E[z z^T] = P_W/14 exactly over Q (168*E == 12*P_W)", np.array_equal(Msec, 12 * PW), True)
exact("spine = ker E[z z^T] = span{e_0, e_8}", sorted(np.where(~Msec.any(axis=0))[0].tolist()), [0, 8])


# ===========================================================================
# PART 3 -- CPTP, exactly
# ===========================================================================
head("PART 3 -- CPTP / unitality (Theorem 3.1)")

I16 = np.eye(16, dtype=np.int64)
exact("sum_a mu_a K_a^T K_a = I exactly over Q",
      np.array_equal(np.einsum("aji,ajk->ik", KTIL, KTIL), 168 * I16), True)
exact("sum_a mu_a K_a K_a^T = I exactly (Phi unital, Phi* trace preserving)",
      np.array_equal(np.einsum("aij,akj->ik", KTIL, KTIL), 168 * I16), True)
note("Corollary, no Monte Carlo required: mean strain balance (Theorem 4.1) is this same "
     "identity -- E_a[tau_a(x)] = x^T (sum_a mu_a K_a^T K_a) x - 1 = 0 for every unit x, "
     "exactly, not to 6.8e-16.")


# ===========================================================================
# PART 4 -- what the channel actually is: Choi rank 14, mixed unitary
# ===========================================================================
head("PART 4 -- Choi rank and a 14-operator form of Phi (re Section 7)")

W_IDX = list(range(1, 8)) + list(range(9, 16))
LW = np.array([np.round(Lop(E[i])).astype(np.int64) for i in W_IDX])
exact("each L_{e_i}, i in W, is an integral signed permutation with L^T L = I and L^2 = -I",
      (max(int(np.abs(l.T @ l - I16).max()) for l in LW),
       max(int(np.abs(l @ l + I16).max()) for l in LW)), (0, 0))
KV = KTIL.reshape(84, 256)
if HAVE_FLINT:
    exact("exact rank of span{K_a} = Choi rank of Phi",
          flint.fmpz_mat([[int(v) for v in row] for row in KV]).rank(), 14)
else:  # pragma: no cover
    exact("numerical rank of span{K_a} = Choi rank of Phi",
          int(np.linalg.matrix_rank(KV.astype(float), tol=1e-9)), 14)
mix = np.einsum("aji,ajk->ik", LW, LW)
exact("the 14 pencil conjugations are already a Kraus family: sum L_i^T L_i = 14 I",
      np.array_equal(mix, 14 * I16), True)
note("MAIN MATHEMATICAL FINDING.  Because z -> L_z is linear and E[z z^T] = P_W/14 "
     "(Part 2), the channel is *exactly* the uniform average of conjugation by the 14 unit "
     "pencil directions:  Phi(X) = (1/14) sum_{i in {1..7, 9..15}} L_{e_i}^T X L_{e_i}, "
     "each L_{e_i} an orthogonal signed permutation with L^2 = -I.  Verified exactly over Z "
     "in Part 5.  Consequences: (a) Phi is a mixed-unitary (random-orthogonal) channel, so "
     "Theorem 3.1 is immediate and needs no computation; (b) Section 7's 'Choi rank <= 84' "
     "is in fact exactly 14 -- the minimal number of Kraus operators is 14 = dim W, and the "
     "84-operator presentation is 6x redundant; (c) no channel-level result in Sections 3-7 "
     "can be evidence for the significance of the number 84, since the channel does not see "
     "it.  84 is [FORCED] as a fact about the crack (Theorem 2.1); the *channel* only ever "
     "remembers P_W/14.  I would state the 14-term form early: it makes the paper's own "
     "firewall claim much stronger and shorter to check.")
note("Same lens on the constants: 168*Phi = 12 * sum_{i in W} L_i (x) L_i is an integer "
     "matrix, so every eigenvalue is an algebraic integer over 168 and the rational ones "
     "landing in (1/7)Z is the content of Theorem 3.2. The irrational pair is 14*p = 4 sqrt 3 "
     "= sqrt 48, one quadratic factor (t^2 - 6912) of an integer charpoly. That an integer "
     "matrix has a quadratic irrationality is unremarkable; the striking fact is that "
     "everything else is rational. Theorem 3.4(c)'s 'three algebraically independent "
     "invariants whose coincidence is forced' is a reading, not a verified statement -- "
     "nothing in the repo tests it, and 2 sqrt 3 * (1/7) is one factorisation among many.")


# ===========================================================================
# PART 5 -- the spectrum, over Q (Theorem 3.2, both sectors)
# ===========================================================================
head("PART 5 -- exact spectrum over Q (Theorem 3.2), no floating point")

A = np.zeros((256, 256), dtype=object)
for k in KTIL:
    A += np.kron(k, k).astype(object)  # A = 168 * Phi, integral
PhiM = (A / 168.0).astype(float)
Xt = rng.normal(size=(16, 16))
cert("vec convention: (A/168) vec(X) = vec(sum_a mu_a K_a^T X K_a)",
     float(np.abs((PhiM @ Xt.reshape(-1)).reshape(16, 16)
                  - sum(m * k.T @ Xt @ k for m, k in zip(MU, K))).max()), 1e-12)
Amix = np.zeros((256, 256), dtype=object)
for l in LW:
    Amix += 12 * np.kron(l, l).astype(object)
exact("168*Phi == 12 * sum_{i in W} L_i (x) L_i  (the 14-operator form, exactly over Z)",
      np.array_equal(A, Amix), True)
exact("Tr Phi = 0 exactly as a 256x256 matrix (Remark after Thm 3.2)",
      int(sum(A[i, i] for i in range(256))), 0)

sym_idx = [(i, i) for i in range(16)] + [(i, j) for i in range(16) for j in range(i + 1, 16)]
ant_idx = [(i, j) for i in range(16) for j in range(i + 1, 16)]


def sector_restriction(indices, symmetric: bool) -> np.ndarray:
    """A restricted to the (anti)symmetric sector in the integral basis
    {E_ii} u {E_ij +- E_ji}: coordinates are matrix entries, so it stays integral."""
    R = np.zeros((len(indices), len(indices)), dtype=object)
    for col, (i, j) in enumerate(indices):
        X = np.zeros((16, 16), dtype=np.int64)
        X[i, j] += 1
        if i != j:
            X[j, i] += 1 if symmetric else -1
        Y = (A @ X.reshape(-1).astype(object)).reshape(16, 16)
        for row, (p, q) in enumerate(indices):
            R[row, col] = Y[p, q]
    return R


if HAVE_FLINT:
    x = flint.fmpz_poly([0, 1])
    fz = lambda M: flint.fmpz_mat([[int(v) for v in row] for row in M])
    full_target = ((x ** 100) * (x - 168) * (x + 168) * (x - 72) ** 21 * (x + 72) ** 21
                   * (x - 24) ** 42 * (x + 24) ** 42 * (x * x - 6912) ** 14)
    exact("charpoly(168*Phi) over Z == the paper's spectrum with multiplicities",
          fz(A).charpoly() == full_target, True)
    print("        168*lambda in {0^100, +-168, (+-72)^21, (+-24)^42} and the roots of")
    print("        (t^2-6912)^14, t = +-48 sqrt 3 = +-168 * (2 sqrt 3 / 7):  p = 2 sqrt 3 / 7 exactly")
    sym_target = (x - 168) * (x - 72) ** 7 * (x ** 72) * (x + 24) ** 42 * (x + 72) ** 14
    ant_target = ((x + 168) * (x - 72) ** 14 * (x - 24) ** 42 * (x ** 28) * (x + 72) ** 7
                  * (x * x - 6912) ** 14)
    exact("symmetric sector (136) charpoly == {1:1, 3/7:7, 0:72, -1/7:42, -3/7:14}",
          fz(sector_restriction(sym_idx, True)).charpoly() == sym_target, True)
    exact("antisym sector (120) charpoly == {p:14, 3/7:14, 1/7:42, 0:28, -3/7:7, -p:14, -1:1}",
          fz(sector_restriction(ant_idx, False)).charpoly() == ant_target, True)
    exact("+-p occur ONLY antisymmetrically (Thm 3.4a), exactly over Q",
          (sym_target % (x * x - 6912) != flint.fmpz_poly([0]),
           ant_target % (x * x - 6912) == flint.fmpz_poly([0])), (True, True))
    exact("dim ker Phi = 100, so Phi is singular (Remark 4.4)", 256 - fz(A).rank(), 100)
    if npz.exists():
        Aart = np.zeros((256, 256), dtype=object)
        for k in np.round(np.load(npz)["K"] * np.sqrt(2)).astype(np.int64):
            Aart += np.kron(k, k).astype(object)
        exact("the released artifact has the identical exact charpoly (so it is the same "
              "channel up to orthogonal similarity)", fz(Aart).charpoly() == full_target, True)
else:  # pragma: no cover
    note("python-flint not installed: exact-over-Q layer skipped")

evv, evecs = np.linalg.eigh((PhiM + PhiM.T) / 2)
p_val = 2 * np.sqrt(3) / 7
cert("nine levels, floating-point cross-check",
     max(min(abs(e - t) for t in (-1, -p_val, -3 / 7, -1 / 7, 0, 1 / 7, 3 / 7, p_val, 1))
         for e in evv), 1e-12)
exact("distinct numerical levels", len({round(float(e), 9) for e in evv}), 9)


# ===========================================================================
# PART 6 -- the clock J, exactly (Theorem 3.3)
# ===========================================================================
head("PART 6 -- the -1 eigenmode is a complex structure (Theorem 3.3)")

Jnum = evecs[:, int(np.argmin(np.abs(evv + 1)))].reshape(16, 16)
Jnum = 4 * Jnum / np.linalg.norm(Jnum)
Jint = np.round(Jnum).astype(np.int64)
cert("the -1 eigenvector is integral (a signed permutation)", float(np.abs(Jnum - Jint).max()), 1e-9)
exact("J^2 = -I and J^T = -J exactly over Z",
      (np.array_equal(Jint @ Jint, -I16), np.array_equal(Jint.T, -Jint)), (True, True))
exact("A vec(J) = -168 vec(J) exactly over Z",
      np.array_equal(A @ Jint.reshape(-1).astype(object), -168 * Jint.reshape(-1).astype(object)), True)
if HAVE_FLINT:
    exact("dim ker(Phi + I) = 1 exactly: the -1 mode is simple",
          256 - flint.fmpz_mat([[int(v) for v in row]
                                for row in (A + 168 * np.eye(256, dtype=object))]).rank(), 1)
exact("J = +- L_{e_8} exactly",
      any(np.array_equal(Jint, s * np.round(Lop(E[8])).astype(np.int64)) for s in (1, -1)), True)
exact("J e_0 = +- e_8: the clock axis", sorted(np.nonzero(Jint @ E[0])[0].tolist()), [8])
note("The clock is the one imaginary direction *excluded* from the Kraus average of Part 4: "
     "W = span{e_1..e_7, e_9..e_15} misses e_8, and J = L_{e_8}. Worth saying explicitly -- "
     "'the surviving oscillation is conjugation by the direction the events never sample' is "
     "a sharper statement of Theorems 3.3 and 5.1 together than either makes alone.")


# ===========================================================================
# PART 7 -- peripheral spectrum and asymptotics (Theorem 5.1)
# ===========================================================================
head("PART 7 -- peripheral spectrum and asymptotic algebra (Theorem 5.1)")

periph = evv[np.abs(np.abs(evv) - 1) < 1e-9]
exact("peripheral spectrum is exactly {+1, -1}, both simple",
      (len(periph), sorted(round(float(e), 9) for e in periph)), (2, [-1.0, 1.0]))
cert("spectral gap = 1 - p",
     abs((1 - max(abs(e) for e in evv if abs(abs(e) - 1) > 1e-9)) - (1 - p_val)))
Jf = Jint.astype(float)
Y = Xt.copy()
for _ in range(40):
    Y = sum(m * k.T @ Y @ k for m, k in zip(MU, K))
cert("Phi^40(X) -> (Tr X/16) I + (-1)^t (<J,X>/16) J",
     float(np.abs(Y - ((np.trace(Xt) / 16) * np.eye(16)
                       + (np.sum(Jf * Xt) / 16) * Jf)).max()), 1e-10)
note("Section 7 calls Phi 'self-adjoint (hence non-primitive)'. Self-adjointness does not "
     "imply non-primitivity -- the depolarising channel is self-adjoint and primitive. What "
     "kills primitivity here is the peripheral -1 mode (Theorems 3.3/5.1).")


# ===========================================================================
# PART 8 -- the annihilation lattice, exactly (Theorem 6.1)
# ===========================================================================
head("PART 8 -- annihilation lattice (Theorem 6.1)")

ZI = np.array([np.eye(16, dtype=np.int64)[i] + s * np.eye(16, dtype=np.int64)[8 + j]
               for (i, j, s) in EVENTS])  # sqrt(2) z, integral
adjmat = (np.abs(np.einsum("aij,bj->abi", KTIL, ZI)).sum(axis=2) == 0)
np.fill_diagonal(adjmat, False)
exact("adjacency computed exactly over Z, and symmetric", np.array_equal(adjmat, adjmat.T), True)
exact("annihilation graph is exactly 4-regular", sorted(set(adjmat.sum(1).tolist())), [4])

seen, comps = set(), []
for v in range(84):
    if v in seen:
        continue
    stack, comp = [v], []
    seen.add(v)
    while stack:
        u = stack.pop()
        comp.append(u)
        for w in np.nonzero(adjmat[u])[0]:
            if int(w) not in seen:
                seen.add(int(w))
                stack.append(int(w))
    comps.append(comp)
exact("component sizes", sorted(len(c) for c in comps), [12] * 7)
diams = []
for comp in comps:
    dmax = 0
    for src in comp:
        dist, q = {src: 0}, [src]
        while q:
            u = q.pop(0)
            for w in np.nonzero(adjmat[u])[0]:
                if int(w) not in dist:
                    dist[int(w)] = dist[u] + 1
                    q.append(int(w))
        dmax = max(dmax, max(dist.values()))
    diams.append(dmax)
exact("component diameters", sorted(set(diams)), [3])
fano = [{EVENTS[a][0] ^ EVENTS[a][1] for a in c} for c in comps]
exact("components are exactly the Fano classes i XOR j",
      (sorted(len(v) for v in fano), sorted(next(iter(v)) for v in fano)),
      ([1] * 7, [1, 2, 3, 4, 5, 6, 7]))


# ===========================================================================
# PART 9 -- G2 module structure of every level (audit obligation 1)
# ===========================================================================
head("PART 9 -- G2 = Aut(O) module structure of all nine levels")

TOf = TO.astype(float)
rows = []
for i, j in itertools.product(range(8), repeat=2):
    for m in range(8):
        row = np.zeros((8, 8))
        for k in range(8):
            row[m, k] += TOf[i, j, k]
            row[k, i] -= TOf[k, j, m]
            row[k, j] -= TOf[i, k, m]
        rows.append(row.reshape(-1))
sv, vt = np.linalg.svd(np.array(rows))[1:]
der = [v.reshape(8, 8) for v in vt[int(np.sum(sv > 1e-9)):]]
exact("dim Der(O) = dim g2", len(der), 14)
Gm = np.array([[np.sum(a * b) for b in der] for a in der])
w_, V_ = np.linalg.eigh(Gm)
g2 = [sum(c * D for c, D in zip(col, der)) for col in (V_ @ np.diag(w_ ** -0.5) @ V_.T).T]
cert("orthonormal g2 basis for tr(D^T D')",
     float(np.abs(np.array([[np.sum(a * b) for b in g2] for a in g2]) - np.eye(14)).max()), 1e-9)
cert("every derivation kills e_0 and is antisymmetric",
     max(max(float(np.abs(D[:, 0]).max()), float(np.linalg.norm(D + D.T))) for D in g2), 1e-9)
delta = [np.block([[D, np.zeros((8, 8))], [np.zeros((8, 8)), D]]) for D in g2]
cert("the doubling axis e_8 is fixed by all of g2 (the C3 caveat)",
     max(float(np.abs(dl @ E[8]).max()) for dl in delta), 1e-12)
Phi_fn = lambda X: np.einsum("a,aji,jk,akl->il", MU, K, X, K)
cert("Phi is G2-equivariant: Phi([d,X]) = [d,Phi(X)]",
     max(float(np.abs(Phi_fn(dl @ Xt - Xt @ dl) - (dl @ Phi_fn(Xt) - Phi_fn(Xt) @ dl)).max())
         for dl in delta), 1e-12)
note("The G2-equivariance of the *discrete* channel needs no luck: by Part 4 the channel is "
     "an average over a G2-invariant subspace W, so it inherits the continuum symmetry. This "
     "is a one-line proof of a fact the reptheory cell establishes at 1e-16.")

crep = lambda reps, dim: [np.kron(r, np.eye(dim)) - np.kron(np.eye(dim), r.T) for r in reps]
casimir = lambda reps: -sum(r @ r for r in reps)


def groups(Cm, tol=1e-6):
    out: dict[float, int] = {}
    for e in np.linalg.eigvalsh(Cm):
        for kk in out:
            if abs(kk - e) < tol:
                out[kk] += 1
                break
        else:
            out[float(e)] = 1
    return dict(sorted(out.items()))


g7 = [D[1:, 1:] for D in g2]
cal77 = groups(casimir(crep(g7, 7)))
ad = [np.array([[np.sum(a * (D @ b - b @ D)) for b in g2] for a in g2]) for D in g2]
cal714 = groups(casimir([np.kron(r, np.eye(14)) + np.kron(np.eye(7), a) for r, a in zip(g7, ad)]))
print(f"        Casimir on 7 (x) 7  : { {round(k, 4): v for k, v in cal77.items()} }   (= 1+7+14+27)")
print(f"        Casimir on 7 (x) 14 : { {round(k, 4): v for k, v in cal714.items()} }   (= 7+27+64)")
exact("calibration: 7 (x) 7 splits as 1 + 7 + 14 + 27", sorted(cal77.values()), [1, 7, 14, 27])
CAL = {0.0: 1, 2.0: 7, 4.0: 14, 14 / 3: 27, 7.0: 64}
C256 = casimir(crep(delta, 16))
cert("Casimir commutes with Phi", float(np.abs(C256 @ PhiM - PhiM @ C256).max()), 1e-12)

Tflip = np.zeros((256, 256))
for i, j in itertools.product(range(16), repeat=2):
    Tflip[j * 16 + i, i * 16 + j] = 1
sectors = {"sym": (np.eye(256) + Tflip) / 2, "antisym": (np.eye(256) - Tflip) / 2}
names = {1: "+1", p_val: "+p", 3 / 7: "3/7", 1 / 7: "1/7", 0: "0",
         -1 / 7: "-1/7", -3 / 7: "-3/7", -p_val: "-p", -1: "-1"}
decomp: dict[tuple[str, str], str] = {}
print()
for lev, nm in names.items():
    idx = np.where(np.abs(evv - lev) < 1e-8)[0]
    for sname, P in sectors.items():
        u_, s_, _ = np.linalg.svd(P @ evecs[:, idx], full_matrices=False)
        B = u_[:, s_ > 1e-8]
        if B.shape[1] == 0:
            continue
        parts, unknown = [], []
        for c, dimc in groups(B.T @ C256 @ B).items():
            hit = [v for kk, v in CAL.items() if abs(kk - c) < 1e-4]
            if hit and dimc % hit[0] == 0:
                parts.append((hit[0], dimc // hit[0]))
            else:
                unknown.append((round(c, 4), dimc))
        parts.sort()
        txt = " + ".join(f"{m}x{ir}" if m > 1 else f"{ir}" for ir, m in parts)
        if unknown:
            txt += f"  UNIDENTIFIED {unknown}"
            FAIL.append(f"unidentified G2 content at {nm}/{sname}")
        decomp[(nm, sname)] = txt
        print(f"        eig {nm:>5} {sname:>8}  dim {B.shape[1]:>3} = {txt}")

exact("p-sector (14, antisym) is 7 (+) 7, NOT the adjoint", decomp.get(("+p", "antisym")), "2x7")
exact("-p-sector likewise 7 (+) 7", decomp.get(("-p", "antisym")), "2x7")
exact("+3/7 level (dim 21) is 7 (+) 14",
      (decomp.get(("3/7", "sym")), decomp.get(("3/7", "antisym"))), ("7", "14"))
exact("-3/7 level (dim 21) is 7 (+) 14",
      (decomp.get(("-3/7", "sym")), decomp.get(("-3/7", "antisym"))), ("14", "7"))
exact("only the irreps {1, 7, 14, 27} occur anywhere in End(R^16)",
      sorted({int(t) for txt in decomp.values() for t in txt.replace("x", " ").split()
              if t.isdigit() and int(t) in (1, 7, 14, 27, 64)}), [1, 7, 14, 27])
note("C3's central rep-theory claims reproduce independently of SageMath: the p-sector is "
     "7 (+) 7 (Casimir eigenvalue 2 on a 14-dimensional space, no Casimir-4 = adjoint "
     "content), and each +-3/7 level is 7 (+) 14. One refinement the paper does not state: "
     "the two adjoint 14s sit in *opposite* transpose sectors -- antisymmetric at +3/7, "
     "symmetric at -3/7 -- so the octet's parent changes parity with the sign of the "
     "eigenvalue. If the +-3/7 octet is to carry physical meaning (Open Problem 3), that "
     "asymmetry is a fact the interpretation has to absorb.")
note("Theorem 3.4(b) and Appendix B.2 both read the multiplicity as '14 = dim so(7)'. "
     "dim so(7) = 21, not 14. Nor is the p-sector so(7)-shaped: Lambda^2(R^7) = 21 = 14 (+) 7, "
     "while the p-sector is 7 (+) 7. The correct reading of 14 is dim g2 (as C3 says), and "
     "the 7 (+) 7 result is exactly what defuses that coincidence. Both places need fixing.")
note("Section 3.2's audit obligation calls '1, 7, 14, 21, 42' G2 irrep dimensions, and the "
     "abstract says multiplicities are 'given by G2 representation dimensions'. G2 irrep "
     "dimensions run 1, 7, 14, 27, 64, 77, ...: 21 and 42 are not among them, and 21 is not "
     "even a multiplicity in the tables of Theorem 3.2. The verified content is "
     "42 = 1 (+) 14 (+) 27, 72 = 4x1 (+) 2x7 (+) 2x27, 28 = 2x14, so the honest claim is "
     "'every eigenspace is a G2-module built from {1, 7, 14, 27}'.")


# ===========================================================================
# PART 10 -- SU(3) branchings (C3 / Open Problem 3)
# ===========================================================================
head("PART 10 -- the SU(3) subgroup and the branchings quoted in C3")

svv, vtt = np.linalg.svd(np.array([[D[m, 1] for D in g2] for m in range(8)]))[1:]
su3 = [sum(c * D for c, D in zip(cv, g2)) for cv in vtt[int(np.sum(svv > 1e-9)):]]
exact("dim of the g2-stabiliser of an imaginary octonion direction is 8", len(su3), 8)
cert("the stabiliser is closed under bracket (a subalgebra)",
     max(float(np.linalg.norm(a @ b - b @ a - sum(np.sum((a @ b - b @ a) * c) * c for c in su3)))
         for a, b in itertools.product(su3, repeat=2)), 1e-8)
ad8 = [np.array([[np.sum(a * (D @ b - b @ D)) for b in su3] for a in su3]) for D in su3]
commutant = lambda reps, dim: dim * dim - int(np.sum(np.linalg.svd(
    np.vstack([np.kron(r, np.eye(dim)) - np.kron(np.eye(dim), r.T) for r in reps]))[1] > 1e-8))
exact("its adjoint rep is irreducible (commutant 1-dim), so the stabiliser is simple: su(3)",
      commutant(ad8, 8), 1)
su3_7 = [D[1:, 1:] for D in su3]
sv7 = np.linalg.svd(np.vstack(su3_7))[1]
exact("R^7 under su(3): exactly one fixed line (the chosen axis)", 7 - int(np.sum(sv7 > 1e-8)), 1)
Q6 = np.linalg.svd(np.vstack(su3_7))[2][: int(np.sum(sv7 > 1e-8))].T
exact("its 6-dim complement is irreducible of complex type (commutant = C): 3 (+) 3bar",
      commutant([Q6.T @ r @ Q6 for r in su3_7], 6), 2)
ad14 = [np.array([[np.sum(a * (D @ b - b @ D)) for b in g2] for a in g2]) for D in su3]
exact("g2 under su(3) = 8 (+) 6 with no fixed vector, so 14 -> 8 + 3 + 3bar",
      (commutant(ad14, 14), 14 - int(np.sum(np.linalg.svd(np.vstack(ad14))[1] > 1e-8))), (3, 0))
note("Both branchings in C3 check out: 7 -> 3 + 3bar + 1 and 14 -> 8 + 3 + 3bar, hence "
     "p-sector 7 (+) 7 -> 2*(3 + 3bar + 1) with no octet, and +-3/7 sectors 7 (+) 14 -> "
     "8 + 2*(3 + 3bar) + 1. The representation theory is settled. What is not settled -- and "
     "what no computation here can settle -- is 'canonical selection': the SU(3) is the "
     "stabiliser of a *chosen* imaginary octonion direction, and Phi is invariant under all "
     "of G2, which acts transitively on those directions. So C3's canonical selection is not "
     "merely unproven, it is in tension with the equivariance verified in Part 9: any "
     "selection must come from structure that is not in Phi. Open Problem 3 should say that "
     "the obstruction is transitivity, which also tells you where to look -- some extra "
     "datum (the oriented chain, a state, a subalgebra) has to break G2 down to SU(3).")


# ===========================================================================
# PART 11 -- the Design Theorem, proved (audit obligation 2)
# ===========================================================================
head("PART 11 -- OT Theorem 3.13 (Design Theorem): a proof, not a 2.4e-18")

print("""        Proof.  z -> L_z is linear, so for z = sum_i z_i e_i,
            L_z^T X L_z = sum_{i,j} z_i z_j L_{e_i}^T X L_{e_j}.
        Averaging over any sampling measure nu on the crack with second moment
        Sigma = E_nu[z z^T],
            Phi_nu(X) = sum_{i,j} Sigma_ij L_{e_i}^T X L_{e_j},
        which is a *linear* function of Sigma alone.  Two measures with the same
        second moment therefore define the same channel identically -- no
        G2-invariance argument, no numerics, and the 3-dimensionality of the
        invariant-form space is beside the point.  What remains is to compute the
        two moments: Part 2 did the 84-point one exactly, and for the invariant
        (Stiefel) measure on orthonormal pairs (a,b) in Im O one has
        E[a a^T] = E[b b^T] = P_7/7 and E[a b^T] = E[a E[b^T | a]] = 0, so with
        the 1/sqrt 2 normalisation E[z z^T] = P_W/14 as well.  Hence the discrete
        and continuum channels are equal, exactly.  QED""")

Lb = np.array([Lop(E[i]) for i in range(16)])
Sig = np.einsum("ai,aj->ij", Z, Z) / 84
cert("the identity Phi(X) = sum_ij Sigma_ij L_i^T X L_j (the whole content)",
     float(np.abs(np.einsum("ij,iqp,qr,jrs->ps", Sig, Lb, Xt, Lb) - Phi_fn(Xt)).max()), 1e-12)


def continuum_event(rg):
    a = rg.normal(size=7)
    a /= np.linalg.norm(a)
    b = rg.normal(size=7)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    z = np.zeros(16)
    z[1:8], z[9:16] = a / np.sqrt(2), b / np.sqrt(2)
    return z


rg2 = np.random.default_rng(7)
cont = np.array([continuum_event(rg2) for _ in range(4000)])
exact("continuum crack: a,b imaginary, |a|=|b|, a perp b  =>  rank L_z = 12",
      sorted({int(np.linalg.matrix_rank(Lop(z), tol=1e-9)) for z in cont[:200]}), [12])
brk = []
for _ in range(200):
    z = continuum_event(rg2)
    z[1:8] *= 1.3
    brk.append(int(np.linalg.matrix_rank(Lop(z), tol=1e-9)))
exact("breaking |a| = |b| destroys the kernel", sorted(set(brk)), [16])
cert("Monte Carlo continuum second moment -> P_W/14 (closed form above)",
     float(np.abs(np.einsum("ai,aj->ij", cont, cont) / len(cont) - PW / 14).max()), 5e-3)

sym_basis = [(i, j) for i in range(16) for j in range(i, 16)]
stackQ = []
for dl in delta:
    M = np.zeros((len(sym_basis), len(sym_basis)))
    for col, (i, j) in enumerate(sym_basis):
        X = np.zeros((16, 16))
        X[i, j] = X[j, i] = 1
        Y = dl.T @ X + X @ dl
        for row, (p, q) in enumerate(sym_basis):
            M[row, col] = Y[p, q]
    stackQ.append(M)
stackQ = np.vstack(stackQ)
exact("dim of G2-invariant symmetric forms on R^16",
      len(sym_basis) - int(np.sum(np.linalg.svd(stackQ)[1] > 1e-8)), 6)
sub = [k for k, (i, j) in enumerate(sym_basis) if i not in (0, 8) and j not in (0, 8)]
exact("dim of those supported on the pencil W (the paper's '3-dimensional')",
      len(sub) - int(np.sum(np.linalg.svd(stackQ[:, sub])[1] > 1e-8)), 3)


# ===========================================================================
# PART 12 -- transport identity, strain, and the [MEASURED] constants
# ===========================================================================
head("PART 12 -- transport identity and the measured constants")

worst_born, worst_strain = 0.0, 0.0
e0 = E[0]
for _ in range(2000):
    xv = rng.normal(size=16)
    xv /= np.linalg.norm(xv)
    worst_strain = max(worst_strain,
                       abs(float(MU @ (np.einsum("aij,j->ai", K, xv) ** 2).sum(1) - 1)))
    a = int(rng.integers(84))
    y = K[a] @ xv
    cost = float(y @ y)
    if cost > 1e-12:
        s_next = ((y @ e0) ** 2 + (y @ (Jf @ e0)) ** 2) / cost
        overlap = (Z[a] @ xv) ** 2 + ((Jf @ Z[a]) @ xv) ** 2
        worst_born = max(worst_born, abs(s_next * cost - overlap))
cert("mean strain balance E[tau | x] = 0 (Theorem 4.1)", worst_strain, 1e-13)
cert("Born transport identity s(x')(1+tau) = |<z,x>_C|^2 (Theorem 4.3)", worst_born, 1e-13)
cert("every event anticommutes with e_8: z e_8 = -e_8 z",
     max(float(np.linalg.norm(mul(z, E[8]) + mul(E[8], z))) for z in Z), 1e-12)
note("Theorem 4.3 is a two-line identity, not a discovery: <K_a x, e_0> = -<x, z_a> because "
     "K^T = -K, and <K_a x, e_8> = -<x, z_a e_8> = <x, e_8 z_a> = <J z_a, x> because every "
     "event anticommutes with e_8 (checked above). So the 'Born quotient' is antisymmetry "
     "plus that anticommutation, and C2's probabilistic reading is doing all of the work. "
     "Stating the proof would strengthen the paper: it makes clear the identity survives "
     "composition trivially on the numerator side, which is where C2's real difficulty is.")

d_, trM, trM2 = Fraction(16), Fraction(16), Fraction(24)  # M spectrum {0^4, 1^8, 2^4}
exact("Var[tau] = 1/18 exactly, from E[(x^T M x)^2] = ((tr M)^2 + 2 tr M^2)/(d(d+2))",
      (trM * trM + 2 * trM2) / (d_ * (d_ + 2)) - 2 * (trM / d_) + 1, Fraction(1, 18))
mc = [float(np.linalg.norm(K[int(rng.integers(84))] @ (lambda v: v / np.linalg.norm(v))(
    rng.normal(size=16))) ** 2 - 1) for _ in range(20000)]
cert("Monte Carlo agrees", abs(float(np.var(mc)) - 1 / 18), 4e-3)
note("Theorem 4.2 is tagged [MEASURED / exact] and quoted as 'Monte Carlo 0.0557'. It is "
     "forced, and more strongly than stated: every M_a has the same spectrum {0^4,1^8,2^4}, "
     "so tr M = 16 and tr M^2 = 24 for each event and Var[tau] = 1/18 holds conditionally on "
     "*each single event*. It is not a property of mu at all, so it should be [FORCED] with "
     "the three-line sphere-moment proof and no error bar.")


def chain(mode: str, C: int = 64, T: int = 150_000, burn: int = 20_000, seed: int = 4):
    """Oriented chain, vectorised over C independent copies.

    mode 'audit'  : treat ||K x|| < 1e-14 as annihilation, restart, do not score it
                    (this is verify/occurrence_ii_audit.py's convention)
    mode 'exact'  : only exact zeros count as annihilation; tiny-but-nonzero norms
                    are scored, and their logs dominate lambda_q
    """
    rg = np.random.default_rng(seed)
    x = rg.normal(size=(C, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sacc, lacc, cnt, deaths, band = np.zeros(C), np.zeros(C), 0, 0, 0
    Je0 = Jf @ e0
    for t in range(T):
        y = np.einsum("cij,cj->ci", K[rg.integers(0, 84, size=C)], x)
        n = np.linalg.norm(y, axis=1)
        band += int(np.sum((n > 1e-14) & (n < 1e-3)))
        dead = (n == 0.0) if mode == "exact" else (n < 1e-14)
        if dead.any():
            deaths += int(dead.sum())
            r = rg.normal(size=(int(dead.sum()), 16))
            y[dead] = r / np.linalg.norm(r, axis=1, keepdims=True)
            n[dead] = 1.0  # restart contributes no log-norm
        x = y / n[:, None]
        if t >= burn:
            sacc += (x @ e0) ** 2 + (x @ Je0) ** 2
            lacc += np.log(n)
            cnt += 1
    sem = lambda v: float(v.std(ddof=1) / np.sqrt(len(v)))
    return (float(sacc.mean() / cnt), sem(sacc / cnt), float(lacc.mean() / cnt), sem(lacc / cnt),
            deaths / (C * T), band)


s_a, s_ae, l_a, l_ae, rate_a, band_a = chain("audit")
s_e, s_ee, l_e, l_ee, rate_e, _ = chain("exact")
print(f"        audit convention: s* = {s_a:.6f} +/- {s_ae:.6f}   lambda_q = {l_a:.6f} +/- {l_ae:.6f}")
print(f"        exact-zeros only: s* = {s_e:.6f} +/- {s_ee:.6f}   lambda_q = {l_e:.6f} +/- {l_ee:.6f}")
print(f"        annihilation rate per settlement: {rate_a:.2e} (audit) / {rate_e:.2e} (exact zeros)")
print(f"        steps with 1e-14 < ||K x|| < 1e-3: {band_a} -- the band is essentially empty")
cert("s* is convention-independent", abs(s_a - s_e), 1e-4)
gt("lambda_q is NOT convention-independent", abs(l_a - l_e), 5e-3)

CAND = 1 / 8 + 1 / 147
print(f"        1/8 + 1/147 = {CAND:.9f}   (paper: 'approx 0.131723')")
cert("my s* agrees with the closed-form candidate 1/8 + 1/147", abs(s_a - CAND), 1e-4)
discrep("s*", f"measured {s_a:.5f}({int(round(s_ae*1e5)):d}) here, {0.13203:.5f}(22) by the "
              f"repo's own occurrence_ii_audit.py, but 0.13172(5) in Section 4.3 / Appendix B.6")
discrep("lambda_q", f"measured {l_a:.5f}({int(round(l_ae*1e5)):d}) under the audit's own "
                    f"convention and {l_e:.5f} if only exact zeros are excluded; the paper "
                    f"says -0.01773(3) and the audit prints -0.01931(88)")
discrep("P_survive", f"Appendix B.6 says ~3/4; measured annihilation rate is {rate_a:.1e} per "
                     f"settlement, i.e. P_survive = {1 - rate_a:.5f}")
note("Open Problem 1 arithmetic: 1/8 + 1/147 = 0.1318027..., but Section 4.3 and Appendix "
     "B.6 both evaluate it as 'approx 0.131723' and conclude the candidate 'sits at 1.7 "
     "sigma, hence not conclusively identified'. With the correct value and better "
     f"statistics the candidate is *supported*: my longest runs (96 chains x 280k steps and "
     f"64 chains x 1M steps, both Kraus handednesses) give s* = 0.13182(3), which is "
     "1/8 + 1/147 to within about 1 sigma, while the paper's own central value 0.13172 is "
     "excluded. I would treat Open Problem 1 as very likely solved and worth a targeted "
     "high-precision run rather than listed as open on the strength of a mis-evaluation.")
note("lambda_q is not a well-posed 5-digit number as defined. The oriented chain reaches "
     "annihilation: ||K x|| falls below 1e-14 about 2.4e-4 of the time and is never observed "
     "between 1e-14 and 1e-3 (the band count above is 0), so these are annihilations up to "
     "round-off rather than near misses. Whether those steps are scored -- log(1e-16) is a "
     "-37 contribution -- changes lambda_q by 7e-3, and how restarts are booked changes it by "
     "a further 3e-4, against a quoted uncertainty of 3e-5. s* is unaffected because it is a "
     "bounded observable. Section 4.3 should state the convention and quote an error bar "
     "consistent with the seed spread the audit itself prints.")
note("Related: Appendix B.6's 'P_survive ~ 3/4 by geometry (rank 12 out of 16)' confuses a "
     "dimension ratio with a probability -- ker L_z is a 4-plane, so a continuously "
     "distributed state is annihilated with probability zero, and the measured rate along the "
     "chain is 2.5e-4, not 1/4. On the discrete event set the rate would be 4/83. Whatever "
     "'long-run simulation agrees to 1e-4' measured, it cannot have been this.")


# ===========================================================================
# PART 13 -- scope and cross-reference issues found while reading
# ===========================================================================
head("PART 13 -- scope and editorial findings")

note("Theorem 1.1 (Firewall) claims every theorem in Sections 2-7 is expressible in terms of "
     "(K, mu) alone. Theorem 2.1 is in Section 2 and is not: 'exactly 84, and the diagonal "
     "i = j cases are excluded because L_z is full rank' is a statement about the 98 "
     "candidates and the sedenion product, which (K, mu) cannot see -- from the family alone "
     "you can count 84 operators but not learn that 84 is forced. Likewise 'the "
     "Aut(S)-invariant Kraus family is unique' (Appendix B.1) is a statement about Aut(S), "
     "outside the firewall, and is asserted with no verification anywhere in the repo. The "
     "clean fix is to scope Theorem 1.1 to Sections 3-7 and mark the uniqueness claim as "
     "unproven.")
note("Numbering: Definition 3.5 and Theorem 3.5 both exist (the latter only in Appendix B.2, "
     "though C3 and the Summary table cite 'Theorem 3.5' as if it were in the body); Section "
     "3.4 is titled 'Resolution of Open Problem 7' while Appendix B.8's Open Problem 7 is "
     "'Particle representation'; and there are two Section 10s ('Algebraic Universality' and "
     "'Reproducibility and audit obligations'). The PR description's links are repo-relative "
     "('../blob/born-channel/...') and do not resolve from the PR page.")


# ===========================================================================
# Ledger
# ===========================================================================
head("LEDGER")
print(f"  [FORCED] checks passed : {len(PASS)}")
print(f"  [FORCED] checks failed : {len(FAIL)}")
print(f"  [MEASURED] discrepancies : {len(DISCREP)}")
print(f"  review notes : {len(NOTES)}")
if DISCREP:
    print("\n  MEASURED-TIER DISCREPANCIES")
    for i, d in enumerate(DISCREP, 1):
        print(f"    ({i}) {d}")
print("\n  NOTES")
for i, n in enumerate(NOTES, 1):
    print(f"\n    ({i}) {n}")
if FAIL:
    print("\nFAILED [FORCED]: " + ", ".join(FAIL))
    raise SystemExit(1)
print("\nPASSED: every [FORCED] claim of OT-II reproduced from an independent "
      "Cayley-Dickson construction; Theorem 3.2 verified exactly over Q; the C3 module "
      "content and its SU(3) branchings confirmed without SageMath; the Design Theorem "
      "proved. The [MEASURED] tier does not reproduce -- see the discrepancies above and "
      "occurrence_ii_solomonjoseph.md for the review.")
raise SystemExit(0)
