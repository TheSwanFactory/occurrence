"""The crack's graph structure and the settlement-channel spectrum.

These lock in results the audit script asserted but never tested. The
annihilation graph claim ("4-regular, 7 components, diameter 3") is true; the
audit simply used an edge predicate that could not detect it.
"""

import numpy as np
import pytest

from topographo.ssd import SedenionAlgebra


@pytest.fixture(scope="module")
def algebra():
    return SedenionAlgebra()


@pytest.fixture(scope="module")
def crack(algebra):
    return algebra.basis_zero_divisors()


@pytest.fixture(scope="module")
def adjacency(algebra, crack):
    """Edges are genuine algebra annihilation: z * w = 0."""
    n = len(crack)
    return {
        i: [
            j
            for j in range(n)
            if i != j and np.linalg.norm(algebra.mul(crack[i], crack[j])) < 1e-9
        ]
        for i in range(n)
    }


def _components(adjacency):
    unseen = set(adjacency)
    out = []
    while unseen:
        stack = [unseen.pop()]
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adjacency[v]:
                if w in unseen:
                    unseen.remove(w)
                    stack.append(w)
        out.append(sorted(comp))
    return out


def test_crack_has_84_elements(crack):
    assert len(crack) == 84


def test_left_multiplication_kernel_is_four_dimensional(algebra, crack):
    for z in crack:
        sv = np.linalg.svd(algebra.left_operator(z), compute_uv=False)
        assert int(np.sum(sv < 1e-9)) == 4


def test_metric_operator_spectrum_is_0_1_2(algebra, crack):
    want = np.array([0.0] * 4 + [1.0] * 8 + [2.0] * 4)
    for z in crack:
        got = np.sort(np.linalg.eigvalsh(algebra.metric_operator(z)))
        assert np.allclose(got, want, atol=1e-12)


def test_operator_singularity_predicate_is_vacuous(algebra, crack):
    """L_z L_w is singular for EVERY pair, so it cannot define a topology.

    Each L_z has a 4-dimensional kernel, so any product of two is singular.
    This is why the original audit's graph was complete (degree 83).
    """
    n = len(crack)
    z0 = algebra.left_operator(crack[0])
    degree = sum(
        1
        for j in range(1, n)
        if np.linalg.svd(z0 @ algebra.left_operator(crack[j]), compute_uv=False)[-1] < 1e-9
    )
    assert degree == n - 1


def test_annihilation_graph_is_four_regular(adjacency):
    assert {len(v) for v in adjacency.values()} == {4}


def test_annihilation_graph_has_seven_components_of_twelve(adjacency):
    comps = _components(adjacency)
    assert len(comps) == 7
    assert sorted(len(c) for c in comps) == [12] * 7


def test_annihilation_graph_diameter_is_three(adjacency):
    diameter = 0
    for src in adjacency:
        dist = {src: 0}
        queue = [src]
        while queue:
            v = queue.pop(0)
            for w in adjacency[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        diameter = max(diameter, max(dist.values()))
    assert diameter == 3


def test_components_are_labelled_by_the_fano_plane(crack, adjacency):
    """Every crack element is (e_i +- e_{j+8})/sqrt(2) with i, j in 1..7.

    The connected component is determined by i XOR j, which ranges over the
    seven nonzero points of F_2^3 -- the Fano plane.
    """

    def split(z):
        support = np.nonzero(np.abs(z) > 1e-9)[0]
        assert len(support) == 2
        i, j = int(support[0]), int(support[1]) - 8
        assert 1 <= i <= 7 and 1 <= j <= 7
        return i, j

    labels = []
    for comp in _components(adjacency):
        values = {np.bitwise_xor(*split(crack[v])) for v in comp}
        assert len(values) == 1, "i XOR j must be constant on a component"
        labels.append(values.pop())

    assert sorted(labels) == [1, 2, 3, 4, 5, 6, 7]


def test_settlement_channel_spectrum(algebra, crack):
    """Thm 3.12: every eigenvalue of Phi lies in (1/7){0, +-1, +-3, +-2sqrt3, +-7}.

    Phi acts on End(R^16), so it is a 256x256 matrix, not 84x84, and its matrix
    trace is 0 (not 1) by the +- symmetry of the spectrum.
    """
    Ls = [algebra.left_operator(z) for z in crack]
    phi = sum(np.kron(L.T, L.T) for L in Ls) / len(Ls)

    assert phi.shape == (256, 256)
    assert abs(np.trace(phi)) < 1e-12

    eigenvalues = np.linalg.eigvals(phi)
    assert np.abs(eigenvalues.imag).max() < 1e-12

    allowed = np.array([0, 1, -1, 3, -3, 2 * np.sqrt(3), -2 * np.sqrt(3), 7, -7]) / 7.0
    for v in eigenvalues.real:
        assert np.abs(allowed - v).min() < 1e-12


def test_channel_is_unital_and_has_exact_parity_eigenvalue(algebra, crack):
    """Thm 3.6 (unital) and Thm 3.9(b): Phi(L_e8) = -L_e8 exactly."""
    identity = np.eye(algebra.dim)
    Ls = [algebra.left_operator(z) for z in crack]

    image_of_identity = sum(L.T @ identity @ L for L in Ls) / len(Ls)
    assert np.allclose(image_of_identity, identity, atol=1e-12)

    L_e8 = algebra.left_operator(algebra.e[8])
    parity = sum(L.T @ L_e8 @ L for L in Ls) / len(Ls)
    assert np.allclose(parity, -L_e8, atol=1e-12)


def test_antisymmetric_sector_multiplicities_sum_to_120(algebra, crack):
    """The paper's Thm 3.12 listed +-1/7 (x42 each) in the antisymmetric sector.

    That makes the multiplicities sum to 162, not 120. Only +1/7 occurs, with
    multiplicity 42.
    """
    Ls = [algebra.left_operator(z) for z in crack]
    phi = sum(np.kron(L.T, L.T) for L in Ls) / len(Ls)

    n = algebra.dim
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            X = np.zeros((n, n))
            X[i, j], X[j, i] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            basis.append(X.flatten())
    B = np.array(basis).T
    assert B.shape[1] == 120

    sector = B.T @ phi @ B
    eigenvalues = np.sort(np.linalg.eigvals(sector).real)

    allowed = np.array([0, 1, -1, 3, -3, 2 * np.sqrt(3), -2 * np.sqrt(3), 7, -7]) / 7.0
    counts = {}
    for v in eigenvalues:
        key = round(float(allowed[np.abs(allowed - v).argmin()]), 6)
        counts[key] = counts.get(key, 0) + 1

    assert sum(counts.values()) == 120
    assert counts[round(1 / 7, 6)] == 42
    assert round(-1 / 7, 6) not in counts
