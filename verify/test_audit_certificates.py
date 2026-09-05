"""Tests for the audit's certificate machinery.

The point of these tests is adversarial: an audit that cannot fail is not an
audit. Each test below breaks something and asserts the audit notices.
"""

import numpy as np
import pytest

import occurrence_i_audit as audit


@pytest.fixture(autouse=True)
def _clear_failures():
    audit._FAILURES.clear()
    yield
    audit._FAILURES.clear()


def test_certify_records_failure_above_tolerance(capsys):
    assert audit.certify("X", "too big", 1.0, tol=1e-12) is False
    assert len(audit._FAILURES) == 1
    assert "FAIL" in capsys.readouterr().out


def test_certify_passes_below_tolerance():
    assert audit.certify("X", "small", 1e-15, tol=1e-12) is True
    assert audit._FAILURES == []


def test_certify_equal_records_mismatch():
    assert audit.certify_equal("X", "count", 83, 84) is False
    assert len(audit._FAILURES) == 1


def test_gates_pass_on_the_real_algebra():
    assert audit.verify_gates(audit.OTAlgebra(dim=16), trials=25) is True
    assert audit._FAILURES == []


def test_gates_test_the_algebra_they_are_given_not_a_fresh_one():
    """Corrupting OT.C must be detected.

    Regression test: an earlier version of verify_gates built its own octonion
    table internally, so it validated a table the audit never used and passed
    even on a corrupted algebra.
    """
    ot = audit.OTAlgebra(dim=16)
    ot.C[1, 2, 3] += 0.5
    assert audit.verify_gates(ot, trials=25) is False
    assert audit._FAILURES


def test_gates_detect_corruption_confined_to_the_sedenion_block():
    """The four classical gates only exercise e_0..e_7.

    A sedenion-only corruption (e_9 * e_10) is invisible to them, so the audit
    must carry gates in dimension 16 as well.
    """
    ot = audit.OTAlgebra(dim=16)
    ot.C[9, 10, 3] += 0.3
    assert audit.verify_gates(ot, trials=25) is False
    assert any("S" in f for f in audit._FAILURES)


def test_main_returns_nonzero_when_gates_fail(monkeypatch):
    monkeypatch.setattr(audit, "verify_gates", lambda ot, **kw: False)
    assert audit.main() == 1


def test_main_returns_zero_on_a_clean_run():
    assert audit.main() == 0


def test_main_returns_nonzero_when_any_certificate_fails(monkeypatch):
    """A single failed certificate anywhere must poison the whole run."""
    monkeypatch.setattr(audit, "verify_gates", lambda ot, **kw: True)
    for name in (
        "test_fundamental_identities",
        "test_zero_divisor_graph",
        "test_no_autonomy",
        "test_invariant_measure",
        "test_ontological_compression",
        "test_first_dynamics",
        "test_strain_field",
        "test_alignment",
        "test_event_state_symmetry",
        "test_minimality_necessity",
    ):
        monkeypatch.setattr(audit, name, lambda ot: None)

    # One section reports a failed certificate.
    def failing(ot):
        audit.certify("BAD", "deliberately failing certificate", 1.0)

    monkeypatch.setattr(audit, "test_alignment", failing)
    assert audit.main() == 1


def test_main_does_not_print_success_when_a_certificate_fails(monkeypatch, capsys):
    """The old audit printed 'survives full algebraic audit' unconditionally."""
    monkeypatch.setattr(audit, "verify_gates", lambda ot, **kw: True)
    for name in (
        "test_fundamental_identities",
        "test_zero_divisor_graph",
        "test_no_autonomy",
        "test_invariant_measure",
        "test_ontological_compression",
        "test_first_dynamics",
        "test_strain_field",
        "test_event_state_symmetry",
        "test_minimality_necessity",
    ):
        monkeypatch.setattr(audit, name, lambda ot: None)
    monkeypatch.setattr(
        audit, "test_alignment", lambda ot: audit.certify("BAD", "fails", 1.0)
    )

    audit.main()
    out = capsys.readouterr().out
    assert "RESULT: FAIL" in out
    assert "RESULT: PASS" not in out


def test_sign_pair_forces_equilibrium_and_is_minimal():
    """Remark 3.6a: a 2-point sign pair gives E[M_z] = I exactly, and 2 is minimal.

    Guards the certificates C9j/C9k/C9l. The mechanism is that L_{e_k} is a signed
    permutation, so M_{e_k} = I and the cross terms cancel between the signs.
    """
    import numpy as np
    from topographo.ssd import SedenionAlgebra

    A = SedenionAlgebra(16, seed=42)
    d, sq = A.dim, 1.0 / np.sqrt(2.0)
    eye = np.eye(d)

    # every basis unit is orthogonal under left multiplication
    for k in range(1, d):
        L = A.Lop(eye[k])
        assert np.abs(L.T @ L - eye).max() < 1e-12

    # all 42 sign pairs average to the identity
    for i in range(1, 8):
        for j in range(1, 8):
            if i == j:
                continue
            zp, zm = np.zeros(d), np.zeros(d)
            zp[i] = zm[i] = sq
            zp[j + 8], zm[j + 8] = sq, -sq
            Lp, Lm = A.Lop(zp), A.Lop(zm)
            assert np.abs((Lp.T @ Lp + Lm.T @ Lm) / 2 - eye).max() < 1e-12

    # n = 1 cannot: the single-Kraus spectrum is {0^4, 1^8, 2^4}, never I
    z = A.basis_zero_divisors()[0]
    L = A.Lop(z)
    spec = np.sort(np.linalg.eigvalsh(L.T @ L))
    assert np.abs(spec - np.array([0.0] * 4 + [1.0] * 8 + [2.0] * 4)).max() < 1e-12


def test_pair_does_not_reproduce_the_channel():
    """Remark 3.6a, second half: Thm 3.13 does NOT inherit the 2-point shortcut.

    Guards C9m/C9n. Equilibrium is a first-moment condition on M (2 points);
    the channel is a second-moment condition on z (needs the design).
    """
    import numpy as np
    from topographo.ssd import SedenionAlgebra

    A = SedenionAlgebra(16, seed=42)
    d, sq = A.dim, 1.0 / np.sqrt(2.0)
    zp, zm = np.zeros(d), np.zeros(d)
    zp[1] = zm[1] = sq
    zp[10], zm[10] = sq, -sq

    P_W = np.eye(d)
    P_W[0, 0] = 0.0
    P_W[8, 8] = 0.0
    pair_moment = (np.outer(zp, zp) + np.outer(zm, zm)) / 2
    assert np.linalg.norm(pair_moment - P_W / 14) > 1e-2   # NOT Aut-invariant

    def channel(zs):
        return sum(np.kron(A.Lop(z).T, A.Lop(z).T) for z in zs) / len(zs)

    assert np.abs(channel([zp, zm]) - channel(A.basis_zero_divisors())).max() > 1e-2


def test_fourteen_points_reproduce_channel_and_are_minimal():
    """Remark 3.6a: a 14-point sign design reproduces Phi, and 14 is minimal."""
    import numpy as np
    from topographo.ssd import SedenionAlgebra

    A = SedenionAlgebra(16, seed=42)
    d, sq = A.dim, 1.0 / np.sqrt(2.0)
    eye = np.eye(d)

    # A cyclic derangement gives seven disjoint supports; both signs on each
    # support cancel cross terms and cover every coordinate of W exactly twice.
    z14 = []
    for i in range(1, 8):
        j = i % 7 + 1
        z14.extend((sq * (eye[i] + eye[j + 8]),
                    sq * (eye[i] - eye[j + 8])))

    P_W = eye.copy()
    P_W[0, 0] = P_W[8, 8] = 0.0
    moment14 = sum(np.outer(z, z) for z in z14) / len(z14)
    assert np.abs(moment14 - P_W / 14).max() < 1e-12

    def channel(zs):
        return sum(np.kron(A.Lop(z).T, A.Lop(z).T) for z in zs) / len(zs)

    z84 = A.basis_zero_divisors()
    assert np.abs(channel(z14) - channel(z84)).max() < 1e-12

    # Choi rank equals the dimension of the Kraus-vector span. Since L_z e0=z,
    # z -> L_z is injective on the 14-dimensional pencil W.
    kraus_vectors = np.stack([A.Lop(z).reshape(-1) for z in z84])
    assert np.linalg.matrix_rank(kraus_vectors) == 14
