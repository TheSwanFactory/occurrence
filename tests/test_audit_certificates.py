"""Tests for the audit's certificate machinery.

The point of these tests is adversarial: an audit that cannot fail is not an
audit. Each test below breaks something and asserts the audit notices.
"""

import numpy as np
import pytest

import occurrence_theory_audit as audit


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
