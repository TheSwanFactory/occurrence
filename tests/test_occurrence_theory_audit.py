import occurrence_theory_audit as audit


def test_audit_gate_function_returns_true_and_reports_all_gates(capsys):
    result = audit.verify_gates(audit.OTAlgebra())

    output = capsys.readouterr().out
    assert result is True
    assert "[G1] Composition" in output
    assert "[G2] Antisymmetry" in output
    assert "[G3] Quadratic" in output
    assert "[G4] Moufang" in output
    assert "All gates pass" in output


def test_main_halts_when_gates_fail(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(audit, "verify_gates", lambda ot: False)
    monkeypatch.setattr(audit, "test_fundamental_identities", lambda ot: calls.append("fundamental"))

    audit.main()

    assert calls == []
    assert "HALTING: Gate verification failed." in capsys.readouterr().out


def test_main_runs_registered_audit_sections_after_gates_pass(monkeypatch):
    calls = []

    monkeypatch.setattr(audit, "verify_gates", lambda ot: calls.append("gates") or True)
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
        monkeypatch.setattr(audit, name, lambda ot, section=name: calls.append(section))

    audit.main()

    assert calls == [
        "gates",
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
    ]
