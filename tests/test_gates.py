from topographo.core import GateResult, verify_gates


def test_verify_gates_returns_named_passing_results():
    results = verify_gates()

    assert [result.name for result in results] == [
        "composition",
        "antisymmetry",
        "quadratic",
        "moufang",
    ]
    assert all(result.error == 0.0 for result in results)
    assert all(result.passed for result in results)


def test_gate_result_uses_strict_tolerance():
    assert GateResult("below", 0.49, tolerance=0.5).passed
    assert not GateResult("equal", 0.5, tolerance=0.5).passed


def test_verify_gates_propagates_custom_tolerance():
    results = verify_gates(tolerance=1.0)

    assert {result.tolerance for result in results} == {1.0}
