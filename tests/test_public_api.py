import topographo
import topographo.core as core
import topographo.ssd as ssd


def test_top_level_exports_core_api():
    assert topographo.CayleyDicksonAlgebra is core.CayleyDicksonAlgebra
    assert topographo.GateResult is core.GateResult
    assert topographo.cayley_dickson_table is core.cayley_dickson_table
    assert topographo.verify_gates is core.verify_gates


def test_subpackage_exports_are_explicit():
    assert set(core.__all__) == {
        "CayleyDicksonAlgebra",
        "GateResult",
        "cayley_dickson_table",
        "verify_gates",
    }
    assert set(ssd.__all__) == {
        "SedenionAlgebra",
        "average_metric_operator",
    }
