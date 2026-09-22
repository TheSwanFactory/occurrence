import topographo
import topographo.born_transport as born_transport
import topographo.core as core
import topographo.ssd as ssd


def test_top_level_exports_core_api():
    assert topographo.CayleyDicksonAlgebra is core.CayleyDicksonAlgebra
    assert topographo.GateResult is core.GateResult
    assert topographo.cayley_dickson_table is core.cayley_dickson_table
    assert topographo.signed_basis_table is core.signed_basis_table
    assert topographo.verify_gates is core.verify_gates


def test_top_level_exports_born_transport_api():
    assert (
        topographo.BornTransportAnnihilation
        is born_transport.BornTransportAnnihilation
    )
    assert topographo.BornTransportResult is born_transport.BornTransportResult
    assert topographo.ot_born_transport is born_transport.ot_born_transport
    assert set(born_transport.__all__) == {
        "BornTransportAnnihilation",
        "BornTransportResult",
        "ot_born_transport",
    }


def test_subpackage_exports_are_explicit():
    assert set(core.__all__) == {
        "CayleyDicksonAlgebra",
        "ExactCayleyDicksonAlgebra",
        "GateResult",
        "cayley_dickson_table",
        "f2_groups",
        "signed_basis_table",
        "verify_gates",
    }
    assert set(ssd.__all__) == {
        "SedenionAlgebra",
        "average_metric_operator",
        "codec",
        "exact",
        "exact_machine",
        "fips_adapter",
        "fips_basic",
        "fixed_head",
        "frames",
        "machine",
        "observers",
        "outcome_runtime",
        "program",
        "projective",
        "seal",
        "structural_control",
    }
