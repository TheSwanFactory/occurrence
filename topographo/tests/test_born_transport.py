from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from topographo import (
    BornTransportAnnihilation,
    BornTransportResult,
    ot_born_transport,
)


@pytest.fixture(scope="module")
def kraus_family() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).resolve().parents[2] / "data" / "kraus84.npz"
    with np.load(path) as data:
        return data["K"].copy(), data["mu"].copy()


@pytest.fixture(scope="module")
def canonical_complex_structure(
    kraus_family: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    operators, weights = kraus_family
    channel = sum(
        weight * np.kron(operator, operator)
        for weight, operator in zip(weights, operators, strict=True)
    )
    channel = (channel + channel.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(channel)
    index = int(np.argmin(np.abs(eigenvalues + 1.0)))
    clock = eigenvectors[:, index].reshape(16, 16)
    clock = (clock - clock.T) / 2.0
    return clock * (4.0 / np.linalg.norm(clock))


@pytest.fixture(scope="module")
def dense_unit_state() -> np.ndarray:
    state = np.arange(1.0, 17.0)
    return state / np.linalg.norm(state)


def direct_formula(
    state: np.ndarray,
    operator: np.ndarray,
    clock: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float, float]:
    e0 = np.eye(16)[0]
    transported = operator @ state
    cost = float(transported @ transported)
    transported_state = transported / np.sqrt(cost)
    event = operator @ e0
    spine_share = float(
        (transported_state @ e0) ** 2
        + (transported_state @ (clock @ e0)) ** 2
    )
    numerator = float((event @ state) ** 2 + ((clock @ event) @ state) ** 2)
    return (
        transported_state,
        cost,
        cost - 1.0,
        spine_share,
        numerator,
        spine_share * cost - numerator,
    )


def test_all_committed_operators_agree_with_direct_formula(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    operators, _ = kraus_family

    for operator in operators:
        result = ot_born_transport(
            dense_unit_state,
            operator,
            canonical_complex_structure,
        )
        expected = direct_formula(
            dense_unit_state,
            operator,
            canonical_complex_structure,
        )

        assert isinstance(result, BornTransportResult)
        np.testing.assert_allclose(
            result.transported_state,
            expected[0],
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            [
                result.normalization_cost,
                result.event_strain,
                result.post_transition_spine_share,
                result.hermitian_numerator,
                result.identity_residual,
            ],
            expected[1:],
            rtol=1e-13,
            atol=1e-14,
        )
        assert abs(result.identity_residual) < 1e-12


def test_state_sign_reverses_only_the_transported_state(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    operator = kraus_family[0][13]
    positive = ot_born_transport(
        dense_unit_state,
        operator,
        canonical_complex_structure,
    )
    negative = ot_born_transport(
        -dense_unit_state,
        operator,
        canonical_complex_structure,
    )

    np.testing.assert_allclose(
        negative.transported_state,
        -positive.transported_state,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        [
            negative.normalization_cost,
            negative.event_strain,
            negative.post_transition_spine_share,
            negative.hermitian_numerator,
            negative.identity_residual,
        ],
        [
            positive.normalization_cost,
            positive.event_strain,
            positive.post_transition_spine_share,
            positive.hermitian_numerator,
            positive.identity_residual,
        ],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("scale", [0.5, 1.001, 2.0])
def test_scaled_nonunit_states_are_rejected(
    scale: float,
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    with pytest.raises(ValueError, match="unit vector"):
        ot_born_transport(
            scale * dense_unit_state,
            kraus_family[0][0],
            canonical_complex_structure,
        )


def test_exact_fixture_annihilation_is_typed_before_division(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
):
    operators, _ = kraus_family
    e0 = np.eye(16)[0]
    state = operators[47] @ e0
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-15)
    np.testing.assert_array_equal(operators[0] @ state, np.zeros(16))

    with pytest.raises(BornTransportAnnihilation) as caught:
        ot_born_transport(state, operators[0], canonical_complex_structure)

    assert caught.value.kind == "exact"
    assert caught.value.cost == 0.0
    assert caught.value.threshold == 1e-12


def test_near_annihilation_and_exact_only_threshold(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
):
    operators, _ = kraus_family
    e0 = np.eye(16)[0]
    state = operators[47] @ e0 + 1e-7 * e0
    state /= np.linalg.norm(state)

    with pytest.raises(BornTransportAnnihilation) as caught:
        ot_born_transport(state, operators[0], canonical_complex_structure)

    assert caught.value.kind == "near"
    assert 0.0 < caught.value.cost <= caught.value.threshold
    assert caught.value.threshold == 1e-12

    exact_only = ot_born_transport(
        state,
        operators[0],
        canonical_complex_structure,
        annihilation_tolerance=0.0,
    )
    assert exact_only.normalization_cost == pytest.approx(caught.value.cost)
    np.testing.assert_allclose(np.linalg.norm(exact_only.transported_state), 1.0)

    with pytest.raises(BornTransportAnnihilation) as boundary:
        ot_born_transport(
            state,
            operators[0],
            canonical_complex_structure,
            annihilation_tolerance=caught.value.cost,
        )
    assert boundary.value.kind == "near"


def test_inputs_are_not_mutated_and_output_is_read_only(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    state = dense_unit_state.copy()
    operator = kraus_family[0][5].copy()
    clock = canonical_complex_structure.copy()
    state_before = state.copy()
    operator_before = operator.copy()
    clock_before = clock.copy()

    result = ot_born_transport(state, operator, clock)

    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(operator, operator_before)
    np.testing.assert_array_equal(clock, clock_before)
    assert not result.transported_state.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.transported_state[0] = 0.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.transported_state.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.normalization_cost = 0.0


@pytest.mark.parametrize(
    ("state_shape", "operator_shape", "clock_shape"),
    [
        ((15,), (16, 16), (16, 16)),
        ((16, 1), (16, 16), (16, 16)),
        ((16,), (15, 15), (16, 16)),
        ((16,), (16, 15), (16, 16)),
        ((16,), (16, 16), (15, 15)),
        ((16,), (16, 16), (15, 16)),
    ],
)
def test_malformed_shapes_are_rejected(
    state_shape: tuple[int, ...],
    operator_shape: tuple[int, ...],
    clock_shape: tuple[int, ...],
):
    state = np.ones(state_shape)
    if state.shape == (16,):
        state /= np.linalg.norm(state)
    with pytest.raises(ValueError, match="shape"):
        ot_born_transport(
            state,
            np.ones(operator_shape),
            np.ones(clock_shape),
        )


@pytest.mark.parametrize("target", ["state", "operator", "clock"])
def test_complex_inputs_are_rejected(target: str, dense_unit_state: np.ndarray):
    state = dense_unit_state.copy()
    operator = np.eye(16)
    clock = np.eye(16)
    if target == "state":
        state = state.astype(np.complex128)
    elif target == "operator":
        operator = operator.astype(np.complex128)
    else:
        clock = clock.astype(np.complex128)

    with pytest.raises(ValueError, match="real-valued"):
        ot_born_transport(state, operator, clock)


@pytest.mark.parametrize("target", ["state", "operator", "clock"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected(
    target: str,
    nonfinite: float,
    dense_unit_state: np.ndarray,
):
    state = dense_unit_state.copy()
    operator = np.eye(16)
    clock = np.eye(16)
    if target == "state":
        state[3] = nonfinite
    elif target == "operator":
        operator[3, 4] = nonfinite
    else:
        clock[3, 4] = nonfinite

    with pytest.raises(ValueError, match="finite values"):
        ot_born_transport(state, operator, clock)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_annihilation_tolerances_are_rejected(
    tolerance: float,
    dense_unit_state: np.ndarray,
):
    with pytest.raises(ValueError, match="nonnegative finite"):
        ot_born_transport(
            dense_unit_state,
            np.eye(16),
            np.eye(16),
            annihilation_tolerance=tolerance,
        )


def test_non_ot_structures_are_rejected(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    operator = kraus_family[0][0]

    with pytest.raises(ValueError, match="settlement_operator must be antisymmetric"):
        ot_born_transport(
            dense_unit_state,
            np.eye(16),
            canonical_complex_structure,
        )
    with pytest.raises(ValueError, match="complex_structure must be antisymmetric"):
        ot_born_transport(dense_unit_state, operator, np.eye(16))
    with pytest.raises(ValueError, match="recover a unit event"):
        ot_born_transport(
            dense_unit_state,
            2.0 * operator,
            canonical_complex_structure,
        )


def test_individually_valid_but_incompatible_presentation_is_rejected(
    kraus_family: tuple[np.ndarray, np.ndarray],
    dense_unit_state: np.ndarray,
):
    standard_clock = np.zeros((16, 16))
    for index in range(0, 16, 2):
        standard_clock[index, index + 1] = -1.0
        standard_clock[index + 1, index] = 1.0
    orthogonal, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((16, 16)))
    incompatible_clock = orthogonal @ standard_clock @ orthogonal.T

    np.testing.assert_allclose(incompatible_clock.T, -incompatible_clock, atol=1e-14)
    np.testing.assert_allclose(
        incompatible_clock @ incompatible_clock,
        -np.eye(16),
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="not a compatible OT Born transport"):
        ot_born_transport(
            dense_unit_state,
            kraus_family[0][0],
            incompatible_clock,
        )


def test_nonfinite_derived_transport_is_rejected(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
    dense_unit_state: np.ndarray,
):
    operator = kraus_family[0][0].copy()
    operator[0, 1] = 1e308
    operator[1, 0] = -1e308
    with pytest.raises(ValueError, match="recover a unit event"):
        ot_born_transport(dense_unit_state, operator, canonical_complex_structure)


@pytest.mark.parametrize(
    "tolerance",
    [complex(0.0), complex(1e-12, 1e-12), np.complex128(0.0), np.complex128(1e-12 + 1e-12j)],
)
def test_complex_annihilation_tolerances_are_rejected(
    tolerance: complex,
    dense_unit_state: np.ndarray,
):
    with pytest.raises(ValueError, match="real nonnegative finite"):
        ot_born_transport(
            dense_unit_state,
            np.eye(16),
            np.eye(16),
            annihilation_tolerance=tolerance,
        )


def test_exact_only_mode_rejects_unrepresentable_squared_cost(
    kraus_family: tuple[np.ndarray, np.ndarray],
    canonical_complex_structure: np.ndarray,
):
    operators, _ = kraus_family
    e0 = np.eye(16)[0]
    state = operators[47] @ e0 + 1e-200 * e0
    state /= np.linalg.norm(state)

    with pytest.raises(BornTransportAnnihilation) as near:
        ot_born_transport(state, operators[0], canonical_complex_structure)
    assert near.value.kind == "near"
    assert near.value.cost == 0.0

    with pytest.raises(ValueError, match="underflowed to zero"):
        ot_born_transport(
            state,
            operators[0],
            canonical_complex_structure,
            annihilation_tolerance=0.0,
        )
