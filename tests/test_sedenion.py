import numpy as np
import pytest

from topographo.core import CayleyDicksonAlgebra
from topographo.ssd import SedenionAlgebra, average_metric_operator


def test_sedenion_algebra_fixes_dimension():
    algebra = SedenionAlgebra()

    assert algebra.dim == 16
    assert algebra.C.shape == (16, 16, 16)
    assert algebra.e.shape == (16, 16)


def test_basis_zero_divisors_are_the_full_unit_crack_design():
    algebra = SedenionAlgebra()
    events = algebra.basis_zero_divisors()

    assert events.shape == (84, 16)
    np.testing.assert_allclose(np.linalg.norm(events, axis=1), 1.0)

    smallest_singular_values = [
        np.linalg.svd(algebra.left_operator(event), compute_uv=False)[-1]
        for event in events
    ]
    assert max(smallest_singular_values) < 1e-9


def test_average_metric_operator_rejects_empty_events():
    algebra = SedenionAlgebra()

    with pytest.raises(ValueError, match="non-empty"):
        average_metric_operator(algebra, np.empty((0, algebra.dim)))


def test_average_metric_operator_is_identity_on_full_basis_crack_design():
    algebra = SedenionAlgebra()
    events = algebra.basis_zero_divisors()

    mean_metric = average_metric_operator(algebra, events)

    np.testing.assert_allclose(mean_metric, np.eye(algebra.dim), atol=1e-12)


def test_basis_zero_divisor_sampling_is_seeded_and_uses_design_points():
    first = SedenionAlgebra(seed=123).sample_basis_zero_divisors(10)
    second = SedenionAlgebra(seed=123).sample_basis_zero_divisors(10)
    design = SedenionAlgebra().basis_zero_divisors()

    np.testing.assert_array_equal(first, second)
    assert all(any(np.array_equal(sample, event) for event in design) for sample in first)


def test_pure_pair_sampling_returns_unit_orthogonal_pure_pairs():
    algebra = SedenionAlgebra(seed=1)
    events = algebra.sample_pure_pair(25)

    lower = events[:, :8]
    upper = events[:, 8:]

    assert events.shape == (25, 16)
    np.testing.assert_allclose(np.linalg.norm(events, axis=1), 1.0)
    np.testing.assert_allclose(events[:, 0], 0.0)
    np.testing.assert_allclose(events[:, 8], 0.0)
    np.testing.assert_allclose(np.sum(lower * upper, axis=1), 0.0, atol=1e-12)


def test_pure_pair_sampling_is_sedenion_only():
    with pytest.raises(ValueError, match="dim=16"):
        CayleyDicksonAlgebra(8).sample_pure_pair(1)
