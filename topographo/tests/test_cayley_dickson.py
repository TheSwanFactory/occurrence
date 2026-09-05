import hashlib

import numpy as np
import pytest

from topographo.core import (
    CayleyDicksonAlgebra,
    cayley_dickson_table,
    signed_basis_table,
)


def test_cayley_dickson_tables_require_positive_power_of_two():
    for dim in (0, -2, 3, 6):
        with pytest.raises(ValueError, match="positive power of two"):
            signed_basis_table(dim)
        with pytest.raises(ValueError, match="positive power of two"):
            cayley_dickson_table(dim)


def test_signed_basis_table_is_exact_immutable_and_canonical():
    signed = signed_basis_table(16)
    tensor = cayley_dickson_table(16)

    assert signed is signed_basis_table(16)
    assert tensor.dtype == np.float64
    for left, row in enumerate(signed):
        for right, (result, sign) in enumerate(row):
            assert 0 <= result < 16
            assert sign in (-1, 1)
            assert tensor[left, right, result] == sign
            assert np.count_nonzero(tensor[left, right]) == 1

    digest = hashlib.sha256(tensor.astype(np.int8).tobytes()).hexdigest()
    assert digest == "56bc5617d33423fa49c943424d8fa39d4677aaa9b830f3c7a11a52c0942c1e21"


def test_quaternion_basis_multiplication_matches_convention():
    algebra = CayleyDicksonAlgebra(4)
    e = algebra.e

    expected = {
        (0, 0): (1, 0),
        (0, 1): (1, 1),
        (1, 0): (1, 1),
        (1, 1): (-1, 0),
        (1, 2): (1, 3),
        (2, 1): (-1, 3),
        (1, 3): (-1, 2),
        (3, 1): (1, 2),
        (2, 3): (1, 1),
        (3, 2): (-1, 1),
    }

    for (left, right), (sign, basis_index) in expected.items():
        actual = algebra.mul(e[left], e[right])
        wanted = sign * e[basis_index]
        np.testing.assert_array_equal(actual, wanted)


def test_operator_wrappers_match_direct_multiplication():
    algebra = CayleyDicksonAlgebra(8)
    x = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 1.5, 0.0, 0.125])
    y = np.array([1.25, 0.5, -0.5, 0.75, 0.25, 0.0, -1.0, 2.0])

    np.testing.assert_allclose(algebra.left_operator(x) @ y, algebra.mul(x, y))
    np.testing.assert_allclose(algebra.right_operator(y) @ x, algebra.mul(x, y))
    np.testing.assert_allclose(algebra.Lop(x), algebra.left_operator(x))
    np.testing.assert_allclose(algebra.Rop(y), algebra.right_operator(y))


def test_stepv_vectorizes_left_settlement():
    algebra = CayleyDicksonAlgebra(8)
    states = np.array([algebra.e[1], algebra.e[2], algebra.e[3]])
    events = np.array([algebra.e[2], algebra.e[3], algebra.e[4]])

    expected = np.array([algebra.mul(event, state) for event, state in zip(events, states)])

    np.testing.assert_array_equal(algebra.stepv(states, events), expected)


def test_metric_operator_and_alternator_definitions():
    algebra = CayleyDicksonAlgebra(8)
    x = np.array([0.0, 1.0, -0.5, 0.25, 0.75, 0.0, -1.25, 0.5])
    left = algebra.left_operator(x)

    np.testing.assert_allclose(algebra.metric_operator(x), left.T @ left)
    np.testing.assert_allclose(
        algebra.alternator(x),
        algebra.left_operator(algebra.mul(x, x)) - left @ left,
    )


def test_conjugate_flips_nonreal_coordinates():
    algebra = CayleyDicksonAlgebra(8)
    x = np.arange(8, dtype=float)

    np.testing.assert_array_equal(
        algebra.conjugate(x), np.array([0, -1, -2, -3, -4, -5, -6, -7])
    )
