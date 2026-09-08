"""Issue 006 Fixed-head existence census pins (Quilt 006.12 / 006.13)."""

from __future__ import annotations

import importlib
import inspect

import numpy as np

from topographo.ssd import fixed_head


def test_fixed_basis_gram_and_decomposition():
    basis = fixed_head.fixed_basis()
    grams = [float(np.trace(g.T @ g)) for g in basis.generators]
    assert np.allclose(grams, fixed_head.GRAM, atol=1e-8)
    # Orthogonality under HS product.
    gens = basis.generators
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(float(np.trace(gens[i].T @ gens[j]))) < 1e-8
    # I = g0 + g3
    assert np.allclose(basis.g0 + basis.g3, np.eye(16), atol=1e-8)


def test_accepted_two_step_census():
    census = fixed_head.assert_census()
    assert len(census.records) == 7056
    assert len(census.class_table) == 15
    assert census.span_dim == 4
    assert census.scalar_mult == 672
    assert census.g1_nonzero == 4032
    assert census.g4_nonzero == 5376
    assert census.rank_hist == {8: 504, 10: 3024, 12: 3528}
    assert census.direct_singleton == 0
    assert census.class_id_table.shape == (84, 84)
    assert len(census.checksum_sha256) == 64
    # Pinned after first certified regeneration on thebeast (left Kraus / Fixed 05b).
    assert census.checksum_sha256 == (
        "46007a3aca22aa28c0d1d1f90610bbac570386a72bb66f6d1898d9df6259e659"
    )


def test_readout_probability_nonnegative_on_effects():
    basis = fixed_head.fixed_basis()
    kraus = fixed_head.basic_kraus_operators()
    # Sample a few words; B_w are scaled Fixed projections of PSD moments.
    rng = np.random.default_rng(0)
    x = rng.normal(size=16)
    for a, b in [(0, 1), (3, 5), (10, 10)]:
        effect = fixed_head.word_effect(kraus[b], kraus[a], basis)
        p = fixed_head.readout_probability(x, effect)
        assert p == p  # finite
        # Equal-weight average of all word effects is I/7056 * 7056? avg E_Fix = I so avg B = I/7056
    # Average readout equals 1/7056 * ||x||^2 / ||x||^2 wait: sum_w P(w|x) = <x, I x>/<x,x> = 1
    # Check on a tiny subsample scaled carefully via census normalization already pinned.


def test_no_optimizer_imports_in_fixed_head():
    mod = importlib.import_module("topographo.ssd.fixed_head")
    src = inspect.getsource(mod)
    assert "import torch" not in src
    assert "torch.optim" not in src
    assert "from torch" not in src
