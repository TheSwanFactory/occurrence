"""Hard FIPS adapter + declared STE learning route (005.07 section 5)."""

from __future__ import annotations

import importlib
import inspect

from topographo.ssd import fips_adapter, fips_basic


def test_hard_forward_matches_fips_fixtures():
    index = fips_adapter.EventIndex.build()
    table = fips_adapter.third_index_table(index)
    assert table.shape == (84, 84)
    assert int((table >= 0).sum()) == 336
    for a, b, t in fips_basic.ORDERED_EDGES:
        assert fips_adapter.hard_third(a, b) == t
        k = fips_adapter.hard_third_index(index.encode(a), index.encode(b), table)
        assert index.decode(k) == t


def test_partner_mask_matches_admissible():
    index = fips_adapter.EventIndex.build()
    mask = fips_adapter.partner_mask(index)
    for i, a in enumerate(index.events):
        partners = set(fips_basic.partners(a))
        for j, b in enumerate(index.events):
            assert bool(mask[i, j]) == (b in partners)


def test_supervised_phi_updates_through_declared_route():
    result = fips_adapter.supervised_phi_smoke(z_idx=0, target_w_idx=0, steps=50)
    assert result.hard_matches_fixtures
    assert result.phi_updated
    assert result.loss_after < result.loss_before
    assert "STE" in result.route


def test_no_optimizer_or_torch_imports_in_runtime_adapters():
    for mod_name in (
        "topographo.ssd.outcome_runtime",
        "topographo.ssd.fips_basic",
        "topographo.ssd.fips_adapter",
        "topographo.ssd.structural_control",
        "topographo.ssd.seal",
    ):
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        assert "import torch" not in src
        assert "torch.optim" not in src
        assert "from torch" not in src
