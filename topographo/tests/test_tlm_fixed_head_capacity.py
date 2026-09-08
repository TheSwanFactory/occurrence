"""CI-safe capacity probe for the Fixed-head experiment (no torch)."""

from __future__ import annotations

from experiments.tlm_fixed_head.capacity import analyze_capacity
from topographo.ssd import fixed_head


def test_capacity_report_p13_and_bottleneck_flag_p97():
    census = fixed_head.two_step_census()
    r13 = analyze_capacity(13, census)
    r97 = analyze_capacity(97, census)
    assert r13.n_classes == 15
    assert r13.n_words == 7056
    assert r13.n_modular_outputs == 13
    assert r97.n_modular_outputs == 97
    # 15 << 97 ⇒ information bottleneck expected under the configured embedding.
    assert r97.fatal_for_p
    assert r97.class_bits < r97.n_modular_outputs  # 15 classes cannot one-hot 97 outputs
