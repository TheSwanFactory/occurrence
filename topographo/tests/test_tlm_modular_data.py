"""Alternate-role modular data partitioning (005.07 §3) — no torch required."""

from __future__ import annotations

from experiments.tlm_modular import data as tlm_data


def test_partition_by_complete_triples_no_leakage():
    split = tlm_data.partition_modular_triples(p=13, train_fraction=0.5, seed=0)
    train_triples = set(split.train_triples)
    test_triples = set(split.test_triples)
    assert train_triples.isdisjoint(test_triples)
    assert len(train_triples) + len(test_triples) == len(split.all_triples)
    # Forward examples drawn only from their triple partition.
    for a, b, c in split.forward_train:
        assert tlm_data.canonical_triple(a, b, c, p=13) in train_triples
    for a, b, c in split.forward_test:
        assert tlm_data.canonical_triple(a, b, c, p=13) in test_triples


def test_alternate_role_views_cover_three_queries():
    split = tlm_data.partition_modular_triples(p=13, train_fraction=0.5, seed=1)
    roles = {ex.role for ex in split.alternate_role_test}
    assert roles == {"c", "b", "a"}
    for ex in split.alternate_role_test:
        a, b, c = ex.a, ex.b, ex.c
        assert (a + b) % 13 == c
        if ex.role == "c":
            assert ex.inputs == (a, b) and ex.target == c
        elif ex.role == "b":
            assert ex.inputs == (a, c) and ex.target == b
        else:
            assert ex.inputs == (b, c) and ex.target == a
