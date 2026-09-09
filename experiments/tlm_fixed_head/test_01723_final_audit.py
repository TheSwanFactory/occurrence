#!/usr/bin/env python3
"""Regression tests for 017.23 catalogue-order scoring and graph classification."""
from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from learned_admissibility_01723_final_audit import (
    classify_rewired_mask,
    first_max_idx,
    synthetic_order_regressions,
)


class Test01723(unittest.TestCase):
    def test_first_max_is_leftmost_argmax(self):
        row = np.array([1.0, 5.0, 5.0, 2.0])
        self.assertEqual(first_max_idx(row), 1)  # leftmost of tied max
        self.assertEqual(first_max_idx(row), int(np.argmax(row)))

    def test_synthetic_tied_untied_order(self):
        r = synthetic_order_regressions()
        self.assertTrue(r["untied_invariant_under_reorder"], r)
        self.assertTrue(r["tied_can_change_selected_physical"], r)
        self.assertTrue(r["ok"], r)

    def test_physical_from_permuted_argmax_not_unpermuted(self):
        """Demonstrate the 017.22 bug pattern: unpermuted argmax would always match."""
        votes = np.zeros((2, 5), dtype=np.float64)
        # token0: unique max at physical 1
        votes[0, 1] = 3.0
        votes[0, 4] = 1.0
        # token1: tie between physical 0 and 2
        votes[1, 0] = 2.0
        votes[1, 2] = 2.0
        # Permutation that puts physical 2 before physical 0 in storage
        # storage positions: [2, 0, 1, 3, 4] means perm = [2,0,1,3,4]
        perm = np.array([2, 0, 1, 3, 4])
        storage = votes[:, perm]
        phys = [int(perm[first_max_idx(storage[t])]) for t in range(2)]
        unperm = [int(first_max_idx(votes[t])) for t in range(2)]
        # token0 unchanged
        self.assertEqual(phys[0], unperm[0])
        # token1: storage order picks physical 2 first (index 0 in storage), vs unperm leftmost 0
        self.assertEqual(phys[1], 2)
        self.assertEqual(unperm[1], 0)
        # Bug pattern: scoring unperm against truth would hide the reorder effect
        self.assertNotEqual(phys[1], unperm[1])

    def test_graph_classification_identical_iso_noniso(self):
        n = 6
        # Path graph
        native = np.zeros((n, n), dtype=np.int8)
        for i in range(n - 1):
            native[i, i + 1] = native[i + 1, i] = 1
        # Identical
        c0 = classify_rewired_mask(native, native.copy())
        self.assertEqual(c0["classification"], "identical_labeled")
        # Isomorphic under relabeling: reverse path (same labeled edges actually...)
        # Use a cycle vs path for non-iso; use relabeled copy for iso
        # Relabel: reverse nodes
        perm = list(range(n - 1, -1, -1))
        rewired_iso = np.zeros_like(native)
        # Build a different labeled tree that is still a path (shifted labels)
        # Actually reverse path on same vertices has SAME undirected edges as path.
        # Use: star vs path (non-iso), and a 2-swap isomorphic labeled-different graph.
        # For isomorphic-but-not-identical: take C4 vs another C4 labeling on 4 nodes embedded in 6.
        native4 = np.zeros((4, 4), dtype=np.int8)
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            native4[i, j] = native4[j, i] = 1
        # Relabel 0->0, 1->2, 2->1, 3->3: edges (0,2),(2,1),(1,3),(3,0) — different labeled, same cycle
        rewired4 = np.zeros((4, 4), dtype=np.int8)
        for i, j in [(0, 2), (2, 1), (1, 3), (3, 0)]:
            rewired4[i, j] = rewired4[j, i] = 1
        c1 = classify_rewired_mask(native4, rewired4)
        self.assertFalse(c1["identical_labeled"])
        self.assertTrue(c1["isomorphic_under_relabeling"])
        self.assertEqual(c1["classification"], "isomorphic_under_relabeling")
        # Non-isomorphic: path vs star
        star = np.zeros((n, n), dtype=np.int8)
        for j in range(1, n):
            star[0, j] = star[j, 0] = 1
        c2 = classify_rewired_mask(native, star)
        self.assertEqual(c2["classification"], "non_isomorphic")
        self.assertEqual(c2["native_self_loops"], 0)
        self.assertEqual(c2["rewired_self_loops"], 0)


if __name__ == "__main__":
    unittest.main()
