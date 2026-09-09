#!/usr/bin/env python3
"""Regression tests for 017.21 Phi tensor path + ambient/native programs."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from learned_admissibility_01709 import Geometry, SedenionAlgebra, basis_vec, densest_vocab_indices, N_VOCAB
from learned_admissibility_01719_exact_eval import (
    exact_from_float_idx,
    exact_p_grp,
    exact_p_seq,
    exact_value_to_float,
    float_to_exact_map,
    rays_equal,
)
from learned_admissibility_01721_structural_controls import (
    PhiGeometry,
    audit_tensor_phi,
    exact_p_grp_ambient,
    mul_t_phi,
    rewire_degree_preserving,
)
from topographo.ssd import exact, fips_basic


class Test01721(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alg = SedenionAlgebra()
        cls.geo = Geometry(cls.alg)
        cls.pgeo = PhiGeometry(cls.geo)
        cls.fte = float_to_exact_map(cls.geo)

    def test_tensor_phi_exhaustive(self):
        a = audit_tensor_phi(self.pgeo, self.fte)
        self.assertTrue(a["phi_ok"], a)
        self.assertEqual(a["basis_256"]["phi_disagreements"], 0)
        self.assertEqual(a["catalogue_84x84"]["phi_disagreements"], 0)
        self.assertEqual(a["programs_at_e4"]["P_seq_phi_disagreements"], 0)
        self.assertEqual(a["programs_at_e4"]["P_grp_cyc_phi_disagreements"], 0)

    def test_ambient_extends_cyc(self):
        re = exact.basis(4)
        true_idx = densest_vocab_indices(self.geo, N_VOCAB)
        cyc = amb = amb_only = 0
        for i in true_idx:
            for j in true_idx:
                ae = exact_from_float_idx(int(i), self.fte)
                be = exact_from_float_idx(int(j), self.fte)
                c = fips_basic.admissible(ae, be)
                pga = exact_p_grp_ambient(ae, be, re)
                cyc += int(c)
                amb += int(pga is not None)
                amb_only += int((pga is not None) and not c)
                if c:
                    # native grp endpoint equals ambient on Cyc edges
                    pgn = exact_p_grp(ae, be, re)
                    if pgn is not None:
                        self.assertIsNotNone(pga)
                        from topographo.ssd import projective
                        self.assertTrue(projective.equivalent(pgn, pga))
        self.assertGreater(amb, cyc)
        self.assertGreater(amb_only, 0)

    def test_rewire_preserves_degree(self):
        true_idx = densest_vocab_indices(self.geo, N_VOCAB)
        n = N_VOCAB
        mask = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                ae = exact_from_float_idx(int(true_idx[i]), self.fte)
                be = exact_from_float_idx(int(true_idx[j]), self.fte)
                mask[i, j] = int(fips_basic.admissible(ae, be))
        info = rewire_degree_preserving(mask, seed=101)
        self.assertTrue(info["degree_sequence_preserved"])
        self.assertTrue(info["non_isomorphic_edge_set"] or info["swaps_done"] == 0)

    def test_phi_mul_t_matches_exact_basis(self):
        M_t = self.pgeo.M_t
        for i in range(16):
            for j in range(16):
                pe = exact.mul(exact.basis(i), exact.basis(j))
                af = torch.tensor(basis_vec(i), dtype=torch.float32)
                bf = torch.tensor(basis_vec(j), dtype=torch.float32)
                pp = mul_t_phi(af, bf, M_t).numpy()
                if pe == exact.zero():
                    self.assertLess(float(np.linalg.norm(pp)), 1e-8)
                else:
                    self.assertTrue(rays_equal(pp, exact_value_to_float(pe)))


if __name__ == "__main__":
    unittest.main()
