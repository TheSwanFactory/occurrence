#!/usr/bin/env python3
"""Regression tests for 017.19 exact evaluator reconciliation.

Covers §D required cases + every identified mismatch cause.
"""
from __future__ import annotations

import unittest

import numpy as np

from learned_admissibility_01709 import (
    Geometry,
    SedenionAlgebra,
    basis_vec,
    densest_vocab_indices,
    N_VOCAB,
    projective_key,
    rays_equal,
    verify_witness,
)
from learned_admissibility_01719_exact_eval import (
    audit_basis_products,
    audit_catalogue_products,
    audit_programs_at_e4,
    exact_from_float_idx,
    exact_occ,
    exact_p_grp,
    exact_p_seq,
    exact_rays_equal,
    exact_value_to_float,
    float_mul_with_phi,
    float_to_exact_map,
    minimal_mismatch_witness,
    phi_signs,
    preferred_target_exact,
)
from topographo.ssd import exact, fips_basic, projective


class Test01719ExactEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alg = SedenionAlgebra()
        cls.geo = Geometry(cls.alg)
        cls.fte = float_to_exact_map(cls.geo)
        cls.re = exact.basis(4)

    def test_float_to_exact_map_bijective(self):
        self.assertEqual(len(self.fte), 84)
        self.assertEqual(len(set(self.fte)), 84)

    def test_phi_signs_match_exact_module(self):
        s = phi_signs()
        self.assertEqual(s.shape, (16,))
        self.assertEqual(int(s[0]), 1)
        self.assertTrue(np.all(s[1:8] == -1))
        self.assertTrue(np.all(s[8:] == 1))

    def test_mismatch_cause_core_C_on_value_coords(self):
        """Identified cause: raw float mul ≠ exact; Phi transport repairs it."""
        w = minimal_mismatch_witness(self.geo, self.fte, self.alg)
        self.assertFalse(w["raw_matches_exact"])
        self.assertTrue(w["phi_matches_exact"])
        cat = audit_catalogue_products(self.geo, self.fte, self.alg)
        self.assertGreater(cat["raw_float_product_disagreements"], 0)
        self.assertEqual(cat["phi_float_product_disagreements"], 0)
        self.assertEqual(cat["cyc_domain_float_vs_fips_admissible_disagreements"], 0)
        prog = audit_programs_at_e4(self.geo, self.fte, self.alg)
        self.assertGreater(prog["P_seq_raw_disagreements"], 0)
        self.assertEqual(prog["P_seq_phi_disagreements"], 0)
        self.assertEqual(prog["P_grp_phi_disagreements"], 0)

    def test_basis_products_phi(self):
        b = audit_basis_products(self.alg)
        self.assertEqual(b["phi_float_vs_exact_disagreements"], 0)

    def test_independent_nonzero_rescaling_and_sign_invariance(self):
        a = fips_basic.EVENTS[self.fte[1]]
        b = fips_basic.EVENTS[self.fte[59]]
        r = self.re
        base = exact_p_seq(a, b, r)
        self.assertIsNotNone(base)
        for sa, sb, sr in [(2, 1, 1), (1, -3, 1), (1, 1, 5), (-1, -1, 1), (7, -2, 3)]:
            aa = exact.scale(sa, a)
            bb = exact.scale(sb, b)
            rr = exact.scale(sr, r)
            out = exact_p_seq(aa, bb, rr)
            self.assertTrue(exact_rays_equal(base, out), msg=(sa, sb, sr))

    def test_exact_zeros_and_undefined_intermediates(self):
        z = exact.zero()
        e1 = exact.basis(1)
        self.assertIsNone(exact_occ(z, e1))
        self.assertFalse(exact_rays_equal(z, z))  # zero has no ray
        self.assertFalse(exact_rays_equal(None, e1))
        self.assertFalse(exact_rays_equal(e1, None))

    def test_legal_vs_off_domain_grouped(self):
        # Find an inadmissible pair among catalogue
        found_bad = found_good = False
        for i in range(84):
            for j in range(84):
                a = exact_from_float_idx(i, self.fte)
                b = exact_from_float_idx(j, self.fte)
                if not fips_basic.admissible(a, b):
                    self.assertIsNone(exact_p_grp(a, b, self.re))
                    found_bad = True
                else:
                    # may still be undefined if Occ fails, but call is legal domain
                    _ = exact_p_grp(a, b, self.re)
                    found_good = True
                if found_bad and found_good:
                    return
        self.fail("did not find both admissible and inadmissible pairs")

    def test_01707_same_leaves_different_tree_witness(self):
        """Exact path must reproduce P_seq ≠ P_grp on (e1+e10, e4+e15, e4)."""
        z = exact.add(exact.basis(1), exact.basis(10))
        w = exact.add(exact.basis(4), exact.basis(15))
        x = exact.basis(4)
        # P_seq = Occ(z, Occ(w, x)) = z*(w*x)
        wx = exact.mul(w, x)
        p_seq = exact.mul(z, wx)
        # P_grp = Occ(Cyc(z,w), x) = (z*w)*x
        zw = exact.mul(z, w)
        p_grp = exact.mul(zw, x)
        self.assertFalse(exact_rays_equal(p_seq, p_grp))
        # Expected rays: [e1] vs [e1+e10]
        self.assertTrue(exact_rays_equal(p_seq, exact.basis(1)))
        self.assertTrue(exact_rays_equal(p_grp, z))
        # Float legacy witness still disagrees (may or may not match exact keys)
        wit = verify_witness(self.alg, self.geo.M)
        self.assertTrue(wit["disagree"])

    def test_target_coordinate_transport_from_true_idx(self):
        true_idx = densest_vocab_indices(self.geo, N_VOCAB)
        # For every vocab pair with an exact preferred target, target must equal
        # the preferred program endpoint under true denotations.
        n_ok = 0
        for i in range(N_VOCAB):
            for j in range(N_VOCAB):
                ti, tj = int(true_idx[i]), int(true_idx[j])
                a = exact_from_float_idx(ti, self.fte)
                b = exact_from_float_idx(tj, self.fte)
                tgt, br, _, _ = preferred_target_exact(a, b, self.re)
                if tgt is None:
                    continue
                if br == "seq":
                    self.assertTrue(exact_rays_equal(exact_p_seq(a, b, self.re), tgt))
                else:
                    self.assertTrue(exact_rays_equal(exact_p_grp(a, b, self.re), tgt))
                n_ok += 1
        self.assertGreater(n_ok, 100)

    def test_H_ge_A_and_four_way_identities_on_toy(self):
        """Structural identities on a tiny hand evaluation."""
        true_idx = densest_vocab_indices(self.geo, N_VOCAB)
        # Use true denotations: H should be 1 for all pairs with defined preferred targets
        s = g = h = p = 0
        counts = {"neither": 0, "seq_only": 0, "grp_only": 0, "both": 0}
        n = 0
        for i in range(N_VOCAB):
            for j in range(N_VOCAB):
                ti, tj = int(true_idx[i]), int(true_idx[j])
                a = exact_from_float_idx(ti, self.fte)
                b = exact_from_float_idx(tj, self.fte)
                tgt, br, _, _ = preferred_target_exact(a, b, self.re)
                if tgt is None:
                    continue
                n += 1
                sq = int(exact_rays_equal(exact_p_seq(a, b, self.re), tgt))
                gq = int(exact_rays_equal(exact_p_grp(a, b, self.re), tgt))
                hq = max(sq, gq)
                # policy = preferred branch
                pq = hq  # oracle policy
                s += sq
                g += gq
                h += hq
                p += pq
                if sq and gq:
                    counts["both"] += 1
                elif sq:
                    counts["seq_only"] += 1
                elif gq:
                    counts["grp_only"] += 1
                else:
                    counts["neither"] += 1
        H, A = h / n, p / n
        self.assertGreaterEqual(H + 1e-15, A)
        self.assertEqual(sum(counts.values()), n)
        self.assertAlmostEqual((1 - A), (1 - H) + (H - A), places=12)
        self.assertEqual(H, 1.0)  # true denotations reach preferred targets

    def test_01715_witness_patterns_explained(self):
        """Legacy F1E0 pattern: float success vs exact fail under mixed semantics."""
        # hard basics 1,59 — true denotations for some pairs; raw float P_seq ≠ exact
        i, j = 1, 59
        a = exact_from_float_idx(i, self.fte)
        b = exact_from_float_idx(j, self.fte)
        ps_e = exact_p_seq(a, b, self.re)
        ps_f = self.geo.p_seq(self.geo.basic[i], self.geo.basic[j], basis_vec(4))
        self.assertIsNotNone(ps_e)
        self.assertIsNotNone(ps_f)
        self.assertFalse(rays_equal(ps_f, exact_value_to_float(ps_e)))
        # Phi float matches exact
        inn = float_mul_with_phi(self.alg, self.geo.basic[j], basis_vec(4))
        ps_p = float_mul_with_phi(self.alg, self.geo.basic[i], inn)
        self.assertTrue(rays_equal(ps_p, exact_value_to_float(ps_e)))


if __name__ == "__main__":
    unittest.main()
