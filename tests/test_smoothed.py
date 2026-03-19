"""Tests for smoothed analysis utilities."""

import unittest
from fractions import Fraction

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.smoothed import random_auxiliary_objective, perturb_rhs, perturb_constraints


def _pair_to_frac(pair):
    """Convert (num, den) tuple to Fraction."""
    return Fraction(pair[0], pair[1])


class TestRandomAuxiliaryObjective(unittest.TestCase):
    def test_linearly_independent_from_c(self):
        c = [3, 2, 5, 1, 4]
        d = random_auxiliary_objective(5, c, seed=42)
        ratios = set()
        for di, ci in zip(d, c):
            if ci != 0 and di != 0:
                ratios.add(Fraction(di, ci))
        self.assertGreater(len(ratios), 1, "d should not be parallel to c")

    def test_nonzero(self):
        d = random_auxiliary_objective(4, [1, 1, 1, 1], seed=7)
        self.assertTrue(any(di != 0 for di in d))

    def test_deterministic_same_seed(self):
        c = [1, 2, 3]
        d1 = random_auxiliary_objective(3, c, seed=99)
        d2 = random_auxiliary_objective(3, c, seed=99)
        self.assertEqual(d1, d2)

    def test_different_seeds_differ(self):
        c = [1, 2, 3]
        d1 = random_auxiliary_objective(3, c, seed=0)
        d2 = random_auxiliary_objective(3, c, seed=1)
        self.assertNotEqual(d1, d2)

    def test_correct_length(self):
        d = random_auxiliary_objective(7, [1] * 7, seed=0)
        self.assertEqual(len(d), 7)

    def test_returns_list_of_ints(self):
        d = random_auxiliary_objective(3, [1, 2, 3], seed=0)
        self.assertIsInstance(d, list)
        for item in d:
            self.assertIsInstance(item, int)

    def test_integer_components(self):
        c = [3, 2, 5, 1, 4]
        d = random_auxiliary_objective(5, c, seed=42)
        for v in d:
            self.assertIsInstance(v, int)
            self.assertLessEqual(abs(v), 11)


class TestPerturbConstraints(unittest.TestCase):
    CONSTRAINTS = [
        ([1, 0, 0], ">=", 0),       # bound: x >= 0
        ([0, 1, 0], ">=", 0),       # bound: y >= 0
        ([0, 0, 1], "<=", 5),       # bound: z <= 5
        ([1, 1, 0], "<=", 10),      # structural
        ([2, 0, 1], "<=", 12),      # structural
    ]

    def test_sigma_zero_returns_identical(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=0.0, seed=0)
        for (oc, orel, orhs), (nc, nrel, nrhs) in zip(self.CONSTRAINTS, out):
            self.assertEqual(orel, nrel)
            self.assertEqual(orhs, nrhs)
            self.assertEqual(oc, nc)

    def test_skip_bounds_preserves_nonnegativity(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=1.0, seed=42,
                                  skip_bounds=True)
        for i in range(2):  # x>=0, y>=0
            oc, orel, orhs = self.CONSTRAINTS[i]
            nc, nrel, nrhs = out[i]
            self.assertEqual(oc, nc)
            self.assertEqual(orel, nrel)
            self.assertEqual(orhs, nrhs)

    def test_skip_bounds_perturbs_upper_bounds(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=1.0, seed=42,
                                  skip_bounds=True)
        # z <= 5 (index 2) is an upper bound, should be perturbed
        oc, _, orhs = self.CONSTRAINTS[2]
        nc, _, nrhs = out[2]
        self.assertTrue(oc != nc or orhs != nrhs,
                        "Upper bound z<=5 should be perturbed")

    def test_skip_bounds_perturbs_structural(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=1.0, seed=42,
                                  skip_bounds=True)
        structural_changed = False
        for i in range(3, 5):
            oc, _, orhs = self.CONSTRAINTS[i]
            nc, _, nrhs = out[i]
            if oc != nc or orhs != nrhs:
                structural_changed = True
        self.assertTrue(structural_changed)

    def test_no_skip_bounds_perturbs_nonnegativity(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=1.0, seed=42,
                                  skip_bounds=False)
        any_nn_changed = False
        for i in range(2):  # x>=0, y>=0
            oc, _, orhs = self.CONSTRAINTS[i]
            nc, _, nrhs = out[i]
            if oc != nc or orhs != nrhs:
                any_nn_changed = True
        self.assertTrue(any_nn_changed)

    def test_deterministic_same_seed(self):
        out1 = perturb_constraints(self.CONSTRAINTS, sigma=0.5, seed=77)
        out2 = perturb_constraints(self.CONSTRAINTS, sigma=0.5, seed=77)
        for (c1, r1, rhs1), (c2, r2, rhs2) in zip(out1, out2):
            self.assertEqual(c1, c2)
            self.assertEqual(r1, r2)
            self.assertEqual(rhs1, rhs2)

    def test_original_not_modified(self):
        import copy
        original = copy.deepcopy(self.CONSTRAINTS)
        perturb_constraints(self.CONSTRAINTS, sigma=1.0, seed=0)
        self.assertEqual(self.CONSTRAINTS, original)

    def test_length_preserved(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=0.1, seed=0)
        self.assertEqual(len(out), len(self.CONSTRAINTS))

    def test_perturbed_coeffs_are_rational_pairs(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=0.1, seed=0,
                                  skip_bounds=True)
        for coeffs, rel, rhs in out[3:]:
            for item in coeffs:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
            self.assertIsInstance(rhs, tuple)
            self.assertEqual(len(rhs), 2)

    def test_denominator_is_bounded(self):
        out = perturb_constraints(self.CONSTRAINTS, sigma=0.1, seed=0,
                                  denom=32, skip_bounds=True)
        for coeffs, rel, rhs in out[3:]:
            for num, den in coeffs:
                self.assertEqual(den, 32)
            self.assertEqual(rhs[1], 32)


class TestPerturbRhs(unittest.TestCase):
    CONSTRAINTS = [
        ([1, 0, 0], ">=", 0),       # bound: x >= 0
        ([0, 1, 0], ">=", 0),       # bound: y >= 0
        ([0, 0, 1], "<=", 5),       # upper bound
        ([1, 1, 0], "<=", 10),      # structural
        ([2, 0, 1], "<=", 12),      # structural
    ]

    def test_sigma_zero_returns_identical(self):
        out = perturb_rhs(self.CONSTRAINTS, sigma=0.0, seed=0)
        for (oc, orel, orhs), (nc, nrel, nrhs) in zip(self.CONSTRAINTS, out):
            self.assertEqual(orel, nrel)
            self.assertEqual(orhs, nrhs)
            self.assertEqual(oc, nc)

    def test_skip_bounds_preserves_nonnegativity(self):
        out = perturb_rhs(self.CONSTRAINTS, sigma=1.0, seed=42,
                          skip_bounds=True)
        for i in range(2):
            oc, orel, orhs = self.CONSTRAINTS[i]
            nc, nrel, nrhs = out[i]
            self.assertEqual(oc, nc)
            self.assertEqual(orel, nrel)
            self.assertEqual(orhs, nrhs)

    def test_coefficients_unchanged(self):
        out = perturb_rhs(self.CONSTRAINTS, sigma=1.0, seed=42,
                          skip_bounds=False)
        for (oc, _, _), (nc, _, _) in zip(self.CONSTRAINTS, out):
            self.assertEqual(oc, nc)

    def test_rhs_perturbed_for_structural(self):
        out = perturb_rhs(self.CONSTRAINTS, sigma=1.0, seed=42,
                          skip_bounds=True)
        any_changed = False
        for i in range(2, 5):
            _, _, orhs = self.CONSTRAINTS[i]
            _, _, nrhs = out[i]
            if orhs != nrhs:
                any_changed = True
        self.assertTrue(any_changed)

    def test_perturbed_rhs_is_rational_pair(self):
        out = perturb_rhs(self.CONSTRAINTS, sigma=0.1, seed=0,
                          skip_bounds=True)
        for _, _, rhs in out[2:]:
            self.assertIsInstance(rhs, tuple)
            self.assertEqual(len(rhs), 2)

    def test_deterministic(self):
        out1 = perturb_rhs(self.CONSTRAINTS, sigma=0.5, seed=77)
        out2 = perturb_rhs(self.CONSTRAINTS, sigma=0.5, seed=77)
        for (c1, r1, rhs1), (c2, r2, rhs2) in zip(out1, out2):
            self.assertEqual(c1, c2)
            self.assertEqual(r1, r2)
            self.assertEqual(rhs1, rhs2)

    def test_original_not_modified(self):
        import copy
        original = copy.deepcopy(self.CONSTRAINTS)
        perturb_rhs(self.CONSTRAINTS, sigma=1.0, seed=0)
        self.assertEqual(self.CONSTRAINTS, original)


if __name__ == "__main__":
    unittest.main()
