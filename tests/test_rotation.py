"""Unit tests for Givens rotation engine and nD projection."""

import sys
import os
import unittest
import math

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from view.rotation import (
    givens_rotation,
    rotation_from_angles,
    rotation_planes,
    project_nd_points,
    rotation_from_direction,
    axis_label,
)

import numpy as np


class TestGivensRotation(unittest.TestCase):
    def test_identity_when_zero_angle(self):
        R = givens_rotation(4, 0, 1, 0.0)
        np.testing.assert_allclose(R, np.eye(4), atol=1e-15)

    def test_orthogonal(self):
        R = givens_rotation(5, 1, 3, 0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(5), atol=1e-14)
        np.testing.assert_allclose(R.T @ R, np.eye(5), atol=1e-14)

    def test_determinant_one(self):
        R = givens_rotation(4, 2, 3, 1.23)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_only_affects_two_axes(self):
        R = givens_rotation(5, 1, 3, math.pi / 4)
        for k in [0, 2, 4]:
            e = np.zeros(5)
            e[k] = 1.0
            np.testing.assert_allclose(R @ e, e, atol=1e-15,
                                       err_msg=f"axis {k} should be unchanged")

    def test_90_deg_xy_rotates_x_to_y(self):
        R = givens_rotation(3, 0, 1, math.pi / 2)
        ex = np.array([1.0, 0.0, 0.0])
        result = R @ ex
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)


class TestRotationFromAngles(unittest.TestCase):
    def test_empty_gives_identity(self):
        R = rotation_from_angles(4, {})
        np.testing.assert_allclose(R, np.eye(4), atol=1e-15)

    def test_single_plane(self):
        theta = 0.5
        R = rotation_from_angles(3, {(0, 2): theta})
        expected = givens_rotation(3, 0, 2, theta)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_composition_is_orthogonal(self):
        angles = {(0, 1): 0.3, (1, 2): -0.7, (0, 2): 1.1}
        R = rotation_from_angles(3, angles)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)


class TestRotationPlanes(unittest.TestCase):
    def test_3d_has_3_planes(self):
        planes = rotation_planes(3)
        self.assertEqual(len(planes), 3)
        self.assertEqual(planes, [(0, 1), (0, 2), (1, 2)])

    def test_4d_has_6_planes(self):
        self.assertEqual(len(rotation_planes(4)), 6)

    def test_5d_has_10_planes(self):
        self.assertEqual(len(rotation_planes(5)), 10)

    def test_nd_formula(self):
        for n in range(2, 8):
            self.assertEqual(len(rotation_planes(n)), n * (n - 1) // 2)


class TestAxisLabel(unittest.TestCase):
    def test_standard_labels(self):
        self.assertEqual(axis_label(0), "X")
        self.assertEqual(axis_label(1), "Y")
        self.assertEqual(axis_label(2), "Z")
        self.assertEqual(axis_label(3), "W")
        self.assertEqual(axis_label(4), "V")

    def test_fallback(self):
        self.assertEqual(axis_label(99), "D99")


class TestProjectNdPoints(unittest.TestCase):
    def test_empty(self):
        R = np.eye(3)
        self.assertEqual(project_nd_points([], R, 2), [])

    def test_identity_keeps_first_axes(self):
        points = [[1, 2, 3], [4, 5, 6]]
        R = np.eye(3)
        out = project_nd_points(points, R, 2)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0][0], 1.0)
        self.assertAlmostEqual(out[0][1], 2.0)
        self.assertAlmostEqual(out[1][0], 4.0)
        self.assertAlmostEqual(out[1][1], 5.0)

    def test_rotation_changes_projection(self):
        R = givens_rotation(3, 0, 2, math.pi / 2)
        pts = project_nd_points([[1.0, 0.0, 0.0]], R, 2)
        np.testing.assert_allclose(pts[0], [0.0, 0.0], atol=1e-14)


class TestDirectionDispatch(unittest.TestCase):
    def test_none_gives_identity(self):
        R = rotation_from_direction(None, 5)
        np.testing.assert_allclose(R, np.eye(5), atol=1e-15)

    def test_dict_spec(self):
        spec = {(0, 1): math.pi / 4}
        R = rotation_from_direction(spec, 3)
        expected = rotation_from_angles(3, spec)
        np.testing.assert_allclose(R, expected, atol=1e-14)


class Test5DProjection(unittest.TestCase):
    def test_5d_identity_projection(self):
        R = np.eye(5)
        pts = project_nd_points([[1, 2, 3, 4, 5]], R, 2)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[0][1], 2.0)

    def test_5d_rotation_orthogonal(self):
        angles = {(0, 4): 0.5, (1, 3): -0.3, (2, 4): 1.0}
        R = rotation_from_angles(5, angles)
        np.testing.assert_allclose(R @ R.T, np.eye(5), atol=1e-13)

    def test_5d_rotation_changes_shadow(self):
        point = [1.0, 0.0, 0.0, 0.0, 0.0]
        R_id = np.eye(5)
        R_rot = rotation_from_angles(5, {(0, 4): math.pi / 3})
        p_id = project_nd_points([point], R_id, 2)[0]
        p_rot = project_nd_points([point], R_rot, 2)[0]
        dist = math.hypot(p_id[0] - p_rot[0], p_id[1] - p_rot[1])
        self.assertGreater(dist, 0.1)


class TestHiddenPlaneComposition(unittest.TestCase):
    """Verify that 'hidden' planes (both axes >= target_dim) affect the
    2D shadow when combined with visible rotations."""

    def test_4d_zw_affects_shadow_with_xz(self):
        point = [0.0, 0.0, 0.0, 1.0]
        R_xz = rotation_from_angles(4, {(0, 2): math.pi / 4})
        R_both = rotation_from_angles(4, {(0, 2): math.pi / 4, (2, 3): math.pi / 3})
        p_xz = project_nd_points([point], R_xz, 2)[0]
        p_both = project_nd_points([point], R_both, 2)[0]
        dist = math.hypot(p_xz[0] - p_both[0], p_xz[1] - p_both[1])
        self.assertGreater(dist, 0.1,
                           "ZW rotation should change shadow when combined with XZ")

    def test_5d_wv_affects_shadow_with_xw(self):
        point = [0.0, 0.0, 0.0, 0.0, 1.0]
        R_xw = rotation_from_angles(5, {(0, 3): math.pi / 4})
        R_both = rotation_from_angles(5, {(0, 3): math.pi / 4, (3, 4): math.pi / 3})
        p_xw = project_nd_points([point], R_xw, 2)[0]
        p_both = project_nd_points([point], R_both, 2)[0]
        dist = math.hypot(p_xw[0] - p_both[0], p_xw[1] - p_both[1])
        self.assertGreater(dist, 0.1,
                           "WV rotation should change shadow when combined with XW")


class TestVertexTiers(unittest.TestCase):
    def test_empty_returns_empty(self):
        from view.rotation import compute_vertex_tiers
        self.assertEqual(compute_vertex_tiers([], 2), [])

    def test_no_extra_dims_all_zero(self):
        from view.rotation import compute_vertex_tiers
        pts = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(compute_vertex_tiers(pts, 2), [0, 0])

    def test_tesseract_4d_to_3d(self):
        """4D tesseract to 3D: W=0 → tier 0 (base), W=1 → tier 1 (4th dim)."""
        from view.rotation import compute_vertex_tiers
        verts = [[x, y, z, w]
                 for x in (0, 1) for y in (0, 1)
                 for z in (0, 1) for w in (0, 1)]
        tiers = compute_vertex_tiers(verts, 3)
        for v, t in zip(verts, tiers):
            expected = 1 if v[3] == 1 else 0
            self.assertEqual(t, expected,
                             f"vertex {v}: expected tier {expected}, got {t}")

    def test_5d_to_3d_highest_wins(self):
        """5D to 3D: highest extra dim takes priority.
        (x,y,z,1,0) → tier 1 (4th dim), (x,y,z,0,1) → tier 2 (5th dim),
        (x,y,z,1,1) → tier 2 (5th dim wins)."""
        from view.rotation import compute_vertex_tiers
        verts = [
            [1, 1, 1, 0, 0],  # base
            [1, 1, 1, 1, 0],  # 4th dim
            [1, 1, 1, 0, 1],  # 5th dim
            [1, 1, 1, 1, 1],  # both → 5th dim wins
        ]
        tiers = compute_vertex_tiers(verts, 3)
        self.assertEqual(tiers, [0, 1, 2, 2])

    def test_4d_to_2d_highest_wins(self):
        """4D to 2D: extra dims are Z and W.
        (x,y,0,0) → tier 0, (x,y,1,0) → tier 1,
        (x,y,0,1) → tier 2, (x,y,1,1) → tier 2."""
        from view.rotation import compute_vertex_tiers
        verts = [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ]
        tiers = compute_vertex_tiers(verts, 2)
        self.assertEqual(tiers, [0, 1, 2, 2])

    def test_continuous_below_midpoint_is_base(self):
        """Vertices with extra-dim values below the midpoint are base tier."""
        from view.rotation import compute_vertex_tiers
        pts = [[0.0, 0.0, 0.1], [0.0, 0.0, 0.9]]
        tiers = compute_vertex_tiers(pts, 2)
        self.assertEqual(tiers[0], 0)
        self.assertEqual(tiers[1], 1)


if __name__ == "__main__":
    unittest.main()
