"""Tests for Givens-controlled nD -> 2D projection."""

import math
import os
import sys
import unittest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)


class TestGivensProjection2D(unittest.TestCase):
    def _project_point(self, angles_deg):
        from view.rotation import rotation_from_angles, project_nd_points
        angles_rad = {k: math.radians(v) for k, v in angles_deg.items()}
        R = rotation_from_angles(3, angles_rad)
        pts = project_nd_points([[1.0, 2.0, 3.0]], R, 2)
        return pts[0]

    def test_continuity_xz_plane(self):
        prev = self._project_point({(0, 2): -5.0})
        for angle in range(-5, 6):
            cur = self._project_point({(0, 2): float(angle)})
            dist = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            self.assertLess(dist, 5.0, msg=f"large jump at XZ={angle}")
            prev = cur

    def test_planes_not_identical_near_origin(self):
        base = self._project_point({})
        diffs = {}
        for plane in [(0, 1), (0, 2), (1, 2)]:
            p = self._project_point({plane: 5.0})
            diffs[plane] = [p[i] - base[i] for i in range(2)]

        def _norm(v):
            return math.hypot(v[0], v[1])

        for plane, d in diffs.items():
            self.assertGreater(_norm(d), 1e-6,
                               msg=f"plane {plane} has no effect")

        def _cos(a, b):
            return (a[0] * b[0] + a[1] * b[1]) / (_norm(a) * _norm(b))

        cos_01_12 = _cos(diffs[(0, 1)], diffs[(1, 2)])
        self.assertLess(abs(cos_01_12), 0.99,
                        msg=f"XY and YZ motions too aligned: cos={cos_01_12}")

    def test_klee_minty_shadow_changes_with_xz_rotation(self):
        from view import Polyhedron, Path
        from view.rotation import rotation_from_angles
        import linprog_core

        n = 3
        objective = [2 ** (n - 1 - i) for i in range(n)]
        constraints = []
        for i in range(1, n + 1):
            coeffs = [2 ** (i - j + 1) for j in range(1, i)] + [1]
            coeffs.extend([0.0] * (n - len(coeffs)))
            constraints.append((coeffs, "<=", 5 ** (i - 1)))
        constraints.append(([1.0, 0.0, 0.0], ">=", 0.0))
        constraints.append(([0.0, 1.0, 0.0], ">=", 0.0))
        constraints.append(([0.0, 0.0, 1.0], ">=", 0.0))
        prob = linprog_core.PyProblem(objective, goal="max")
        for coeffs, rel, rhs in constraints:
            prob.add_constraint(coeffs, rel, rhs)
        solver = linprog_core.PySimplexSolver()
        _, history = solver.solve_with_history(prob)

        path = Path.from_history(history)
        poly = Polyhedron.from_constraints(constraints, path.points if path.points else None)
        self.assertFalse(poly.is_empty())

        shadow0 = poly.to_2d(direction=None)
        shadow90 = poly.to_2d(direction={(0, 2): math.radians(90.0)})

        moved = any(
            math.hypot(v1[0] - v0[0], v1[1] - v0[1]) > 1e-3
            for v0, v1 in zip(shadow0.vertices, shadow90.vertices)
        )
        self.assertTrue(moved, "XZ rotation should change at least one 2D shadow vertex")


if __name__ == "__main__":
    unittest.main()
