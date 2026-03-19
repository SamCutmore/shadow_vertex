"""Test that to_2d() preserves structure and coordinate correctness."""

import sys
import os
import unittest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)


class TestProjection(unittest.TestCase):
    @staticmethod
    def _demo_3d_problem():
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
        _, history, _stats = solver.solve_with_history(prob)
        return constraints, history

    def test_path_length_unchanged_by_projection(self):
        from view import Path, Polyhedron
        constraints, history = self._demo_3d_problem()
        path = Path.from_history(history)
        n_orig = len(path.points)
        self.assertGreaterEqual(n_orig, 2)
        self.assertEqual(len(path.to_2d().points), n_orig)

        for i in range(n_orig):
            p = path.points[i]
            q2 = path.to_2d().points[i]
            self.assertEqual(len(q2), 2)
            self.assertAlmostEqual(q2[0], p[0])
            self.assertAlmostEqual(q2[1], p[1])

    def test_polyhedron_structure_unchanged_by_projection(self):
        from view import Path, Polyhedron
        constraints, history = self._demo_3d_problem()
        path = Path.from_history(history)
        poly = Polyhedron.from_constraints(constraints, path.points if path.points else None)
        self.assertFalse(poly.is_empty())

        n_v, n_f, n_e = len(poly.vertices), len(poly.faces), len(poly.edges)
        poly_2d = poly.to_2d()
        self.assertEqual(len(poly_2d.vertices), n_v)
        self.assertEqual(len(poly_2d.faces), n_f)
        self.assertEqual(len(poly_2d.edges), n_e)

        for i in range(n_v):
            v = poly.vertices[i]
            u2 = poly_2d.vertices[i]
            self.assertEqual(len(u2), 2)
            self.assertAlmostEqual(u2[0], v[0])
            self.assertAlmostEqual(u2[1], v[1])

    def test_segment_preserves_path_length(self):
        from view import Path
        path = Path([[0, 0], [1, 0], [1, 1], [0.5, 0.5], [0, 1]])
        for i in range(5):
            seg = path.segment(i)
            self.assertEqual(len(seg.points), min(i + 1, 5))

    def test_3d_to_2d_identity_direction_matches_no_direction(self):
        from view import Path, Polyhedron
        constraints, history = self._demo_3d_problem()
        path = Path.from_history(history)
        poly = Polyhedron.from_constraints(constraints, path.points if path.points else None)

        path_std = path.to_2d()
        path_dir = path.to_2d(direction={})
        for i in range(len(path.points)):
            for j in range(2):
                self.assertAlmostEqual(
                    path_std.points[i][j], path_dir.points[i][j], places=6,
                    msg=f"path point {i} coord {j}",
                )

        poly_std = poly.to_2d()
        poly_dir = poly.to_2d(direction={})
        self.assertEqual(len(poly_std.vertices), len(poly_dir.vertices))
        for i in range(len(poly.vertices)):
            for j in range(2):
                self.assertAlmostEqual(
                    poly_std.vertices[i][j], poly_dir.vertices[i][j], places=6,
                    msg=f"vertex {i} coord {j}",
                )

    def test_structure_unchanged_with_identity_direction(self):
        from view import Path, Polyhedron
        constraints, history = self._demo_3d_problem()
        path = Path.from_history(history)
        poly = Polyhedron.from_constraints(constraints, path.points if path.points else None)
        n_path = len(path.points)
        n_v, n_f, n_e = len(poly.vertices), len(poly.faces), len(poly.edges)
        path_2d = path.to_2d(direction={})
        poly_2d = poly.to_2d(direction={})
        self.assertEqual(len(path_2d.points), n_path)
        self.assertEqual(len(poly_2d.vertices), n_v)
        self.assertEqual(len(poly_2d.faces), n_f)
        self.assertEqual(len(poly_2d.edges), n_e)


if __name__ == "__main__":
    unittest.main()
