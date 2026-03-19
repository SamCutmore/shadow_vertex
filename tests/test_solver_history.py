"""Tests for solver step history: no duplicates, includes initial BFS."""

import unittest
import linprog_core


def _tesseract_problem(n):
    """Unit hypercube [0,1]^n with objective max sum(x_i)."""
    objective = [1.0] * n
    prob = linprog_core.PyProblem(objective, goal="max")
    constraints = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        prob.add_constraint(e, ">=", 0.0)
        constraints.append((e, ">=", 0.0))
        prob.add_constraint(e, "<=", 1.0)
        constraints.append((e, "<=", 1.0))
    return prob, constraints


def _kleeminty_problem(n):
    """Klee-Minty cube in n dimensions."""
    objective = [2 ** (n - 1 - i) for i in range(n)]
    prob = linprog_core.PyProblem(objective, goal="max")
    constraints = []
    for i in range(1, n + 1):
        coeffs = [2 ** (i - j + 1) for j in range(1, i)] + [1]
        coeffs.extend([0.0] * (n - len(coeffs)))
        prob.add_constraint(coeffs, "<=", 5 ** (i - 1))
        constraints.append((coeffs, "<=", 5 ** (i - 1)))
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        prob.add_constraint(e, ">=", 0.0)
        constraints.append((e, ">=", 0.0))
    return prob, constraints


def _primals(history):
    return [tuple(round(x, 10) for x in s.primal) for s in history]


class TestSimplexHistory(unittest.TestCase):

    def test_tesseract_3d_has_4_unique_vertices(self):
        prob, _ = _tesseract_problem(3)
        solver = linprog_core.PySimplexSolver()
        sol, history = solver.solve_with_history(prob)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 3.0)
        self.assertEqual(len(primals), 4)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0))
        self.assertEqual(primals[-1], (1.0, 1.0, 1.0))
        self.assertEqual(len(primals), len(set(primals)),
                         "No consecutive duplicate primals")

    def test_tesseract_4d_starts_at_origin(self):
        prob, _ = _tesseract_problem(4)
        solver = linprog_core.PySimplexSolver()
        sol, history = solver.solve_with_history(prob)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 4.0)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(primals[-1], (1.0, 1.0, 1.0, 1.0))
        for i in range(1, len(primals)):
            self.assertNotEqual(primals[i], primals[i - 1],
                                f"Steps {i-1} and {i} should differ")

    def test_kleeminty_3d_starts_at_origin_ends_at_optimal(self):
        prob, _ = _kleeminty_problem(3)
        solver = linprog_core.PySimplexSolver()
        sol, history = solver.solve_with_history(prob)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 25.0)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0))
        for i in range(1, len(primals)):
            self.assertNotEqual(primals[i], primals[i - 1])

    def test_no_duplicate_final_step(self):
        """The final vertex must not appear twice."""
        prob, _ = _tesseract_problem(3)
        solver = linprog_core.PySimplexSolver()
        _, history = solver.solve_with_history(prob)
        primals = _primals(history)
        self.assertNotEqual(len(primals), 0)
        if len(primals) >= 2:
            self.assertNotEqual(primals[-1], primals[-2])


class TestShadowVertexHistory(unittest.TestCase):

    def _solve(self, n, builder):
        prob, constraints = builder(n)
        solver = linprog_core.PyShadowVertexSimplexSolver()
        solver.set_auxiliary_objective(
            [-1.0] * n, [0.0] * len(constraints), 0.0)
        sol, history, shadow_pts = solver.solve_with_shadow_history(prob)
        return sol, history, shadow_pts

    def test_tesseract_3d_has_4_unique_vertices(self):
        sol, history, shadow_pts = self._solve(3, _tesseract_problem)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 3.0)
        self.assertEqual(len(primals), 4)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0))
        self.assertEqual(primals[-1], (1.0, 1.0, 1.0))

    def test_shadow_points_match_history_length(self):
        """Each unique vertex has a corresponding shadow point."""
        sol, history, shadow_pts = self._solve(3, _tesseract_problem)
        self.assertEqual(len(shadow_pts), len(history))

    def test_shadow_points_no_consecutive_duplicates(self):
        sol, history, shadow_pts = self._solve(3, _tesseract_problem)
        for i in range(1, len(shadow_pts)):
            self.assertNotEqual(shadow_pts[i], shadow_pts[i - 1],
                                f"Shadow points {i-1} and {i} should differ")

    def test_kleeminty_3d_no_duplicates(self):
        sol, history, shadow_pts = self._solve(3, _kleeminty_problem)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 25.0)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0))
        for i in range(1, len(primals)):
            self.assertNotEqual(primals[i], primals[i - 1])
        self.assertEqual(len(shadow_pts), len(history))

    def test_tesseract_4d_starts_at_origin(self):
        sol, history, shadow_pts = self._solve(4, _tesseract_problem)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 4.0)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(primals[-1], (1.0, 1.0, 1.0, 1.0))
        for i in range(1, len(primals)):
            self.assertNotEqual(primals[i], primals[i - 1])

    def test_tesseract_5d(self):
        sol, history, shadow_pts = self._solve(5, _tesseract_problem)
        primals = _primals(history)
        self.assertEqual(sol.status, "optimal")
        self.assertEqual(sol.objective, 5.0)
        self.assertEqual(primals[0], (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(primals[-1], (1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(len(shadow_pts), len(history))


if __name__ == "__main__":
    unittest.main()
