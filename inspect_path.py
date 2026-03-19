#!/usr/bin/env python3
import argparse
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)


def main():
    parser = argparse.ArgumentParser(description="Inspect shadow vertex path")
    parser.add_argument("--dim", type=int, default=4, help="Dimension (default 4)")
    parser.add_argument("--kleeminty", action="store_true", help="Use Klee-Minty instead of tesseract")
    args = parser.parse_args()

    import linprog_core

    if args.kleeminty:
        n = args.dim
        objective = [2 ** (n - 1 - i) for i in range(n)]
        constraints = []
        for i in range(1, n + 1):
            coeffs = [2 ** (i - j + 1) for j in range(1, i)] + [1]
            coeffs.extend([0] * (n - len(coeffs)))
            constraints.append((coeffs, "<=", 5 ** (i - 1)))
        for i in range(n):
            e = [0] * n
            e[i] = 1
            constraints.append((e, ">=", 0))
        name = f"Klee-Minty dim {n}"
    else:
        n = args.dim
        objective = [1] * n
        constraints = []
        for i in range(n):
            e = [0] * n
            e[i] = 1
            constraints.append((e, ">=", 0))
            constraints.append((e, "<=", 1))
        name = f"Tesseract dim {n}"

    prob = linprog_core.PyProblem(objective, goal="max")
    for coeffs, rel, rhs in constraints:
        prob.add_constraint(coeffs, rel, rhs)

    d_coeffs = [0] * n
    d_coeffs[-1] = -1
    solver = linprog_core.PyShadowVertexSimplexSolver()
    solver.set_auxiliary_objective(d_coeffs, [0] * len(constraints), 0)
    solution, history, shadow_points = solver.solve_with_shadow_history(prob)

    print(f"{name}")
    print(f"  Status: {solution.status}  Objective: {solution.objective}")
    print(f"  Path length: {len(history)}")
    print()
    for i, step in enumerate(history):
        print(f"  {i}: {step.primal}  (iter {step.iteration}, status={step.status})")
    print()
    print(f"  Shadow (d,c): {shadow_points}")


if __name__ == "__main__":
    main()
