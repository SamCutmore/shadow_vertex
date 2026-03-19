#!/usr/bin/env python3
import argparse
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)


def main():
    parser = argparse.ArgumentParser(description="Inspect shadow vertex path")
    parser.add_argument("--dim", type=int, default=4, help="Dimension (default 4)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--kleeminty", action="store_true",
                       help="Use Klee-Minty instead of tesseract")
    group.add_argument("--degenerate", action="store_true",
                       help="Use degenerate polytope (extra constraints at origin)")
    args = parser.parse_args()

    from start import _kleeminty, _tesseract, _degenerate

    import linprog_core

    if args.kleeminty:
        objective, constraints = _kleeminty(args.dim)
        name = f"Klee-Minty dim {args.dim}"
    elif args.degenerate:
        objective, constraints = _degenerate(args.dim)
        name = f"Degenerate dim {args.dim}"
    else:
        objective, constraints = _tesseract(args.dim)
        name = f"Tesseract dim {args.dim}"

    n = args.dim
    prob = linprog_core.PyProblem(objective, goal="max")
    for coeffs, rel, rhs in constraints:
        prob.add_constraint(coeffs, rel, rhs)

    d_coeffs = [0] * n
    d_coeffs[-1] = -1
    solver = linprog_core.PyShadowVertexSimplexSolver()
    solver.set_auxiliary_objective(d_coeffs, [0] * len(constraints), 0)
    solution, history, shadow_points, _stats = solver.solve_with_shadow_history(prob)

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
