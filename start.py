#!/usr/bin/env python3
import argparse
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

_lib = os.path.join(_root, "lib")
if os.path.isdir(_lib) and "SHADOW_VERTEX_LD_SET" not in os.environ:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _lib + (os.pathsep + env.get("LD_LIBRARY_PATH", ""))
    env["SHADOW_VERTEX_LD_SET"] = "1"
    os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


# ---------------------------------------------------------------------------
# LP problem builders
# ---------------------------------------------------------------------------

def _kleeminty(n):
    """Klee-Minty cube in n dimensions."""
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
    return objective, constraints


def _tesseract(n):
    """Unit hypercube [0,1]^n with objective max sum(x_i)."""
    objective = [1] * n
    constraints = []
    for i in range(n):
        e = [0] * n
        e[i] = 1
        constraints.append((e, ">=", 0))
        constraints.append((e, "<=", 1))
    return objective, constraints


def _degenerate(n):
    """Unit hypercube with extra constraints making the origin degenerate.

    The origin has n non-negativity constraints plus n-1 additional
    constraints of the form x_i + x_{i+1} <= 1, all tight at the origin.
    This gives l = 2n - 1 tight constraints at the origin (for n >= 2),
    so perturbing b splits it into up to C(2n-1, n) nearby vertices.
    """
    objective = [1] * n
    constraints = []
    for i in range(n):
        e = [0] * n
        e[i] = 1
        constraints.append((e, ">=", 0))
        constraints.append((e, "<=", 1))
    for i in range(n - 1):
        row = [0] * n
        row[i] = 1
        row[i + 1] = 1
        constraints.append((row, "<=", 1))
    return objective, constraints


PROBLEMS = {
    "kleeminty": _kleeminty,
    "tesseract": _tesseract,
    "degenerate": _degenerate,
}


def _rat_to_float(v):
    if isinstance(v, tuple):
        return v[0] / v[1]
    return float(v)


def _constraints_to_float(constraints):
    out = []
    for coeffs, rel, rhs in constraints:
        out.append(([_rat_to_float(c) for c in coeffs], rel, _rat_to_float(rhs)))
    return out


# ---------------------------------------------------------------------------
# Solver runners
# ---------------------------------------------------------------------------

def _run_simplex(prob, args):
    import linprog_core

    solver = linprog_core.PySimplexSolver()
    solution, history, _stats = solver.solve_with_history(prob)
    return solution, history, None, None


def _run_shadow_vertex(prob, args, objective, constraints, n_vars):
    import linprog_core
    from utils.smoothed import random_auxiliary_objective

    if args.d_mode == "random":
        d_coeffs = random_auxiliary_objective(n_vars, objective, args.seed)
    else:
        d_coeffs = [0] * n_vars
        d_coeffs[-1] = -1

    solver = linprog_core.PyShadowVertexSimplexSolver()
    n_slack = len(constraints)
    solver.set_auxiliary_objective(d_coeffs, [0] * n_slack, 0)
    solution, history, shadow_points, _stats = solver.solve_with_shadow_history(prob)

    d_coeffs_float = [_rat_to_float(v) for v in d_coeffs]
    return solution, history, shadow_points, d_coeffs_float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LP solver visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kleeminty", action="store_const", const="kleeminty",
                       dest="problem", help="Klee-Minty distorted cube LP")
    group.add_argument("--tesseract", action="store_const", const="tesseract",
                       dest="problem", help="Unit hypercube LP")
    group.add_argument("--degenerate", action="store_const", const="degenerate",
                       dest="problem",
                       help="Unit hypercube with extra constraints (degenerate origin)")
    parser.add_argument("--dim", "--dimensions", type=int, default=3,
                        help="Number of decision variables (default: 3)")
    parser.add_argument("--solver", choices=["simplex", "shadow_vertex"],
                        default="shadow_vertex",
                        help="Solver algorithm (default: shadow_vertex)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for d generation and perturbation")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Gaussian perturbation magnitude for smoothed analysis (0=off)")
    parser.add_argument("--d-mode", choices=["axis", "random"], default="axis",
                        help="Auxiliary objective: 'axis' (-e_n) or 'random' (Gaussian)")
    args = parser.parse_args()

    if args.dim < 2:
        parser.error("--dim must be at least 2")

    sys.argv = [sys.argv[0]]

    import linprog_core
    from view import run_visualization_2d
    from utils.smoothed import perturb_rhs

    objective, constraints = PROBLEMS[args.problem](args.dim)
    n_vars = args.dim

    if args.sigma > 0:
        constraints = perturb_rhs(constraints, args.sigma, args.seed)

    prob = linprog_core.PyProblem(objective, goal="max")
    for coeffs, rel, rhs in constraints:
        prob.add_constraint(coeffs, rel, rhs)

    if args.solver == "simplex":
        solution, history, shadow_points, d_coeffs_float = _run_simplex(prob, args)
    else:
        solution, history, shadow_points, d_coeffs_float = _run_shadow_vertex(
            prob, args, objective, constraints, n_vars)

    d_label = {
        "simplex": "n/a",
        "shadow_vertex": "random" if args.d_mode == "random" else "axis(-e_n)",
    }[args.solver]
    sigma_str = f", sigma={args.sigma}" if args.sigma > 0 else ""
    print(f"Problem: {args.problem} {args.dim}D  "
          f"(solver={args.solver}, d={d_label}, seed={args.seed}{sigma_str})")
    print(f"Status: {solution.status}, objective: {solution.objective}")
    print(f"Path: {len(history)} vertices. Use arrow keys to step through.")
    if shadow_points:
        print(f"Shadow points: {len(shadow_points)}")

    run_visualization_2d(
        constraints=_constraints_to_float(constraints),
        history=history,
        solution=solution,
        shadow_points=shadow_points if shadow_points else None,
        objective=[float(v) for v in objective],
        d_objective=d_coeffs_float,
        window_size=(900, 700),
    )


if __name__ == "__main__":
    main()
