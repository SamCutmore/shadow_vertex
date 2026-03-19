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

parser = argparse.ArgumentParser(description="5D generic LP demo")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for d generation and perturbation")
parser.add_argument("--sigma", type=float, default=0.0,
                    help="Gaussian perturbation magnitude for smoothed analysis (0=off)")
parser.add_argument("--d-mode", choices=["axis", "random"], default="axis",
                    help="Auxiliary objective: 'axis' (-e_n) or 'random' (Gaussian)")
args = parser.parse_args()
sys.argv = [sys.argv[0]]

import linprog_core
from view import run_visualization_2d
from utils.smoothed import random_auxiliary_objective, perturb_constraints


def _rat_to_float(v):
    """Convert a value that may be int, float, or (num, den) tuple to float."""
    if isinstance(v, tuple):
        return v[0] / v[1]
    return float(v)


def _constraints_to_float(constraints):
    """Convert constraints with possible rational tuples to pure floats for the view layer."""
    out = []
    for coeffs, rel, rhs in constraints:
        out.append(([_rat_to_float(c) for c in coeffs], rel, _rat_to_float(rhs)))
    return out


n = 5
objective = [3, 2, 5, 1, 4]

constraints = [
    # upper bounds per variable (loose, asymmetric)
    ([1, 0, 0, 0, 0], "<=", 6),
    ([0, 1, 0, 0, 0], "<=", 8),
    ([0, 0, 1, 0, 0], "<=", 5),
    ([0, 0, 0, 1, 0], "<=", 10),
    ([0, 0, 0, 0, 1], "<=", 7),
    # non-negativity
    ([1, 0, 0, 0, 0], ">=", 0),
    ([0, 1, 0, 0, 0], ">=", 0),
    ([0, 0, 1, 0, 0], ">=", 0),
    ([0, 0, 0, 1, 0], ">=", 0),
    ([0, 0, 0, 0, 1], ">=", 0),
    # cross-dimension constraints that cut the feasible region
    ([1, 1, 0, 0, 0], "<=", 10),
    ([0, 0, 1, 1, 1], "<=", 14),
    ([2, 0, 1, 0, 0], "<=", 12),
    ([0, 1, 0, 2, 0], "<=", 15),
    ([1, 0, 0, 0, 2], "<=", 16),
    ([1, 1, 1, 0, 0], "<=", 13),
    ([0, 0, 1, 1, 0], "<=", 9),
    ([1, 0, 0, 1, 1], "<=", 15),
]

if args.sigma > 0:
    constraints = perturb_constraints(constraints, args.sigma, args.seed)

if args.d_mode == "random":
    d_coeffs = random_auxiliary_objective(n, objective, args.seed)
else:
    d_coeffs = [0, 0, 0, 0, -1]

prob = linprog_core.PyProblem(objective, goal="max")
for coeffs, rel, rhs in constraints:
    prob.add_constraint(coeffs, rel, rhs)

n_slack = len(constraints)
solver = linprog_core.PyShadowVertexSimplexSolver()
solver.set_auxiliary_objective(d_coeffs, [0] * n_slack, 0)
solution, history, shadow_points = solver.solve_with_shadow_history(prob)

d_label = "random" if args.d_mode == "random" else "axis(-e_n)"
print(f"Generic 5D LP  (d={d_label}, seed={args.seed}, sigma={args.sigma})")
print(f"Objective: max {objective}")
print(f"Status: {solution.status}, objective: {solution.objective}")
print(f"Path: {len(history)} vertices. Use arrow keys to step through.")
print(f"Shadow points: {len(shadow_points)}")

d_coeffs_float = [_rat_to_float(v) for v in d_coeffs]

run_visualization_2d(
    constraints=_constraints_to_float(constraints),
    history=history,
    solution=solution,
    shadow_points=shadow_points,
    objective=[float(v) for v in objective],
    d_objective=d_coeffs_float,
    window_size=(900, 700),
)
