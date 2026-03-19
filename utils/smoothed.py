from fractions import Fraction
from typing import List, Tuple, Union

import numpy as np

Rational = Tuple[int, int]


def _frac_to_pair(f: Fraction) -> Rational:
    return (int(f.numerator), int(f.denominator))


def random_auxiliary_objective(
    n: int, c: List[int], seed: int = 0
) -> List[int]:
    """Generate a random auxiliary objective linearly independent from *c*.

    Components are drawn as random integers in [-10, 10].  The vector is
    checked for linear independence from *c*; if it happens to be a scalar
    multiple, a coordinate is bumped until independence is achieved.

    Returns plain integers (denominator 1) to minimise Rational64 denominator
    growth during the shadow vertex solver's Phase I and parametric pivots.
    The shadow vertex algorithm requires only linear independence from *c*,
    not orthogonality.

    Deterministic for a given *seed*.
    """
    rng = np.random.RandomState(seed)
    d = [int(x) for x in rng.randint(-10, 11, size=n)]

    def _is_parallel(a, b):
        """True when a and b are scalar multiples (or one is zero)."""
        ratio = None
        for ai, bi in zip(a, b):
            if ai == 0 and bi == 0:
                continue
            if ai == 0 or bi == 0:
                return False
            r = Fraction(ai, bi)
            if ratio is None:
                ratio = r
            elif r != ratio:
                return False
        return True

    if all(di == 0 for di in d) or _is_parallel(d, c):
        for i in range(n):
            d[i] += 1
            if not _is_parallel(d, c) and any(di != 0 for di in d):
                break
            d[i] -= 1

    return d


def _is_nonnegativity(coeffs, rel: str, rhs) -> bool:
    """True for non-negativity constraints like x_i >= 0."""
    if rel.strip().lower() not in (">=", "geq"):
        return False
    if isinstance(rhs, tuple):
        return rhs[0] == 0
    return rhs == 0


def perturb_constraints(
    constraints: List[Tuple[list, str, int]],
    sigma: float,
    seed: int = 0,
    skip_bounds: bool = True,
    denom: int = 32,
) -> List[Tuple[list, str, Union[int, Rational]]]:
    """Apply discretised Gaussian perturbation using exact rationals.

    Each coefficient *a* (integer) becomes ``(a * D + p, D)`` where *D* is
    *denom* and *p* is drawn from a discretised Gaussian clamped to
    ``[-k, k]`` with ``k = round(sigma * D)``.  This keeps all denominators
    bounded by *D*, preventing Rational64 overflow during simplex pivots.

    The default ``denom=32`` keeps Rational64 arithmetic safe for the
    shadow vertex solver's Phase I and parametric pivots on 5D+ problems.
    Larger values (e.g. 100+) risk i64 overflow in complex LPs.

    Parameters
    ----------
    constraints : list of (coeffs, rel, rhs)
        Original constraint list with **integer** coefficients.
    sigma : float
        Perturbation magnitude.  ``k = round(sigma * denom)`` controls the
        range of the integer noise.  0 means no perturbation.
    seed : int
        Random seed for reproducibility.
    skip_bounds : bool
        When True, non-negativity constraints (x_i >= 0) are left untouched.
        Upper bounds and structural constraints are always perturbed.
    denom : int
        Fixed denominator for perturbed rationals (default 100).

    Returns a **new** list; the original is not modified.
    """
    if sigma <= 0:
        return list(constraints)

    rng = np.random.RandomState(seed)
    k = max(1, round(sigma * denom))
    out: list = []

    for coeffs, rel, rhs in constraints:
        if skip_bounds and _is_nonnegativity(coeffs, rel, rhs):
            out.append((list(coeffs), rel, rhs))
            continue
        n_cols = len(coeffs)
        noise_c = np.clip(np.round(rng.randn(n_cols) * k).astype(int), -k, k)
        noise_r = int(np.clip(np.round(rng.randn() * k), -k, k))

        new_coeffs = [
            (int(c) * denom + int(p), denom) for c, p in zip(coeffs, noise_c)
        ]
        new_rhs = (int(rhs) * denom + noise_r, denom)
        out.append((new_coeffs, rel, new_rhs))

    return out
