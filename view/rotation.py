import math
import numpy as np
from typing import Dict, List, Optional, Tuple

RotationSpec = Optional[Dict[Tuple[int, int], float]]

AXIS_LABELS = ["X", "Y", "Z", "W", "V", "U", "S", "T"]


def axis_label(i: int) -> str:
    if i < len(AXIS_LABELS):
        return AXIS_LABELS[i]
    return f"D{i}"


def rotation_planes(n: int) -> List[Tuple[int, int]]:
    """All n(n-1)/2 rotation planes for an n-dimensional space,
    ordered by (i, j) with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ---------------------------------------------------------------------------
# Givens rotation
# ---------------------------------------------------------------------------

def givens_rotation(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """n x n rotation matrix that rotates in the (i, j) plane by theta."""
    R = np.eye(n, dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def rotation_from_angles(n: int, angles: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Compose Givens rotations for all provided (i, j) -> theta pairs.

    Right-multiplied so that higher-index planes (iterated later) are applied
    to data points first, allowing "hidden" plane rotations (both axes beyond
    the projection target) to feed into "visible" plane rotations."""
    R = np.eye(n, dtype=np.float64)
    for (i, j), theta in angles.items():
        if abs(theta) > 1e-12:
            R = R @ givens_rotation(n, i, j, theta)
    return R


# ---------------------------------------------------------------------------
# Generic projector
# ---------------------------------------------------------------------------

def project_nd_points(
    points: List[List[float]], R: np.ndarray, target_dim: int,
) -> List[List[float]]:
    """Rotate each point by R, keep the first target_dim coordinates."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float64)
    rotated = pts @ R.T
    return rotated[:, :target_dim].tolist()


def compute_vertex_tiers(
    vertices: List[List[float]], target_dim: int,
) -> List[int]:
    """Return the highest extra-dimension tier each vertex occupies.

    0 = base (only lives in the first target_dim dimensions).
    1 = first extra dimension (index target_dim) is the highest occupied.
    2 = second extra dimension (index target_dim+1) is the highest occupied.
    ...and so on. Higher tiers take priority: a vertex that occupies both
    the 4th and 5th dimension gets tier 2 (the 5th dim tier).

    A dimension is considered "occupied" when the vertex's coordinate in
    that dimension is significantly different from the dimension's minimum
    across all vertices (normalized magnitude > 0.5).

    Uses original (un-rotated) coordinates so colours are rotation-stable."""
    if not vertices:
        return []
    pts = np.array(vertices, dtype=np.float64)
    n = pts.shape[1]
    if n <= target_dim:
        return [0] * len(pts)
    extra = pts[:, target_dim:]
    lo = extra.min(axis=0)
    hi = extra.max(axis=0)
    span = hi - lo
    span[span < 1e-12] = 1.0
    normed = (extra - lo) / span
    significant = normed > 0.5
    tiers = []
    for row in significant:
        tier = 0
        for k in range(len(row)):
            if row[k]:
                tier = k + 1
        tiers.append(tier)
    return tiers


# ---------------------------------------------------------------------------
# Givens decomposition (inverse of rotation_from_angles)
# ---------------------------------------------------------------------------

def decompose_givens(R: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Decompose an orthogonal matrix R into Givens rotation angles.

    Returns angles dict compatible with rotation_from_angles: iterating
    the dict in insertion order and composing via right-multiplication
    reproduces R (up to floating-point tolerance).

    Algorithm: peel Givens rotations from the right in reverse
    rotation_planes order, zeroing below-diagonal elements of R."""
    n = R.shape[0]
    M = R.astype(np.float64).copy()
    raw: Dict[Tuple[int, int], float] = {}
    for (i, j) in reversed(rotation_planes(n)):
        theta = math.atan2(M[j, i], M[j, j])
        raw[(i, j)] = theta
        if abs(theta) > 1e-15:
            M = M @ givens_rotation(n, i, j, -theta)
    return {plane: raw[plane] for plane in rotation_planes(n) if plane in raw}


def angles_from_target_plane(
    c: List[float], d: List[float], n: int,
) -> Optional[Dict[Tuple[int, int], float]]:
    """Compute Givens slider angles that project onto the plane spanned by
    c and d.  Row 0 of the resulting rotation aligns with d (normalised),
    row 1 with the component of c orthogonal to d.

    Returns None if c and d are parallel or degenerate."""
    c_arr = np.asarray(c, dtype=np.float64)[:n]
    d_arr = np.asarray(d, dtype=np.float64)[:n]

    d_norm = np.linalg.norm(d_arr)
    if d_norm < 1e-12:
        return None
    e1 = d_arr / d_norm

    c_orth = c_arr - np.dot(c_arr, e1) * e1
    c_orth_norm = np.linalg.norm(c_orth)
    if c_orth_norm < 1e-12:
        return None
    e2 = c_orth / c_orth_norm

    rows = [e1, e2]
    for k in range(n):
        ek = np.zeros(n, dtype=np.float64)
        ek[k] = 1.0
        v = ek.copy()
        for r in rows:
            v -= np.dot(v, r) * r
        if np.linalg.norm(v) > 1e-10:
            rows.append(v / np.linalg.norm(v))
        if len(rows) == n:
            break

    R = np.array(rows, dtype=np.float64)
    if np.linalg.det(R) < 0:
        R[-1] *= -1

    return decompose_givens(R)


# ---------------------------------------------------------------------------
# Direction-spec dispatcher
# ---------------------------------------------------------------------------

def rotation_from_direction(direction: RotationSpec, n: int) -> np.ndarray:
    """Convert a RotationSpec to an n x n rotation matrix.
    Returns identity when direction is None."""
    if direction is None:
        return np.eye(n, dtype=np.float64)
    return rotation_from_angles(n, direction)
