import numpy as np
from typing import List, Tuple, Optional, Any

from .rotation import RotationSpec, rotation_from_direction, project_nd_points


class Polyhedron:
    """Convex polyhedron in n dimensions: vertices + faces.

    Built from LP constraints via from_constraints().
    Use to_2d() with an optional RotationSpec to get a 2D shadow."""

    __slots__ = ("_vertices", "_faces")

    def __init__(self, vertices: List[List[float]], faces: List[List[int]]) -> None:
        self._vertices = [list(v) for v in vertices]
        self._faces = list(faces)

    @property
    def vertices(self) -> List[List[float]]:
        return self._vertices

    @property
    def faces(self) -> List[List[int]]:
        return self._faces

    @property
    def dimension(self) -> int:
        if not self._vertices:
            return 0
        return len(self._vertices[0])

    @property
    def edges(self) -> List[Tuple[int, int]]:
        seen: set = set()
        out: List[Tuple[int, int]] = []
        for face in self._faces:
            for a in range(len(face)):
                for b in range(a + 1, len(face)):
                    i, j = face[a], face[b]
                    if i > j:
                        i, j = j, i
                    if (i, j) not in seen:
                        seen.add((i, j))
                        out.append((i, j))
        return out

    def is_empty(self) -> bool:
        return len(self._vertices) == 0

    def to_2d(self, direction: RotationSpec = None) -> "Polyhedron":
        R = rotation_from_direction(direction, self.dimension) if self.dimension >= 2 else np.eye(max(self.dimension, 1))
        verts = project_nd_points(self._vertices, R, 2)
        return Polyhedron(verts, self._faces)

    # ------------------------------------------------------------------
    # Construction from LP constraints
    # ------------------------------------------------------------------

    @classmethod
    def from_constraints(
        cls,
        constraints: List[Tuple[List[float], str, float]],
        interior_or_path: Optional[List[List[float]]] = None,
    ) -> "Polyhedron":
        if not constraints:
            return cls([], [])
        n_vars = len(constraints[0][0])
        if n_vars == 2:
            verts, faces = cls._from_2d(constraints, interior_or_path)
            return cls(verts, faces)
        if n_vars >= 3:
            verts, faces = cls._from_nd(constraints, n_vars, interior_or_path)
            return cls(verts, faces)
        return cls([], [])

    @staticmethod
    def _norm(coeffs: List[float], rel: str, rhs: float) -> Tuple[np.ndarray, float]:
        r = (rel or "<=").strip().lower()
        a = np.array(coeffs, dtype=np.float64)
        b = float(rhs)
        if r in (">=", "geq"):
            a, b = -a, -b
        return a, b

    @staticmethod
    def _clip_2d(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
        if len(poly) < 3:
            return poly
        out = []
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            dp = np.dot(normal, p) - offset
            dq = np.dot(normal, q) - offset
            if dp <= 0:
                out.append(p)
            if (dp <= 0) != (dq <= 0):
                t = dp / (dp - dq) if (dp - dq) != 0 else 0
                out.append(p + t * (q - p))
        return np.array(out) if out else np.zeros((0, 2))

    @classmethod
    def _from_2d(
        cls,
        constraints: List[Tuple[List[float], str, float]],
        interior_or_path: Optional[List[List[float]]],
    ) -> Tuple[List[List[float]], List[List[int]]]:
        box: Optional[Tuple[float, float, float, float]] = None
        if interior_or_path and len(interior_or_path) >= 1:
            arr = np.array([[p[0], p[1]] for p in interior_or_path])
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            pad = max(float(np.ptp(arr)) * 0.5, 0.5)
            box = (float(lo[0]) - pad, float(lo[1]) - pad, float(hi[0]) + pad, float(hi[1]) + pad)
        poly = cls._polygon_2d(constraints, box)
        if len(poly) < 3:
            return [], []
        verts = [[float(p[0]), float(p[1])] for p in poly]
        return verts, [list(range(len(verts)))]

    @classmethod
    def _polygon_2d(
        cls,
        constraints: List[Tuple[List[float], str, float]],
        box: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        if box is None:
            box = (-1e6, -1e6, 1e6, 1e6)
        x0, y0, x1, y1 = box
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        for coeffs, rel, rhs in constraints:
            if len(coeffs) != 2:
                continue
            a, b = cls._norm(coeffs, rel, rhs)
            poly = cls._clip_2d(poly, a, b)
            if len(poly) < 3:
                return np.zeros((0, 2))
            if rel.strip().lower() in ("=", "eq", "=="):
                a, b = -a, -b
                poly = cls._clip_2d(poly, a, b)
                if len(poly) < 3:
                    return np.zeros((0, 2))
        if len(poly) >= 3:
            cw = (poly[1, 0] - poly[0, 0]) * (poly[2, 1] - poly[0, 1]) - (poly[1, 1] - poly[0, 1]) * (poly[2, 0] - poly[0, 0])
            if cw < 0:
                poly = poly[::-1]
        return poly

    @classmethod
    def _from_nd(
        cls,
        constraints: List[Tuple[List[float], str, float]],
        n_dim: int,
        interior_or_path: Optional[List[List[float]]],
    ) -> Tuple[List[List[float]], List[List[int]]]:
        try:
            from scipy.spatial import HalfspaceIntersection, ConvexHull
        except ImportError:
            return [], []
        rows = []
        for coeffs, rel, rhs in constraints:
            if len(coeffs) != n_dim:
                continue
            a, b = cls._norm(coeffs, rel, rhs)
            rows.append(np.append(a, -b))
            if rel.strip().lower() in ("=", "eq", "=="):
                rows.append(np.append(-a, b))
        if not rows:
            return [], []
        half = np.array(rows, dtype=np.float64)
        if interior_or_path and len(interior_or_path) >= 1:
            interior = np.asarray(interior_or_path[0], dtype=np.float64).reshape(n_dim)
        else:
            interior = np.ones(n_dim, dtype=np.float64) * 0.5
        if np.allclose(interior, 0):
            interior = np.ones(n_dim, dtype=np.float64) * 0.5
        try:
            hs = HalfspaceIntersection(half, interior)
            v = hs.intersections
            if len(v) < n_dim + 1:
                return [list(map(float, p)) for p in v], []
            hull = ConvexHull(v)
            verts = [list(map(float, v[i])) for i in range(len(v))]
            faces = [list(s) for s in hull.simplices]
            return verts, faces
        except Exception:
            return [], []


class Path:
    """Ordered n-D points (e.g. simplex path). Built from solver history."""

    __slots__ = ("_points",)

    def __init__(self, points: List[List[float]]) -> None:
        self._points = [list(p) for p in points]

    @property
    def points(self) -> List[List[float]]:
        return self._points

    @property
    def dimension(self) -> int:
        if not self._points:
            return 0
        return len(self._points[0])

    def to_2d(self, direction: RotationSpec = None) -> "Path":
        R = rotation_from_direction(direction, self.dimension) if self.dimension >= 2 else np.eye(max(self.dimension, 1))
        return Path(project_nd_points(self._points, R, 2))

    def segment(self, up_to_index: int) -> "Path":
        if not self._points:
            return Path([])
        idx = min(max(0, up_to_index), len(self._points) - 1)
        return Path(self._points[: idx + 1])

    def current_point(self, step_index: int) -> Optional[List[float]]:
        if not self._points or step_index < 0:
            return None
        return self._points[min(step_index, len(self._points) - 1)].copy()

    @classmethod
    def from_history(cls, history: Any) -> "Path":
        points = [list(s.primal) for s in history if hasattr(s, "primal")]
        return cls(points)
