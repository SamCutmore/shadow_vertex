"""2D wireframe: polyhedron + path (domain types) → OpenGL. Pan, zoom, ←/→ step."""

import numpy as np
from typing import List, Tuple, Optional, Union

from .polyhedron import Polyhedron, Path

VERTEX_SHADER = """#version 330
in vec2 in_vert;
in vec3 in_color;
uniform mat3 mvp;
out vec3 v_color;
void main() {
  vec2 p = (mvp * vec3(in_vert, 1.0)).xy;
  gl_Position = vec4(p, 0.0, 1.0);
  v_color = in_color;
}
"""
FRAGMENT_SHADER = """#version 330
in vec3 v_color;
out vec4 f_color;
void main() { f_color = vec4(v_color, 1.0); }
"""


def _verts_2d(v: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
        return np.zeros((0, 2), dtype=np.float64)
    if isinstance(v, np.ndarray):
        v = np.atleast_2d(v)
        if v.shape[1] >= 2:
            return v[:, :2].astype(np.float64)
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.array(v, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.atleast_2d(arr)
    return arr[:, :2].astype(np.float64) if arr.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)


def _center_scale_2d(verts: np.ndarray, path: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    pts = verts if verts is not None and verts.size >= 2 else np.zeros((0, 2))
    if path is not None and path.size >= 2:
        pts = np.vstack([pts, np.atleast_2d(path)[:, :2]]) if pts.size else np.atleast_2d(path)[:, :2]
    if pts.size == 0:
        return np.zeros(2, dtype=np.float32), np.float32(1.0)
    c = pts.mean(axis=0).astype(np.float32)
    r = float(np.linalg.norm(pts - c, axis=1).max())
    return c, np.float32(1.8 / max(r, 1e-6))


def _line_vbo_2d(verts: np.ndarray, edges: List[Tuple[int, int]],
                 color: Tuple[float, float, float],
                 vertex_colors: Optional[np.ndarray] = None) -> bytes:
    if verts.size == 0 or not edges:
        return b""
    data = []
    for i, j in edges:
        for idx in (i, j):
            x, y = float(verts[idx, 0]), float(verts[idx, 1])
            if vertex_colors is not None:
                r, g, b = vertex_colors[idx]
            else:
                r, g, b = color
            data.extend([x, y, r, g, b])
    return np.array(data, dtype=np.float32).tobytes()


def _path_vbo_2d(path: List[List[float]], color: Tuple[float, float, float]) -> bytes:
    if not path or len(path) < 2:
        return b""
    data = []
    for p in path:
        x = p[0] if len(p) > 0 else 0.0
        y = p[1] if len(p) > 1 else 0.0
        data.extend([x, y, color[0], color[1], color[2]])
    return np.array(data, dtype=np.float32).tobytes()


def _point_vbo_2d(x: float, y: float, color: Tuple[float, float, float], size: float) -> bytes:
    h = size
    # small square as two triangles (6 vertices) or 4 line segments
    corners = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    data = []
    for i, j in edges:
        for idx in (i, j):
            data.extend([corners[idx][0], corners[idx][1], color[0], color[1], color[2]])
    return np.array(data, dtype=np.float32).tobytes()


def _path_2d(path: Union[Path, List[List[float]]]) -> List[List[float]]:
    if isinstance(path, Path):
        return path.points
    if not path:
        return []
    return [[p[0], p[1] if len(p) > 1 else 0.0] for p in path]


class Renderer2D:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self._wire_vao = self._path_vao = self._point_vao = None
        self._center = np.zeros(2, dtype=np.float32)
        self._scale = 1.0
        self._pan_x = self._pan_y = 0.0
        self._zoom = 1.0

    def set_geometry(self, polyhedron: Polyhedron, path: Path,
                     vertex_colors: Optional[np.ndarray] = None,
                     vertex_tiers: Optional[List[int]] = None) -> None:
        if polyhedron.dimension != 2:
            polyhedron = polyhedron.to_2d()
        if path.dimension != 2:
            path = path.to_2d()
        v = _verts_2d(polyhedron.vertices)
        p2 = path.points
        path_arr = np.array(p2) if p2 else None
        self._center, self._scale = _center_scale_2d(v, path_arr)
        self._pan_x = self._pan_y = 0.0
        edges = polyhedron.edges
        if vertex_tiers is not None and edges:
            edges = sorted(edges,
                           key=lambda e: max(vertex_tiers[e[0]], vertex_tiers[e[1]]),
                           reverse=True)
        if v.size >= 2 and edges:
            buf = _line_vbo_2d(v, edges, (0.5, 0.65, 0.95),
                               vertex_colors=vertex_colors)
            self._wire_vao = self.ctx.vertex_array(
                self.prog, [(self.ctx.buffer(buf), "2f 3f", "in_vert", "in_color")]
            )
        else:
            self._wire_vao = None
        if p2 and len(p2) >= 2:
            self._path_vao = self.ctx.vertex_array(
                self.prog, [(self.ctx.buffer(_path_vbo_2d(p2, (1.0, 1.0, 1.0))), "2f 3f", "in_vert", "in_color")]
            )
        else:
            self._path_vao = None

    def set_path(self, path: Union[Path, List[List[float]]]) -> None:
        if isinstance(path, Path) and path.dimension != 2:
            path = path.to_2d()
        p2 = _path_2d(path)
        if p2 and len(p2) >= 2:
            self._path_vao = self.ctx.vertex_array(
                self.prog, [(self.ctx.buffer(_path_vbo_2d(p2, (1.0, 1.0, 1.0))), "2f 3f", "in_vert", "in_color")]
            )
        else:
            self._path_vao = None

    def set_current_point(self, x: float, y: float) -> None:
        sz = 0.02 / max(abs(self._scale), 1e-6)
        buf = _point_vbo_2d(x, y, (1.0, 0.45, 0.45), sz)
        self._point_vao = self.ctx.vertex_array(
            self.prog, [(self.ctx.buffer(buf), "2f 3f", "in_vert", "in_color")]
        )

    def pan(self, dx: float, dy: float) -> None:
        k = 0.002 / max(abs(self._scale), 1e-6)
        self._pan_x += dx * k
        self._pan_y -= dy * k

    def zoom(self, delta: float) -> None:
        self._zoom = max(0.2, min(10.0, self._zoom * (1.0 + delta * 0.1)))

    def _mvp(self, aspect: float) -> np.ndarray:
        c, s = self._center[0], self._center[1]
        scale = self._scale * self._zoom
        tx = -c + self._pan_x
        ty = -s + self._pan_y
        # orthographic: scale then translate; NDC is -1..1
        sx = scale / max(aspect, 1e-6)
        sy = scale
        m = np.array([
            [sx, 0, tx * sx],
            [0, sy, ty * sy],
            [0, 0, 1],
        ], dtype=np.float32)
        return m.T.astype(np.float32).tobytes()

    def draw(self, aspect: float) -> None:
        self.prog["mvp"].write(self._mvp(aspect))
        if self._wire_vao:
            self._wire_vao.render(self.ctx.LINES)
        if self._path_vao:
            self._path_vao.render(self.ctx.LINE_STRIP)
        if self._point_vao:
            self._point_vao.render(self.ctx.LINES)
