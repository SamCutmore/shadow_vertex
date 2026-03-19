import math
import subprocess
import numpy as np
from typing import List, Tuple, Optional, Any, Dict

from .polyhedron import Polyhedron, Path
from .rotation import (rotation_planes, axis_label,
                       compute_vertex_tiers, angles_from_target_plane)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_scene(polyhedron: Polyhedron, path: Path):
    """Log-compressed normalization: reduces extreme scale ratios between
    dimensions while preserving relative proportions.  For equal-range
    polytopes (e.g. tesseract) this is identical to per-axis [-1,1] scaling."""
    if polyhedron.is_empty():
        return polyhedron, path
    verts = np.array(polyhedron.vertices, dtype=np.float64)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    half_range = (hi - lo) / 2.0
    half_range[half_range < 1e-12] = 1.0
    center = (lo + hi) / 2.0

    effective_half = half_range / np.log2(1.0 + half_range)

    norm_verts = ((verts - center) / effective_half).tolist()
    polyhedron = Polyhedron(norm_verts, polyhedron.faces)

    if path.points:
        pts = np.array(path.points, dtype=np.float64)
        norm_pts = ((pts - center) / effective_half).tolist()
        path = Path(norm_pts)

    return polyhedron, path


def _build_scene(constraints, history):
    """Build Polyhedron + Path from LP constraints and solver history.

    Returns (raw_poly, raw_path, norm_poly, norm_path) where raw_* are
    un-normalized and norm_* have per-dimension nD normalization applied.
    """
    path = Path.from_history(history)
    n_vars = len(constraints[0][0]) if constraints else 0
    interior_guess = [[0.5] * n_vars] if n_vars >= 3 else None
    polyhedron = Polyhedron.from_constraints(
        constraints, path.points if path.points else interior_guess,
    )
    if polyhedron.is_empty() and interior_guess is not None:
        polyhedron = Polyhedron.from_constraints(constraints, interior_guess)
    if polyhedron.is_empty():
        polyhedron = Polyhedron([], [])
    norm_poly, norm_path = _normalize_scene(polyhedron, path)
    return polyhedron, path, norm_poly, norm_path


def _ortho_basis(c_vec, d_vec):
    """Gram-Schmidt: e1 = d/||d||, e2 = (c - proj_d c) / norm.

    Returns (e1, e2, d_norm, c_orth_norm) or None if degenerate."""
    d_norm = np.linalg.norm(d_vec)
    if d_norm < 1e-12:
        return None
    e1 = d_vec / d_norm
    c_orth = c_vec - np.dot(c_vec, e1) * e1
    c_orth_norm = np.linalg.norm(c_orth)
    if c_orth_norm < 1e-12:
        return None
    e2 = c_orth / c_orth_norm
    return e1, e2, d_norm, c_orth_norm


def _project_to_ortho_plane(raw_dc, c_vec, d_vec, basis):
    """Transform raw (d·x, c·x) pairs into orthogonalised (e1·x, e2·x)."""
    e1, e2, d_norm, c_orth_norm = basis
    c_dot_d = np.dot(c_vec, d_vec)
    d_norm_sq = d_norm * d_norm
    out = []
    for d_val, c_val in raw_dc:
        px = d_val / d_norm
        py = (c_val - (c_dot_d / d_norm_sq) * d_val) / c_orth_norm
        out.append([float(px), float(py)])
    return out


def _build_shadow_scene(
    shadow_points: List[Tuple[float, float]],
    all_vertices: Optional[List[List[float]]] = None,
    objective: Optional[List[float]] = None,
    d_objective: Optional[List[float]] = None,
):
    """Build the 2D shadow polygon and solver path.

    When *all_vertices*, *objective*, and *d_objective* are provided the
    polygon is the convex hull of ALL polytope vertices projected onto the
    (d, c) plane – the true shadow polygon.  Coordinates use the same
    orthogonalised basis (e1, e2) as angles_from_target_plane so the
    shadow view and the "Shadow plane" snap produce identical geometry.

    Falls back to the raw (d·x, c·x) convex hull of visited *shadow_points*
    when the full polytope is unavailable."""
    if not shadow_points or len(shadow_points) < 2:
        return Polyhedron([], []), Path([])

    have_full = (all_vertices and objective and d_objective)
    basis = None
    if have_full:
        c_vec = np.asarray(objective, dtype=np.float64)
        d_vec = np.asarray(d_objective, dtype=np.float64)
        basis = _ortho_basis(c_vec, d_vec)

    if have_full and basis is not None:
        e1, e2 = basis[0], basis[1]
        verts_nd = np.asarray(all_vertices, dtype=np.float64)
        all_proj = np.column_stack([verts_nd @ e1, verts_nd @ e2])
        unique = np.unique(all_proj, axis=0)
        path_2d = _project_to_ortho_plane(shadow_points, c_vec, d_vec, basis)
    else:
        pts_2d = [[float(d), float(c)] for d, c in shadow_points]
        arr = np.array(pts_2d, dtype=np.float64)
        unique = np.unique(arr, axis=0)
        path_2d = pts_2d

    shadow_path = Path(path_2d)

    if len(unique) < 3:
        verts = unique.tolist()
        edges = [[0, 1]] if len(unique) == 2 else []
        poly = Polyhedron(verts, edges)
        return _normalize_scene(poly, shadow_path)

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(unique)
        hull_pts = unique[hull.vertices]
        verts = [list(map(float, row)) for row in hull_pts]
        n_hull = len(verts)
        cx = sum(v[0] for v in verts) / n_hull
        cy = sum(v[1] for v in verts) / n_hull
        order = sorted(range(n_hull),
                        key=lambda i: math.atan2(verts[i][1] - cy,
                                                 verts[i][0] - cx))
        verts = [verts[i] for i in order]
        faces = [list(range(n_hull))]
        poly = Polyhedron(verts, faces)
    except Exception:
        verts = [list(map(float, row)) for row in unique]
        edges = [[i, i + 1] for i in range(len(verts) - 1)]
        poly = Polyhedron(verts, edges)
    return _normalize_scene(poly, shadow_path)


_TIER_PALETTE = np.array([
    [0.40, 0.58, 0.93],  # tier 0 - base fallback (cornflower blue)
    [0.95, 0.85, 0.25],  # tier 1 - X  (gold)
    [1.00, 0.60, 0.20],  # tier 2 - Y  (orange)
    [0.30, 0.80, 0.40],  # tier 3 - Z  (green)
    [0.90, 0.35, 0.35],  # tier 4 - W  (coral)
    [0.65, 0.40, 0.85],  # tier 5 - V  (purple)
    [0.25, 0.80, 0.80],  # tier 6 - U  (teal)
    [0.95, 0.50, 0.70],  # tier 7 - S  (pink)
    [0.60, 0.85, 0.25],  # tier 8 - T  (lime)
], dtype=np.float32)


_BASE_DIM = 0


def _tier_colors(polyhedron: Polyhedron,
                 dim_toggles: Dict[int, List[bool]],
                 ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """Assign each vertex a color based on its highest occupied dimension.
    Disabled tiers (via *dim_toggles*) fall back to the base color.
    Returns (colors, tiers) or (None, None) if no tier toggle is enabled."""
    if polyhedron.is_empty() or polyhedron.dimension <= _BASE_DIM:
        return None, None
    if not any(v[0] for v in dim_toggles.values()):
        return None, None
    tiers = compute_vertex_tiers(polyhedron.vertices, _BASE_DIM)
    n_palette = len(_TIER_PALETTE)
    colors = []
    for t in tiers:
        if t > 0 and not dim_toggles.get(t, [True])[0]:
            colors.append(_TIER_PALETTE[0])
        else:
            colors.append(_TIER_PALETTE[min(t, n_palette - 1)])
    return np.array(colors, dtype=np.float32), tiers


def _direction_from_sliders(dim, sliders):
    """Build a RotationSpec dict from the current slider values."""
    planes = rotation_planes(dim)
    angles: Dict[Tuple[int, int], float] = {}
    for plane in planes:
        val = sliders.get(plane, [0.0])[0]
        if abs(val) > 1e-12:
            angles[plane] = math.radians(val)
    return angles if angles else None


PANEL_WIDTH = 250


def _angles_to_string(sliders, dim):
    """Format current slider angles as a copyable string."""
    planes = rotation_planes(dim)
    parts = []
    for (i, j) in planes:
        val = sliders.get((i, j), [0.0])[0]
        label = f"{axis_label(i)}{axis_label(j)}"
        parts.append(f"{label}={val:.1f}")
    return ",".join(parts)


def _string_to_angles(text, sliders, dim):
    """Parse an angle string and update sliders. Returns True if at least
    one angle was recognised."""
    planes = rotation_planes(dim)
    plane_map = {}
    for (i, j) in planes:
        label = f"{axis_label(i)}{axis_label(j)}"
        plane_map[label] = (i, j)
    matched = 0
    try:
        for token in text.split(","):
            token = token.strip()
            if "=" not in token:
                continue
            name, val_s = token.split("=", 1)
            plane = plane_map.get(name.strip())
            if plane is not None:
                sliders[plane][0] = float(val_s.strip())
                matched += 1
    except (ValueError, KeyError):
        return False
    return matched > 0


def _clipboard_copy(text):
    for cmd in [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ]:
        try:
            subprocess.run(cmd, input=text.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    import logging
    logging.debug("clipboard copy failed: xclip/xsel not found")


def _clipboard_paste():
    for cmd in [
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, check=True)
            return r.stdout.decode().strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    import logging
    logging.debug("clipboard paste failed: xclip/xsel not found")
    return None

_VIEW_PROJECTION = 0
_VIEW_SHADOW = 1


def _draw_imgui_panel(dim, sliders, win_w, win_h, dim_toggles,
                      has_shadow, view_mode, normalize,
                      objective=None, d_objective=None,
                      title="Projection"):
    """Draw the ImGui rotation-control panel anchored to the right edge.

    Returns True if any control changed (sliders, toggles, view mode).
    *normalize* ([bool]): True = per-dimension nD normalization,
    False = raw polytope. "Shadow plane" sets it to False; user can
    re-check it via the checkbox."""
    import imgui

    panel_x = max(0, win_w - PANEL_WIDTH)
    imgui.set_next_window_position(panel_x, 0)
    imgui.set_next_window_size(PANEL_WIDTH, win_h)
    flags = (imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE
             | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR)
    imgui.begin(title, None, flags)

    changed = False

    if has_shadow:
        imgui.text("View mode")
        if imgui.radio_button("Projection",
                              view_mode[0] == _VIEW_PROJECTION):
            if view_mode[0] != _VIEW_PROJECTION:
                view_mode[0] = _VIEW_PROJECTION
                changed = True
        imgui.same_line()
        if imgui.radio_button("Shadow",
                              view_mode[0] == _VIEW_SHADOW):
            if view_mode[0] != _VIEW_SHADOW:
                view_mode[0] = _VIEW_SHADOW
                changed = True
        imgui.separator()

    if view_mode[0] == _VIEW_SHADOW:
        imgui.text("Shadow polygon")
        imgui.spacing()
        imgui.text("d' x  vs  c' x")
        imgui.spacing()
        imgui.text("Use arrow keys to step.")
        imgui.end()
        return changed

    imgui.text("Projection")
    imgui.separator()

    planes = rotation_planes(dim)

    if not planes:
        imgui.text(f"No controls for {dim}D.")
        imgui.end()
        return changed

    prev_i = -1
    for (i, j) in planes:
        if i != prev_i:
            if prev_i >= 0:
                imgui.spacing()
                imgui.separator()
            imgui.text(f"{axis_label(i)} planes")
            imgui.spacing()
            prev_i = i
        label = f"{axis_label(i)}{axis_label(j)}"
        c, sliders[(i, j)][0] = imgui.slider_float(
            label, sliders[(i, j)][0], -180.0, 180.0)
        changed = changed or c

    imgui.spacing()
    if imgui.button("Reset"):
        for plane in planes:
            sliders[plane][0] = 0.0
        changed = True

    if objective is not None and d_objective is not None:
        imgui.same_line()
        if imgui.button("Shadow plane"):
            snap = angles_from_target_plane(objective, d_objective, dim)
            if snap is not None:
                for plane in planes:
                    sliders[plane][0] = math.degrees(snap.get(plane, 0.0))
                normalize[0] = False
                changed = True

    imgui.spacing()
    if imgui.button("Copy angles"):
        _clipboard_copy(_angles_to_string(sliders, dim))
    imgui.same_line()
    if imgui.button("Paste angles"):
        clip = _clipboard_paste()
        if clip is not None:
            if _string_to_angles(clip, sliders, dim):
                changed = True
            else:
                import logging
                logging.debug("Paste ignored: unrecognised angle format")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    toggled, normalize[0] = imgui.checkbox("Normalize", normalize[0])
    if toggled:
        changed = True

    if dim_toggles:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text("Dimension colours")
        tiers = sorted(dim_toggles)
        items_per_row = 3
        for idx, tier in enumerate(tiers):
            lbl = axis_label(_BASE_DIM + tier - 1)
            col = _TIER_PALETTE[min(tier, len(_TIER_PALETTE) - 1)]
            imgui.color_button(f"##swatch{tier}", col[0], col[1], col[2], 1.0,
                               0, 10, 10)
            imgui.same_line()
            toggled, dim_toggles[tier][0] = imgui.checkbox(
                lbl, dim_toggles[tier][0])
            if toggled:
                changed = True
            if idx < len(tiers) - 1 and (idx + 1) % items_per_row != 0:
                imgui.same_line()

    imgui.end()
    return changed


def _make_sliders(dim):
    """Create mutable slider values for all rotation planes of dimension dim."""
    return {plane: [0.0] for plane in rotation_planes(dim)}


def _imgui_mouse_button(wnd, btn, pressed):
    """Map window button id to ImGui mouse_down index."""
    import imgui
    io = imgui.get_io()
    idx = {wnd.mouse.left: 0, wnd.mouse.right: 1, wnd.mouse.middle: 2}.get(btn)
    if idx is not None:
        io.mouse_down[idx] = pressed


def _handle_scroll(x_offset, y_offset):
    """Feed scroll to ImGui (works around pyimgui attribute naming)."""
    import imgui
    io = imgui.get_io()
    io.mouse_wheel = float(y_offset)
    if hasattr(io, "mouse_wheel_horizontal"):
        io.mouse_wheel_horizontal = float(x_offset)


# ---------------------------------------------------------------------------
# 2D viewer
# ---------------------------------------------------------------------------

def run_visualization_2d(
    constraints: List[Tuple[List[float], str, float]],
    history: List[Any],
    solution: Optional[Any] = None,
    shadow_points: Optional[List[Tuple[float, float]]] = None,
    objective: Optional[List[float]] = None,
    d_objective: Optional[List[float]] = None,
    window_size: Tuple[int, int] = (800, 600),
) -> None:
    """2D shadow window. R-drag pan, wheel zoom, arrow-keys step.

    If *shadow_points* (from the shadow vertex solver) are provided, a
    radio toggle lets the user switch between the polytope projection
    and a fixed shadow polygon view.  When *objective* and *d_objective*
    are also provided, a "Shadow plane" button snaps the Givens sliders
    so the projection aligns with the (c, d) plane.
    """
    import moderngl_window
    from moderngl_window import WindowConfig
    from moderngl_window.integrations.imgui import ModernglWindowRenderer
    import moderngl
    import imgui

    poly_raw, path_raw, polyhedron, path = _build_scene(constraints, history)
    has_shadow = shadow_points is not None and len(shadow_points) >= 2
    shadow_poly, shadow_path = (
        _build_shadow_scene(shadow_points,
                            all_vertices=poly_raw.vertices if not poly_raw.is_empty() else None,
                            objective=objective,
                            d_objective=d_objective)
        if has_shadow
        else (Polyhedron([], []), Path([]))
    )

    step_idx = [0]
    shadow_step_idx = [0]
    n_steps = max(1, len(history))
    n_shadow_steps = max(1, len(shadow_points)) if shadow_points else 1
    win_w, win_h = window_size
    sliders = _make_sliders(polyhedron.dimension)
    n_extra = max(0, polyhedron.dimension - _BASE_DIM)
    dim_toggles: Dict[int, List[bool]] = {
        t: [False] for t in range(1, n_extra + 1)
    }
    view_mode = [_VIEW_PROJECTION]
    normalize = [True]

    class Cfg(WindowConfig):
        gl_version = (3, 3)
        window_size = (win_w, win_h)
        title = "2D Projection"

        def __init__(self, **kw):
            super().__init__(**kw)
            from .renderer_2d import Renderer2D
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.renderer = Renderer2D(self.ctx)
            imgui.create_context()
            self._imgui = ModernglWindowRenderer(self.wnd)
            self._mr = False
            self._refresh()

        def _dir(self):
            return _direction_from_sliders(polyhedron.dimension, sliders)

        def _active_poly_path(self):
            if normalize[0]:
                return polyhedron, path
            return poly_raw, path_raw

        def _refresh(self):
            if view_mode[0] == _VIEW_SHADOW:
                self.renderer.set_geometry(shadow_poly, shadow_path)
                self._step_shadow()
            else:
                d = self._dir()
                src_poly, src_path = self._active_poly_path()
                vcols, vtiers = _tier_colors(src_poly, dim_toggles)
                self.renderer.set_geometry(src_poly.to_2d(d),
                                           src_path.to_2d(d),
                                           vertex_colors=vcols,
                                           vertex_tiers=vtiers)
                self._step()

        def _step(self):
            i = step_idx[0]
            d = self._dir()
            _, src_path = self._active_poly_path()
            p2 = src_path.to_2d(d)
            self.renderer.set_path(p2.segment(i))
            pt = p2.current_point(i)
            if pt:
                self.renderer.set_current_point(pt[0], pt[1])
            else:
                self.renderer._point_vao = None

        def _step_shadow(self):
            i = shadow_step_idx[0]
            self.renderer.set_path(shadow_path.segment(i))
            pt = shadow_path.current_point(i)
            if pt:
                self.renderer.set_current_point(pt[0], pt[1])
            else:
                self.renderer._point_vao = None

        def on_render(self, time, frame_time):
            w, h = self.wnd.size
            bw, bh = self.wnd.buffer_size
            scene_w = max(1, w - PANEL_WIDTH)

            self.ctx.clear(0.08, 0.08, 0.12, 1.0)
            self.renderer.draw(scene_w / max(h, 1))

            self.ctx.viewport = (0, 0, w, h)
            io = imgui.get_io()
            io.display_size = (w, h)
            io.display_fb_scale = (1.0, 1.0)
            imgui.new_frame()
            if _draw_imgui_panel(polyhedron.dimension, sliders, w, h,
                                 dim_toggles, has_shadow, view_mode,
                                 normalize,
                                 objective=objective,
                                 d_objective=d_objective,
                                 title="Controls"):
                self._refresh()
            imgui.render()
            dd = imgui.get_draw_data()
            if dd:
                self._imgui.render(dd)
            self.ctx.viewport = (0, 0, bw, bh)

        def on_key_event(self, key, action, mods):
            self._imgui.key_event(key, action, mods)
            if imgui.get_io().want_capture_keyboard:
                return
            if action != self.wnd.keys.ACTION_PRESS:
                return
            if view_mode[0] == _VIEW_SHADOW:
                if key in (self.wnd.keys.RIGHT, self.wnd.keys.PERIOD):
                    shadow_step_idx[0] = min(
                        shadow_step_idx[0] + 1, n_shadow_steps - 1)
                    self._step_shadow()
                elif key in (self.wnd.keys.LEFT, self.wnd.keys.COMMA):
                    shadow_step_idx[0] = max(0, shadow_step_idx[0] - 1)
                    self._step_shadow()
            else:
                if key in (self.wnd.keys.RIGHT, self.wnd.keys.PERIOD):
                    step_idx[0] = min(step_idx[0] + 1, n_steps - 1)
                    self._step()
                elif key in (self.wnd.keys.LEFT, self.wnd.keys.COMMA):
                    step_idx[0] = max(0, step_idx[0] - 1)
                    self._step()

        def on_mouse_press_event(self, x, y, btn):
            imgui.get_io().mouse_pos = (x, y)
            _imgui_mouse_button(self.wnd, btn, True)
            if btn == 2: self._mr = True

        def on_mouse_release_event(self, x, y, btn):
            imgui.get_io().mouse_pos = (x, y)
            _imgui_mouse_button(self.wnd, btn, False)
            if btn == 2: self._mr = False

        def on_mouse_drag_event(self, x, y, dx, dy):
            imgui.get_io().mouse_pos = (x, y)
            if imgui.get_io().want_capture_mouse:
                return
            if self._mr:
                self.renderer.pan(float(dx), float(dy))

        def on_mouse_scroll_event(self, xo, yo):
            _handle_scroll(xo, yo)
            if imgui.get_io().want_capture_mouse:
                return
            self.renderer.zoom(yo)

        def on_mouse_position_event(self, x, y, dx, dy):
            imgui.get_io().mouse_pos = (x, y)

    moderngl_window.run_window_config(Cfg)
