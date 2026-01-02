"""Open3D GUI application for primitive fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import threading
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from primitives import (
    PlaneParam,
    CylinderParam,
    adaptive_seed_indices,
    expand_plane_from_seed,
    probe_cylinder_from_seed,
    expand_cylinder_from_seed,
    fit_cylinder,
    extract_stair_planes,
    create_cylinder_mesh,
    set_gpu_enabled,
    snap_axis_to_reference,
    fit_cylinder_with_axis,
    recompute_cylinder_length_from_inliers,
    score_cylinder_fit,
    robust_axis_from_cylinder_points,
    center_weighted_sample,
)


@dataclass
class SensorProfile:
    name: str
    voxel_size: float
    r_min: float
    r_max: float
    r_step: float
    min_points: int
    plane_distance_threshold: float
    cylinder_distance_threshold: float


SENSOR_PROFILES: Dict[str, SensorProfile] = {
    "mid70_map": SensorProfile(
        name="Livox Mid-70 (Map)",
        voxel_size=0.02,
        r_min=0.2,
        r_max=1.0,
        r_step=0.1,
        min_points=50,
        plane_distance_threshold=0.01,
        cylinder_distance_threshold=0.02,
    ),
    "velodyne_map": SensorProfile(
        name="Velodyne (Map)",
        voxel_size=0.05,
        r_min=0.3,
        r_max=1.5,
        r_step=0.15,
        min_points=60,
        plane_distance_threshold=0.03,
        cylinder_distance_threshold=0.04,
    ),
    "mid70_stairs": SensorProfile(
        name="Livox Mid-70 (Stairs Mode)",
        voxel_size=0.03,
        r_min=0.5,
        r_max=3.0,
        r_step=0.2,
        min_points=200,
        plane_distance_threshold=0.025,
        cylinder_distance_threshold=0.03,
    ),
}

PROFILE_CUSTOM_LABEL = "カスタム"
PROFILE_LABELS: Dict[str, str] = {
    "mid70_map": "mid70_map（Livox Mid-70/マップ）",
    "velodyne_map": "velodyne_map（Velodyne/マップ）",
    "mid70_stairs": "mid70_stairs（Livox Mid-70/階段）",
}

MODE_PLANE = "平面"
MODE_CYLINDER = "円柱"
MODE_STAIRS = "階段"

EXPAND_METHOD_LABELS: Dict[str, str] = {
    "component": "連結成分",
    "bfs": "BFS",
}
PATCH_SHAPE_LABELS: Dict[str, str] = {
    "hull": "凸包",
    "rect": "矩形",
}


# =============================================================================
# Geometry helpers (subset copied from main.py to avoid import cycles)
# =============================================================================

def generate_plane_colors(n: int) -> List[np.ndarray]:
    """Generate n distinct colors for plane visualization."""
    colors: List[np.ndarray] = []
    if n <= 0:
        return colors
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.75
        v = 0.9
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1 / 6:
            r, g, b = c, x, 0
        elif h < 2 / 6:
            r, g, b = x, c, 0
        elif h < 3 / 6:
            r, g, b = 0, c, x
        elif h < 4 / 6:
            r, g, b = 0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append(np.array([r + m, g + m, b + m]))
    return colors


def _plane_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-12:
        raise ValueError("Invalid plane normal")
    normal = normal / normal_norm

    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))

    u_norm = float(np.linalg.norm(u))
    if not np.isfinite(u_norm) or u_norm < 1e-12:
        raise ValueError("Failed to compute plane basis")
    u = u / u_norm
    v = np.cross(normal, u)
    return u, v


def _convex_hull_2d(points_2d: np.ndarray, *, round_decimals: int = 6) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must be (N, 2)")

    points_2d = points_2d[np.all(np.isfinite(points_2d), axis=1)]
    if len(points_2d) == 0:
        return np.empty((0, 2), dtype=float)

    pts = np.unique(np.round(points_2d, decimals=round_decimals), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    return hull


def _polygon_area_2d(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _minimum_area_bounding_rectangle(
    hull_2d: np.ndarray,
) -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
    hull_2d = np.asarray(hull_2d, dtype=float)
    if hull_2d.ndim != 2 or hull_2d.shape[1] != 2 or len(hull_2d) < 3:
        return None

    best_area = float("inf")
    best_rect: Optional[np.ndarray] = None
    best_extents: Optional[Tuple[float, float]] = None

    num = len(hull_2d)
    for i in range(num):
        p0 = hull_2d[i]
        p1 = hull_2d[(i + 1) % num]
        edge = p1 - p0
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm < 1e-12 or not np.isfinite(edge_norm):
            continue

        c = edge[0] / edge_norm
        s = edge[1] / edge_norm
        rot = np.array([[c, s], [-s, c]])
        rotated = hull_2d @ rot.T

        min_x, max_x = float(rotated[:, 0].min()), float(rotated[:, 0].max())
        min_y, max_y = float(rotated[:, 1].min()), float(rotated[:, 1].max())

        extent_u = max_x - min_x
        extent_v = max_y - min_y
        area = extent_u * extent_v
        if not np.isfinite(area):
            continue

        if area < best_area:
            best_area = area
            inv_rot = np.array([[c, -s], [s, c]])
            rect = np.array(
                [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ],
                dtype=float,
            )
            best_rect = rect @ inv_rot.T
            best_extents = (float(extent_u), float(extent_v))

    if best_rect is None or best_extents is None or best_area <= 0.0:
        return None
    return best_rect, float(best_area), best_extents


def create_plane_patch_mesh(
    plane: PlaneParam,
    roi_points: np.ndarray,
    color: np.ndarray,
    *,
    padding: float = 0.0,
    patch_shape: str = "hull",
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, object]]:
    roi_points = np.asarray(roi_points, dtype=float)
    if roi_points.ndim != 2 or roi_points.shape[1] != 3:
        raise ValueError("roi_points must be (N, 3)")

    if plane.inlier_indices is None or len(plane.inlier_indices) == 0:
        raise ValueError("Plane has no inlier indices")

    indices = np.asarray(plane.inlier_indices, dtype=int).reshape(-1)
    indices = indices[(0 <= indices) & (indices < len(roi_points))]
    inlier_points = roi_points[indices]
    inlier_points = inlier_points[np.all(np.isfinite(inlier_points), axis=1)]
    if len(inlier_points) < 3:
        raise ValueError("Too few inlier points")

    normal = np.asarray(plane.normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-12:
        raise ValueError("Invalid plane normal")
    normal = normal / normal_norm

    origin = np.asarray(plane.point, dtype=float)
    rel = inlier_points - origin
    distances = rel @ normal
    projected = inlier_points - distances[:, None] * normal[None, :]
    origin = projected.mean(axis=0)

    u, v = _plane_basis_from_normal(normal)
    local = projected - origin
    coords = np.column_stack((local @ u, local @ v))

    extent_u = float(coords[:, 0].max() - coords[:, 0].min())
    extent_v = float(coords[:, 1].max() - coords[:, 1].min())

    hull_2d = _convex_hull_2d(coords)
    hull_area = _polygon_area_2d(hull_2d)

    def axis_aligned_bbox() -> np.ndarray:
        min_u, max_u = float(coords[:, 0].min()), float(coords[:, 0].max())
        min_v, max_v = float(coords[:, 1].min()), float(coords[:, 1].max())
        return np.array(
            [
                [min_u, min_v],
                [max_u, min_v],
                [max_u, max_v],
                [min_u, max_v],
            ],
            dtype=float,
        )

    hull_valid = len(hull_2d) >= 3
    requested_shape = patch_shape.lower()
    selected_shape = requested_shape if requested_shape in ("hull", "rect") else "hull"
    fallback_reason = ""
    rect_corners_2d: Optional[np.ndarray] = None
    rect_extents: Optional[Tuple[float, float]] = None

    if selected_shape == "rect" and hull_valid:
        rect_result = _minimum_area_bounding_rectangle(hull_2d)
        if rect_result is not None:
            rect_corners_2d, rect_area, rect_extents = rect_result
            if rect_area < 1e-12 or min(rect_extents) < 1e-8:
                rect_corners_2d = None
        if rect_corners_2d is None:
            selected_shape = "hull"
            fallback_reason = "rect_degenerate"
    elif selected_shape == "rect" and not hull_valid:
        selected_shape = "hull"
        fallback_reason = "insufficient_hull_points"

    if selected_shape == "rect" and rect_corners_2d is not None:
        polygon_2d = rect_corners_2d
    elif hull_valid:
        polygon_2d = hull_2d
    else:
        polygon_2d = axis_aligned_bbox()

    if padding > 0:
        centroid_2d = polygon_2d.mean(axis=0)
        vec = polygon_2d - centroid_2d
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        polygon_2d = centroid_2d + vec * ((norms + padding) / np.maximum(norms, 1e-9))

    vertices = origin + polygon_2d[:, 0:1] * u[None, :] + polygon_2d[:, 1:2] * v[None, :]

    if len(vertices) >= 3:
        poly_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if float(np.dot(poly_normal, normal)) < 0:
            polygon_2d = polygon_2d[::-1]
            vertices = vertices[::-1]

    area = _polygon_area_2d(polygon_2d)

    triangles = np.array([[0, i, i + 1] for i in range(1, len(vertices) - 1)], dtype=int)
    if len(triangles) == 0:
        raise ValueError("Failed to triangulate plane patch")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(vertices), 1)))
    mesh.compute_vertex_normals()

    rect_corners_world = None
    if selected_shape == "rect" and len(polygon_2d) == 4:
        rect_corners_world = origin + polygon_2d[:, 0:1] * u[None, :] + polygon_2d[:, 1:2] * v[None, :]
        edge_u_vec = polygon_2d[1] - polygon_2d[0]
        edge_v_vec = polygon_2d[3] - polygon_2d[0]
        rect_extents = (
            float(np.linalg.norm(edge_u_vec)),
            float(np.linalg.norm(edge_v_vec)),
        )

    metrics: Dict[str, object] = {
        "extent_u": extent_u,
        "extent_v": extent_v,
        "area": float(area),
        "hull_area": float(hull_area),
        "patch_shape": selected_shape,
        "patch_shape_requested": requested_shape,
        "corners_world": vertices,
    }
    if rect_corners_world is not None:
        metrics["rect_corners_world"] = rect_corners_world
    if rect_extents is not None:
        metrics["rect_extent_u"] = float(rect_extents[0])
        metrics["rect_extent_v"] = float(rect_extents[1])
        metrics["rect_area"] = float(metrics["rect_extent_u"] * metrics["rect_extent_v"])
    if fallback_reason:
        metrics["patch_fallback_reason"] = fallback_reason

    return mesh, metrics


# =============================================================================
# Result I/O
# =============================================================================

def load_results(filepath: str) -> dict:
    path = Path(filepath)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"planes": [], "cylinders": []}


def save_results(results: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def append_plane_result(results: dict, plane: PlaneParam) -> dict:
    plane_id = len(results["planes"])
    results["planes"].append(
        {
            "id": plane_id,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "inlier_count": plane.inlier_count,
        }
    )
    return results


def append_cylinder_result(results: dict, cylinder: CylinderParam) -> dict:
    cylinder_id = len(results["cylinders"])
    results["cylinders"].append(
        {
            "id": cylinder_id,
            "axis_point": cylinder.axis_point.tolist(),
            "axis_direction": cylinder.axis_direction.tolist(),
            "radius": cylinder.radius,
            "length": cylinder.length,
            "inlier_count": cylinder.inlier_count,
        }
    )
    return results


def save_stairs_results(
    planes: List[PlaneParam],
    roi_points: np.ndarray,
    filepath: str,
    *,
    patch_shape: str = "hull",
):
    result = {
        "mode": "stairs",
        "plane_count": len(planes),
        "planes": [],
    }

    for i, plane in enumerate(planes):
        plane_entry = {
            "id": i,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "height": plane.height,
            "inlier_count": plane.inlier_count,
        }
        try:
            _, metrics = create_plane_patch_mesh(
                plane,
                roi_points,
                np.array([0.2, 0.8, 0.2]),
                padding=0.02,
                patch_shape=patch_shape,
            )
            plane_entry["patch_shape"] = metrics.get("patch_shape", patch_shape)
            if metrics.get("rect_corners_world") is not None:
                plane_entry["rect_corners_world"] = np.asarray(
                    metrics["rect_corners_world"]
                ).tolist()
            if metrics.get("patch_fallback_reason"):
                plane_entry["patch_fallback_reason"] = metrics["patch_fallback_reason"]
        except Exception as exc:
            plane_entry["patch_shape"] = patch_shape
            plane_entry["patch_error"] = str(exc)
        result["planes"].append(plane_entry)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)


def _profile_display(key: str) -> str:
    return PROFILE_LABELS.get(key, key)


def _profile_key_from_display(text: str) -> Optional[str]:
    if text == PROFILE_CUSTOM_LABEL:
        return None
    for key, label in PROFILE_LABELS.items():
        if text == label:
            return key
    if text in SENSOR_PROFILES:
        return text
    return None


def _expand_method_value(text: str) -> str:
    for key, label in EXPAND_METHOD_LABELS.items():
        if text == label:
            return key
    return text if text in EXPAND_METHOD_LABELS else "component"


def _patch_shape_value(text: str) -> str:
    for key, label in PATCH_SHAPE_LABELS.items():
        if text == label:
            return key
    return text if text in PATCH_SHAPE_LABELS else "hull"


def _set_japanese_font(app: gui.Application, font_override: Optional[str] = None) -> None:
    """Install a CJK-capable font for Japanese UI labels if available."""
    candidates: List[str] = []
    if font_override:
        candidates.append(font_override)
    candidates += [
        "NotoSansCJK",
        "Noto Sans CJK JP",
        "NotoSansCJKJP",
        "Noto Sans CJK",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc",
    ]

    for typeface in candidates:
        if typeface.startswith("/") and not Path(typeface).exists():
            continue
        try:
            font_desc = gui.FontDescription()
            font_desc.add_typeface_for_language(typeface, "ja")
            app.set_font(app.DEFAULT_FONT_ID, font_desc)
            return
        except Exception:
            continue

    print(
        "Warning: 日本語フォントが見つかりません。"
        "GUI表示が文字化けする可能性があります。"
    )


# =============================================================================
# Point cloud I/O
# =============================================================================

def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"点群の読み込みに失敗しました: {filepath}")
    return pcd


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    normal_radius: float = 0.05,
) -> o3d.geometry.PointCloud:
    result = pcd
    if voxel_size > 0:
        result = result.voxel_down_sample(voxel_size)
    if nb_neighbors > 0:
        result, _ = result.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    result.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    result.orient_normals_consistent_tangent_plane(k=15)
    return result


# =============================================================================
# GUI Application
# =============================================================================

class PrimitiveFittingApp:
    def __init__(
        self,
        *,
        initial_path: Optional[str] = None,
        initial_profile: Optional[str] = None,
        output_path: str = "fit_results.json",
        stairs_output_path: str = "stairs_results.json",
    ):
        self._app = gui.Application.instance
        self.window = self._app.create_window("Primitive Fitting GUI", 1400, 900)
        self.window.set_on_layout(self._on_layout)

        self._pcd_material = rendering.MaterialRecord()
        self._pcd_material.shader = "defaultUnlit"
        self._pcd_material.point_size = 3 * self.window.scaling

        self._mesh_material = rendering.MaterialRecord()
        self._mesh_material.shader = "defaultLit"
        self._line_material = rendering.MaterialRecord()
        self._line_material.shader = "unlitLine"
        self._line_material.line_width = 1.0 * self.window.scaling
        self._drag_rect_material = rendering.MaterialRecord()
        self._drag_rect_material.shader = "unlitLine"
        self._drag_rect_material.line_width = 2.0 * self.window.scaling
        self._highlight_material = rendering.MaterialRecord()
        self._highlight_material.shader = "unlitLine"
        self._highlight_material.line_width = 2.5 * self.window.scaling

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        self.scene_widget.set_on_mouse(self._on_mouse)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.scene_widget)

        self.axes_widget = gui.SceneWidget()
        self.axes_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.axes_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
        self.axes_widget.scene.add_geometry("axes", axes, self._mesh_material)
        self.window.add_child(self.axes_widget)

        self.panel = gui.ScrollableVert(0, gui.Margins(8, 8, 8, 8))
        self.window.add_child(self.panel)
        self.window.set_on_key(self._on_key)

        self._rng = np.random.default_rng(1234)

        self._result_names: List[str] = []
        self._result_meshes: List[o3d.geometry.TriangleMesh] = []
        self._seed_name: Optional[str] = None
        self._pointcloud_name: Optional[str] = None
        self._grid_name: Optional[str] = None

        self._objects: Dict[str, Dict[str, object]] = {}
        self._outliner_names: List[str] = []
        self._selected_outliner_index: int = -1
        self._object_counter = 0
        self._selection_highlight_name: Optional[str] = None
        self._selection_highlight_target: Optional[str] = None
        self._loading: bool = False
        self._pcd_t: Optional[o3d.t.geometry.PointCloud] = None
        self._cuda_available: bool = self._check_cuda_available()
        self._last_loaded_path: Optional[str] = None
        self._roi: Optional[Dict[str, object]] = None
        self._roi_name: Optional[str] = None
        self._roi_gizmo_name: Optional[str] = None
        self._roi_drag_start: Optional[Tuple[int, int]] = None
        self._roi_drag_start_center: Optional[np.ndarray] = None
        self._roi_drag_start_rot: Optional[np.ndarray] = None
        self._roi_drag_start_params: Optional[Tuple[float, float, float]] = None
        self._last_mouse_pos: Tuple[int, int] = (0, 0)
        self._transform_mode: Optional[str] = None
        self._transform_axis: Optional[str] = None
        self._transform_backup: Optional[Dict[str, object]] = None

        self.pcd_raw: Optional[o3d.geometry.PointCloud] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.all_points: Optional[np.ndarray] = None
        self.all_normals: Optional[np.ndarray] = None
        self._kdtree: Optional[o3d.geometry.KDTreeFlann] = None
        self.ground_plane: Optional[PlaneParam] = None
        self._ground_name: Optional[str] = None
        self.ceiling_plane: Optional[PlaneParam] = None
        self._ceiling_name: Optional[str] = None
        self._all_points_original: Optional[np.ndarray] = None
        self._all_normals_original: Optional[np.ndarray] = None
        self._edit_mask: Optional[np.ndarray] = None
        self._edit_history: List[np.ndarray] = []
        self._current_indices: Optional[np.ndarray] = None
        self._box_select_start: Optional[Tuple[int, int]] = None
        self._drag_rect_name: Optional[str] = None
        self._shift_pick_start: Optional[Tuple[int, int]] = None
        self._shift_pick_dragging: bool = False

        self.last_pick: Optional[np.ndarray] = None

        self.results = load_results(output_path)
        self.output_path = output_path
        self.stairs_output_path = stairs_output_path

        self._build_controls(initial_profile)

        if initial_path:
            self._load_point_cloud(initial_path)

    def _build_controls(self, initial_profile: Optional[str]):
        self.status = gui.Label("点群を読み込み、Shift+クリックでseed/ROIを指定してください。")
        self.panel.add_child(self.status)

        self.ui_mode_combo = gui.Combobox()
        self.ui_mode_combo.add_item("基本")
        self.ui_mode_combo.add_item("詳細")
        self.ui_mode_combo.selected_text = "基本"
        self.ui_mode_combo.set_on_selection_changed(self._on_ui_mode_changed)
        self.panel.add_child(self._labeled_row("UIモード", self.ui_mode_combo))

        self.file_group = gui.CollapsableVert("入力", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.file_group)
        self.file_group.set_is_open(False)

        self.input_path = gui.TextEdit()
        self.input_path.placeholder_text = "PCD/PLY のパス"
        self.file_group.add_child(self.input_path)

        file_buttons = gui.Horiz(4)
        self.open_button = gui.Button("開く...")
        self.open_button.set_on_clicked(self._on_open_dialog)
        self.load_button = gui.Button("読み込み")
        self.load_button.set_on_clicked(self._on_load_clicked)
        file_buttons.add_child(self.open_button)
        file_buttons.add_child(self.load_button)
        self.file_group.add_child(file_buttons)

        self.load_normals_checkbox = gui.Checkbox("読み込み時に法線推定")
        self.load_normals_checkbox.checked = True
        self.file_group.add_child(self.load_normals_checkbox)

        self.use_gpu_checkbox = gui.Checkbox("GPUを使用（前処理/計算）")
        self.use_gpu_checkbox.checked = False
        self.file_group.add_child(self.use_gpu_checkbox)

        self.drop_raw_checkbox = gui.Checkbox("メモリ節約（RAW保持しない）")
        self.drop_raw_checkbox.checked = True
        self.file_group.add_child(self.drop_raw_checkbox)

        self.preprocess_group = gui.CollapsableVert("前処理", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.preprocess_group)
        self.preprocess_group.set_is_open(False)

        self.preprocess_checkbox = gui.Checkbox("前処理を有効化")
        self.preprocess_checkbox.checked = True
        self.preprocess_group.add_child(self.preprocess_checkbox)

        self.voxel_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.voxel_size.double_value = 0.02
        voxel_row = gui.Horiz(4)
        voxel_row.add_child(gui.Label("ボクセルサイズ"))
        voxel_row.add_child(self.voxel_size)
        self.preprocess_group.add_child(voxel_row)

        preprocess_buttons = gui.Horiz(4)
        self.apply_preprocess_button = gui.Button("前処理を適用")
        self.apply_preprocess_button.set_on_clicked(self._on_apply_preprocess)
        self.reset_preprocess_button = gui.Button("前処理をリセット")
        self.reset_preprocess_button.set_on_clicked(self._on_reset_preprocess)
        preprocess_buttons.add_child(self.apply_preprocess_button)
        preprocess_buttons.add_child(self.reset_preprocess_button)
        self.preprocess_group.add_child(preprocess_buttons)

        self.sensor_profile_group = gui.CollapsableVert("センサプロファイル", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.sensor_profile_group)
        self.sensor_profile_group.set_is_open(False)

        self.profile_combo = gui.Combobox()
        self.profile_combo.add_item(PROFILE_CUSTOM_LABEL)
        for key in SENSOR_PROFILES:
            self.profile_combo.add_item(_profile_display(key))
        self.profile_combo.set_on_selection_changed(self._on_profile_changed)
        if initial_profile and initial_profile in SENSOR_PROFILES:
            self.profile_combo.selected_text = _profile_display(initial_profile)
        else:
            self.profile_combo.selected_text = PROFILE_CUSTOM_LABEL
        self.sensor_profile_group.add_child(self.profile_combo)

        self.mode_group = gui.CollapsableVert("モード", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.mode_group)
        self.mode_group.set_is_open(False)

        self.mode_combo = gui.Combobox()
        self.mode_combo.add_item(MODE_PLANE)
        self.mode_combo.add_item(MODE_CYLINDER)
        self.mode_combo.add_item(MODE_STAIRS)
        self.mode_combo.selected_text = MODE_PLANE
        self.mode_combo.set_on_selection_changed(self._on_mode_changed)
        self.mode_group.add_child(self.mode_combo)

        self.auto_run_checkbox = gui.Checkbox("クリック時はseedのみ（自動実行しない）")
        self.auto_run_checkbox.checked = False
        self.mode_group.add_child(self.auto_run_checkbox)


        self.keep_results_checkbox = gui.Checkbox("結果を保持")
        self.keep_results_checkbox.checked = True
        self.mode_group.add_child(self.keep_results_checkbox)

        run_row = gui.Horiz(4)
        self.run_button = gui.Button("実行")
        self.run_button.set_on_clicked(self._on_run_clicked)
        self.clear_button = gui.Button("結果クリア")
        self.clear_button.set_on_clicked(self._on_clear_clicked)
        run_row.add_child(self.run_button)
        run_row.add_child(self.clear_button)
        self.mode_group.add_child(run_row)

        self.outliner_group = gui.CollapsableVert("アウトライナー", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.outliner_group)
        self.outliner_group.set_is_open(False)

        self.outliner = gui.ListView()
        self.outliner.set_max_visible_items(8)
        self.outliner.set_on_selection_changed(self._on_outliner_select)
        self.outliner_group.add_child(self.outliner)

        outliner_row = gui.Horiz(4)
        self.outliner_toggle_button = gui.Button("表示/非表示")
        self.outliner_toggle_button.set_on_clicked(self._on_toggle_visibility)
        self.outliner_solid_button = gui.Button("ソリッド")
        self.outliner_solid_button.set_on_clicked(lambda: self._on_set_display_mode("solid"))
        self.outliner_wire_button = gui.Button("ワイヤ")
        self.outliner_wire_button.set_on_clicked(lambda: self._on_set_display_mode("wire"))
        outliner_row.add_child(self.outliner_toggle_button)
        outliner_row.add_child(self.outliner_solid_button)
        outliner_row.add_child(self.outliner_wire_button)
        self.outliner_group.add_child(outliner_row)

        outliner_action_row = gui.Horiz(4)
        self.outliner_delete_button = gui.Button("選択を削除")
        self.outliner_delete_button.set_on_clicked(self._on_delete_selected)
        outliner_action_row.add_child(self.outliner_delete_button)
        self.outliner_group.add_child(outliner_action_row)

        outliner_assign_row = gui.Horiz(4)
        self.outliner_ground_button = gui.Button("地面に設定")
        self.outliner_ground_button.set_on_clicked(lambda: self._on_assign_plane("ground"))
        self.outliner_ceiling_button = gui.Button("天井に設定")
        self.outliner_ceiling_button.set_on_clicked(lambda: self._on_assign_plane("ceiling"))
        outliner_assign_row.add_child(self.outliner_ground_button)
        outliner_assign_row.add_child(self.outliner_ceiling_button)
        self.outliner_group.add_child(outliner_assign_row)

        self.outliner_color = gui.ColorEdit()
        self.outliner_color.color_value = gui.Color(0.8, 0.2, 0.2, 1.0)
        self.outliner_group.add_child(self._labeled_row("色", self.outliner_color))
        self.outliner_apply_color = gui.Button("色を適用")
        self.outliner_apply_color.set_on_clicked(self._on_apply_color)
        self.outliner_group.add_child(self.outliner_apply_color)

        self.roi_group = gui.CollapsableVert("ROI / シード", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.roi_group)
        self.roi_group.set_is_open(False)

        roi_create_row = gui.Horiz(4)
        self.roi_cylinder_button = gui.Button("円柱ROI作成")
        self.roi_cylinder_button.set_on_clicked(lambda: self._create_roi("cylinder"))
        self.roi_box_button = gui.Button("箱ROI作成")
        self.roi_box_button.set_on_clicked(lambda: self._create_roi("box"))
        self.roi_clear_button = gui.Button("ROI削除")
        self.roi_clear_button.set_on_clicked(self._clear_roi)
        roi_create_row.add_child(self.roi_cylinder_button)
        roi_create_row.add_child(self.roi_box_button)
        roi_create_row.add_child(self.roi_clear_button)
        self.roi_group.add_child(roi_create_row)

        self.roi_use_checkbox = gui.Checkbox("ROIを使用")
        self.roi_use_checkbox.checked = False
        self.roi_group.add_child(self.roi_use_checkbox)

        self.roi_edit_checkbox = gui.Checkbox("ROI編集モード")
        self.roi_edit_checkbox.checked = False
        self.roi_edit_checkbox.set_on_checked(self._on_roi_edit_toggle)
        self.roi_group.add_child(self.roi_edit_checkbox)

        self.roi_transform_combo = gui.Combobox()
        self.roi_transform_combo.add_item("移動")
        self.roi_transform_combo.add_item("回転")
        self.roi_transform_combo.add_item("スケール")
        self.roi_transform_combo.selected_text = "移動"
        self.roi_group.add_child(self._labeled_row("変換", self.roi_transform_combo))

        self.roi_axis_combo = gui.Combobox()
        self.roi_axis_combo.add_item("自由")
        self.roi_axis_combo.add_item("X")
        self.roi_axis_combo.add_item("Y")
        self.roi_axis_combo.add_item("Z")
        self.roi_axis_combo.selected_text = "自由"
        self.roi_group.add_child(self._labeled_row("軸", self.roi_axis_combo))

        self.roi_axis_lock_checkbox = gui.Checkbox("ROI軸を固定（上級）")
        self.roi_axis_lock_checkbox.checked = False
        self.roi_group.add_child(self.roi_axis_lock_checkbox)

        self.roi_data_axis_checkbox = gui.Checkbox("ROI内から初期円柱を推定")
        self.roi_data_axis_checkbox.checked = True
        self.roi_group.add_child(self.roi_data_axis_checkbox)

        self.roi_add_combo = gui.Combobox()
        self.roi_add_combo.add_item("円柱")
        self.roi_add_combo.add_item("箱")
        self.roi_add_combo.selected_text = "円柱"
        self.roi_group.add_child(self._labeled_row("Shift+A追加", self.roi_add_combo))

        self.roi_apply_button = gui.Button("この形でROI適用（seed設定）")
        self.roi_apply_button.set_on_clicked(self._apply_roi_to_seed)
        self.roi_group.add_child(self.roi_apply_button)

        self.adaptive_roi_checkbox = gui.Checkbox("適応ROI")
        self.adaptive_roi_checkbox.checked = True
        self.roi_group.add_child(self.adaptive_roi_checkbox)

        self.roi_r_min = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.roi_r_min.double_value = 0.2
        self.roi_r_max = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.roi_r_max.double_value = 1.0
        self.roi_r_step = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.roi_r_step.double_value = 0.1
        self.roi_min_points = gui.NumberEdit(gui.NumberEdit.INT)
        self.roi_min_points.int_value = 50
        self.pick_snap_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.pick_snap_radius.double_value = 0.05

        self.roi_group.add_child(self._labeled_row("r_min (最小半径)", self.roi_r_min))
        self.roi_group.add_child(self._labeled_row("r_max (最大半径)", self.roi_r_max))
        self.roi_group.add_child(self._labeled_row("r_step (増分)", self.roi_r_step))
        self.roi_group.add_child(self._labeled_row("最小点数", self.roi_min_points))
        self.roi_group.add_child(self._labeled_row("クリックスナップ半径", self.pick_snap_radius))

        self.fit_group = gui.CollapsableVert("平面/円柱パラメータ", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.fit_group)
        self.fit_group.set_is_open(False)

        self.plane_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.plane_threshold.double_value = 0.01
        self.cylinder_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cylinder_threshold.double_value = 0.02
        self.normal_th = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.normal_th.double_value = 30.0

        self.fit_group.add_child(self._labeled_row("平面しきい値", self.plane_threshold))
        self.fit_group.add_child(self._labeled_row("円柱しきい値", self.cylinder_threshold))
        self.fit_group.add_child(self._labeled_row("法線閾値(度)", self.normal_th))

        self.cyl_preset = gui.Combobox()
        self.cyl_preset.add_item("カスタム")
        self.cyl_preset.add_item("細い柱 (φ≤10cm)")
        self.cyl_preset.add_item("中くらいの柱 (φ10-30cm)")
        self.cyl_preset.add_item("太い柱 (φ30cm-1m)")
        self.cyl_preset.add_item("大きな円柱 (φ>1m)")
        self.cyl_preset.add_item("高精度モード")
        self.cyl_preset.selected_text = "カスタム"
        self.cyl_preset.set_on_selection_changed(self._on_cyl_preset_changed)
        self.fit_group.add_child(self._labeled_row("円柱プリセット", self.cyl_preset))

        self.auto_tune_cylinder = gui.Checkbox("円柱パラメータ自動調整")
        self.auto_tune_cylinder.checked = False
        self.fit_group.add_child(self.auto_tune_cylinder)

        self.target_diameter = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_diameter.double_value = 0.0
        self.target_height = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_height.double_value = 0.0
        self.fit_group.add_child(self._labeled_row("想定直径(m)", self.target_diameter))
        self.fit_group.add_child(self._labeled_row("想定高さ(m)", self.target_height))

        self.cyl_radius_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cyl_radius_scale.double_value = 1.0
        self.fit_group.add_child(self._labeled_row("半径補正(×)", self.cyl_radius_scale))

        self.target_diameter_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_diameter_tol.double_value = 0.0
        self.target_height_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_height_tol.double_value = 0.0
        self.fit_group.add_child(self._labeled_row("直径許容(±m)", self.target_diameter_tol))
        self.fit_group.add_child(self._labeled_row("高さ許容(±m)", self.target_height_tol))

        self.expand_method = gui.Combobox()
        self.expand_method.add_item(EXPAND_METHOD_LABELS["component"])
        self.expand_method.add_item(EXPAND_METHOD_LABELS["bfs"])
        self.expand_method.selected_text = EXPAND_METHOD_LABELS["component"]
        self.fit_group.add_child(self._labeled_row("拡張方法", self.expand_method))

        self.max_expand_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.max_expand_radius.double_value = 5.0
        self.work_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.work_radius.double_value = 0.0
        self.grow_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.grow_radius.double_value = 0.15
        self.max_refine_iters = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_refine_iters.int_value = 3
        self.fit_group.add_child(self._labeled_row("最大拡張半径", self.max_expand_radius))
        self.fit_group.add_child(self._labeled_row("計算範囲半径(0=最大拡張)", self.work_radius))
        self.fit_group.add_child(self._labeled_row("成長半径", self.grow_radius))
        self.fit_group.add_child(self._labeled_row("再フィット回数", self.max_refine_iters))

        self.max_expanded_points = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_expanded_points.int_value = 200000
        self.max_frontier = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_frontier.int_value = 200000
        self.max_steps = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_steps.int_value = 1000000
        self.fit_group.add_child(self._labeled_row("最大点数", self.max_expanded_points))
        self.fit_group.add_child(self._labeled_row("フロンティア上限", self.max_frontier))
        self.fit_group.add_child(self._labeled_row("最大ステップ", self.max_steps))

        self.cyl_auto_sample_checkbox = gui.Checkbox("大規模点群は自動サンプリング")
        self.cyl_auto_sample_checkbox.checked = True
        self.cyl_sample_cap = gui.NumberEdit(gui.NumberEdit.INT)
        self.cyl_sample_cap.int_value = 200000
        self.fit_group.add_child(self.cyl_auto_sample_checkbox)
        self.fit_group.add_child(self._labeled_row("サンプル上限", self.cyl_sample_cap))

        self.cyl_fast_roi_checkbox = gui.Checkbox("ROI高速モード")
        self.cyl_fast_roi_checkbox.checked = True
        self.fit_group.add_child(self.cyl_fast_roi_checkbox)

        self.adaptive_plane_refine = gui.Checkbox("平面しきい値の自動調整")
        self.adaptive_plane_refine.checked = False
        self.fit_group.add_child(self.adaptive_plane_refine)

        self.adaptive_plane_refine_k = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_k.double_value = 3.0
        self.adaptive_plane_refine_min_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_min_scale.double_value = 0.5
        self.adaptive_plane_refine_max_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_max_scale.double_value = 2.0
        self.fit_group.add_child(self._labeled_row("適応k", self.adaptive_plane_refine_k))
        self.fit_group.add_child(self._labeled_row("最小スケール", self.adaptive_plane_refine_min_scale))
        self.fit_group.add_child(self._labeled_row("最大スケール", self.adaptive_plane_refine_max_scale))

        self.patch_shape = gui.Combobox()
        self.patch_shape.add_item(PATCH_SHAPE_LABELS["hull"])
        self.patch_shape.add_item(PATCH_SHAPE_LABELS["rect"])
        self.patch_shape.selected_text = PATCH_SHAPE_LABELS["hull"]
        self.fit_group.add_child(self._labeled_row("パッチ形状", self.patch_shape))

        self.consistency_group = gui.CollapsableVert("整合性/地面", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.consistency_group)
        self.consistency_group.set_is_open(False)

        self.use_ground_plane = gui.Checkbox("地面平面を使う")
        self.use_ground_plane.checked = False
        self.consistency_group.add_child(self.use_ground_plane)

        ground_row = gui.Horiz(4)
        self.estimate_ground_button = gui.Button("地面推定")
        self.estimate_ground_button.set_on_clicked(self._on_estimate_ground)
        self.clear_ground_button = gui.Button("地面クリア")
        self.clear_ground_button.set_on_clicked(self._on_clear_ground)
        ground_row.add_child(self.estimate_ground_button)
        ground_row.add_child(self.clear_ground_button)
        self.consistency_group.add_child(ground_row)

        self.ground_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ground_threshold.double_value = 0.02
        self.ground_ransac_n = gui.NumberEdit(gui.NumberEdit.INT)
        self.ground_ransac_n.int_value = 3
        self.ground_num_iterations = gui.NumberEdit(gui.NumberEdit.INT)
        self.ground_num_iterations.int_value = 1000
        self.ground_max_tilt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ground_max_tilt.double_value = 20.0
        self.consistency_group.add_child(self._labeled_row("地面しきい値", self.ground_threshold))
        self.consistency_group.add_child(self._labeled_row("地面RANSAC n", self.ground_ransac_n))
        self.consistency_group.add_child(self._labeled_row("地面反復回数", self.ground_num_iterations))
        self.consistency_group.add_child(self._labeled_row("地面最大傾斜角", self.ground_max_tilt))

        self.use_ceiling_plane = gui.Checkbox("天井平面を使う")
        self.use_ceiling_plane.checked = False
        self.consistency_group.add_child(self.use_ceiling_plane)

        ceiling_row = gui.Horiz(4)
        self.estimate_ceiling_button = gui.Button("天井推定")
        self.estimate_ceiling_button.set_on_clicked(self._on_estimate_ceiling)
        self.clear_ceiling_button = gui.Button("天井クリア")
        self.clear_ceiling_button.set_on_clicked(self._on_clear_ceiling)
        ceiling_row.add_child(self.estimate_ceiling_button)
        ceiling_row.add_child(self.clear_ceiling_button)
        self.consistency_group.add_child(ceiling_row)

        self.ceiling_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ceiling_threshold.double_value = 0.02
        self.ceiling_max_tilt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ceiling_max_tilt.double_value = 20.0
        self.consistency_group.add_child(self._labeled_row("天井しきい値", self.ceiling_threshold))
        self.consistency_group.add_child(self._labeled_row("天井最大傾斜角", self.ceiling_max_tilt))

        self.cyl_vertical_constraint = gui.Checkbox("円柱の垂直制約")
        self.cyl_vertical_constraint.checked = False
        self.consistency_group.add_child(self.cyl_vertical_constraint)
        self.cyl_vertical_deg = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cyl_vertical_deg.double_value = 12.0
        self.consistency_group.add_child(self._labeled_row("許容傾き(度)", self.cyl_vertical_deg))

        self.snap_axis_checkbox = gui.Checkbox("地面に垂直スナップ")
        self.snap_axis_checkbox.checked = False
        self.consistency_group.add_child(self.snap_axis_checkbox)
        self.snap_axis_deg = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.snap_axis_deg.double_value = 10.0
        self.consistency_group.add_child(self._labeled_row("スナップ角度(度)", self.snap_axis_deg))

        self.stairs_group = gui.CollapsableVert("階段パラメータ", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.stairs_group)
        self.stairs_group.set_is_open(False)

        self.max_planes = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_planes.int_value = 20
        self.min_inliers = gui.NumberEdit(gui.NumberEdit.INT)
        self.min_inliers.int_value = 50
        self.stairs_ransac_n = gui.NumberEdit(gui.NumberEdit.INT)
        self.stairs_ransac_n.int_value = 3
        self.stairs_num_iterations = gui.NumberEdit(gui.NumberEdit.INT)
        self.stairs_num_iterations.int_value = 1000
        self.max_tilt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.max_tilt.double_value = 15.0
        self.height_eps = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.height_eps.double_value = 0.03
        self.no_horizontal_filter = gui.Checkbox("水平面フィルタ無効")
        self.no_horizontal_filter.checked = False
        self.no_height_merge = gui.Checkbox("高さマージ無効")
        self.no_height_merge.checked = False

        self.stairs_group.add_child(self._labeled_row("最大平面数", self.max_planes))
        self.stairs_group.add_child(self._labeled_row("最小インライヤ数", self.min_inliers))
        self.stairs_group.add_child(self._labeled_row("RANSAC n", self.stairs_ransac_n))
        self.stairs_group.add_child(self._labeled_row("反復回数", self.stairs_num_iterations))
        self.stairs_group.add_child(self._labeled_row("最大傾斜角", self.max_tilt))
        self.stairs_group.add_child(self._labeled_row("高さ許容", self.height_eps))
        self.stairs_group.add_child(self.no_horizontal_filter)
        self.stairs_group.add_child(self.no_height_merge)

        self.output_group = gui.CollapsableVert("出力", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.output_group)
        self.output_group.set_is_open(False)

        self.output_path_edit = gui.TextEdit()
        self.output_path_edit.text_value = self.output_path
        self.output_group.add_child(self._labeled_row("出力JSON", self.output_path_edit))

        self.stairs_output_edit = gui.TextEdit()
        self.stairs_output_edit.text_value = self.stairs_output_path
        self.output_group.add_child(self._labeled_row("階段JSON", self.stairs_output_edit))

        self.export_mesh_edit = gui.TextEdit()
        self.export_mesh_edit.placeholder_text = "メッシュ出力 (PLY/OBJ)"
        self.output_group.add_child(self._labeled_row("メッシュ出力", self.export_mesh_edit))

        self.auto_save_checkbox = gui.Checkbox("結果を自動保存")
        self.auto_save_checkbox.checked = True
        self.output_group.add_child(self.auto_save_checkbox)

        self.cylinder_profile_group = gui.CollapsableVert("円柱プロファイル", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.cylinder_profile_group)
        self.cylinder_profile_group.set_is_open(False)

        self.cylinder_profile_path = gui.TextEdit()
        self.cylinder_profile_path.text_value = "cylinder_profile.json"
        self.cylinder_profile_group.add_child(self._labeled_row("保存先JSON", self.cylinder_profile_path))

        self.save_profile_button = gui.Button("プロファイル保存")
        self.save_profile_button.set_on_clicked(self._on_save_profile)
        self.cylinder_profile_group.add_child(self.save_profile_button)

        self.edit_group = gui.CollapsableVert("編集", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(self.edit_group)
        self.edit_group.set_is_open(False)

        self.edit_mode = gui.Checkbox("編集モード（Shift+クリックで削除）")
        self.edit_mode.checked = False
        self.edit_mode.set_on_checked(self._on_edit_mode_toggle)
        self.edit_group.add_child(self.edit_mode)

        self.box_select_mode = gui.Checkbox("矩形削除モード（2クリック）")
        self.box_select_mode.checked = False
        self.box_select_mode.set_on_checked(self._on_box_select_toggle)
        self.edit_group.add_child(self.box_select_mode)

        self.edit_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.edit_radius.double_value = 0.10
        self.edit_group.add_child(self._labeled_row("削除半径(m)", self.edit_radius))

        edit_row = gui.Horiz(4)
        self.edit_undo_button = gui.Button("削除を戻す")
        self.edit_undo_button.set_on_clicked(self._on_edit_undo)
        self.edit_reset_button = gui.Button("編集リセット")
        self.edit_reset_button.set_on_clicked(self._on_edit_reset)
        edit_row.add_child(self.edit_undo_button)
        edit_row.add_child(self.edit_reset_button)
        self.edit_group.add_child(edit_row)

        self._fit_group = self.fit_group
        self._stairs_group = self.stairs_group
        self._apply_ui_mode(self.ui_mode_combo.selected_text)
        self._update_mode_visibility()
        self._update_view_controls()

        if self.profile_combo.selected_text != PROFILE_CUSTOM_LABEL:
            key = _profile_key_from_display(self.profile_combo.selected_text)
            if key is not None:
                self._apply_profile(key)


    def _labeled_row(self, label: str, widget: gui.Widget) -> gui.Widget:
        row = gui.Horiz(4)
        row.add_child(gui.Label(label))
        row.add_child(widget)
        return row

    def _on_ui_mode_changed(self, text: str, _index: int):
        self._apply_ui_mode(text)

    def _apply_ui_mode(self, mode_text: str):
        basic = mode_text == "基本"
        basic_groups = [
            self.file_group,
            self.mode_group,
            self.roi_group,
            self.fit_group,
            self.output_group,
        ]
        advanced_groups = [
            self.preprocess_group,
            self.sensor_profile_group,
            self.outliner_group,
            self.consistency_group,
            self.stairs_group,
            self.cylinder_profile_group,
            self.edit_group,
        ]
        for group in basic_groups:
            if group is not None:
                group.visible = True
        for group in advanced_groups:
            if group is not None:
                group.visible = not basic
        if basic:
            for group in advanced_groups:
                if group is not None:
                    group.set_is_open(False)
        if self.roi_axis_lock_checkbox is not None:
            self.roi_axis_lock_checkbox.visible = not basic

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(360 * self.window.scaling)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        axes_size = int(160 * self.window.scaling)
        self.axes_widget.frame = gui.Rect(r.x + 8, r.y + 8, axes_size, axes_size)

    def _on_open_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "点群ファイルを選択", self.window.theme)
        dlg.add_filter(".pcd .ply", "点群ファイル (.pcd, .ply)")
        dlg.add_filter("", "すべてのファイル")
        dlg.set_on_cancel(self._on_dialog_cancel)
        dlg.set_on_done(self._on_open_done)
        self.window.show_dialog(dlg)

    def _on_dialog_cancel(self):
        self.window.close_dialog()

    def _on_open_done(self, filename):
        self.window.close_dialog()
        if filename:
            self.input_path.text_value = filename
            self._load_point_cloud(filename)

    def _on_load_clicked(self):
        path = self.input_path.text_value.strip()
        if not path:
            self._set_status("入力パスが空です。")
            return
        self._load_point_cloud(path)

    def _on_profile_changed(self, text: str, index: int):
        _ = index
        key = _profile_key_from_display(text)
        if key is None:
            return
        self._apply_profile(key)
        self._set_status("プロファイルを更新しました。前処理を適用して反映できます。")

    def _apply_profile(self, key: str):
        profile = SENSOR_PROFILES.get(key)
        if profile is None:
            return
        self.voxel_size.double_value = profile.voxel_size
        self.roi_r_min.double_value = profile.r_min
        self.roi_r_max.double_value = profile.r_max
        self.roi_r_step.double_value = profile.r_step
        self.roi_min_points.int_value = profile.min_points
        self.plane_threshold.double_value = profile.plane_distance_threshold
        self.cylinder_threshold.double_value = profile.cylinder_distance_threshold

    def _on_mode_changed(self, text: str, index: int):
        _ = text
        _ = index
        self._update_mode_visibility()

    def _on_apply_preprocess(self):
        if self._loading:
            self._set_status("読み込み中です。完了後に前処理を適用してください。")
            return
        if self.pcd_raw is None:
            self._set_status("点群を読み込んでから前処理を適用してください。")
            return
        if self.use_gpu_checkbox.checked and not self._cuda_available:
            self._set_status("CUDAが利用できないためCPUで前処理します。")
        self._apply_preprocess(reset_camera=False)
        self._set_status("前処理を適用しました。")

    def _on_reset_preprocess(self):
        if self._loading:
            self._set_status("読み込み中です。完了後にリセットしてください。")
            return
        if self.pcd_raw is None:
            self._set_status("点群を読み込んでからリセットしてください。")
            return
        self._set_current_from_raw(reset_camera=False)
        self._set_status("前処理をリセットしました。")

    def _update_mode_visibility(self):
        mode = self.mode_combo.selected_text
        self._fit_group.visible = mode in (MODE_PLANE, MODE_CYLINDER)
        self._stairs_group.visible = mode == MODE_STAIRS

    def _on_run_clicked(self):
        self._run_fit()

    def _on_clear_clicked(self):
        self._clear_results()
        self.results = {"planes": [], "cylinders": []}
        self._set_status("結果をクリアしました。")

    def _modifier_state(self, event, modifier) -> Tuple[bool, bool]:
        """Return (is_down, known) for a modifier key."""
        try:
            return bool(event.is_modifier_down(modifier)), True
        except Exception:
            pass
        mods = getattr(event, "modifiers", None)
        if mods is None:
            return False, False
        try:
            return bool(mods & modifier), True
        except Exception:
            try:
                return bool(mods & int(modifier)), True
            except Exception:
                return False, False

    def _on_mouse(self, event):
        self._last_mouse_pos = (int(event.x), int(event.y))
        shift_down, _ = self._modifier_state(event, gui.KeyModifier.SHIFT)
        if self._transform_mode is not None and self._roi is not None:
            if event.type == gui.MouseEvent.Type.MOVE or event.type == gui.MouseEvent.Type.DRAG:
                x_local, y_local, inside = self._event_to_widget_coords(event)
                if inside and self._roi_drag_start is not None:
                    self._update_roi_drag(x_local, y_local)
                return gui.Widget.EventCallbackResult.HANDLED
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                if event.is_button_down(gui.MouseButton.LEFT):
                    self._end_roi_transform(commit=True)
                    return gui.Widget.EventCallbackResult.HANDLED
                if event.is_button_down(gui.MouseButton.RIGHT):
                    self._end_roi_transform(commit=False)
                    return gui.Widget.EventCallbackResult.HANDLED
            if event.type in (gui.MouseEvent.Type.BUTTON_UP, gui.MouseEvent.Type.WHEEL):
                return gui.Widget.EventCallbackResult.HANDLED
        if self.edit_mode.checked and self.box_select_mode.checked:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                if not event.is_button_down(gui.MouseButton.LEFT):
                    return gui.Widget.EventCallbackResult.HANDLED
                x_local, y_local, inside = self._event_to_widget_coords(event)
                if not inside:
                    return gui.Widget.EventCallbackResult.HANDLED
                if self._box_select_start is None:
                    self._box_select_start = (x_local, y_local)
                    self._update_drag_rect(self._box_select_start, self._box_select_start)
                    self._set_status("矩形削除: 2点目をクリックしてください")
                else:
                    start = self._box_select_start
                    end = (x_local, y_local)
                    self._box_select_start = None
                    self._clear_drag_rect()
                    dx = abs(end[0] - start[0])
                    dy = abs(end[1] - start[1])
                    if max(dx, dy) >= 4:
                        gui.Application.instance.post_to_main_thread(
                            self.window,
                            lambda s=start, e=end: self._erase_points_in_rect(s, e)
                        )
                return gui.Widget.EventCallbackResult.HANDLED
            if event.type == gui.MouseEvent.Type.MOVE and self._box_select_start is not None:
                x_local, y_local, inside = self._event_to_widget_coords(event)
                if inside:
                    self._update_drag_rect(self._box_select_start, (x_local, y_local))
                return gui.Widget.EventCallbackResult.HANDLED
            if event.type in (gui.MouseEvent.Type.DRAG, gui.MouseEvent.Type.BUTTON_UP, gui.MouseEvent.Type.WHEEL):
                return gui.Widget.EventCallbackResult.HANDLED
        if shift_down:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                self._shift_pick_start = (int(event.x), int(event.y))
                self._shift_pick_dragging = False
                return gui.Widget.EventCallbackResult.IGNORED
            if event.type == gui.MouseEvent.Type.DRAG and self._shift_pick_start is not None:
                dx = abs(int(event.x) - self._shift_pick_start[0])
                dy = abs(int(event.y) - self._shift_pick_start[1])
                if max(dx, dy) > 4:
                    self._shift_pick_dragging = True
                return gui.Widget.EventCallbackResult.IGNORED
            if event.type == gui.MouseEvent.Type.BUTTON_UP and self._shift_pick_start is not None:
                if not self._shift_pick_dragging:
                    x0, y0 = self._shift_pick_start
                    self._handle_shift_pick_at(x0, y0)
                    self._shift_pick_start = None
                    self._shift_pick_dragging = False
                    return gui.Widget.EventCallbackResult.HANDLED
                self._shift_pick_start = None
                self._shift_pick_dragging = False
                return gui.Widget.EventCallbackResult.IGNORED
        if shift_down:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            x_local, y_local, inside = self._event_to_widget_coords(event)
            if inside and self._hit_test_roi(x_local, y_local):
                self._select_object(self._roi_name)
                return gui.Widget.EventCallbackResult.HANDLED
        if event.type in (gui.MouseEvent.Type.DRAG, gui.MouseEvent.Type.BUTTON_UP, gui.MouseEvent.Type.WHEEL):
            self._sync_axes_camera()
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event):
        if event.type == gui.KeyEvent.Type.DOWN:
            shift_down, shift_known = self._modifier_state(event, gui.KeyModifier.SHIFT)
            if event.key == gui.KeyName.A and (shift_down or not shift_known):
                kind = "cylinder" if self.roi_add_combo.selected_text == "円柱" else "box"
                self._create_roi(kind)
                if not shift_known:
                    self._set_status("Shift判定不可のため A キーでROIを追加しました。")
                return True
            if event.key == gui.KeyName.G:
                return self._begin_roi_transform("移動")
            if event.key == gui.KeyName.R:
                return self._begin_roi_transform("回転")
            if event.key == gui.KeyName.S:
                return self._begin_roi_transform("スケール")
            if event.key in (gui.KeyName.X, gui.KeyName.Y, gui.KeyName.Z):
                if self._transform_mode is not None:
                    axis = {gui.KeyName.X: "X", gui.KeyName.Y: "Y", gui.KeyName.Z: "Z"}[event.key]
                    self._transform_axis = axis
                    self._set_status(f"変換軸: {axis}")
                    return True
            if event.key in (gui.KeyName.ESCAPE, gui.KeyName.ENTER):
                if self._transform_mode is not None:
                    commit = event.key == gui.KeyName.ENTER
                    self._end_roi_transform(commit=commit)
                    return True
            if event.key == gui.KeyName.B and self.edit_mode.checked:
                self.box_select_mode.checked = not self.box_select_mode.checked
                self._update_view_controls()
                state = "ON" if self.box_select_mode.checked else "OFF"
                self._set_status(f"矩形削除モード: {state}（Bキー）")
                return True
        return False

    def _on_box_select_toggle(self, is_checked: bool):
        _ = is_checked
        if not is_checked:
            self._box_select_start = None
            self._clear_drag_rect()
        self._update_view_controls()

    def _on_edit_mode_toggle(self, is_checked: bool):
        if not is_checked:
            self.box_select_mode.checked = False
            self._clear_drag_rect()
            self._box_select_start = None
        self._update_view_controls()

    def _update_view_controls(self):
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _select_object(self, name: Optional[str]):
        if name is None:
            return
        if name not in self._outliner_names:
            return
        self._selected_outliner_index = self._outliner_names.index(name)
        self._refresh_outliner()

    def _on_roi_edit_toggle(self, is_checked: bool):
        _ = is_checked
        self._update_roi_gizmo()

    def _clear_roi(self):
        if self._roi_name is not None:
            self._remove_object(self._roi_name)
        if self._roi_gizmo_name is not None:
            try:
                self.scene_widget.scene.remove_geometry(self._roi_gizmo_name)
            except Exception:
                pass
        self._roi = None
        self._roi_name = None
        self._roi_gizmo_name = None
        self.roi_use_checkbox.checked = False
        self.roi_edit_checkbox.checked = False
        self._set_status("ROIを削除しました。")

    def _apply_roi_to_seed(self):
        if self._roi is None:
            self._set_status("ROIがありません。")
            return
        center = np.asarray(self._roi.get("center"), dtype=float)
        self._set_seed(center)
        self.roi_use_checkbox.checked = True
        self._set_status("ROI中心をseedに設定しました。")

    def _create_roi(self, kind: str):
        if self.all_points is None or len(self.all_points) == 0:
            self._set_status("点群が読み込まれていません。")
            return
        bounds = self.scene_widget.scene.bounding_box
        center = np.asarray(bounds.get_center(), dtype=float)
        extent = np.asarray(bounds.get_extent(), dtype=float)
        extent = np.where(np.isfinite(extent), extent, 1.0)
        extent = np.maximum(extent, 0.2)
        if kind == "cylinder":
            radius = 0.25 * float(max(extent[0], extent[1]))
            height = 0.6 * float(extent[2])
            if height <= 0:
                height = 1.0
            self._roi = {
                "type": "cylinder",
                "center": center,
                "rotation": np.eye(3),
                "radius": float(max(radius, 0.05)),
                "height": float(max(height, 0.1)),
            }
        else:
            size = 0.35 * extent
            self._roi = {
                "type": "box",
                "center": center,
                "rotation": np.eye(3),
                "extent": np.maximum(size, 0.1),
            }
        self.roi_use_checkbox.checked = True
        self.roi_edit_checkbox.checked = True
        self._update_roi_geometry()
        self._set_status("ROIを作成しました。ギズモで移動/回転/スケールできます。")

    def _update_object_geometry(self, name: str, geometry: o3d.geometry.Geometry):
        obj = self._objects.get(name)
        if obj is None:
            return
        obj["solid"] = geometry
        if obj.get("kind") == "mesh":
            try:
                line = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)
                color = obj.get("color")
                if color is not None:
                    line.paint_uniform_color(color)
                else:
                    line.paint_uniform_color((0.9, 0.9, 0.9))
                obj["wire"] = line
            except Exception:
                obj["wire"] = None
        self._sync_object_visibility(name)

    def _update_roi_geometry(self):
        if self._roi is None:
            return
        mesh = self._build_roi_mesh(self._roi)
        if mesh is None:
            return
        label = "ROI: 円柱" if self._roi.get("type") == "cylinder" else "ROI: 箱"
        color = np.array([0.95, 0.6, 0.15], dtype=float)
        if self._roi_name is None:
            name = self._register_object(
                label=label,
                geometry=mesh,
                kind="mesh",
                category="roi",
                color=color,
            )
            self._roi_name = name
        else:
            obj = self._objects.get(self._roi_name)
            if obj is not None:
                obj["label"] = label
                obj["color"] = color
            self._update_object_geometry(self._roi_name, mesh)
            self._refresh_outliner()
        self._update_roi_gizmo()

    def _update_roi_gizmo(self):
        if self._roi_gizmo_name is not None:
            try:
                self.scene_widget.scene.remove_geometry(self._roi_gizmo_name)
            except Exception:
                pass
            self._roi_gizmo_name = None
        if self._roi is None or not self.roi_edit_checkbox.checked:
            return
        if self._roi_name is not None:
            obj = self._objects.get(self._roi_name)
            if obj is not None and not obj.get("visible", True):
                return
        center = np.asarray(self._roi.get("center"), dtype=float)
        rotation = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
        axes = self._roi_local_axes(rotation)
        scale = 0.2
        if self._roi.get("type") == "cylinder":
            scale = max(scale, float(self._roi.get("radius", 0.2)) * 1.2)
        else:
            extent = np.asarray(self._roi.get("extent", [0.2, 0.2, 0.2]), dtype=float)
            scale = max(scale, float(np.max(extent)) * 0.6)
        pts = [
            center,
            center + axes[0] * scale,
            center,
            center + axes[1] * scale,
            center,
            center + axes[2] * scale,
        ]
        lines = [[0, 1], [2, 3], [4, 5]]
        colors = [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.5, 1.0]]
        gizmo = o3d.geometry.LineSet()
        gizmo.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
        gizmo.lines = o3d.utility.Vector2iVector(lines)
        gizmo.colors = o3d.utility.Vector3dVector(colors)
        name = "__roi_gizmo__"
        self.scene_widget.scene.add_geometry(name, gizmo, self._line_material)
        self._roi_gizmo_name = name

    def _hit_test_roi(self, x: int, y: int) -> bool:
        if self._roi is None or self._roi_name is None:
            return False
        obj = self._objects.get(self._roi_name)
        if obj is None or not obj.get("visible", True):
            return False
        geom = obj.get("solid")
        if geom is None:
            return False
        try:
            aabb = geom.get_axis_aligned_bounding_box()
            corners = np.asarray(aabb.get_box_points(), dtype=float)
        except Exception:
            return False
        proj = []
        for pt in corners:
            screen = self._project_point(pt)
            if screen is not None:
                proj.append(screen)
        if not proj:
            return False
        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 6.0
        inside = (min_x - pad) <= x <= (max_x + pad) and (min_y - pad) <= y <= (max_y + pad)
        return inside

    def _begin_roi_transform(self, mode_label: str) -> bool:
        if self._roi is None:
            self._set_status("ROIがありません。Shift+Aで作成してください。")
            return True
        if self._roi_name is not None:
            self._select_object(self._roi_name)
        self.roi_edit_checkbox.checked = True
        self._update_roi_gizmo()
        self._transform_mode = mode_label
        self._transform_axis = None
        self._transform_backup = {
            "type": self._roi.get("type"),
            "center": np.array(self._roi.get("center"), dtype=float).copy(),
            "rotation": np.array(self._roi.get("rotation", np.eye(3)), dtype=float).copy(),
        }
        if self._roi.get("type") == "cylinder":
            self._transform_backup["radius"] = float(self._roi.get("radius", 0.1))
            self._transform_backup["height"] = float(self._roi.get("height", 0.2))
        else:
            self._transform_backup["extent"] = np.array(self._roi.get("extent", [0.2, 0.2, 0.2]), dtype=float).copy()
        x, y = self._last_mouse_pos
        x_local, y_local, inside = self._coords_to_widget(x, y, self.scene_widget)
        if not inside:
            x_local = int(self.scene_widget.frame.width // 2)
            y_local = int(self.scene_widget.frame.height // 2)
        self._roi_drag_start = (x_local, y_local)
        self._roi_drag_start_center = np.asarray(self._roi.get("center"), dtype=float)
        self._roi_drag_start_rot = np.asarray(self._roi.get("rotation"), dtype=float)
        if self._roi.get("type") == "cylinder":
            self._roi_drag_start_params = (
                float(self._roi.get("radius", 0.1)),
                float(self._roi.get("height", 0.2)),
                0.0,
            )
        else:
            self._roi_drag_start_params = tuple(
                np.asarray(self._roi.get("extent", [0.2, 0.2, 0.2]), dtype=float)
            )
        self._set_status("変換中: マウス移動で操作 / 左クリック確定 / 右クリック取消 / X,Y,Zで軸")
        return True

    def _end_roi_transform(self, *, commit: bool):
        if not commit and self._transform_backup is not None and self._roi is not None:
            self._roi["type"] = self._transform_backup.get("type", self._roi.get("type"))
            self._roi["center"] = np.asarray(self._transform_backup.get("center"), dtype=float)
            self._roi["rotation"] = np.asarray(self._transform_backup.get("rotation"), dtype=float)
            if self._roi.get("type") == "cylinder":
                self._roi["radius"] = float(self._transform_backup.get("radius", self._roi.get("radius", 0.1)))
                self._roi["height"] = float(self._transform_backup.get("height", self._roi.get("height", 0.2)))
            else:
                self._roi["extent"] = np.asarray(self._transform_backup.get("extent"), dtype=float)
            self._update_roi_geometry()
        self._transform_mode = None
        self._transform_axis = None
        self._transform_backup = None
        self._roi_drag_start = None
        self._roi_drag_start_center = None
        self._roi_drag_start_rot = None
        self._roi_drag_start_params = None
        self._set_status("変換終了。")

    def _build_roi_mesh(self, roi: Dict[str, object]) -> Optional[o3d.geometry.TriangleMesh]:
        if roi is None:
            return None
        center = np.asarray(roi.get("center"), dtype=float)
        rotation = np.asarray(roi.get("rotation", np.eye(3)), dtype=float)
        try:
            if roi.get("type") == "cylinder":
                radius = float(roi.get("radius", 0.1))
                height = float(roi.get("height", 0.2))
                radius = max(radius, 0.01)
                height = max(height, 0.02)
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
            else:
                extent = np.asarray(roi.get("extent", [0.2, 0.2, 0.2]), dtype=float)
                extent = np.maximum(extent, 0.02)
                mesh = o3d.geometry.TriangleMesh.create_box(
                    width=float(extent[0]), height=float(extent[1]), depth=float(extent[2])
                )
            mesh.compute_vertex_normals()
            try:
                aabb = mesh.get_axis_aligned_bounding_box()
                mesh.translate(-aabb.get_center())
            except Exception:
                pass
            mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
            mesh.translate(center)
            return mesh
        except Exception:
            return None

    def _roi_local_axes(self, rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rot = np.asarray(rotation, dtype=float)
        if rot.shape != (3, 3):
            rot = np.eye(3)
        return rot[:, 0], rot[:, 1], rot[:, 2]

    def _rotation_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm <= 1e-8:
            return np.eye(3)
        axis = axis / norm
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        C = 1.0 - c
        return np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=float,
        )

    def _camera_forward(self) -> np.ndarray:
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
        if view.shape != (4, 4):
            return np.array([0.0, 0.0, -1.0], dtype=float)
        rot = view[:3, :3]
        forward = -rot.T[:, 2]
        norm = np.linalg.norm(forward)
        if norm <= 1e-8:
            return np.array([0.0, 0.0, -1.0], dtype=float)
        return forward / norm

    def _project_point(self, point: np.ndarray) -> Optional[Tuple[float, float, float]]:
        width = int(self.scene_widget.frame.width)
        height = int(self.scene_widget.frame.height)
        if width <= 1 or height <= 1:
            return None
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=np.float32)
        camera = self.scene_widget.scene.camera
        try:
            proj = np.asarray(camera.get_projection_matrix(width, height), dtype=np.float32)
        except TypeError:
            proj = np.asarray(camera.get_projection_matrix(), dtype=np.float32)
        if not (np.isfinite(view).all() and np.isfinite(proj).all()):
            return None
        pt = np.asarray(point, dtype=np.float32).reshape(1, 3)
        ones = np.ones((1, 1), dtype=np.float32)
        pt_h = np.hstack([pt, ones])
        clip = (pt_h @ view.T) @ proj.T
        w = clip[0, 3]
        if abs(w) <= 1e-6:
            return None
        ndc = clip[0, :3] / w
        x = (ndc[0] + 1.0) * 0.5 * width
        y = (1.0 - ndc[1]) * 0.5 * height
        depth = (ndc[2] + 1.0) * 0.5
        return float(x), float(y), float(depth)

    def _screen_drag_to_world_delta(
        self,
        start_xy: Tuple[int, int],
        end_xy: Tuple[int, int],
        depth: float,
    ) -> Optional[np.ndarray]:
        width = int(self.scene_widget.frame.width)
        height = int(self.scene_widget.frame.height)
        if width <= 1 or height <= 1:
            return None
        camera = self.scene_widget.scene.camera
        try:
            p0 = camera.unproject(
                start_xy[0], start_xy[1], float(depth), width, height
            )
            p1 = camera.unproject(
                end_xy[0], end_xy[1], float(depth), width, height
            )
        except Exception:
            return None
        return np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)

    def _update_roi_drag(self, x: int, y: int):
        if (
            self._roi is None
            or self._roi_drag_start is None
            or self._roi_drag_start_center is None
            or self._roi_drag_start_rot is None
            or self._roi_drag_start_params is None
        ):
            return
        start_x, start_y = self._roi_drag_start
        proj = self._project_point(self._roi_drag_start_center)
        if proj is None:
            return
        _, _, depth = proj
        delta = self._screen_drag_to_world_delta(
            (start_x, start_y), (x, y), depth
        )
        if delta is None:
            return
        mode = self._transform_mode if self._transform_mode is not None else self.roi_transform_combo.selected_text
        axis_key = self._transform_axis if self._transform_axis is not None else self.roi_axis_combo.selected_text
        rotation0 = np.asarray(self._roi_drag_start_rot, dtype=float)
        axes = self._roi_local_axes(rotation0)
        axis_vec = None
        if axis_key == "X":
            axis_vec = axes[0]
        elif axis_key == "Y":
            axis_vec = axes[1]
        elif axis_key == "Z":
            axis_vec = axes[2]

        if mode == "移動":
            move = delta
            if axis_vec is not None:
                move = axis_vec * float(np.dot(delta, axis_vec))
            self._roi["center"] = self._roi_drag_start_center + move
        elif mode == "回転":
            rot_axis = axis_vec if axis_vec is not None else self._camera_forward()
            dx = float(x - start_x)
            dy = float(y - start_y)
            angle = 0.005 * (dx + dy)
            rot = self._rotation_from_axis_angle(rot_axis, angle)
            self._roi["rotation"] = rot @ rotation0
        elif mode == "スケール":
            dx = float(x - start_x)
            dy = float(y - start_y)
            sign = 1.0 if (dx + dy) >= 0 else -1.0
            amount = float(np.linalg.norm(delta)) * sign
            if self._roi.get("type") == "cylinder":
                if axis_key == "Z":
                    height0 = float(self._roi_drag_start_params[1])
                    height = max(0.02, height0 + amount * 2.0)
                    self._roi["height"] = height
                else:
                    radius0 = float(self._roi_drag_start_params[0])
                    radius = max(0.01, radius0 + amount)
                    self._roi["radius"] = radius
            else:
                extent0 = np.asarray(self._roi_drag_start_params, dtype=float)
                if axis_vec is None:
                    extent = extent0 + amount * 2.0
                else:
                    extent = extent0.copy()
                    axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis_key, 0)
                    extent[axis_idx] = extent0[axis_idx] + amount * 2.0
                extent = np.maximum(extent, 0.02)
                self._roi["extent"] = extent
        self._update_roi_geometry()

    def _handle_shift_pick_at(self, x: int, y: int):
        x_local, y_local, inside = self._coords_to_widget(x, y, self.scene_widget)
        if not inside:
            return
        if self.edit_mode.checked:
            def depth_callback(depth_image):
                x = x_local
                y = y_local
                if x < 0 or y < 0 or x >= self.scene_widget.frame.width or y >= self.scene_widget.frame.height:
                    return
                depth = np.asarray(depth_image)[y, x]
                if depth == 1.0:
                    return
                world = self.scene_widget.scene.camera.unproject(
                    x,
                    y,
                    depth,
                    self.scene_widget.frame.width,
                    self.scene_widget.frame.height,
                )

                def update():
                    snapped = self._snap_to_point(np.array(world, dtype=float))
                    if snapped is None:
                        self._set_status("クリック位置の近くに点がありません。ズームして再クリックしてください。")
                        return
                    self._erase_points_at(snapped)

                gui.Application.instance.post_to_main_thread(self.window, update)

            self.scene_widget.scene.scene.render_to_depth_image(depth_callback)
            return

        def depth_callback(depth_image):
            x = x_local
            y = y_local
            if x < 0 or y < 0 or x >= self.scene_widget.frame.width or y >= self.scene_widget.frame.height:
                return
            depth = np.asarray(depth_image)[y, x]
            if depth == 1.0:
                return
            world = self.scene_widget.scene.camera.unproject(
                x,
                y,
                depth,
                self.scene_widget.frame.width,
                self.scene_widget.frame.height,
            )

            def update():
                snapped = self._snap_to_point(np.array(world, dtype=float))
                if snapped is None:
                    self._set_status("クリック位置の近くに点がありません。ズームして再クリックしてください。")
                    return
                self._set_seed(snapped)
                return

            gui.Application.instance.post_to_main_thread(self.window, update)

        self.scene_widget.scene.scene.render_to_depth_image(depth_callback)

    def _set_status(self, text: str):
        self.status.text = text

    def _sync_gpu_mode(self) -> bool:
        enabled = bool(self.use_gpu_checkbox.checked) and bool(self._cuda_available)
        set_gpu_enabled(enabled)
        return enabled

    def _segment_plane(self, distance_threshold: float, ransac_n: int, num_iterations: int):
        use_gpu = self._sync_gpu_mode()
        if use_gpu and self.pcd is not None:
            try:
                device = o3d.core.Device("CUDA:0")
                if self._pcd_t is not None and self._pcd_t.device == device:
                    pcd_t = self._pcd_t
                else:
                    pcd_t = o3d.t.geometry.PointCloud.from_legacy(self.pcd, device=device)
                plane_model, inliers = pcd_t.segment_plane(
                    distance_threshold=float(distance_threshold),
                    ransac_n=int(ransac_n),
                    num_iterations=int(num_iterations),
                )
                return plane_model.cpu().numpy(), inliers.cpu().numpy().astype(int)
            except Exception:
                pass
        if self.pcd is None:
            return None, None
        try:
            plane_model, inliers = self.pcd.segment_plane(
                distance_threshold=float(distance_threshold),
                ransac_n=int(ransac_n),
                num_iterations=int(num_iterations),
            )
            return plane_model, np.asarray(inliers, dtype=int)
        except Exception:
            return None, None

    def _check_cuda_available(self) -> bool:
        try:
            return bool(o3d.core.cuda.is_available() and o3d.core.cuda.device_count() > 0)
        except Exception:
            return False

    def _load_point_cloud(self, path: str):
        if self._loading:
            self._set_status("読み込み中です。完了後に再試行してください。")
            return
        self._loading = True
        self._last_loaded_path = path
        if self.use_gpu_checkbox.checked and not self._cuda_available:
            self._set_status("点群を読み込み中です…（CUDA無効のためCPU）")
        else:
            self._set_status("点群を読み込み中です…")

        load_normals = bool(self.load_normals_checkbox.checked)
        try:
            file_size = os.path.getsize(path)
        except Exception:
            file_size = 0
        if file_size > 300 * 1024 * 1024 and load_normals:
            load_normals = False
            self._set_status("巨大ファイルのため読み込み時法線推定をスキップします。")
        use_gpu = bool(self.use_gpu_checkbox.checked) and self._cuda_available

        def worker():
            try:
                pcd_raw = load_point_cloud(path)
            except Exception as exc:
                msg = f"読み込み失敗: {exc}"
                def fail(msg=msg):
                    self._loading = False
                    self._set_status(msg)
                gui.Application.instance.post_to_main_thread(self.window, fail)
                return

            pcd = pcd_raw
            pcd_t = None
            if load_normals and not pcd.has_normals():
                try:
                    if use_gpu:
                        device = o3d.core.Device("CUDA:0")
                        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd, device=device)
                        pcd_t.estimate_normals(max_nn=30, radius=0.05)
                        pcd_t.orient_normals_consistent_tangent_plane(k=15)
                        pcd = pcd_t.to_legacy()
                    else:
                        pcd.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                        )
                        pcd.orient_normals_consistent_tangent_plane(k=15)
                except Exception as exc:
                    msg = f"法線推定に失敗: {exc}"
                    def fail_normals(msg=msg):
                        self._loading = False
                        self._set_status(msg)
                    gui.Application.instance.post_to_main_thread(self.window, fail_normals)
                    return

            def done():
                self._loading = False
                self.pcd_raw = pcd_raw if not self.drop_raw_checkbox.checked else None
                self._pcd_t = pcd_t
                self._set_current_from_pcd(pcd, reset_camera=True)
                self._set_status(
                    f"{len(self.all_points)}点を読み込みました。前処理は「前処理を適用」で反映されます。"
                )

            gui.Application.instance.post_to_main_thread(self.window, done)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_preprocess(self, *, reset_camera: bool = False):
        if self.pcd_raw is None:
            if self._last_loaded_path:
                try:
                    self.pcd_raw = load_point_cloud(self._last_loaded_path)
                except Exception as exc:
                    self._set_status(f"再読み込み失敗: {exc}")
                    return
            else:
                return
        use_gpu = bool(self.use_gpu_checkbox.checked) and self._cuda_available
        if self.preprocess_checkbox.checked:
            try:
                if use_gpu:
                    device = o3d.core.Device("CUDA:0")
                    pcd_t = o3d.t.geometry.PointCloud.from_legacy(self.pcd_raw, device=device)
                    if self.voxel_size.double_value > 0:
                        pcd_t = pcd_t.voxel_down_sample(self.voxel_size.double_value)
                    pcd_t, _ = pcd_t.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
                    pcd_t.estimate_normals(max_nn=30, radius=0.05)
                    pcd_t.orient_normals_consistent_tangent_plane(k=15)
                    self.pcd = pcd_t.to_legacy()
                    self._pcd_t = pcd_t
                else:
                    self.pcd = preprocess_point_cloud(self.pcd_raw, self.voxel_size.double_value)
                    self._pcd_t = None
            except Exception as exc:
                self._set_status(f"前処理失敗: {exc}")
                return
        else:
            self.pcd = o3d.geometry.PointCloud(self.pcd_raw)
            if not self.pcd.has_normals():
                self.pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                )
                self.pcd.orient_normals_consistent_tangent_plane(k=15)
            self._pcd_t = None
        self._set_current_from_pcd(self.pcd, reset_camera=reset_camera)
        if self.drop_raw_checkbox.checked:
            self.pcd_raw = None

    def _set_current_from_raw(self, *, reset_camera: bool = False):
        if self.pcd_raw is None:
            if self._last_loaded_path:
                try:
                    self.pcd_raw = load_point_cloud(self._last_loaded_path)
                except Exception as exc:
                    self._set_status(f"再読み込み失敗: {exc}")
                    return
            else:
                return
        self.pcd = o3d.geometry.PointCloud(self.pcd_raw)
        if not self.pcd.has_normals():
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
            self.pcd.orient_normals_consistent_tangent_plane(k=15)
        self._set_current_from_pcd(self.pcd, reset_camera=reset_camera)
        if self.drop_raw_checkbox.checked:
            self.pcd_raw = None

    def _set_current_from_pcd(self, pcd: o3d.geometry.PointCloud, *, reset_camera: bool = False):
        self.pcd = pcd
        self._finalize_point_cloud(reset_camera=reset_camera)
        self._clear_results()
        self._clear_ground_plane()
        self._clear_ceiling_plane()

    def _finalize_point_cloud(self, *, reset_camera: bool = False):
        self.all_points = np.asarray(self.pcd.points)
        self.all_normals = np.asarray(self.pcd.normals) if self.pcd.has_normals() else None
        # Avoid extra copies for large point clouds.
        self._all_points_original = self.all_points
        self._all_normals_original = self.all_normals
        self._edit_mask = np.ones(len(self._all_points_original), dtype=bool)
        self._edit_history = []

        self._reset_scene_objects()
        self._update_point_cloud_view(reset_camera=reset_camera)

        if reset_camera:
            bounds = self.scene_widget.scene.bounding_box
            center = bounds.get_center()
            self.scene_widget.setup_camera(60, bounds, center)
            self.scene_widget.look_at(center, center - [0, 0, 3], [0, -1, 0])

    def _set_seed(self, center: np.ndarray):
        self.last_pick = np.asarray(center, dtype=float)
        if self._seed_name is not None:
            self.scene_widget.scene.remove_geometry(self._seed_name)
        snap_radius = float(self.pick_snap_radius.double_value)
        radius = max(0.03, min(0.2, snap_radius if snap_radius > 0 else 0.05))
        seed = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        seed.paint_uniform_color([1.0, 0.8, 0.1])
        seed.translate(self.last_pick)
        seed.compute_vertex_normals()
        self._seed_name = "seed_marker"
        self.scene_widget.scene.add_geometry(self._seed_name, seed, self._mesh_material)
        self._set_status(f"シード選択: {np.round(self.last_pick, 4).tolist()}")

    def _update_point_cloud_view(self, *, reset_camera: bool = False):
        if self._all_points_original is None:
            return
        mask = self._edit_mask if self._edit_mask is not None else np.ones(len(self._all_points_original), dtype=bool)
        indices = np.flatnonzero(mask)
        self._current_indices = indices
        pts = self._all_points_original[indices]
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pts)
        if self._all_normals_original is not None and len(self._all_normals_original) == len(self._all_points_original):
            norms = self._all_normals_original[indices]
            self.pcd.normals = o3d.utility.Vector3dVector(norms)
        self.all_points = np.asarray(self.pcd.points)
        self.all_normals = np.asarray(self.pcd.normals) if self.pcd.has_normals() else None

        try:
            self._kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        except Exception:
            self._kdtree = None

        if self._pointcloud_name is not None:
            self._remove_object(self._pointcloud_name)
        self._pointcloud_name = self._register_object(
            label="点群",
            geometry=self.pcd,
            kind="pcd",
            category="base",
        )

        if reset_camera:
            bounds = self.scene_widget.scene.bounding_box
            center = bounds.get_center()
            self.scene_widget.setup_camera(60, bounds, center)
            self.scene_widget.look_at(center, center - [0, 0, 3], [0, -1, 0])
            self._sync_axes_camera()
        self._ensure_grid()

    def _erase_points_at(self, center: np.ndarray):
        if self._all_points_original is None or self._edit_mask is None:
            return
        radius = float(self.edit_radius.double_value)
        if radius <= 0:
            return
        diff = self._all_points_original - center
        dist2 = np.einsum("ij,ij->i", diff, diff)
        to_remove = np.where(self._edit_mask & (dist2 <= radius * radius))[0]
        if len(to_remove) == 0:
            self._set_status("削除対象が見つかりませんでした。")
            return
        self._edit_mask[to_remove] = False
        self._edit_history.append(to_remove)
        self._update_point_cloud_view()
        self._set_status(f"{len(to_remove)}点を削除しました（元データは保持）")

    def _erase_points_in_rect(self, start: Tuple[int, int], end: Tuple[int, int]):
        if self._all_points_original is None or self._edit_mask is None or self.all_points is None:
            return
        if self._current_indices is None:
            return
        x0, y0 = start
        x1, y1 = end
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        if x_max < 0 or y_max < 0 or x_min >= self.scene_widget.frame.width or y_min >= self.scene_widget.frame.height:
            self._set_status("選択範囲がビュー外です。")
            return

        x_min = max(0, min(self.scene_widget.frame.width - 1, x_min))
        x_max = max(0, min(self.scene_widget.frame.width - 1, x_max))
        y_min = max(0, min(self.scene_widget.frame.height - 1, y_min))
        y_max = max(0, min(self.scene_widget.frame.height - 1, y_max))
        if x_max <= x_min or y_max <= y_min:
            return

        coords, valid = self._project_points_to_screen(self.all_points)
        if coords is None:
            return
        xs = coords[:, 0]
        ys = coords[:, 1]
        in_rect = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        idx_local = np.where(valid & in_rect)[0]
        if len(idx_local) == 0:
            self._set_status("選択範囲に点がありません。")
            return
        idx_global = self._current_indices[idx_local]
        self._edit_mask[idx_global] = False
        self._edit_history.append(idx_global)
        self._update_point_cloud_view()
        self._set_status(f"{len(idx_global)}点を削除しました（元データは保持）")

    def _update_drag_rect(self, start: Tuple[int, int], end: Tuple[int, int]):
        if start is None or end is None:
            return
        width = int(self.scene_widget.frame.width)
        height = int(self.scene_widget.frame.height)
        if width <= 1 or height <= 1:
            return
        x_min = max(0, min(width - 1, min(start[0], end[0])))
        x_max = max(0, min(width - 1, max(start[0], end[0])))
        y_min = max(0, min(height - 1, min(start[1], end[1])))
        y_max = max(0, min(height - 1, max(start[1], end[1])))
        if x_max <= x_min or y_max <= y_min:
            return
        depth = 0.02
        camera = self.scene_widget.scene.camera
        try:
            c0 = camera.unproject(x_min, y_min, depth, width, height)
            c1 = camera.unproject(x_max, y_min, depth, width, height)
            c2 = camera.unproject(x_max, y_max, depth, width, height)
            c3 = camera.unproject(x_min, y_max, depth, width, height)
        except Exception:
            return
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([c0, c1, c2, c3])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.6, 0.2]] * 4)
        name = "__drag_rect__"
        if self.scene_widget.scene.has_geometry(name):
            self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, line_set, self._drag_rect_material)
        self._drag_rect_name = name
        self.scene_widget.force_redraw()

    def _clear_drag_rect(self):
        if self._drag_rect_name and self.scene_widget.scene.has_geometry(self._drag_rect_name):
            self.scene_widget.scene.remove_geometry(self._drag_rect_name)
        self._drag_rect_name = None

    def _coords_to_widget(self, x: int, y: int, widget: Optional[gui.Widget] = None) -> Tuple[int, int, bool]:
        if widget is None:
            widget = self.scene_widget
        width = int(widget.frame.width)
        height = int(widget.frame.height)
        if 0 <= x < width and 0 <= y < height:
            return x, y, True
        x_local = x - int(widget.frame.x)
        y_local = y - int(widget.frame.y)
        inside = 0 <= x_local < width and 0 <= y_local < height
        return x_local, y_local, inside

    def _event_to_widget_coords(self, event, widget: Optional[gui.Widget] = None) -> Tuple[int, int, bool]:
        return self._coords_to_widget(int(event.x), int(event.y), widget)

    def _project_points_to_screen(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if points is None or len(points) == 0:
            return None, None
        width = int(self.scene_widget.frame.width)
        height = int(self.scene_widget.frame.height)
        if width <= 1 or height <= 1:
            return None, None
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=np.float32)
        camera = self.scene_widget.scene.camera
        try:
            proj = np.asarray(camera.get_projection_matrix(width, height), dtype=np.float32)
        except TypeError:
            proj = np.asarray(camera.get_projection_matrix(), dtype=np.float32)
        if not (np.isfinite(view).all() and np.isfinite(proj).all()):
            return None, None
        pts = np.asarray(points, dtype=np.float32)
        n = len(pts)
        coords = np.empty((n, 2), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        batch = 200000
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = pts[start:end]
            ones = np.ones((len(chunk), 1), dtype=np.float32)
            pts_h = np.hstack([chunk, ones])
            clip = (pts_h @ view.T) @ proj.T
            w = clip[:, 3]
            v = w > 1e-6
            ndc = np.zeros((len(chunk), 3), dtype=np.float32)
            if np.any(v):
                ndc[v] = clip[v, :3] / w[v, None]
            z_ok = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            v &= z_ok
            xs = (ndc[:, 0] + 1.0) * 0.5 * width
            ys = (1.0 - ndc[:, 1]) * 0.5 * height
            coords[start:end, 0] = xs
            coords[start:end, 1] = ys
            valid[start:end] = v
        return coords, valid

    def _ensure_grid(self):
        if self.pcd is None or self.all_points is None or len(self.all_points) == 0:
            return
        if self._grid_name is not None:
            return
        bounds = self.scene_widget.scene.bounding_box
        min_bound = np.asarray(bounds.min_bound)
        max_bound = np.asarray(bounds.max_bound)
        extent = max(max_bound[0] - min_bound[0], max_bound[1] - min_bound[1])
        spacing = 1.0
        if extent < 5.0:
            spacing = 0.5
        if extent < 2.0:
            spacing = 0.2
        z = float(min_bound[2])
        lines = self._create_grid_lines(min_bound[0], max_bound[0], min_bound[1], max_bound[1], z, spacing)
        self._grid_name = self._register_object(
            label="グリッド",
            geometry=lines,
            kind="line",
            category="base",
            color=np.array([0.3, 0.3, 0.3]),
        )

    def _create_grid_lines(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z: float,
        spacing: float,
    ) -> o3d.geometry.LineSet:
        x0 = np.floor(x_min / spacing) * spacing
        x1 = np.ceil(x_max / spacing) * spacing
        y0 = np.floor(y_min / spacing) * spacing
        y1 = np.ceil(y_max / spacing) * spacing
        xs = np.arange(x0, x1 + spacing * 0.5, spacing)
        ys = np.arange(y0, y1 + spacing * 0.5, spacing)
        points = []
        lines = []
        idx = 0
        for x in xs:
            points.append([x, y0, z])
            points.append([x, y1, z])
            lines.append([idx, idx + 1])
            idx += 2
        for y in ys:
            points.append([x0, y, z])
            points.append([x1, y, z])
            lines.append([idx, idx + 1])
            idx += 2
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
        return ls

    def _sync_axes_camera(self):
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
        if view.shape != (4, 4):
            return
        if not np.isfinite(view).all():
            return
        R = view[:3, :3]
        forward = -R[2]
        up = R[1]
        if not (np.isfinite(forward).all() and np.isfinite(up).all()):
            return
        if np.linalg.norm(forward) < 1e-6 or np.linalg.norm(up) < 1e-6:
            return
        eye = forward * 2.0
        try:
            self.axes_widget.scene.camera.look_at(np.array([0.0, 0.0, 0.0]), eye, up)
        except Exception:
            return

    def _on_edit_undo(self):
        if not self._edit_history or self._edit_mask is None:
            self._set_status("戻す操作がありません。")
            return
        last = self._edit_history.pop()
        self._edit_mask[last] = True
        self._update_point_cloud_view()
        self._set_status("削除を戻しました。")

    def _on_edit_reset(self):
        if self._all_points_original is None:
            return
        self._edit_mask = np.ones(len(self._all_points_original), dtype=bool)
        self._edit_history = []
        self._update_point_cloud_view()
        self._set_status("編集をリセットしました。")

    def _snap_to_point(self, world: np.ndarray) -> Optional[np.ndarray]:
        if self._kdtree is None or self.all_points is None:
            return world
        try:
            _, idx, dist2 = self._kdtree.search_knn_vector_3d(world, 1)
            if not idx:
                return None
            snap_radius = float(self.pick_snap_radius.double_value)
            if snap_radius > 0 and dist2[0] > snap_radius * snap_radius:
                return None
            return self.all_points[int(idx[0])]
        except Exception:
            return world

    def _clear_results(self):
        to_remove = [
            name for name, obj in self._objects.items()
            if obj.get("category") == "result"
        ]
        for name in to_remove:
            self._remove_object(name)
        self._result_names = []
        self._result_meshes = []

    def _reset_scene_objects(self):
        self.scene_widget.scene.clear_geometry()
        self._clear_selection_highlight()
        self._outliner_names = []
        self.outliner.set_items([])
        self._selected_outliner_index = -1
        self._objects = {}
        self._object_counter = 0
        self._pointcloud_name = None
        self._roi = None
        self._roi_name = None
        self._roi_gizmo_name = None

    def _outliner_label(self, name: str) -> str:
        obj = self._objects.get(name, {})
        label = str(obj.get("label", name))
        if not obj.get("visible", True):
            prefix = "[H]"
        else:
            mode = obj.get("mode", "solid")
            prefix = "[W]" if mode == "wire" else "[S]"
        return f"{prefix} {label}"

    def _refresh_outliner(self):
        items = [self._outliner_label(n) for n in self._outliner_names]
        self.outliner.set_items(items)
        if 0 <= self._selected_outliner_index < len(items):
            self.outliner.selected_index = self._selected_outliner_index
        self._update_selection_highlight()

    def _register_object(
        self,
        *,
        label: str,
        geometry: o3d.geometry.Geometry,
        kind: str,
        category: str,
        color: Optional[np.ndarray] = None,
        plane_param: Optional[PlaneParam] = None,
    ) -> str:
        self._object_counter += 1
        base_name = f"{category}_{self._object_counter}"
        solid_name = base_name
        wire_name = f"{base_name}_wire"

        obj: Dict[str, object] = {
            "label": label,
            "kind": kind,
            "category": category,
            "solid_name": solid_name,
            "wire_name": wire_name,
            "solid": geometry,
            "wire": None,
            "mode": "solid",
            "visible": True,
        }
        if plane_param is not None:
            obj["plane_param"] = plane_param

        if kind == "pcd":
            self.scene_widget.scene.add_geometry(solid_name, geometry, self._pcd_material)
        elif kind == "line":
            if color is not None:
                try:
                    geometry.colors = o3d.utility.Vector3dVector(
                        np.tile(color, (len(geometry.lines), 1))
                    )
                except Exception:
                    pass
            self.scene_widget.scene.add_geometry(solid_name, geometry, self._line_material)
        else:
            if color is not None:
                try:
                    geometry.paint_uniform_color(color)
                except Exception:
                    pass
            try:
                if hasattr(geometry, "has_vertex_normals") and not geometry.has_vertex_normals():
                    geometry.compute_vertex_normals()
            except Exception:
                pass
            self.scene_widget.scene.add_geometry(solid_name, geometry, self._mesh_material)
            try:
                line = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)
                if color is not None:
                    line.paint_uniform_color(color)
                else:
                    line.paint_uniform_color((0.9, 0.9, 0.9))
                obj["wire"] = line
            except Exception:
                obj["wire"] = None

        self._objects[base_name] = obj
        self._outliner_names.append(base_name)
        self._refresh_outliner()
        return base_name

    def _remove_object(self, name: str):
        obj = self._objects.get(name)
        if obj is None:
            return
        if self._selection_highlight_target == name:
            self._clear_selection_highlight()
        solid_name = obj.get("solid_name")
        wire_name = obj.get("wire_name")
        if solid_name:
            self.scene_widget.scene.remove_geometry(str(solid_name))
        if wire_name:
            self.scene_widget.scene.remove_geometry(str(wire_name))
        self._objects.pop(name, None)
        if name in self._outliner_names:
            self._outliner_names.remove(name)
            self._refresh_outliner()

    def _sync_object_visibility(self, name: str):
        obj = self._objects.get(name)
        if obj is None:
            return
        solid_name = str(obj.get("solid_name"))
        wire_name = str(obj.get("wire_name"))
        self.scene_widget.scene.remove_geometry(solid_name)
        if wire_name:
            self.scene_widget.scene.remove_geometry(wire_name)
        if not obj.get("visible", True):
            return
        mode = obj.get("mode", "solid")
        kind = obj.get("kind")
        if mode == "wire" and kind == "mesh" and obj.get("wire") is not None:
            self.scene_widget.scene.add_geometry(wire_name, obj["wire"], self._line_material)
        else:
            if kind == "line":
                mat = self._line_material
            else:
                mat = self._mesh_material if kind == "mesh" else self._pcd_material
            self.scene_widget.scene.add_geometry(solid_name, obj["solid"], mat)
        self._update_selection_highlight()

    def _on_toggle_visibility(self):
        idx = int(self.outliner.selected_index)
        if idx < 0 or idx >= len(self._outliner_names):
            return
        name = self._outliner_names[idx]
        if name is None:
            return
        obj = self._objects.get(name)
        if obj is None:
            return
        obj["visible"] = not bool(obj.get("visible", True))
        self._sync_object_visibility(name)
        self._refresh_outliner()
        if name == self._roi_name:
            self._update_roi_gizmo()

    def _on_set_display_mode(self, mode: str):
        idx = int(self.outliner.selected_index)
        if idx < 0 or idx >= len(self._outliner_names):
            return
        name = self._outliner_names[idx]
        if name is None:
            return
        obj = self._objects.get(name)
        if obj is None:
            return
        if mode == "wire" and obj.get("kind") != "mesh":
            return
        obj["mode"] = mode
        obj["visible"] = True if mode != "hidden" else False
        self._sync_object_visibility(name)
        self._refresh_outliner()
        if name == self._roi_name:
            self._update_roi_gizmo()

    def _on_apply_color(self):
        idx = int(self.outliner.selected_index)
        if idx < 0 or idx >= len(self._outliner_names):
            return
        name = self._outliner_names[idx]
        obj = self._objects.get(name)
        if obj is None or obj.get("kind") != "mesh":
            return
        color = self.outliner_color.color_value
        rgb = np.array([color.red, color.green, color.blue], dtype=float)
        mesh = obj.get("solid")
        if mesh is not None:
            try:
                mesh.paint_uniform_color(rgb)
            except Exception:
                pass
        wire = obj.get("wire")
        if wire is not None:
            try:
                wire.colors = o3d.utility.Vector3dVector(
                    np.tile(rgb, (len(wire.lines), 1))
                )
            except Exception:
                pass
        elif obj.get("kind") == "line":
            try:
                mesh.colors = o3d.utility.Vector3dVector(
                    np.tile(rgb, (len(mesh.lines), 1))
                )
            except Exception:
                pass
        obj["color"] = rgb
        self._sync_object_visibility(name)

    def _on_delete_selected(self):
        idx = int(self.outliner.selected_index)
        if idx < 0 or idx >= len(self._outliner_names):
            self._set_status("アウトライナーで対象を選択してください。")
            return
        name = self._outliner_names[idx]
        obj = self._objects.get(name)
        if obj is None:
            return
        if obj.get("category") == "base":
            self._set_status("点群/グリッドは削除できません。")
            return
        if name == self._roi_name:
            self._clear_roi()
            return
        if name == self._ground_name:
            self._clear_ground_plane()
            self._set_status("地面を削除しました。")
            return
        if name == self._ceiling_name:
            self._clear_ceiling_plane()
            self._set_status("天井を削除しました。")
            return
        if name in self._result_names:
            idx_res = self._result_names.index(name)
            self._result_names.pop(idx_res)
            if idx_res < len(self._result_meshes):
                self._result_meshes.pop(idx_res)
        self._remove_object(name)
        self._set_status("選択オブジェクトを削除しました。")

    def _on_assign_plane(self, kind: str):
        idx = int(self.outliner.selected_index)
        if idx < 0 or idx >= len(self._outliner_names):
            self._set_status("アウトライナーで対象メッシュを選択してください。")
            return
        name = self._outliner_names[idx]
        obj = self._objects.get(name)
        if obj is None:
            self._set_status("選択オブジェクトが無効です。")
            return
        plane = obj.get("plane_param")
        if plane is None:
            self._set_status("選択オブジェクトは平面情報がありません。")
            return
        if kind == "ground":
            self.ground_plane = plane
            self._ground_name = None
            self.use_ground_plane.checked = True
            self._set_status(f"地面平面に設定: {obj.get('label', name)}")
        elif kind == "ceiling":
            self.ceiling_plane = plane
            self._ceiling_name = None
            self.use_ceiling_plane.checked = True
            self._set_status(f"天井平面に設定: {obj.get('label', name)}")

    def _on_outliner_select(self, _value, _is_dbl_click=False):
        try:
            self._selected_outliner_index = int(self.outliner.selected_index)
        except Exception:
            self._selected_outliner_index = -1
        self._update_selection_highlight()

    def _clear_selection_highlight(self):
        if self._selection_highlight_name is not None:
            try:
                self.scene_widget.scene.remove_geometry(self._selection_highlight_name)
            except Exception:
                pass
        self._selection_highlight_name = None
        self._selection_highlight_target = None

    def _update_selection_highlight(self):
        if self._selection_highlight_name is not None:
            try:
                self.scene_widget.scene.remove_geometry(self._selection_highlight_name)
            except Exception:
                pass
        self._selection_highlight_name = None
        self._selection_highlight_target = None

        idx = int(self.outliner.selected_index) if self.outliner is not None else -1
        if idx < 0 or idx >= len(self._outliner_names):
            return
        name = self._outliner_names[idx]
        obj = self._objects.get(name)
        if obj is None:
            return
        if not obj.get("visible", True):
            return
        if obj.get("category") == "base":
            return
        geom = obj.get("solid")
        if geom is None:
            return
        try:
            aabb = geom.get_axis_aligned_bounding_box()
        except Exception:
            return
        try:
            extent = np.asarray(aabb.get_extent(), dtype=float)
        except Exception:
            return
        max_extent = float(np.max(extent)) if extent.size else 0.0
        if not np.isfinite(max_extent) or max_extent <= 0:
            return
        pad = 0.02 * max_extent
        try:
            scale = (max_extent + 2.0 * pad) / max_extent
            aabb = aabb.scale(scale, aabb.get_center())
        except Exception:
            pass
        try:
            highlight = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        except Exception:
            return
        try:
            color = np.array([1.0, 0.85, 0.2], dtype=float)
            highlight.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(highlight.lines), 1))
            )
        except Exception:
            pass
        highlight_name = f"__highlight_{name}"
        try:
            self.scene_widget.scene.add_geometry(highlight_name, highlight, self._highlight_material)
        except Exception:
            return
        self._selection_highlight_name = highlight_name
        self._selection_highlight_target = name

    def _on_save_profile(self):
        path = self.cylinder_profile_path.text_value.strip()
        if not path:
            self._set_status("保存先が空です。")
            return
        profile = {
            "target": {
                "diameter": float(self.target_diameter.double_value),
                "height": float(self.target_height.double_value),
                "diameter_tol": float(self.target_diameter_tol.double_value),
                "height_tol": float(self.target_height_tol.double_value),
            },
            "roi": {
                "r_min": float(self.roi_r_min.double_value),
                "r_max": float(self.roi_r_max.double_value),
                "r_step": float(self.roi_r_step.double_value),
                "min_points": int(self.roi_min_points.int_value),
            },
            "cylinder": {
                "threshold": float(self.cylinder_threshold.double_value),
                "normal_deg": float(self.normal_th.double_value),
                "grow_radius": float(self.grow_radius.double_value),
                "max_expand_radius": float(self.max_expand_radius.double_value),
                "max_refine_iters": int(self.max_refine_iters.int_value),
                "max_expanded_points": int(self.max_expanded_points.int_value),
                "max_frontier": int(self.max_frontier.int_value),
                "max_steps": int(self.max_steps.int_value),
            },
        }
        try:
            with open(path, "w") as f:
                json.dump(profile, f, indent=2)
        except Exception as exc:
            self._set_status(f"保存失敗: {exc}")
            return
        self._set_status(f"プロファイル保存: {path}")

    def _on_cyl_preset_changed(self, text: str, _idx: int) -> None:
        """Apply cylinder fitting preset parameters."""
        presets = {
            "細い柱 (φ≤10cm)": {
                "cylinder_threshold": 0.01,
                "normal_th": 25.0,
                "roi_r_min": 0.05,
                "roi_r_max": 0.2,
                "roi_r_step": 0.02,
                "grow_radius": 0.08,
                "max_expand_radius": 2.0,
            },
            "中くらいの柱 (φ10-30cm)": {
                "cylinder_threshold": 0.02,
                "normal_th": 30.0,
                "roi_r_min": 0.1,
                "roi_r_max": 0.5,
                "roi_r_step": 0.05,
                "grow_radius": 0.15,
                "max_expand_radius": 4.0,
            },
            "太い柱 (φ30cm-1m)": {
                "cylinder_threshold": 0.03,
                "normal_th": 35.0,
                "roi_r_min": 0.3,
                "roi_r_max": 1.2,
                "roi_r_step": 0.1,
                "grow_radius": 0.25,
                "max_expand_radius": 6.0,
            },
            "大きな円柱 (φ>1m)": {
                "cylinder_threshold": 0.05,
                "normal_th": 40.0,
                "roi_r_min": 1.0,
                "roi_r_max": 3.0,
                "roi_r_step": 0.2,
                "grow_radius": 0.4,
                "max_expand_radius": 10.0,
            },
            "高精度モード": {
                "cylinder_threshold": 0.008,
                "normal_th": 20.0,
                "roi_r_min": 0.15,
                "roi_r_max": 0.8,
                "roi_r_step": 0.05,
                "grow_radius": 0.1,
                "max_expand_radius": 3.0,
            },
        }
        preset = presets.get(text)
        if preset is None:
            return
        self.cylinder_threshold.double_value = preset["cylinder_threshold"]
        self.normal_th.double_value = preset["normal_th"]
        self.roi_r_min.double_value = preset["roi_r_min"]
        self.roi_r_max.double_value = preset["roi_r_max"]
        self.roi_r_step.double_value = preset["roi_r_step"]
        self.grow_radius.double_value = preset["grow_radius"]
        self.max_expand_radius.double_value = preset["max_expand_radius"]
        self._set_status(f"プリセット '{text}' を適用しました。")

    def _clear_ground_plane(self):
        if self._ground_name is not None:
            self._remove_object(self._ground_name)
        self._ground_name = None
        self.ground_plane = None

    def _clear_ceiling_plane(self):
        if self._ceiling_name is not None:
            self._remove_object(self._ceiling_name)
        self._ceiling_name = None
        self.ceiling_plane = None

    def _run_fit(self):
        if self.pcd is None or self.all_points is None:
            self._set_status("点群が読み込まれていません。")
            return
        if self.last_pick is None:
            if self.roi_use_checkbox.checked and self._roi is not None:
                seed_center = np.asarray(self._roi.get("center"), dtype=float)
                self._set_seed(seed_center)
            else:
                self._set_status("先にShift+クリックでseed/ROIを指定してください。")
                return
        self._sync_gpu_mode()

        seed_center = self.last_pick
        mode = self.mode_combo.selected_text
        if mode == MODE_STAIRS:
            seed_indices, _ = self._compute_seed_indices(seed_center)
            if len(seed_indices) == 0:
                self._set_status("ROIが空です。r_min/r_maxを調整するか再クリックしてください。")
                return
            self._run_stairs(seed_center, seed_indices)
            return

        max_expand_radius = float(self.max_expand_radius.double_value)
        work_radius = float(self.work_radius.double_value)
        if not np.isfinite(work_radius) or work_radius <= 0:
            work_radius = max_expand_radius
        work_points, work_normals = self._get_working_cloud(seed_center, work_radius)
        seed_indices, seed_radius = self._compute_seed_indices_for_points(work_points, seed_center)
        if len(seed_indices) == 0:
            self._set_status("ROIが空です。r_min/r_maxを調整するか再クリックしてください。")
            return

        if mode == MODE_PLANE:
            self._run_plane(work_points, work_normals, seed_center, seed_radius)
        elif mode == MODE_CYLINDER:
            self._run_cylinder(work_points, work_normals, seed_center, seed_radius)
        else:
            self._set_status(f"未対応のモード: {mode}")

    def _compute_seed_indices(self, seed_center: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.adaptive_roi_checkbox.checked:
            radius = float(self.roi_r_min.double_value)
            diff = self.all_points - seed_center
            dist2 = np.einsum("ij,ij->i", diff, diff)
            mask = dist2 <= radius * radius
            return np.where(mask)[0], radius

        indices, radius = adaptive_seed_indices(
            self.all_points,
            seed_center,
            seed_radius_start=float(self.roi_r_min.double_value),
            seed_radius_max=float(self.roi_r_max.double_value),
            seed_radius_step=float(self.roi_r_step.double_value),
            min_seed_points=int(self.roi_min_points.int_value),
        )
        return indices, radius

    def _compute_seed_indices_for_points(
        self,
        points: np.ndarray,
        seed_center: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        if not self.adaptive_roi_checkbox.checked:
            radius = float(self.roi_r_min.double_value)
            diff = points - seed_center
            dist2 = np.einsum("ij,ij->i", diff, diff)
            mask = dist2 <= radius * radius
            return np.where(mask)[0], radius

        indices, radius = adaptive_seed_indices(
            points,
            seed_center,
            seed_radius_start=float(self.roi_r_min.double_value),
            seed_radius_max=float(self.roi_r_max.double_value),
            seed_radius_step=float(self.roi_r_step.double_value),
            min_seed_points=int(self.roi_min_points.int_value),
        )
        return indices, radius

    def _filter_points_by_roi(
        self,
        points: Optional[np.ndarray],
        normals: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if points is None or self._roi is None or not self.roi_use_checkbox.checked:
            return points, normals
        center = np.asarray(self._roi.get("center"), dtype=float)
        if self._roi.get("type") == "cylinder":
            radius = float(self._roi.get("radius", 0.1))
            height = float(self._roi.get("height", 0.2))
            rotation = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
            axis = rotation[:, 2]
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 1e-8:
                axis = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                axis = axis / axis_norm
            vec = points - center
            axial = vec @ axis
            radial = vec - np.outer(axial, axis)
            dist2 = np.einsum("ij,ij->i", radial, radial)
            mask = (np.abs(axial) <= 0.5 * height) & (dist2 <= radius * radius)
        else:
            extent = np.asarray(self._roi.get("extent", [0.2, 0.2, 0.2]), dtype=float)
            rotation = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
            local = (points - center) @ rotation.T
            half = 0.5 * extent
            mask = (np.abs(local) <= half).all(axis=1)
        if mask.size == 0:
            return points, normals
        points_f = points[mask]
        normals_f = normals[mask] if normals is not None and len(normals) == len(points) else None
        return points_f, normals_f

    def _get_working_cloud(
        self,
        seed_center: np.ndarray,
        max_expand_radius: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        points, normals = self._filter_points_by_roi(self.all_points, self.all_normals)
        if (
            points is None
            or max_expand_radius <= 0
        ):
            return points, normals
        if self.roi_use_checkbox.checked and self._roi is not None:
            diff = points - seed_center
            dist2 = np.einsum("ij,ij->i", diff, diff)
            mask = dist2 <= float(max_expand_radius) ** 2
            points = points[mask]
            normals = normals[mask] if normals is not None and len(normals) == len(diff) else None
            return points, normals
        if self._kdtree is None:
            return points, normals
        try:
            k, idx, _ = self._kdtree.search_radius_vector_3d(
                seed_center, float(max_expand_radius)
            )
            if k <= 0:
                return points, normals
            indices = np.asarray(idx[:k], dtype=int)
            sub_points = points[indices] if points is not None else points
            if normals is not None and len(normals) == len(points):
                sub_normals = normals[indices]
            else:
                sub_normals = None
            return sub_points, sub_normals
        except Exception:
            return points, normals

    def _run_plane(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        seed_center: np.ndarray,
        seed_radius: float,
    ):
        if not self.keep_results_checkbox.checked:
            self._clear_results()
        result = expand_plane_from_seed(
            points,
            seed_center,
            normals=normals,
            seed_radius=seed_radius,
            max_expand_radius=float(self.max_expand_radius.double_value),
            grow_radius=float(self.grow_radius.double_value),
            distance_threshold=float(self.plane_threshold.double_value),
            normal_threshold_deg=float(self.normal_th.double_value),
            expand_method=_expand_method_value(self.expand_method.selected_text),
            max_refine_iters=int(self.max_refine_iters.int_value),
            adaptive_refine_threshold=bool(self.adaptive_plane_refine.checked),
            adaptive_refine_k=float(self.adaptive_plane_refine_k.double_value),
            adaptive_refine_min_scale=float(self.adaptive_plane_refine_min_scale.double_value),
            adaptive_refine_max_scale=float(self.adaptive_plane_refine_max_scale.double_value),
            max_expanded_points=int(self.max_expanded_points.int_value),
            max_frontier=int(self.max_frontier.int_value),
            max_steps=int(self.max_steps.int_value),
            verbose=False,
        )
        if not result.success or result.plane is None:
            self._set_status(f"平面抽出失敗: {result.message}")
            return

        try:
            mesh, _ = create_plane_patch_mesh(
                result.plane,
                points,
                np.array([0.2, 0.8, 0.2]),
                padding=0.02,
                patch_shape=_patch_shape_value(self.patch_shape.selected_text),
            )
        except Exception as exc:
            self._set_status(f"平面パッチ生成失敗: {exc}")
            return

        color = self._rng.uniform(0.2, 0.9, size=3)
        label = f"平面 {len(self._result_names) + 1}"
        name = self._register_object(
            label=label,
            geometry=mesh,
            kind="mesh",
            category="result",
            color=color,
            plane_param=result.plane,
        )
        self._result_names.append(name)
        self._result_meshes.append(mesh)

        self.results = append_plane_result(self.results, result.plane)
        if self.auto_save_checkbox.checked:
            save_results(self.results, self.output_path_edit.text_value)
        self._set_status(
            f"平面OK: インライヤ={result.plane.inlier_count}, 面積={result.area:.3f} m^2"
        )

    def _run_cylinder(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray],
        seed_center: np.ndarray,
        seed_radius: float,
    ):
        if not self.keep_results_checkbox.checked:
            self._clear_results()
        self._maybe_autotune_cylinder()
        if self.use_ground_plane.checked and self.ground_plane is None:
            self._estimate_ground_plane()
        if self.use_ceiling_plane.checked and self.ceiling_plane is None:
            self._estimate_ceiling_plane()
        points_fit = points
        normals_fit = normals
        if (
            self.cyl_auto_sample_checkbox.checked
            and points is not None
            and len(points) > int(self.cyl_sample_cap.int_value)
        ):
            cap = max(1000, int(self.cyl_sample_cap.int_value))
            if len(points) > cap:
                indices = self._rng.choice(len(points), size=cap, replace=False)
                points_fit = points[indices]
                if normals is not None and len(normals) == len(points):
                    normals_fit = normals[indices]
                else:
                    normals_fit = None
                self._set_status(f"大規模点群のため {cap} 点をサンプリングして推定します。")
        if self.roi_use_checkbox.checked and self._roi is not None and self._roi.get("type") == "cylinder":
            try:
                roi_center = np.asarray(self._roi.get("center"), dtype=float)
                roi_rot = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
                roi_axis = roi_rot[:, 2]
                roi_axis = roi_axis / max(np.linalg.norm(roi_axis), 1e-9)
                roi_radius = float(self._roi.get("radius", 0.1))
                if np.isfinite(roi_radius) and roi_radius > 0 and points_fit is not None:
                    diff = points_fit - roi_center
                    proj = diff @ roi_axis
                    radial_vec = diff - np.outer(proj, roi_axis)
                    radial_dist = np.linalg.norm(radial_vec, axis=1)
                    margin = max(float(self.cylinder_threshold.double_value) * 3.0, 0.02)
                    band_lo = max(roi_radius * 0.4, roi_radius - margin)
                    band_hi = roi_radius + margin
                    band_mask = (radial_dist >= band_lo) & (radial_dist <= band_hi)
                    if np.count_nonzero(band_mask) >= 200:
                        points_fit = points_fit[band_mask]
                        if normals_fit is not None and len(normals_fit) == len(radial_dist):
                            normals_fit = normals_fit[band_mask]
                        self._set_status("ROI円柱の表面近傍に点を絞って推定します。")
            except Exception:
                pass
        if points_fit is not None and self.use_ground_plane.checked and self.ground_plane is not None:
            try:
                plane_n = np.asarray(self.ground_plane.normal, dtype=float)
                plane_n = plane_n / max(np.linalg.norm(plane_n), 1e-9)
                plane_p = np.asarray(self.ground_plane.point, dtype=float)
                dist = np.abs((points_fit - plane_p) @ plane_n)
                margin = max(float(self.ground_threshold.double_value) * 2.0, 0.02)
                keep = dist > margin
                if np.count_nonzero(keep) >= 200:
                    points_fit = points_fit[keep]
                    if normals_fit is not None and len(normals_fit) == len(dist):
                        normals_fit = normals_fit[keep]
                    self._set_status("地面近傍の点を除去して推定します。")
            except Exception:
                pass
        seed_radius_start = float(self.roi_r_min.double_value)
        seed_radius_max = float(self.roi_r_max.double_value)
        seed_radius_step = float(self.roi_r_step.double_value)
        min_seed_points = int(self.roi_min_points.int_value)
        max_expand_radius_local = float(self.max_expand_radius.double_value)
        if self.roi_use_checkbox.checked and self._roi is not None:
            if self._roi.get("type") == "cylinder":
                try:
                    roi_radius = float(self._roi.get("radius", 0.1))
                except Exception:
                    roi_radius = 0.0
                try:
                    roi_height = float(self._roi.get("height", 0.2))
                except Exception:
                    roi_height = 0.0
                if np.isfinite(roi_radius) and roi_radius > 0:
                    seed_radius_start = max(0.8 * roi_radius, 0.02)
                    seed_radius_max = max(1.4 * roi_radius, seed_radius_start + 0.02)
                    seed_radius_step = max(0.2 * roi_radius, 0.01)
                    min_seed_points = max(30, min_seed_points)
                    if np.isfinite(roi_height) and roi_height > 0:
                        roi_max = float(np.sqrt((0.5 * roi_height) ** 2 + roi_radius ** 2))
                        if np.isfinite(roi_max) and roi_max > 0:
                            max_expand_radius_local = min(max_expand_radius_local, roi_max * 1.05)
        result = None
        final_cyl = None
        if (
            self.roi_use_checkbox.checked
            and self._roi is not None
            and self.roi_data_axis_checkbox.checked
        ):
            try:
                sample_points = points_fit
                sample_normals = normals_fit
                max_sample = 20000
                if points_fit is not None and len(points_fit) > max_sample:
                    indices = center_weighted_sample(points_fit, seed_center, max_samples=max_sample)
                    sample_points = points_fit[indices]
                    if normals_fit is not None and len(normals_fit) == len(points_fit):
                        sample_normals = normals_fit[indices]
                    else:
                        sample_normals = None
                radius_min = 0.01
                radius_max = 1.0
                if self._roi.get("type") == "cylinder":
                    roi_radius = float(self._roi.get("radius", 0.1))
                    if np.isfinite(roi_radius) and roi_radius > 0:
                        radius_min = max(0.3 * roi_radius, 0.01)
                        radius_max = max(1.7 * roi_radius, radius_min + 0.02)
                target_d = float(self.target_diameter.double_value)
                if target_d > 0:
                    radius = target_d * 0.5
                    radius_min = max(0.4 * radius, 0.01)
                    radius_max = max(1.6 * radius, radius_min + 0.02)

                candidates: List[Tuple[np.ndarray, str, float]] = []
                include_vertical = self.ground_plane is not None
                vertical_axis = None
                if self.ground_plane is not None:
                    try:
                        vertical_axis = np.asarray(self.ground_plane.normal, dtype=float)
                    except Exception:
                        vertical_axis = None
                if sample_points is not None and len(sample_points) >= 6:
                    candidates = robust_axis_from_cylinder_points(
                        sample_points,
                        sample_normals,
                        include_vertical=include_vertical,
                        vertical_axis=vertical_axis,
                    )
                roi_axis = None
                if self._roi is not None and self._roi.get("type") == "cylinder":
                    roi_rot = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
                    roi_axis = roi_rot[:, 2]
                if self.roi_axis_lock_checkbox.checked and roi_axis is not None:
                    candidates = [(roi_axis, "roi_axis", 0.95)]
                if not candidates and sample_points is not None and len(sample_points) >= 6:
                    centered = sample_points - sample_points.mean(axis=0)
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    candidates = [(vh[0], "pca0", 0.5)]

                best_score = -1.0
                best_cyl = None
                for axis_prior, source, confidence in candidates:
                    axis_norm = np.linalg.norm(axis_prior)
                    if not np.isfinite(axis_norm) or axis_norm <= 1e-8:
                        continue
                    axis_prior = axis_prior / axis_norm
                    cand = fit_cylinder_with_axis(
                        sample_points,
                        axis_prior,
                        circle_ransac_iters=200,
                        circle_inlier_threshold=float(self.cylinder_threshold.double_value),
                        length_margin=0.05,
                    )
                    if cand is None:
                        continue
                    inliers, median_res = score_cylinder_fit(
                        sample_points,
                        cand.axis_point,
                        cand.axis_direction,
                        cand.radius,
                    )
                    base_score = float(inliers) / max(float(median_res), 1e-6)
                    score = base_score * (0.7 + 0.3 * float(confidence))
                    if source in ("ransac", "normals", "trimmed_pca"):
                        score *= 1.15
                    elif source == "vertical":
                        if vertical_axis is not None:
                            cos_vert = abs(np.dot(cand.axis_direction, vertical_axis))
                            if cos_vert > 0.95:
                                score *= 1.1
                    if score > best_score:
                        best_score = score
                        best_cyl = cand

                initial_cyl = best_cyl
                if initial_cyl is None:
                    initial_cyl = fit_cylinder(
                        sample_points,
                        sample_normals,
                        distance_threshold=float(self.cylinder_threshold.double_value),
                        radius_min=radius_min,
                        radius_max=radius_max,
                        num_iterations=300,
                        normal_angle_threshold_deg=float(self.normal_th.double_value),
                    )
                if initial_cyl is not None and self.cyl_fast_roi_checkbox.checked:
                    axis_dir = np.asarray(initial_cyl.axis_direction, dtype=float)
                    axis_dir = axis_dir / max(np.linalg.norm(axis_dir), 1e-9)
                    diff = points_fit - initial_cyl.axis_point
                    proj = diff @ axis_dir
                    radial_vec = diff - np.outer(proj, axis_dir)
                    radial_dist = np.linalg.norm(radial_vec, axis=1)
                    residual = np.abs(radial_dist - float(initial_cyl.radius))
                    inlier_mask = residual < float(self.cylinder_threshold.double_value)
                    if normals_fit is not None and len(normals_fit) == len(points_fit):
                        normals_unit = normals_fit / np.maximum(
                            np.linalg.norm(normals_fit, axis=1, keepdims=True), 1e-8
                        )
                        radial_dir = radial_vec / np.maximum(radial_dist[:, None], 1e-8)
                        normal_alignment = np.abs(np.einsum("ij,ij->i", radial_dir, normals_unit))
                        inlier_mask &= normal_alignment > np.cos(np.deg2rad(self.normal_th.double_value))
                    inlier_points = points_fit[inlier_mask]
                    if len(inlier_points) >= 6:
                        refined = fit_cylinder_with_axis(
                            inlier_points,
                            axis_dir,
                            circle_ransac_iters=200,
                            circle_inlier_threshold=float(self.cylinder_threshold.double_value),
                            length_margin=0.05,
                        )
                        if refined is not None:
                            initial_cyl = refined
                        new_length, new_axis_point = recompute_cylinder_length_from_inliers(
                            inlier_points,
                            initial_cyl.axis_point,
                            axis_dir,
                            quantile_lo=0.01,
                            quantile_hi=0.99,
                            margin=0.0,
                        )
                        initial_cyl = CylinderParam(
                            axis_point=new_axis_point,
                            axis_direction=axis_dir,
                            radius=float(initial_cyl.radius),
                            length=float(new_length),
                            inlier_count=int(len(inlier_points)),
                            inlier_indices=np.where(inlier_mask)[0],
                        )
                    final_cyl = initial_cyl
                if initial_cyl is not None and final_cyl is None:
                    seed_r = float(max(seed_radius_max, 0.1))
                    expand = expand_cylinder_from_seed(
                        points_fit,
                        initial_cyl.axis_point,
                        normals=normals_fit,
                        initial_cylinder=initial_cyl,
                        seed_radius=seed_r,
                        max_expand_radius=max_expand_radius_local,
                        grow_radius=float(self.grow_radius.double_value),
                        distance_threshold=float(self.cylinder_threshold.double_value),
                        normal_threshold_deg=float(self.normal_th.double_value),
                        expand_method=_expand_method_value(self.expand_method.selected_text),
                        max_refine_iters=int(self.max_refine_iters.int_value),
                        max_expanded_points=int(self.max_expanded_points.int_value),
                        max_frontier=int(self.max_frontier.int_value),
                        max_steps=int(self.max_steps.int_value),
                        radius_min=radius_min,
                        radius_max=radius_max,
                        num_iterations=300,
                        verbose=False,
                    )
                    if expand.success and expand.cylinder is not None:
                        final_cyl = expand.cylinder
                    else:
                        final_cyl = initial_cyl
            except Exception:
                final_cyl = None

        if final_cyl is None:
            result = probe_cylinder_from_seed(
                points_fit,
                seed_center,
                normals=normals_fit,
                seed_radius_start=seed_radius_start,
                seed_radius_max=seed_radius_max,
                seed_radius_step=seed_radius_step,
                min_seed_points=min_seed_points,
                circle_ransac_iters=200,
                circle_inlier_threshold=float(self.cylinder_threshold.double_value),
                length_margin=0.05,
                surface_threshold=float(self.cylinder_threshold.double_value),
                cap_margin=0.05,
                grow_radius=float(self.grow_radius.double_value),
                max_expand_radius=max_expand_radius_local,
                max_expanded_points=int(self.max_expanded_points.int_value),
                max_frontier=int(self.max_frontier.int_value),
                max_steps=int(self.max_steps.int_value),
                refine_iters=int(self.max_refine_iters.int_value),
            )
            if not result.success or result.final is None:
                self._set_status(f"円柱抽出失敗: {result.message}")
                return
            final_cyl = result.final

        if (
            self.roi_use_checkbox.checked
            and self._roi is not None
            and self._roi.get("type") == "cylinder"
            and self.roi_axis_lock_checkbox.checked
        ):
            try:
                roi_rot = np.asarray(self._roi.get("rotation", np.eye(3)), dtype=float)
                roi_axis = roi_rot[:, 2]
                roi_axis = roi_axis / max(np.linalg.norm(roi_axis), 1e-9)
                axis_fit = fit_cylinder_with_axis(
                    points_fit,
                    roi_axis,
                    circle_ransac_iters=200,
                    circle_inlier_threshold=float(self.cylinder_threshold.double_value),
                    length_margin=0.05,
                )
                if axis_fit is not None:
                    if np.dot(axis_fit.axis_direction, roi_axis) < 0:
                        axis_fit.axis_direction = -axis_fit.axis_direction
                    final_cyl = axis_fit
            except Exception:
                pass

        target_d = float(self.target_diameter.double_value)
        tol_d = float(self.target_diameter_tol.double_value)
        if target_d > 0.0 and tol_d > 0.0:
            diameter = float(final_cyl.radius) * 2.0
            if abs(diameter - target_d) > tol_d:
                self._set_status(
                    f"円柱直径が範囲外: {diameter:.3f}m (許容±{tol_d:.3f}m)"
                )
                return

        target_h = float(self.target_height.double_value)
        tol_h = float(self.target_height_tol.double_value)
        if target_h > 0.0 and tol_h > 0.0:
            length = float(final_cyl.length)
            if abs(length - target_h) > tol_h:
                self._set_status(
                    f"円柱高さが範囲外: {length:.3f}m (許容±{tol_h:.3f}m)"
                )
                return

        if self.cyl_vertical_constraint.checked:
            if self.ground_plane is None:
                self._set_status("地面平面が未推定です。")
                return
            axis = np.asarray(final_cyl.axis_direction, dtype=float)
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            normal = np.asarray(self.ground_plane.normal, dtype=float)
            normal = normal / max(np.linalg.norm(normal), 1e-9)
            angle_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, normal)), -1.0, 1.0))))
            if angle_deg > float(self.cyl_vertical_deg.double_value):
                self._set_status(f"円柱の傾きが大きいです: {angle_deg:.1f}°")
                return
        cyl_for_bounds = final_cyl
        if self.snap_axis_checkbox.checked:
            ref = None
            if self.ground_plane is not None:
                ref = np.asarray(self.ground_plane.normal, dtype=float)
            if ref is None or ref.size != 3 or not np.isfinite(ref).all():
                ref = np.array([0.0, 0.0, 1.0], dtype=float)
            snapped = snap_axis_to_reference(
                cyl_for_bounds.axis_direction,
                ref,
                snap_threshold_deg=float(self.snap_axis_deg.double_value),
            )
            cyl_for_bounds = CylinderParam(
                axis_point=cyl_for_bounds.axis_point,
                axis_direction=snapped,
                radius=cyl_for_bounds.radius,
                length=cyl_for_bounds.length,
                inlier_count=cyl_for_bounds.inlier_count,
                inlier_indices=cyl_for_bounds.inlier_indices,
            )

        roi_height = None
        if self.roi_use_checkbox.checked and self._roi is not None and self._roi.get("type") == "cylinder":
            try:
                roi_height = float(self._roi.get("height", 0.0))
            except Exception:
                roi_height = None
        result_cyl = self._apply_cylinder_plane_bounds(
            cyl_for_bounds,
            points=points_fit,
            roi_height=roi_height,
        )
        if result_cyl is None:
            self._set_status("地面/天井平面との整合に失敗しました。")
            return

        radius_scale = float(self.cyl_radius_scale.double_value)
        if not np.isfinite(radius_scale) or radius_scale <= 0:
            radius_scale = 1.0
        adjusted_radius = float(result_cyl.radius) * radius_scale

        cyl_mesh = create_cylinder_mesh(
            result_cyl.axis_point,
            result_cyl.axis_direction,
            adjusted_radius,
            result_cyl.length,
        )
        if cyl_mesh is None:
            self._set_status("円柱メッシュ生成失敗")
            return

        color = self._rng.uniform(0.2, 0.9, size=3)
        label = f"円柱 {len(self._result_names) + 1}"
        name = self._register_object(
            label=label,
            geometry=cyl_mesh,
            kind="mesh",
            category="result",
            color=color,
        )
        self._result_names.append(name)
        self._result_meshes.append(cyl_mesh)

        if abs(radius_scale - 1.0) > 1e-6:
            adjusted = CylinderParam(
                axis_point=result_cyl.axis_point,
                axis_direction=result_cyl.axis_direction,
                radius=adjusted_radius,
                length=result_cyl.length,
                inlier_count=result_cyl.inlier_count,
                inlier_indices=result_cyl.inlier_indices,
            )
            self.results = append_cylinder_result(self.results, adjusted)
        else:
            self.results = append_cylinder_result(self.results, result_cyl)
        if self.auto_save_checkbox.checked:
            save_results(self.results, self.output_path_edit.text_value)
        self._set_status(
            f"円柱OK: 半径={adjusted_radius:.4f} m, 長さ={result_cyl.length:.3f} m"
        )

    def _maybe_autotune_cylinder(self) -> None:
        if not self.auto_tune_cylinder.checked:
            return
        diameter = float(self.target_diameter.double_value)
        height = float(self.target_height.double_value)
        if diameter <= 0.0 and height <= 0.0:
            return

        def _clamp(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val))

        if diameter > 0.0:
            radius = diameter * 0.5
            r_min = _clamp(0.4 * radius, 0.05, 0.25)
            r_step = _clamp(0.2 * radius, 0.02, 0.10)
            r_max = _clamp(1.6 * radius, max(r_min + 0.05, 0.2), 1.2)
            self.roi_r_min.double_value = r_min
            self.roi_r_step.double_value = r_step
            self.roi_r_max.double_value = r_max

            threshold = _clamp(0.02 * diameter + 0.005, 0.01, 0.05)
            self.cylinder_threshold.double_value = threshold

            grow_radius = _clamp(0.7 * radius, 0.05, 0.25)
            self.grow_radius.double_value = grow_radius

            min_points = int(_clamp(200.0 * (radius / 0.2), 60.0, 300.0))
            self.roi_min_points.int_value = max(10, min_points)

            if height > 0.0:
                max_expand = float(np.sqrt((height * 0.5) ** 2 + radius ** 2) * 1.1)
                self.max_expand_radius.double_value = max(0.5, max_expand)
        elif height > 0.0:
            max_expand = float(height * 0.6)
            self.max_expand_radius.double_value = max(0.5, max_expand)

    def _on_estimate_ground(self):
        self._estimate_ground_plane()

    def _on_clear_ground(self):
        self._clear_ground_plane()
        self._set_status("地面平面をクリアしました。")

    def _on_estimate_ceiling(self):
        self._estimate_ceiling_plane()

    def _on_clear_ceiling(self):
        self._clear_ceiling_plane()
        self._set_status("天井平面をクリアしました。")

    def _estimate_ground_plane(self) -> None:
        if self.pcd is None or self.all_points is None:
            self._set_status("点群が読み込まれていません。")
            return
        plane_model, inliers = self._segment_plane(
            distance_threshold=float(self.ground_threshold.double_value),
            ransac_n=int(self.ground_ransac_n.int_value),
            num_iterations=int(self.ground_num_iterations.int_value),
        )
        if plane_model is None or inliers is None:
            self._set_status("地面推定失敗: 平面抽出に失敗しました。")
            return

        if len(inliers) == 0:
            self._set_status("地面推定失敗: インライヤなし")
            return

        normal = np.asarray(plane_model[:3], dtype=float)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            self._set_status("地面推定失敗: 法線が不正")
            return
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal

        nz = float(np.clip(abs(normal[2]), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(nz)))
        if tilt_deg > float(self.ground_max_tilt.double_value):
            self._set_status(f"地面推定失敗: 傾斜が大きい ({tilt_deg:.1f}°)")
            return

        inlier_points = self.all_points[np.asarray(inliers, dtype=int)]
        point = inlier_points.mean(axis=0)
        plane = PlaneParam(
            normal=normal,
            point=point,
            inlier_count=len(inliers),
            inlier_indices=np.asarray(inliers, dtype=int),
            height=float(point[2]),
        )
        self.ground_plane = plane

        if self._ground_name is not None:
            self._remove_object(self._ground_name)
        try:
            mesh, _ = create_plane_patch_mesh(
                plane,
                self.all_points,
                np.array([0.7, 0.2, 0.7]),
                padding=0.05,
                patch_shape=_patch_shape_value(self.patch_shape.selected_text),
            )
            self._ground_name = self._register_object(
                label="地面",
                geometry=mesh,
                kind="mesh",
                category="ground",
                color=np.array([0.7, 0.2, 0.7]),
                plane_param=plane,
            )
        except Exception:
            self._ground_name = None

        self._set_status(f"地面推定OK: 傾斜={tilt_deg:.1f}°, inliers={len(inliers)}")

    def _estimate_ceiling_plane(self) -> None:
        if self.pcd is None or self.all_points is None:
            self._set_status("点群が読み込まれていません。")
            return
        plane_model, inliers = self._segment_plane(
            distance_threshold=float(self.ceiling_threshold.double_value),
            ransac_n=int(self.ground_ransac_n.int_value),
            num_iterations=int(self.ground_num_iterations.int_value),
        )
        if plane_model is None or inliers is None:
            self._set_status("天井推定失敗: 平面抽出に失敗しました。")
            return

        if len(inliers) == 0:
            self._set_status("天井推定失敗: インライヤなし")
            return

        normal = np.asarray(plane_model[:3], dtype=float)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            self._set_status("天井推定失敗: 法線が不正")
            return
        normal = normal / norm
        if normal[2] > 0:
            normal = -normal

        nz = float(np.clip(abs(normal[2]), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(nz)))
        if tilt_deg > float(self.ceiling_max_tilt.double_value):
            self._set_status(f"天井推定失敗: 傾斜が大きい ({tilt_deg:.1f}°)")
            return

        inlier_points = self.all_points[np.asarray(inliers, dtype=int)]
        point = inlier_points.mean(axis=0)
        plane = PlaneParam(
            normal=normal,
            point=point,
            inlier_count=len(inliers),
            inlier_indices=np.asarray(inliers, dtype=int),
            height=float(point[2]),
        )
        self.ceiling_plane = plane

        if self._ceiling_name is not None:
            self._remove_object(self._ceiling_name)
        try:
            mesh, _ = create_plane_patch_mesh(
                plane,
                self.all_points,
                np.array([0.2, 0.4, 0.9]),
                padding=0.05,
                patch_shape=_patch_shape_value(self.patch_shape.selected_text),
            )
            self._ceiling_name = self._register_object(
                label="天井",
                geometry=mesh,
                kind="mesh",
                category="ceiling",
                color=np.array([0.2, 0.4, 0.9]),
                plane_param=plane,
            )
        except Exception:
            self._ceiling_name = None

        self._set_status(f"天井推定OK: 傾斜={tilt_deg:.1f}°, inliers={len(inliers)}")

    def _apply_cylinder_plane_bounds(
        self,
        cyl: CylinderParam,
        *,
        points: Optional[np.ndarray] = None,
        roi_height: Optional[float] = None,
    ) -> Optional[CylinderParam]:
        if cyl is None:
            return None
        axis_dir = np.asarray(cyl.axis_direction, dtype=float)
        axis_norm = float(np.linalg.norm(axis_dir))
        if not np.isfinite(axis_norm) or axis_norm < 1e-9:
            return None
        axis_dir = axis_dir / axis_norm
        axis_point = np.asarray(cyl.axis_point, dtype=float)
        length = float(cyl.length)

        base_length = max(length, 1e-6)
        t_min = -0.5 * base_length
        t_max = 0.5 * base_length
        has_inlier_range = False
        if points is not None and cyl.inlier_indices is not None:
            try:
                idx = np.asarray(cyl.inlier_indices, dtype=int).reshape(-1)
                if idx.size > 0:
                    idx = idx[(idx >= 0) & (idx < len(points))]
                if idx.size > 0:
                    pts_in = np.asarray(points, dtype=float)[idx]
                    t_vals = (pts_in - axis_point) @ axis_dir
                    t_min = float(np.min(t_vals))
                    t_max = float(np.max(t_vals))
                    base_length = max(base_length, t_max - t_min)
                    has_inlier_range = True
            except Exception:
                has_inlier_range = False
        if base_length <= 1e-6 and roi_height is not None:
            if np.isfinite(roi_height) and roi_height > 0:
                base_length = float(roi_height)
                t_min = -0.5 * base_length
                t_max = 0.5 * base_length

        if not has_inlier_range:
            t_min = -0.5 * base_length
            t_max = 0.5 * base_length

        gate = max(0.2 * base_length, 0.05)
        if roi_height is not None and np.isfinite(roi_height) and roi_height > 0:
            gate = max(gate, 0.25 * float(roi_height))

        def intersect_t(plane: PlaneParam) -> Optional[float]:
            n = np.asarray(plane.normal, dtype=float)
            denom = float(np.dot(n, axis_dir))
            if not np.isfinite(denom) or abs(denom) < 1e-6:
                return None
            return float(np.dot(n, np.asarray(plane.point, dtype=float) - axis_point) / denom)

        def is_t_valid(t: float) -> bool:
            return (t >= t_min - gate) and (t <= t_max + gate)

        ts = []
        ground_t = None
        ceiling_t = None
        if self.use_ground_plane.checked and self.ground_plane is not None:
            t = intersect_t(self.ground_plane)
            if t is not None and is_t_valid(t):
                ts.append(t)
                ground_t = t
        if self.use_ceiling_plane.checked and self.ceiling_plane is not None:
            t = intersect_t(self.ceiling_plane)
            if t is not None and is_t_valid(t):
                ts.append(t)
                ceiling_t = t

        if len(ts) == 0:
            return CylinderParam(
                axis_point=axis_point,
                axis_direction=axis_dir,
                radius=float(cyl.radius),
                length=float(base_length),
                inlier_count=int(cyl.inlier_count),
                inlier_indices=cyl.inlier_indices,
            )
        if len(ts) == 1:
            t_plane = ts[0]
            new_length = float(base_length)
            half = 0.5 * max(new_length, 1e-6)
            t_cap_minus = -half
            t_cap_plus = half
            target = t_cap_minus if abs(t_plane - t_cap_minus) <= abs(t_plane - t_cap_plus) else t_cap_plus
            shift = t_plane - target
            axis_point = axis_point + axis_dir * shift
            return CylinderParam(
                axis_point=axis_point,
                axis_direction=axis_dir,
                radius=float(cyl.radius),
                length=float(new_length),
                inlier_count=int(cyl.inlier_count),
                inlier_indices=cyl.inlier_indices,
            )

        t0, t1 = sorted(ts[:2])
        new_length = abs(t1 - t0)
        if not np.isfinite(new_length) or new_length <= 1e-6:
            return cyl
        axis_point = axis_point + axis_dir * ((t0 + t1) * 0.5)
        return CylinderParam(
            axis_point=axis_point,
            axis_direction=axis_dir,
            radius=float(cyl.radius),
            length=float(new_length),
            inlier_count=int(cyl.inlier_count),
            inlier_indices=cyl.inlier_indices,
        )

    def _run_stairs(self, seed_center: np.ndarray, seed_indices: np.ndarray):
        if not self.keep_results_checkbox.checked:
            self._clear_results()
        roi_points = self.all_points[seed_indices]
        if len(roi_points) == 0:
            self._set_status("ROIに点がありません。")
            return

        planes = extract_stair_planes(
            roi_points,
            max_planes=int(self.max_planes.int_value),
            min_inliers=int(self.min_inliers.int_value),
            distance_threshold=float(self.plane_threshold.double_value),
            ransac_n=int(self.stairs_ransac_n.int_value),
            num_iterations=int(self.stairs_num_iterations.int_value),
            max_tilt_deg=float(self.max_tilt.double_value),
            height_eps=float(self.height_eps.double_value),
            horizontal_only=not self.no_horizontal_filter.checked,
            merge_by_height=not self.no_height_merge.checked,
            verbose=False,
        )
        if len(planes) == 0:
            self._set_status("階段平面が検出できませんでした。")
            return

        colors = generate_plane_colors(len(planes))
        for i, (plane, color) in enumerate(zip(planes, colors)):
            try:
                mesh, _ = create_plane_patch_mesh(
                    plane,
                    roi_points,
                    color,
                    padding=0.02,
                    patch_shape=_patch_shape_value(self.patch_shape.selected_text),
                )
            except Exception:
                continue
            color = self._rng.uniform(0.2, 0.9, size=3)
            label = f"階段平面 {len(self._result_names) + 1}"
            name = self._register_object(
                label=label,
                geometry=mesh,
                kind="mesh",
                category="result",
                color=color,
                plane_param=plane,
            )
            self._result_names.append(name)
            self._result_meshes.append(mesh)

        if self.auto_save_checkbox.checked:
            save_stairs_results(
                planes,
                roi_points,
                self.stairs_output_edit.text_value,
                patch_shape=_patch_shape_value(self.patch_shape.selected_text),
            )

        export_path = self.export_mesh_edit.text_value.strip()
        if export_path:
            combined = _combine_meshes(self._result_meshes)
            o3d.io.write_triangle_mesh(export_path, combined)

        self._set_status(f"階段OK: {len(planes)}平面")


def _combine_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    if len(meshes) == 0:
        return o3d.geometry.TriangleMesh()

    vertices_list: List[np.ndarray] = []
    triangles_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []

    vertex_offset = 0
    for mesh in meshes:
        v = np.asarray(mesh.vertices)
        t = np.asarray(mesh.triangles)
        c = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

        vertices_list.append(v)
        triangles_list.append(t + vertex_offset)
        if c is None or len(c) != len(v):
            colors_list.append(np.zeros((len(v), 3), dtype=float))
        else:
            colors_list.append(c)
        vertex_offset += len(v)

    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(np.vstack(vertices_list))
    combined.triangles = o3d.utility.Vector3iVector(np.vstack(triangles_list))
    combined.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors_list))
    combined.compute_vertex_normals()
    return combined


def launch_gui(args=None):
    app = gui.Application.instance
    app.initialize()
    font_override = getattr(args, "gui_font", None) if args is not None else None
    _set_japanese_font(app, font_override=font_override)
    initial_path = None
    initial_profile = None
    output_path = "fit_results.json"
    stairs_output = "stairs_results.json"
    if args is not None:
        initial_path = getattr(args, "input", None)
        initial_profile = getattr(args, "sensor_profile", None)
        output_path = getattr(args, "output", output_path)
        stairs_output = getattr(args, "stairs_output", stairs_output)

    PrimitiveFittingApp(
        initial_path=initial_path,
        initial_profile=initial_profile,
        output_path=output_path,
        stairs_output_path=stairs_output,
    )
    app.run()


if __name__ == "__main__":
    launch_gui()
