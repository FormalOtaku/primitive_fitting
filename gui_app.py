"""Open3D GUI application for primitive fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

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
    extract_stair_planes,
    create_cylinder_mesh,
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

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        self.scene_widget.set_on_mouse(self._on_mouse)
        self.window.add_child(self.scene_widget)

        self.panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        self.window.add_child(self.panel)

        self._pcd_material = rendering.MaterialRecord()
        self._pcd_material.shader = "defaultUnlit"
        self._pcd_material.point_size = 3 * self.window.scaling

        self._mesh_material = rendering.MaterialRecord()
        self._mesh_material.shader = "defaultLit"
        self._line_material = rendering.MaterialRecord()
        self._line_material.shader = "unlitLine"
        self._line_material.line_width = 1.0 * self.window.scaling

        self._result_names: List[str] = []
        self._result_meshes: List[o3d.geometry.TriangleMesh] = []
        self._seed_name: Optional[str] = None

        self._objects: Dict[str, Dict[str, object]] = {}
        self._outliner_items: Dict[int, str] = {}
        self._outliner_root: Optional[int] = None
        self._object_counter = 0

        self.pcd_raw: Optional[o3d.geometry.PointCloud] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.all_points: Optional[np.ndarray] = None
        self.all_normals: Optional[np.ndarray] = None
        self._kdtree: Optional[o3d.geometry.KDTreeFlann] = None
        self.ground_plane: Optional[PlaneParam] = None
        self._ground_name: Optional[str] = None
        self.ceiling_plane: Optional[PlaneParam] = None
        self._ceiling_name: Optional[str] = None

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

        file_group = gui.CollapsableVert("入力", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(file_group)

        self.input_path = gui.TextEdit()
        self.input_path.placeholder_text = "PCD/PLY のパス"
        file_group.add_child(self.input_path)

        file_buttons = gui.Horiz(4)
        self.open_button = gui.Button("開く...")
        self.open_button.set_on_clicked(self._on_open_dialog)
        self.load_button = gui.Button("読み込み")
        self.load_button.set_on_clicked(self._on_load_clicked)
        file_buttons.add_child(self.open_button)
        file_buttons.add_child(self.load_button)
        file_group.add_child(file_buttons)

        preprocess_group = gui.CollapsableVert("前処理", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(preprocess_group)

        self.preprocess_checkbox = gui.Checkbox("前処理を有効化")
        self.preprocess_checkbox.checked = True
        preprocess_group.add_child(self.preprocess_checkbox)

        self.voxel_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.voxel_size.double_value = 0.02
        voxel_row = gui.Horiz(4)
        voxel_row.add_child(gui.Label("ボクセルサイズ"))
        voxel_row.add_child(self.voxel_size)
        preprocess_group.add_child(voxel_row)

        profile_group = gui.CollapsableVert("センサプロファイル", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(profile_group)

        self.profile_combo = gui.Combobox()
        self.profile_combo.add_item(PROFILE_CUSTOM_LABEL)
        for key in SENSOR_PROFILES:
            self.profile_combo.add_item(_profile_display(key))
        self.profile_combo.set_on_selection_changed(self._on_profile_changed)
        if initial_profile and initial_profile in SENSOR_PROFILES:
            self.profile_combo.selected_text = _profile_display(initial_profile)
        else:
            self.profile_combo.selected_text = PROFILE_CUSTOM_LABEL
        profile_group.add_child(self.profile_combo)

        mode_group = gui.CollapsableVert("モード", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(mode_group)

        self.mode_combo = gui.Combobox()
        self.mode_combo.add_item(MODE_PLANE)
        self.mode_combo.add_item(MODE_CYLINDER)
        self.mode_combo.add_item(MODE_STAIRS)
        self.mode_combo.selected_text = MODE_PLANE
        self.mode_combo.set_on_selection_changed(self._on_mode_changed)
        mode_group.add_child(self.mode_combo)

        self.auto_run_checkbox = gui.Checkbox("クリック時に自動実行")
        self.auto_run_checkbox.checked = True
        mode_group.add_child(self.auto_run_checkbox)

        self.keep_results_checkbox = gui.Checkbox("結果を保持")
        self.keep_results_checkbox.checked = True
        mode_group.add_child(self.keep_results_checkbox)

        run_row = gui.Horiz(4)
        self.run_button = gui.Button("実行")
        self.run_button.set_on_clicked(self._on_run_clicked)
        self.clear_button = gui.Button("結果クリア")
        self.clear_button.set_on_clicked(self._on_clear_clicked)
        run_row.add_child(self.run_button)
        run_row.add_child(self.clear_button)
        mode_group.add_child(run_row)

        outliner_group = gui.CollapsableVert("アウトライナー", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(outliner_group)

        self.outliner = gui.TreeView()
        self.outliner.can_select_items_with_children = True
        outliner_group.add_child(self.outliner)

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
        outliner_group.add_child(outliner_row)

        roi_group = gui.CollapsableVert("ROI / シード", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(roi_group)

        self.adaptive_roi_checkbox = gui.Checkbox("適応ROI")
        self.adaptive_roi_checkbox.checked = True
        roi_group.add_child(self.adaptive_roi_checkbox)

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

        roi_group.add_child(self._labeled_row("r_min (最小半径)", self.roi_r_min))
        roi_group.add_child(self._labeled_row("r_max (最大半径)", self.roi_r_max))
        roi_group.add_child(self._labeled_row("r_step (増分)", self.roi_r_step))
        roi_group.add_child(self._labeled_row("最小点数", self.roi_min_points))
        roi_group.add_child(self._labeled_row("クリックスナップ半径", self.pick_snap_radius))

        fit_group = gui.CollapsableVert("平面/円柱パラメータ", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(fit_group)

        self.plane_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.plane_threshold.double_value = 0.01
        self.cylinder_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cylinder_threshold.double_value = 0.02
        self.normal_th = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.normal_th.double_value = 30.0

        fit_group.add_child(self._labeled_row("平面しきい値", self.plane_threshold))
        fit_group.add_child(self._labeled_row("円柱しきい値", self.cylinder_threshold))
        fit_group.add_child(self._labeled_row("法線閾値(度)", self.normal_th))

        self.auto_tune_cylinder = gui.Checkbox("円柱パラメータ自動調整")
        self.auto_tune_cylinder.checked = False
        fit_group.add_child(self.auto_tune_cylinder)

        self.target_diameter = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_diameter.double_value = 0.0
        self.target_height = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_height.double_value = 0.0
        fit_group.add_child(self._labeled_row("想定直径(m)", self.target_diameter))
        fit_group.add_child(self._labeled_row("想定高さ(m)", self.target_height))

        self.target_diameter_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_diameter_tol.double_value = 0.0
        self.target_height_tol = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.target_height_tol.double_value = 0.0
        fit_group.add_child(self._labeled_row("直径許容(±m)", self.target_diameter_tol))
        fit_group.add_child(self._labeled_row("高さ許容(±m)", self.target_height_tol))

        self.expand_method = gui.Combobox()
        self.expand_method.add_item(EXPAND_METHOD_LABELS["component"])
        self.expand_method.add_item(EXPAND_METHOD_LABELS["bfs"])
        self.expand_method.selected_text = EXPAND_METHOD_LABELS["component"]
        fit_group.add_child(self._labeled_row("拡張方法", self.expand_method))

        self.max_expand_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.max_expand_radius.double_value = 5.0
        self.grow_radius = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.grow_radius.double_value = 0.15
        self.max_refine_iters = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_refine_iters.int_value = 3
        fit_group.add_child(self._labeled_row("最大拡張半径", self.max_expand_radius))
        fit_group.add_child(self._labeled_row("成長半径", self.grow_radius))
        fit_group.add_child(self._labeled_row("再フィット回数", self.max_refine_iters))

        self.max_expanded_points = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_expanded_points.int_value = 200000
        self.max_frontier = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_frontier.int_value = 200000
        self.max_steps = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_steps.int_value = 1000000
        fit_group.add_child(self._labeled_row("最大点数", self.max_expanded_points))
        fit_group.add_child(self._labeled_row("フロンティア上限", self.max_frontier))
        fit_group.add_child(self._labeled_row("最大ステップ", self.max_steps))

        self.adaptive_plane_refine = gui.Checkbox("平面しきい値の自動調整")
        self.adaptive_plane_refine.checked = False
        fit_group.add_child(self.adaptive_plane_refine)

        self.adaptive_plane_refine_k = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_k.double_value = 3.0
        self.adaptive_plane_refine_min_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_min_scale.double_value = 0.5
        self.adaptive_plane_refine_max_scale = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.adaptive_plane_refine_max_scale.double_value = 2.0
        fit_group.add_child(self._labeled_row("適応k", self.adaptive_plane_refine_k))
        fit_group.add_child(self._labeled_row("最小スケール", self.adaptive_plane_refine_min_scale))
        fit_group.add_child(self._labeled_row("最大スケール", self.adaptive_plane_refine_max_scale))

        self.patch_shape = gui.Combobox()
        self.patch_shape.add_item(PATCH_SHAPE_LABELS["hull"])
        self.patch_shape.add_item(PATCH_SHAPE_LABELS["rect"])
        self.patch_shape.selected_text = PATCH_SHAPE_LABELS["hull"]
        fit_group.add_child(self._labeled_row("パッチ形状", self.patch_shape))

        consistency_group = gui.CollapsableVert("整合性/地面", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(consistency_group)

        self.use_ground_plane = gui.Checkbox("地面平面を使う")
        self.use_ground_plane.checked = False
        consistency_group.add_child(self.use_ground_plane)

        ground_row = gui.Horiz(4)
        self.estimate_ground_button = gui.Button("地面推定")
        self.estimate_ground_button.set_on_clicked(self._on_estimate_ground)
        self.clear_ground_button = gui.Button("地面クリア")
        self.clear_ground_button.set_on_clicked(self._on_clear_ground)
        ground_row.add_child(self.estimate_ground_button)
        ground_row.add_child(self.clear_ground_button)
        consistency_group.add_child(ground_row)

        self.ground_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ground_threshold.double_value = 0.02
        self.ground_ransac_n = gui.NumberEdit(gui.NumberEdit.INT)
        self.ground_ransac_n.int_value = 3
        self.ground_num_iterations = gui.NumberEdit(gui.NumberEdit.INT)
        self.ground_num_iterations.int_value = 1000
        self.ground_max_tilt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ground_max_tilt.double_value = 20.0
        consistency_group.add_child(self._labeled_row("地面しきい値", self.ground_threshold))
        consistency_group.add_child(self._labeled_row("地面RANSAC n", self.ground_ransac_n))
        consistency_group.add_child(self._labeled_row("地面反復回数", self.ground_num_iterations))
        consistency_group.add_child(self._labeled_row("地面最大傾斜角", self.ground_max_tilt))

        self.use_ceiling_plane = gui.Checkbox("天井平面を使う")
        self.use_ceiling_plane.checked = False
        consistency_group.add_child(self.use_ceiling_plane)

        ceiling_row = gui.Horiz(4)
        self.estimate_ceiling_button = gui.Button("天井推定")
        self.estimate_ceiling_button.set_on_clicked(self._on_estimate_ceiling)
        self.clear_ceiling_button = gui.Button("天井クリア")
        self.clear_ceiling_button.set_on_clicked(self._on_clear_ceiling)
        ceiling_row.add_child(self.estimate_ceiling_button)
        ceiling_row.add_child(self.clear_ceiling_button)
        consistency_group.add_child(ceiling_row)

        self.ceiling_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ceiling_threshold.double_value = 0.02
        self.ceiling_max_tilt = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ceiling_max_tilt.double_value = 20.0
        self.ceiling_min_ratio = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.ceiling_min_ratio.double_value = 0.7
        consistency_group.add_child(self._labeled_row("天井しきい値", self.ceiling_threshold))
        consistency_group.add_child(self._labeled_row("天井最大傾斜角", self.ceiling_max_tilt))
        consistency_group.add_child(self._labeled_row("天井下限比率", self.ceiling_min_ratio))

        self.cyl_vertical_constraint = gui.Checkbox("円柱の垂直制約")
        self.cyl_vertical_constraint.checked = False
        consistency_group.add_child(self.cyl_vertical_constraint)
        self.cyl_vertical_deg = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.cyl_vertical_deg.double_value = 12.0
        consistency_group.add_child(self._labeled_row("許容傾き(度)", self.cyl_vertical_deg))

        stairs_group = gui.CollapsableVert("階段パラメータ", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(stairs_group)

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

        stairs_group.add_child(self._labeled_row("最大平面数", self.max_planes))
        stairs_group.add_child(self._labeled_row("最小インライヤ数", self.min_inliers))
        stairs_group.add_child(self._labeled_row("RANSAC n", self.stairs_ransac_n))
        stairs_group.add_child(self._labeled_row("反復回数", self.stairs_num_iterations))
        stairs_group.add_child(self._labeled_row("最大傾斜角", self.max_tilt))
        stairs_group.add_child(self._labeled_row("高さ許容", self.height_eps))
        stairs_group.add_child(self.no_horizontal_filter)
        stairs_group.add_child(self.no_height_merge)

        output_group = gui.CollapsableVert("出力", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(output_group)

        self.output_path_edit = gui.TextEdit()
        self.output_path_edit.text_value = self.output_path
        output_group.add_child(self._labeled_row("出力JSON", self.output_path_edit))

        self.stairs_output_edit = gui.TextEdit()
        self.stairs_output_edit.text_value = self.stairs_output_path
        output_group.add_child(self._labeled_row("階段JSON", self.stairs_output_edit))

        self.export_mesh_edit = gui.TextEdit()
        self.export_mesh_edit.placeholder_text = "メッシュ出力 (PLY/OBJ)"
        output_group.add_child(self._labeled_row("メッシュ出力", self.export_mesh_edit))

        self.auto_save_checkbox = gui.Checkbox("結果を自動保存")
        self.auto_save_checkbox.checked = True
        output_group.add_child(self.auto_save_checkbox)

        profile_group = gui.CollapsableVert("円柱プロファイル", 0, gui.Margins(6, 6, 6, 6))
        self.panel.add_child(profile_group)

        self.cylinder_profile_path = gui.TextEdit()
        self.cylinder_profile_path.text_value = "cylinder_profile.json"
        profile_group.add_child(self._labeled_row("保存先JSON", self.cylinder_profile_path))

        self.save_profile_button = gui.Button("プロファイル保存")
        self.save_profile_button.set_on_clicked(self._on_save_profile)
        profile_group.add_child(self.save_profile_button)

        self._fit_group = fit_group
        self._stairs_group = stairs_group
        self._update_mode_visibility()

        if self.profile_combo.selected_text != PROFILE_CUSTOM_LABEL:
            key = _profile_key_from_display(self.profile_combo.selected_text)
            if key is not None:
                self._apply_profile(key)

        if self._outliner_root is None:
            self._outliner_root = self.outliner.get_root_item()

    def _labeled_row(self, label: str, widget: gui.Widget) -> gui.Widget:
        row = gui.Horiz(4)
        row.add_child(gui.Label(label))
        row.add_child(widget)
        return row

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = int(360 * self.window.scaling)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

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

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
            gui.KeyModifier.SHIFT
        ):
            def depth_callback(depth_image):
                x = int(event.x - self.scene_widget.frame.x)
                y = int(event.y - self.scene_widget.frame.y)
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
                    if self.auto_run_checkbox.checked:
                        self._run_fit()

                gui.Application.instance.post_to_main_thread(self.window, update)

            self.scene_widget.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _set_status(self, text: str):
        self.status.text = text

    def _load_point_cloud(self, path: str):
        try:
            self.pcd_raw = load_point_cloud(path)
        except Exception as exc:
            self._set_status(f"読み込み失敗: {exc}")
            return

        if self.preprocess_checkbox.checked:
            try:
                self.pcd = preprocess_point_cloud(self.pcd_raw, self.voxel_size.double_value)
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

        self.all_points = np.asarray(self.pcd.points)
        self.all_normals = np.asarray(self.pcd.normals) if self.pcd.has_normals() else None
        try:
            self._kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        except Exception:
            self._kdtree = None

        self._reset_scene_objects()
        self._register_object(
            label="点群",
            geometry=self.pcd,
            kind="pcd",
            category="base",
        )

        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        self.scene_widget.setup_camera(60, bounds, center)
        self.scene_widget.look_at(center, center - [0, 0, 3], [0, -1, 0])

        self._clear_results()
        self._clear_ground_plane()
        self._clear_ceiling_plane()
        self._set_status(f"{len(self.all_points)}点を読み込みました。Shift+クリックで指定してください。")

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
        for item_id in list(self._outliner_items.keys()):
            self.outliner.remove_item(item_id)
        self._outliner_items = {}
        self._objects = {}
        self._object_counter = 0

    def _register_object(
        self,
        *,
        label: str,
        geometry: o3d.geometry.Geometry,
        kind: str,
        category: str,
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

        if kind == "pcd":
            self.scene_widget.scene.add_geometry(solid_name, geometry, self._pcd_material)
        else:
            self.scene_widget.scene.add_geometry(solid_name, geometry, self._mesh_material)
            try:
                line = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)
                line.paint_uniform_color((0.9, 0.9, 0.9))
                obj["wire"] = line
            except Exception:
                obj["wire"] = None

        if self._outliner_root is None:
            self._outliner_root = self.outliner.get_root_item()
        item_id = self.outliner.add_text_item(self._outliner_root, label)
        self._outliner_items[item_id] = base_name
        obj["item_id"] = item_id

        self._objects[base_name] = obj
        return base_name

    def _remove_object(self, name: str):
        obj = self._objects.get(name)
        if obj is None:
            return
        solid_name = obj.get("solid_name")
        wire_name = obj.get("wire_name")
        if solid_name:
            self.scene_widget.scene.remove_geometry(str(solid_name))
        if wire_name:
            self.scene_widget.scene.remove_geometry(str(wire_name))
        item_id = obj.get("item_id")
        if item_id in self._outliner_items:
            self.outliner.remove_item(int(item_id))
            self._outliner_items.pop(int(item_id), None)
        self._objects.pop(name, None)

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
            self.scene_widget.scene.add_geometry(solid_name, obj["solid"], self._mesh_material if kind == "mesh" else self._pcd_material)

    def _on_toggle_visibility(self):
        item = self.outliner.selected_item
        name = self._outliner_items.get(item)
        if name is None:
            return
        obj = self._objects.get(name)
        if obj is None:
            return
        obj["visible"] = not bool(obj.get("visible", True))
        self._sync_object_visibility(name)

    def _on_set_display_mode(self, mode: str):
        item = self.outliner.selected_item
        name = self._outliner_items.get(item)
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
            self._set_status("先にShift+クリックでseed/ROIを指定してください。")
            return

        seed_center = self.last_pick
        seed_indices, seed_radius = self._compute_seed_indices(seed_center)
        if len(seed_indices) == 0:
            self._set_status("ROIが空です。r_min/r_maxを調整するか再クリックしてください。")
            return

        mode = self.mode_combo.selected_text
        if mode == MODE_PLANE:
            self._run_plane(seed_center, seed_radius)
        elif mode == MODE_CYLINDER:
            self._run_cylinder(seed_center, seed_radius)
        elif mode == MODE_STAIRS:
            self._run_stairs(seed_center, seed_indices)
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

    def _run_plane(self, seed_center: np.ndarray, seed_radius: float):
        if not self.keep_results_checkbox.checked:
            self._clear_results()
        result = expand_plane_from_seed(
            self.all_points,
            seed_center,
            normals=self.all_normals,
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
                self.all_points,
                np.array([0.2, 0.8, 0.2]),
                padding=0.02,
                patch_shape=_patch_shape_value(self.patch_shape.selected_text),
            )
        except Exception as exc:
            self._set_status(f"平面パッチ生成失敗: {exc}")
            return

        label = f"平面 {len(self._result_names) + 1}"
        name = self._register_object(
            label=label,
            geometry=mesh,
            kind="mesh",
            category="result",
        )
        self._result_names.append(name)
        self._result_meshes.append(mesh)

        self.results = append_plane_result(self.results, result.plane)
        if self.auto_save_checkbox.checked:
            save_results(self.results, self.output_path_edit.text_value)
        self._set_status(
            f"平面OK: インライヤ={result.plane.inlier_count}, 面積={result.area:.3f} m^2"
        )

    def _run_cylinder(self, seed_center: np.ndarray, seed_radius: float):
        if not self.keep_results_checkbox.checked:
            self._clear_results()
        self._maybe_autotune_cylinder()
        if self.use_ground_plane.checked and self.ground_plane is None:
            self._estimate_ground_plane()
        if self.use_ceiling_plane.checked and self.ceiling_plane is None:
            self._estimate_ceiling_plane()
        result = probe_cylinder_from_seed(
            self.all_points,
            seed_center,
            normals=self.all_normals,
            seed_radius_start=float(self.roi_r_min.double_value),
            seed_radius_max=float(self.roi_r_max.double_value),
            seed_radius_step=float(self.roi_r_step.double_value),
            min_seed_points=int(self.roi_min_points.int_value),
            circle_ransac_iters=200,
            circle_inlier_threshold=float(self.cylinder_threshold.double_value),
            length_margin=0.05,
            surface_threshold=float(self.cylinder_threshold.double_value),
            cap_margin=0.05,
            grow_radius=float(self.grow_radius.double_value),
            max_expand_radius=float(self.max_expand_radius.double_value),
            max_expanded_points=int(self.max_expanded_points.int_value),
            max_frontier=int(self.max_frontier.int_value),
            max_steps=int(self.max_steps.int_value),
            refine_iters=int(self.max_refine_iters.int_value),
        )
        if not result.success or result.final is None:
            self._set_status(f"円柱抽出失敗: {result.message}")
            return

        target_d = float(self.target_diameter.double_value)
        tol_d = float(self.target_diameter_tol.double_value)
        if target_d > 0.0 and tol_d > 0.0:
            diameter = float(result.final.radius) * 2.0
            if abs(diameter - target_d) > tol_d:
                self._set_status(
                    f"円柱直径が範囲外: {diameter:.3f}m (許容±{tol_d:.3f}m)"
                )
                return

        target_h = float(self.target_height.double_value)
        tol_h = float(self.target_height_tol.double_value)
        if target_h > 0.0 and tol_h > 0.0:
            length = float(result.final.length)
            if abs(length - target_h) > tol_h:
                self._set_status(
                    f"円柱高さが範囲外: {length:.3f}m (許容±{tol_h:.3f}m)"
                )
                return

        if self.cyl_vertical_constraint.checked:
            if self.ground_plane is None:
                self._set_status("地面平面が未推定です。")
                return
            axis = np.asarray(result.final.axis_direction, dtype=float)
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            normal = np.asarray(self.ground_plane.normal, dtype=float)
            normal = normal / max(np.linalg.norm(normal), 1e-9)
            angle_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, normal)), -1.0, 1.0))))
            if angle_deg > float(self.cyl_vertical_deg.double_value):
                self._set_status(f"円柱の傾きが大きいです: {angle_deg:.1f}°")
                return

        cyl_mesh = create_cylinder_mesh(
            result.final.axis_point,
            result.final.axis_direction,
            result.final.radius,
            result.final.length,
        )
        if cyl_mesh is None:
            self._set_status("円柱メッシュ生成失敗")
            return

        label = f"円柱 {len(self._result_names) + 1}"
        name = self._register_object(
            label=label,
            geometry=cyl_mesh,
            kind="mesh",
            category="result",
        )
        self._result_names.append(name)
        self._result_meshes.append(cyl_mesh)

        self.results = append_cylinder_result(self.results, result.final)
        if self.auto_save_checkbox.checked:
            save_results(self.results, self.output_path_edit.text_value)
        self._set_status(
            f"円柱OK: 半径={result.final.radius:.4f} m, 長さ={result.final.length:.3f} m"
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
        try:
            plane_model, inliers = self.pcd.segment_plane(
                distance_threshold=float(self.ground_threshold.double_value),
                ransac_n=int(self.ground_ransac_n.int_value),
                num_iterations=int(self.ground_num_iterations.int_value),
            )
        except Exception as exc:
            self._set_status(f"地面推定失敗: {exc}")
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
            )
        except Exception:
            self._ground_name = None

        self._set_status(f"地面推定OK: 傾斜={tilt_deg:.1f}°, inliers={len(inliers)}")

    def _estimate_ceiling_plane(self) -> None:
        if self.all_points is None or self.pcd is None:
            self._set_status("点群が読み込まれていません。")
            return
        z_vals = self.all_points[:, 2]
        if z_vals.size == 0:
            self._set_status("天井推定失敗: 点がありません")
            return
        ratio = float(self.ceiling_min_ratio.double_value)
        ratio = max(0.1, min(0.95, ratio))
        z_min = float(z_vals.min())
        z_max = float(z_vals.max())
        z_thr = z_min + ratio * (z_max - z_min)
        mask = z_vals >= z_thr
        idx = np.where(mask)[0]
        if len(idx) < 50:
            self._set_status("天井推定失敗: 上部点が少なすぎます")
            return
        subset = self.all_points[idx]
        pcd_sub = o3d.geometry.PointCloud()
        pcd_sub.points = o3d.utility.Vector3dVector(subset)
        try:
            plane_model, inliers = pcd_sub.segment_plane(
                distance_threshold=float(self.ceiling_threshold.double_value),
                ransac_n=int(self.ground_ransac_n.int_value),
                num_iterations=int(self.ground_num_iterations.int_value),
            )
        except Exception as exc:
            self._set_status(f"天井推定失敗: {exc}")
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

        inlier_global = idx[np.asarray(inliers, dtype=int)]
        inlier_points = self.all_points[inlier_global]
        point = inlier_points.mean(axis=0)
        plane = PlaneParam(
            normal=normal,
            point=point,
            inlier_count=len(inliers),
            inlier_indices=inlier_global,
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
            )
        except Exception:
            self._ceiling_name = None

        self._set_status(f"天井推定OK: 傾斜={tilt_deg:.1f}°, inliers={len(inliers)}")

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
            label = f"階段平面 {len(self._result_names) + 1}"
            name = self._register_object(
                label=label,
                geometry=mesh,
                kind="mesh",
                category="result",
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
