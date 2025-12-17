#!/usr/bin/env python3
"""
main.py - Primitive Fitting Tool for LiDAR point clouds

CLI tool for fitting plane and cylinder primitives to point cloud data.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import open3d as o3d

from primitives import fit_plane, fit_cylinder, PlaneParam, CylinderParam, extract_stair_planes


# =============================================================================
# Sensor Profile System
# =============================================================================

@dataclass
class SensorProfile:
    """
    Configuration profile for specific sensor/map combinations.

    Attributes:
        name: Human-readable profile name
        voxel_size: Voxel size for downsampling
        r_min: Minimum ROI radius for adaptive selection
        r_max: Maximum ROI radius for adaptive selection
        r_step: Step size to increase radius when points are insufficient
        min_points: Minimum number of points required in ROI
        plane_distance_threshold: RANSAC distance threshold for plane fitting
        cylinder_distance_threshold: RANSAC distance threshold for cylinder fitting
    """
    name: str
    voxel_size: float = 0.01
    r_min: float = 0.2
    r_max: float = 1.0
    r_step: float = 0.1
    min_points: int = 80
    plane_distance_threshold: float = 0.01
    cylinder_distance_threshold: float = 0.02


# Built-in sensor profiles
SENSOR_PROFILES: Dict[str, SensorProfile] = {
    "default": SensorProfile(
        name="Default",
        voxel_size=0.01,
        r_min=0.2,
        r_max=0.5,
        r_step=0.1,
        min_points=50,
        plane_distance_threshold=0.01,
        cylinder_distance_threshold=0.02,
    ),
    "mid70_map": SensorProfile(
        name="Livox Mid-70 (FAST-LIO2 Map)",
        voxel_size=0.05,
        r_min=0.2,
        r_max=1.0,
        r_step=0.1,
        min_points=80,
        plane_distance_threshold=0.02,
        cylinder_distance_threshold=0.03,
    ),
    "mid70_dense": SensorProfile(
        name="Livox Mid-70 (Dense Scan)",
        voxel_size=0.02,
        r_min=0.1,
        r_max=0.5,
        r_step=0.05,
        min_points=100,
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

# tkinter availability check (lazy import)
_TKINTER_AVAILABLE: Optional[bool] = None


def _check_tkinter() -> bool:
    """Check if tkinter is available."""
    global _TKINTER_AVAILABLE
    if _TKINTER_AVAILABLE is None:
        try:
            import tkinter as tk
            # Test that we can actually create a root window
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            _TKINTER_AVAILABLE = True
        except Exception:
            _TKINTER_AVAILABLE = False
    return _TKINTER_AVAILABLE


def select_file_with_dialog() -> Optional[str]:
    """
    Open a file dialog to select a point cloud file.

    Returns:
        Selected file path, or None if cancelled.

    Raises:
        RuntimeError: If tkinter is not available.
    """
    if not _check_tkinter():
        raise RuntimeError(
            "tkinter is not available. "
            "Please install tkinter or use --input to specify the file path."
        )

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    filepath = filedialog.askopenfilename(
        title="Select Point Cloud File",
        filetypes=[
            ("Point Cloud Files", "*.pcd *.ply"),
            ("PCD Files", "*.pcd"),
            ("PLY Files", "*.ply"),
            ("All Files", "*.*"),
        ],
        initialdir=Path.home(),
    )

    root.destroy()

    if filepath:
        return filepath
    return None


# =============================================================================
# Point Cloud I/O and Preprocessing
# =============================================================================

def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from PCD or PLY file."""
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud from {filepath}")
    print(f"Loaded {len(pcd.points)} points from {filepath}")
    return pcd


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    estimate_normals: bool = True,
    normal_radius: float = 0.05
) -> o3d.geometry.PointCloud:
    """
    Preprocess point cloud: downsample, remove outliers, estimate normals.

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling (0 to skip)
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal
        estimate_normals: Whether to estimate normals
        normal_radius: Search radius for normal estimation

    Returns:
        Preprocessed point cloud
    """
    result = pcd

    # Voxel downsampling
    if voxel_size > 0:
        result = result.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling: {len(result.points)} points")

    # Statistical outlier removal
    if nb_neighbors > 0:
        result, _ = result.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        print(f"After outlier removal: {len(result.points)} points")

    # Normal estimation
    if estimate_normals:
        result.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=30
            )
        )
        result.orient_normals_consistent_tangent_plane(k=15)
        print("Normals estimated")

    return result


# =============================================================================
# ROI Selection
# =============================================================================

@dataclass
class AdaptiveROIResult:
    """Result of adaptive ROI selection."""
    roi_pcd: Optional[o3d.geometry.PointCloud]
    final_radius: float
    point_count: int
    success: bool
    message: str


class ROISelector:
    """Interactive ROI selection using Open3D visualizer."""

    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.selected_indices: Optional[np.ndarray] = None
        self.vis = None
        self._pcd_tree: Optional[o3d.geometry.KDTreeFlann] = None

    def _get_kdtree(self) -> o3d.geometry.KDTreeFlann:
        """Lazily build and cache KD-tree."""
        if self._pcd_tree is None:
            self._pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        return self._pcd_tree

    def select_roi_adaptive(
        self,
        r_min: float = 0.2,
        r_max: float = 1.0,
        r_step: float = 0.1,
        min_points: int = 80
    ) -> AdaptiveROIResult:
        """
        Select ROI with adaptive radius expansion based on point density.

        The radius starts at r_min and expands by r_step until either:
        - min_points are found in the ROI, or
        - r_max is exceeded (returns warning)

        Args:
            r_min: Initial/minimum radius
            r_max: Maximum allowed radius
            r_step: Step size for radius expansion
            min_points: Minimum points required in ROI

        Returns:
            AdaptiveROIResult containing ROI point cloud and metadata
        """
        print("\n=== ROI Selection Mode (Adaptive Radius) ===")
        print(f"Parameters: r_min={r_min}m, r_max={r_max}m, step={r_step}m, min_points={min_points}")
        print("1. Shift + Left Click to pick a center point")
        print("2. Press 'Q' to confirm selection")
        print("3. Press 'Escape' to cancel")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Select ROI Center Point")
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        picked_indices = vis.get_picked_points()

        if len(picked_indices) == 0:
            return AdaptiveROIResult(
                roi_pcd=None,
                final_radius=0.0,
                point_count=0,
                success=False,
                message="No point selected, cancelling..."
            )

        # Get the first picked point as center
        center_idx = picked_indices[0]
        center_point = np.asarray(self.pcd.points)[center_idx]
        print(f"Selected center point: {center_point}")

        # Adaptive radius expansion
        pcd_tree = self._get_kdtree()
        current_radius = r_min
        point_count = 0
        idx = []

        while current_radius <= r_max:
            [k, idx, _] = pcd_tree.search_radius_vector_3d(center_point, current_radius)
            point_count = k

            if point_count >= min_points:
                break

            print(f"  Radius {current_radius:.2f}m: {point_count} points (< {min_points}), expanding...")
            current_radius += r_step

        # Check if we exceeded r_max without finding enough points
        if point_count < min_points:
            msg = (
                f"WARNING: ROI is too sparse. Found only {point_count} points "
                f"at maximum radius {r_max}m (required: {min_points}). "
                "Fitting may be unreliable or skipped."
            )
            print(f"\n{msg}")
            if point_count == 0:
                return AdaptiveROIResult(
                    roi_pcd=None,
                    final_radius=current_radius,
                    point_count=0,
                    success=False,
                    message=msg
                )

        self.selected_indices = np.array(idx)
        roi_pcd = self.pcd.select_by_index(self.selected_indices)

        # Log final ROI radius
        final_radius = min(current_radius, r_max)
        print(f"\nFinal ROI radius: {final_radius:.2f} m, points in ROI: {len(roi_pcd.points)}")

        return AdaptiveROIResult(
            roi_pcd=roi_pcd,
            final_radius=final_radius,
            point_count=len(roi_pcd.points),
            success=point_count >= min_points,
            message=f"ROI selected with {len(roi_pcd.points)} points at radius {final_radius:.2f}m"
        )

    def select_roi_picking(self, radius: float = 0.1) -> Optional[o3d.geometry.PointCloud]:
        """
        Select ROI by picking a point and selecting neighbors within radius.

        Args:
            radius: Radius around picked point to include

        Returns:
            Point cloud containing selected ROI, or None if cancelled
        """
        print("\n=== ROI Selection Mode ===")
        print("1. Shift + Left Click to pick a center point")
        print("2. Press 'Q' to confirm selection")
        print("3. Press 'Escape' to cancel")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Select ROI Center Point")
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        picked_indices = vis.get_picked_points()

        if len(picked_indices) == 0:
            print("No point selected, cancelling...")
            return None

        # Get the first picked point as center
        center_idx = picked_indices[0]
        center_point = np.asarray(self.pcd.points)[center_idx]
        print(f"Selected center point: {center_point}")

        # Find all points within radius
        pcd_tree = self._get_kdtree()
        [k, idx, _] = pcd_tree.search_radius_vector_3d(center_point, radius)

        if k == 0:
            print("No points found in radius")
            return None

        self.selected_indices = np.array(idx)
        roi_pcd = self.pcd.select_by_index(self.selected_indices)
        print(f"Selected {len(roi_pcd.points)} points in ROI")

        return roi_pcd

    def select_roi_crop(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Select ROI by cropping with a bounding box.

        Returns:
            Point cloud containing selected ROI, or None if cancelled
        """
        print("\n=== ROI Crop Selection Mode ===")
        print("Use Open3D crop tool to select region")
        print("Press 'K' to lock/unlock view for crop box editing")
        print("Press 'C' to crop")
        print("Press 'Q' to finish")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Crop ROI")
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        # VisualizerWithEditing returns picked points, not cropped geometry
        # For actual crop, we'd need different approach
        # This is a simplified version using picked points

        picked = vis.get_picked_points()
        if len(picked) < 2:
            print("Need at least 2 points to define crop region")
            return None

        # Use picked points to define bounding box
        picked_points = np.asarray(self.pcd.points)[picked]
        min_bound = np.min(picked_points, axis=0)
        max_bound = np.max(picked_points, axis=0)

        # Add margin
        margin = 0.1
        min_bound -= margin
        max_bound += margin

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        roi_pcd = self.pcd.crop(bbox)

        if roi_pcd.is_empty():
            print("No points in crop region")
            return None

        print(f"Cropped {len(roi_pcd.points)} points")
        return roi_pcd


# =============================================================================
# Visualization
# =============================================================================

def visualize_plane_fit(
    pcd: o3d.geometry.PointCloud,
    plane: PlaneParam,
    roi_pcd: Optional[o3d.geometry.PointCloud] = None
):
    """Visualize fitted plane with point cloud."""
    geometries = [pcd]

    # Color inlier points
    if roi_pcd is not None:
        roi_colored = o3d.geometry.PointCloud(roi_pcd)
        roi_colored.paint_uniform_color([1.0, 0.0, 0.0])  # Red for ROI
        geometries.append(roi_colored)

    # Create plane mesh for visualization
    # Create a small plane mesh centered at the point
    plane_size = 0.5
    plane_mesh = o3d.geometry.TriangleMesh.create_box(
        width=plane_size, height=plane_size, depth=0.001
    )
    plane_mesh.translate([-plane_size/2, -plane_size/2, 0])

    # Rotate to align with normal
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, plane.normal)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, plane.normal), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        plane_mesh.rotate(R, center=[0, 0, 0])

    plane_mesh.translate(plane.point)
    plane_mesh.paint_uniform_color([0.0, 0.8, 0.0])  # Green
    plane_mesh.compute_vertex_normals()
    geometries.append(plane_mesh)

    print("\nPlane Parameters:")
    print(f"  Normal: {plane.normal}")
    print(f"  Point: {plane.point}")
    print(f"  Inliers: {plane.inlier_count}")

    o3d.visualization.draw_geometries(geometries, window_name="Plane Fit Result")


def visualize_cylinder_fit(
    pcd: o3d.geometry.PointCloud,
    cylinder: CylinderParam,
    roi_pcd: Optional[o3d.geometry.PointCloud] = None
):
    """Visualize fitted cylinder with point cloud."""
    geometries = [pcd]

    # Color ROI points
    if roi_pcd is not None:
        roi_colored = o3d.geometry.PointCloud(roi_pcd)
        roi_colored.paint_uniform_color([1.0, 0.0, 0.0])  # Red for ROI
        geometries.append(roi_colored)

    # Create cylinder mesh
    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=cylinder.radius,
        height=cylinder.length,
        resolution=20,
        split=4
    )

    # Rotate to align with axis direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, cylinder.axis_direction)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, cylinder.axis_direction), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        cylinder_mesh.rotate(R, center=[0, 0, 0])

    cylinder_mesh.translate(cylinder.axis_point)
    cylinder_mesh.paint_uniform_color([0.0, 0.0, 0.8])  # Blue
    cylinder_mesh.compute_vertex_normals()
    geometries.append(cylinder_mesh)

    # Draw axis line
    axis_start = cylinder.axis_point - cylinder.axis_direction * cylinder.length / 2
    axis_end = cylinder.axis_point + cylinder.axis_direction * cylinder.length / 2
    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector([axis_start, axis_end])
    axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # Yellow
    geometries.append(axis_line)

    print("\nCylinder Parameters:")
    print(f"  Axis Point: {cylinder.axis_point}")
    print(f"  Axis Direction: {cylinder.axis_direction}")
    print(f"  Radius: {cylinder.radius:.4f}")
    print(f"  Length: {cylinder.length:.4f}")
    print(f"  Inliers: {cylinder.inlier_count}")

    o3d.visualization.draw_geometries(geometries, window_name="Cylinder Fit Result")


def generate_plane_colors(n: int) -> List[np.ndarray]:
    """Generate n distinct colors for plane visualization."""
    colors = []
    for i in range(n):
        # Use HSV color space for distinct colors
        hue = i / max(n, 1)
        # Convert HSV to RGB (saturation=0.8, value=0.9)
        h = hue * 6.0
        c = 0.9 * 0.8
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
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


def create_plane_patch_mesh(
    plane: PlaneParam,
    roi_points: np.ndarray,
    color: np.ndarray,
    *,
    padding: float = 0.0,
    use_convex_hull: bool = True,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, float]]:
    """
    Create a plane patch mesh from inlier points.

    - Inliers are projected onto the plane
    - A 2D convex hull is computed in plane-local coordinates
    - The hull is triangulated (fan triangulation)
    - Vertex colors are assigned (PLY keeps them)
    """
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

    # Project inliers to the plane to remove normal-direction noise.
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

    hull_2d: np.ndarray
    if use_convex_hull:
        hull_2d = _convex_hull_2d(coords)
        if len(hull_2d) < 3:
            use_convex_hull = False

    if not use_convex_hull:
        min_u, max_u = float(coords[:, 0].min()), float(coords[:, 0].max())
        min_v, max_v = float(coords[:, 1].min()), float(coords[:, 1].max())
        hull_2d = np.array(
            [
                [min_u, min_v],
                [max_u, min_v],
                [max_u, max_v],
                [min_u, max_v],
            ],
            dtype=float,
        )

    if padding > 0:
        centroid_2d = hull_2d.mean(axis=0)
        vec = hull_2d - centroid_2d
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        hull_2d = centroid_2d + vec * ((norms + padding) / np.maximum(norms, 1e-9))

    area = _polygon_area_2d(hull_2d)
    vertices = origin + hull_2d[:, 0:1] * u[None, :] + hull_2d[:, 1:2] * v[None, :]

    triangles = np.array([[0, i, i + 1] for i in range(1, len(vertices) - 1)], dtype=int)
    if len(triangles) == 0:
        raise ValueError("Failed to triangulate plane patch")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(vertices), 1)))
    mesh.compute_vertex_normals()

    return mesh, {"extent_u": extent_u, "extent_v": extent_v, "area": float(area)}


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


def visualize_stair_planes(
    roi_pcd: o3d.geometry.PointCloud,
    planes: List[PlaneParam],
    show_roi_points: bool = True
):
    """
    Visualize multiple stair planes with the ROI point cloud.

    Args:
        roi_pcd: Point cloud of the ROI
        planes: List of PlaneParam for detected planes
        show_roi_points: If True, show ROI points in light gray
    """
    geometries = []
    roi_points = np.asarray(roi_pcd.points)

    # Add ROI point cloud in light gray
    if show_roi_points:
        roi_vis = o3d.geometry.PointCloud(roi_pcd)
        roi_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
        geometries.append(roi_vis)

    # Generate colors for planes
    colors = generate_plane_colors(len(planes))

    # Create mesh for each plane
    for i, (plane, color) in enumerate(zip(planes, colors)):
        try:
            mesh, metrics = create_plane_patch_mesh(plane, roi_points, color, padding=0.02, use_convex_hull=True)
            geometries.append(mesh)

            height = float(plane.height) if plane.height is not None else float(np.asarray(plane.point)[2])
            nz = float(np.clip(np.asarray(plane.normal, dtype=float)[2], -1.0, 1.0))
            tilt_deg = float(np.rad2deg(np.arccos(abs(nz))))
            small_patch = metrics["area"] < 0.01 or min(metrics["extent_u"], metrics["extent_v"]) < 0.10
            print(
                f"  [{i:02d}] height={height:+.3f}m tilt={tilt_deg:4.1f}deg "
                f"inliers={plane.inlier_count:6d} extent=({metrics['extent_u']:.2f} x {metrics['extent_v']:.2f})m "
                f"area={metrics['area']:.3f}m^2{'  <SMALL>' if small_patch else ''}"
            )
        except Exception as exc:
            print(f"  [{i:02d}] Failed to create plane patch mesh: {type(exc).__name__}: {exc}")

        # Also highlight inlier points with the same color
        if plane.inlier_indices is not None and len(plane.inlier_indices) > 0:
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(roi_points[plane.inlier_indices])
            inlier_pcd.paint_uniform_color(color * 0.8)  # Slightly darker
            geometries.append(inlier_pcd)

    print(f"\nVisualization: {len(planes)} plane patches")
    print("  - Light gray: ROI points")
    print("  - Colored patches: Detected stair planes")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Stair Planes ({len(planes)} detected)"
    )


def export_stair_planes_mesh(
    planes: List[PlaneParam],
    roi_points: np.ndarray,
    filepath: str,
    padding: float = 0.0
):
    """
    Export all stair plane meshes to a single PLY file.

    Args:
        planes: List of PlaneParam
        roi_points: Points in the ROI
        filepath: Output file path (PLY or OBJ)
        padding: Padding around plane patches
    """
    if len(planes) == 0:
        print("No planes to export")
        return

    colors = generate_plane_colors(len(planes))
    meshes: List[o3d.geometry.TriangleMesh] = []

    for i, (plane, color) in enumerate(zip(planes, colors)):
        try:
            mesh, metrics = create_plane_patch_mesh(plane, roi_points, color, padding=padding, use_convex_hull=True)
            meshes.append(mesh)
            height = float(plane.height) if plane.height is not None else float(np.asarray(plane.point)[2])
            small_patch = metrics["area"] < 0.01 or min(metrics["extent_u"], metrics["extent_v"]) < 0.10
            print(
                f"  Export[{i:02d}] height={height:+.3f}m inliers={plane.inlier_count:6d} "
                f"extent=({metrics['extent_u']:.2f} x {metrics['extent_v']:.2f})m "
                f"area={metrics['area']:.3f}m^2{'  <SMALL>' if small_patch else ''}"
            )
        except Exception as exc:
            print(f"  Export[{i:02d}] skipped: {type(exc).__name__}: {exc}")

    if len(meshes) == 0:
        print("No valid plane meshes to export")
        return

    combined_mesh = _combine_meshes(meshes)

    # Export
    success = o3d.io.write_triangle_mesh(filepath, combined_mesh)
    if success:
        print(f"Exported {len(meshes)} plane meshes to {filepath}")
    else:
        print(f"Failed to export mesh to {filepath}")


# =============================================================================
# Result I/O
# =============================================================================

def load_results(filepath: str) -> dict:
    """Load existing results from JSON file."""
    path = Path(filepath)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {"planes": [], "cylinders": []}


def save_results(results: dict, filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def append_plane_result(results: dict, plane: PlaneParam) -> dict:
    """Append plane result to results dictionary."""
    plane_id = len(results["planes"])
    results["planes"].append({
        "id": plane_id,
        "normal": plane.normal.tolist(),
        "point": plane.point.tolist(),
        "inlier_count": plane.inlier_count
    })
    return results


def append_cylinder_result(results: dict, cylinder: CylinderParam) -> dict:
    """Append cylinder result to results dictionary."""
    cylinder_id = len(results["cylinders"])
    results["cylinders"].append({
        "id": cylinder_id,
        "axis_point": cylinder.axis_point.tolist(),
        "axis_direction": cylinder.axis_direction.tolist(),
        "radius": cylinder.radius,
        "length": cylinder.length,
        "inlier_count": cylinder.inlier_count
    })
    return results


def save_stairs_results(
    planes: List[PlaneParam],
    filepath: str
):
    """
    Save stair plane results to JSON file.

    Args:
        planes: List of PlaneParam detected as stair planes
        filepath: Output JSON file path
    """
    result = {
        "mode": "stairs",
        "plane_count": len(planes),
        "planes": []
    }

    for i, plane in enumerate(planes):
        result["planes"].append({
            "id": i,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "height": plane.height,
            "inlier_count": plane.inlier_count
        })

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Stairs results saved to {filepath}")


# =============================================================================
# Main Application
# =============================================================================

def list_sensor_profiles() -> str:
    """Return a formatted string listing available sensor profiles."""
    lines = ["Available sensor profiles:"]
    for key, profile in SENSOR_PROFILES.items():
        lines.append(f"  {key}: {profile.name}")
        lines.append(f"      voxel_size={profile.voxel_size}, r_min={profile.r_min}, "
                     f"r_max={profile.r_max}, min_points={profile.min_points}")
    return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Primitive Fitting Tool for LiDAR point clouds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=list_sensor_profiles()
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        default=None,
        help="Path to input PCD or PLY file (optional: opens file dialog if not specified)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="fit_results.json",
        help="Output JSON file for results (default: fit_results.json)"
    )

    # Sensor profile
    parser.add_argument(
        "--sensor-profile",
        type=str,
        default=None,
        choices=list(SENSOR_PROFILES.keys()),
        metavar="PROFILE",
        help=f"Sensor profile to use. Available: {', '.join(SENSOR_PROFILES.keys())}"
    )

    # Preprocessing options (can override profile)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for downsampling (overrides profile setting)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        dest="no_preprocess",
        help="Skip preprocessing (downsampling, outlier removal)"
    )

    # ROI adaptive radius options (can override profile)
    roi_group = parser.add_argument_group("ROI Adaptive Radius Options")
    roi_group.add_argument(
        "--roi-r-min",
        type=float,
        default=None,
        help="Minimum ROI radius in meters (overrides profile)"
    )
    roi_group.add_argument(
        "--roi-r-max",
        type=float,
        default=None,
        help="Maximum ROI radius in meters (overrides profile)"
    )
    roi_group.add_argument(
        "--roi-r-step",
        type=float,
        default=None,
        help="ROI radius step size in meters (overrides profile)"
    )
    roi_group.add_argument(
        "--roi-min-points",
        type=int,
        default=None,
        help="Minimum points required in ROI (overrides profile)"
    )
    roi_group.add_argument(
        "--no-adaptive-roi",
        action="store_true",
        dest="no_adaptive_roi",
        help="Disable adaptive ROI, use fixed radius from --roi-r-min"
    )

    # RANSAC thresholds (can override profile)
    ransac_group = parser.add_argument_group("RANSAC Options")
    ransac_group.add_argument(
        "--plane-threshold",
        type=float,
        default=None,
        help="Distance threshold for plane RANSAC (overrides profile)"
    )
    ransac_group.add_argument(
        "--cylinder-threshold",
        type=float,
        default=None,
        help="Distance threshold for cylinder RANSAC (overrides profile)"
    )

    # Legacy argument for backward compatibility
    parser.add_argument(
        "--roi_radius",
        type=float,
        default=None,
        help=argparse.SUPPRESS  # Hidden, for backward compatibility
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help=argparse.SUPPRESS  # Hidden, for backward compatibility
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        dest="no_preprocess_legacy",
        help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        dest="no_gui",
        help="Disable GUI file dialog (requires --input to be specified)"
    )

    # Stairs mode options
    stairs_group = parser.add_argument_group("Stairs Mode Options")
    stairs_group.add_argument(
        "--stairs-mode",
        action="store_true",
        dest="stairs_mode",
        help="Enable stairs mode: extract multiple horizontal planes from ROI"
    )
    stairs_group.add_argument(
        "--max-planes",
        type=int,
        default=20,
        help="Maximum number of planes to extract in stairs mode (default: 20)"
    )
    stairs_group.add_argument(
        "--min-inliers",
        type=int,
        default=50,
        help="Minimum inliers required for a plane in stairs mode (default: 50)"
    )
    stairs_group.add_argument(
        "--stairs-ransac-n",
        type=int,
        default=3,
        help="RANSAC n (sample size) for plane extraction in stairs mode (default: 3)"
    )
    stairs_group.add_argument(
        "--stairs-num-iterations",
        type=int,
        default=1000,
        help="RANSAC iterations per plane in stairs mode (default: 1000)"
    )
    stairs_group.add_argument(
        "--max-tilt",
        type=float,
        default=15.0,
        help="Maximum tilt angle (degrees) for horizontal plane filter (default: 15.0)"
    )
    stairs_group.add_argument(
        "--height-eps",
        type=float,
        default=0.03,
        help="Height tolerance (meters) for merging planes (default: 0.03)"
    )
    stairs_group.add_argument(
        "--no-horizontal-filter",
        action="store_true",
        dest="no_horizontal_filter",
        help="Disable horizontal plane filter (keep all planes)"
    )
    stairs_group.add_argument(
        "--no-height-merge",
        action="store_true",
        dest="no_height_merge",
        help="Disable height-based plane merging"
    )
    stairs_group.add_argument(
        "--stairs-output",
        type=str,
        default="stairs_results.json",
        help="Output JSON file for stairs mode (default: stairs_results.json)"
    )
    stairs_group.add_argument(
        "--export-mesh",
        type=str,
        default=None,
        metavar="FILE",
        help="Export stair plane meshes to PLY/OBJ file (e.g., stairs_planes.ply)"
    )

    return parser.parse_args()


def build_effective_config(args) -> Tuple[SensorProfile, dict]:
    """
    Build effective configuration by merging profile defaults with CLI overrides.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (effective SensorProfile, additional config dict)
    """
    # Start with default or specified profile
    if args.sensor_profile:
        profile = SENSOR_PROFILES[args.sensor_profile]
        print(f"Using sensor profile: {profile.name}")
    else:
        profile = SENSOR_PROFILES["default"]

    # Create a mutable copy of profile values
    config = {
        "voxel_size": profile.voxel_size,
        "r_min": profile.r_min,
        "r_max": profile.r_max,
        "r_step": profile.r_step,
        "min_points": profile.min_points,
        "plane_distance_threshold": profile.plane_distance_threshold,
        "cylinder_distance_threshold": profile.cylinder_distance_threshold,
    }

    # Apply CLI overrides
    # Handle both new and legacy voxel_size arguments
    if args.voxel_size is not None:
        config["voxel_size"] = args.voxel_size

    if args.roi_r_min is not None:
        config["r_min"] = args.roi_r_min
    elif args.roi_radius is not None:  # Legacy fallback
        config["r_min"] = args.roi_radius

    if args.roi_r_max is not None:
        config["r_max"] = args.roi_r_max

    if args.roi_r_step is not None:
        config["r_step"] = args.roi_r_step

    if args.roi_min_points is not None:
        config["min_points"] = args.roi_min_points

    if args.plane_threshold is not None:
        config["plane_distance_threshold"] = args.plane_threshold

    if args.cylinder_threshold is not None:
        config["cylinder_distance_threshold"] = args.cylinder_threshold

    # Handle legacy no_preprocess
    no_preprocess = args.no_preprocess or getattr(args, 'no_preprocess_legacy', False)

    extra_config = {
        "no_preprocess": no_preprocess,
        "no_adaptive_roi": args.no_adaptive_roi,
    }

    # Print effective configuration
    print("\nEffective configuration:")
    print(f"  Voxel size: {config['voxel_size']}")
    print(f"  ROI: r_min={config['r_min']}m, r_max={config['r_max']}m, "
          f"step={config['r_step']}m, min_points={config['min_points']}")
    print(f"  RANSAC thresholds: plane={config['plane_distance_threshold']}, "
          f"cylinder={config['cylinder_distance_threshold']}")
    if extra_config["no_adaptive_roi"]:
        print(f"  Adaptive ROI: DISABLED (fixed radius: {config['r_min']}m)")
    else:
        print("  Adaptive ROI: ENABLED")

    # Create effective profile
    effective_profile = SensorProfile(
        name=f"{profile.name} (with overrides)" if args.sensor_profile else "Custom",
        voxel_size=config["voxel_size"],
        r_min=config["r_min"],
        r_max=config["r_max"],
        r_step=config["r_step"],
        min_points=config["min_points"],
        plane_distance_threshold=config["plane_distance_threshold"],
        cylinder_distance_threshold=config["cylinder_distance_threshold"],
    )

    return effective_profile, extra_config


def resolve_input_file(args) -> str:
    """
    Resolve the input file path from args or file dialog.

    Args:
        args: Parsed command line arguments.

    Returns:
        Path to the input file.

    Raises:
        SystemExit: If no file is specified/selected or required conditions not met.
    """
    # Case 1: --input is specified
    if args.input is not None:
        return args.input

    # Case 2: --no-gui is specified but --input is not
    if args.no_gui:
        print("Error: --no-gui requires --input to be specified.", file=sys.stderr)
        sys.exit(1)

    # Case 3: Try to open file dialog
    if not _check_tkinter():
        print(
            "Warning: tkinter is not available for file dialog.\n"
            "Please specify the input file with --input option.",
            file=sys.stderr
        )
        sys.exit(1)

    print("No input file specified. Opening file dialog...")
    filepath = select_file_with_dialog()

    if filepath is None:
        print("No file selected. Exiting.", file=sys.stderr)
        sys.exit(1)

    return filepath


def prompt_primitive_type() -> str:
    """Prompt user to select primitive type."""
    while True:
        print("\nSelect primitive type to fit:")
        print("  [p] Plane")
        print("  [c] Cylinder")
        print("  [q] Quit")
        choice = input("Enter choice: ").strip().lower()
        if choice in ['p', 'c', 'q']:
            return choice
        print("Invalid choice, please try again.")


def run_stairs_mode(
    pcd: o3d.geometry.PointCloud,
    args,
    profile: SensorProfile,
    extra_config: dict
):
    """
    Run stairs mode: extract multiple horizontal planes from ROI.

    Args:
        pcd: Preprocessed point cloud
        args: Parsed command line arguments
        profile: Effective sensor profile
        extra_config: Additional configuration
    """
    print("\n" + "=" * 60)
    print("  STAIRS MODE - Multi-plane Extraction")
    print("=" * 60)
    print("Parameters:")
    print(f"  Max planes: {args.max_planes}")
    print(f"  Min inliers: {args.min_inliers}")
    print(f"  Plane threshold: {profile.plane_distance_threshold}m")
    print(f"  RANSAC: n={args.stairs_ransac_n}, iters={args.stairs_num_iterations}")
    print(f"  Max tilt: {args.max_tilt}°")
    print(f"  Height epsilon: {args.height_eps}m")
    print(f"  Horizontal filter: {'OFF' if args.no_horizontal_filter else 'ON'}")
    print(f"  Height merge: {'OFF' if args.no_height_merge else 'ON'}")

    roi_selector = ROISelector(pcd)

    # Show point cloud first
    print("\n=== Point Cloud Viewer ===")
    print("Close the viewer window to proceed with ROI selection")
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud (Stairs Mode)")

    # Select ROI (adaptive or fixed)
    if extra_config["no_adaptive_roi"]:
        print(f"\nSelecting ROI (fixed radius: {profile.r_min}m)...")
        roi_pcd = roi_selector.select_roi_picking(radius=profile.r_min)
        if roi_pcd is None or roi_pcd.is_empty():
            print("ROI selection cancelled or empty")
            return
    else:
        roi_result = roi_selector.select_roi_adaptive(
            r_min=profile.r_min,
            r_max=profile.r_max,
            r_step=profile.r_step,
            min_points=profile.min_points
        )

        if roi_result.roi_pcd is None or roi_result.point_count == 0:
            print(roi_result.message)
            return

        roi_pcd = roi_result.roi_pcd

        if not roi_result.success:
            print("\nWARNING: ROI has fewer points than required.")
            proceed = input("Proceed with stair extraction anyway? [y/n]: ").strip().lower()
            if proceed != 'y':
                print("Stairs mode cancelled.")
                return

    # Get points from ROI
    points = np.asarray(roi_pcd.points)
    print(f"\nROI contains {len(points)} points")

    # Extract stair planes
    planes = extract_stair_planes(
        points,
        max_planes=args.max_planes,
        min_inliers=args.min_inliers,
        distance_threshold=profile.plane_distance_threshold,
        ransac_n=args.stairs_ransac_n,
        num_iterations=args.stairs_num_iterations,
        max_tilt_deg=args.max_tilt,
        height_eps=args.height_eps,
        horizontal_only=not args.no_horizontal_filter,
        merge_by_height=not args.no_height_merge
    )

    if len(planes) == 0:
        print("\nNo stair planes detected!")
        return

    # Visualize results
    visualize_stair_planes(roi_pcd, planes)

    # Save results to JSON
    save_stairs_results(planes, args.stairs_output)

    # Export mesh if requested
    if args.export_mesh:
        export_stair_planes_mesh(planes, points, args.export_mesh)

    # Summary
    print("\n" + "=" * 60)
    print("  STAIRS MODE COMPLETE")
    print("=" * 60)
    print(f"Detected {len(planes)} stair planes:")
    for i, p in enumerate(planes):
        print(f"  [{i}] height={p.height:.3f}m, inliers={p.inlier_count}")

    if len(planes) >= 2:
        heights = [p.height for p in planes]
        height_diffs = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        avg_step = np.mean(height_diffs) if height_diffs else 0
        print(f"\nAverage step height: {avg_step:.3f}m ({avg_step*100:.1f}cm)")

    print(f"\nResults saved to: {args.stairs_output}")
    if args.export_mesh:
        print(f"Mesh exported to: {args.export_mesh}")


def main():
    """Main entry point."""
    args = parse_args()

    # Build effective configuration from profile and CLI overrides
    profile, extra_config = build_effective_config(args)

    # Resolve input file (from args or file dialog)
    input_file = resolve_input_file(args)

    # Load point cloud
    print(f"\nLoading point cloud from {input_file}...")
    pcd = load_point_cloud(input_file)

    # Preprocess
    if not extra_config["no_preprocess"]:
        print("\nPreprocessing point cloud...")
        pcd = preprocess_point_cloud(pcd, voxel_size=profile.voxel_size)

    # Check if stairs mode
    if args.stairs_mode:
        run_stairs_mode(pcd, args, profile, extra_config)
        return

    # Load existing results
    results = load_results(args.output)
    print(f"\nLoaded {len(results['planes'])} planes and {len(results['cylinders'])} cylinders from previous session")

    # Main interaction loop
    roi_selector = ROISelector(pcd)

    while True:
        # Show current point cloud
        print("\n=== Point Cloud Viewer ===")
        print("Close the viewer window to proceed with ROI selection")
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

        # Select primitive type
        primitive_type = prompt_primitive_type()
        if primitive_type == 'q':
            break

        # Select ROI (adaptive or fixed)
        if extra_config["no_adaptive_roi"]:
            # Fixed radius mode (legacy behavior)
            print(f"\nSelecting ROI (fixed radius: {profile.r_min}m)...")
            roi_pcd = roi_selector.select_roi_picking(radius=profile.r_min)
            roi_result = None
            if roi_pcd is None or roi_pcd.is_empty():
                print("ROI selection cancelled or empty")
                continue
        else:
            # Adaptive radius mode
            roi_result = roi_selector.select_roi_adaptive(
                r_min=profile.r_min,
                r_max=profile.r_max,
                r_step=profile.r_step,
                min_points=profile.min_points
            )

            if roi_result.roi_pcd is None or roi_result.point_count == 0:
                print(roi_result.message)
                continue

            roi_pcd = roi_result.roi_pcd

            # Warn and optionally skip if ROI is too sparse
            if not roi_result.success:
                print("\nWARNING: ROI has fewer points than required.")
                proceed = input("Proceed with fitting anyway? [y/n]: ").strip().lower()
                if proceed != 'y':
                    print("Skipping fitting for this ROI.")
                    continue

        # Get points and normals from ROI
        points = np.asarray(roi_pcd.points)
        normals = np.asarray(roi_pcd.normals) if roi_pcd.has_normals() else None

        # Fit primitive
        if primitive_type == 'p':
            print("\nFitting plane...")
            plane = fit_plane(
                points,
                distance_threshold=profile.plane_distance_threshold
            )
            if plane is not None:
                print(
                    "Plane fit OK | normal="
                    f"{np.round(plane.normal, 4).tolist()}, point="
                    f"{np.round(plane.point, 4).tolist()}, inliers={plane.inlier_count}"
                )
                visualize_plane_fit(pcd, plane, roi_pcd)
                results = append_plane_result(results, plane)
                save_results(results, args.output)
            else:
                print("Plane fitting failed")

        elif primitive_type == 'c':
            print("\nFitting cylinder...")
            cylinder = fit_cylinder(
                points,
                normals,
                distance_threshold=profile.cylinder_distance_threshold
            )
            if cylinder is not None:
                print(
                    "Cylinder fit OK | axis_point="
                    f"{np.round(cylinder.axis_point, 4).tolist()}, axis_dir="
                    f"{np.round(cylinder.axis_direction, 4).tolist()}, radius="
                    f"{cylinder.radius:.4f}, length={cylinder.length:.4f}, "
                    f"inliers={cylinder.inlier_count}"
                )
                visualize_cylinder_fit(pcd, cylinder, roi_pcd)
                results = append_cylinder_result(results, cylinder)
                save_results(results, args.output)
            else:
                print("Cylinder fitting failed")

        # Ask if user wants to continue
        cont = input("\nFit another primitive? [y/n]: ").strip().lower()
        if cont != 'y':
            break

    print("\n=== Session Complete ===")
    print(f"Total planes fitted: {len(results['planes'])}")
    print(f"Total cylinders fitted: {len(results['cylinders'])}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
