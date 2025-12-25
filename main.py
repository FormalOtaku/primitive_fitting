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

from primitives import (
    PlaneParam, CylinderParam, extract_stair_planes,
    expand_plane_from_seed, expand_cylinder_from_seed,
    SeedExpandPlaneResult, SeedExpandCylinderResult,
    compute_cylinder_proxy_from_seed, finalize_cylinder_from_proxy,
    adaptive_seed_indices, compute_plane_residual_stats,
    auto_select_primitive
)

SEED_EXPAND_RESULT_VERSION = 4
CYLINDER_PROBE_RESULT_VERSION = 1
SESSION_RESULT_VERSION = 1


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
    center: Optional[np.ndarray] = None


class ROISelector:
    """Interactive ROI selection using Open3D visualizer."""

    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.selected_indices: Optional[np.ndarray] = None
        self.vis = None
        self._pcd_tree: Optional[o3d.geometry.KDTreeFlann] = None
        self.last_center: Optional[np.ndarray] = None

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
        self.last_center = center_point

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
            message=f"ROI selected with {len(roi_pcd.points)} points at radius {final_radius:.2f}m",
            center=center_point,
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
        self.last_center = center_point

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
        self.last_center = bbox.get_center()

        if roi_pcd.is_empty():
            print("No points in crop region")
            return None

        print(f"Cropped {len(roi_pcd.points)} points")
        return roi_pcd


# =============================================================================
# Visualization
# =============================================================================

def create_context_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    *,
    center: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Create a lightly downsampled context cloud for visualization.

    Optionally limits the context to points within the given radius
    of the specified center.
    """
    if pcd.is_empty():
        return None

    if radius is not None and radius > 0.0 and center is not None:
        center_arr = np.asarray(center, dtype=float).reshape(3)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return None
        mask = np.linalg.norm(pts - center_arr[None, :], axis=1) <= radius
        if not np.any(mask):
            return None

        base = o3d.geometry.PointCloud()
        base.points = o3d.utility.Vector3dVector(pts[mask])
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            if normals.shape == pts.shape:
                base.normals = o3d.utility.Vector3dVector(normals[mask])
    else:
        base = o3d.geometry.PointCloud(pcd)

    if voxel_size is not None and voxel_size > 0.0:
        base = base.voxel_down_sample(voxel_size)

    if base.is_empty():
        return None

    base.paint_uniform_color([0.85, 0.85, 0.85])  # Light gray context
    return base


def create_wireframe_sphere(
    center: Optional[np.ndarray],
    radius: float,
    color: Tuple[float, float, float] = (1.0, 0.85, 0.2),
    resolution: int = 12,
) -> Optional[o3d.geometry.LineSet]:
    """
    Create a wireframe sphere to visualize ROI/seed radius.
    """
    if center is None or radius is None or radius <= 0.0:
        return None

    try:
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=max(resolution, 5),
        )
        sphere_mesh.translate(np.asarray(center, dtype=float))
        sphere_lines = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
        sphere_lines.paint_uniform_color(color)
        return sphere_lines
    except Exception:
        return None


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


def _minimum_area_bounding_rectangle(
    hull_2d: np.ndarray
) -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """
    Compute the minimum-area bounding rectangle of a 2D convex hull.

    Returns:
        (rect_corners, area, (extent_u, extent_v)) or None if degenerate.
        rect_corners are in counter-clockwise order.
    """
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
        # Rotate by -theta so edge aligns with +x axis
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
            # Rotate corners back to original orientation
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
    """
    Create a plane patch mesh from inlier points.

    - Inliers are projected onto the plane
    - A 2D convex hull is computed in plane-local coordinates
    - Patch shape is either the hull polygon or the minimum-area oriented rectangle
    - The patch polygon is triangulated (fan triangulation)
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
    show_roi_points: bool = True,
    patch_shape: str = "hull",
    context_pcd: Optional[o3d.geometry.PointCloud] = None,
    roi_center: Optional[np.ndarray] = None,
    roi_radius: Optional[float] = None,
):
    """
    Visualize multiple stair planes with the ROI point cloud.

    Args:
        roi_pcd: Point cloud of the ROI
        planes: List of PlaneParam for detected planes
        show_roi_points: If True, show ROI points in light gray
        patch_shape: 'hull' or 'rect' for patch generation
        context_pcd: Downsampled background point cloud to show context
        roi_center: Center of ROI selection (for context radius visualization)
        roi_radius: ROI radius (drawn as wireframe sphere when provided)
    """
    geometries = []
    roi_points = np.asarray(roi_pcd.points)

    if context_pcd is not None and not context_pcd.is_empty():
        geometries.append(context_pcd)

    roi_sphere = create_wireframe_sphere(roi_center, roi_radius)
    if roi_sphere is not None:
        geometries.append(roi_sphere)

    # Add ROI point cloud in light gray
    if show_roi_points:
        roi_vis = o3d.geometry.PointCloud(roi_pcd)
        roi_vis.paint_uniform_color([0.6, 0.6, 0.6])  # Darker gray than context
        geometries.append(roi_vis)

    # Generate colors for planes
    colors = generate_plane_colors(len(planes))

    # Create mesh for each plane
    for i, (plane, color) in enumerate(zip(planes, colors)):
        try:
            mesh, metrics = create_plane_patch_mesh(
                plane, roi_points, color, padding=0.02, patch_shape=patch_shape
            )
            geometries.append(mesh)

            height = float(plane.height) if plane.height is not None else float(np.asarray(plane.point)[2])
            nz = float(np.clip(np.asarray(plane.normal, dtype=float)[2], -1.0, 1.0))
            tilt_deg = float(np.rad2deg(np.arccos(abs(nz))))
            small_patch = metrics["area"] < 0.01 or min(metrics["extent_u"], metrics["extent_v"]) < 0.10
            patch_info = f"patch={metrics.get('patch_shape', 'hull')}"
            if metrics.get("patch_shape") == "rect":
                rect_u = float(metrics.get("rect_extent_u", metrics.get("extent_u", 0.0)))
                rect_v = float(metrics.get("rect_extent_v", metrics.get("extent_v", 0.0)))
                patch_info += f" rect=({rect_u:.2f} x {rect_v:.2f})m area={metrics.get('area', 0.0):.3f}m^2"
            else:
                hull_area = float(metrics.get("hull_area", metrics.get("area", 0.0)))
                patch_info += (
                    f" area={metrics.get('area', 0.0):.3f}m^2"
                    f" hull_area={hull_area:.3f}m^2"
                )
            if metrics.get("patch_fallback_reason"):
                patch_info += f" fallback={metrics['patch_fallback_reason']}"
            print(
                f"  [{i:02d}] height={height:+.3f}m tilt={tilt_deg:4.1f}deg "
                f"inliers={plane.inlier_count:6d} extent=({metrics['extent_u']:.2f} x {metrics['extent_v']:.2f})m "
                f"{patch_info}{'  <SMALL>' if small_patch else ''}"
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
    legend = [
        "Colored patches: Detected stair planes",
        "Gray points: ROI (darker) and optional context (lighter)",
        "Wireframe sphere: ROI radius" if roi_sphere is not None else None,
    ]
    for line in legend:
        if line:
            print(f"  - {line}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Stair Planes ({len(planes)} detected)"
    )


def export_stair_planes_mesh(
    planes: List[PlaneParam],
    roi_points: np.ndarray,
    filepath: str,
    padding: float = 0.0,
    patch_shape: str = "hull",
):
    """
    Export all stair plane meshes to a single PLY file.

    Args:
        planes: List of PlaneParam
        roi_points: Points in the ROI
        filepath: Output file path (PLY or OBJ)
        padding: Padding around plane patches
        patch_shape: 'hull' or 'rect' for patch generation
    """
    if len(planes) == 0:
        print("No planes to export")
        return

    colors = generate_plane_colors(len(planes))
    meshes: List[o3d.geometry.TriangleMesh] = []

    for i, (plane, color) in enumerate(zip(planes, colors)):
        try:
            mesh, metrics = create_plane_patch_mesh(
                plane,
                roi_points,
                color,
                padding=padding,
                patch_shape=patch_shape,
            )
            meshes.append(mesh)
            height = float(plane.height) if plane.height is not None else float(np.asarray(plane.point)[2])
            small_patch = metrics["area"] < 0.01 or min(metrics["extent_u"], metrics["extent_v"]) < 0.10
            patch_info = f"patch={metrics.get('patch_shape', 'hull')}"
            if metrics.get("patch_shape") == "rect":
                rect_u = float(metrics.get("rect_extent_u", metrics.get("extent_u", 0.0)))
                rect_v = float(metrics.get("rect_extent_v", metrics.get("extent_v", 0.0)))
                patch_info += f" rect=({rect_u:.2f} x {rect_v:.2f})m area={metrics.get('area', 0.0):.3f}m^2"
            else:
                hull_area = float(metrics.get("hull_area", metrics.get("area", 0.0)))
                patch_info += (
                    f" area={metrics.get('area', 0.0):.3f}m^2"
                    f" hull_area={hull_area:.3f}m^2"
                )
            if metrics.get("patch_fallback_reason"):
                patch_info += f" fallback={metrics['patch_fallback_reason']}"
            print(
                f"  Export[{i:02d}] height={height:+.3f}m inliers={plane.inlier_count:6d} "
                f"extent=({metrics['extent_u']:.2f} x {metrics['extent_v']:.2f})m "
                f"{patch_info}{'  <SMALL>' if small_patch else ''}"
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
    roi_points: np.ndarray,
    filepath: str,
    *,
    patch_shape: str = "hull",
):
    """
    Save stair plane results to JSON file.

    Args:
        planes: List of PlaneParam detected as stair planes
        roi_points: Points in the ROI (for patch computation)
        filepath: Output JSON file path
        patch_shape: Requested patch shape
    """
    result = {
        "mode": "stairs",
        "version": 2,
        "plane_count": len(planes),
        "planes": []
    }

    for i, plane in enumerate(planes):
        plane_entry = {
            "id": i,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "height": plane.height,
            "inlier_count": plane.inlier_count
        }

        try:
            _, metrics = create_plane_patch_mesh(
                plane,
                roi_points,
                np.array([0.5, 0.5, 0.5]),
                padding=0.0,
                patch_shape=patch_shape,
            )
            plane_entry["patch_shape"] = metrics.get("patch_shape", patch_shape)
            if metrics.get("rect_corners_world") is not None:
                plane_entry["rect_corners_world"] = np.asarray(metrics["rect_corners_world"]).tolist()
            if metrics.get("patch_fallback_reason"):
                plane_entry["patch_fallback_reason"] = metrics["patch_fallback_reason"]
        except Exception as exc:
            plane_entry["patch_shape"] = patch_shape
            plane_entry["patch_error"] = str(exc)

        result["planes"].append(plane_entry)

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
        "--gui-app",
        action="store_true",
        dest="gui_app",
        help="Launch Open3D GUI app with side panel controls"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="fit_results.json",
        help="Output JSON file for results (default: fit_results.json)"
    )
    parser.add_argument(
        "--patch-shape",
        type=str,
        default="hull",
        choices=["hull", "rect"],
        help="Plane patch shape: convex hull or oriented rectangle (default: hull)"
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--show-context",
        action="store_true",
        help="Show downsampled original cloud as light-gray background in visualizations"
    )
    viz_group.add_argument(
        "--context-voxel",
        type=float,
        default=0.10,
        help="Voxel size for the background context cloud (default: 0.10m)"
    )
    viz_group.add_argument(
        "--context-radius",
        type=float,
        default=None,
        help="If set, only show context points within this radius of the ROI/seed center"
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

    # Session mode options
    parser.add_argument(
        "--session",
        action="store_true",
        dest="session_mode",
        help="Enable session mode: repeatedly click to extract and keep multiple primitives"
    )
    parser.add_argument(
        "--session-file",
        type=str,
        default="session.json",
        help="Session JSON file to read/write (default: session.json)"
    )
    parser.add_argument(
        "--session-reset",
        action="store_true",
        dest="session_reset",
        help="Start session from empty (ignore existing session-file)"
    )
    parser.add_argument(
        "--export-all",
        type=str,
        default=None,
        metavar="FILE",
        help="Export all primitives as a combined mesh (PLY/OBJ)"
    )
    parser.add_argument(
        "--force",
        type=str,
        default="auto",
        choices=["auto", "plane", "cylinder"],
        help="Force primitive type in session mode (default: auto)"
    )
    parser.add_argument(
        "--session-dedup",
        action="store_true",
        dest="session_dedup",
        default=True,
        help="Enable session deduplication (default: on)"
    )
    parser.add_argument(
        "--session-no-dedup",
        action="store_false",
        dest="session_dedup",
        help="Disable session deduplication"
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
        help="Export meshes to PLY/OBJ file (stairs mode or cylinder probe)"
    )

    # Seed-expand mode options
    seed_group = parser.add_argument_group("Seed-Expand Mode Options")
    seed_group.add_argument(
        "--seed-expand",
        action="store_true",
        dest="seed_expand",
        help="Enable seed-expand mode: fit primitive on seed, then expand to connected region"
    )
    seed_group.add_argument(
        "--seed-radius",
        type=float,
        default=0.3,
        help="Radius for initial seed region (default: 0.3m)"
    )
    seed_group.add_argument(
        "--max-expand-radius",
        type=float,
        default=5.0,
        help="Maximum radius from seed center for expansion (default: 5.0m)"
    )
    seed_group.add_argument(
        "--grow-radius",
        type=float,
        default=0.15,
        help="Neighbor radius for connectivity/growth (default: 0.15m)"
    )
    seed_group.add_argument(
        "--expand-method",
        type=str,
        default="component",
        choices=["component", "bfs"],
        help="Expansion method: 'component' (connected component) or 'bfs' (breadth-first search)"
    )
    seed_group.add_argument(
        "--max-refine-iters",
        type=int,
        default=3,
        help="Number of refit iterations after expansion (plane: up to N, cylinder: up to 2) (default: 3)"
    )
    seed_group.add_argument(
        "--max-expanded-points",
        type=int,
        default=200_000,
        help="Hard cap on expanded inlier points (default: 200000)"
    )
    seed_group.add_argument(
        "--max-frontier",
        type=int,
        default=200_000,
        help="Hard cap on BFS frontier size (default: 200000)"
    )
    seed_group.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Hard cap on BFS steps (default: 1000000)"
    )
    seed_group.add_argument(
        "--adaptive-plane-refine-th",
        action="store_true",
        dest="adaptive_plane_refine_th",
        help="Adapt plane distance threshold during refinement using median/MAD (default: off)"
    )
    seed_group.add_argument(
        "--adaptive-plane-refine-k",
        type=float,
        default=3.0,
        help="k for adaptive thresholding: median + k*sigma(MAD) (default: 3.0)"
    )
    seed_group.add_argument(
        "--adaptive-plane-refine-min-scale",
        type=float,
        default=0.5,
        help="Minimum adaptive threshold scale of plane threshold (default: 0.5)"
    )
    seed_group.add_argument(
        "--adaptive-plane-refine-max-scale",
        type=float,
        default=1.5,
        help="Maximum adaptive threshold scale of plane threshold (default: 1.5)"
    )
    seed_group.add_argument(
        "--normal-th",
        type=float,
        default=30.0,
        help="Normal angle threshold in degrees (default: 30.0, skipped if normals unavailable)"
    )
    seed_group.add_argument(
        "--seed-output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output JSON file for seed-expand results (default: seed_expand_results.json)"
    )
    seed_group.add_argument(
        "--export-inliers",
        type=str,
        default=None,
        metavar="FILE",
        help="Export expanded inlier points to PLY file (for debugging)"
    )

    session_seed_group = parser.add_argument_group("Session Seed Options")
    session_seed_group.add_argument(
        "--seed-radius-start",
        type=float,
        default=0.05,
        help="Session seed radius start (meters, default: 0.05)"
    )
    session_seed_group.add_argument(
        "--seed-radius-max",
        type=float,
        default=0.6,
        help="Session seed radius max (meters, default: 0.6)"
    )
    session_seed_group.add_argument(
        "--seed-radius-step",
        type=float,
        default=0.05,
        help="Session seed radius step (meters, default: 0.05)"
    )
    session_seed_group.add_argument(
        "--min-seed-points",
        type=int,
        default=80,
        help="Minimum points required in seed region (default: 80)"
    )

    # Cylinder probe mode options
    probe_group = parser.add_argument_group("Cylinder Probe Mode Options")
    probe_group.add_argument(
        "--cyl-probe",
        action="store_true",
        dest="cyl_probe",
        help="Enable interactive cylinder probe mode"
    )
    probe_group.add_argument(
        "--cyl-probe-seed-start",
        type=float,
        default=0.05,
        help="Initial seed radius for probe (meters, default: 0.05)"
    )
    probe_group.add_argument(
        "--cyl-probe-seed-max",
        type=float,
        default=0.5,
        help="Maximum seed radius for probe (meters, default: 0.5)"
    )
    probe_group.add_argument(
        "--cyl-probe-seed-step",
        type=float,
        default=0.05,
        help="Seed radius growth step (meters, default: 0.05)"
    )
    probe_group.add_argument(
        "--cyl-probe-min-seed-points",
        type=int,
        default=80,
        help="Minimum points required in seed for probe (default: 80)"
    )
    probe_group.add_argument(
        "--cyl-probe-surface-th",
        type=float,
        default=None,
        help="Surface distance threshold for probe (default: use cylinder threshold)"
    )
    probe_group.add_argument(
        "--cyl-probe-cap-margin",
        type=float,
        default=0.05,
        help="End-cap margin for probe selection (meters, default: 0.05)"
    )
    probe_group.add_argument(
        "--cyl-probe-refine-iters",
        type=int,
        default=2,
        help="Refinement iterations for probe (default: 2)"
    )
    probe_group.add_argument(
        "--cyl-probe-axis-refit",
        action="store_true",
        dest="cyl_probe_axis_refit",
        help="Allow axis direction refit during probe finalization (default: off, keeps proxy/user axis)"
    )
    probe_group.add_argument(
        "--cyl-probe-output",
        type=str,
        default="cyl_probe_results.json",
        help="Output JSON file for cylinder probe results (default: cyl_probe_results.json)"
    )
    probe_group.add_argument(
        "--cyl-probe-axis-snap-deg",
        type=float,
        default=0.0,
        dest="cyl_probe_axis_snap_deg",
        help="Snap axis to vertical if within N degrees (default: 0 = disabled)"
    )
    probe_group.add_argument(
        "--cyl-probe-axis-reg-weight",
        type=float,
        default=0.02,
        dest="cyl_probe_axis_reg_weight",
        help="Axis regularization weight (penalize deviation from reference, default: 0.02)"
    )
    probe_group.add_argument(
        "--cyl-probe-no-recompute-length",
        action="store_true",
        dest="cyl_probe_no_recompute_length",
        help="Disable length recomputation from final inliers"
    )
    probe_group.add_argument(
        "--cyl-probe-diagnostics-dir",
        type=str,
        default=None,
        dest="cyl_probe_diagnostics_dir",
        help="Directory to export debug PLY files (seed, candidates, inliers)"
    )

    # Cylinder prior options
    cyl_prior_group = parser.add_argument_group("Cylinder Prior Options")
    cyl_prior_group.add_argument(
        "--cylinder-init-axis-point",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Initial cylinder axis point (use with --cylinder-init-axis-dir and --cylinder-init-radius)"
    )
    cyl_prior_group.add_argument(
        "--cylinder-init-axis-dir",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Initial cylinder axis direction (use with --cylinder-init-axis-point and --cylinder-init-radius)"
    )
    cyl_prior_group.add_argument(
        "--cylinder-init-radius",
        type=float,
        default=None,
        help="Initial cylinder radius in meters"
    )
    cyl_prior_group.add_argument(
        "--cylinder-init-length",
        type=float,
        default=None,
        help="Initial cylinder length in meters (optional; used if expansion fails)"
    )

    return parser.parse_args()


def resolve_initial_cylinder(args) -> Optional[CylinderParam]:
    """Resolve an optional initial cylinder prior from CLI args."""
    axis_point = args.cylinder_init_axis_point
    axis_dir = args.cylinder_init_axis_dir
    radius = args.cylinder_init_radius
    length = args.cylinder_init_length

    if axis_point is None and axis_dir is None and radius is None and length is None:
        return None

    missing = []
    if axis_point is None:
        missing.append("--cylinder-init-axis-point")
    if axis_dir is None:
        missing.append("--cylinder-init-axis-dir")
    if radius is None:
        missing.append("--cylinder-init-radius")
    if missing:
        raise ValueError("Cylinder prior requires: " + ", ".join(missing))

    axis_point = np.asarray(axis_point, dtype=float).reshape(-1)
    axis_dir = np.asarray(axis_dir, dtype=float).reshape(-1)
    if axis_point.size != 3 or axis_dir.size != 3:
        raise ValueError("Cylinder prior axis point/dir must be 3D vectors")

    axis_norm = np.linalg.norm(axis_dir)
    if not np.isfinite(axis_norm) or axis_norm < 1e-12:
        raise ValueError("Cylinder prior axis direction is invalid")

    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Cylinder prior radius must be positive")

    length_val = float(length) if length is not None else 0.0
    if not np.isfinite(length_val):
        length_val = 0.0

    return CylinderParam(
        axis_point=axis_point,
        axis_direction=axis_dir,
        radius=radius,
        length=length_val,
        inlier_count=0,
        inlier_indices=None,
    )


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


def visualize_seed_expand_plane(
    pcd: o3d.geometry.PointCloud,
    result: SeedExpandPlaneResult,
    seed_center: np.ndarray,
    all_points: np.ndarray,
    *,
    patch_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    patch_shape: str = "hull",
    context_pcd: Optional[o3d.geometry.PointCloud] = None,
    roi_radius: Optional[float] = None,
):
    """Visualize seed-expand plane result."""
    geometries = []

    # Background/context point cloud
    if context_pcd is not None and not context_pcd.is_empty():
        geometries.append(context_pcd)
    else:
        pcd_gray = o3d.geometry.PointCloud(pcd)
        pcd_gray.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_gray)

    # Seed region points (if available)
    if result.seed_inlier_indices is not None and len(result.seed_inlier_indices) > 0:
        seed_pcd = o3d.geometry.PointCloud()
        seed_pcd.points = o3d.utility.Vector3dVector(all_points[result.seed_inlier_indices])
        seed_pcd.paint_uniform_color([0.0, 0.5, 1.0])  # Blue for seed region
        geometries.append(seed_pcd)

    # Expanded inlier points in red
    if result.expanded_inlier_indices is not None and len(result.expanded_inlier_indices) > 0:
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(all_points[result.expanded_inlier_indices])
        inlier_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for expanded inliers
        geometries.append(inlier_pcd)

    # Plane patch mesh in green
    if result.plane is not None and result.expanded_inlier_indices is not None:
        try:
            if patch_mesh is None:
                patch_mesh, _ = create_plane_patch_mesh(
                    result.plane,
                    all_points,
                    np.array([0.0, 0.8, 0.0]),  # Green
                    padding=0.02,
                    patch_shape=patch_shape,
                )
            geometries.append(patch_mesh)
        except Exception as e:
            print(f"  Warning: Could not create plane patch mesh: {e}")

    roi_sphere = create_wireframe_sphere(seed_center, roi_radius)
    if roi_sphere is not None:
        geometries.append(roi_sphere)

    # Seed center marker
    seed_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    seed_marker.translate(seed_center)
    seed_marker.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    seed_marker.compute_vertex_normals()
    geometries.append(seed_marker)

    print("\nVisualization:")
    print("  - Gray: Context/original point cloud")
    print("  - Blue: Seed region points")
    print("  - Red: Expanded inlier points")
    print("  - Green patch: Fitted plane")
    print("  - Yellow sphere: Seed center")
    if roi_sphere is not None:
        print("  - Orange wireframe: Seed radius")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Seed-Expand Plane ({result.expanded_inlier_count} inliers)"
    )


def visualize_seed_expand_cylinder(
    pcd: o3d.geometry.PointCloud,
    result: SeedExpandCylinderResult,
    seed_center: np.ndarray,
    all_points: np.ndarray,
    *,
    context_pcd: Optional[o3d.geometry.PointCloud] = None,
    roi_radius: Optional[float] = None,
):
    """Visualize seed-expand cylinder result."""
    geometries = []

    # Background/context point cloud
    if context_pcd is not None and not context_pcd.is_empty():
        geometries.append(context_pcd)
    else:
        pcd_gray = o3d.geometry.PointCloud(pcd)
        pcd_gray.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_gray)

    # Seed region points
    if result.seed_inlier_indices is not None and len(result.seed_inlier_indices) > 0:
        seed_pcd = o3d.geometry.PointCloud()
        seed_pcd.points = o3d.utility.Vector3dVector(all_points[result.seed_inlier_indices])
        seed_pcd.paint_uniform_color([0.0, 0.5, 1.0])  # Blue for seed region
        geometries.append(seed_pcd)

    # Expanded inlier points in red
    if result.expanded_inlier_indices is not None and len(result.expanded_inlier_indices) > 0:
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(all_points[result.expanded_inlier_indices])
        inlier_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for expanded inliers
        geometries.append(inlier_pcd)

    # Cylinder mesh in blue
    if result.cylinder is not None:
        cyl = result.cylinder
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cyl.radius,
            height=cyl.length,
            resolution=20,
            split=4
        )

        # Rotate to align with axis direction
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, cyl.axis_direction)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, cyl.axis_direction), -1, 1))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            cylinder_mesh.rotate(R, center=[0, 0, 0])

        cylinder_mesh.translate(cyl.axis_point)
        cylinder_mesh.paint_uniform_color([0.0, 0.0, 0.8])  # Blue
        cylinder_mesh.compute_vertex_normals()
        geometries.append(cylinder_mesh)

        # Axis line in yellow
        axis_start = cyl.axis_point - cyl.axis_direction * cyl.length / 2
        axis_end = cyl.axis_point + cyl.axis_direction * cyl.length / 2
        axis_line = o3d.geometry.LineSet()
        axis_line.points = o3d.utility.Vector3dVector([axis_start, axis_end])
        axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        axis_line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])
        geometries.append(axis_line)

    roi_sphere = create_wireframe_sphere(seed_center, roi_radius)
    if roi_sphere is not None:
        geometries.append(roi_sphere)

    # Seed center marker
    seed_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    seed_marker.translate(seed_center)
    seed_marker.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    seed_marker.compute_vertex_normals()
    geometries.append(seed_marker)

    print("\nVisualization:")
    print("  - Gray: Context/original point cloud")
    print("  - Blue points: Seed region")
    print("  - Red points: Expanded inliers")
    print("  - Blue cylinder: Fitted cylinder mesh")
    print("  - Yellow line: Cylinder axis")
    if roi_sphere is not None:
        print("  - Orange wireframe: Seed radius")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Seed-Expand Cylinder ({result.expanded_inlier_count} inliers)"
    )


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector or zeros if invalid."""
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.size != 3:
        return np.zeros(3, dtype=float)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros(3, dtype=float)
    return vec / norm


def plane_tilt_deg(normal: np.ndarray) -> float:
    normal = unit_vector(normal)
    nz = float(np.clip(abs(normal[2]), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(nz)))


def cylinder_endpoints(axis_point: np.ndarray, axis_dir: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    axis_point = np.asarray(axis_point, dtype=float).reshape(-1)
    axis_dir = unit_vector(axis_dir)
    if axis_point.size != 3:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    half = 0.5 * float(length)
    return axis_point - axis_dir * half, axis_point + axis_dir * half


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vec around axis by angle_deg using Rodrigues' formula."""
    vec = np.asarray(vec, dtype=float).reshape(-1)
    axis = unit_vector(axis)
    if vec.size != 3 or axis.size != 3:
        return vec
    angle = np.deg2rad(float(angle_deg))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return (
        vec * cos_a
        + np.cross(axis, vec) * sin_a
        + axis * float(np.dot(axis, vec)) * (1.0 - cos_a)
    )


def create_cylinder_mesh(
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    length: float,
    color: np.ndarray,
    *,
    resolution: int = 40,
    split: int = 4,
) -> o3d.geometry.TriangleMesh:
    """Create a cylinder mesh oriented to axis_dir."""
    radius = float(max(radius, 1e-4))
    length = float(max(length, 1e-4))
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=resolution,
        split=split,
    )
    axis_dir = unit_vector(axis_dir)
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, axis_dir), -1.0, 1.0))
    if dot < 1.0 - 1e-8:
        if dot < -1.0 + 1e-8:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0.0, 0.0])
        else:
            rot_axis = np.cross(z_axis, axis_dir)
            rot_axis = rot_axis / np.maximum(np.linalg.norm(rot_axis), 1e-8)
            angle = float(np.arccos(dot))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        mesh.rotate(R, center=[0, 0, 0])
    mesh.translate(np.asarray(axis_point, dtype=float))
    mesh.paint_uniform_color(np.asarray(color, dtype=float))
    mesh.compute_vertex_normals()
    return mesh


def create_axis_line(
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    length: float,
    color: np.ndarray,
) -> o3d.geometry.LineSet:
    """Create a LineSet for a cylinder axis."""
    axis_dir = unit_vector(axis_dir)
    start, end = cylinder_endpoints(axis_point, axis_dir, length)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([start, end])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([np.asarray(color, dtype=float)])
    return line


def colorize_point_cloud_by_height(
    pcd: o3d.geometry.PointCloud,
    *,
    axis: int = 2,
) -> o3d.geometry.PointCloud:
    """Apply a simple height colormap to a point cloud copy."""
    colored = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(colored.points)
    if pts.size == 0:
        return colored
    z = pts[:, axis]
    z_min = float(np.percentile(z, 5))
    z_max = float(np.percentile(z, 95))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or abs(z_max - z_min) < 1e-9:
        colored.paint_uniform_color([0.7, 0.7, 0.7])
        return colored
    t = (z - z_min) / (z_max - z_min)
    t = np.clip(t, 0.0, 1.0)
    colors = np.column_stack([t, 0.2 + 0.6 * (1.0 - t), 1.0 - t])
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


def build_mesh_from_polygon(
    corners_world: np.ndarray,
    color: np.ndarray,
) -> Optional[o3d.geometry.TriangleMesh]:
    corners = np.asarray(corners_world, dtype=float)
    if corners.ndim != 2 or corners.shape[1] != 3 or len(corners) < 3:
        return None
    triangles = np.array([[0, i, i + 1] for i in range(1, len(corners) - 1)], dtype=int)
    if len(triangles) == 0:
        return None
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(corners), 1)))
    mesh.compute_vertex_normals()
    return mesh


def save_seed_expand_result(
    result,
    primitive_type: str,
    filepath: str,
    seed_center: np.ndarray,
    params: dict,
    patch_metrics: Optional[Dict[str, object]] = None,
):
    """Save seed-expand result to JSON."""
    data = {
        "mode": "seed_expand",
        "version": SEED_EXPAND_RESULT_VERSION,
        "primitive_type": primitive_type,
        "seed_center": seed_center.tolist(),
        "success": result.success,
        "message": result.message,
        "expanded_inlier_count": result.expanded_inlier_count,
        "seed_point_count": int(getattr(result, "seed_point_count", 0)),
        "candidate_count": int(getattr(result, "candidate_count", 0)),
        "stopped_early": bool(getattr(result, "stopped_early", False)),
        "stop_reason": str(getattr(result, "stop_reason", "")),
        "params": params,
        "seed": {
            "center": seed_center.tolist(),
            "radius": float(params.get("seed_radius", 0.0)),
            "point_count": int(getattr(result, "seed_point_count", 0)),
        },
        "stats": {
            "steps": int(getattr(result, "steps", 0)),
            "max_frontier_size": int(getattr(result, "max_frontier_size", 0)),
            "residual_median": float(getattr(result, "residual_median", 0.0)),
            "residual_mad": float(getattr(result, "residual_mad", 0.0)),
            "residual_p90": float(getattr(result, "residual_p90", 0.0)),
            "residual_p95": float(getattr(result, "residual_p95", 0.0)),
        },
    }

    if primitive_type == "plane" and result.plane is not None:
        tilt = plane_tilt_deg(result.plane.normal)
        data["plane"] = {
            "normal": result.plane.normal.tolist(),
            "point": result.plane.point.tolist(),
            "tilt_deg": tilt,
            "inlier_count": result.plane.inlier_count,
            "area": float(result.area),
            "extent_u": float(result.extent_u),
            "extent_v": float(result.extent_v),
        }
        data["area"] = float(result.area)
        data["extent_u"] = float(result.extent_u)
        data["extent_v"] = float(result.extent_v)
        if patch_metrics is None:
            patch_metrics = {
                "patch_shape": "unknown",
                "corners_world": [],
                "patch_fallback_reason": "patch_metrics_missing",
            }
        plane_patch_shape = patch_metrics.get("patch_shape") if isinstance(patch_metrics, dict) else None
        if plane_patch_shape:
            data["plane"]["patch_shape"] = plane_patch_shape
        if isinstance(patch_metrics, dict) and patch_metrics.get("corners_world") is not None:
            data["plane"]["corners_world"] = np.asarray(
                patch_metrics["corners_world"]
            ).tolist()
        if isinstance(patch_metrics, dict) and patch_metrics.get("rect_corners_world") is not None:
            data["plane"]["rect_corners_world"] = np.asarray(
                patch_metrics["rect_corners_world"]
            ).tolist()
        if isinstance(patch_metrics, dict) and patch_metrics.get("patch_fallback_reason"):
            data["plane"]["patch_fallback_reason"] = patch_metrics["patch_fallback_reason"]
        if isinstance(patch_metrics, dict) and patch_metrics.get("hull_area") is not None:
            data["plane"]["hull_area"] = float(patch_metrics["hull_area"])
        if isinstance(patch_metrics, dict) and patch_metrics.get("extent_u") is not None:
            data["plane"]["patch_extent_u"] = float(patch_metrics["extent_u"])
        if isinstance(patch_metrics, dict) and patch_metrics.get("extent_v") is not None:
            data["plane"]["patch_extent_v"] = float(patch_metrics["extent_v"])
    elif primitive_type == "cylinder" and result.cylinder is not None:
        axis_dir = unit_vector(result.cylinder.axis_direction)
        end0, end1 = cylinder_endpoints(
            result.cylinder.axis_point,
            axis_dir,
            result.cylinder.length,
        )
        data["cylinder"] = {
            "axis_point": result.cylinder.axis_point.tolist(),
            "axis_direction": axis_dir.tolist(),
            "radius": result.cylinder.radius,
            "diameter": float(result.cylinder.radius * 2.0),
            "length": result.cylinder.length,
            "end_center_0": end0.tolist(),
            "end_center_1": end1.tolist(),
            "inlier_count": result.cylinder.inlier_count,
            "residual_median": float(getattr(result, "residual_median", 0.0)),
            "residual_mad": float(getattr(result, "residual_mad", 0.0)),
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Seed-expand results saved to {filepath}")


def save_cylinder_probe_result(
    result,
    proxy: Optional[CylinderParam],
    final: Optional[CylinderParam],
    filepath: str,
    seed_center: np.ndarray,
    params: dict,
    control_log: dict,
):
    """Save cylinder probe result to JSON."""
    data = {
        "mode": "cylinder_probe",
        "version": CYLINDER_PROBE_RESULT_VERSION,
        "seed_center": np.asarray(seed_center, dtype=float).tolist(),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "params": params,
        "controls": control_log,
        "stats": {
            "candidate_count": int(getattr(result, "candidate_count", 0)),
            "inlier_count": int(getattr(result, "inlier_count", 0)),
            "residual_median": float(getattr(result, "residual_median", 0.0)),
            "residual_mad": float(getattr(result, "residual_mad", 0.0)),
            "stopped_early": bool(getattr(result, "stopped_early", False)),
            "stop_reason": str(getattr(result, "stop_reason", "")),
            "steps": int(getattr(result, "steps", 0)),
            "max_frontier_size": int(getattr(result, "max_frontier_size", 0)),
        },
        "seed": {
            "center": np.asarray(seed_center, dtype=float).tolist(),
            "radius": float(params.get("seed_radius_used", 0.0)),
            "point_count": int(params.get("seed_point_count", 0)),
        },
    }

    if proxy is not None:
        proxy_axis = unit_vector(proxy.axis_direction)
        p0, p1 = cylinder_endpoints(proxy.axis_point, proxy_axis, proxy.length)
        data["proxy"] = {
            "axis_point": np.asarray(proxy.axis_point, dtype=float).tolist(),
            "axis_direction": proxy_axis.tolist(),
            "radius": float(proxy.radius),
            "diameter": float(proxy.radius * 2.0),
            "length": float(proxy.length),
            "end_center_0": p0.tolist(),
            "end_center_1": p1.tolist(),
        }

    if final is not None:
        final_axis = unit_vector(final.axis_direction)
        f0, f1 = cylinder_endpoints(final.axis_point, final_axis, final.length)
        data["final"] = {
            "axis_point": np.asarray(final.axis_point, dtype=float).tolist(),
            "axis_direction": final_axis.tolist(),
            "radius": float(final.radius),
            "diameter": float(final.radius * 2.0),
            "length": float(final.length),
            "end_center_0": f0.tolist(),
            "end_center_1": f1.tolist(),
            "inlier_count": int(final.inlier_count),
            "residual_median": float(getattr(result, "residual_median", 0.0)),
            "residual_mad": float(getattr(result, "residual_mad", 0.0)),
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Cylinder probe results saved to {filepath}")


def load_session(filepath: str) -> dict:
    """Load session JSON or initialize new."""
    path = Path(filepath)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        if "objects" not in data:
            data["objects"] = []
        if "version" not in data:
            data["version"] = SESSION_RESULT_VERSION
        return data
    return {"version": SESSION_RESULT_VERSION, "objects": []}


def save_session(filepath: str, data: dict) -> None:
    """Save session JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Session saved to {filepath}")


def mesh_from_session_object(obj: dict) -> Optional[o3d.geometry.TriangleMesh]:
    """Rebuild mesh from session object."""
    obj_type = obj.get("type")
    color = np.asarray(obj.get("color", [0.2, 0.6, 0.9]), dtype=float)
    params = obj.get("params", {})
    if obj_type == "plane":
        corners = params.get("corners_world")
        if corners is None:
            return None
        return build_mesh_from_polygon(np.asarray(corners, dtype=float), color)
    if obj_type == "cylinder":
        return create_cylinder_mesh(
            params.get("axis_point", [0.0, 0.0, 0.0]),
            params.get("axis_direction", [0.0, 0.0, 1.0]),
            params.get("radius", 0.1),
            params.get("length", 0.1),
            color,
        )
    return None


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = unit_vector(a)
    b = unit_vector(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(abs(dot))))


def _axis_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d0 = unit_vector(d0)
    d1 = unit_vector(d1)
    cross = np.cross(d0, d1)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        diff = p1 - p0
        return float(np.linalg.norm(diff - np.dot(diff, d0) * d0))
    diff = p1 - p0
    return float(abs(np.dot(diff, cross)) / cross_norm)


def find_duplicate_cylinder(
    new_obj: dict,
    objects: list,
    *,
    angle_deg: float = 12.0,
    axis_dist: float = 0.15,
    radius_rel: float = 0.25,
    radius_abs: float = 0.03,
) -> Optional[int]:
    params = new_obj.get("params", {})
    axis_point = np.asarray(params.get("axis_point", [0.0, 0.0, 0.0]), dtype=float)
    axis_dir = np.asarray(params.get("axis_direction", [0.0, 0.0, 1.0]), dtype=float)
    radius = float(params.get("radius", 0.0))

    for idx, obj in enumerate(objects):
        if obj.get("type") != "cylinder":
            continue
        p = obj.get("params", {})
        op = np.asarray(p.get("axis_point", [0.0, 0.0, 0.0]), dtype=float)
        od = np.asarray(p.get("axis_direction", [0.0, 0.0, 1.0]), dtype=float)
        orad = float(p.get("radius", 0.0))
        if radius <= 0 or orad <= 0:
            continue
        if _axis_angle_deg(axis_dir, od) > angle_deg:
            continue
        if _axis_distance(axis_point, axis_dir, op, od) > axis_dist:
            continue
        if abs(radius - orad) > max(radius_abs, radius_rel * radius):
            continue
        return idx
    return None


def run_session_mode(
    pcd: o3d.geometry.PointCloud,
    args,
    profile: SensorProfile,
    extra_config: dict,
):
    """Run workspace session mode for one-click extraction."""
    print("\n" + "=" * 60)
    print("  SESSION MODE")
    print("=" * 60)
    print("Controls after extraction:")
    print("  [Enter] next seed | U undo | D delete last | S save | E export | Q quit")

    all_points = np.asarray(pcd.points)
    all_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    if args.session_reset:
        session = {"version": SESSION_RESULT_VERSION, "objects": []}
        print("Session reset: starting from empty.")
    else:
        session = load_session(args.session_file)
    meshes: List[o3d.geometry.TriangleMesh] = []
    for obj in session.get("objects", []):
        mesh = mesh_from_session_object(obj)
        if mesh is not None:
            meshes.append(mesh)

    context_radius = args.context_radius if args.context_radius and args.context_radius > 0.0 else None
    context_pcd = None
    if args.show_context:
        context_pcd = create_context_cloud(
            pcd,
            args.context_voxel,
            center=None,
            radius=context_radius,
        )
        if context_pcd is not None:
            context_pcd = colorize_point_cloud_by_height(context_pcd)

    rng = np.random.default_rng(1234)

    def _apply_render_options(vis_obj):
        opt = vis_obj.get_render_option()
        if opt is not None:
            opt.point_size = 2.0
            opt.background_color = np.array([0.03, 0.03, 0.03])

    def show_scene_for_pick():
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Session Pick (Shift+Click, close to confirm)")
        _apply_render_options(vis)
        pick_cloud = colorize_point_cloud_by_height(pcd)
        vis.add_geometry(pick_cloud)
        for mesh in meshes:
            vis.add_geometry(mesh)
        vis.run()
        picked = vis.get_picked_points()
        vis.destroy_window()
        pick_points = np.asarray(pick_cloud.points)
        if not picked:
            return None
        idx = picked[0]
        if idx < 0 or idx >= len(pick_points):
            return None
        return pick_points[idx]

    def show_scene():
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Session Workspace")
        _apply_render_options(vis)
        if context_pcd is not None and not context_pcd.is_empty():
            vis.add_geometry(context_pcd)
        else:
            base = colorize_point_cloud_by_height(pcd)
            vis.add_geometry(base)
        for mesh in meshes:
            vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()

    while True:
        seed_center = show_scene_for_pick()
        if seed_center is None:
            print("No point selected. Exiting session.")
            break

        print(f"\nSeed: {np.round(seed_center, 4).tolist()}")

        seed_indices, seed_radius = adaptive_seed_indices(
            all_points,
            seed_center,
            seed_radius_start=args.seed_radius_start,
            seed_radius_max=args.seed_radius_max,
            seed_radius_step=args.seed_radius_step,
            min_seed_points=args.min_seed_points,
        )
        print(f"  Seed radius used: {seed_radius:.3f} m ({len(seed_indices)} pts)")

        plane_result = None
        cylinder_result = None
        plane_mesh = None
        plane_metrics = None
        plane_quality = {}
        cylinder_quality = {}
        cyl_proxy = None
        plane_color = None

        surface_th = profile.cylinder_distance_threshold

        if args.force in ("auto", "plane"):
            plane_result = expand_plane_from_seed(
                all_points,
                seed_center,
                normals=all_normals,
                seed_radius=seed_radius,
                max_expand_radius=args.max_expand_radius,
                grow_radius=args.grow_radius,
                distance_threshold=profile.plane_distance_threshold,
                normal_threshold_deg=args.normal_th,
                expand_method=args.expand_method,
                max_refine_iters=args.max_refine_iters,
                adaptive_refine_threshold=args.adaptive_plane_refine_th,
                adaptive_refine_k=args.adaptive_plane_refine_k,
                adaptive_refine_min_scale=args.adaptive_plane_refine_min_scale,
                adaptive_refine_max_scale=args.adaptive_plane_refine_max_scale,
                max_expanded_points=args.max_expanded_points,
                max_frontier=args.max_frontier,
                max_steps=args.max_steps,
                verbose=False,
            )
            if plane_result.success and plane_result.plane is not None:
                plane_color = rng.uniform(0.2, 0.9, size=3)
                plane_indices = plane_result.expanded_inlier_indices
                if plane_indices is None and plane_result.plane.inlier_indices is not None:
                    plane_indices = plane_result.plane.inlier_indices
                if plane_indices is None:
                    plane_indices = np.empty((0,), dtype=int)
                plane_res_median, plane_res_mad = compute_plane_residual_stats(
                    all_points[np.asarray(plane_indices, dtype=int)],
                    plane_result.plane.point,
                    plane_result.plane.normal,
                )
                plane_quality = {
                    "inlier_count": int(plane_result.expanded_inlier_count),
                    "residual_median": float(plane_res_median),
                    "residual_mad": float(plane_res_mad),
                    "area": float(plane_result.area),
                    "extent_u": float(plane_result.extent_u),
                    "extent_v": float(plane_result.extent_v),
                    "tilt_deg": float(plane_tilt_deg(plane_result.plane.normal)),
                    "stop_reason": str(plane_result.stop_reason),
                }
                try:
                    plane_mesh, plane_metrics = create_plane_patch_mesh(
                        plane_result.plane,
                        all_points,
                        plane_color,
                        padding=0.02,
                        patch_shape=args.patch_shape,
                    )
                except Exception as exc:
                    plane_metrics = {
                        "patch_shape": args.patch_shape,
                        "corners_world": np.empty((0, 3), dtype=float),
                        "patch_fallback_reason": str(exc),
                        "area": float(plane_result.area),
                        "extent_u": float(plane_result.extent_u),
                        "extent_v": float(plane_result.extent_v),
                    }
            else:
                if plane_result is not None:
                    print(f"  Plane failed: {plane_result.message}")

        if args.force in ("auto", "cylinder"):
            proxy_init = compute_cylinder_proxy_from_seed(
                all_points,
                seed_center,
                normals=all_normals,
                seed_radius_start=args.seed_radius_start,
                seed_radius_max=args.seed_radius_max,
                seed_radius_step=args.seed_radius_step,
                min_seed_points=args.min_seed_points,
                circle_ransac_iters=200,
                circle_inlier_threshold=surface_th,
                length_margin=0.05,
            )
            if proxy_init.success and proxy_init.cylinder is not None:
                cyl_proxy = proxy_init.cylinder
                cylinder_result = finalize_cylinder_from_proxy(
                    all_points,
                    seed_center,
                    proxy_init.seed_indices,
                    cyl_proxy,
                    surface_threshold=surface_th,
                    cap_margin=0.05,
                    grow_radius=args.grow_radius,
                    max_expand_radius=args.max_expand_radius,
                    max_expanded_points=args.max_expanded_points,
                    max_frontier=args.max_frontier,
                    max_steps=args.max_steps,
                    normals=all_normals,
                    normal_angle_threshold_deg=args.normal_th,
                    refine_iters=min(3, max(1, int(args.max_refine_iters))),
                    circle_ransac_iters=200,
                    circle_inlier_threshold=surface_th,
                    allow_length_growth=True,
                    axis_regularization_weight=0.02,
                    recompute_length_from_inliers=True,
                )
                if cylinder_result.success and cylinder_result.final is not None:
                    cylinder_quality = {
                        "inlier_count": int(cylinder_result.inlier_count),
                        "residual_median": float(cylinder_result.residual_median),
                        "residual_mad": float(cylinder_result.residual_mad),
                        "stop_reason": str(cylinder_result.stop_reason),
                        "candidate_count": int(cylinder_result.candidate_count),
                    }
                else:
                    if cylinder_result is not None:
                        print(f"  Cylinder failed: {cylinder_result.message}")
            else:
                print(f"  Cylinder proxy init failed: {proxy_init.message}")

        chosen_type = args.force
        plane_score = -1.0
        cylinder_score = -1.0
        if args.force == "auto":
            select = auto_select_primitive(
                plane_result,
                cylinder_result,
                plane_threshold=profile.plane_distance_threshold,
                cylinder_threshold=profile.cylinder_distance_threshold,
            )
            chosen_type = select.chosen
            plane_score = select.plane_score
            cylinder_score = select.cylinder_score
            print(
                f"  Auto scores: plane={plane_score:.2f}, cylinder={cylinder_score:.2f} -> {chosen_type}"
            )

        if chosen_type == "none":
            print("  No valid primitive found. Continuing.")
            continue

        if chosen_type == "plane":
            if plane_result is None or not plane_result.success or plane_result.plane is None:
                print("  Plane extraction failed. Continuing.")
                continue
            color = plane_color if plane_color is not None else rng.uniform(0.2, 0.9, size=3)
            if plane_mesh is None and plane_metrics is not None:
                plane_mesh = build_mesh_from_polygon(
                    plane_metrics.get("corners_world", np.empty((0, 3))),
                    color,
                )
            if plane_mesh is None:
                print("  Plane mesh unavailable; skipping.")
                continue
            meshes.append(plane_mesh)

            corners_world = None
            if plane_metrics is not None and plane_metrics.get("corners_world") is not None:
                corners_world = np.asarray(plane_metrics["corners_world"], dtype=float).tolist()

            obj = {
                "id": len(session["objects"]),
                "type": "plane",
                "seed": {
                    "center": seed_center.tolist(),
                    "radius": float(seed_radius),
                    "point_count": int(len(seed_indices)),
                },
                "params": {
                    "normal": plane_result.plane.normal.tolist(),
                    "point": plane_result.plane.point.tolist(),
                    "tilt_deg": float(plane_tilt_deg(plane_result.plane.normal)),
                    "area": float(plane_result.area),
                    "extent_u": float(plane_result.extent_u),
                    "extent_v": float(plane_result.extent_v),
                    "patch_shape": plane_metrics.get("patch_shape") if plane_metrics else None,
                    "corners_world": corners_world,
                    "patch_fallback_reason": plane_metrics.get("patch_fallback_reason") if plane_metrics else None,
                },
                "quality": {
                    **plane_quality,
                    "score": float(plane_score if args.force == "auto" else plane_quality.get("inlier_count", 0)),
                },
                "stop_reason": str(plane_result.stop_reason),
                "color": color.tolist(),
            }
            session["objects"].append(obj)
        elif chosen_type == "cylinder":
            if cylinder_result is None or not cylinder_result.success or cylinder_result.final is None:
                print("  Cylinder extraction failed. Continuing.")
                continue
            final = cylinder_result.final
            axis_dir = unit_vector(final.axis_direction)
            end0, end1 = cylinder_endpoints(final.axis_point, axis_dir, final.length)
            color = rng.uniform(0.2, 0.9, size=3)
            cyl_mesh = create_cylinder_mesh(
                final.axis_point,
                axis_dir,
                final.radius,
                final.length,
                color,
            )
            obj = {
                "id": len(session["objects"]),
                "type": "cylinder",
                "seed": {
                    "center": seed_center.tolist(),
                    "radius": float(seed_radius),
                    "point_count": int(len(seed_indices)),
                },
                "params": {
                    "axis_point": final.axis_point.tolist(),
                    "axis_direction": axis_dir.tolist(),
                    "radius": float(final.radius),
                    "diameter": float(final.radius * 2.0),
                    "length": float(final.length),
                    "end_center_0": end0.tolist(),
                    "end_center_1": end1.tolist(),
                },
                "quality": {
                    **cylinder_quality,
                    "score": float(cylinder_score if args.force == "auto" else cylinder_quality.get("inlier_count", 0)),
                },
                "stop_reason": str(cylinder_result.stop_reason),
                "color": color.tolist(),
            }
            if cyl_proxy is not None:
                proxy_axis = unit_vector(cyl_proxy.axis_direction)
                p0, p1 = cylinder_endpoints(cyl_proxy.axis_point, proxy_axis, cyl_proxy.length)
                obj["proxy"] = {
                    "axis_point": cyl_proxy.axis_point.tolist(),
                    "axis_direction": proxy_axis.tolist(),
                    "radius": float(cyl_proxy.radius),
                    "diameter": float(cyl_proxy.radius * 2.0),
                    "length": float(cyl_proxy.length),
                    "end_center_0": p0.tolist(),
                    "end_center_1": p1.tolist(),
                }
            if args.session_dedup:
                dup_idx = find_duplicate_cylinder(obj, session["objects"])
            else:
                dup_idx = None

            if dup_idx is not None:
                existing = session["objects"][dup_idx]
                old_score = float(existing.get("quality", {}).get("score", 0.0))
                new_score = float(obj.get("quality", {}).get("score", 0.0))
                if new_score >= old_score:
                    session["objects"][dup_idx] = obj
                    if dup_idx < len(meshes):
                        meshes[dup_idx] = cyl_mesh
                    print("  Updated existing cylinder (dedup).")
                else:
                    print("  Skipped duplicate cylinder (existing better).")
            else:
                session["objects"].append(obj)
                meshes.append(cyl_mesh)
        else:
            print("  Unknown selection, skipping.")
            continue

        show_scene()

        while True:
            cmd = input("\n[Enter] next | U undo | D delete | S save | E export | Q quit: ").strip().lower()
            if cmd == "":
                break
            if cmd == "u" or cmd == "d":
                if session["objects"]:
                    session["objects"].pop()
                    if meshes:
                        meshes.pop()
                    print("Undid last object.")
                break
            if cmd == "s":
                save_session(args.session_file, session)
                continue
            if cmd == "e":
                if args.export_all:
                    combined = _combine_meshes(meshes)
                    o3d.io.write_triangle_mesh(args.export_all, combined)
                    print(f"Exported all meshes to {args.export_all}")
                else:
                    print("No --export-all specified.")
                continue
            if cmd == "q":
                save_session(args.session_file, session)
                return
            print("Unknown command.")

    save_session(args.session_file, session)


def run_seed_expand_mode(
    pcd: o3d.geometry.PointCloud,
    args,
    profile: SensorProfile,
    extra_config: dict,
    initial_cylinder: Optional[CylinderParam] = None,
):
    """
    Run seed-expand mode: fit primitive on seed, then expand to connected region.
    """
    print("\n" + "=" * 60)
    print("  SEED-EXPAND MODE")
    print("=" * 60)
    print("Parameters:")
    print(f"  Seed radius: {args.seed_radius}m")
    print(f"  Max expand radius: {args.max_expand_radius}m")
    print(f"  Grow radius: {args.grow_radius}m")
    print(f"  Expand method: {args.expand_method}")
    print(f"  Max refine iterations: {args.max_refine_iters}")
    print(f"  Max expanded points: {args.max_expanded_points}")
    print(f"  Max frontier: {args.max_frontier}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Normal threshold: {args.normal_th}°")
    print(f"  Plane threshold: {profile.plane_distance_threshold}m")
    print(f"  Cylinder threshold: {profile.cylinder_distance_threshold}m")
    if initial_cylinder is not None:
        print(
            "  Initial cylinder prior: axis_point="
            f"{np.round(initial_cylinder.axis_point, 4).tolist()}, axis_dir="
            f"{np.round(initial_cylinder.axis_direction, 4).tolist()}, "
            f"radius={initial_cylinder.radius:.4f}, length={initial_cylinder.length:.3f}"
        )
    print(
        "  Adaptive plane refine threshold: "
        f"{'on' if args.adaptive_plane_refine_th else 'off'}"
        f" (k={args.adaptive_plane_refine_k}, "
        f"scale={args.adaptive_plane_refine_min_scale}..{args.adaptive_plane_refine_max_scale})"
    )

    # Get all points and normals
    all_points = np.asarray(pcd.points)
    all_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    # Show point cloud first
    print("\n=== Point Cloud Viewer ===")
    print("Close the viewer window to proceed with seed selection")
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud (Seed-Expand Mode)")

    # Select seed center using ROI selection
    print("\n=== Seed Selection ===")
    print("Shift + Left Click to pick the seed center point")
    print("Press 'Q' to confirm selection")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Select Seed Center Point")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_indices = vis.get_picked_points()

    if len(picked_indices) == 0:
        print("No point selected, cancelling seed-expand mode")
        return

    seed_idx = picked_indices[0]
    seed_center = all_points[seed_idx]
    print(f"Selected seed center: {np.round(seed_center, 4).tolist()}")

    context_radius = args.context_radius if args.context_radius and args.context_radius > 0.0 else None
    context_pcd = None
    if args.show_context:
        context_center = seed_center if context_radius is not None else None
        context_pcd = create_context_cloud(
            pcd,
            args.context_voxel,
            center=context_center,
            radius=context_radius,
        )
        if context_pcd is None:
            print(
                "Context cloud is empty after downsampling/radius filter; "
                "visualization will fall back to the original cloud."
            )
        else:
            radius_info = (
                f"radius={context_radius}m" if context_radius is not None else "full cloud"
            )
            print(
                f"Prepared context cloud: {len(context_pcd.points)} points "
                f"({radius_info}, voxel={args.context_voxel})"
            )

    # Select primitive type
    print("\nSelect primitive type:")
    print("  [p] Plane")
    print("  [c] Cylinder")
    print("  [q] Cancel")
    choice = input("Enter choice: ").strip().lower()

    if choice == 'q':
        print("Cancelled")
        return

    output_file = args.seed_output or "seed_expand_results.json"
    seed_params = {
        "sensor_profile": profile.name,
        "seed_radius": args.seed_radius,
        "max_expand_radius": args.max_expand_radius,
        "grow_radius": args.grow_radius,
        "expand_method": args.expand_method,
        "max_refine_iters": args.max_refine_iters,
        "max_expanded_points": args.max_expanded_points,
        "max_frontier": args.max_frontier,
        "max_steps": args.max_steps,
        "normal_threshold_deg": args.normal_th,
        "plane_distance_threshold": profile.plane_distance_threshold,
        "cylinder_distance_threshold": profile.cylinder_distance_threshold,
        "adaptive_plane_refine_threshold": args.adaptive_plane_refine_th,
        "adaptive_plane_refine_k": args.adaptive_plane_refine_k,
        "adaptive_plane_refine_min_scale": args.adaptive_plane_refine_min_scale,
        "adaptive_plane_refine_max_scale": args.adaptive_plane_refine_max_scale,
    }
    if initial_cylinder is not None:
        seed_params["initial_cylinder"] = {
            "axis_point": initial_cylinder.axis_point.tolist(),
            "axis_direction": initial_cylinder.axis_direction.tolist(),
            "radius": float(initial_cylinder.radius),
            "length": float(initial_cylinder.length),
        }

    if choice == 'p':
        # Plane seed-expand
        print("\n=== Plane Seed-Expand ===")
        result = expand_plane_from_seed(
            all_points,
            seed_center,
            normals=all_normals,
            seed_radius=args.seed_radius,
            max_expand_radius=args.max_expand_radius,
            grow_radius=args.grow_radius,
            distance_threshold=profile.plane_distance_threshold,
            normal_threshold_deg=args.normal_th,
            expand_method=args.expand_method,
            max_refine_iters=args.max_refine_iters,
            adaptive_refine_threshold=args.adaptive_plane_refine_th,
            adaptive_refine_k=args.adaptive_plane_refine_k,
            adaptive_refine_min_scale=args.adaptive_plane_refine_min_scale,
            adaptive_refine_max_scale=args.adaptive_plane_refine_max_scale,
            max_expanded_points=args.max_expanded_points,
            max_frontier=args.max_frontier,
            max_steps=args.max_steps,
            verbose=True
        )

        if result.success and result.plane is not None:
            print("\n" + "-" * 40)
            print("Plane Seed-Expand Result:")
            print(f"  Normal: {np.round(result.plane.normal, 4).tolist()}")
            print(f"  Point: {np.round(result.plane.point, 4).tolist()}")
            print(f"  Tilt: {plane_tilt_deg(result.plane.normal):.2f} deg")
            print(f"  Expanded inliers: {result.expanded_inlier_count}")
            print(f"  Area: {result.area:.3f} m²")
            print(f"  Extent: ({result.extent_u:.2f} x {result.extent_v:.2f}) m")
            patch_mesh = None
            patch_metrics = None
            try:
                patch_mesh, patch_metrics = create_plane_patch_mesh(
                    result.plane,
                    all_points,
                    np.array([0.0, 0.8, 0.0]),
                    padding=0.02,
                    patch_shape=args.patch_shape,
                )
                patch_shape_used = patch_metrics.get("patch_shape", args.patch_shape)
                corners_world = patch_metrics.get("corners_world")
                corners_array = None
                if corners_world is not None:
                    corners_array = np.asarray(corners_world, dtype=float)
                corners_count = int(corners_array.shape[0]) if corners_array is not None and corners_array.ndim == 2 else 0
                if patch_shape_used == "rect":
                    rect_u = float(patch_metrics.get("rect_extent_u", 0.0))
                    rect_v = float(patch_metrics.get("rect_extent_v", 0.0))
                    print(
                        f"  Patch: rect ({rect_u:.2f} x {rect_v:.2f}) m "
                        f"area={float(patch_metrics.get('area', 0.0)):.3f} m²"
                    )
                else:
                    hull_area = float(patch_metrics.get("hull_area", patch_metrics.get("area", 0.0)))
                    print(
                        f"  Patch: hull area={float(patch_metrics.get('area', 0.0)):.3f} m² "
                        f"hull_area={hull_area:.3f} m²"
                    )
                if corners_array is not None and corners_array.ndim == 2 and corners_count > 0:
                    rounded = np.round(corners_array, 4).tolist()
                    print(f"  Patch corners_world ({corners_count}): {rounded}")
                if patch_metrics.get("patch_fallback_reason"):
                    print(f"  Patch fallback: {patch_metrics['patch_fallback_reason']}")
            except Exception as exc:
                patch_metrics = {
                    "patch_shape": args.patch_shape,
                    "corners_world": np.empty((0, 3), dtype=float),
                    "patch_fallback_reason": str(exc),
                    "area": float(result.area),
                    "extent_u": float(result.extent_u),
                    "extent_v": float(result.extent_v),
                }
                print(f"  Patch fallback: {exc}")
                print(f"  Patch shape: {args.patch_shape}")
                print("  Patch corners_world (0): []")

            # Visualize
            visualize_seed_expand_plane(
                pcd,
                result,
                seed_center,
                all_points,
                patch_mesh=patch_mesh,
                patch_shape=args.patch_shape,
                context_pcd=context_pcd,
                roi_radius=args.seed_radius,
            )

            # Save results
            save_seed_expand_result(
                result,
                "plane",
                output_file,
                seed_center,
                seed_params,
                patch_metrics=patch_metrics,
            )

            # Export inliers if requested
            if args.export_inliers and result.expanded_inlier_indices is not None:
                inlier_pcd = o3d.geometry.PointCloud()
                inlier_pcd.points = o3d.utility.Vector3dVector(all_points[result.expanded_inlier_indices])
                o3d.io.write_point_cloud(args.export_inliers, inlier_pcd)
                print(f"Exported inlier points to {args.export_inliers}")
        else:
            print(f"\nPlane seed-expand failed: {result.message}")

    elif choice == 'c':
        # Cylinder seed-expand
        print("\n=== Cylinder Seed-Expand ===")
        result = expand_cylinder_from_seed(
            all_points,
            seed_center,
            normals=all_normals,
            initial_cylinder=initial_cylinder,
            seed_radius=args.seed_radius,
            max_expand_radius=args.max_expand_radius,
            grow_radius=args.grow_radius,
            distance_threshold=profile.cylinder_distance_threshold,
            normal_threshold_deg=args.normal_th,
            expand_method=args.expand_method,
            max_refine_iters=args.max_refine_iters,
            max_expanded_points=args.max_expanded_points,
            max_frontier=args.max_frontier,
            max_steps=args.max_steps,
            verbose=True
        )

        if result.success and result.cylinder is not None:
            print("\n" + "-" * 40)
            print("Cylinder Seed-Expand Result:")
            axis_dir = unit_vector(result.cylinder.axis_direction)
            end0, end1 = cylinder_endpoints(
                result.cylinder.axis_point,
                axis_dir,
                result.cylinder.length,
            )
            print(f"  Axis point: {np.round(result.cylinder.axis_point, 4).tolist()}")
            print(f"  Axis direction: {np.round(axis_dir, 4).tolist()}")
            print(f"  Radius: {result.cylinder.radius:.4f} m")
            print(f"  Diameter: {result.cylinder.radius * 2.0:.4f} m")
            print(f"  Length: {result.cylinder.length:.3f} m")
            print(f"  End center 0: {np.round(end0, 4).tolist()}")
            print(f"  End center 1: {np.round(end1, 4).tolist()}")
            print(
                "  Residual median/MAD: "
                f"{getattr(result, 'residual_median', 0.0):.4f}/"
                f"{getattr(result, 'residual_mad', 0.0):.4f} m"
            )
            print(
                f"  Inliers: {result.cylinder.inlier_count} "
                f"(expanded={result.expanded_inlier_count})"
            )

            # Visualize
            visualize_seed_expand_cylinder(
                pcd,
                result,
                seed_center,
                all_points,
                context_pcd=context_pcd,
                roi_radius=args.seed_radius,
            )

            # Save results
            save_seed_expand_result(result, "cylinder", output_file, seed_center, seed_params)

            # Export inliers if requested
            if args.export_inliers and result.expanded_inlier_indices is not None:
                inlier_pcd = o3d.geometry.PointCloud()
                inlier_pcd.points = o3d.utility.Vector3dVector(all_points[result.expanded_inlier_indices])
                o3d.io.write_point_cloud(args.export_inliers, inlier_pcd)
                print(f"Exported inlier points to {args.export_inliers}")
        else:
            print(f"\nCylinder seed-expand failed: {result.message}")

    else:
        print("Invalid choice")
        return

    print("\n" + "=" * 60)
    print("  SEED-EXPAND MODE COMPLETE")
    print("=" * 60)


def visualize_cylinder_probe_result(
    pcd: o3d.geometry.PointCloud,
    all_points: np.ndarray,
    proxy: CylinderParam,
    final: CylinderParam,
    inlier_indices: Optional[np.ndarray],
    *,
    seed_center: np.ndarray,
    context_pcd: Optional[o3d.geometry.PointCloud] = None,
):
    """Visualize cylinder probe with proxy and final meshes."""
    geometries = []

    if context_pcd is not None and not context_pcd.is_empty():
        geometries.append(context_pcd)
    else:
        base = o3d.geometry.PointCloud(pcd)
        base.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(base)

    if inlier_indices is not None and len(inlier_indices) > 0:
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(all_points[inlier_indices])
        inlier_pcd.paint_uniform_color([0.9, 0.2, 0.2])
        geometries.append(inlier_pcd)

    proxy_mesh = create_cylinder_mesh(
        proxy.axis_point,
        proxy.axis_direction,
        proxy.radius,
        proxy.length,
        np.array([0.4, 0.7, 1.0]),
    )
    proxy_axis = create_axis_line(
        proxy.axis_point,
        proxy.axis_direction,
        proxy.length,
        np.array([0.4, 0.7, 1.0]),
    )
    geometries.extend([proxy_mesh, proxy_axis])

    final_mesh = create_cylinder_mesh(
        final.axis_point,
        final.axis_direction,
        final.radius,
        final.length,
        np.array([0.0, 0.0, 0.8]),
    )
    final_axis = create_axis_line(
        final.axis_point,
        final.axis_direction,
        final.length,
        np.array([1.0, 1.0, 0.0]),
    )
    geometries.extend([final_mesh, final_axis])

    seed_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    seed_marker.translate(seed_center)
    seed_marker.paint_uniform_color([1.0, 1.0, 0.0])
    seed_marker.compute_vertex_normals()
    geometries.append(seed_marker)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Cylinder Probe Result",
    )


def run_cylinder_probe_mode(
    pcd: o3d.geometry.PointCloud,
    args,
    profile: SensorProfile,
    extra_config: dict,
):
    """
    Run interactive cylinder probe mode.
    """
    print("\n" + "=" * 60)
    print("  CYLINDER PROBE MODE")
    print("=" * 60)
    surface_th = (
        args.cyl_probe_surface_th
        if args.cyl_probe_surface_th is not None
        else profile.cylinder_distance_threshold
    )
    print("Parameters:")
    print(f"  Seed radius start/max/step: {args.cyl_probe_seed_start}/{args.cyl_probe_seed_max}/{args.cyl_probe_seed_step} m")
    print(f"  Min seed points: {args.cyl_probe_min_seed_points}")
    print(f"  Surface threshold: {surface_th} m")
    print(f"  Cap margin: {args.cyl_probe_cap_margin} m")
    print(f"  Refine iters: {args.cyl_probe_refine_iters}")
    print(f"  Axis refit: {'ON' if args.cyl_probe_axis_refit else 'OFF'}")
    print("  Length growth: ON")
    print(f"  Grow radius: {args.grow_radius} m")
    print(f"  Max expand radius: {args.max_expand_radius} m")

    # Get all points
    all_points = np.asarray(pcd.points)
    all_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    # Show point cloud first
    print("\n=== Point Cloud Viewer ===")
    print("Shift + Left Click to pick a cylinder surface point")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Point Cloud (Cylinder Probe)")
    vis.add_geometry(pcd)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if not picked_indices:
        print("No point selected, cancelling cylinder probe.")
        return

    seed_idx = picked_indices[0]
    seed_center = all_points[seed_idx]
    print(f"Selected seed point: {np.round(seed_center, 4).tolist()}")

    context_radius = args.context_radius if args.context_radius and args.context_radius > 0.0 else None
    context_pcd = None
    if args.show_context:
        context_center = seed_center if context_radius is not None else None
        context_pcd = create_context_cloud(
            pcd,
            args.context_voxel,
            center=context_center,
            radius=context_radius,
        )
        if context_pcd is None:
            print("Context cloud is empty; using full cloud.")

    proxy_init = compute_cylinder_proxy_from_seed(
        all_points,
        seed_center,
        normals=all_normals,
        seed_radius_start=args.cyl_probe_seed_start,
        seed_radius_max=args.cyl_probe_seed_max,
        seed_radius_step=args.cyl_probe_seed_step,
        min_seed_points=args.cyl_probe_min_seed_points,
        circle_ransac_iters=200,
        circle_inlier_threshold=surface_th,
        length_margin=0.05,
    )
    if not proxy_init.success or proxy_init.cylinder is None:
        print(f"Cylinder probe init failed: {proxy_init.message}")
        return

    proxy = proxy_init.cylinder
    axis_dir_pca = unit_vector(proxy_init.axis_dir_pca) if proxy_init.axis_dir_pca is not None else unit_vector(proxy.axis_direction)
    print("Initial proxy cylinder:")
    print(f"  Axis point: {np.round(proxy.axis_point, 4).tolist()}")
    print(f"  Axis direction: {np.round(proxy.axis_direction, 4).tolist()}")
    print(f"  Radius: {proxy.radius:.4f} m")
    print(f"  Length: {proxy.length:.3f} m")
    print(f"  Seed radius used: {proxy_init.seed_radius:.3f} m ({proxy_init.seed_point_count} pts)")

    help_text = (
        "Keys: [/] radius, -/= length, WASD move, R/F axis move, "
        "arrows rotate, X reset, V snap Z, Enter confirm, Esc cancel"
    )
    print("\nCylinder Probe Controls:")
    print(f"  {help_text}")

    class ProbeState:
        def __init__(self, cylinder: CylinderParam, axis_dir_reset: np.ndarray):
            self.axis_point = np.asarray(cylinder.axis_point, dtype=float)
            self.axis_dir = unit_vector(cylinder.axis_direction)
            self.radius = float(cylinder.radius)
            self.length = float(cylinder.length)
            self.axis_dir_reset = unit_vector(axis_dir_reset)
            self.snap_z = False
            self.axis_dir_before_snap = self.axis_dir.copy()
            self.confirmed = False
            self.cancelled = False
            self.ops = {
                "radius_adjust": 0,
                "length_adjust": 0,
                "move_perp": 0,
                "move_axis": 0,
                "rotate": 0,
                "reset_axis": 0,
                "snap_toggle": 0,
            }

        def ensure_snap_off(self):
            if self.snap_z:
                self.snap_z = False
                self.axis_dir_before_snap = self.axis_dir.copy()
                self.ops["snap_toggle"] += 1

    state = ProbeState(proxy, axis_dir_pca)

    radius_step = 0.002
    length_step = 0.05
    move_step = 0.02
    axis_step = 0.02
    rot_step = 2.0

    proxy_mesh = create_cylinder_mesh(
        state.axis_point,
        state.axis_dir,
        state.radius,
        state.length,
        np.array([0.4, 0.7, 1.0]),
    )
    proxy_axis = create_axis_line(
        state.axis_point,
        state.axis_dir,
        state.length,
        np.array([0.4, 0.7, 1.0]),
    )

    seed_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    seed_marker.translate(seed_center)
    seed_marker.paint_uniform_color([1.0, 1.0, 0.0])
    seed_marker.compute_vertex_normals()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Cylinder Probe ({help_text})")
    if context_pcd is not None and not context_pcd.is_empty():
        vis.add_geometry(context_pcd)
    else:
        base = o3d.geometry.PointCloud(pcd)
        base.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(base)
    vis.add_geometry(proxy_mesh)
    vis.add_geometry(proxy_axis)
    vis.add_geometry(seed_marker)

    def refresh_geometry():
        new_mesh = create_cylinder_mesh(
            state.axis_point,
            state.axis_dir,
            state.radius,
            state.length,
            np.array([0.4, 0.7, 1.0]),
        )
        proxy_mesh.vertices = new_mesh.vertices
        proxy_mesh.triangles = new_mesh.triangles
        proxy_mesh.vertex_colors = new_mesh.vertex_colors
        proxy_mesh.compute_vertex_normals()

        new_axis = create_axis_line(
            state.axis_point,
            state.axis_dir,
            state.length,
            np.array([0.4, 0.7, 1.0]),
        )
        proxy_axis.points = new_axis.points
        proxy_axis.lines = new_axis.lines
        proxy_axis.colors = new_axis.colors

        vis.update_geometry(proxy_mesh)
        vis.update_geometry(proxy_axis)

    def close_window():
        try:
            vis.close()
        except Exception:
            vis.destroy_window()

    def on_radius_minus(vis_obj):
        state.radius = max(state.radius - radius_step, 0.001)
        state.ops["radius_adjust"] += 1
        refresh_geometry()
        return True

    def on_radius_plus(vis_obj):
        state.radius = max(state.radius + radius_step, 0.001)
        state.ops["radius_adjust"] += 1
        refresh_geometry()
        return True

    def on_length_minus(vis_obj):
        state.length = max(state.length - length_step, 0.02)
        state.ops["length_adjust"] += 1
        refresh_geometry()
        return True

    def on_length_plus(vis_obj):
        state.length = max(state.length + length_step, 0.02)
        state.ops["length_adjust"] += 1
        refresh_geometry()
        return True

    def move_perp(dx, dy):
        u, v = _plane_basis_from_normal(state.axis_dir)
        state.axis_point = state.axis_point + u * dx + v * dy
        state.ops["move_perp"] += 1
        refresh_geometry()

    def move_axis(dt):
        state.axis_point = state.axis_point + state.axis_dir * dt
        state.ops["move_axis"] += 1
        refresh_geometry()

    def rotate_axis(rot_axis, angle_deg):
        state.ensure_snap_off()
        state.axis_dir = unit_vector(rotate_vector(state.axis_dir, rot_axis, angle_deg))
        state.ops["rotate"] += 1
        refresh_geometry()

    def on_w(vis_obj):
        move_perp(0.0, move_step)
        return True

    def on_s(vis_obj):
        move_perp(0.0, -move_step)
        return True

    def on_a(vis_obj):
        move_perp(-move_step, 0.0)
        return True

    def on_d(vis_obj):
        move_perp(move_step, 0.0)
        return True

    def on_r(vis_obj):
        move_axis(axis_step)
        return True

    def on_f(vis_obj):
        move_axis(-axis_step)
        return True

    def on_up(vis_obj):
        u, _ = _plane_basis_from_normal(state.axis_dir)
        rotate_axis(u, rot_step)
        return True

    def on_down(vis_obj):
        u, _ = _plane_basis_from_normal(state.axis_dir)
        rotate_axis(u, -rot_step)
        return True

    def on_left(vis_obj):
        _, v = _plane_basis_from_normal(state.axis_dir)
        rotate_axis(v, rot_step)
        return True

    def on_right(vis_obj):
        _, v = _plane_basis_from_normal(state.axis_dir)
        rotate_axis(v, -rot_step)
        return True

    def on_reset_axis(vis_obj):
        state.axis_dir = unit_vector(state.axis_dir_reset)
        state.snap_z = False
        state.ops["reset_axis"] += 1
        refresh_geometry()
        return True

    def on_snap_z(vis_obj):
        if not state.snap_z:
            state.axis_dir_before_snap = state.axis_dir.copy()
            state.axis_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            state.snap_z = True
        else:
            state.axis_dir = state.axis_dir_before_snap.copy()
            state.snap_z = False
        state.ops["snap_toggle"] += 1
        refresh_geometry()
        return True

    def on_confirm(vis_obj):
        state.confirmed = True
        close_window()
        return True

    def on_cancel(vis_obj):
        state.cancelled = True
        close_window()
        return True

    vis.register_key_callback(ord("["), on_radius_minus)
    vis.register_key_callback(ord("]"), on_radius_plus)
    vis.register_key_callback(ord("-"), on_length_minus)
    vis.register_key_callback(ord("="), on_length_plus)
    vis.register_key_callback(ord("W"), on_w)
    vis.register_key_callback(ord("S"), on_s)
    vis.register_key_callback(ord("A"), on_a)
    vis.register_key_callback(ord("D"), on_d)
    vis.register_key_callback(ord("R"), on_r)
    vis.register_key_callback(ord("F"), on_f)
    vis.register_key_callback(ord("X"), on_reset_axis)
    vis.register_key_callback(ord("V"), on_snap_z)
    vis.register_key_callback(265, on_up)    # GLFW_KEY_UP
    vis.register_key_callback(264, on_down)  # GLFW_KEY_DOWN
    vis.register_key_callback(263, on_left)  # GLFW_KEY_LEFT
    vis.register_key_callback(262, on_right) # GLFW_KEY_RIGHT
    vis.register_key_callback(257, on_confirm)  # GLFW_KEY_ENTER
    vis.register_key_callback(256, on_cancel)   # GLFW_KEY_ESCAPE

    vis.run()
    vis.destroy_window()

    if state.cancelled or not state.confirmed:
        print("Cylinder probe cancelled.")
        return

    proxy_final = CylinderParam(
        axis_point=state.axis_point,
        axis_direction=state.axis_dir,
        radius=float(state.radius),
        length=float(state.length),
        inlier_count=0,
        inlier_indices=None,
    )

    result = finalize_cylinder_from_proxy(
        all_points,
        seed_center,
        proxy_init.seed_indices,
        proxy_final,
        surface_threshold=surface_th,
        cap_margin=args.cyl_probe_cap_margin,
        grow_radius=args.grow_radius,
        max_expand_radius=args.max_expand_radius,
        max_expanded_points=args.max_expanded_points,
        max_frontier=args.max_frontier,
        max_steps=args.max_steps,
        normals=all_normals,
        normal_angle_threshold_deg=args.normal_th,
        refine_iters=args.cyl_probe_refine_iters,
        circle_ransac_iters=200,
        circle_inlier_threshold=surface_th,
        allow_length_growth=True,
        allow_axis_refit=args.cyl_probe_axis_refit,
        axis_snap_to_vertical_deg=args.cyl_probe_axis_snap_deg,
        axis_regularization_weight=args.cyl_probe_axis_reg_weight,
        recompute_length_from_inliers=not args.cyl_probe_no_recompute_length,
    )

    if args.cyl_probe_diagnostics_dir and result.success:
        from primitives import export_cylinder_diagnostics_ply
        diag_files = export_cylinder_diagnostics_ply(
            all_points,
            args.cyl_probe_diagnostics_dir,
            prefix="cyl_probe",
            seed_indices=proxy_init.seed_indices,
            inlier_indices=result.inlier_indices,
            cylinder=result.final,
        )
        print(f"  Diagnostics exported: {len(diag_files)} files to {args.cyl_probe_diagnostics_dir}")

    if not result.success or result.final is None:
        print(f"Cylinder probe failed: {result.message}")
        return

    final = result.final
    final_axis = unit_vector(final.axis_direction)
    end0, end1 = cylinder_endpoints(final.axis_point, final_axis, final.length)

    print("\n" + "-" * 40)
    print("Cylinder Probe Result:")
    print(f"  Proxy axis point: {np.round(proxy_final.axis_point, 4).tolist()}")
    print(f"  Proxy axis dir: {np.round(unit_vector(proxy_final.axis_direction), 4).tolist()}")
    print(f"  Proxy radius/length: {proxy_final.radius:.4f} m / {proxy_final.length:.3f} m")
    print(f"  Final axis point: {np.round(final.axis_point, 4).tolist()}")
    print(f"  Final axis dir: {np.round(final_axis, 4).tolist()}")
    print(f"  Final radius/diameter: {final.radius:.4f} / {final.radius * 2.0:.4f} m")
    print(f"  Final length: {final.length:.3f} m")
    print(f"  End centers: {np.round(end0, 4).tolist()} -> {np.round(end1, 4).tolist()}")
    print(
        f"  Residual median/MAD: {result.residual_median:.4f}/{result.residual_mad:.4f} m"
    )
    print(
        f"  Inliers: {final.inlier_count} (candidates={result.candidate_count})"
    )
    if result.stopped_early:
        print(f"  WARNING: expansion stopped early ({result.stop_reason})")

    visualize_cylinder_probe_result(
        pcd,
        all_points,
        proxy_final,
        final,
        result.inlier_indices,
        seed_center=seed_center,
        context_pcd=context_pcd,
    )

    params = {
        "seed_radius_start": args.cyl_probe_seed_start,
        "seed_radius_max": args.cyl_probe_seed_max,
        "seed_radius_step": args.cyl_probe_seed_step,
        "min_seed_points": args.cyl_probe_min_seed_points,
        "seed_radius_used": float(proxy_init.seed_radius),
        "seed_point_count": int(proxy_init.seed_point_count),
        "surface_threshold": float(surface_th),
        "cap_margin": float(args.cyl_probe_cap_margin),
        "grow_radius": float(args.grow_radius),
        "max_expand_radius": float(args.max_expand_radius),
        "max_expanded_points": int(args.max_expanded_points),
        "max_frontier": int(args.max_frontier),
        "max_steps": int(args.max_steps),
        "refine_iters": int(args.cyl_probe_refine_iters),
    }
    save_cylinder_probe_result(
        result,
        proxy_final,
        final,
        args.cyl_probe_output,
        seed_center,
        params,
        state.ops,
    )

    if args.export_mesh:
        mesh = create_cylinder_mesh(
            final.axis_point,
            final.axis_direction,
            final.radius,
            final.length,
            np.array([0.0, 0.0, 0.8]),
        )
        o3d.io.write_triangle_mesh(args.export_mesh, mesh)
        print(f"Exported cylinder mesh to {args.export_mesh}")


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

    roi_center: Optional[np.ndarray] = None
    roi_radius_used: Optional[float] = None
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
        roi_center = roi_selector.last_center
        roi_radius_used = profile.r_min
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
        roi_center = roi_result.center
        roi_radius_used = roi_result.final_radius

        if not roi_result.success:
            print("\nWARNING: ROI has fewer points than required.")
            proceed = input("Proceed with stair extraction anyway? [y/n]: ").strip().lower()
            if proceed != 'y':
                print("Stairs mode cancelled.")
                return

    # Get points from ROI
    points = np.asarray(roi_pcd.points)
    print(f"\nROI contains {len(points)} points")

    context_radius = args.context_radius if args.context_radius and args.context_radius > 0.0 else None
    context_pcd = None
    if args.show_context:
        context_center = roi_center if context_radius is not None else None
        if args.context_radius is not None and args.context_radius > 0.0 and context_center is None:
            print("Context radius specified but ROI center unavailable; showing full cloud instead.")
        context_pcd = create_context_cloud(
            pcd,
            args.context_voxel,
            center=context_center,
            radius=context_radius,
        )
        if context_pcd is None:
            print(
                "Context cloud is empty after downsampling/radius filter; "
                "visualization will show ROI and patches only."
            )
        else:
            radius_info = (
                f"radius={context_radius}m" if context_radius is not None else "full cloud"
            )
            print(
                f"Prepared context cloud: {len(context_pcd.points)} points "
                f"({radius_info}, voxel={args.context_voxel})"
            )

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
    visualize_stair_planes(
        roi_pcd,
        planes,
        patch_shape=args.patch_shape,
        context_pcd=context_pcd,
        roi_center=roi_center,
        roi_radius=roi_radius_used,
    )

    # Save results to JSON
    save_stairs_results(planes, points, args.stairs_output, patch_shape=args.patch_shape)

    # Export mesh if requested
    if args.export_mesh:
        export_stair_planes_mesh(planes, points, args.export_mesh, patch_shape=args.patch_shape)

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
    if args.gui_app:
        from gui_app import launch_gui
        launch_gui(args)
        return
    try:
        initial_cylinder = resolve_initial_cylinder(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(2)

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

    if args.cyl_probe and args.seed_expand:
        print("Error: --cyl-probe and --seed-expand cannot be used together.")
        sys.exit(2)
    if args.session_mode and (args.cyl_probe or args.seed_expand or args.stairs_mode):
        print("Error: --session cannot be combined with other modes.")
        sys.exit(2)

    if args.session_mode:
        run_session_mode(pcd, args, profile, extra_config)
        return

    # Cylinder probe mode
    if args.cyl_probe:
        run_cylinder_probe_mode(pcd, args, profile, extra_config)
        return

    # Check if seed-expand mode
    if args.seed_expand:
        run_seed_expand_mode(pcd, args, profile, extra_config, initial_cylinder=initial_cylinder)
        return

    # Check if stairs mode
    if args.stairs_mode:
        run_stairs_mode(pcd, args, profile, extra_config)
        return

    all_points = np.asarray(pcd.points)
    all_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

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
            roi_center = roi_selector.last_center
            roi_radius_used = profile.r_min
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

            roi_center = roi_result.center
            roi_radius_used = roi_result.final_radius

        if roi_center is None:
            roi_points = np.asarray(roi_pcd.points)
            if roi_points.size == 0:
                print("ROI has no points; skipping.")
                continue
            roi_center = np.mean(roi_points, axis=0)

        seed_radius = roi_radius_used if roi_radius_used and roi_radius_used > 0 else args.seed_radius

        context_radius = args.context_radius if args.context_radius and args.context_radius > 0.0 else None
        context_pcd = None
        if args.show_context:
            context_center = roi_center if context_radius is not None else None
            context_pcd = create_context_cloud(
                pcd,
                args.context_voxel,
                center=context_center,
                radius=context_radius,
            )
            if context_pcd is None:
                print(
                    "Context cloud is empty after downsampling/radius filter; "
                    "visualization will fall back to the original cloud."
                )

        # Fit primitive
        if primitive_type == 'p':
            print("\nExpanding plane from seed...")
            result = expand_plane_from_seed(
                all_points,
                roi_center,
                normals=all_normals,
                seed_radius=seed_radius,
                max_expand_radius=args.max_expand_radius,
                grow_radius=args.grow_radius,
                distance_threshold=profile.plane_distance_threshold,
                normal_threshold_deg=args.normal_th,
                expand_method=args.expand_method,
                max_refine_iters=args.max_refine_iters,
                adaptive_refine_threshold=args.adaptive_plane_refine_th,
                adaptive_refine_k=args.adaptive_plane_refine_k,
                adaptive_refine_min_scale=args.adaptive_plane_refine_min_scale,
                adaptive_refine_max_scale=args.adaptive_plane_refine_max_scale,
                max_expanded_points=args.max_expanded_points,
                max_frontier=args.max_frontier,
                max_steps=args.max_steps,
                verbose=True
            )

            if result.success and result.plane is not None:
                print(
                    "Plane expansion OK | normal="
                    f"{np.round(result.plane.normal, 4).tolist()}, point="
                    f"{np.round(result.plane.point, 4).tolist()}, "
                    f"tilt={plane_tilt_deg(result.plane.normal):.2f}deg, "
                    f"inliers={result.expanded_inlier_count}, "
                    f"area={result.area:.3f}m^2, "
                    f"extent=({result.extent_u:.2f} x {result.extent_v:.2f})m"
                )
                patch_mesh = None
                patch_metrics = None
                try:
                    patch_mesh, patch_metrics = create_plane_patch_mesh(
                        result.plane,
                        all_points,
                        np.array([0.0, 0.8, 0.0]),
                        padding=0.02,
                        patch_shape=args.patch_shape,
                    )
                    if patch_metrics is not None:
                        patch_shape_used = patch_metrics.get("patch_shape", args.patch_shape)
                        corners_world = patch_metrics.get("corners_world")
                        corners_array = None
                        if corners_world is not None:
                            corners_array = np.asarray(corners_world, dtype=float)
                        corners_count = int(corners_array.shape[0]) if corners_array is not None and corners_array.ndim == 2 else 0
                        print(f"  Patch shape: {patch_shape_used}")
                        if corners_array is not None and corners_array.ndim == 2 and corners_count > 0:
                            rounded = np.round(corners_array, 4).tolist()
                            print(f"  Patch corners_world ({corners_count}): {rounded}")
                        if patch_metrics.get("patch_fallback_reason"):
                            print(f"  Patch fallback: {patch_metrics['patch_fallback_reason']}")
                except Exception as exc:
                    patch_metrics = {
                        "patch_shape": args.patch_shape,
                        "corners_world": np.empty((0, 3), dtype=float),
                        "patch_fallback_reason": str(exc),
                        "area": float(result.area),
                        "extent_u": float(result.extent_u),
                        "extent_v": float(result.extent_v),
                    }
                    print(f"  Patch fallback: {exc}")
                    print(f"  Patch shape: {args.patch_shape}")
                    print("  Patch corners_world (0): []")

                visualize_seed_expand_plane(
                    pcd,
                    result,
                    roi_center,
                    all_points,
                    patch_mesh=patch_mesh,
                    patch_shape=args.patch_shape,
                    context_pcd=context_pcd,
                    roi_radius=seed_radius,
                )
                results = append_plane_result(results, result.plane)
                save_results(results, args.output)
            else:
                print(f"Plane expansion failed: {result.message}")

        elif primitive_type == 'c':
            print("\nExpanding cylinder from seed...")
            result = expand_cylinder_from_seed(
                all_points,
                roi_center,
                normals=all_normals,
                initial_cylinder=initial_cylinder,
                seed_radius=seed_radius,
                max_expand_radius=args.max_expand_radius,
                grow_radius=args.grow_radius,
                distance_threshold=profile.cylinder_distance_threshold,
                normal_threshold_deg=args.normal_th,
                expand_method=args.expand_method,
                max_refine_iters=args.max_refine_iters,
                max_expanded_points=args.max_expanded_points,
                max_frontier=args.max_frontier,
                max_steps=args.max_steps,
                verbose=True
            )

            if result.success and result.cylinder is not None:
                axis_dir = unit_vector(result.cylinder.axis_direction)
                end0, end1 = cylinder_endpoints(
                    result.cylinder.axis_point,
                    axis_dir,
                    result.cylinder.length,
                )
                print(
                    "Cylinder expansion OK | axis_point="
                    f"{np.round(result.cylinder.axis_point, 4).tolist()}, axis_dir="
                    f"{np.round(axis_dir, 4).tolist()}, radius="
                    f"{result.cylinder.radius:.4f}, diameter={result.cylinder.radius * 2.0:.4f}, "
                    f"length={result.cylinder.length:.4f}, "
                    f"end0={np.round(end0, 4).tolist()}, end1={np.round(end1, 4).tolist()}, "
                    f"residual_median/MAD={getattr(result, 'residual_median', 0.0):.4f}/"
                    f"{getattr(result, 'residual_mad', 0.0):.4f}, "
                    f"inliers={result.cylinder.inlier_count} "
                    f"(expanded={result.expanded_inlier_count})"
                )
                visualize_seed_expand_cylinder(
                    pcd,
                    result,
                    roi_center,
                    all_points,
                    context_pcd=context_pcd,
                    roi_radius=seed_radius,
                )
                results = append_cylinder_result(results, result.cylinder)
                save_results(results, args.output)
            else:
                print(f"Cylinder expansion failed: {result.message}")

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
