#!/usr/bin/env python3
"""
main.py - Primitive Fitting Tool for LiDAR point clouds

CLI tool for fitting plane and cylinder primitives to point cloud data.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from primitives import fit_plane, fit_cylinder, PlaneParam, CylinderParam


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

class ROISelector:
    """Interactive ROI selection using Open3D visualizer."""

    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.selected_indices: Optional[np.ndarray] = None
        self.vis = None

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
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
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

        cropped_indices = vis.get_cropped_geometry()

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


# =============================================================================
# Main Application
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Primitive Fitting Tool for LiDAR point clouds"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input PCD or PLY file"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling (default: 0.01, 0 to skip)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="fit_results.json",
        help="Output JSON file for results (default: fit_results.json)"
    )
    parser.add_argument(
        "--roi_radius",
        type=float,
        default=0.2,
        help="Radius for ROI selection around picked point (default: 0.2)"
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Skip preprocessing (downsampling, outlier removal)"
    )
    return parser.parse_args()


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


def main():
    """Main entry point."""
    args = parse_args()

    # Load point cloud
    print(f"\nLoading point cloud from {args.input}...")
    pcd = load_point_cloud(args.input)

    # Preprocess
    if not args.no_preprocess:
        print("\nPreprocessing point cloud...")
        pcd = preprocess_point_cloud(pcd, voxel_size=args.voxel_size)

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

        # Select ROI
        print(f"\nSelecting ROI (radius: {args.roi_radius})...")
        roi_pcd = roi_selector.select_roi_picking(radius=args.roi_radius)

        if roi_pcd is None or roi_pcd.is_empty():
            print("ROI selection cancelled or empty")
            continue

        # Get points and normals from ROI
        points = np.asarray(roi_pcd.points)
        normals = np.asarray(roi_pcd.normals) if roi_pcd.has_normals() else None

        # Fit primitive
        if primitive_type == 'p':
            print("\nFitting plane...")
            plane = fit_plane(points)
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
            cylinder = fit_cylinder(points, normals)
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
