"""
primitives.py - Primitive fitting functions for plane and cylinder
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import open3d as o3d


@dataclass
class PlaneParam:
    """Parameters for a fitted plane."""
    normal: np.ndarray        # (nx, ny, nz)
    point: np.ndarray         # (px, py, pz) - representative point on plane
    inlier_count: int
    inlier_indices: Optional[np.ndarray] = None


@dataclass
class CylinderParam:
    """Parameters for a fitted cylinder."""
    axis_point: np.ndarray      # (ax, ay, az) - point on axis
    axis_direction: np.ndarray  # (dx, dy, dz) - normalized axis direction
    radius: float
    length: float
    inlier_count: int
    inlier_indices: Optional[np.ndarray] = None


def fit_plane(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000
) -> Optional[PlaneParam]:
    """
    Fit a plane to the given points using RANSAC.

    Args:
        points: (N, 3) array of 3D points
        distance_threshold: Maximum distance from plane for inliers
        ransac_n: Number of points to sample for each iteration
        num_iterations: Number of RANSAC iterations

    Returns:
        PlaneParam with fitted plane parameters, or None if fitting fails
    """
    if points is None or len(points) < 3:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
    except RuntimeError:
        return None

    if len(inliers) == 0:
        return None

    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        return None
    normal = normal / normal_norm

    inlier_points = points[inliers]
    point_on_plane = inlier_points.mean(axis=0)

    return PlaneParam(
        normal=normal,
        point=point_on_plane,
        inlier_count=len(inliers),
        inlier_indices=np.array(inliers, dtype=int)
    )


def fit_cylinder(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    distance_threshold: float = 0.02,
    radius_min: float = 0.01,
    radius_max: float = 1.0,
    num_iterations: int = 1000,
    normal_angle_threshold_deg: float = 60.0
) -> Optional[CylinderParam]:
    """
    Fit a cylinder to the given points using RANSAC.

    Args:
        points: (N, 3) array of 3D points
        normals: (N, 3) array of surface normals (optional but recommended)
        distance_threshold: Maximum distance from cylinder surface for inliers
        radius_min: Minimum allowed radius
        radius_max: Maximum allowed radius
        num_iterations: Number of RANSAC iterations
        normal_angle_threshold_deg: Max angle (degrees) between radial direction
            and provided normals for inliers

    Returns:
        CylinderParam with fitted cylinder parameters, or None if fitting fails
    """
    if len(points) < 6:
        return None

    normals_valid = (
        normals is not None and
        len(normals) == len(points) and
        np.all(np.isfinite(normals))
    )
    if normals_valid:
        normals_unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    else:
        normals_unit = None

    def compute_axis_metrics(axis_point: np.ndarray, axis_dir: np.ndarray):
        diff = points - axis_point
        projections = diff @ axis_dir
        radial_vec = diff - np.outer(projections, axis_dir)
        radial_dist = np.linalg.norm(radial_vec, axis=1)
        return projections, radial_dist, radial_vec

    rng = np.random.default_rng()
    best_inliers: Optional[np.ndarray] = None
    normal_cos_threshold = np.cos(np.deg2rad(normal_angle_threshold_deg))

    for _ in range(num_iterations):
        sample_size = min(len(points), 30)
        sample_indices = rng.choice(len(points), size=sample_size, replace=False)
        sample_points = points[sample_indices]
        sample_center = sample_points.mean(axis=0)
        centered = sample_points - sample_center
        _, _, vh = np.linalg.svd(centered)
        axis_dir = vh[0]
        axis_norm = np.linalg.norm(axis_dir)
        if axis_norm == 0:
            continue
        axis_dir = axis_dir / axis_norm

        _, radial_distances, radial_vec = compute_axis_metrics(sample_center, axis_dir)
        radius_candidate = np.median(radial_distances)
        if not (radius_min <= radius_candidate <= radius_max):
            continue

        residual = np.abs(radial_distances - radius_candidate)
        inlier_mask = residual < distance_threshold

        if normals_unit is not None:
            radial_dir = radial_vec / np.maximum(radial_distances[:, None], 1e-8)
            normal_alignment = np.einsum("ij,ij->i", radial_dir, normals_unit)
            inlier_mask &= normal_alignment > normal_cos_threshold

        inliers = np.where(inlier_mask)[0]

        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers

    if best_inliers is None or len(best_inliers) < 6:
        return None

    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    _, _, vh = np.linalg.svd(centered)
    axis_direction = vh[0]
    axis_direction /= np.linalg.norm(axis_direction)

    projections, radial_distances, radial_vec = compute_axis_metrics(centroid, axis_direction)
    radius = float(np.median(radial_distances[best_inliers]))

    residual = np.abs(radial_distances - radius)
    inlier_mask = residual < distance_threshold
    if normals_unit is not None:
        radial_dir = radial_vec / np.maximum(radial_distances[:, None], 1e-8)
        normal_alignment = np.einsum("ij,ij->i", radial_dir, normals_unit)
        inlier_mask &= normal_alignment > normal_cos_threshold
    final_inliers = np.where(inlier_mask)[0]
    if len(final_inliers) == 0:
        final_inliers = best_inliers

    final_inliers = np.array(final_inliers, dtype=int)

    projections = projections[final_inliers]
    radial_distances = radial_distances[final_inliers]

    length = float(np.max(projections) - np.min(projections)) if len(projections) > 0 else 0.0
    radius = float(np.median(radial_distances)) if len(radial_distances) > 0 else float(radius)

    return CylinderParam(
        axis_point=centroid,
        axis_direction=axis_direction,
        radius=radius,
        length=length,
        inlier_count=len(final_inliers),
        inlier_indices=final_inliers
    )
