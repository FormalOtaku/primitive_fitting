"""
primitives.py - Primitive fitting functions for plane and cylinder
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, List, Tuple
import numpy as np
import open3d as o3d


@dataclass
class PlaneParam:
    """Parameters for a fitted plane."""
    normal: np.ndarray        # (nx, ny, nz)
    point: np.ndarray         # (px, py, pz) - representative point on plane
    inlier_count: int
    inlier_indices: Optional[np.ndarray] = None
    height: Optional[float] = None  # z-coordinate of representative point (for stairs)


@dataclass
class CylinderParam:
    """Parameters for a fitted cylinder."""
    axis_point: np.ndarray      # (ax, ay, az) - point on axis
    axis_direction: np.ndarray  # (dx, dy, dz) - normalized axis direction
    radius: float
    length: float
    inlier_count: int
    inlier_indices: Optional[np.ndarray] = None


def _orient_direction(vec: np.ndarray, *, prefer_positive_z: bool) -> np.ndarray:
    """Return vec or -vec with a deterministic sign convention."""
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (3,):
        vec = vec.reshape(3)
    if prefer_positive_z:
        return vec if vec[2] >= 0 else -vec
    dominant = int(np.argmax(np.abs(vec)))
    return vec if vec[dominant] >= 0 else -vec


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

    plane_normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(plane_normal)
    if not np.isfinite(normal_norm) or normal_norm < 1e-12:
        return None
    normal = plane_normal / normal_norm

    inlier_points = points[inliers]
    centroid = inlier_points.mean(axis=0)
    d = float(plane_model[3]) / float(normal_norm)
    # Project centroid onto the RANSAC plane to get a point guaranteed to lie on the plane.
    point_on_plane = centroid - (normal @ centroid + d) * normal

    normal = _orient_direction(normal, prefer_positive_z=True)

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
    axis_direction = _orient_direction(axis_direction, prefer_positive_z=False)

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


# =============================================================================
# Multi-plane extraction for stairs mode
# =============================================================================

@dataclass
class MultiPlaneResult:
    """Result of multi-plane extraction."""
    planes: List[PlaneParam]
    remaining_points: np.ndarray
    remaining_indices: np.ndarray


def extract_multi_planes(
    points: np.ndarray,
    *,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_inliers: int = 50,
    max_planes: int = 20,
    max_failures: int = 3,
    verbose: bool = True,
) -> MultiPlaneResult:
    """
    Extract multiple planes from point cloud using iterative RANSAC.

    Args:
        points: (N, 3) array of 3D points
        distance_threshold: Maximum distance from plane for inliers
        ransac_n: Number of points to sample for each RANSAC iteration
        num_iterations: Number of RANSAC iterations per plane
        min_inliers: Minimum inliers required to accept a plane
        max_planes: Maximum number of planes to extract
        max_failures: Stop after this many consecutive failures
        verbose: If True, print progress logs

    Returns:
        MultiPlaneResult containing list of planes and remaining points
    """
    if points is None:
        return MultiPlaneResult(planes=[], remaining_points=np.empty((0, 3)), remaining_indices=np.empty((0,), dtype=int))

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        if verbose:
            print(f"  extract_multi_planes: invalid points shape {points.shape}, expected (N, 3)")
        return MultiPlaneResult(planes=[], remaining_points=np.empty((0, 3)), remaining_indices=np.empty((0,), dtype=int))

    if max_planes <= 0:
        return MultiPlaneResult(planes=[], remaining_points=points.copy(), remaining_indices=np.arange(len(points), dtype=int))

    # Basic parameter validation / normalization (be permissive to avoid crashing).
    ransac_n = int(max(3, ransac_n))
    num_iterations = int(max(1, num_iterations))
    min_inliers = int(max(ransac_n, min_inliers))
    max_failures = int(max(1, max_failures))
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        distance_threshold = 0.02

    finite_mask = np.all(np.isfinite(points), axis=1)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        if verbose:
            print(f"  extract_multi_planes: dropped {dropped} non-finite points (NaN/Inf)")
        points = points[finite_mask]
        valid_indices = np.flatnonzero(finite_mask).astype(int)
    else:
        valid_indices = np.arange(len(points), dtype=int)

    planes: List[PlaneParam] = []
    remaining_points = points.copy()
    remaining_indices = valid_indices.copy()

    failures = 0

    for i in range(max_planes):
        if len(remaining_points) < min_inliers:
            if verbose:
                print(f"  Stopping: only {len(remaining_points)} points remaining (< {min_inliers})")
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining_points)

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
        except Exception as exc:
            failures += 1
            if verbose:
                print(f"  Plane {i}: segment_plane failed ({type(exc).__name__}): {exc}")
            if failures >= max_failures:
                if verbose:
                    print(f"  Stopping: too many consecutive failures ({failures}/{max_failures})")
                break
            continue

        if len(inliers) < min_inliers:
            if verbose:
                print(f"  Stopping: plane {i} has only {len(inliers)} inliers (< {min_inliers})")
            break

        # Extract plane parameters
        normal = np.asarray(plane_model[:3], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if not np.isfinite(normal_norm) or normal_norm < 1e-12:
            failures += 1
            if verbose:
                print(f"  Plane {i}: invalid normal from segment_plane, skipping")
            if failures >= max_failures:
                if verbose:
                    print(f"  Stopping: too many consecutive failures ({failures}/{max_failures})")
                break
            continue
        normal = normal / normal_norm

        inlier_points = remaining_points[inliers]
        if inlier_points.size == 0 or not np.all(np.isfinite(inlier_points)):
            failures += 1
            if verbose:
                print(f"  Plane {i}: non-finite inlier points, skipping")
            if failures >= max_failures:
                if verbose:
                    print(f"  Stopping: too many consecutive failures ({failures}/{max_failures})")
                break
            continue
        point_on_plane = inlier_points.mean(axis=0)
        if not np.all(np.isfinite(point_on_plane)):
            failures += 1
            if verbose:
                print(f"  Plane {i}: non-finite plane point, skipping")
            if failures >= max_failures:
                if verbose:
                    print(f"  Stopping: too many consecutive failures ({failures}/{max_failures})")
                break
            continue
        height = float(point_on_plane[2])

        # Map local inlier indices to original indices
        global_inlier_indices = remaining_indices[np.asarray(inliers, dtype=int)]

        plane = PlaneParam(
            normal=normal,
            point=point_on_plane,
            inlier_count=len(inliers),
            inlier_indices=global_inlier_indices,
            height=height
        )
        planes.append(plane)

        failures = 0
        if verbose:
            print(
                f"  Plane {i}: {len(inliers)} inliers, height={height:.3f}m, "
                f"normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]"
            )

        # Remove inlier points for next iteration
        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]

    return MultiPlaneResult(
        planes=planes,
        remaining_points=remaining_points,
        remaining_indices=remaining_indices
    )


def filter_horizontal_planes(
    planes: List[PlaneParam],
    max_tilt_deg: float = 15.0,
    verbose: bool = True,
) -> List[PlaneParam]:
    """
    Filter planes to keep only horizontal ones (for stair treads and landings).

    A plane is considered horizontal if its normal is close to +Z.
    After filtering, normals are oriented to satisfy nz >= 0.

    Args:
        planes: List of PlaneParam to filter
        max_tilt_deg: Maximum tilt angle from horizontal (degrees)

    Returns:
        List of horizontal planes
    """
    max_tilt_deg = float(max(0.0, max_tilt_deg))
    cos_threshold = float(np.cos(np.deg2rad(max_tilt_deg)))
    horizontal_planes: List[PlaneParam] = []

    for i, plane in enumerate(planes):
        normal = np.asarray(plane.normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if not np.isfinite(normal_norm) or normal_norm < 1e-12:
            if verbose:
                print(f"  Plane[{i}] invalid normal, dropped")
            continue
        normal = normal / normal_norm
        if normal[2] < 0:
            normal = -normal

        nz = float(np.clip(normal[2], -1.0, 1.0))
        tilt_deg = float(np.rad2deg(np.arccos(nz)))
        height = float(plane.height) if plane.height is not None else float(np.asarray(plane.point, dtype=float)[2])
        keep = nz >= cos_threshold
        if verbose:
            print(
                f"  Plane[{i}] tilt={tilt_deg:5.1f}deg height={height:+.3f}m "
                f"inliers={plane.inlier_count:6d} -> {'KEEP' if keep else 'DROP'}"
            )
        if keep:
            horizontal_planes.append(
                PlaneParam(
                    normal=normal,
                    point=np.asarray(plane.point, dtype=float),
                    inlier_count=plane.inlier_count,
                    inlier_indices=None if plane.inlier_indices is None else np.asarray(plane.inlier_indices, dtype=int),
                    height=height,
                )
            )

    return horizontal_planes


def merge_planes_by_height(
    planes: List[PlaneParam],
    points: np.ndarray,
    *,
    height_eps: float = 0.03,
    distance_threshold: Optional[float] = None,
    verbose: bool = True,
) -> List[PlaneParam]:
    """
    Merge planes that are at similar heights (same stair step).

    Args:
        planes: List of PlaneParam to merge
        points: (N, 3) point cloud that inlier_indices refer to
        height_eps: Maximum height difference to consider same cluster (meters)
        distance_threshold: Optional inlier distance threshold for refit filtering
        verbose: If True, print merge logs

    Returns:
        List of merged planes (one per height cluster)
    """
    if len(planes) == 0:
        return []

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        if verbose:
            print("  merge_planes_by_height: invalid points, skipping refit and returning input planes")
        return sorted(planes, key=lambda p: (p.height if p.height is not None else float(p.point[2])))

    height_eps = float(max(0.0, height_eps))
    if distance_threshold is not None:
        distance_threshold = float(distance_threshold)
        if not np.isfinite(distance_threshold) or distance_threshold <= 0:
            distance_threshold = None

    def height_of(plane: PlaneParam) -> float:
        return float(plane.height) if plane.height is not None else float(np.asarray(plane.point, dtype=float)[2])

    # Sort by height
    sorted_planes = sorted(planes, key=height_of)

    # Cluster by height
    clusters: List[List[PlaneParam]] = []
    current_cluster: List[PlaneParam] = [sorted_planes[0]]

    for plane in sorted_planes[1:]:
        if height_of(plane) - height_of(current_cluster[-1]) <= height_eps:
            # Same cluster
            current_cluster.append(plane)
        else:
            # New cluster
            clusters.append(current_cluster)
            current_cluster = [plane]

    clusters.append(current_cluster)

    # Merge each cluster into a single plane
    merged_planes: List[PlaneParam] = []

    for cluster in clusters:
        cluster_indices_list: List[np.ndarray] = []
        for p in cluster:
            if p.inlier_indices is None:
                continue
            idx = np.asarray(p.inlier_indices, dtype=int).reshape(-1)
            if idx.size == 0:
                continue
            idx = idx[(0 <= idx) & (idx < len(points))]
            if idx.size == 0:
                continue
            cluster_indices_list.append(idx)

        if len(cluster_indices_list) == 0:
            # Fallback: keep the first plane if we cannot refit.
            merged_planes.append(cluster[0])
            continue

        cluster_indices = np.unique(np.concatenate(cluster_indices_list))
        cluster_points = points[cluster_indices]
        finite_mask = np.all(np.isfinite(cluster_points), axis=1)
        cluster_points = cluster_points[finite_mask]
        cluster_indices = cluster_indices[finite_mask]

        if len(cluster_points) < 3:
            merged_planes.append(cluster[0])
            continue

        # Refit plane using least squares on the merged inlier points.
        centroid = cluster_points.mean(axis=0)
        centered = cluster_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if not np.isfinite(normal_norm) or normal_norm < 1e-12:
            merged_planes.append(cluster[0])
            continue
        normal = normal / normal_norm
        if normal[2] < 0:
            normal = -normal

        if distance_threshold is not None:
            distances = np.abs(np.dot(cluster_points - centroid, normal))
            inlier_mask = distances <= distance_threshold
            if int(inlier_mask.sum()) >= 3 and int(inlier_mask.sum()) < len(cluster_points):
                cluster_points = cluster_points[inlier_mask]
                cluster_indices = cluster_indices[inlier_mask]
                centroid = cluster_points.mean(axis=0)
                centered = cluster_points - centroid
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                normal_norm = np.linalg.norm(normal)
                if not np.isfinite(normal_norm) or normal_norm < 1e-12:
                    merged_planes.append(cluster[0])
                    continue
                normal = normal / normal_norm
                if normal[2] < 0:
                    normal = -normal

        merged = PlaneParam(
            normal=normal,
            point=centroid,
            inlier_count=int(len(cluster_indices)),
            inlier_indices=cluster_indices,
            height=float(centroid[2]),
        )
        merged_planes.append(merged)

        if verbose:
            heights = [height_of(p) for p in cluster]
            print(
                f"  Merged {len(cluster)} planes: height {min(heights):+.3f}..{max(heights):+.3f}m "
                f"-> {merged.height:+.3f}m ({merged.inlier_count} inliers)"
            )

    return merged_planes


# =============================================================================
# Seed-expand plane extraction
# =============================================================================


@dataclass
class SeedExpandPlaneResult:
    """Result of seed-expand plane extraction."""
    plane: Optional[PlaneParam]
    expanded_inlier_indices: Optional[np.ndarray]
    expanded_inlier_count: int
    area: float
    extent_u: float
    extent_v: float
    seed_inlier_indices: Optional[np.ndarray]
    success: bool
    message: str
    seed_point_count: int = 0
    candidate_count: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    steps: int = 0
    max_frontier_size: int = 0
    residual_median: float = 0.0
    residual_p90: float = 0.0
    residual_p95: float = 0.0


def expand_plane_from_seed(
    points: np.ndarray,
    seed_center: np.ndarray,
    normals: Optional[np.ndarray] = None,
    *,
    seed_radius: float = 0.3,
    max_expand_radius: float = 5.0,
    grow_radius: float = 0.15,
    distance_threshold: float = 0.02,
    normal_threshold_deg: float = 30.0,
    expand_method: str = "component",
    max_refine_iters: int = 3,
    adaptive_refine_threshold: bool = False,
    adaptive_refine_k: float = 3.0,
    adaptive_refine_min_scale: float = 0.5,
    adaptive_refine_max_scale: float = 1.5,
    max_expanded_points: int = 200_000,
    max_frontier: int = 200_000,
    max_steps: int = 1_000_000,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    verbose: bool = True,
) -> SeedExpandPlaneResult:
    """
    Expand plane from seed point using region growing.

    Algorithm:
    1. Extract points within seed_radius from seed_center
    2. Fit initial plane using RANSAC on seed points
    3. Expand to find connected planar region using either:
       - component: find all candidates, build neighbor graph, extract seed's connected component
       - bfs: BFS from seed inliers, adding neighbors that satisfy plane/normal conditions

    Args:
        points: (N, 3) array of 3D points
        seed_center: (3,) seed center point
        normals: (N, 3) array of point normals (optional)
        seed_radius: radius for initial seed region
        max_expand_radius: maximum radius from seed center for expansion
        grow_radius: neighbor radius for connectivity (component) or BFS growth
        distance_threshold: max distance from plane to be considered inlier
        normal_threshold_deg: max angle between point normal and plane normal (if normals provided)
        expand_method: "component" or "bfs"
        max_refine_iters: number of plane refit iterations after expansion
        adaptive_refine_threshold: If True, adjust distance_threshold per refine step using median/MAD
        adaptive_refine_k: Robust scale multiplier for adaptive thresholding (median + k*sigma)
        adaptive_refine_min_scale: Minimum adaptive threshold as scale of distance_threshold
        adaptive_refine_max_scale: Maximum adaptive threshold as scale of distance_threshold
        ransac_n: RANSAC sample size for initial plane fit
        num_iterations: RANSAC iterations for initial plane fit
        verbose: print progress logs

    Returns:
        SeedExpandPlaneResult with expanded plane and metadata
    """
    points = np.asarray(points, dtype=float)
    seed_center = np.asarray(seed_center, dtype=float).flatten()
    time_start = perf_counter()

    if points.ndim != 2 or points.shape[1] != 3:
        return SeedExpandPlaneResult(
            plane=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            area=0.0, extent_u=0.0, extent_v=0.0, seed_inlier_indices=None,
            success=False, message="Invalid points shape"
        )

    if len(seed_center) != 3:
        return SeedExpandPlaneResult(
            plane=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            area=0.0, extent_u=0.0, extent_v=0.0, seed_inlier_indices=None,
            success=False, message="Invalid seed_center shape"
        )

    # Check normals
    has_normals = (
        normals is not None and
        len(normals) == len(points) and
        np.all(np.isfinite(normals))
    )
    if has_normals:
        normals = np.asarray(normals, dtype=float)
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normals_norm, 1e-8)
    normal_cos_threshold = np.cos(np.deg2rad(normal_threshold_deg))

    # Step 1: Extract seed points (squared distances for speed)
    delta = points - seed_center
    distances2_to_seed = np.einsum("ij,ij->i", delta, delta)
    seed_radius2 = float(seed_radius) * float(seed_radius)
    seed_mask = distances2_to_seed <= seed_radius2
    seed_indices = np.where(seed_mask)[0]

    if len(seed_indices) < 3:
        return SeedExpandPlaneResult(
            plane=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            area=0.0, extent_u=0.0, extent_v=0.0, seed_inlier_indices=seed_indices,
            success=False, message=f"Too few seed points: {len(seed_indices)}",
            seed_point_count=len(seed_indices),
        )

    seed_points = points[seed_indices]
    time_seed = perf_counter()
    if verbose:
        ms = (time_seed - time_start) * 1000.0
        print(f"  Seed region: {len(seed_points)} points within {seed_radius}m ({ms:.1f} ms)")

    # Step 2: Fit initial plane on seed points
    initial_plane = fit_plane(
        seed_points,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    if initial_plane is None:
        return SeedExpandPlaneResult(
            plane=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            area=0.0, extent_u=0.0, extent_v=0.0, seed_inlier_indices=seed_indices,
            success=False, message="Failed to fit initial plane on seed points",
            seed_point_count=len(seed_indices),
        )

    plane_normal = initial_plane.normal
    plane_point = initial_plane.point
    time_fit = perf_counter()

    if verbose:
        ms = (time_fit - time_seed) * 1000.0
        print(
            f"  Initial plane: normal={np.round(plane_normal, 3).tolist()}, "
            f"inliers={initial_plane.inlier_count}/{len(seed_points)} ({ms:.1f} ms)"
        )

    # Step 3: Expand from seed
    # Limit expansion to max_expand_radius (strictly subset before any neighbor search)
    max_expand_radius2 = float(max_expand_radius) * float(max_expand_radius)
    expand_mask = distances2_to_seed <= max_expand_radius2
    expand_indices = np.where(expand_mask)[0]
    expand_points = points[expand_indices]

    if verbose:
        print(f"  Expansion region: {len(expand_points)} points within {max_expand_radius}m")

    # Compute plane distance for all expansion candidates
    plane_distances = np.abs(np.dot(expand_points - plane_point, plane_normal))
    candidate_mask = plane_distances < distance_threshold

    # Optional: apply normal condition
    if has_normals:
        expand_normals = normals[expand_indices]
        normal_alignment = np.abs(np.dot(expand_normals, plane_normal))
        candidate_mask &= normal_alignment > normal_cos_threshold

    candidate_local_indices = np.where(candidate_mask)[0]
    candidate_global_indices = expand_indices[candidate_local_indices]
    time_candidates = perf_counter()

    if verbose:
        ms = (time_candidates - time_fit) * 1000.0
        print(
            f"  Plane candidates (dist < {distance_threshold}m): {len(candidate_local_indices)} "
            f"({ms:.1f} ms)"
        )

    if len(candidate_local_indices) == 0:
        # Return seed plane only
        return SeedExpandPlaneResult(
            plane=initial_plane,
            expanded_inlier_indices=seed_indices[initial_plane.inlier_indices] if initial_plane.inlier_indices is not None else seed_indices,
            expanded_inlier_count=initial_plane.inlier_count,
            area=0.0, extent_u=0.0, extent_v=0.0,
            seed_inlier_indices=seed_indices,
            success=True, message="No expansion candidates found, returning seed plane",
            seed_point_count=len(seed_indices),
            candidate_count=0,
        )

    # Map seed inliers to global indices
    if initial_plane.inlier_indices is not None:
        seed_inlier_global = seed_indices[initial_plane.inlier_indices]
    else:
        seed_inlier_global = seed_indices

    # Expand using selected method (both are connected-component extraction with different semantics)
    expanded_indices, cc_stats = _extract_connected_component(
        points,
        candidate_global_indices,
        seed_inlier_global,
        grow_radius,
        max_expanded_points=max_expanded_points,
        max_frontier=max_frontier,
        max_steps=max_steps,
        enforce_seed_in_component=True,
    )
    time_expand = perf_counter()

    if verbose:
        extra = f", stop={cc_stats.stop_reason}" if cc_stats.stopped_early else ""
        ms = (time_expand - time_candidates) * 1000.0
        print(
            f"  Expanded region ({expand_method}): {len(expanded_indices)} points "
            f"(seed_inliers={len(seed_inlier_global)}, frontier_max={cc_stats.max_frontier_size}, "
            f"steps={cc_stats.steps}{extra}) ({ms:.1f} ms)"
        )

    if len(expanded_indices) < 3:
        return SeedExpandPlaneResult(
            plane=initial_plane,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            area=0.0, extent_u=0.0, extent_v=0.0,
            seed_inlier_indices=seed_indices,
            success=True, message="Expansion resulted in too few points",
            seed_point_count=len(seed_indices),
            candidate_count=len(candidate_local_indices),
            stopped_early=cc_stats.stopped_early,
            stop_reason=cc_stats.stop_reason,
            steps=cc_stats.steps,
            max_frontier_size=cc_stats.max_frontier_size,
        )

    # Step 4: Iterative refinement
    current_indices = expanded_indices
    final_plane = initial_plane
    final_stats = cc_stats

    for refine_iter in range(max_refine_iters):
        iter_start = perf_counter()
        if len(current_indices) < 10:
            if verbose:
                print(f"    Refine iter {refine_iter + 1}: skipped (too few points: {len(current_indices)})")
            break
        # Refit plane using expanded inliers
        expanded_points = points[current_indices]
        centroid = expanded_points.mean(axis=0)
        centered = expanded_points - centroid
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        if s.size < 2 or not np.all(np.isfinite(s)):
            if verbose:
                print(f"    Refine iter {refine_iter + 1}: skipped (degenerate SVD)")
            break
        if s[0] < 1e-12 or (s[1] / s[0]) < 1e-3:
            if verbose:
                print(f"    Refine iter {refine_iter + 1}: skipped (degenerate points)")
            break
        new_normal = vh[-1]
        new_norm = np.linalg.norm(new_normal)
        if not np.isfinite(new_norm) or new_norm < 1e-12:
            if verbose:
                print(f"    Refine iter {refine_iter + 1}: skipped (invalid normal)")
            break
        new_normal = new_normal / new_norm
        new_normal = _orient_direction(new_normal, prefer_positive_z=True)

        refine_distance_threshold = float(distance_threshold)
        if adaptive_refine_threshold:
            residuals = np.abs((expanded_points - centroid) @ new_normal)
            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med)))
            sigma = 1.4826 * mad
            estimate = med + float(adaptive_refine_k) * sigma
            min_th = float(distance_threshold) * float(adaptive_refine_min_scale)
            max_th = float(distance_threshold) * float(adaptive_refine_max_scale)
            min_th = max(min_th, 1e-6)
            max_th = max(max_th, min_th)
            if np.isfinite(estimate):
                refine_distance_threshold = float(np.clip(estimate, min_th, max_th))

        # Re-expand with new plane
        plane_distances = np.abs(np.dot(expand_points - centroid, new_normal))
        candidate_mask = plane_distances < refine_distance_threshold
        if has_normals:
            expand_normals = normals[expand_indices]
            normal_alignment = np.abs(np.dot(expand_normals, new_normal))
            candidate_mask &= normal_alignment > normal_cos_threshold

        candidate_local_indices = np.where(candidate_mask)[0]
        candidate_global_indices = expand_indices[candidate_local_indices]

        new_indices, refine_stats = _extract_connected_component(
            points,
            candidate_global_indices,
            seed_inlier_global,
            grow_radius,
            max_expanded_points=max_expanded_points,
            max_frontier=max_frontier,
            max_steps=max_steps,
            enforce_seed_in_component=True,
        )
        iter_end = perf_counter()

        if verbose:
            extra = f", stop={refine_stats.stop_reason}" if refine_stats.stopped_early else ""
            th_info = (
                f", th={refine_distance_threshold:.4f}m"
                if adaptive_refine_threshold
                else ""
            )
            ms = (iter_end - iter_start) * 1000.0
            print(f"    Refine iter {refine_iter + 1}: {len(new_indices)} points{th_info}{extra} ({ms:.1f} ms)")
        final_stats = refine_stats

        # Check convergence
        if len(new_indices) == len(current_indices) and np.array_equal(np.sort(new_indices), np.sort(current_indices)):
            break

        current_indices = new_indices
        final_plane = PlaneParam(
            normal=new_normal,
            point=centroid,
            inlier_count=len(current_indices),
            inlier_indices=current_indices,
            height=float(centroid[2])
        )
        plane_normal = new_normal
        plane_point = centroid
        final_stats = refine_stats

    # Compute area and extent
    area, extent_u, extent_v = _compute_plane_metrics(points[current_indices], final_plane.normal)
    residuals = np.abs((points[current_indices] - final_plane.point) @ final_plane.normal)
    if residuals.size > 0:
        residual_median, residual_p90, residual_p95 = np.percentile(residuals, [50, 90, 95]).astype(float)
    else:
        residual_median = residual_p90 = residual_p95 = 0.0
    time_end = perf_counter()

    if verbose:
        ms_total = (time_end - time_start) * 1000.0
        print(
            f"  Final: {len(current_indices)} inliers, area={area:.3f}m², "
            f"extent=({extent_u:.2f} x {extent_v:.2f})m, "
            f"residual_median/p95={residual_median:.4f}/{residual_p95:.4f}m "
            f"({ms_total:.1f} ms)"
        )
        if final_stats.stopped_early:
            print(f"  WARNING: expansion stopped early ({final_stats.stop_reason})")
        if max_expanded_points > 0 and len(current_indices) >= max_expanded_points:
            print(f"  WARNING: hit max_expanded_points={max_expanded_points}")
        if max(extent_u, extent_v) > (2.0 * float(max_expand_radius) * 0.9):
            print("  WARNING: extent is close to max_expand_radius; check for leakage")
        area_max = np.pi * float(max_expand_radius) * float(max_expand_radius)
        if area_max > 0 and area > (area_max * 0.9):
            print("  WARNING: area is close to the max allowed by max_expand_radius; check for leakage")

    return SeedExpandPlaneResult(
        plane=PlaneParam(
            normal=final_plane.normal,
            point=final_plane.point,
            inlier_count=len(current_indices),
            inlier_indices=current_indices,
            height=float(final_plane.point[2])
        ),
        expanded_inlier_indices=current_indices,
        expanded_inlier_count=len(current_indices),
        area=area,
        extent_u=extent_u,
        extent_v=extent_v,
        seed_inlier_indices=seed_indices,
        success=True,
        message=(
            f"Plane expansion stopped early: {final_stats.stop_reason}"
            if final_stats.stopped_early
            else "Plane expansion successful"
        ),
        seed_point_count=len(seed_indices),
        candidate_count=len(candidate_local_indices),
        stopped_early=final_stats.stopped_early,
        stop_reason=final_stats.stop_reason,
        steps=final_stats.steps,
        max_frontier_size=final_stats.max_frontier_size,
        residual_median=float(residual_median),
        residual_p90=float(residual_p90),
        residual_p95=float(residual_p95),
    )


@dataclass
class _ConnectedComponentStats:
    stopped_early: bool
    stop_reason: str
    steps: int
    max_frontier_size: int


def _extract_connected_component(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    seed_indices: np.ndarray,
    grow_radius: float,
    *,
    max_expanded_points: int,
    max_frontier: int,
    max_steps: int,
    enforce_seed_in_component: bool,
) -> tuple[np.ndarray, _ConnectedComponentStats]:
    """Extract connected component from candidates using radius connectivity (spatial hashing)."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    seed_indices = np.asarray(seed_indices, dtype=int)
    if candidate_indices.size == 0 or seed_indices.size == 0:
        return seed_indices.copy(), _ConnectedComponentStats(
            stopped_early=False, stop_reason="", steps=0, max_frontier_size=0
        )

    grow_radius = float(grow_radius)
    if not np.isfinite(grow_radius) or grow_radius <= 0:
        return seed_indices.copy(), _ConnectedComponentStats(
            stopped_early=False, stop_reason="", steps=0, max_frontier_size=0
        )

    # Sort for deterministic outputs and efficient seed->local mapping.
    order = np.argsort(candidate_indices)
    cand_global = candidate_indices[order]
    cand_points = points[cand_global]

    seed_unique = np.unique(seed_indices)
    pos = np.searchsorted(cand_global, seed_unique)
    pos_ok = pos < cand_global.size
    if np.any(pos_ok):
        pos_valid = pos[pos_ok]
        seed_valid = seed_unique[pos_ok]
        match = cand_global[pos_valid] == seed_valid
        seed_local = pos_valid[match]
    else:
        seed_local = np.empty((0,), dtype=int)

    if seed_local.size == 0:
        reason = "seed_not_in_candidates" if enforce_seed_in_component else ""
        if enforce_seed_in_component:
            return seed_indices.copy(), _ConnectedComponentStats(
                stopped_early=False, stop_reason=reason, steps=0, max_frontier_size=0
            )
        # Fall back to empty component (caller can decide).
        return np.empty((0,), dtype=int), _ConnectedComponentStats(
            stopped_early=False, stop_reason=reason, steps=0, max_frontier_size=0
        )

    max_expanded_points = int(max_expanded_points)
    max_frontier = int(max_frontier)
    max_steps = int(max_steps)

    visited = np.zeros(cand_global.size, dtype=bool)
    visited[seed_local] = True
    visited_count = int(seed_local.size)

    q: deque[int] = deque(int(i) for i in seed_local)
    steps = 0
    max_frontier_size = len(q)
    stopped_early = False
    stop_reason = ""

    def should_stop_before_push() -> bool:
        nonlocal stopped_early, stop_reason
        if max_expanded_points > 0 and visited_count >= max_expanded_points:
            stopped_early = True
            stop_reason = "max_expanded_points"
            return True
        if max_frontier > 0 and len(q) >= max_frontier:
            stopped_early = True
            stop_reason = "max_frontier"
            return True
        return False

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        cKDTree = None

    if cKDTree is not None:
        tree = cKDTree(cand_points)

        while q:
            if max_steps > 0 and steps >= max_steps:
                stopped_early = True
                stop_reason = "max_steps"
                break
            if max_expanded_points > 0 and visited_count >= max_expanded_points:
                stopped_early = True
                stop_reason = "max_expanded_points"
                break

            max_frontier_size = max(max_frontier_size, len(q))

            current = q.popleft()
            steps += 1

            neighbors = tree.query_ball_point(cand_points[current], grow_radius, return_sorted=True)
            for nbr in neighbors:
                if visited[nbr]:
                    continue
                if should_stop_before_push():
                    break
                visited[nbr] = True
                visited_count += 1
                q.append(int(nbr))
            if stopped_early:
                break
    else:
        # Spatial hash grid fallback: cell size == grow_radius (27 neighboring cells cover all candidates).
        cell_size = grow_radius
        cell_coords = np.floor(cand_points / cell_size).astype(np.int64)
        cell_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for i, (cx, cy, cz) in enumerate(cell_coords):
            cell_to_indices.setdefault((int(cx), int(cy), int(cz)), []).append(i)

        grow_radius2 = grow_radius * grow_radius

        while q:
            if max_steps > 0 and steps >= max_steps:
                stopped_early = True
                stop_reason = "max_steps"
                break
            if max_expanded_points > 0 and visited_count >= max_expanded_points:
                stopped_early = True
                stop_reason = "max_expanded_points"
                break

            max_frontier_size = max(max_frontier_size, len(q))

            current = q.popleft()
            steps += 1

            cx, cy, cz = cell_coords[current]
            base_point = cand_points[current]

            stop_now = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        bucket = cell_to_indices.get((int(cx + dx), int(cy + dy), int(cz + dz)))
                        if not bucket:
                            continue
                        for nbr in bucket:
                            if visited[nbr]:
                                continue
                            diff = cand_points[nbr] - base_point
                            if float(diff @ diff) <= grow_radius2:
                                if should_stop_before_push():
                                    stop_now = True
                                    break
                                visited[nbr] = True
                                visited_count += 1
                                q.append(nbr)
                        if stop_now:
                            break
                    if stop_now:
                        break
                if stop_now:
                    break
            if stop_now:
                break

    component_local = np.flatnonzero(visited)
    component_global = cand_global[component_local]
    return component_global, _ConnectedComponentStats(
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        steps=steps,
        max_frontier_size=max_frontier_size,
    )


def _compute_plane_metrics(inlier_points: np.ndarray, normal: np.ndarray) -> tuple:
    """Compute area and extent of plane inliers."""
    if len(inlier_points) < 3:
        return 0.0, 0.0, 0.0

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Compute plane basis
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project to 2D
    centroid = inlier_points.mean(axis=0)
    local = inlier_points - centroid
    coords_2d = np.column_stack((local @ u, local @ v))

    extent_u = float(coords_2d[:, 0].max() - coords_2d[:, 0].min())
    extent_v = float(coords_2d[:, 1].max() - coords_2d[:, 1].min())

    # Compute convex hull area
    try:
        from scipy.spatial import ConvexHull
        if len(coords_2d) >= 3:
            hull = ConvexHull(coords_2d)
            area = float(hull.volume)  # In 2D, volume is area
        else:
            area = 0.0
    except Exception:
        # Fallback: bounding box area
        area = extent_u * extent_v

    return area, extent_u, extent_v


# =============================================================================
# Seed-expand cylinder extraction
# =============================================================================


@dataclass
class SeedExpandCylinderResult:
    """Result of seed-expand cylinder extraction."""
    cylinder: Optional[CylinderParam]
    expanded_inlier_indices: Optional[np.ndarray]
    expanded_inlier_count: int
    seed_inlier_indices: Optional[np.ndarray]
    success: bool
    message: str
    seed_point_count: int = 0
    candidate_count: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    steps: int = 0
    max_frontier_size: int = 0
    residual_median: float = 0.0
    residual_mad: float = 0.0
    residual_p90: float = 0.0
    residual_p95: float = 0.0


@dataclass
class CylinderProxyInit:
    """Initial proxy cylinder estimate from a seed point."""
    cylinder: Optional[CylinderParam]
    seed_indices: np.ndarray
    seed_radius: float
    axis_dir_pca: Optional[np.ndarray]
    seed_point_count: int
    success: bool
    message: str


@dataclass
class CylinderSurfaceExtractResult:
    """Result of extracting cylinder surface points with connectivity."""
    inlier_indices: np.ndarray
    candidate_count: int
    seed_inlier_count: int
    stopped_early: bool
    stop_reason: str
    steps: int
    max_frontier_size: int
    success: bool
    message: str


@dataclass
class CylinderProbeResult:
    """Result of interactive cylinder probe extraction."""
    proxy: Optional[CylinderParam]
    final: Optional[CylinderParam]
    inlier_indices: Optional[np.ndarray]
    inlier_count: int
    residual_median: float
    residual_mad: float
    candidate_count: int
    stopped_early: bool
    stop_reason: str
    steps: int
    max_frontier_size: int
    success: bool
    message: str
    seed_radius: float = 0.0
    seed_point_count: int = 0


@dataclass
class AutoSelectResult:
    """Auto selection result for plane vs cylinder."""
    chosen: str
    plane_score: float
    cylinder_score: float
    reason: str


def expand_cylinder_from_seed(
    points: np.ndarray,
    seed_center: np.ndarray,
    normals: Optional[np.ndarray] = None,
    *,
    initial_cylinder: Optional[CylinderParam] = None,
    seed_radius: float = 0.3,
    max_expand_radius: float = 5.0,
    grow_radius: float = 0.15,
    distance_threshold: float = 0.02,
    normal_threshold_deg: float = 30.0,
    expand_method: str = "component",
    max_refine_iters: int = 3,
    max_expanded_points: int = 200_000,
    max_frontier: int = 200_000,
    max_steps: int = 1_000_000,
    radius_min: float = 0.01,
    radius_max: float = 1.0,
    num_iterations: int = 1000,
    verbose: bool = True,
) -> SeedExpandCylinderResult:
    """
    Expand cylinder from seed point using region growing.

    Args:
        points: (N, 3) array of 3D points
        seed_center: (3,) seed center point
        normals: (N, 3) array of point normals (optional)
        initial_cylinder: Optional initial cylinder prior (axis/radius) to skip seed RANSAC
        seed_radius: radius for initial seed region
        max_expand_radius: maximum radius from seed center for expansion
        grow_radius: neighbor radius for connectivity
        distance_threshold: max distance from cylinder surface for inliers
        normal_threshold_deg: max angle between point normal and radial direction
        expand_method: "component" or "bfs"
        radius_min: minimum cylinder radius
        radius_max: maximum cylinder radius
        num_iterations: RANSAC iterations for initial fit
        verbose: print progress logs

    Returns:
        SeedExpandCylinderResult with expanded cylinder and metadata
    """
    points = np.asarray(points, dtype=float)
    seed_center = np.asarray(seed_center, dtype=float).flatten()
    time_start = perf_counter()

    if points.ndim != 2 or points.shape[1] != 3:
        return SeedExpandCylinderResult(
            cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            seed_inlier_indices=None, success=False, message="Invalid points shape"
        )

    if len(seed_center) != 3:
        return SeedExpandCylinderResult(
            cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            seed_inlier_indices=None, success=False, message="Invalid seed_center shape"
        )

    # Check normals
    has_normals = (
        normals is not None and
        len(normals) == len(points) and
        np.all(np.isfinite(normals))
    )
    if has_normals:
        normals = np.asarray(normals, dtype=float)
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normals_norm, 1e-8)
    normal_cos_threshold = np.cos(np.deg2rad(normal_threshold_deg))

    # Step 1: Extract seed points (squared distances for speed)
    delta = points - seed_center
    distances2_to_seed = np.einsum("ij,ij->i", delta, delta)
    seed_radius2 = float(seed_radius) * float(seed_radius)
    seed_mask = distances2_to_seed <= seed_radius2
    seed_indices = np.where(seed_mask)[0]

    if len(seed_indices) < 6 and initial_cylinder is None:
        return SeedExpandCylinderResult(
            cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            seed_inlier_indices=seed_indices, success=False,
            message=f"Too few seed points: {len(seed_indices)}",
            seed_point_count=len(seed_indices),
        )

    seed_points = points[seed_indices]
    seed_normals = normals[seed_indices] if has_normals else None
    time_seed = perf_counter()

    if verbose:
        ms = (time_seed - time_start) * 1000.0
        print(f"  Seed region: {len(seed_points)} points within {seed_radius}m ({ms:.1f} ms)")

    # Step 2: Fit initial cylinder on seed points (or use provided prior)
    if initial_cylinder is None:
        initial_cylinder = fit_cylinder(
            seed_points,
            seed_normals,
            distance_threshold=distance_threshold,
            radius_min=radius_min,
            radius_max=radius_max,
            num_iterations=num_iterations
        )

        if initial_cylinder is None:
            return SeedExpandCylinderResult(
                cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
                seed_inlier_indices=seed_indices, success=False,
                message="Failed to fit initial cylinder on seed points",
                seed_point_count=len(seed_indices),
            )
    else:
        axis_dir = np.asarray(initial_cylinder.axis_direction, dtype=float).reshape(-1)
        axis_point = np.asarray(initial_cylinder.axis_point, dtype=float).reshape(-1)
        if axis_dir.size != 3 or axis_point.size != 3:
            return SeedExpandCylinderResult(
                cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
                seed_inlier_indices=seed_indices, success=False,
                message="Initial cylinder axis must be 3D",
                seed_point_count=len(seed_indices),
            )
        axis_norm = np.linalg.norm(axis_dir)
        if not np.isfinite(axis_norm) or axis_norm < 1e-12:
            return SeedExpandCylinderResult(
                cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
                seed_inlier_indices=seed_indices, success=False,
                message="Initial cylinder axis direction is invalid",
                seed_point_count=len(seed_indices),
            )
        axis_dir = axis_dir / axis_norm
        axis_dir = _orient_direction(axis_dir, prefer_positive_z=False)
        cyl_radius = float(initial_cylinder.radius)
        if not np.isfinite(cyl_radius) or cyl_radius <= 0:
            return SeedExpandCylinderResult(
                cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
                seed_inlier_indices=seed_indices, success=False,
                message="Initial cylinder radius is invalid",
                seed_point_count=len(seed_indices),
            )
        cyl_length = float(initial_cylinder.length) if np.isfinite(initial_cylinder.length) else 0.0
        initial_cylinder = CylinderParam(
            axis_point=np.asarray(axis_point, dtype=float),
            axis_direction=np.asarray(axis_dir, dtype=float),
            radius=cyl_radius,
            length=cyl_length,
            inlier_count=0,
            inlier_indices=None,
        )

    axis_point = initial_cylinder.axis_point
    axis_dir = initial_cylinder.axis_direction
    cyl_radius = initial_cylinder.radius
    time_fit = perf_counter()

    if verbose:
        ms = (time_fit - time_seed) * 1000.0
        print(
            f"  Initial cylinder: radius={cyl_radius:.4f}m, "
            f"axis_dir={np.round(axis_dir, 3).tolist()}, "
            f"inliers={initial_cylinder.inlier_count}/{len(seed_points)} ({ms:.1f} ms)"
        )

    # Step 3: Expand from seed (strictly limit by max_expand_radius before any neighbor search)
    max_expand_radius2 = float(max_expand_radius) * float(max_expand_radius)
    expand_mask = distances2_to_seed <= max_expand_radius2
    expand_indices = np.where(expand_mask)[0]
    expand_points = points[expand_indices]

    if verbose:
        print(f"  Expansion region: {len(expand_points)} points within {max_expand_radius}m")

    # Compute cylinder distance for all expansion candidates
    diff = expand_points - axis_point
    projections = diff @ axis_dir
    radial_vec = diff - np.outer(projections, axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    cylinder_dist = np.abs(radial_dist - cyl_radius)

    candidate_mask = cylinder_dist < distance_threshold

    # Optional: apply normal condition
    if has_normals:
        expand_normals = normals[expand_indices]
        radial_dir = radial_vec / np.maximum(radial_dist[:, None], 1e-8)
        normal_alignment = np.abs(np.einsum('ij,ij->i', radial_dir, expand_normals))
        candidate_mask &= normal_alignment > normal_cos_threshold

    candidate_local_indices = np.where(candidate_mask)[0]
    candidate_global_indices = expand_indices[candidate_local_indices]
    time_candidates = perf_counter()

    if verbose:
        ms = (time_candidates - time_fit) * 1000.0
        print(
            f"  Cylinder candidates (dist < {distance_threshold}m): {len(candidate_local_indices)} "
            f"({ms:.1f} ms)"
        )

    seed_inlier_global = seed_indices
    if initial_cylinder.inlier_indices is not None:
        seed_inlier_global = seed_indices[initial_cylinder.inlier_indices]
    elif seed_indices.size > 0:
        diff_seed = seed_points - axis_point
        seed_proj = diff_seed @ axis_dir
        radial_seed = diff_seed - np.outer(seed_proj, axis_dir)
        radial_seed_dist = np.linalg.norm(radial_seed, axis=1)
        seed_dist = np.abs(radial_seed_dist - cyl_radius)
        seed_mask = seed_dist < distance_threshold
        if has_normals:
            radial_dir_seed = radial_seed / np.maximum(radial_seed_dist[:, None], 1e-8)
            seed_normal_alignment = np.abs(np.einsum("ij,ij->i", radial_dir_seed, seed_normals))
            seed_mask &= seed_normal_alignment > normal_cos_threshold
        seed_inlier_global = seed_indices[np.where(seed_mask)[0]]

    if seed_inlier_global.size == 0 and candidate_global_indices.size > 0:
        candidate_points = points[candidate_global_indices]
        delta_seed = candidate_points - seed_center
        distances2 = np.einsum("ij,ij->i", delta_seed, delta_seed)
        nearest = int(np.argmin(distances2))
        seed_inlier_global = np.array([candidate_global_indices[nearest]], dtype=int)
        if verbose:
            print("  Seed had no cylinder inliers; using nearest candidate as seed")

    if len(candidate_local_indices) == 0:
        return SeedExpandCylinderResult(
            cylinder=initial_cylinder,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            seed_inlier_indices=seed_indices,
            success=True, message="No expansion candidates found, returning seed cylinder",
            seed_point_count=len(seed_indices),
            candidate_count=0,
        )

    expanded_indices, cc_stats = _extract_connected_component(
        points,
        candidate_global_indices,
        seed_inlier_global,
        grow_radius,
        max_expanded_points=max_expanded_points,
        max_frontier=max_frontier,
        max_steps=max_steps,
        enforce_seed_in_component=True,
    )
    time_expand = perf_counter()

    if verbose:
        extra = f", stop={cc_stats.stop_reason}" if cc_stats.stopped_early else ""
        ms = (time_expand - time_candidates) * 1000.0
        print(
            f"  Expanded region ({expand_method}): {len(expanded_indices)} points "
            f"(seed_inliers={len(seed_inlier_global)}, frontier_max={cc_stats.max_frontier_size}, "
            f"steps={cc_stats.steps}{extra}) ({ms:.1f} ms)"
        )

    if len(expanded_indices) < 6:
        return SeedExpandCylinderResult(
            cylinder=initial_cylinder,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            seed_inlier_indices=seed_indices,
            success=True, message="Expansion resulted in too few points",
            seed_point_count=len(seed_indices),
            candidate_count=len(candidate_local_indices),
            stopped_early=cc_stats.stopped_early,
            stop_reason=cc_stats.stop_reason,
            steps=cc_stats.steps,
            max_frontier_size=cc_stats.max_frontier_size,
        )

    # Recompute cylinder parameters from expanded inliers (1-2 refinement rounds)
    refined_indices = expanded_indices
    final_axis_point = axis_point
    final_axis_dir = axis_dir
    final_radius = float(cyl_radius)
    final_length = float(initial_cylinder.length)

    refine_rounds = max(1, min(int(max_refine_iters), 2))
    for refine_iter in range(refine_rounds):
        iter_start = perf_counter()
        refined_points = points[refined_indices]
        centroid = refined_points.mean(axis=0)
        centered = refined_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        new_axis_dir = vh[0]
        new_axis_dir = new_axis_dir / np.linalg.norm(new_axis_dir)

        # Orient consistently to initial axis direction
        if np.dot(new_axis_dir, axis_dir) < 0:
            new_axis_dir = -new_axis_dir

        diff = refined_points - centroid
        projections = diff @ new_axis_dir
        radial_vec = diff - np.outer(projections, new_axis_dir)
        radial_dist = np.linalg.norm(radial_vec, axis=1)
        new_radius = float(np.median(radial_dist))
        new_length = float(np.max(projections) - np.min(projections))

        # Re-inlier selection using updated cylinder on the full expansion region
        diff_all = expand_points - centroid
        projections_all = diff_all @ new_axis_dir
        radial_vec_all = diff_all - np.outer(projections_all, new_axis_dir)
        radial_dist_all = np.linalg.norm(radial_vec_all, axis=1)
        cylinder_dist = np.abs(radial_dist_all - new_radius)
        new_candidate_mask = cylinder_dist < distance_threshold
        if has_normals:
            expand_normals = normals[expand_indices]
            radial_dir = radial_vec_all / np.maximum(radial_dist_all[:, None], 1e-8)
            normal_alignment = np.abs(np.einsum("ij,ij->i", radial_dir, expand_normals))
            new_candidate_mask &= normal_alignment > normal_cos_threshold

        new_candidate_local = np.where(new_candidate_mask)[0]
        new_candidate_global = expand_indices[new_candidate_local]
        new_refined_indices, refine_stats = _extract_connected_component(
            points,
            new_candidate_global,
            seed_inlier_global,
            grow_radius,
            max_expanded_points=max_expanded_points,
            max_frontier=max_frontier,
            max_steps=max_steps,
            enforce_seed_in_component=True,
        )
        iter_end = perf_counter()

        if verbose:
            extra = f", stop={refine_stats.stop_reason}" if refine_stats.stopped_early else ""
            ms = (iter_end - iter_start) * 1000.0
            print(
                f"    Refine iter {refine_iter + 1}: {len(new_refined_indices)} points, "
                f"radius={new_radius:.4f}m, length={new_length:.3f}m{extra} ({ms:.1f} ms)"
            )

        # Convergence: stable membership.
        if (
            len(new_refined_indices) == len(refined_indices)
            and np.array_equal(np.sort(new_refined_indices), np.sort(refined_indices))
        ):
            refined_indices = new_refined_indices
            final_axis_point = centroid
            final_axis_dir = new_axis_dir
            final_radius = new_radius
            final_length = new_length
            cc_stats = refine_stats
            break

        refined_indices = new_refined_indices
        final_axis_point = centroid
        final_axis_dir = new_axis_dir
        final_radius = new_radius
        final_length = new_length
        cc_stats = refine_stats

    inlier_points = points[refined_indices]
    diff = inlier_points - final_axis_point
    proj = diff @ final_axis_dir
    radial_vec = diff - np.outer(proj, final_axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    residuals = np.abs(radial_dist - final_radius)
    if residuals.size > 0:
        residual_median, residual_p90, residual_p95 = np.percentile(residuals, [50, 90, 95]).astype(float)
        residual_mad = float(np.median(np.abs(residuals - residual_median)))
    else:
        residual_median = residual_p90 = residual_p95 = 0.0
        residual_mad = 0.0
    time_end = perf_counter()

    if verbose:
        ms_total = (time_end - time_start) * 1000.0
        print(
            f"  Final: {len(refined_indices)} inliers, radius={final_radius:.4f}m, "
            f"length={final_length:.3f}m, residual_median/p95={residual_median:.4f}/{residual_p95:.4f}m "
            f"({ms_total:.1f} ms)"
        )
        if cc_stats.stopped_early:
            print(f"  WARNING: expansion stopped early ({cc_stats.stop_reason})")
        if max_expanded_points > 0 and len(refined_indices) >= max_expanded_points:
            print(f"  WARNING: hit max_expanded_points={max_expanded_points}")
        if final_length > (float(max_expand_radius) * 1.8):
            print("  WARNING: length is close to max_expand_radius; check for leakage")
        if residual_p95 > (float(distance_threshold) * 1.2):
            print("  WARNING: residual p95 is close to distance_threshold; consider tuning thresholds")

    final_cylinder = CylinderParam(
        axis_point=final_axis_point,
        axis_direction=final_axis_dir,
        radius=final_radius,
        length=final_length,
        inlier_count=len(refined_indices),
        inlier_indices=refined_indices,
    )

    return SeedExpandCylinderResult(
        cylinder=final_cylinder,
        expanded_inlier_indices=refined_indices,
        expanded_inlier_count=len(refined_indices),
        seed_inlier_indices=seed_indices,
        success=True,
        message=(
            f"Cylinder expansion stopped early: {cc_stats.stop_reason}"
            if cc_stats.stopped_early
            else "Cylinder expansion successful"
        ),
        seed_point_count=len(seed_indices),
        candidate_count=len(candidate_local_indices),
        stopped_early=cc_stats.stopped_early,
        stop_reason=cc_stats.stop_reason,
        steps=cc_stats.steps,
        max_frontier_size=cc_stats.max_frontier_size,
        residual_median=float(residual_median),
        residual_mad=float(residual_mad),
        residual_p90=float(residual_p90),
        residual_p95=float(residual_p95),
    )


def _orthonormal_basis_from_axis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis (u, v) perpendicular to axis_dir."""
    axis_dir = np.asarray(axis_dir, dtype=float).reshape(-1)
    axis_dir = axis_dir / np.maximum(np.linalg.norm(axis_dir), 1e-8)
    if abs(axis_dir[2]) < 0.9:
        u = np.cross(axis_dir, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(axis_dir, np.array([1.0, 0.0, 0.0]))
    u = u / np.maximum(np.linalg.norm(u), 1e-8)
    v = np.cross(axis_dir, u)
    v = v / np.maximum(np.linalg.norm(v), 1e-8)
    return u, v


def _circle_from_3pts(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Compute circle center/radius from 3 points in 2D."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = 2.0 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
    if abs(d) < 1e-12:
        return None
    p0_sq = p0[0] * p0[0] + p0[1] * p0[1]
    p1_sq = p1[0] * p1[0] + p1[1] * p1[1]
    p2_sq = p2[0] * p2[0] + p2[1] * p2[1]
    ux = (p0_sq * (p1[1] - p2[1]) + p1_sq * (p2[1] - p0[1]) + p2_sq * (p0[1] - p1[1])) / d
    uy = (p0_sq * (p2[0] - p1[0]) + p1_sq * (p0[0] - p2[0]) + p2_sq * (p1[0] - p0[0])) / d
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(center - p0))
    if not np.isfinite(radius) or radius <= 0:
        return None
    return center, radius


def _fit_circle_2d_least_squares(points_2d: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Least-squares circle fit in 2D."""
    pts = np.asarray(points_2d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, c = sol
    r_sq = c + cx * cx + cy * cy
    if not np.isfinite(r_sq) or r_sq <= 0:
        return None
    center = np.array([cx, cy], dtype=float)
    radius = float(np.sqrt(r_sq))
    return center, radius


def _fit_circle_2d_ransac(
    points_2d: np.ndarray,
    *,
    num_iterations: int = 200,
    inlier_threshold: float = 0.01,
    min_inliers: int = 6,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """RANSAC circle fit with least-squares refinement."""
    pts = np.asarray(points_2d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None
    n = len(pts)
    rng = np.random.default_rng()

    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    best_median = float("inf")
    best_center = None
    best_radius = None

    for _ in range(max(1, int(num_iterations))):
        sample = rng.choice(n, size=3, replace=False)
        circle = _circle_from_3pts(pts[sample[0]], pts[sample[1]], pts[sample[2]])
        if circle is None:
            continue
        center, radius = circle
        residuals = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        inliers = residuals < float(inlier_threshold)
        count = int(np.count_nonzero(inliers))
        median_res = float(np.median(residuals)) if residuals.size > 0 else float("inf")
        if count > best_count or (count == best_count and median_res < best_median):
            best_inliers = inliers
            best_count = count
            best_median = median_res
            best_center = center
            best_radius = radius

    if best_inliers is None:
        ls = _fit_circle_2d_least_squares(pts)
        if ls is None:
            return None
        center, radius = ls
        inliers = np.ones((n,), dtype=bool)
        return center, radius, inliers

    if best_count >= min_inliers:
        ls = _fit_circle_2d_least_squares(pts[best_inliers])
        if ls is not None:
            best_center, best_radius = ls

    return best_center, float(best_radius), best_inliers


def _estimate_cylinder_from_points(
    points: np.ndarray,
    *,
    axis_dir: Optional[np.ndarray] = None,
    circle_ransac_iters: int = 200,
    circle_inlier_threshold: float = 0.01,
    length_margin: float = 0.05,
) -> Optional[CylinderParam]:
    """Estimate cylinder parameters from points given an axis direction."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 6:
        return None

    if axis_dir is None:
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_dir = vh[0]
    axis_dir = np.asarray(axis_dir, dtype=float).reshape(-1)
    axis_norm = np.linalg.norm(axis_dir)
    if not np.isfinite(axis_norm) or axis_norm < 1e-12:
        return None
    axis_dir = axis_dir / axis_norm
    axis_dir = _orient_direction(axis_dir, prefer_positive_z=False)

    projections = pts @ axis_dir
    plane_points = pts - np.outer(projections, axis_dir)
    origin = plane_points.mean(axis=0)
    u, v = _orthonormal_basis_from_axis(axis_dir)
    coords = np.column_stack([(plane_points - origin) @ u, (plane_points - origin) @ v])

    circle_fit = _fit_circle_2d_ransac(
        coords,
        num_iterations=circle_ransac_iters,
        inlier_threshold=circle_inlier_threshold,
        min_inliers=6,
    )
    if circle_fit is None:
        return None

    center_2d, radius, inliers = circle_fit
    axis_point = origin + center_2d[0] * u + center_2d[1] * v

    t = (pts - axis_point) @ axis_dir
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    raw_length = max(0.0, t_max - t_min)
    margin = max(float(length_margin), raw_length * 0.05)
    length = float(raw_length + 2.0 * margin)
    center_shift = 0.5 * (t_min + t_max)
    axis_point = axis_point + axis_dir * center_shift

    residuals = np.abs(np.linalg.norm(pts - axis_point - np.outer((pts - axis_point) @ axis_dir, axis_dir), axis=1) - radius)
    inlier_mask = residuals < float(circle_inlier_threshold)
    inlier_indices = np.where(inlier_mask)[0] if inliers is None else np.where(inlier_mask)[0]

    return CylinderParam(
        axis_point=axis_point,
        axis_direction=axis_dir,
        radius=float(radius),
        length=length,
        inlier_count=int(len(inlier_indices)),
        inlier_indices=inlier_indices,
    )


def adaptive_seed_indices(
    points: np.ndarray,
    seed_center: np.ndarray,
    *,
    seed_radius_start: float,
    seed_radius_max: float,
    seed_radius_step: float,
    min_seed_points: int,
) -> Tuple[np.ndarray, float]:
    """Collect seed indices by expanding radius until min_seed_points or max."""
    pts = np.asarray(points, dtype=float)
    seed_center = np.asarray(seed_center, dtype=float).reshape(-1)
    if seed_center.size != 3:
        return np.empty((0,), dtype=int), 0.0
    radius = float(seed_radius_start)
    radius_max = float(seed_radius_max)
    step = max(float(seed_radius_step), 1e-6)
    min_points = max(int(min_seed_points), 1)

    best_indices = np.empty((0,), dtype=int)
    tree = None
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
    except Exception:
        tree = None
    while radius <= radius_max + 1e-9:
        if tree is not None:
            k, idx, _ = tree.search_radius_vector_3d(seed_center, radius)
            indices = np.array(idx[:k], dtype=int)
        else:
            diff = pts - seed_center
            dist2 = np.einsum("ij,ij->i", diff, diff)
            mask = dist2 <= radius * radius
            indices = np.where(mask)[0]
        best_indices = indices
        if len(indices) >= min_points:
            break
        radius += step

    return best_indices, float(min(radius, radius_max))


def compute_cylinder_proxy_from_seed(
    points: np.ndarray,
    seed_center: np.ndarray,
    *,
    seed_radius_start: float = 0.05,
    seed_radius_max: float = 0.5,
    seed_radius_step: float = 0.05,
    min_seed_points: int = 80,
    circle_ransac_iters: int = 200,
    circle_inlier_threshold: float = 0.01,
    length_margin: float = 0.05,
) -> CylinderProxyInit:
    """Estimate a proxy cylinder from an adaptive seed region."""
    seed_indices, seed_radius = adaptive_seed_indices(
        points,
        seed_center,
        seed_radius_start=seed_radius_start,
        seed_radius_max=seed_radius_max,
        seed_radius_step=seed_radius_step,
        min_seed_points=min_seed_points,
    )
    if seed_indices.size < 6:
        return CylinderProxyInit(
            cylinder=None,
            seed_indices=seed_indices,
            seed_radius=seed_radius,
            axis_dir_pca=None,
            seed_point_count=int(seed_indices.size),
            success=False,
            message=f"Too few seed points: {len(seed_indices)}",
        )

    seed_points = np.asarray(points, dtype=float)[seed_indices]
    centroid = seed_points.mean(axis=0)
    centered = seed_points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_dir_pca = _orient_direction(vh[0], prefer_positive_z=False)

    cylinder = _estimate_cylinder_from_points(
        seed_points,
        axis_dir=axis_dir_pca,
        circle_ransac_iters=circle_ransac_iters,
        circle_inlier_threshold=circle_inlier_threshold,
        length_margin=length_margin,
    )
    if cylinder is None:
        return CylinderProxyInit(
            cylinder=None,
            seed_indices=seed_indices,
            seed_radius=seed_radius,
            axis_dir_pca=axis_dir_pca,
            seed_point_count=int(seed_indices.size),
            success=False,
            message="Failed to fit proxy cylinder from seed points",
        )

    return CylinderProxyInit(
        cylinder=cylinder,
        seed_indices=seed_indices,
        seed_radius=seed_radius,
        axis_dir_pca=axis_dir_pca,
        seed_point_count=int(seed_indices.size),
        success=True,
        message="Proxy cylinder initialized",
    )


def extract_cylinder_surface_component(
    points: np.ndarray,
    seed_center: np.ndarray,
    seed_indices: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    length: float,
    *,
    surface_threshold: float,
    cap_margin: float,
    grow_radius: float,
    max_expand_radius: float,
    max_expanded_points: int,
    max_frontier: int,
    max_steps: int,
) -> CylinderSurfaceExtractResult:
    """Extract a connected set of points on a cylinder surface."""
    pts = np.asarray(points, dtype=float)
    axis_point = np.asarray(axis_point, dtype=float).reshape(-1)
    axis_dir = np.asarray(axis_dir, dtype=float).reshape(-1)
    seed_indices = np.asarray(seed_indices, dtype=int).reshape(-1)
    if axis_point.size != 3 or axis_dir.size != 3:
        return CylinderSurfaceExtractResult(
            inlier_indices=np.empty((0,), dtype=int),
            candidate_count=0,
            seed_inlier_count=0,
            stopped_early=False,
            stop_reason="invalid_axis",
            steps=0,
            max_frontier_size=0,
            success=False,
            message="Invalid axis for cylinder extraction",
        )

    axis_norm = np.linalg.norm(axis_dir)
    if not np.isfinite(axis_norm) or axis_norm < 1e-12:
        return CylinderSurfaceExtractResult(
            inlier_indices=np.empty((0,), dtype=int),
            candidate_count=0,
            seed_inlier_count=0,
            stopped_early=False,
            stop_reason="invalid_axis",
            steps=0,
            max_frontier_size=0,
            success=False,
            message="Invalid axis direction for cylinder extraction",
        )
    axis_dir = axis_dir / axis_norm

    max_expand_radius = float(max_expand_radius)
    if max_expand_radius > 0:
        delta = pts - np.asarray(seed_center, dtype=float).reshape(-1)
        dist2 = np.einsum("ij,ij->i", delta, delta)
        expand_mask = dist2 <= max_expand_radius * max_expand_radius
        expand_indices = np.where(expand_mask)[0]
    else:
        expand_indices = np.arange(len(pts), dtype=int)

    if expand_indices.size == 0:
        return CylinderSurfaceExtractResult(
            inlier_indices=np.empty((0,), dtype=int),
            candidate_count=0,
            seed_inlier_count=0,
            stopped_early=False,
            stop_reason="no_expand_points",
            steps=0,
            max_frontier_size=0,
            success=False,
            message="No points within max_expand_radius",
        )

    expand_points = pts[expand_indices]
    diff = expand_points - axis_point
    t = diff @ axis_dir
    radial_vec = diff - np.outer(t, axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    half_len = 0.5 * float(length)
    cap_margin = float(cap_margin)

    candidate_mask = (
        np.abs(radial_dist - float(radius)) < float(surface_threshold)
    ) & (
        t >= -half_len - cap_margin
    ) & (
        t <= half_len + cap_margin
    )
    candidate_local = np.where(candidate_mask)[0]
    candidate_indices = expand_indices[candidate_local]

    if candidate_indices.size == 0:
        return CylinderSurfaceExtractResult(
            inlier_indices=np.empty((0,), dtype=int),
            candidate_count=0,
            seed_inlier_count=0,
            stopped_early=False,
            stop_reason="no_candidates",
            steps=0,
            max_frontier_size=0,
            success=False,
            message="No candidate points on cylinder surface",
        )

    seed_inlier_indices = np.empty((0,), dtype=int)
    if seed_indices.size > 0:
        seed_points = pts[seed_indices]
        diff_seed = seed_points - axis_point
        t_seed = diff_seed @ axis_dir
        radial_seed = diff_seed - np.outer(t_seed, axis_dir)
        radial_seed_dist = np.linalg.norm(radial_seed, axis=1)
        seed_mask = (
            np.abs(radial_seed_dist - float(radius)) < float(surface_threshold)
        ) & (
            t_seed >= -half_len - cap_margin
        ) & (
            t_seed <= half_len + cap_margin
        )
        seed_inlier_indices = seed_indices[np.where(seed_mask)[0]]

    if seed_inlier_indices.size == 0:
        candidate_points = pts[candidate_indices]
        delta = candidate_points - np.asarray(seed_center, dtype=float).reshape(-1)
        dist2 = np.einsum("ij,ij->i", delta, delta)
        nearest = int(np.argmin(dist2))
        seed_inlier_indices = np.array([candidate_indices[nearest]], dtype=int)

    component, cc_stats = _extract_connected_component(
        pts,
        candidate_indices,
        seed_inlier_indices,
        grow_radius,
        max_expanded_points=max_expanded_points,
        max_frontier=max_frontier,
        max_steps=max_steps,
        enforce_seed_in_component=True,
    )

    return CylinderSurfaceExtractResult(
        inlier_indices=component,
        candidate_count=int(candidate_indices.size),
        seed_inlier_count=int(seed_inlier_indices.size),
        stopped_early=bool(cc_stats.stopped_early),
        stop_reason=str(cc_stats.stop_reason),
        steps=int(cc_stats.steps),
        max_frontier_size=int(cc_stats.max_frontier_size),
        success=True,
        message="Cylinder surface extraction OK",
    )


def _compute_cylinder_residual_stats(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
) -> Tuple[float, float]:
    pts = np.asarray(points, dtype=float)
    diff = pts - np.asarray(axis_point, dtype=float).reshape(-1)
    axis_dir = np.asarray(axis_dir, dtype=float).reshape(-1)
    axis_dir = axis_dir / np.maximum(np.linalg.norm(axis_dir), 1e-8)
    t = diff @ axis_dir
    radial_vec = diff - np.outer(t, axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    residuals = np.abs(radial_dist - float(radius))
    if residuals.size == 0:
        return 0.0, 0.0
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return median, mad


def compute_plane_residual_stats(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[float, float]:
    """Compute plane residual median/MAD."""
    pts = np.asarray(points, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float).reshape(-1)
    plane_normal = np.asarray(plane_normal, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3 or plane_point.size != 3 or plane_normal.size != 3:
        return 0.0, 0.0
    norm = np.linalg.norm(plane_normal)
    if not np.isfinite(norm) or norm < 1e-12:
        return 0.0, 0.0
    plane_normal = plane_normal / norm
    residuals = np.abs((pts - plane_point) @ plane_normal)
    if residuals.size == 0:
        return 0.0, 0.0
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return median, mad


def auto_select_primitive(
    plane_result: Optional[SeedExpandPlaneResult],
    cylinder_result: Optional[CylinderProbeResult],
    *,
    eps: float = 1e-6,
    plane_threshold: float = 1.0,
    cylinder_threshold: float = 1.0,
) -> AutoSelectResult:
    """Pick plane or cylinder based on scores."""
    plane_score = -1.0
    cylinder_score = -1.0
    reason = ""

    if plane_result is not None and plane_result.success and plane_result.expanded_inlier_count > 0:
        norm = max(float(plane_threshold), eps)
        plane_score = plane_result.expanded_inlier_count / max(float(plane_result.residual_median) / norm, eps)

    if cylinder_result is not None and cylinder_result.success and cylinder_result.inlier_count > 0:
        norm = max(float(cylinder_threshold), eps)
        cylinder_score = cylinder_result.inlier_count / max(float(cylinder_result.residual_median) / norm, eps)

    if plane_score < 0 and cylinder_score < 0:
        return AutoSelectResult(
            chosen="none",
            plane_score=plane_score,
            cylinder_score=cylinder_score,
            reason="both_failed",
        )

    if cylinder_score > plane_score:
        reason = "cylinder_score_higher"
        chosen = "cylinder"
    else:
        reason = "plane_score_higher_or_equal"
        chosen = "plane"

    return AutoSelectResult(
        chosen=chosen,
        plane_score=float(plane_score),
        cylinder_score=float(cylinder_score),
        reason=reason,
    )


def finalize_cylinder_from_proxy(
    points: np.ndarray,
    seed_center: np.ndarray,
    seed_indices: np.ndarray,
    proxy: CylinderParam,
    *,
    surface_threshold: float,
    cap_margin: float,
    grow_radius: float,
    max_expand_radius: float,
    max_expanded_points: int,
    max_frontier: int,
    max_steps: int,
    refine_iters: int = 2,
    circle_ransac_iters: int = 200,
    circle_inlier_threshold: float = 0.01,
    allow_length_growth: bool = False,
) -> CylinderProbeResult:
    """Use a proxy cylinder to extract points and refit a final cylinder."""
    if proxy is None:
        return CylinderProbeResult(
            proxy=None,
            final=None,
            inlier_indices=None,
            inlier_count=0,
            residual_median=0.0,
            residual_mad=0.0,
            candidate_count=0,
            stopped_early=False,
            stop_reason="no_proxy",
            steps=0,
            max_frontier_size=0,
            success=False,
            message="Proxy cylinder is missing",
        )

    axis_point = np.asarray(proxy.axis_point, dtype=float)
    axis_dir = np.asarray(proxy.axis_direction, dtype=float)
    radius = float(proxy.radius)
    length = float(proxy.length)

    length_for_extract = length
    if allow_length_growth and max_expand_radius > 0:
        length_for_extract = max(length_for_extract, float(max_expand_radius) * 2.0)

    extract = extract_cylinder_surface_component(
        points,
        seed_center,
        seed_indices,
        axis_point,
        axis_dir,
        radius,
        length_for_extract,
        surface_threshold=surface_threshold,
        cap_margin=cap_margin,
        grow_radius=grow_radius,
        max_expand_radius=max_expand_radius,
        max_expanded_points=max_expanded_points,
        max_frontier=max_frontier,
        max_steps=max_steps,
    )
    if not extract.success or extract.inlier_indices.size < 6:
        return CylinderProbeResult(
            proxy=proxy,
            final=None,
            inlier_indices=extract.inlier_indices if extract.success else None,
            inlier_count=int(extract.inlier_indices.size) if extract.success else 0,
            residual_median=0.0,
            residual_mad=0.0,
            candidate_count=int(extract.candidate_count),
            stopped_early=bool(extract.stopped_early),
            stop_reason=str(extract.stop_reason),
            steps=int(extract.steps),
            max_frontier_size=int(extract.max_frontier_size),
            success=False,
            message="Too few inliers for final cylinder",
        )

    inlier_indices = extract.inlier_indices
    last_extract = extract
    axis_dir_ref = axis_dir / np.maximum(np.linalg.norm(axis_dir), 1e-8)

    refine_iters = max(1, min(int(refine_iters), 3))
    for _ in range(refine_iters):
        inlier_points = np.asarray(points, dtype=float)[inlier_indices]
        cyl = _estimate_cylinder_from_points(
            inlier_points,
            axis_dir=axis_dir_ref,
            circle_ransac_iters=circle_ransac_iters,
            circle_inlier_threshold=circle_inlier_threshold,
        )
        if cyl is None:
            break
        if np.dot(cyl.axis_direction, axis_dir_ref) < 0:
            cyl.axis_direction = -cyl.axis_direction
        axis_dir_ref = cyl.axis_direction
        axis_point = cyl.axis_point
        radius = float(cyl.radius)
        length = float(cyl.length)

        length_for_extract = length
        if allow_length_growth and max_expand_radius > 0:
            length_for_extract = max(length_for_extract, float(max_expand_radius) * 2.0)

        last_extract = extract_cylinder_surface_component(
            points,
            seed_center,
            seed_indices,
            axis_point,
            axis_dir_ref,
            radius,
            length_for_extract,
            surface_threshold=surface_threshold,
            cap_margin=cap_margin,
            grow_radius=grow_radius,
            max_expand_radius=max_expand_radius,
            max_expanded_points=max_expanded_points,
            max_frontier=max_frontier,
            max_steps=max_steps,
        )
        if not last_extract.success or last_extract.inlier_indices.size < 6:
            break
        inlier_indices = last_extract.inlier_indices

    final_cylinder = CylinderParam(
        axis_point=axis_point,
        axis_direction=axis_dir_ref,
        radius=radius,
        length=length,
        inlier_count=int(inlier_indices.size),
        inlier_indices=inlier_indices,
    )
    residual_median, residual_mad = _compute_cylinder_residual_stats(
        np.asarray(points, dtype=float)[inlier_indices],
        axis_point,
        axis_dir_ref,
        radius,
    )

    return CylinderProbeResult(
        proxy=proxy,
        final=final_cylinder,
        inlier_indices=inlier_indices,
        inlier_count=int(inlier_indices.size),
        residual_median=float(residual_median),
        residual_mad=float(residual_mad),
        candidate_count=int(last_extract.candidate_count),
        stopped_early=bool(last_extract.stopped_early),
        stop_reason=str(last_extract.stop_reason),
        steps=int(last_extract.steps),
        max_frontier_size=int(last_extract.max_frontier_size),
        success=True,
        message="Cylinder probe fit successful",
    )


def probe_cylinder_from_seed(
    points: np.ndarray,
    seed_center: np.ndarray,
    *,
    seed_radius_start: float = 0.05,
    seed_radius_max: float = 0.5,
    seed_radius_step: float = 0.05,
    min_seed_points: int = 80,
    circle_ransac_iters: int = 200,
    circle_inlier_threshold: float = 0.01,
    length_margin: float = 0.05,
    surface_threshold: float = 0.02,
    cap_margin: float = 0.05,
    grow_radius: float = 0.15,
    max_expand_radius: float = 5.0,
    max_expanded_points: int = 200_000,
    max_frontier: int = 200_000,
    max_steps: int = 1_000_000,
    refine_iters: int = 2,
) -> CylinderProbeResult:
    """Full probe pipeline from seed click to final cylinder fit."""
    proxy_init = compute_cylinder_proxy_from_seed(
        points,
        seed_center,
        seed_radius_start=seed_radius_start,
        seed_radius_max=seed_radius_max,
        seed_radius_step=seed_radius_step,
        min_seed_points=min_seed_points,
        circle_ransac_iters=circle_ransac_iters,
        circle_inlier_threshold=circle_inlier_threshold,
        length_margin=length_margin,
    )
    if not proxy_init.success or proxy_init.cylinder is None:
        return CylinderProbeResult(
            proxy=None,
            final=None,
            inlier_indices=None,
            inlier_count=0,
            residual_median=0.0,
            residual_mad=0.0,
            candidate_count=0,
            stopped_early=False,
            stop_reason="proxy_init_failed",
            steps=0,
            max_frontier_size=0,
            success=False,
            message=proxy_init.message,
            seed_radius=float(proxy_init.seed_radius),
            seed_point_count=int(proxy_init.seed_point_count),
        )

    result = finalize_cylinder_from_proxy(
        points,
        seed_center,
        proxy_init.seed_indices,
        proxy_init.cylinder,
        surface_threshold=surface_threshold,
        cap_margin=cap_margin,
        grow_radius=grow_radius,
        max_expand_radius=max_expand_radius,
        max_expanded_points=max_expanded_points,
        max_frontier=max_frontier,
        max_steps=max_steps,
        refine_iters=refine_iters,
        circle_ransac_iters=circle_ransac_iters,
        circle_inlier_threshold=circle_inlier_threshold,
        allow_length_growth=True,
    )
    result.seed_radius = float(proxy_init.seed_radius)
    result.seed_point_count = int(proxy_init.seed_point_count)
    return result


def extract_stair_planes(
    points: np.ndarray,
    max_planes: int = 20,
    min_inliers: int = 50,
    distance_threshold: float = 0.02,
    max_tilt_deg: float = 15.0,
    height_eps: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    horizontal_only: bool = True,
    merge_by_height: bool = True,
    verbose: bool = True,
) -> List[PlaneParam]:
    """
    Extract stair planes (horizontal surfaces) from point cloud.

    This is a convenience function that combines:
    1. Iterative multi-plane RANSAC
    2. Horizontal plane filtering (optional)
    3. Height-based plane merging (optional)

    Args:
        points: (N, 3) array of 3D points
        max_planes: Maximum number of planes to extract
        min_inliers: Minimum inliers required to accept a plane
        distance_threshold: Maximum distance from plane for inliers
        max_tilt_deg: Maximum tilt angle for horizontal filter
        height_eps: Height tolerance for merging planes
        ransac_n: Number of points for RANSAC sampling
        num_iterations: RANSAC iterations per plane
        horizontal_only: If True, filter to keep only horizontal planes
        merge_by_height: If True, merge planes at similar heights

    Returns:
        List of PlaneParam for detected stair planes, sorted by height
    """
    if verbose:
        print("\n=== Extracting Stair Planes ===")
        print(
            f"Parameters: max_planes={max_planes}, min_inliers={min_inliers}, "
            f"threshold={distance_threshold}m"
        )

    # Step 1: Extract all planes
    result = extract_multi_planes(
        points,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        min_inliers=min_inliers,
        max_planes=max_planes,
        verbose=verbose,
    )

    planes = result.planes
    if verbose:
        print(f"\nExtracted {len(planes)} planes total")

    # Step 2: Filter horizontal planes
    if horizontal_only:
        if verbose:
            print(f"\nFiltering horizontal planes (max_tilt={max_tilt_deg}°)...")
        planes = filter_horizontal_planes(planes, max_tilt_deg=max_tilt_deg, verbose=verbose)
        if verbose:
            print(f"After filter: {len(planes)} horizontal planes")

    # Step 3: Merge by height
    if merge_by_height and len(planes) > 1:
        if verbose:
            print(f"\nMerging planes by height (eps={height_eps}m)...")
        planes = merge_planes_by_height(
            planes,
            points,
            height_eps=height_eps,
            distance_threshold=distance_threshold,
            verbose=verbose,
        )
        if verbose:
            print(f"After merge: {len(planes)} planes")

    # Sort by height
    planes = sorted(
        planes,
        key=lambda p: (p.height if p.height is not None else float(np.asarray(p.point, dtype=float)[2])),
    )

    if verbose:
        print(f"\n=== Final Result: {len(planes)} stair planes ===")
        for i, p in enumerate(planes):
            height = float(p.height) if p.height is not None else float(np.asarray(p.point, dtype=float)[2])
            print(
                f"  [{i}] height={height:.3f}m, inliers={p.inlier_count}, "
                f"normal=[{p.normal[0]:.3f}, {p.normal[1]:.3f}, {p.normal[2]:.3f}]"
            )

    return planes
