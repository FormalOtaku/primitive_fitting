"""
primitives.py - Primitive fitting functions for plane and cylinder
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, List
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
    pos_ok = (pos < cand_global.size) & (cand_global[pos] == seed_unique)
    seed_local = pos[pos_ok]

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
    residual_p90: float = 0.0
    residual_p95: float = 0.0


def expand_cylinder_from_seed(
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

    if len(seed_indices) < 6:
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

    # Step 2: Fit initial cylinder on seed points
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

    if len(candidate_local_indices) == 0:
        seed_inlier_global = seed_indices[initial_cylinder.inlier_indices] if initial_cylinder.inlier_indices is not None else seed_indices
        return SeedExpandCylinderResult(
            cylinder=initial_cylinder,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            seed_inlier_indices=seed_indices,
            success=True, message="No expansion candidates found, returning seed cylinder",
            seed_point_count=len(seed_indices),
            candidate_count=0,
        )

    # Map seed inliers to global indices
    if initial_cylinder.inlier_indices is not None:
        seed_inlier_global = seed_indices[initial_cylinder.inlier_indices]
    else:
        seed_inlier_global = seed_indices

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
    else:
        residual_median = residual_p90 = residual_p95 = 0.0
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
        residual_p90=float(residual_p90),
        residual_p95=float(residual_p95),
    )


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
