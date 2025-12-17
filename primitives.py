"""
primitives.py - Primitive fitting functions for plane and cylinder
"""

from dataclasses import dataclass
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
        ransac_n: RANSAC sample size for initial plane fit
        num_iterations: RANSAC iterations for initial plane fit
        verbose: print progress logs

    Returns:
        SeedExpandPlaneResult with expanded plane and metadata
    """
    points = np.asarray(points, dtype=float)
    seed_center = np.asarray(seed_center, dtype=float).flatten()

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

    # Step 1: Extract seed points
    distances_to_seed = np.linalg.norm(points - seed_center, axis=1)
    seed_mask = distances_to_seed <= seed_radius
    seed_indices = np.where(seed_mask)[0]

    if len(seed_indices) < 3:
        return SeedExpandPlaneResult(
            plane=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            area=0.0, extent_u=0.0, extent_v=0.0, seed_inlier_indices=None,
            success=False, message=f"Too few seed points: {len(seed_indices)}"
        )

    seed_points = points[seed_indices]
    if verbose:
        print(f"  Seed region: {len(seed_points)} points within {seed_radius}m")

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
            success=False, message="Failed to fit initial plane on seed points"
        )

    plane_normal = initial_plane.normal
    plane_point = initial_plane.point

    if verbose:
        print(f"  Initial plane: normal={np.round(plane_normal, 3).tolist()}, "
              f"inliers={initial_plane.inlier_count}/{len(seed_points)}")

    # Step 3: Expand from seed
    # Limit expansion to max_expand_radius
    expand_mask = distances_to_seed <= max_expand_radius
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

    if verbose:
        print(f"  Plane candidates (distance < {distance_threshold}m): {len(candidate_local_indices)}")

    if len(candidate_local_indices) == 0:
        # Return seed plane only
        return SeedExpandPlaneResult(
            plane=initial_plane,
            expanded_inlier_indices=seed_indices[initial_plane.inlier_indices] if initial_plane.inlier_indices is not None else seed_indices,
            expanded_inlier_count=initial_plane.inlier_count,
            area=0.0, extent_u=0.0, extent_v=0.0,
            seed_inlier_indices=seed_indices,
            success=True, message="No expansion candidates found, returning seed plane"
        )

    # Map seed inliers to global indices
    if initial_plane.inlier_indices is not None:
        seed_inlier_global = seed_indices[initial_plane.inlier_indices]
    else:
        seed_inlier_global = seed_indices

    # Expand using selected method
    if expand_method == "bfs":
        expanded_indices = _expand_bfs(
            points, candidate_global_indices, seed_inlier_global, grow_radius
        )
    else:  # component
        expanded_indices = _expand_component(
            points, candidate_global_indices, seed_inlier_global, grow_radius
        )

    if verbose:
        print(f"  Expanded region ({expand_method}): {len(expanded_indices)} points")

    if len(expanded_indices) < 3:
        return SeedExpandPlaneResult(
            plane=initial_plane,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            area=0.0, extent_u=0.0, extent_v=0.0,
            seed_inlier_indices=seed_indices,
            success=True, message="Expansion resulted in too few points"
        )

    # Step 4: Iterative refinement
    current_indices = expanded_indices
    final_plane = initial_plane

    for refine_iter in range(max_refine_iters):
        # Refit plane using expanded inliers
        expanded_points = points[current_indices]
        centroid = expanded_points.mean(axis=0)
        centered = expanded_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        new_normal = vh[-1]
        new_normal = new_normal / np.linalg.norm(new_normal)

        # Orient normal consistently
        if np.dot(new_normal, plane_normal) < 0:
            new_normal = -new_normal

        # Re-expand with new plane
        plane_distances = np.abs(np.dot(expand_points - centroid, new_normal))
        candidate_mask = plane_distances < distance_threshold
        if has_normals:
            expand_normals = normals[expand_indices]
            normal_alignment = np.abs(np.dot(expand_normals, new_normal))
            candidate_mask &= normal_alignment > normal_cos_threshold

        candidate_local_indices = np.where(candidate_mask)[0]
        candidate_global_indices = expand_indices[candidate_local_indices]

        if expand_method == "bfs":
            new_indices = _expand_bfs(
                points, candidate_global_indices, seed_inlier_global, grow_radius
            )
        else:
            new_indices = _expand_component(
                points, candidate_global_indices, seed_inlier_global, grow_radius
            )

        if verbose:
            print(f"    Refine iter {refine_iter + 1}: {len(new_indices)} points")

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

    # Compute area and extent
    area, extent_u, extent_v = _compute_plane_metrics(points[current_indices], final_plane.normal)

    if verbose:
        print(f"  Final: {len(current_indices)} inliers, area={area:.3f}m², "
              f"extent=({extent_u:.2f} x {extent_v:.2f})m")

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
        message="Plane expansion successful"
    )


def _expand_bfs(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    seed_indices: np.ndarray,
    grow_radius: float
) -> np.ndarray:
    """Expand using BFS from seed inliers."""
    if len(candidate_indices) == 0:
        return seed_indices.copy()

    # Build set for fast lookup
    candidate_set = set(candidate_indices.tolist())

    # Initialize with seed inliers that are also candidates
    frontier = [i for i in seed_indices if i in candidate_set]
    if len(frontier) == 0:
        # Fall back to closest candidate to any seed point
        seed_points = points[seed_indices]
        candidate_points = points[candidate_indices]
        min_dist = float('inf')
        closest = candidate_indices[0] if len(candidate_indices) > 0 else None
        for cp_idx, cp in zip(candidate_indices, candidate_points):
            d = np.min(np.linalg.norm(seed_points - cp, axis=1))
            if d < min_dist:
                min_dist = d
                closest = cp_idx
        if closest is not None and min_dist <= grow_radius:
            frontier = [closest]
        else:
            return seed_indices.copy()

    visited = set(frontier)
    result = list(frontier)

    # BFS
    while frontier:
        current = frontier.pop(0)
        current_point = points[current]

        # Find neighbors
        for idx in candidate_indices:
            if idx in visited:
                continue
            if np.linalg.norm(points[idx] - current_point) <= grow_radius:
                visited.add(idx)
                result.append(idx)
                frontier.append(idx)

    return np.array(result, dtype=int)


def _expand_component(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    seed_indices: np.ndarray,
    grow_radius: float
) -> np.ndarray:
    """Expand using connected component from seed."""
    if len(candidate_indices) == 0:
        return seed_indices.copy()

    candidate_list = list(candidate_indices)
    n_candidates = len(candidate_list)
    idx_to_local = {idx: i for i, idx in enumerate(candidate_list)}

    # Build adjacency list using radius search
    adjacency: List[List[int]] = [[] for _ in range(n_candidates)]
    candidate_points = points[candidate_list]

    # Simple O(n²) neighbor search for now (could use KD-tree for large datasets)
    for i in range(n_candidates):
        for j in range(i + 1, n_candidates):
            if np.linalg.norm(candidate_points[i] - candidate_points[j]) <= grow_radius:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Find component containing seed
    seed_local = [idx_to_local[idx] for idx in seed_indices if idx in idx_to_local]
    if len(seed_local) == 0:
        # Find closest candidate to seed
        seed_points = points[seed_indices]
        min_dist = float('inf')
        closest_local = None
        for i, cp in enumerate(candidate_points):
            d = np.min(np.linalg.norm(seed_points - cp, axis=1))
            if d < min_dist:
                min_dist = d
                closest_local = i
        if closest_local is not None and min_dist <= grow_radius:
            seed_local = [closest_local]
        else:
            return seed_indices.copy()

    # BFS from seed to find connected component
    visited = set(seed_local)
    frontier = list(seed_local)

    while frontier:
        current = frontier.pop(0)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)

    return np.array([candidate_list[i] for i in visited], dtype=int)


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

    # Step 1: Extract seed points
    distances_to_seed = np.linalg.norm(points - seed_center, axis=1)
    seed_mask = distances_to_seed <= seed_radius
    seed_indices = np.where(seed_mask)[0]

    if len(seed_indices) < 6:
        return SeedExpandCylinderResult(
            cylinder=None, expanded_inlier_indices=None, expanded_inlier_count=0,
            seed_inlier_indices=None, success=False,
            message=f"Too few seed points: {len(seed_indices)}"
        )

    seed_points = points[seed_indices]
    seed_normals = normals[seed_indices] if has_normals else None

    if verbose:
        print(f"  Seed region: {len(seed_points)} points within {seed_radius}m")

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
            message="Failed to fit initial cylinder on seed points"
        )

    axis_point = initial_cylinder.axis_point
    axis_dir = initial_cylinder.axis_direction
    cyl_radius = initial_cylinder.radius

    if verbose:
        print(f"  Initial cylinder: radius={cyl_radius:.4f}m, "
              f"axis_dir={np.round(axis_dir, 3).tolist()}, "
              f"inliers={initial_cylinder.inlier_count}/{len(seed_points)}")

    # Step 3: Expand from seed
    expand_mask = distances_to_seed <= max_expand_radius
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

    if verbose:
        print(f"  Cylinder candidates (surface dist < {distance_threshold}m): {len(candidate_local_indices)}")

    if len(candidate_local_indices) == 0:
        seed_inlier_global = seed_indices[initial_cylinder.inlier_indices] if initial_cylinder.inlier_indices is not None else seed_indices
        return SeedExpandCylinderResult(
            cylinder=initial_cylinder,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            seed_inlier_indices=seed_indices,
            success=True, message="No expansion candidates found, returning seed cylinder"
        )

    # Map seed inliers to global indices
    if initial_cylinder.inlier_indices is not None:
        seed_inlier_global = seed_indices[initial_cylinder.inlier_indices]
    else:
        seed_inlier_global = seed_indices

    # Expand using selected method
    if expand_method == "bfs":
        expanded_indices = _expand_bfs(
            points, candidate_global_indices, seed_inlier_global, grow_radius
        )
    else:
        expanded_indices = _expand_component(
            points, candidate_global_indices, seed_inlier_global, grow_radius
        )

    if verbose:
        print(f"  Expanded region ({expand_method}): {len(expanded_indices)} points")

    if len(expanded_indices) < 6:
        return SeedExpandCylinderResult(
            cylinder=initial_cylinder,
            expanded_inlier_indices=seed_inlier_global,
            expanded_inlier_count=len(seed_inlier_global),
            seed_inlier_indices=seed_indices,
            success=True, message="Expansion resulted in too few points"
        )

    # Recompute cylinder parameters from expanded inliers
    expanded_points = points[expanded_indices]
    centroid = expanded_points.mean(axis=0)
    centered = expanded_points - centroid
    _, _, vh = np.linalg.svd(centered)
    new_axis_dir = vh[0]
    new_axis_dir = new_axis_dir / np.linalg.norm(new_axis_dir)

    # Orient consistently
    if np.dot(new_axis_dir, axis_dir) < 0:
        new_axis_dir = -new_axis_dir

    # Recompute radius and length
    diff = expanded_points - centroid
    projections = diff @ new_axis_dir
    radial_vec = diff - np.outer(projections, new_axis_dir)
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    new_radius = float(np.median(radial_dist))
    new_length = float(np.max(projections) - np.min(projections))

    if verbose:
        print(f"  Final: {len(expanded_indices)} inliers, radius={new_radius:.4f}m, length={new_length:.3f}m")

    final_cylinder = CylinderParam(
        axis_point=centroid,
        axis_direction=new_axis_dir,
        radius=new_radius,
        length=new_length,
        inlier_count=len(expanded_indices),
        inlier_indices=expanded_indices
    )

    return SeedExpandCylinderResult(
        cylinder=final_cylinder,
        expanded_inlier_indices=expanded_indices,
        expanded_inlier_count=len(expanded_indices),
        seed_inlier_indices=seed_indices,
        success=True,
        message="Cylinder expansion successful"
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
