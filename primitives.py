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
