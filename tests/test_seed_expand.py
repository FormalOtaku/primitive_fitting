"""Tests for seed-expand plane and cylinder extraction."""

import numpy as np

from primitives import (
    expand_plane_from_seed,
    expand_cylinder_from_seed,
)


def _make_horizontal_plane_points(
    *,
    z: float,
    x_range: tuple,
    y_range: tuple,
    n: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate points on a horizontal plane."""
    x = rng.uniform(x_range[0], x_range[1], size=n)
    y = rng.uniform(y_range[0], y_range[1], size=n)
    z_vals = np.full(n, z, dtype=float) + rng.normal(scale=noise_std, size=n)
    return np.column_stack([x, y, z_vals])


def _make_cylinder_points(
    *,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    length: float,
    n: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate points on a cylinder surface."""
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # Create orthonormal basis
    if abs(axis_dir[2]) < 0.9:
        u = np.cross(axis_dir, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(axis_dir, np.array([1.0, 0.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(axis_dir, u)

    # Generate random angles and positions along axis
    theta = rng.uniform(0, 2 * np.pi, size=n)
    t = rng.uniform(-length / 2, length / 2, size=n)

    # Generate points on cylinder surface with noise
    r = radius + rng.normal(scale=noise_std, size=n)
    points = (
        axis_point
        + np.outer(t, axis_dir)
        + np.outer(r * np.cos(theta), u)
        + np.outer(r * np.sin(theta), v)
    )
    return points


def _make_cylinder_grid_points(
    *,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    length: float,
    n_theta: int,
    n_t: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a dense, well-connected set of points on a cylinder surface."""
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    if abs(axis_dir[2]) < 0.9:
        u = np.cross(axis_dir, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(axis_dir, np.array([1.0, 0.0, 0.0]))
    u = u / np.linalg.norm(u)
    v = np.cross(axis_dir, u)

    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    t = np.linspace(-length / 2, length / 2, n_t)
    theta_grid, t_grid = np.meshgrid(theta, t)
    theta_flat = theta_grid.ravel()
    t_flat = t_grid.ravel()
    r = radius + rng.normal(scale=noise_std, size=len(theta_flat))

    points = (
        axis_point
        + np.outer(t_flat, axis_dir)
        + np.outer(r * np.cos(theta_flat), u)
        + np.outer(r * np.sin(theta_flat), v)
    )
    return points


def test_expand_plane_from_seed_connected_region():
    """Test that plane expansion only extracts connected region from seed."""
    rng = np.random.default_rng(42)

    # Create two separate planes at the same height but disconnected
    # Plane A: near the origin (where seed will be)
    plane_a = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        n=500,
        noise_std=0.002,
        rng=rng,
    )

    # Plane B: far away, same height (should NOT be included)
    plane_b = _make_horizontal_plane_points(
        z=0.0,
        x_range=(5.0, 7.0),
        y_range=(5.0, 7.0),
        n=500,
        noise_std=0.002,
        rng=rng,
    )

    # Combine points
    points = np.vstack([plane_a, plane_b])

    # Seed center in the middle of plane A
    seed_center = np.array([0.0, 0.0, 0.0])

    # Run seed-expand with component method
    result = expand_plane_from_seed(
        points,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=10.0,  # Large enough to include both planes
        grow_radius=0.2,
        distance_threshold=0.02,
        expand_method="component",
        max_refine_iters=3,
        verbose=False,
    )

    assert result.success, f"Expansion failed: {result.message}"
    assert result.plane is not None

    # Should only get points from plane A (first 500 points)
    # Not plane B (indices 500-999)
    expanded_indices = set(result.expanded_inlier_indices.tolist())

    # Most of the expanded points should be from plane A
    plane_a_indices = set(range(len(plane_a)))
    plane_b_indices = set(range(len(plane_a), len(points)))

    overlap_a = len(expanded_indices & plane_a_indices)
    overlap_b = len(expanded_indices & plane_b_indices)

    # Almost all expanded points should be from plane A
    assert overlap_a > 400, f"Expected most points from plane A, got {overlap_a}"
    # Very few or no points from plane B
    assert overlap_b < 10, f"Expected few points from plane B, got {overlap_b}"


def test_expand_plane_from_seed_bfs_method():
    """Test BFS expansion method."""
    rng = np.random.default_rng(43)

    # Create a single plane with higher point density
    plane = _make_horizontal_plane_points(
        z=1.0,
        x_range=(-1.5, 1.5),  # Smaller area for better connectivity
        y_range=(-1.5, 1.5),
        n=1000,
        noise_std=0.003,
        rng=rng,
    )

    seed_center = np.array([0.0, 0.0, 1.0])

    result = expand_plane_from_seed(
        plane,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=5.0,
        grow_radius=0.25,  # Larger grow radius for better connectivity
        distance_threshold=0.02,
        expand_method="bfs",
        verbose=False,
    )

    assert result.success
    assert result.plane is not None
    # Should capture a significant portion of the plane
    assert result.expanded_inlier_count > 500


def test_expand_plane_normal_validation():
    """Test that normal threshold filters out points with wrong normals."""
    rng = np.random.default_rng(44)

    # Create a horizontal plane
    plane = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        n=500,
        noise_std=0.001,
        rng=rng,
    )

    # Create normals - all pointing up (+Z)
    normals = np.zeros_like(plane)
    normals[:, 2] = 1.0

    seed_center = np.array([0.0, 0.0, 0.0])

    result = expand_plane_from_seed(
        plane,
        seed_center,
        normals=normals,
        seed_radius=0.3,
        max_expand_radius=5.0,
        grow_radius=0.15,
        distance_threshold=0.02,
        normal_threshold_deg=30.0,
        expand_method="component",
        verbose=False,
    )

    assert result.success
    assert result.plane is not None
    # Normal should be close to +Z or -Z
    assert abs(result.plane.normal[2]) > 0.95


def test_expand_cylinder_from_seed_connected_region():
    """Test that cylinder expansion only extracts connected cylinder."""
    rng = np.random.default_rng(45)

    # Create two separate cylinders with higher point density
    # Cylinder A: vertical cylinder at origin (where seed will be)
    cylinder_a = _make_cylinder_points(
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        radius=0.1,
        length=1.5,  # Shorter for better density
        n=800,  # More points
        noise_std=0.002,
        rng=rng,
    )

    # Cylinder B: same radius but far away (should NOT be included)
    cylinder_b = _make_cylinder_points(
        axis_point=np.array([5.0, 5.0, 0.0]),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        radius=0.1,
        length=1.5,
        n=500,
        noise_std=0.002,
        rng=rng,
    )

    points = np.vstack([cylinder_a, cylinder_b])

    # Seed at center of cylinder A
    seed_center = np.array([0.1, 0.0, 0.0])

    result = expand_cylinder_from_seed(
        points,
        seed_center,
        seed_radius=0.3,
        max_expand_radius=10.0,
        grow_radius=0.2,  # Larger grow radius for better connectivity
        distance_threshold=0.03,  # Slightly larger threshold
        expand_method="component",
        verbose=False,
    )

    assert result.success, f"Expansion failed: {result.message}"
    assert result.cylinder is not None

    # Check that mostly cylinder A points are included
    expanded_indices = set(result.expanded_inlier_indices.tolist())
    cylinder_a_indices = set(range(len(cylinder_a)))
    cylinder_b_indices = set(range(len(cylinder_a), len(points)))

    overlap_a = len(expanded_indices & cylinder_a_indices)
    overlap_b = len(expanded_indices & cylinder_b_indices)

    # Should capture a significant portion of cylinder A
    assert overlap_a > 200, f"Expected most points from cylinder A, got {overlap_a}"
    assert overlap_b < 10, f"Expected few points from cylinder B, got {overlap_b}"

    # Check cylinder parameters are reasonable
    assert 0.07 < result.cylinder.radius < 0.15, f"Radius {result.cylinder.radius} not close to 0.1"


def test_expand_cylinder_bfs_method():
    """Test BFS expansion method for cylinder."""
    rng = np.random.default_rng(46)

    cylinder = _make_cylinder_points(
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_dir=np.array([1.0, 0.0, 0.0]),  # Horizontal cylinder
        radius=0.15,
        length=3.0,
        n=1000,
        noise_std=0.003,
        rng=rng,
    )

    seed_center = np.array([0.15, 0.0, 0.0])

    result = expand_cylinder_from_seed(
        cylinder,
        seed_center,
        seed_radius=0.4,
        max_expand_radius=5.0,
        grow_radius=0.15,
        distance_threshold=0.03,
        expand_method="bfs",
        verbose=False,
    )

    assert result.success
    assert result.cylinder is not None
    assert result.expanded_inlier_count > 700


def test_expand_plane_too_few_seed_points():
    """Test handling of too few seed points."""
    rng = np.random.default_rng(47)

    # Very sparse points
    points = rng.uniform(-10, 10, size=(5, 3))
    seed_center = np.array([0.0, 0.0, 0.0])

    result = expand_plane_from_seed(
        points,
        seed_center,
        seed_radius=0.1,  # Very small - will have few points
        verbose=False,
    )

    # Should fail gracefully
    assert not result.success or result.expanded_inlier_count < 3


def test_expand_cylinder_too_few_seed_points():
    """Test handling of too few seed points for cylinder."""
    rng = np.random.default_rng(48)

    points = rng.uniform(-10, 10, size=(5, 3))
    seed_center = np.array([0.0, 0.0, 0.0])

    result = expand_cylinder_from_seed(
        points,
        seed_center,
        seed_radius=0.1,
        verbose=False,
    )

    assert not result.success or result.expanded_inlier_count < 6


def test_expand_plane_iterative_refinement():
    """Test that iterative refinement improves the result."""
    rng = np.random.default_rng(49)

    # Create a tilted plane with higher density
    n = 1000
    x = rng.uniform(-1.5, 1.5, n)
    y = rng.uniform(-1.5, 1.5, n)
    # z = 0.1*x + 0.05*y (slightly tilted plane)
    z = 0.1 * x + 0.05 * y + rng.normal(scale=0.002, size=n)
    points = np.column_stack([x, y, z])

    seed_center = np.array([0.0, 0.0, 0.0])

    # With refinement - use larger threshold for tilted plane
    result_refined = expand_plane_from_seed(
        points,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=5.0,
        grow_radius=0.25,  # Larger for better connectivity
        distance_threshold=0.05,  # Larger threshold for tilted plane
        max_refine_iters=5,  # More iterations
        verbose=False,
    )

    assert result_refined.success
    # Should capture a significant portion after refinement
    assert result_refined.expanded_inlier_count > 200


def test_expand_plane_area_computation():
    """Test that area and extent are computed correctly."""
    rng = np.random.default_rng(50)

    # Create a 2x3 meter horizontal plane
    plane = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-1.0, 1.0),  # 2m in x
        y_range=(-1.5, 1.5),  # 3m in y
        n=1000,
        noise_std=0.001,
        rng=rng,
    )

    seed_center = np.array([0.0, 0.0, 0.0])

    result = expand_plane_from_seed(
        plane,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=5.0,
        grow_radius=0.2,
        distance_threshold=0.02,
        verbose=False,
    )

    assert result.success
    # Area should be approximately 6 m² (2m x 3m)
    assert 4.0 < result.area < 8.0, f"Area {result.area} not in expected range"
    # Extents should be approximately 2m and 3m
    extents = sorted([result.extent_u, result.extent_v])
    assert 1.5 < extents[0] < 2.5, f"Smaller extent {extents[0]} not close to 2m"
    assert 2.5 < extents[1] < 3.5, f"Larger extent {extents[1]} not close to 3m"


def test_expand_plane_respects_max_expand_radius():
    """Points outside max_expand_radius must never be included, even if the plane continues."""
    rng = np.random.default_rng(51)

    # A long strip of planar points extending beyond max_expand_radius.
    x = np.linspace(0.0, 3.0, 61)
    y = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(x, y)
    zz = rng.normal(scale=0.001, size=xx.size)
    plane = np.column_stack([xx.ravel(), yy.ravel(), zz])

    seed_center = np.array([0.0, 0.0, 0.0])
    max_expand_radius = 1.0

    result = expand_plane_from_seed(
        plane,
        seed_center,
        seed_radius=0.3,
        max_expand_radius=max_expand_radius,
        grow_radius=0.15,
        distance_threshold=0.01,
        expand_method="component",
        verbose=False,
    )

    assert result.success
    assert result.expanded_inlier_indices is not None
    d = np.linalg.norm(plane[result.expanded_inlier_indices] - seed_center, axis=1)
    assert d.max() <= max_expand_radius + 1e-6


def test_expand_cylinder_nearby_cylinder_does_not_leak_when_disconnected():
    """Nearby cylinder points can be candidates but must not leak across a connectivity gap."""
    rng = np.random.default_rng(52)

    radius = 0.1
    length = 1.0
    axis_dir = np.array([0.0, 0.0, 1.0])

    cylinder_a = _make_cylinder_grid_points(
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_dir=axis_dir,
        radius=radius,
        length=length,
        n_theta=60,
        n_t=80,
        noise_std=0.0005,
        rng=rng,
    )
    # Slightly offset (gap ~2 cm): some points fall within cylinder_dist threshold, but gap > grow_radius.
    cylinder_b = _make_cylinder_grid_points(
        axis_point=np.array([0.22, 0.0, 0.0]),
        axis_dir=axis_dir,
        radius=radius,
        length=length,
        n_theta=60,
        n_t=80,
        noise_std=0.0005,
        rng=rng,
    )

    points = np.vstack([cylinder_a, cylinder_b])
    # Seed on the opposite side of cylinder A so the seed region does not include cylinder B.
    seed_center = np.array([-radius, 0.0, 0.0])

    result = expand_cylinder_from_seed(
        points,
        seed_center,
        seed_radius=0.2,
        max_expand_radius=2.0,
        grow_radius=0.015,  # smaller than the gap
        distance_threshold=0.03,  # allows some nearby-cylinder candidates
        expand_method="component",
        max_refine_iters=2,
        verbose=False,
    )

    assert result.success
    assert result.expanded_inlier_indices is not None
    expanded = set(result.expanded_inlier_indices.tolist())
    a_idx = set(range(len(cylinder_a)))
    b_idx = set(range(len(cylinder_a), len(points)))
    overlap_a = len(expanded & a_idx)
    overlap_b = len(expanded & b_idx)
    assert overlap_a > 1000
    assert overlap_b < 10


def test_expand_plane_stops_at_max_expanded_points():
    """max_expanded_points must cap the expansion and report the stop reason."""
    rng = np.random.default_rng(53)
    plane = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        n=6000,
        noise_std=0.001,
        rng=rng,
    )
    seed_center = np.array([0.0, 0.0, 0.0])

    result = expand_plane_from_seed(
        plane,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=5.0,
        grow_radius=0.2,
        distance_threshold=0.02,
        max_expanded_points=500,
        max_frontier=1_000_000,
        max_steps=1_000_000,
        verbose=False,
    )

    assert result.success
    assert result.expanded_inlier_count <= 500
    assert result.stopped_early
    assert result.stop_reason == "max_expanded_points"
