"""Tests for cylinder probe core pipeline."""

import numpy as np

from primitives import probe_cylinder_from_seed


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
    """Generate a dense set of points on a cylinder surface."""
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


def test_probe_cylinder_recovers_radius_length():
    rng = np.random.default_rng(123)

    radius = 0.18
    length = 2.4
    points = _make_cylinder_grid_points(
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        radius=radius,
        length=length,
        n_theta=72,
        n_t=80,
        noise_std=0.002,
        rng=rng,
    )

    seed_center = np.array([radius, 0.0, 0.0])

    result = probe_cylinder_from_seed(
        points,
        seed_center,
        seed_radius_start=0.05,
        seed_radius_max=0.4,
        seed_radius_step=0.05,
        min_seed_points=80,
        surface_threshold=0.03,
        cap_margin=0.08,
        grow_radius=0.2,
        max_expand_radius=3.0,
        refine_iters=2,
    )

    assert result.success
    assert result.final is not None
    assert abs(result.final.radius - radius) < 0.04
    assert abs(result.final.length - length) < 0.4
