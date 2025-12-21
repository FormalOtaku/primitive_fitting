"""Tests for session auto selection."""

import numpy as np

from primitives import (
    expand_plane_from_seed,
    probe_cylinder_from_seed,
    auto_select_primitive,
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
    x = rng.uniform(x_range[0], x_range[1], size=n)
    y = rng.uniform(y_range[0], y_range[1], size=n)
    z_vals = np.full(n, z, dtype=float) + rng.normal(scale=noise_std, size=n)
    return np.column_stack([x, y, z_vals])


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


def test_auto_selects_plane():
    rng = np.random.default_rng(222)
    plane = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
        n=1500,
        noise_std=0.002,
        rng=rng,
    )

    seed_center = np.array([0.0, 0.0, 0.0])
    plane_result = expand_plane_from_seed(
        plane,
        seed_center,
        seed_radius=0.5,
        max_expand_radius=5.0,
        grow_radius=0.2,
        distance_threshold=0.02,
        expand_method="component",
        verbose=False,
    )
    cylinder_result = probe_cylinder_from_seed(
        plane,
        seed_center,
        seed_radius_start=0.05,
        seed_radius_max=0.4,
        seed_radius_step=0.05,
        min_seed_points=80,
        surface_threshold=0.02,
        cap_margin=0.05,
        grow_radius=0.15,
        max_expand_radius=2.0,
        refine_iters=2,
    )
    choice = auto_select_primitive(
        plane_result,
        cylinder_result,
        plane_threshold=0.02,
        cylinder_threshold=0.03,
    )
    assert choice.chosen == "plane"


def test_auto_selects_cylinder():
    rng = np.random.default_rng(223)
    cylinder = _make_cylinder_grid_points(
        axis_point=np.array([0.0, 0.0, 0.0]),
        axis_dir=np.array([0.0, 0.0, 1.0]),
        radius=0.12,
        length=2.0,
        n_theta=60,
        n_t=60,
        noise_std=0.002,
        rng=rng,
    )

    seed_center = np.array([0.12, 0.0, 0.0])
    plane_result = expand_plane_from_seed(
        cylinder,
        seed_center,
        seed_radius=0.3,
        max_expand_radius=4.0,
        grow_radius=0.2,
        distance_threshold=0.02,
        expand_method="component",
        verbose=False,
    )
    cylinder_result = probe_cylinder_from_seed(
        cylinder,
        seed_center,
        seed_radius_start=0.05,
        seed_radius_max=0.5,
        seed_radius_step=0.05,
        min_seed_points=80,
        surface_threshold=0.03,
        cap_margin=0.08,
        grow_radius=0.2,
        max_expand_radius=3.0,
        refine_iters=2,
    )
    choice = auto_select_primitive(
        plane_result,
        cylinder_result,
        plane_threshold=0.02,
        cylinder_threshold=0.03,
    )
    assert choice.chosen == "cylinder"
