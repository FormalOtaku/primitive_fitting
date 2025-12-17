import numpy as np

from primitives import PlaneParam, extract_stair_planes, filter_horizontal_planes, merge_planes_by_height


def _make_horizontal_plane_points(
    *,
    z: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = rng.uniform(x_range[0], x_range[1], size=n)
    y = rng.uniform(y_range[0], y_range[1], size=n)
    z_vals = np.full(n, z, dtype=float) + rng.normal(scale=noise_std, size=n)
    return np.column_stack([x, y, z_vals])


def test_extract_stair_planes_synthetic_stairs():
    rng = np.random.default_rng(0)
    num_steps = 6
    step_height = 0.18
    tread_depth = 0.30

    planes = []
    for i in range(num_steps):
        planes.append(
            _make_horizontal_plane_points(
                z=i * step_height,
                x_range=(-0.6, 0.6),
                y_range=(i * tread_depth, (i + 1) * tread_depth),
                n=400,
                noise_std=0.002,
                rng=rng,
            )
        )

    outliers = rng.uniform(
        low=(-1.0, -1.0, -0.2),
        high=(1.0, num_steps * tread_depth + 1.0, num_steps * step_height + 0.5),
        size=(200, 3),
    )
    points = np.vstack([*planes, outliers])

    extracted = extract_stair_planes(
        points,
        max_planes=30,
        min_inliers=150,
        distance_threshold=0.01,
        max_tilt_deg=10.0,
        height_eps=0.03,
        ransac_n=3,
        num_iterations=2000,
        horizontal_only=True,
        merge_by_height=True,
        verbose=False,
    )

    assert len(extracted) >= num_steps
    assert all(p.normal[2] >= 0.0 for p in extracted)


def test_filter_horizontal_planes_rejects_tilted_and_flips_normal_up():
    horizontal_down = PlaneParam(
        normal=np.array([0.0, 0.0, -1.0]),
        point=np.array([0.0, 0.0, 0.0]),
        inlier_count=100,
        inlier_indices=np.arange(100, dtype=int),
        height=0.0,
    )

    tilted = PlaneParam(
        normal=np.array([0.5, 0.0, 0.8660254]),  # 30 deg tilt from vertical
        point=np.array([0.0, 0.0, 1.0]),
        inlier_count=80,
        inlier_indices=np.arange(80, dtype=int),
        height=1.0,
    )

    filtered = filter_horizontal_planes([horizontal_down, tilted], max_tilt_deg=15.0, verbose=False)
    assert len(filtered) == 1
    assert filtered[0].normal[2] > 0.99


def test_merge_planes_by_height_merges_and_refits_combined_points():
    rng = np.random.default_rng(1)
    p1 = _make_horizontal_plane_points(
        z=0.0,
        x_range=(-1.0, -0.1),
        y_range=(0.0, 1.0),
        n=200,
        noise_std=0.001,
        rng=rng,
    )
    p2 = _make_horizontal_plane_points(
        z=0.002,
        x_range=(0.1, 1.0),
        y_range=(0.0, 1.0),
        n=250,
        noise_std=0.001,
        rng=rng,
    )
    points = np.vstack([p1, p2])
    idx1 = np.arange(len(p1), dtype=int)
    idx2 = np.arange(len(p1), len(points), dtype=int)

    plane1 = PlaneParam(
        normal=np.array([0.0, 0.0, 1.0]),
        point=p1.mean(axis=0),
        inlier_count=len(idx1),
        inlier_indices=idx1,
        height=float(p1.mean(axis=0)[2]),
    )
    plane2 = PlaneParam(
        normal=np.array([0.0, 0.0, -1.0]),
        point=p2.mean(axis=0),
        inlier_count=len(idx2),
        inlier_indices=idx2,
        height=float(p2.mean(axis=0)[2]),
    )

    merged = merge_planes_by_height(
        [plane1, plane2],
        points,
        height_eps=0.03,
        distance_threshold=0.01,
        verbose=False,
    )

    assert len(merged) == 1
    merged_plane = merged[0]
    assert merged_plane.inlier_count == len(points)
    assert merged_plane.normal[2] > 0.99
    assert set(merged_plane.inlier_indices.tolist()) == set(range(len(points)))

