import numpy as np

import main
from primitives import PlaneParam


def test_rect_patch_generation_contains_points():
    rng = np.random.default_rng(123)

    normal = np.array([0.2, -0.1, 1.0])
    normal = normal / np.linalg.norm(normal)
    u, v = main._plane_basis_from_normal(normal)

    width = 1.4
    height = 0.7
    angle = 0.65
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    samples = rng.uniform(
        low=[-width / 2, -height / 2],
        high=[width / 2, height / 2],
        size=(800, 2),
    )
    samples = samples @ rot.T
    noise = rng.normal(scale=0.002, size=len(samples))
    points = (
        samples[:, 0:1] * u[None, :]
        + samples[:, 1:2] * v[None, :]
        + noise[:, None] * normal[None, :]
    )

    plane = PlaneParam(
        normal=normal,
        point=np.zeros(3),
        inlier_count=len(points),
        inlier_indices=np.arange(len(points), dtype=int),
        height=float(points[:, 2].mean()),
    )

    mesh, metrics = main.create_plane_patch_mesh(
        plane,
        points,
        color=np.array([0.0, 0.5, 0.0]),
        padding=0.0,
        patch_shape="rect",
    )

    assert metrics.get("patch_shape") == "rect"
    rect_world = np.asarray(metrics.get("rect_corners_world"))
    assert rect_world.shape == (4, 3)

    triangles = np.asarray(mesh.triangles)
    assert triangles.shape[0] == 2

    edge_u = rect_world[1] - rect_world[0]
    edge_v = rect_world[3] - rect_world[0]
    u_len = float(np.linalg.norm(edge_u))
    v_len = float(np.linalg.norm(edge_v))
    u_dir = edge_u / max(u_len, 1e-9)
    v_dir = edge_v / max(v_len, 1e-9)

    projected = points - np.dot(points - rect_world[0], normal)[:, None] * normal[None, :]
    rel = projected - rect_world[0]
    u_coords = rel @ u_dir
    v_coords = rel @ v_dir

    tol = 0.03
    assert np.all(u_coords >= -tol)
    assert np.all(v_coords >= -tol)
    assert np.all(u_coords <= u_len + tol)
    assert np.all(v_coords <= v_len + tol)
