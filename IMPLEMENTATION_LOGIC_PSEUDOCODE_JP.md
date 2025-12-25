# 擬似コード付き 実装ロジック整理（卒論用）

最終更新: 2025-12-25

本書は `primitives.py` / `main.py` / `gui_app.py` をベースに、
**主要アルゴリズムを擬似コードで再構成**したものです。

---

## 1. 全体フロー（CLI/GUI共通）

```text
input: pointcloud_path, mode, parameters
output: fitted primitives (JSON + visualization)

pcd = load_point_cloud(pointcloud_path)
if preprocess:
    pcd = voxel_downsample(pcd, voxel_size)
    pcd = remove_outlier(pcd, nb_neighbors, std_ratio)
    pcd = estimate_normals(pcd, normal_radius)

seed_or_roi = user_pick(shift_click)

switch mode:
  plane:
      plane = fit_plane(roi_points)
      visualize_and_save(plane)
  cylinder:
      cyl = probe_cylinder_from_seed(seed_center)
      visualize_and_save(cyl)
  seed_expand:
      result = expand_plane_from_seed(...) or expand_cylinder_from_seed(...)
      visualize_and_save(result)
  stairs:
      planes = extract_stair_planes(roi_points)
      visualize_and_save(planes)
```

---

## 2. ROI/seed選択（適応半径）

```text
function select_roi_adaptive(center, r_min, r_max, r_step, min_points):
    r = r_min
    while r <= r_max:
        idx = points_within_radius(center, r)
        if count(idx) >= min_points:
            return ROI(points[idx], center, r)
        r += r_step
    return failure
```

---

## 3. 平面推定（RANSAC）

```text
function fit_plane(points, distance_threshold, ransac_n, iters):
    if len(points) < 3: return None
    model, inliers = RANSAC_plane(points)
    if inliers empty: return None
    n = normalize(model.normal)
    if n.z < 0: n = -n
    centroid = mean(points[inliers])
    p0 = project(centroid, model)
    return PlaneParam(normal=n, point=p0, inliers=inliers)
```

---

## 4. 円柱推定（RANSACベース）

```text
function fit_cylinder(points, normals=None):
    best = None
    repeat num_iterations:
        sample = random_sample(points)
        axis_dir = first_pc_axis(sample)   # SVD
        radius = median(radial_distance(sample, axis_dir))
        if radius not in [min,max]: continue

        inliers = |radial_distance(points, axis_dir) - radius| < threshold
        if normals:
            inliers &= dot(radial_dir, normals) > cos(angle_th)

        if better: best = inliers

    if best too small: return None
    axis_dir = first_pc_axis(points[best])
    radius = median(radial_distance(points[best], axis_dir))
    length = robust_length(project(points[best], axis_dir))
    return CylinderParam(axis_point=centroid, axis_dir, radius, length, inliers)
```

---

## 5. Seed‑Expand 平面

```text
function expand_plane_from_seed(points, seed_center):
    seed = points within seed_radius
    initial_plane = fit_plane(seed)
    if fail: return failure

    candidates = points within max_expand_radius
    mask = plane_distance(candidates, initial_plane) < th
    if normals: mask &= normal_alignment > cos(normal_th)
    candidate_points = candidates[mask]

    expanded = extract_connected_component(candidate_points, seed_inliers)

    for iter in 1..max_refine_iters:
        n = SVD_normal(points[expanded])
        if adaptive_threshold:
            th = median + k * MAD
        expanded = extract_connected_component(... using new plane ...)
        if no change: break

    area, extent = compute_plane_metrics(points[expanded])
    return SeedExpandPlaneResult(plane, expanded, area, extent)
```

---

## 6. Seed‑Expand 円柱

```text
function expand_cylinder_from_seed(points, seed_center):
    seed = points within seed_radius
    initial_cyl = fit_cylinder(seed)
    if fail: return failure

    candidates = points within max_expand_radius
    mask = |radial_distance - radius| < threshold
    candidate_points = candidates[mask]

    expanded = extract_connected_component(candidate_points, seed_inliers)

    for iter in 1..refine_iters:
        cyl = estimate_cylinder_from_points(points[expanded])
        expanded = extract_connected_component(... using new cyl ...)
        if no change: break

    length = recompute_length(expanded)
    return SeedExpandCylinderResult(cyl, expanded)
```

---

## 7. Cylinder Probe（円柱専用の強化版）

```text
function probe_cylinder_from_seed(points, seed_center):
    proxy = compute_cylinder_proxy_from_seed(seed_center)
    if fail: return failure
    final = finalize_cylinder_from_proxy(proxy)
    return CylinderProbeResult(proxy, final)
```

### 7.1 代理円柱の探索
```text
function compute_cylinder_proxy_from_seed(seed_center):
    best = None
    for seed_radius in [start..max step]:
        seed = points within seed_radius
        cyl = fit_cylinder(seed)
        score = inliers / max(median_residual, eps)
        if score best: best = cyl
    return best
```

### 7.2 最終化
```text
function finalize_cylinder_from_proxy(proxy):
    surface = extract_cylinder_surface_component(proxy)
    cyl = estimate_cylinder_from_points(surface)
    if axis_refit: refine_axis(cyl)
    length = recompute_length(surface)
    return cyl
```

---

## 8. Stairs Mode

```text
function extract_stair_planes(points):
    planes = []
    remaining = points
    while remaining large and planes < max:
        plane = RANSAC_plane(remaining)
        if inliers too small: break
        planes.append(plane)
        remaining = remaining - inliers

    planes = filter_horizontal_planes(planes, tilt_threshold)
    planes = merge_planes_by_height(planes)
    return planes
```

---

## 9. GUIの主要イベント

```text
on_shift_click():
    if edit_mode:
        delete_points_in_radius()
    else:
        seed_center = picked_point
        if auto_run: run_fit()

on_box_select_mode (2-click):
    if first click: store start
    if second click: delete points in rect
```

---

必要なら「関数単位で完全分解版（入力・出力・計算量も付記）」を作成できます。
