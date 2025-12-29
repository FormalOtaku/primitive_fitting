# Cylinder Probe Improvement Notes

## Goal
Reduce two common probe failures:
- Cylinder axis appears tilted (slanted) even when the object is vertical.
- Cylinder mesh appears too short (ends are truncated).

These notes explain the current fix, the reasoning, and what to try next if the issue persists.

## Likely Root Causes
1. **Length truncation during probe**
   The probe pipeline uses the proxy cylinder length to define the extraction slab. If the seed region is small, the proxy length is short, and the extraction step only sees a short segment. The final fit can never grow beyond that.

2. **Axis drift during refinement**
   The final refinement step refits the axis using PCA of the extracted points. If the extracted points are short or sparse, PCA can yield a slanted axis, even if the proxy axis was correct.

## Changes Implemented
### 1) Allow length growth during probe extraction
- In `main.py`, the probe finalize call now sets `allow_length_growth=True`.
- This expands the extraction slab to `max_expand_radius * 2`, letting the fit see the full cylinder.

### 2) Make axis refit opt-in for probe
- New CLI flag: `--cyl-probe-axis-refit` (default OFF).
- By default, probe refinement keeps the proxy/user axis direction. This avoids PCA-induced tilt after manual adjustment.

### 3) More robust axis selection when refit is enabled
- In `primitives.py`, probe refinement now tests multiple axis candidates:
  - the current reference axis
  - PCA axis of inlier points
  - axis inferred from normals (if available)
- The best candidate is chosen by residual median, with a tie-break that prefers smaller deviation from the reference axis.

### 4) Slight preference for normal-derived axis in proxy init
- During proxy initialization, candidates derived from normals get a small score bonus.
- This helps when PCA is ambiguous on small seed patches.

## How To Use
Recommended probe command (same as before):

```
python main.py --input <your_cloud.ply> --cyl-probe --sensor-profile default
```

Optional axis refit (if you want PCA/normals to update the axis):

```
python main.py --input <your_cloud.ply> --cyl-probe --sensor-profile default --cyl-probe-axis-refit
```

## Quick Sanity Checks
- If the cylinder looks too short, increase `--max-expand-radius` or verify that length growth is ON in the logs.
- If the cylinder is still slanted, try:
  1) Use the probe UI to rotate the axis and press `Enter`.
  2) Run with `--cyl-probe-axis-refit` and compare.
  3) Increase normal estimation radius in preprocessing (`preprocess_point_cloud` in `main.py`) if normals are noisy.

## Implemented Improvements

### 1. Axis regularization ✅
Added `compute_axis_regularized_score()` function in `primitives.py`. During axis candidate selection in `finalize_cylinder_from_proxy`, axes that deviate from the reference axis receive a penalty score proportional to their angle difference. This keeps the axis stable unless residuals improve significantly.

- **CLI flag**: `--cyl-probe-axis-reg-weight` (default: 0.02)
- Formula: `score = residual_median * (1 + weight * angle_deg)`

### 2. Length estimation from inliers ✅
Added `recompute_cylinder_length_from_inliers()` function. After final inliers are extracted, the length is recomputed using 1% and 99% quantiles of axis-projected positions. The axis center is also adjusted.

- **CLI flag**: `--cyl-probe-no-recompute-length` (to disable)
- Default quantiles: 1% and 99%

### 3. Normals-aware axis estimation ✅
Already implemented via `_axis_from_normals()`. The axis is computed as the smallest eigenvector of the normal covariance matrix. In `compute_cylinder_proxy_from_seed`, normal-derived axes receive a 1.1x score bonus.

### 4. Axis snapping heuristics ✅
Added `snap_axis_to_vertical()` function. If the fitted axis is within N degrees of +Z/-Z, it snaps to exactly vertical.

- **CLI flag**: `--cyl-probe-axis-snap-deg` (default: 0 = disabled)
- Example: `--cyl-probe-axis-snap-deg 5` snaps axes within 5° of vertical

### 5. Export diagnostics ✅
Added `export_cylinder_diagnostics_ply()` and `create_cylinder_mesh()` functions. These export PLY files for visualization:
- `*_seed.ply`: Seed points (yellow)
- `*_candidates.ply`: Candidate surface points (cyan)
- `*_inliers.ply`: Final inliers (green)
- `*_cylinder.ply`: Fitted cylinder mesh (red)

- **CLI flag**: `--cyl-probe-diagnostics-dir <dir>`

## Files Touched
- `main.py`
  - Added `--cyl-probe-axis-refit`
  - Added `--cyl-probe-axis-snap-deg`
  - Added `--cyl-probe-axis-reg-weight`
  - Added `--cyl-probe-no-recompute-length`
  - Added `--cyl-probe-diagnostics-dir`
  - Probe finalize now uses length growth and axis lock by default

- `primitives.py`
  - Added `recompute_cylinder_length_from_inliers()` - quantile-based length estimation
  - Added `snap_axis_to_vertical()` - axis snapping heuristic
  - Added `compute_axis_regularized_score()` - angle-penalized axis scoring
  - Added `export_cylinder_diagnostics_ply()` - PLY export for debugging
  - Added `create_cylinder_mesh()` - helper for mesh generation
  - `finalize_cylinder_from_proxy()` now accepts new parameters for regularization, snapping, and length recomputation
  - Proxy init prefers normals slightly
  - Probe refinement considers multiple axis candidates with regularized scoring
