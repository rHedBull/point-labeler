#!/usr/bin/env python3
"""RANSAC primitive fitting prototype for industrial point cloud instance segmentation.

Strategy v4: principal curvature (k1, k2) based classification + RANSAC.

Key insight: k1 and k2 principal curvatures identify surface type directly:
  - Plane:    k1 ≈ 0, k2 ≈ 0
  - Cylinder: k1 ≈ 1/r, k2 ≈ 0  (one direction curves, other is straight)
  - Sphere:   k1 ≈ k2 ≈ 1/r     (curves equally in all directions)
  - Saddle:   k1 > 0, k2 < 0     (junction/transition point)

Pipeline:
  1. Compute normals + principal curvatures (k1, k2)
  2. Classify each point by (k1, k2) signature → plane/cylinder/sphere/saddle
  3. Planes: iterative RANSAC
  4. Cylinders: cluster by (position + axis direction + radius), fit per cluster
  5. Classify by geometry
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from collections import Counter
from pathlib import Path
import time
import json


def _compute_curvatures_at_scale(points, normals, nn_idx):
    """Vectorized principal curvature computation for a single neighbor scale.

    Args:
        points: (N, 3)
        normals: (N, 3)
        nn_idx: (N, k) neighbor indices (excluding self)

    Returns:
        k1, k2: (N,) principal curvatures
        directions, axes: (N, 3) principal directions
        residuals: (N,) quadratic fit residuals (lower = better fit)
    """
    n = len(points)
    k = nn_idx.shape[1]

    # Build local tangent frames for all points at once
    # Pick reference vector: [1,0,0] unless normal is near-parallel to it
    ref = np.where(np.abs(normals[:, 0:1]) < 0.9,
                   np.broadcast_to([1, 0, 0], (n, 3)),
                   np.broadcast_to([0, 1, 0], (n, 3)))
    t1 = np.cross(normals, ref)
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True) + 1e-10
    t1 = t1 / t1_norm
    t2 = np.cross(normals, t1)

    # Gather neighbors: (N, k, 3)
    neighbors = points[nn_idx]

    # diff: (N, k, 3)
    diff = neighbors - points[:, np.newaxis, :]

    # Project to local frames: u, v, w are (N, k)
    u = np.einsum('nkd,nd->nk', diff, t1)
    v = np.einsum('nkd,nd->nk', diff, t2)
    w = np.einsum('nkd,nd->nk', diff, normals)

    # Build design matrix A: (N, k, 3) for quadratic fit w ≈ a*u² + b*u*v + c*v²
    A = np.stack([u**2, u*v, v**2], axis=-1)  # (N, k, 3)

    # Solve all least-squares problems at once via normal equations: (AᵀA) x = Aᵀw
    AtA = np.einsum('nki,nkj->nij', A, A)  # (N, 3, 3)
    Atw = np.einsum('nki,nk->ni', A, w)    # (N, 3)

    # Regularize to avoid singular matrices
    AtA[:, 0, 0] += 1e-8
    AtA[:, 1, 1] += 1e-8
    AtA[:, 2, 2] += 1e-8

    # Batched solve
    try:
        coeffs = np.linalg.solve(AtA, Atw)  # (N, 3) -> [a, b, c] per point
    except np.linalg.LinAlgError:
        # Fallback to per-point solve for numerical issues
        coeffs = np.zeros((n, 3))
        for i in range(n):
            try:
                coeffs[i] = np.linalg.solve(AtA[i], Atw[i])
            except np.linalg.LinAlgError:
                pass

    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]

    # Compute fit residuals: ||w - A @ coeffs||² / k
    w_pred = np.einsum('nki,ni->nk', A, coeffs)  # (N, k)
    residuals = np.mean((w - w_pred)**2, axis=1)   # (N,)

    # Shape operator eigenvalues via closed-form 2x2 symmetric eigen
    # S = [[2a, b], [b, 2c]]
    trace = 2*a + 2*c
    det = 4*a*c - b*b
    disc = np.sqrt(np.maximum(trace**2 - 4*det, 0))

    lam1 = (trace + disc) / 2  # larger eigenvalue
    lam2 = (trace - disc) / 2  # smaller eigenvalue

    # Sort by absolute value
    abs1 = np.abs(lam1)
    abs2 = np.abs(lam2)
    swap = abs1 < abs2

    k1 = np.where(swap, lam2, lam1)
    k2 = np.where(swap, lam1, lam2)

    # Eigenvectors for the 2x2 shape operator
    # For eigenvalue lam: (2a - lam) * v0 + b * v1 = 0 => v = [-b, 2a - lam] (unnormalized)
    # Use the eigenvector for k1 (max |curvature|) and k2 (min |curvature|)
    lam_max = np.where(swap, lam2, lam1)
    lam_min = np.where(swap, lam1, lam2)

    vmax_0 = -b
    vmax_1 = 2*a - lam_max
    vmax_norm = np.sqrt(vmax_0**2 + vmax_1**2) + 1e-10
    vmax_0 /= vmax_norm
    vmax_1 /= vmax_norm

    vmin_0 = -b
    vmin_1 = 2*a - lam_min
    vmin_norm = np.sqrt(vmin_0**2 + vmin_1**2) + 1e-10
    vmin_0 /= vmin_norm
    vmin_1 /= vmin_norm

    # Convert to world coordinates
    directions = vmax_0[:, np.newaxis] * t1 + vmax_1[:, np.newaxis] * t2
    axes = vmin_0[:, np.newaxis] * t1 + vmin_1[:, np.newaxis] * t2

    return k1, k2, directions, axes, residuals


def compute_principal_curvatures(points, normals, k=20):
    """Compute principal curvatures k1, k2 and principal directions per point.

    Uses shape operator (Weingarten map) estimated from local neighborhoods.
    Multi-scale: computes at k=10, 20, 40 and picks the scale with the
    lowest quadratic fit residual per point.

    Returns:
        k1: (N,) larger principal curvature (abs)
        k2: (N,) smaller principal curvature (abs)
        directions: (N, 3) principal direction for k1 (cylinder axis is perpendicular to this)
        axes: (N, 3) estimated cylinder axis (direction of zero curvature)
    """
    n = len(points)
    tree = cKDTree(points)

    scales = [10, k, 40]
    max_k = max(scales)
    _, nn_idx_all = tree.query(points, k=max_k + 1)
    nn_idx_all = nn_idx_all[:, 1:]  # drop self

    # Compute curvatures at each scale
    all_k1 = []
    all_k2 = []
    all_dirs = []
    all_axes = []
    all_residuals = []

    for s in scales:
        nn_idx_s = nn_idx_all[:, :s]
        k1_s, k2_s, dir_s, ax_s, res_s = _compute_curvatures_at_scale(
            points, normals, nn_idx_s)
        all_k1.append(k1_s)
        all_k2.append(k2_s)
        all_dirs.append(dir_s)
        all_axes.append(ax_s)
        all_residuals.append(res_s)

    # Stack: (n_scales, N)
    residuals = np.stack(all_residuals, axis=0)  # (3, N)
    best_scale = np.argmin(residuals, axis=0)     # (N,) index into scales

    # Gather best results per point
    k1 = np.choose(best_scale, all_k1)
    k2 = np.choose(best_scale, all_k2)

    # For 3D arrays, need manual indexing
    all_dirs = np.stack(all_dirs, axis=0)   # (3, N, 3)
    all_axes = np.stack(all_axes, axis=0)   # (3, N, 3)
    idx = np.arange(n)
    directions = all_dirs[best_scale, idx]
    axes = all_axes[best_scale, idx]

    # Report scale usage
    for i, s in enumerate(scales):
        count = (best_scale == i).sum()
        print(f"    scale k={s}: {count} points ({100*count/n:.1f}%)")

    # Canonicalize axes (consistent direction)
    flip_mask = axes[:, 2] < 0
    flip_mask2 = (np.abs(axes[:, 2]) < 0.1) & (axes[:, 0] < 0)
    axes[flip_mask | flip_mask2] *= -1

    return k1, k2, directions, axes


def classify_points_by_curvature(k1, k2, cylinder_ratio_thresh=3.0, flat_thresh=0.5):
    """Classify points into surface types based on principal curvatures.

    Returns:
        labels: (N,) array with:
            0 = flat (plane)
            1 = cylindrical (one curvature >> other)
            2 = spherical (both curvatures similar and high)
            3 = saddle/junction (curvatures have opposite signs or complex)
            4 = edge/noisy (very high curvature, unreliable)
    """
    n = len(k1)
    labels = np.full(n, 3, dtype=np.int32)  # default: saddle/junction

    abs_k1 = np.abs(k1)
    abs_k2 = np.abs(k2)
    k_max = np.maximum(abs_k1, abs_k2)
    k_min = np.minimum(abs_k1, abs_k2)

    # Flat: both curvatures near zero
    flat = k_max < flat_thresh
    labels[flat] = 0

    # Cylindrical: one curvature >> other
    ratio = k_max / (k_min + 1e-6)
    cylindrical = ~flat & (ratio > cylinder_ratio_thresh) & (k_max < 50)
    labels[cylindrical] = 1

    # Spherical: both curvatures similar and high
    spherical = ~flat & ~cylindrical & (ratio < 2.0) & (k_max < 50)
    labels[spherical] = 2

    # Edge/noisy: extremely high curvature
    noisy = k_max > 50
    labels[noisy] = 4

    return labels


def iterative_plane_ransac(points, normals, indices,
                           distance_threshold=0.025,
                           min_inliers=100,
                           max_planes=30):
    """Extract planes iteratively from flat points."""
    remaining = indices.copy()
    planes = []

    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[remaining])

        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=500,
        )

        if len(inliers) < min_inliers:
            break

        global_inliers = remaining[inliers]
        a, b, c, d = model
        extent_pts = points[global_inliers]
        extent = np.linalg.norm(extent_pts.max(axis=0) - extent_pts.min(axis=0))

        planes.append({
            "type": "plane",
            "label": "flat_surface",
            "normal": [float(a), float(b), float(c)],
            "d": float(d),
            "n_points": len(global_inliers),
            "extent": float(extent),
            "global_indices": global_inliers,
        })

        remaining = np.setdiff1d(remaining, global_inliers)
        print(f"    Plane: {len(global_inliers)} pts, "
              f"normal=[{a:.2f},{b:.2f},{c:.2f}], extent={extent:.2f}m")

    return planes, remaining


def fit_cylinder_to_cluster(points, normals, axis_hint=None,
                            distance_threshold=0.03,
                            min_radius=0.015, max_radius=2.0):
    """Fit cylinder using given or estimated axis, then circle fit perpendicular to it."""
    n = len(points)
    if n < 20:
        return None

    if axis_hint is not None:
        axis = axis_hint / (np.linalg.norm(axis_hint) + 1e-10)
    else:
        mean_n = normals.mean(axis=0)
        centered_n = normals - mean_n
        try:
            _, s, Vt = np.linalg.svd(centered_n, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        axis = Vt[-1]
        axis = axis / (np.linalg.norm(axis) + 1e-10)

    centroid = points.mean(axis=0)
    vecs = points - centroid
    along = np.dot(vecs, axis)
    perp = vecs - along[:, np.newaxis] * axis

    e1 = np.array([1, 0, 0]) - np.dot([1, 0, 0], axis) * axis
    if np.linalg.norm(e1) < 0.1:
        e1 = np.array([0, 1, 0]) - np.dot([0, 1, 0], axis) * axis
    e1 = e1 / (np.linalg.norm(e1) + 1e-10)
    e2 = np.cross(axis, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-10)

    coords_2d = np.column_stack([np.dot(perp, e1), np.dot(perp, e2)])

    # Algebraic circle fit
    A_mat = np.column_stack([coords_2d, np.ones(n)])
    b_vec = -(coords_2d[:, 0]**2 + coords_2d[:, 1]**2)
    try:
        result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        D, E, F = result[0]
    except np.linalg.LinAlgError:
        return None

    cx, cy = -D/2, -E/2
    r_sq = cx**2 + cy**2 - F
    if r_sq <= 0:
        return None
    radius = np.sqrt(r_sq)
    if radius < min_radius or radius > max_radius:
        return None

    dists = np.sqrt((coords_2d[:, 0] - cx)**2 + (coords_2d[:, 1] - cy)**2)
    residuals = np.abs(dists - radius)
    best_mask = residuals < distance_threshold
    best_n = best_mask.sum()
    best_r = radius
    best_cx, best_cy = cx, cy

    # RANSAC refinement
    for _ in range(min(200, n * 3)):
        idx = np.random.choice(n, 3, replace=False)
        sub = coords_2d[idx]
        A_s = np.column_stack([sub, np.ones(3)])
        b_s = -(sub[:, 0]**2 + sub[:, 1]**2)
        try:
            res = np.linalg.lstsq(A_s, b_s, rcond=None)
            D, E, F = res[0]
        except np.linalg.LinAlgError:
            continue
        cx_t, cy_t = -D/2, -E/2
        r_sq_t = cx_t**2 + cy_t**2 - F
        if r_sq_t <= 0:
            continue
        r_t = np.sqrt(r_sq_t)
        if r_t < min_radius or r_t > max_radius:
            continue
        d_t = np.sqrt((coords_2d[:, 0] - cx_t)**2 + (coords_2d[:, 1] - cy_t)**2)
        mask_t = np.abs(d_t - r_t) < distance_threshold
        n_t = mask_t.sum()
        if n_t > best_n:
            best_n = n_t
            best_r = r_t
            best_cx, best_cy = cx_t, cy_t
            best_mask = mask_t

    center_3d = centroid + best_cx * e1 + best_cy * e2
    inlier_along = along[best_mask]
    length = inlier_along.max() - inlier_along.min() if best_mask.sum() > 0 else 0

    return {
        "center": center_3d,
        "axis": axis,
        "radius": best_r,
        "length": length,
        "n_inliers": int(best_n),
        "inlier_ratio": best_n / n,
    }


def classify_cylinder(radius, length):
    aspect = length / (2 * radius) if radius > 0 else 0
    if radius > 0.4:
        return "tank"
    elif radius > 0.15:
        return "tank" if aspect < 3 else "large_pipe"
    elif radius > 0.03:
        if aspect > 5:
            return "pipe"
        elif aspect > 1.5:
            return "short_pipe"
        else:
            return "fitting"
    else:
        return "small_pipe"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RANSAC primitive segmentation for industrial point clouds")
    parser.add_argument("--input", required=True, help="Input PLY point cloud")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points).copy()
    print(f"  {len(points)} points, extent: {points.ptp(axis=0).round(2)}")

    print("\nEstimating normals...")
    t0 = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    normals = np.asarray(pcd.normals).copy()
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Step 1: Principal curvatures
    print("\n=== Step 1: Principal Curvature Estimation ===")
    t0 = time.time()
    k1, k2, directions, cyl_axes = compute_principal_curvatures(points, normals, k=20)
    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"  k1: median={np.median(np.abs(k1)):.3f}, p95={np.percentile(np.abs(k1), 95):.3f}")
    print(f"  k2: median={np.median(np.abs(k2)):.3f}, p95={np.percentile(np.abs(k2), 95):.3f}")

    # Step 2: Classify by (k1, k2) signature
    print("\n=== Step 2: Surface Classification by (k1, k2) ===")
    surface_labels = classify_points_by_curvature(k1, k2, cylinder_ratio_thresh=3.0, flat_thresh=0.5)

    type_names = {0: "flat", 1: "cylindrical", 2: "spherical", 3: "saddle", 4: "edge/noisy"}
    for t, name in type_names.items():
        count = (surface_labels == t).sum()
        print(f"  {name}: {count} ({100*count/len(points):.1f}%)")

    instance_labels = np.full(len(points), -1, dtype=np.int32)
    primitives = []
    next_id = 0

    # Step 3: Extract planes from flat points
    print(f"\n=== Step 3: Plane Extraction (flat points) ===")
    t0 = time.time()
    flat_indices = np.where(surface_labels == 0)[0]
    planes, flat_remaining = iterative_plane_ransac(
        points, normals, flat_indices,
        distance_threshold=0.025,
        min_inliers=80,
        max_planes=25,
    )
    for p in planes:
        p["id"] = next_id
        instance_labels[p["global_indices"]] = next_id
        next_id += 1
    primitives.extend(planes)
    print(f"  {len(planes)} planes, {sum(p['n_points'] for p in planes)} pts ({time.time()-t0:.1f}s)")

    # Step 4: Region-growing cylinder clustering
    # Instead of DBSCAN in 7D feature space, grow regions along pipes:
    # two neighboring cylindrical points belong to the same instance if
    # they have similar axis direction and similar estimated radius.
    print(f"\n=== Step 4: Region-Growing Cylinder Clustering ===")
    t0 = time.time()
    cyl_indices = np.where(surface_labels == 1)[0]
    cyl_pts = points[cyl_indices]
    cyl_ax = cyl_axes[cyl_indices]
    cyl_k1 = np.abs(k1[cyl_indices])

    # Estimated radius from k1
    est_radius = np.where(cyl_k1 > 0.1, 1.0 / cyl_k1, 10.0)
    est_radius = np.clip(est_radius, 0.01, 5.0)

    print(f"  {len(cyl_indices)} cylindrical points")
    print(f"  Radius distribution: median={np.median(est_radius):.3f}m, "
          f"mean={est_radius.mean():.3f}m")

    # Build spatial KD-tree on cylindrical points only
    cyl_tree = cKDTree(cyl_pts)

    # Region growing parameters
    search_radius = 0.12  # spatial neighbor distance (meters)
    axis_thresh = 0.92    # cos(~23°) for axis similarity
    radius_ratio = 1.8    # max ratio between estimated radii

    # Precompute neighbor lists (faster than querying per-point during growth)
    print(f"  Building neighbor graph (r={search_radius}m)...")
    neighbor_lists = cyl_tree.query_ball_tree(cyl_tree, r=search_radius)

    cyl_cluster = np.full(len(cyl_indices), -1, dtype=np.int32)
    cluster_id = 0

    # Sort by point density (start from denser regions = more reliable)
    density = np.array([len(nl) for nl in neighbor_lists])
    seed_order = np.argsort(-density)

    print(f"  Growing regions...")
    for seed in seed_order:
        if cyl_cluster[seed] >= 0:
            continue

        # Start new region
        region = [seed]
        cyl_cluster[seed] = cluster_id
        queue = [seed]
        seed_axis = cyl_ax[seed].copy()
        seed_radius = est_radius[seed]

        # Running axis estimate (updated as region grows)
        axis_sum = cyl_ax[seed].copy()
        n_in_region = 1

        while queue:
            current = queue.pop(0)
            cur_axis = axis_sum / (np.linalg.norm(axis_sum) + 1e-10)
            cur_radius = seed_radius  # keep seed radius as reference

            for nb in neighbor_lists[current]:
                if cyl_cluster[nb] >= 0:
                    continue

                # Check axis compatibility (account for sign flip)
                dot = abs(np.dot(cyl_ax[nb], cur_axis))
                if dot < axis_thresh:
                    continue

                # Check radius compatibility
                r_ratio = max(est_radius[nb], cur_radius) / (min(est_radius[nb], cur_radius) + 1e-6)
                if r_ratio > radius_ratio:
                    continue

                # Accept this point
                cyl_cluster[nb] = cluster_id
                queue.append(nb)
                region.append(nb)

                # Update running axis (sign-consistent)
                sign = 1.0 if np.dot(cyl_ax[nb], cur_axis) >= 0 else -1.0
                axis_sum += sign * cyl_ax[nb]
                n_in_region += 1

        cluster_id += 1

    n_clusters = cluster_id
    noise_count = (cyl_cluster == -1).sum()
    assigned_count = (cyl_cluster >= 0).sum()
    print(f"  {n_clusters} raw regions, {assigned_count}/{len(cyl_indices)} assigned "
          f"({100*assigned_count/len(cyl_indices):.0f}%)")

    # Filter small clusters and fit cylinders
    cylinders_found = 0
    min_cluster_pts = 20

    # Collect cluster sizes for reporting
    cluster_sizes = []
    for c in range(n_clusters):
        cluster_sizes.append(int((cyl_cluster == c).sum()))
    big_clusters = sum(1 for s in cluster_sizes if s >= min_cluster_pts)
    print(f"  {big_clusters} clusters with >= {min_cluster_pts} points")
    top_sizes = sorted(cluster_sizes, reverse=True)[:15]
    print(f"  Top cluster sizes: {top_sizes}")

    for c in range(n_clusters):
        c_mask = cyl_cluster == c
        n_pts = c_mask.sum()
        if n_pts < min_cluster_pts:
            continue

        c_global = cyl_indices[c_mask]
        c_pts = points[c_global]
        c_normals = normals[c_global]
        c_axis = cyl_ax[c_mask]
        # Consistent axis direction
        signs = np.sign(np.dot(c_axis, c_axis.mean(axis=0)))
        signs[signs == 0] = 1
        mean_axis = (c_axis * signs[:, np.newaxis]).mean(axis=0)
        mean_axis = mean_axis / (np.linalg.norm(mean_axis) + 1e-10)

        cyl = fit_cylinder_to_cluster(c_pts, c_normals, axis_hint=mean_axis,
                                      distance_threshold=0.03)

        if cyl and cyl["inlier_ratio"] > 0.25:
            label = classify_cylinder(cyl["radius"], cyl["length"])
            primitives.append({
                "id": next_id,
                "type": "cylinder",
                "label": label,
                "radius": float(cyl["radius"]),
                "length": float(cyl["length"]),
                "center": cyl["center"].tolist(),
                "axis": cyl["axis"].tolist(),
                "n_points": int(n_pts),
                "inlier_ratio": float(cyl["inlier_ratio"]),
                "global_indices": c_global,
            })
            instance_labels[c_global] = next_id
            next_id += 1
            cylinders_found += 1

            if cylinders_found <= 20:
                med_r = np.median(est_radius[c_mask])
                print(f"    Cyl #{cylinders_found}: r={cyl['radius']:.3f}m (est:{med_r:.3f}m), "
                      f"L={cyl['length']:.2f}m, {n_pts} pts -> {label}")
        else:
            primitives.append({
                "id": next_id,
                "type": "unknown",
                "label": "curved_unclassified",
                "n_points": int(n_pts),
                "global_indices": c_global,
            })
            instance_labels[c_global] = next_id
            next_id += 1

    print(f"  {cylinders_found} cylinders ({time.time()-t0:.1f}s)")

    # Step 4b: Spherical points — cluster by position
    print(f"\n=== Step 4b: Spherical Clustering ===")
    sphere_indices = np.where(surface_labels == 2)[0]
    if len(sphere_indices) > 50:
        sphere_pts = points[sphere_indices]
        # Use small eps to avoid giant catchall clusters
        sphere_eps = 0.05
        print(f"  {len(sphere_indices)} spherical points, eps={sphere_eps:.3f}m")
        db_sphere = DBSCAN(eps=sphere_eps, min_samples=15).fit(sphere_pts)
        spheres_found = 0
        max_sphere_pts = len(points) // 20  # cap at 5% of total points
        for c in range(db_sphere.labels_.max() + 1):
            c_mask = db_sphere.labels_ == c
            if c_mask.sum() < 30 or c_mask.sum() > max_sphere_pts:
                continue
            c_global = sphere_indices[c_mask]
            primitives.append({
                "id": next_id,
                "type": "sphere",
                "label": "spherical",
                "n_points": int(c_mask.sum()),
                "global_indices": c_global,
            })
            instance_labels[c_global] = next_id
            next_id += 1
            spheres_found += 1
        print(f"  {spheres_found} sphere clusters")

    # Step 5: Post-assignment — absorb unassigned points into nearby instances
    # Now includes normal consistency: a point's normal must be compatible
    # with the instance it's joining (prevents cross-surface bleeding).
    print(f"\n=== Step 5: Post-Assignment (with normal consistency) ===")
    t0 = time.time()
    unassigned = np.where(instance_labels == -1)[0]
    print(f"  {len(unassigned)} unassigned points before post-assignment")

    # Build KD-tree of all points
    full_tree = cKDTree(points)

    # Precompute mean normal per instance for normal compatibility check
    instance_mean_normals = {}
    instance_types = {}
    for prim in primitives:
        pid = prim["id"]
        idx = prim.get("global_indices")
        if idx is not None and len(idx) > 0:
            mn = normals[idx].mean(axis=0)
            mn_norm = np.linalg.norm(mn)
            instance_mean_normals[pid] = mn / (mn_norm + 1e-10) if mn_norm > 0.1 else None
            instance_types[pid] = prim["type"]
        else:
            instance_mean_normals[pid] = None
            instance_types[pid] = prim.get("type", "unknown")

    # For each unassigned point, look at assigned neighbors and vote
    assign_radius = 0.10  # 10cm search
    normal_thresh_plane = 0.85   # cos(~32°) — planes need tighter normal agreement
    normal_thresh_other = 0.5    # cos(~60°) — cylinders/spheres are more permissive
    newly_assigned = 0
    normal_rejected = 0
    passes = 3  # multiple passes to propagate

    for pass_num in range(passes):
        unassigned = np.where(instance_labels == -1)[0]
        if len(unassigned) == 0:
            break

        changed = 0
        pass_rejected = 0
        # Batch query for speed
        un_pts = points[unassigned]
        un_neighbors = full_tree.query_ball_point(un_pts, r=assign_radius)

        for i, un_idx in enumerate(unassigned):
            nbs = un_neighbors[i]
            if len(nbs) < 3:
                continue

            # Count votes from assigned neighbors
            nb_labels = instance_labels[nbs]
            assigned_nbs = nb_labels[nb_labels >= 0]
            if len(assigned_nbs) < 3:
                continue

            # Majority vote
            counts = Counter(assigned_nbs.tolist())
            point_normal = normals[un_idx]

            # Try candidates in order of vote count
            assigned = False
            for cand_label, cand_count in counts.most_common():
                if cand_count < max(3, len(assigned_nbs) * 0.4):
                    break

                # Normal consistency check
                mean_n = instance_mean_normals.get(cand_label)
                if mean_n is not None:
                    itype = instance_types.get(cand_label, "unknown")
                    thresh = normal_thresh_plane if itype == "plane" else normal_thresh_other
                    dot = abs(np.dot(point_normal, mean_n))
                    if dot < thresh:
                        pass_rejected += 1
                        continue  # try next candidate

                instance_labels[un_idx] = cand_label
                changed += 1
                assigned = True
                break

        newly_assigned += changed
        normal_rejected += pass_rejected
        print(f"  Pass {pass_num+1}: assigned {changed} points, {pass_rejected} rejected by normal check")
        if changed == 0:
            break

    # Update primitive point counts and global_indices
    for prim in primitives:
        mask = instance_labels == prim["id"]
        prim["n_points"] = int(mask.sum())
        prim["global_indices"] = np.where(mask)[0]

    print(f"  Total newly assigned: {newly_assigned}, "
          f"normal-rejected: {normal_rejected} ({time.time()-t0:.1f}s)")

    # Step 6: Merge over-segmented cylinders
    print(f"\n=== Step 6: Merge Over-Segmented Cylinders ===")
    t0 = time.time()
    cyl_prims = [p for p in primitives if p["type"] == "cylinder"]
    merged = set()
    merge_count = 0

    for i, p1 in enumerate(cyl_prims):
        if p1["id"] in merged:
            continue
        for j, p2 in enumerate(cyl_prims):
            if j <= i or p2["id"] in merged:
                continue

            # Check axis alignment
            a1 = np.array(p1["axis"])
            a2 = np.array(p2["axis"])
            dot = abs(np.dot(a1, a2))
            if dot < 0.95:
                continue

            # Check radius similarity
            r_ratio = max(p1["radius"], p2["radius"]) / (min(p1["radius"], p2["radius"]) + 1e-6)
            if r_ratio > 1.4:
                continue

            # Check spatial proximity (centers should be close, accounting for along-axis distance)
            c1 = np.array(p1["center"])
            c2 = np.array(p2["center"])
            diff = c2 - c1
            along_dist = abs(np.dot(diff, a1))
            perp_dist = np.linalg.norm(diff - np.dot(diff, a1) * a1)

            # Perpendicular distance should be small (same pipe)
            if perp_dist > max(p1["radius"], p2["radius"]) * 0.5 + 0.05:
                continue

            # Along-axis gap should be small
            half_len = (p1["length"] + p2["length"]) / 2
            if along_dist > half_len + 0.3:
                continue

            # Merge p2 into p1
            idx2 = p2["global_indices"]
            instance_labels[idx2] = p1["id"]
            merged.add(p2["id"])
            merge_count += 1

    # Remove merged primitives and update
    primitives = [p for p in primitives if p["id"] not in merged]
    for prim in primitives:
        mask = instance_labels == prim["id"]
        prim["n_points"] = int(mask.sum())
        prim["global_indices"] = np.where(mask)[0]
        if prim["type"] == "cylinder" and mask.sum() > 0:
            cyl = fit_cylinder_to_cluster(
                points[prim["global_indices"]], normals[prim["global_indices"]],
                axis_hint=np.array(prim["axis"]), distance_threshold=0.03)
            if cyl:
                prim["radius"] = float(cyl["radius"])
                prim["length"] = float(cyl["length"])
                prim["center"] = cyl["center"].tolist()
                prim["axis"] = cyl["axis"].tolist()
                prim["inlier_ratio"] = float(cyl["inlier_ratio"])
                prim["label"] = classify_cylinder(cyl["radius"], cyl["length"])

    print(f"  Merged {merge_count} cylinder pairs ({time.time()-t0:.1f}s)")

    # Summary
    print(f"\n{'='*50}")
    print(f"=== RESULTS ===")
    print(f"{'='*50}")
    assigned = (instance_labels >= 0).sum()
    print(f"Points assigned: {assigned}/{len(points)} ({100*assigned/len(points):.1f}%)")
    print(f"Total instances: {next_id}")

    type_counts = Counter(p["label"] for p in primitives)
    print(f"\nBy type:")
    for label, count in sorted(type_counts.items()):
        pts = sum(p["n_points"] for p in primitives if p["label"] == label)
        print(f"  {label}: {count} instances, {pts} points")

    # Colorize
    type_colors = {
        "tank":                np.array([1.0, 0.15, 0.15]),
        "large_pipe":          np.array([0.15, 0.15, 1.0]),
        "pipe":                np.array([0.15, 0.8, 0.15]),
        "small_pipe":          np.array([0.0, 0.9, 0.9]),
        "short_pipe":          np.array([1.0, 0.5, 0.0]),
        "fitting":             np.array([1.0, 0.0, 1.0]),
        "flat_surface":        np.array([0.8, 0.8, 0.15]),
        "curved_unclassified": np.array([0.6, 0.3, 0.6]),
        "spherical":           np.array([0.9, 0.6, 0.1]),
    }

    colors = np.full((len(points), 3), 0.15)
    for prim in primitives:
        idx = prim.get("global_indices")
        if idx is None:
            continue
        base = type_colors.get(prim["label"], np.array([0.5, 0.5, 0.5]))
        variation = 0.1 * ((prim["id"] * 7) % 5 - 2) / 2.0
        colors[idx] = np.clip(base + variation, 0, 1)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    pcd_out.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path / "ransac_segmentation.ply"), pcd_out)

    # Instance colors
    instance_colors = np.full((len(points), 3), 0.15)
    for prim in primitives:
        idx = prim.get("global_indices")
        if idx is None:
            continue
        hue = (prim["id"] * 0.618033988) % 1.0
        h = hue * 6
        x = 1.0 - abs(h % 2 - 1)
        if h < 1: rgb = [1, x, 0]
        elif h < 2: rgb = [x, 1, 0]
        elif h < 3: rgb = [0, 1, x]
        elif h < 4: rgb = [0, x, 1]
        elif h < 5: rgb = [x, 0, 1]
        else: rgb = [1, 0, x]
        instance_colors[idx] = rgb

    pcd_inst = o3d.geometry.PointCloud()
    pcd_inst.points = o3d.utility.Vector3dVector(points)
    pcd_inst.colors = o3d.utility.Vector3dVector(instance_colors)
    o3d.io.write_point_cloud(str(output_path / "instance_colors.ply"), pcd_inst)

    # Also save curvature visualization
    k1_abs = np.abs(k1)
    k_vis = np.clip(k1_abs / np.percentile(k1_abs, 95), 0, 1)
    curv_colors = np.zeros((len(points), 3))
    curv_colors[:, 0] = k_vis  # red = high curvature
    curv_colors[:, 2] = 1 - k_vis  # blue = flat
    pcd_curv = o3d.geometry.PointCloud()
    pcd_curv.points = o3d.utility.Vector3dVector(points)
    pcd_curv.colors = o3d.utility.Vector3dVector(curv_colors)
    o3d.io.write_point_cloud(str(output_path / "curvature_k1.ply"), pcd_curv)

    # Surface type visualization
    surface_type_colors = {
        0: [0.8, 0.8, 0.2],   # flat = yellow
        1: [0.2, 0.8, 0.2],   # cylindrical = green
        2: [0.2, 0.2, 0.8],   # spherical = blue
        3: [0.8, 0.2, 0.8],   # saddle = magenta
        4: [0.8, 0.2, 0.2],   # edge = red
    }
    type_colors_arr = np.full((len(points), 3), 0.3)
    for t, color in surface_type_colors.items():
        mask = surface_labels == t
        type_colors_arr[mask] = color
    pcd_types = o3d.geometry.PointCloud()
    pcd_types.points = o3d.utility.Vector3dVector(points)
    pcd_types.colors = o3d.utility.Vector3dVector(type_colors_arr)
    o3d.io.write_point_cloud(str(output_path / "surface_types.ply"), pcd_types)

    np.save(str(output_path / "instance_labels.npy"), instance_labels)
    np.save(str(output_path / "k1.npy"), k1)
    np.save(str(output_path / "k2.npy"), k2)
    np.save(str(output_path / "surface_labels.npy"), surface_labels)

    summary = [{k: v for k, v in p.items() if k != "global_indices"} for p in primitives]
    with open(output_path / "ransac_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_path}/")
    print(f"  ransac_segmentation.ply  — colored by type")
    print(f"  instance_colors.ply      — colored by instance")
    print(f"  curvature_k1.ply         — k1 magnitude (blue=flat, red=curved)")
    print(f"  surface_types.ply        — surface classification (yellow=flat, green=cyl, blue=sphere, magenta=saddle)")
    print(f"  k1.npy, k2.npy           — raw curvature arrays")
    print(f"  ransac_summary.json      — primitive parameters")


if __name__ == "__main__":
    main()
