"""Segment preparation, split logic, and export functions for the annotation tool."""

import numpy as np


def prepare_segments(xyz, rgb, labels, summary=None):
    """Build per-segment data for the browser, keeping full indices server-side.

    Returns (segments, context, scene_center, scene_extent, point_indices_map).
    """
    np.random.seed(42)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids >= 0]

    summary_map = {}
    if summary:
        for p in summary:
            summary_map[p["id"]] = p

    segments = []
    point_indices_map = {}

    for sid in unique_ids:
        sid = int(sid)
        mask = labels == sid
        count = int(mask.sum())
        if count < 5:
            continue

        pts = xyz[mask]
        cols = rgb[mask]
        indices = np.where(mask)[0]
        point_indices_map[sid] = indices

        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()
        center = ((pts.min(axis=0) + pts.max(axis=0)) / 2).tolist()
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

        seg = {
            "id": sid,
            "n_points": count,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "center": center,
            "extent": extent,
        }

        meta = summary_map.get(sid, {})
        seg["type"] = meta.get("type", "unknown")
        if "class_label" in meta:
            seg["class_label"] = meta["class_label"]
        if "radius" in meta:
            seg["radius"] = round(meta["radius"], 4)
        if "length" in meta:
            seg["length"] = round(meta["length"], 3)
        if "axis" in meta:
            seg["axis"] = meta["axis"]
        if "normal" in meta:
            seg["normal"] = meta["normal"]

        if count > 2000:
            idx = np.random.choice(count, 2000, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        seg["x"] = pts[:, 0].tolist()
        seg["y"] = pts[:, 1].tolist()
        seg["z"] = pts[:, 2].tolist()
        seg["r"] = cols[:, 0].tolist()
        seg["g"] = cols[:, 1].tolist()
        seg["b"] = cols[:, 2].tolist()
        segments.append(seg)

    segments.sort(key=lambda s: -s["n_points"])

    max_ctx = 50000
    if len(xyz) > max_ctx:
        idx = np.random.choice(len(xyz), max_ctx, replace=False)
        idx.sort()
        ctx_xyz = xyz[idx]
        ctx_rgb = rgb[idx]
    else:
        ctx_xyz = xyz
        ctx_rgb = rgb

    context = {
        "x": ctx_xyz[:, 0].tolist(),
        "y": ctx_xyz[:, 1].tolist(),
        "z": ctx_xyz[:, 2].tolist(),
        "r": ctx_rgb[:, 0].tolist(),
        "g": ctx_rgb[:, 1].tolist(),
        "b": ctx_rgb[:, 2].tolist(),
    }

    scene_center = ((xyz.min(axis=0) + xyz.max(axis=0)) / 2).tolist()
    scene_extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))

    return segments, context, scene_center, scene_extent, point_indices_map


def prepare_graph_segments(xyz, rgb, labels, graph_nodes):
    """Build per-segment data for process-relevant graph nodes only.

    Returns (segments, context, scene_center, scene_extent).
    """
    np.random.seed(42)

    node_map = {n["id"]: n for n in graph_nodes}
    node_ids = set(node_map.keys())

    segments = []
    for sid in sorted(node_ids):
        mask = labels == sid
        count = int(mask.sum())
        if count < 5:
            continue

        pts = xyz[mask]
        cols = rgb[mask]

        node = node_map[sid]
        center = ((pts.min(axis=0) + pts.max(axis=0)) / 2).tolist()
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

        seg = {
            "id": sid,
            "n_points": count,
            "label": node.get("label", "unknown"),
            "center": center,
            "extent": extent,
        }

        if count > 2000:
            idx = np.random.choice(count, 2000, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        seg["x"] = pts[:, 0].tolist()
        seg["y"] = pts[:, 1].tolist()
        seg["z"] = pts[:, 2].tolist()
        seg["r"] = cols[:, 0].tolist()
        seg["g"] = cols[:, 1].tolist()
        seg["b"] = cols[:, 2].tolist()
        segments.append(seg)

    max_ctx = 50000
    if len(xyz) > max_ctx:
        idx = np.random.choice(len(xyz), max_ctx, replace=False)
        idx.sort()
        ctx_xyz = xyz[idx]
        ctx_rgb = rgb[idx]
    else:
        ctx_xyz = xyz
        ctx_rgb = rgb

    context = {
        "x": ctx_xyz[:, 0].tolist(),
        "y": ctx_xyz[:, 1].tolist(),
        "z": ctx_xyz[:, 2].tolist(),
        "r": ctx_rgb[:, 0].tolist(),
        "g": ctx_rgb[:, 1].tolist(),
        "b": ctx_rgb[:, 2].tolist(),
    }

    scene_center = ((xyz.min(axis=0) + xyz.max(axis=0)) / 2).tolist()
    scene_extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))

    return segments, context, scene_center, scene_extent
