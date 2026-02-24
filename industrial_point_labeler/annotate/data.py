"""Segment preparation, split logic, and export functions for the annotation tool."""

import datetime
import json
from pathlib import Path

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

        # Attach RANSAC metadata if available
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

        # Subsample for browser
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

    # Context cloud
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

    # Context cloud
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


# --- Split logic ---


def get_split_data(xyz, rgb, point_indices_map, segment_ids):
    """Return full-resolution point data for the given segments (no subsampling).

    Returns dict with x, y, z, r, g, b, segment_ids (per-point) lists.
    """
    all_pts = []
    all_cols = []
    all_seg_ids = []

    for sid in segment_ids:
        if sid not in point_indices_map:
            continue
        indices = point_indices_map[sid]
        all_pts.append(xyz[indices])
        all_cols.append(rgb[indices])
        all_seg_ids.append(np.full(len(indices), sid, dtype=np.int32))

    if not all_pts:
        return {"x": [], "y": [], "z": [], "r": [], "g": [], "b": [], "segment_ids": []}

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    seg_ids = np.concatenate(all_seg_ids)

    return {
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "r": cols[:, 0].tolist(),
        "g": cols[:, 1].tolist(),
        "b": cols[:, 2].tolist(),
        "segment_ids": seg_ids.tolist(),
    }


def apply_split(labels, point_indices_map, segments, split_result, next_segment_id):
    """Apply a split operation: create new segments, remove originals.

    Mutates labels and point_indices_map in place.
    Returns (new_segment_entries, next_segment_id).
    """
    original_ids = split_result["original_segment_ids"]
    groups = split_result["groups"]

    # Gather all original point indices in order
    all_original_indices = []
    for sid in original_ids:
        if sid in point_indices_map:
            all_original_indices.append(point_indices_map[sid])
    if not all_original_indices:
        return [], next_segment_id
    all_original_indices = np.concatenate(all_original_indices)

    # Track which points are assigned to a group
    assigned = np.zeros(len(all_original_indices), dtype=bool)
    new_segs = []
    cur_id = next_segment_id

    for g in groups:
        pidx = np.array(g["point_indices"], dtype=np.int64)
        real_indices = all_original_indices[pidx]
        labels[real_indices] = cur_id
        point_indices_map[cur_id] = real_indices
        assigned[pidx] = True
        new_segs.append({"id": cur_id, "group": g["group"], "n_points": len(real_indices)})
        cur_id += 1

    # Remainder: unassigned points
    remainder_pidx = np.where(~assigned)[0]
    if len(remainder_pidx) > 0:
        real_indices = all_original_indices[remainder_pidx]
        labels[real_indices] = cur_id
        point_indices_map[cur_id] = real_indices
        new_segs.append({"id": cur_id, "group": "_remainder", "n_points": len(real_indices)})
        cur_id += 1

    # Remove originals
    for sid in original_ids:
        point_indices_map.pop(sid, None)
        for i, s in enumerate(segments):
            if s["id"] == sid:
                segments.pop(i)
                break

    return new_segs, cur_id


# --- Export functions with versioning ---


def _next_version(output_dir, prefix, input_fingerprints):
    """Determine next version number. Returns (version, needs_new_version).

    If the latest version's metadata has matching input fingerprints,
    returns (current_version, False) — overwrite in place.
    Otherwise returns (current_version + 1, True) — create new version.
    """
    output_dir = Path(output_dir)
    # Find latest version
    existing = sorted(output_dir.glob(f"{prefix}_metadata.v*.json"))
    if not existing:
        return 1, True

    latest = existing[-1]
    # Extract version number
    version = int(latest.stem.split(".v")[1])
    with open(latest) as f:
        meta = json.load(f)

    if meta.get("input_fingerprints") == input_fingerprints:
        return version, False  # same inputs, overwrite
    return version + 1, True


def export_patches(session, labels, point_indices_map, output_dir, class_map, input_fingerprints):
    """Save patch-level outputs from stage 2.

    Writes patch_ids.v{N}.npy, patch_classes.v{N}.npy, patch_metadata.v{N}.json.
    New version only if input fingerprints differ from last version.
    Returns metadata dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version, _ = _next_version(output_dir, "patch", input_fingerprints)

    N = len(labels)
    patch_ids = np.full(N, -1, dtype=np.int32)
    patch_classes = np.full(N, -1, dtype=np.int32)

    confirmed = session.get("confirmed", [])
    for gt_seg in confirmed:
        gt_id = gt_seg["gt_id"]
        class_id = class_map.get(gt_seg["label"], -1)
        for src_id in gt_seg["source_segment_ids"]:
            if src_id in point_indices_map:
                idx = point_indices_map[src_id]
                patch_ids[idx] = gt_id
                patch_classes[idx] = class_id

    np.save(output_dir / f"patch_ids.v{version}.npy", patch_ids)
    np.save(output_dir / f"patch_classes.v{version}.npy", patch_classes)

    meta = {
        "version": version,
        "input_fingerprints": input_fingerprints,
        "created": datetime.datetime.now().isoformat(),
        "n_points": N,
        "n_patches": len(confirmed),
        "n_labeled_points": int((patch_ids >= 0).sum()),
        "class_map": class_map,
        "patches": [
            {
                "gt_id": s["gt_id"],
                "label": s["label"],
                "class_id": class_map.get(s["label"], -1),
                "source_segment_ids": s["source_segment_ids"],
                "n_points": int(sum(
                    len(point_indices_map.get(sid, []))
                    for sid in s["source_segment_ids"]
                )),
            }
            for s in confirmed
        ],
    }
    with open(output_dir / f"patch_metadata.v{version}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def export_instances(session, labels, point_indices_map, output_dir, class_map, input_fingerprints):
    """Save instance-level outputs from stage 3.

    Writes instance_ids.v{N}.npy, instance_classes.v{N}.npy, instance_metadata.v{N}.json.
    New version only if input fingerprints differ from last version.
    Returns metadata dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version, _ = _next_version(output_dir, "instance", input_fingerprints)

    N = len(labels)
    instance_ids = np.full(N, -1, dtype=np.int32)
    instance_classes = np.full(N, -1, dtype=np.int32)

    instances = session.get("instances", [])
    for inst in instances:
        inst_id = inst["instance_id"]
        class_id = class_map.get(inst["label"], -1)
        for patch_id in inst["patch_ids"]:
            if patch_id in point_indices_map:
                idx = point_indices_map[patch_id]
                instance_ids[idx] = inst_id
                instance_classes[idx] = class_id

    np.save(output_dir / f"instance_ids.v{version}.npy", instance_ids)
    np.save(output_dir / f"instance_classes.v{version}.npy", instance_classes)

    meta = {
        "version": version,
        "input_fingerprints": input_fingerprints,
        "created": datetime.datetime.now().isoformat(),
        "n_points": N,
        "n_instances": len(instances),
        "n_labeled_points": int((instance_ids >= 0).sum()),
        "class_map": class_map,
        "instances": [
            {
                "instance_id": inst["instance_id"],
                "label": inst["label"],
                "class_id": class_map.get(inst["label"], -1),
                "patch_ids": inst["patch_ids"],
                "n_points": int(sum(
                    len(point_indices_map.get(pid, []))
                    for pid in inst["patch_ids"]
                )),
            }
            for inst in instances
        ],
    }
    with open(output_dir / f"instance_metadata.v{version}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def export_graph(edges, nodes, session, output_dir, input_fingerprints):
    """Save connectivity graph from stage 4.

    Writes connectivity_graph.v{N}.json containing only accepted edges
    and all nodes (including isolated ones with no connections).
    Review progress stays in graph_session.json.
    New version only if input fingerprints differ from last version.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Graph has no separate _metadata file — version info is embedded in the JSON
    existing = sorted(output_dir.glob("connectivity_graph.v*.json"))
    if not existing:
        version = 1
    else:
        latest = existing[-1]
        version = int(latest.stem.split(".v")[1])
        with open(latest) as f:
            meta = json.load(f).get("metadata", {})
        if meta.get("input_fingerprints") != input_fingerprints:
            version += 1  # different inputs → new version

    reviewed = session.get("graph_reviewed", [])
    reviewed_map = {(r["source"], r["target"]): r["action"] for r in reviewed}

    # Only accepted edges — no action field
    accepted_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"])
        action = reviewed_map.get(key, "pending")
        if action != "accept":
            continue
        entry = {
            "source": edge["source"],
            "target": edge["target"],
            "min_dist": edge["min_dist"],
            "n_close_pairs": edge["n_close_pairs"],
        }
        if "endpoint" in edge:
            entry["endpoint"] = edge["endpoint"]
        if "endpoint_target" in edge:
            entry["endpoint_target"] = edge["endpoint_target"]
        if "connection_type" in edge:
            entry["connection_type"] = edge["connection_type"]
        accepted_edges.append(entry)

    output = {
        "nodes": nodes,
        "edges": accepted_edges,
        "metadata": {
            "version": version,
            "input_fingerprints": input_fingerprints,
            "created": datetime.datetime.now().isoformat(),
            "n_nodes": len(nodes),
            "n_edges": len(accepted_edges),
        },
    }

    with open(output_dir / f"connectivity_graph.v{version}.json", "w") as f:
        json.dump(output, f, indent=2)

    return output["metadata"]
