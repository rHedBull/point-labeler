#!/usr/bin/env python3
"""Build pipe-endpoint connectivity graph for industrial point clouds.

For each pipe segment, finds the two endpoints via PCA projection and checks
whether those endpoints are close to points from other process segments
(pipe, tank, equipment). Produces a clean P&ID-style connectivity graph.

Usage:
    python -m industrial_point_labeler.graph.builder \
        --points ground_truth.ply --labels gt_labels.npy \
        --metadata gt_metadata.json --output adjacency_graph.json \
        --threshold 0.15
"""

import argparse
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from industrial_point_labeler.io import load_ply_xyz


ENDPOINT_FRACTION = 0.10  # bottom/top 10% of axis projection


def load_metadata(path):
    """Return dict mapping segment id -> class label."""
    with open(path) as f:
        meta = json.load(f)
    return {seg["gt_id"]: seg["label"] for seg in meta["segments"]}


def compute_pipe_endpoints(points, fraction=ENDPOINT_FRACTION):
    """PCA on segment points, return two endpoint centroids."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    # 1st principal component via SVD
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]

    projections = centered @ axis
    pmin, pmax = projections.min(), projections.max()
    span = pmax - pmin
    if span < 1e-6:
        # Degenerate (blob-like) segment — return centroid as both endpoints
        return np.array([centroid, centroid])

    lo_thresh = pmin + span * fraction
    hi_thresh = pmax - span * fraction

    lo_mask = projections <= lo_thresh
    hi_mask = projections >= hi_thresh

    ep_lo = points[lo_mask].mean(axis=0) if lo_mask.any() else centroid
    ep_hi = points[hi_mask].mean(axis=0) if hi_mask.any() else centroid

    return np.array([ep_lo, ep_hi])


def build_graph(xyz, labels, label_map, threshold, process_classes):
    """Build connectivity graph from pipe endpoints."""
    # Identify process segments
    unique_ids = np.unique(labels)
    process_segments = {}
    for sid in unique_ids:
        sid = int(sid)
        if sid < 0:
            continue
        cls = label_map.get(sid)
        if cls not in process_classes:
            continue
        mask = labels == sid
        pts = xyz[mask]
        if len(pts) < 10:
            continue
        center = ((pts.min(axis=0) + pts.max(axis=0)) / 2).tolist()
        process_segments[sid] = {
            "id": sid,
            "label": cls,
            "n_points": int(mask.sum()),
            "center": center,
        }

    print(f"  {len(process_segments)} process segments "
          f"({'/'.join(sorted(process_classes))})")

    # Build KD-tree of all process-segment points, with segment ID per point
    all_pts = []
    all_seg_ids = []
    for sid, info in process_segments.items():
        mask = labels == sid
        pts = xyz[mask]
        all_pts.append(pts)
        all_seg_ids.append(np.full(len(pts), sid, dtype=np.int32))

    all_pts = np.vstack(all_pts)
    all_seg_ids = np.concatenate(all_seg_ids)
    tree = cKDTree(all_pts)
    print(f"  KD-tree: {len(all_pts)} process points")

    # Find pipe endpoints and query for connections
    pipe_ids = [sid for sid, info in process_segments.items()
                if info["label"] == "pipe"]
    print(f"  {len(pipe_ids)} pipe segments to process")

    edge_dict = {}  # (min_id, max_id) -> edge info

    for sid in pipe_ids:
        mask = labels == sid
        pts = xyz[mask]
        endpoints = compute_pipe_endpoints(pts)

        for ep in endpoints:
            # Query nearby points from other segments
            idxs = tree.query_ball_point(ep, threshold)
            if not idxs:
                continue

            # Group by segment id
            neighbor_sids = all_seg_ids[idxs]
            for other_sid in np.unique(neighbor_sids):
                other_sid = int(other_sid)
                if other_sid == sid:
                    continue

                # Compute stats for this connection
                other_mask = neighbor_sids == other_sid
                other_pts = all_pts[np.array(idxs)[other_mask]]
                dists = np.linalg.norm(other_pts - ep, axis=1)
                min_dist = float(dists.min())
                n_close = int((dists < threshold).sum())

                # Closest point on target segment to this endpoint
                closest_idx = np.argmin(dists)
                target_ep = other_pts[closest_idx]

                edge_key = (min(sid, other_sid), max(sid, other_sid))
                if edge_key not in edge_dict or min_dist < edge_dict[edge_key]["min_dist"]:
                    src_label = process_segments[sid]["label"]
                    tgt_label = process_segments[other_sid]["label"]
                    edge_dict[edge_key] = {
                        "source": sid,
                        "target": other_sid,
                        "min_dist": round(min_dist, 4),
                        "n_close_pairs": n_close,
                        "endpoint": [round(float(x), 4) for x in ep],
                        "endpoint_target": [round(float(x), 4) for x in target_ep],
                        "connection_type": f"{src_label}-{tgt_label}",
                    }

    # Build nodes list
    nodes = []
    for sid, info in sorted(process_segments.items()):
        nodes.append({
            "id": info["id"],
            "label": info["label"],
            "n_points": info["n_points"],
            "center": info["center"],
        })

    # Build sorted edge list
    edges = sorted(edge_dict.values(), key=lambda e: e["min_dist"])

    return {"nodes": nodes, "edges": edges, "method": "pipe_endpoint",
            "threshold": threshold}


def main():
    parser = argparse.ArgumentParser(
        description="Build pipe-endpoint connectivity graph")
    parser.add_argument("--points", required=True, help="Input PLY file")
    parser.add_argument("--labels", required=True, help="GT labels .npy")
    parser.add_argument("--metadata", required=True, help="GT metadata JSON")
    parser.add_argument("--output", required=True, help="Output graph JSON")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Max distance (m) for endpoint connections")
    parser.add_argument("--config", help="Classes YAML config file")
    args = parser.parse_args()

    # Load process classes from config
    from industrial_point_labeler.config import load_classes
    _, _, _, process_classes = load_classes(args.config)

    print("Loading point cloud...")
    xyz = load_ply_xyz(args.points)
    print(f"  {len(xyz)} points")

    print("Loading labels...")
    labels = np.load(args.labels)
    print(f"  {len(np.unique(labels[labels >= 0]))} segments")

    print("Loading metadata...")
    label_map = load_metadata(args.metadata)
    print(f"  {len(label_map)} segment labels")

    print(f"Building connectivity graph (threshold={args.threshold}m)...")
    graph = build_graph(xyz, labels, label_map, args.threshold, process_classes)

    print(f"\nResult: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # Summarize connection types
    type_counts = {}
    for e in graph["edges"]:
        ct = e["connection_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct}: {count}")

    with open(args.output, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
