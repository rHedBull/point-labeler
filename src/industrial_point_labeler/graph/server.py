#!/usr/bin/env python3
"""Connection graph review tool for industrial point cloud segmentation.

Review precomputed spatial adjacency edges between GT segments: accept, reject,
or skip each connection to build a functional connectivity graph.

Usage:
    python -m industrial_point_labeler.graph.server --points scan.ply --labels gt_labels.npy \
        --graph adjacency_graph.json [--resume] [--port 8768] \
        [--output-dir output]
"""

import argparse
import datetime
import json
import os
import sys
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

from industrial_point_labeler.io import load_ply, labels_fingerprint


def prepare_segments(xyz, rgb, labels, graph_nodes):
    """Build per-segment data for process-relevant nodes only."""
    np.random.seed(42)

    node_map = {n["id"]: n for n in graph_nodes}
    # Only build point clouds for nodes in the filtered graph
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


class GraphHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app_state=None, **kwargs):
        self.app = app_state
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if args and isinstance(args[0], str) and args[0].startswith("4"):
            super().log_message(format, *args)

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/data":
            self._serve_data()
        elif self.path == "/mesh.glb":
            self._serve_mesh()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            self._handle_save()
        elif self.path == "/export":
            self._handle_export()
        elif self.path == "/add-edge":
            self._handle_add_edge()
        else:
            self.send_error(404)

    def _serve_html(self):
        html_path = Path(__file__).parent / "viewer.html"
        if not html_path.exists():
            self.send_error(500, "viewer.html not found")
            return
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _serve_mesh(self):
        mesh_path = self.app.get("mesh_path")
        if not mesh_path or not mesh_path.exists():
            self.send_error(404, "No mesh file configured")
            return
        content = mesh_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _serve_data(self):
        # Stitch pre-cached static JSON with dynamic fields to avoid
        # re-serializing ~MBs of point data on every page load
        dynamic = json.dumps({
            "edges": self.app["edges"],
            "session": self.app["session"],
            "graph_method": self.app.get("graph_method", "proximity"),
            "has_mesh": self.app.get("mesh_path") is not None,
        })
        # Merge: strip closing } from static, strip opening { from dynamic
        body = (self.app["static_json"][:-1] + "," + dynamic[1:]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        session = json.loads(body)

        session["labels_fingerprint"] = self.app["labels_fingerprint"]
        session["last_saved"] = datetime.datetime.now().isoformat()

        self.app["session"] = session

        out_dir = self.app["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "graph_session.json", "w") as f:
            json.dump(session, f, indent=2)

        n_reviewed = len(session.get("reviewed", []))
        n_total = len(self.app["edges"])
        resp = json.dumps({"ok": True, "n_reviewed": n_reviewed, "n_total": n_total}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"  Saved: {n_reviewed}/{n_total} edges reviewed")

    def _handle_export(self):
        session = self.app["session"]
        reviewed = session.get("reviewed", [])

        # Build lookup from reviewed edges
        reviewed_map = {}
        for r in reviewed:
            reviewed_map[(r["source"], r["target"])] = r["action"]

        # Full edge list with verdicts
        export_edges = []
        for i, edge in enumerate(self.app["edges"]):
            key = (edge["source"], edge["target"])
            action = reviewed_map.get(key, "pending")
            entry = {
                "edge_id": i,
                "source": edge["source"],
                "target": edge["target"],
                "min_dist": edge["min_dist"],
                "n_close_pairs": edge["n_close_pairs"],
                "action": action,
            }
            if "endpoint" in edge:
                entry["endpoint"] = edge["endpoint"]
            if "connection_type" in edge:
                entry["connection_type"] = edge["connection_type"]
            export_edges.append(entry)

        n_accepted = sum(1 for e in export_edges if e["action"] == "accept")
        n_rejected = sum(1 for e in export_edges if e["action"] == "reject")
        n_pending = sum(1 for e in export_edges if e["action"] == "pending")

        output = {
            "edges": export_edges,
            "summary": {
                "total": len(export_edges),
                "accepted": n_accepted,
                "rejected": n_rejected,
                "pending": n_pending,
            },
            "exported": datetime.datetime.now().isoformat(),
            "labels_fingerprint": self.app["labels_fingerprint"],
        }

        out_dir = self.app["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "reviewed_edges.json", "w") as f:
            json.dump(output, f, indent=2)

        resp = json.dumps({"ok": True, "summary": output["summary"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"Exported: {n_accepted} accepted, {n_rejected} rejected, "
              f"{n_pending} pending → {out_dir / 'reviewed_edges.json'}")

    def _handle_add_edge(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        req = json.loads(body)
        source = int(req["source"])
        target = int(req["target"])

        xyz = self.app["xyz"]
        labels = self.app["labels"]

        src_mask = labels == source
        tgt_mask = labels == target
        src_pts = xyz[src_mask]
        tgt_pts = xyz[tgt_mask]

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            resp = json.dumps({"ok": False, "error": "Segment has no points"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)
            return

        # Compute pairwise distances to find closest point pair
        # Use chunked approach to avoid memory issues with large segments
        min_dist = float("inf")
        best_src_idx = 0
        chunk_size = 2000
        n_close = 0
        threshold = 0.1  # 10cm threshold for close pairs

        for i in range(0, len(src_pts), chunk_size):
            src_chunk = src_pts[i:i + chunk_size]
            # Broadcast: (chunk, 1, 3) - (1, tgt, 3)
            diffs = src_chunk[:, None, :] - tgt_pts[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            chunk_min_idx = np.unravel_index(dists.argmin(), dists.shape)
            chunk_min = dists[chunk_min_idx]
            if chunk_min < min_dist:
                min_dist = chunk_min
                best_src_idx = i + chunk_min_idx[0]
            n_close += int((dists < threshold).sum())

        endpoint = src_pts[best_src_idx].tolist()

        # Find closest point on target to the source endpoint
        dists_to_ep = np.linalg.norm(tgt_pts - src_pts[best_src_idx], axis=1)
        endpoint_target = tgt_pts[np.argmin(dists_to_ep)].tolist()

        edge = {
            "source": source,
            "target": target,
            "min_dist": float(min_dist),
            "n_close_pairs": n_close,
            "endpoint": endpoint,
            "endpoint_target": endpoint_target,
            "connection_type": "manual",
        }
        self.app["edges"].append(edge)

        resp = json.dumps({"ok": True, "edge": edge}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"  Added manual edge: {source} → {target} (dist={min_dist:.4f}m, {n_close} close pairs)")


def main():
    parser = argparse.ArgumentParser(description="Connection graph review tool")
    parser.add_argument("--points", required=True, help="Input PLY point cloud")
    parser.add_argument("--labels", required=True, help="GT labels .npy")
    parser.add_argument("--graph", required=True, help="Adjacency graph JSON")
    parser.add_argument("--resume", action="store_true", help="Resume from graph_session.json in output-dir")
    parser.add_argument("--mesh", help="GLB mesh file for split-view (optional)")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    print("Loading point cloud...")
    xyz, rgb = load_ply(args.points)
    print(f"  {len(xyz)} points")

    print("Loading labels...")
    labels = np.load(args.labels)
    current_fp = labels_fingerprint(labels)
    n_segs = len(np.unique(labels[labels >= 0]))
    print(f"  {n_segs} segments  (fingerprint: {current_fp})")

    print("Loading connectivity graph...")
    with open(args.graph) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    edges = graph["edges"]
    method = graph.get("method", "proximity")
    print(f"  {len(nodes)} nodes, {len(edges)} edges (method: {method})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create session
    session = {"reviewed": [], "current_edge_index": 0}
    session_path = output_dir / "graph_session.json"
    if args.resume and session_path.exists():
        with open(session_path) as f:
            session = json.load(f)
        n_reviewed = len(session.get("reviewed", []))
        print(f"  Resumed session: {n_reviewed} edges reviewed")

        saved_fp = session.get("labels_fingerprint")
        if saved_fp and saved_fp != current_fp:
            print(f"\n  ERROR: labels fingerprint mismatch!")
            print(f"    Session expects: {saved_fp}")
            print(f"    --labels file:   {current_fp}")
            print(f"  Refusing to load — would corrupt review state.")
            sys.exit(1)

    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"Warning: mesh file not found: {mesh_path}")
            mesh_path = None
        else:
            print(f"  Mesh file: {mesh_path}")

    print("Preparing segments...")
    segments, context, scene_center, scene_extent = \
        prepare_segments(xyz, rgb, labels, nodes)
    print(f"  {len(segments)} segments prepared for browser")

    # Pre-serialize the heavy static data once
    print("Serializing data for browser...")
    static_json = json.dumps({
        "segments": segments,
        "context": context,
        "scene_center": scene_center,
        "scene_extent": scene_extent,
    })
    print(f"  {len(static_json) // 1024}KB payload cached")

    app_state = {
        "segments": segments,
        "context": context,
        "scene_center": scene_center,
        "scene_extent": scene_extent,
        "edges": edges,
        "session": session,
        "output_dir": output_dir,
        "labels_fingerprint": current_fp,
        "graph_method": method,
        "mesh_path": mesh_path,
        "xyz": xyz,
        "labels": labels,
        "static_json": static_json,
    }

    handler = partial(GraphHandler, app_state=app_state)
    server = HTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\nGraph Review Tool running at {url}")
    print("Press Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
