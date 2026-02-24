#!/usr/bin/env python3
"""Equipment description tool for industrial point cloud segmentation.

Browse equipment segments with point cloud + mesh views and assign
descriptive labels (valve, flange, pump, etc.).

Usage:
    python -m industrial_point_labeler.equipment.server --points data/input/ground_truth.ply \
        --labels data/gt/merged_pipes2/gt_labels.npy \
        --summary data/gt/merged_pipes2/summary.json \
        --graph data/gt/merged_pipes2/adjacency_graph.json \
        --mesh data/input/source.glb [--port 8767]
"""

import argparse
import datetime
import json
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

from industrial_point_labeler.io import load_ply


def prepare_equipment_segments(xyz, rgb, labels, summary, graph):
    """Build per-segment data for equipment segments from the adjacency graph."""
    np.random.seed(42)

    # Use graph labels (latest reviewed result) instead of summary class_label
    graph_nodes = {n["id"]: n for n in graph["nodes"]}
    equip_ids = {n["id"] for n in graph["nodes"] if n["label"] == "equipment"}

    summary_map = {}
    for p in summary:
        summary_map[p["id"]] = p

    segments = []
    for sid in sorted(equip_ids):
        mask = labels == sid
        count = int(mask.sum())
        if count < 5:
            continue

        pts = xyz[mask]
        cols = rgb[mask]

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

    segments.sort(key=lambda s: s["id"])
    return segments


class EquipHandler(SimpleHTTPRequestHandler):
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

    def _serve_data(self):
        data = {
            "segments": self.app["segments"],
            "descriptions": self.app["descriptions"],
            "has_mesh": self.app.get("mesh_path") is not None,
        }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

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

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        descriptions = json.loads(body)

        self.app["descriptions"] = descriptions

        # Save dedicated descriptions file
        out_path = self.app["descriptions_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(descriptions, f, indent=2)

        # Update gt_metadata.json — add/update "description" field on segments
        meta_path = self.app["metadata_path"]
        with open(meta_path) as f:
            metadata = json.load(f)
        for seg in metadata["segments"]:
            key = str(seg["gt_id"])
            if key in descriptions:
                seg["description"] = descriptions[key]["description"]
            else:
                seg.pop("description", None)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        n_described = len(descriptions)
        resp = json.dumps({"ok": True, "n_described": n_described}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"  Saved: {n_described} equipment descriptions")


def main():
    parser = argparse.ArgumentParser(description="Equipment description tool")
    parser.add_argument("--points", required=True, help="Input PLY point cloud")
    parser.add_argument("--labels", required=True, help="GT instance labels .npy (re-indexed)")
    parser.add_argument("--summary", required=True, help="RANSAC summary JSON")
    parser.add_argument("--graph", required=True, help="Adjacency graph JSON")
    parser.add_argument("--mesh", help="GLB mesh file (optional)")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    print("Loading point cloud...")
    xyz, rgb = load_ply(args.points)
    print(f"  {len(xyz)} points")

    print("Loading instance labels...")
    labels = np.load(args.labels)
    print(f"  {len(np.unique(labels[labels >= 0]))} instances")

    print("Loading summary...")
    with open(args.summary) as f:
        summary = json.load(f)

    print("Loading adjacency graph...")
    with open(args.graph) as f:
        graph = json.load(f)
    print(f"  {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    print("Preparing equipment segments...")
    segments = prepare_equipment_segments(xyz, rgb, labels, summary, graph)
    print(f"  {len(segments)} equipment segments")

    # Descriptions file path — next to summary
    desc_path = Path(args.summary).parent / "equipment_descriptions.json"
    descriptions = {}
    if desc_path.exists():
        with open(desc_path) as f:
            descriptions = json.load(f)
        print(f"  Loaded {len(descriptions)} existing descriptions")

    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"  Warning: mesh file not found: {mesh_path}")
            mesh_path = None
        else:
            print(f"  Mesh file: {mesh_path}")

    metadata_path = Path(args.summary).parent / "gt_metadata.json"

    app_state = {
        "segments": segments,
        "descriptions": descriptions,
        "descriptions_path": desc_path,
        "metadata_path": metadata_path,
        "mesh_path": mesh_path,
    }

    handler = partial(EquipHandler, app_state=app_state)
    server = HTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\nEquipment Description Tool running at {url}")
    print("Press Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
