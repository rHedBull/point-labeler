#!/usr/bin/env python3
"""Ground truth annotation tool for industrial point cloud segmentation.

Merge-based workflow: start from RANSAC pre-segments, merge groups into
labeled GT segments (pipe, tank, equipment, structural). Serves a Three.js
frontend for interactive annotation.

Usage:
    python -m industrial_point_labeler.labeler.server --points scan.ply --labels instance_labels.npy \
        [--summary ransac_summary.json] [--resume gt_session.json] \
        [--port 8766] [--output-dir gt_output] [--config classes.yaml]
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

from industrial_point_labeler.config import load_classes
from industrial_point_labeler.io import labels_fingerprint, load_ply


def prepare_segments(xyz, rgb, labels, summary=None):
    """Build per-segment data for the browser, keeping full indices server-side."""
    np.random.seed(42)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids >= 0]

    summary_map = {}
    if summary:
        for p in summary:
            summary_map[p["id"]] = p

    segments = []
    point_indices_map = {}  # seg_id -> array of point indices (kept server-side)

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


def export_ground_truth(session, labels, point_indices_map, output_dir, class_map):
    """Resolve confirmed GT segments to per-point label arrays."""
    N = len(labels)
    gt_labels = np.full(N, -1, dtype=np.int32)
    gt_classes = np.full(N, -1, dtype=np.int32)

    confirmed = session.get("confirmed", [])
    for gt_seg in confirmed:
        gt_id = gt_seg["gt_id"]
        class_id = class_map.get(gt_seg["label"], -1)
        for src_id in gt_seg["source_segment_ids"]:
            if src_id in point_indices_map:
                idx = point_indices_map[src_id]
                gt_labels[idx] = gt_id
                gt_classes[idx] = class_id

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gt_labels.npy", gt_labels)
    np.save(output_dir / "gt_classes.npy", gt_classes)

    # Metadata
    meta = {
        "n_points": N,
        "n_gt_segments": len(confirmed),
        "n_labeled_points": int((gt_labels >= 0).sum()),
        "class_map": class_map,
        "segments": [
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
    with open(output_dir / "gt_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


class GTHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the GT annotation tool."""

    def __init__(self, *args, app_state=None, **kwargs):
        self.app = app_state
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Quiet logging — only errors
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
        data = {
            "segments": self.app["segments"],
            "context": self.app["context"],
            "scene_center": self.app["scene_center"],
            "scene_extent": self.app["scene_extent"],
            "session": self.app["session"],
            "has_mesh": self.app.get("mesh_path") is not None,
            "class_map": self.app["class_map"],
            "class_colors": {k: v for k, v in self.app["class_colors"].items()},
            "class_keys": self.app["class_keys"],
        }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        session = json.loads(body)

        # Stamp integrity metadata from server state (not client data)
        session["labels_fingerprint"] = self.app["labels_fingerprint"]
        session["labels_source_path"] = self.app["labels_source_path"]
        session["points_source_path"] = self.app["points_source_path"]
        session["last_saved"] = datetime.datetime.now().isoformat()
        session["n_points"] = int(len(self.app["labels"]))
        session["n_source_segments"] = len(self.app["point_indices_map"])

        self.app["session"] = session

        out_dir = self.app["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "gt_session.json", "w") as f:
            json.dump(session, f, indent=2)

        # Auto-export per-point labels on every save
        meta = export_ground_truth(
            session,
            self.app["labels"],
            self.app["point_indices_map"],
            out_dir,
            self.app["class_map"],
        )

        resp = json.dumps({"ok": True, "meta": meta}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"  Saved + exported: {meta['n_gt_segments']} segments, "
              f"{meta['n_labeled_points']} labeled points")

    def _handle_export(self):
        meta = export_ground_truth(
            self.app["session"],
            self.app["labels"],
            self.app["point_indices_map"],
            self.app["output_dir"],
            self.app["class_map"],
        )
        resp = json.dumps({"ok": True, "meta": meta}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)
        print(f"Exported GT: {meta['n_gt_segments']} segments, "
              f"{meta['n_labeled_points']} labeled points → {self.app['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Ground truth annotation tool")
    parser.add_argument("--points", required=True, help="Input PLY point cloud")
    parser.add_argument("--labels", required=True, help="Instance labels .npy")
    parser.add_argument("--summary", help="RANSAC summary JSON (optional)")
    parser.add_argument("--resume", help="Resume from gt_session.json")
    parser.add_argument("--mesh", help="GLB mesh file for split-view (optional)")
    parser.add_argument("--config", help="Classes YAML config (optional)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", default="gt_output")
    args = parser.parse_args()

    class_map, class_colors, class_keys, _ = load_classes(args.config)

    print("Loading point cloud...")
    xyz, rgb = load_ply(args.points)
    print(f"  {len(xyz)} points")

    print("Loading instance labels...")
    labels = np.load(args.labels)
    n_instances = len(np.unique(labels[labels >= 0]))
    current_fp = labels_fingerprint(labels)
    print(f"  {n_instances} instances  (fingerprint: {current_fp})")

    summary = None
    if args.summary:
        with open(args.summary) as f:
            summary = json.load(f)
        print(f"  {len(summary)} RANSAC primitives loaded")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or create session ---
    session = {"confirmed": [], "hidden_segments": [], "mode": "unconfirmed", "next_gt_id": 0}
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            session = json.load(f)
        print(f"  Resumed session: {len(session.get('confirmed', []))} confirmed segments")

        # --- Fingerprint validation ---
        saved_fp = session.get("labels_fingerprint")
        if saved_fp and saved_fp != current_fp:
            print(f"\n  WARNING: labels fingerprint mismatch!")
            print(f"    Session expects: {saved_fp}")
            print(f"    --labels file:   {current_fp}")

            # Try to find the matching backup
            recovered = False
            search_dirs = []
            resume_dir = Path(args.resume).resolve().parent
            if resume_dir not in search_dirs:
                search_dirs.append(resume_dir)
            if output_dir.resolve() not in search_dirs:
                search_dirs.append(output_dir.resolve())

            for search_dir in search_dirs:
                backup = search_dir / "source_instance_labels.npy"
                if backup.exists():
                    backup_labels = np.load(backup)
                    backup_fp = labels_fingerprint(backup_labels)
                    if backup_fp == saved_fp:
                        labels = backup_labels
                        current_fp = backup_fp
                        n_instances = len(np.unique(labels[labels >= 0]))
                        print(f"  RECOVERED: using backup {backup}")
                        print(f"    ({n_instances} instances, fingerprint: {current_fp})")
                        recovered = True
                        break

            if not recovered:
                print(f"\n  ERROR: Cannot find labels matching session fingerprint {saved_fp}")
                print(f"  The --labels file has changed (e.g. re-ran segmentation) and no")
                print(f"  matching source_instance_labels.npy backup was found.")
                print(f"  Refusing to load — this would silently corrupt your GT annotations.")
                sys.exit(1)
        elif not saved_fp:
            print(f"  Note: legacy session without fingerprint — will stamp on next save")

    # Stamp creation metadata (setdefault so it's never overwritten on re-save)
    session.setdefault("created", datetime.datetime.now().isoformat())
    session.setdefault("labels_source_path", str(Path(args.labels).resolve()))
    session.setdefault("points_source_path", str(Path(args.points).resolve()))

    print("Preparing segments...")
    segments, context, scene_center, scene_extent, point_indices_map = \
        prepare_segments(xyz, rgb, labels, summary)
    print(f"  {len(segments)} segments prepared for browser")

    # --- Validate source_segment_ids exist in current labels ---
    missing_refs = []
    for gt_seg in session.get("confirmed", []):
        for src_id in gt_seg.get("source_segment_ids", []):
            if src_id not in point_indices_map:
                missing_refs.append((gt_seg["gt_id"], gt_seg.get("label", "?"), src_id))
    if missing_refs:
        print(f"\n  ERROR: {len(missing_refs)} source_segment_id references not found in labels!")
        for gt_id, label, src_id in missing_refs[:10]:
            print(f"    GT segment {gt_id} ({label}): source segment {src_id} missing")
        if len(missing_refs) > 10:
            print(f"    ... and {len(missing_refs) - 10} more")
        print(f"  The labels file does not match this session. Aborting.")
        sys.exit(1)
    elif session.get("confirmed"):
        print(f"  All source_segment_ids validated OK")

    # Preserve source instance_labels alongside session so GT can always be reconstructed
    src_labels_backup = output_dir / "source_instance_labels.npy"
    if not src_labels_backup.exists():
        np.save(src_labels_backup, labels)
        print(f"  Backed up instance_labels → {src_labels_backup}")
    else:
        print(f"  Source labels backup exists: {src_labels_backup}")

    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"Warning: mesh file not found: {mesh_path}")
            mesh_path = None
        else:
            print(f"  Mesh file: {mesh_path}")

    app_state = {
        "segments": segments,
        "context": context,
        "scene_center": scene_center,
        "scene_extent": scene_extent,
        "session": session,
        "labels": labels,
        "point_indices_map": point_indices_map,
        "output_dir": output_dir,
        "mesh_path": mesh_path,
        "labels_fingerprint": current_fp,
        "labels_source_path": str(Path(args.labels).resolve()),
        "points_source_path": str(Path(args.points).resolve()),
        "class_map": class_map,
        "class_colors": class_colors,
        "class_keys": class_keys,
    }

    handler = partial(GTHandler, app_state=app_state)
    server = HTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\nGT Annotation Tool running at {url}")
    print("Press Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
