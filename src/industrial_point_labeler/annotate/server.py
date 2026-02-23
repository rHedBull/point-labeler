"""Unified HTTP server for the annotation tool (stages 2a, 2b, 2c)."""

import argparse
import datetime
import json
import sys
import urllib.parse
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

from industrial_point_labeler.annotate.data import (
    apply_split,
    export_graph,
    export_instances,
    export_patches,
    get_split_data,
    prepare_graph_segments,
    prepare_segments,
)
from industrial_point_labeler.config import load_classes
from industrial_point_labeler.io import labels_fingerprint, load_ply


class AnnotateHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the unified annotation tool."""

    def __init__(self, *args, app_state=None, **kwargs):
        self.app = app_state
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if args and isinstance(args[0], str) and args[0].startswith("4"):
            super().log_message(format, *args)

    @property
    def stage(self):
        return self.app["session"].get("stage", "label")

    # --- Routing ---

    def do_GET(self):
        if self.path == "/":
            self._serve_viewer()
        elif self.path == "/data":
            self._serve_data()
        elif self.path == "/mesh.glb":
            self._serve_mesh()
        elif self.path == "/split" or self.path.startswith("/split?"):
            if not self._require_stage("label"):
                return
            self._serve_html("splitter.html")
        elif self.path.startswith("/split-data"):
            if not self._require_stage("label"):
                return
            self._serve_split_data()
        elif self.path == "/graph-data":
            if not self._require_stage("graph"):
                return
            self._serve_graph_data()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            self._handle_save()
        elif self.path == "/apply-split":
            if not self._require_stage("label"):
                return
            self._handle_apply_split()
        elif self.path == "/finalize-patches":
            if not self._require_stage("label"):
                return
            self._handle_finalize_patches()
        elif self.path == "/finalize-instances":
            if not self._require_stage("instance"):
                return
            self._handle_finalize_instances()
        elif self.path == "/add-edge":
            if not self._require_stage("graph"):
                return
            self._handle_add_edge()
        elif self.path == "/review-edge":
            if not self._require_stage("graph"):
                return
            self._handle_review_edge()
        elif self.path == "/export-graph":
            if not self._require_stage("graph"):
                return
            self._handle_export_graph()
        else:
            self.send_error(404)

    # --- Stage enforcement ---

    def _require_stage(self, required):
        if self.stage != required:
            self._send_json(409, {"error": f"Cannot use this route in '{self.stage}' stage"})
            return False
        return True

    # --- Shared utilities ---

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body)

    def _serve_html(self, filename):
        html_path = Path(__file__).parent / filename
        if not html_path.exists():
            self.send_error(500, f"{filename} not found")
            return
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _serve_viewer(self):
        if self.stage == "graph":
            self._serve_html("graph_viewer.html")
        else:
            self._serve_html("viewer.html")

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

    # --- Data routes ---

    def _serve_data(self):
        stage = self.stage
        if stage in ("label", "instance"):
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
            self._send_json(200, data)
        elif stage == "graph":
            self._serve_graph_data()

    def _handle_save(self):
        session = self._read_json()

        # Preserve server-side integrity fields
        session["labels_fingerprint"] = self.app["labels_fingerprint"]
        session["labels_source_path"] = self.app.get("labels_source_path", "")
        session["points_source_path"] = self.app.get("points_source_path", "")
        session["last_saved"] = datetime.datetime.now().isoformat()
        session["n_points"] = int(len(self.app["labels"]))
        session["stage"] = self.stage  # preserve current stage

        self.app["session"] = session

        out_dir = self.app["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "gt_session.json", "w") as f:
            json.dump(session, f, indent=2)

        self._send_json(200, {"ok": True})
        print(f"  Saved session (stage={self.stage})")

    # --- Split routes ---

    def _serve_split_data(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        seg_ids_str = params.get("segments", [""])[0]
        if not seg_ids_str:
            self._send_json(400, {"error": "Missing segments parameter"})
            return

        segment_ids = [int(s) for s in seg_ids_str.split(",")]
        data = get_split_data(
            self.app["xyz"], self.app["rgb"],
            self.app["point_indices_map"], segment_ids,
        )
        self._send_json(200, data)

    def _handle_apply_split(self):
        split_result = self._read_json()
        new_segs, next_id = apply_split(
            self.app["labels"],
            self.app["point_indices_map"],
            self.app["segments"],
            split_result,
            self.app.get("next_segment_id", 1000),
        )
        self.app["next_segment_id"] = next_id

        # Record in session
        self.app["session"].setdefault("splits", []).append({
            "original_segment_ids": split_result["original_segment_ids"],
            "new_segments": new_segs,
        })

        # Re-prepare segment data for the new IDs
        xyz, rgb, labels = self.app["xyz"], self.app["rgb"], self.app["labels"]
        segments, context, sc, se, pim = prepare_segments(
            xyz, rgb, labels, self.app.get("summary"),
        )
        self.app["segments"] = segments
        self.app["context"] = context
        self.app["scene_center"] = sc
        self.app["scene_extent"] = se
        self.app["point_indices_map"] = pim

        self._send_json(200, {"ok": True, "new_segments": new_segs})
        print(f"  Applied split: {len(new_segs)} new segments")

    # --- Finalize routes ---

    def _handle_finalize_patches(self):
        meta = export_patches(
            self.app["session"],
            self.app["labels"],
            self.app["point_indices_map"],
            self.app["output_dir"],
            self.app["class_map"],
        )

        # Advance stage
        self.app["session"]["stage"] = "instance"
        self.app["session"]["patches_finalized"] = True

        # Save session
        out_dir = self.app["output_dir"]
        with open(out_dir / "gt_session.json", "w") as f:
            json.dump(self.app["session"], f, indent=2)

        self._send_json(200, {"ok": True, "meta": meta})
        print(f"  Finalized patches: {meta['n_patches']} patches, "
              f"{meta['n_labeled_points']} labeled points")

    def _handle_finalize_instances(self):
        meta = export_instances(
            self.app["session"],
            self.app["labels"],
            self.app["point_indices_map"],
            self.app["output_dir"],
            self.app["class_map"],
        )

        # Auto-run graph builder
        print("  Building connectivity graph...")
        from industrial_point_labeler.graph.builder import build_graph

        output_dir = self.app["output_dir"]
        instance_labels = np.load(output_dir / "instance_ids.npy")
        instance_meta_path = output_dir / "instance_metadata.json"
        with open(instance_meta_path) as f:
            instance_meta = json.load(f)
        label_map = {inst["instance_id"]: inst["label"] for inst in instance_meta["instances"]}

        graph = build_graph(
            self.app["xyz"], instance_labels, label_map,
            threshold=0.15,
            process_classes=self.app.get("process_classes", {"pipe", "tank", "equipment"}),
        )
        with open(output_dir / "adjacency_graph.json", "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

        # Prepare graph data
        graph_segs, graph_ctx, gc, ge = prepare_graph_segments(
            self.app["xyz"], self.app["rgb"], instance_labels, graph["nodes"],
        )
        self.app["graph_segments"] = graph_segs
        self.app["graph_context"] = graph_ctx
        self.app["graph_center"] = gc
        self.app["graph_extent"] = ge
        self.app["edges"] = graph["edges"]
        self.app["graph_labels"] = instance_labels

        # Pre-cache static graph JSON
        self.app["graph_static_json"] = json.dumps({
            "segments": graph_segs,
            "context": graph_ctx,
            "scene_center": gc,
            "scene_extent": ge,
        })

        # Advance stage
        self.app["session"]["stage"] = "graph"
        self.app["session"]["instances_finalized"] = True
        self.app["session"]["graph_reviewed"] = []

        with open(output_dir / "gt_session.json", "w") as f:
            json.dump(self.app["session"], f, indent=2)

        self._send_json(200, {"ok": True, "meta": meta,
                               "graph_nodes": len(graph["nodes"]),
                               "graph_edges": len(graph["edges"])})

    # --- Graph routes ---

    def _serve_graph_data(self):
        dynamic = json.dumps({
            "edges": self.app["edges"],
            "session": self.app["session"],
            "graph_method": self.app.get("graph_method", "proximity"),
            "has_mesh": self.app.get("mesh_path") is not None,
        })
        static = self.app.get("graph_static_json", "{}")
        body = (static[:-1] + "," + dynamic[1:]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_add_edge(self):
        req = self._read_json()
        source = int(req["source"])
        target = int(req["target"])

        xyz = self.app["xyz"]
        labels = self.app.get("graph_labels", self.app["labels"])

        src_pts = xyz[labels == source]
        tgt_pts = xyz[labels == target]

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            self._send_json(400, {"error": "Segment has no points"})
            return

        # Chunked distance computation
        min_dist = float("inf")
        best_src_idx = 0
        chunk_size = 2000
        n_close = 0
        threshold = 0.1

        for i in range(0, len(src_pts), chunk_size):
            src_chunk = src_pts[i:i + chunk_size]
            diffs = src_chunk[:, None, :] - tgt_pts[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            chunk_min_idx = np.unravel_index(dists.argmin(), dists.shape)
            chunk_min = dists[chunk_min_idx]
            if chunk_min < min_dist:
                min_dist = chunk_min
                best_src_idx = i + chunk_min_idx[0]
            n_close += int((dists < threshold).sum())

        endpoint = src_pts[best_src_idx].tolist()
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

        self._send_json(200, {"ok": True, "edge": edge})
        print(f"  Added edge: {source} → {target} (dist={min_dist:.4f}m)")

    def _handle_review_edge(self):
        req = self._read_json()
        reviewed = self.app["session"].setdefault("graph_reviewed", [])
        reviewed.append(req)

        # Auto-save session
        out_dir = self.app["output_dir"]
        with open(out_dir / "gt_session.json", "w") as f:
            json.dump(self.app["session"], f, indent=2)

        self._send_json(200, {"ok": True})

    def _handle_export_graph(self):
        summary = export_graph(
            self.app["edges"],
            self.app["session"],
            self.app["output_dir"],
            self.app["labels_fingerprint"],
        )
        self._send_json(200, {"ok": True, "summary": summary})
        print(f"  Exported graph: {summary['accepted']} accepted, "
              f"{summary['rejected']} rejected, {summary['pending']} pending")


# --- main() ---


def main():
    parser = argparse.ArgumentParser(description="Unified annotation tool")
    parser.add_argument("project_dir", nargs="?", help="Project directory with project.yaml")
    parser.add_argument("--points", help="Input PLY point cloud")
    parser.add_argument("--labels", help="Instance labels .npy")
    parser.add_argument("--summary", help="RANSAC summary JSON (optional)")
    parser.add_argument("--resume", help="Resume from gt_session.json")
    parser.add_argument("--mesh", help="GLB mesh file (optional)")
    parser.add_argument("--config", help="Classes YAML config (optional)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", default="annotation")
    args = parser.parse_args()

    # TODO: project_dir mode reads project.yaml — for now, require explicit args
    if not args.points or not args.labels:
        if not args.project_dir:
            parser.error("Provide project_dir or --points and --labels")
        parser.error("Project dir mode not yet implemented — use --points and --labels")

    class_map, class_colors, class_keys, process_classes = load_classes(args.config)

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
    session = {
        "stage": "label",
        "confirmed": [],
        "hidden_segments": [],
        "splits": [],
        "patches_finalized": False,
        "instances_finalized": False,
        "graph_reviewed": [],
        "next_gt_id": 0,
    }

    session_path = output_dir / "gt_session.json"
    resume_path = Path(args.resume) if args.resume else session_path
    if resume_path.exists():
        with open(resume_path) as f:
            session = json.load(f)
        stage = session.get("stage", "label")
        print(f"  Resumed session at stage '{stage}'")
        n_confirmed = len(session.get("confirmed", []))
        if n_confirmed:
            print(f"  {n_confirmed} confirmed segments")

        # Fingerprint validation
        saved_fp = session.get("labels_fingerprint")
        if saved_fp and saved_fp != current_fp:
            print(f"\n  WARNING: labels fingerprint mismatch!")
            print(f"    Session expects: {saved_fp}")
            print(f"    --labels file:   {current_fp}")

            recovered = False
            backup = output_dir / "source_instance_labels.npy"
            if backup.exists():
                backup_labels = np.load(backup)
                backup_fp = labels_fingerprint(backup_labels)
                if backup_fp == saved_fp:
                    labels = backup_labels
                    current_fp = backup_fp
                    n_instances = len(np.unique(labels[labels >= 0]))
                    print(f"  RECOVERED: using backup {backup}")
                    recovered = True

            if not recovered:
                print(f"\n  ERROR: Cannot find labels matching session fingerprint.")
                print(f"  Refusing to load — would corrupt annotations.")
                sys.exit(1)

    # Stamp creation metadata
    session.setdefault("created", datetime.datetime.now().isoformat())
    session.setdefault("labels_source_path", str(Path(args.labels).resolve()))
    session.setdefault("points_source_path", str(Path(args.points).resolve()))

    # Source backup
    src_labels_backup = output_dir / "source_instance_labels.npy"
    if not src_labels_backup.exists():
        np.save(src_labels_backup, labels)
        print(f"  Backed up instance_labels → {src_labels_backup}")

    # --- Stage-specific data preparation ---
    stage = session.get("stage", "label")

    print("Preparing segments...")
    segments, context, scene_center, scene_extent, point_indices_map = \
        prepare_segments(xyz, rgb, labels, summary)
    print(f"  {len(segments)} segments prepared for browser")

    # Compute next_segment_id
    max_id = max((s["id"] for s in segments), default=0)
    next_segment_id = max(max_id + 1, 1000)

    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"  Warning: mesh file not found: {mesh_path}")
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
        "xyz": xyz,
        "rgb": rgb,
        "point_indices_map": point_indices_map,
        "output_dir": output_dir,
        "mesh_path": mesh_path,
        "labels_fingerprint": current_fp,
        "labels_source_path": str(Path(args.labels).resolve()),
        "points_source_path": str(Path(args.points).resolve()),
        "class_map": class_map,
        "class_colors": class_colors,
        "class_keys": class_keys,
        "process_classes": process_classes,
        "summary": summary,
        "next_segment_id": next_segment_id,
    }

    # If resuming at graph stage, load graph data
    if stage == "graph":
        graph_path = output_dir / "adjacency_graph.json"
        if graph_path.exists():
            with open(graph_path) as f:
                graph = json.load(f)
            instance_labels = np.load(output_dir / "instance_ids.npy")
            graph_segs, graph_ctx, gc, ge = prepare_graph_segments(
                xyz, rgb, instance_labels, graph["nodes"],
            )
            app_state["graph_segments"] = graph_segs
            app_state["graph_context"] = graph_ctx
            app_state["graph_center"] = gc
            app_state["graph_extent"] = ge
            app_state["edges"] = graph["edges"]
            app_state["graph_labels"] = instance_labels
            app_state["graph_method"] = graph.get("method", "proximity")
            app_state["graph_static_json"] = json.dumps({
                "segments": graph_segs,
                "context": graph_ctx,
                "scene_center": gc,
                "scene_extent": ge,
            })
            print(f"  Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        else:
            print(f"  WARNING: graph stage but no adjacency_graph.json found")

    handler = partial(AnnotateHandler, app_state=app_state)
    server = HTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\nAnnotation Tool running at {url} (stage: {stage})")
    print("Press Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
