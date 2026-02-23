# Industrial Point Labeler — Design

## Overview

A 5-stage interactive pipeline for segmenting and annotating industrial LiDAR point clouds (pipe stations, pumping stations, process plants). Extracted from a NavVis scan annotation project, generalized for any industrial point cloud.

## Architecture

Each stage is a self-contained Python HTTP server + Three.js browser frontend. No framework (Flask, FastAPI) — just stdlib `http.server`. The tools share:

- **`io.py`** — PLY loading, label fingerprinting
- **`config.py`** — YAML-based class definitions (labels, colors, keyboard shortcuts)
- **`project.py`** — Project file management (auto-wires stage inputs/outputs)

## Pipeline Stages

```
                        ┌─────────────┐
                        │  scan.ply   │  User's LiDAR point cloud
                        │  mesh.glb   │  Optional textured mesh
                        └──────┬──────┘
                               │
                    ┌──────────▼──────────┐
            Stage 1 │   ipl segment       │  RANSAC primitive fitting
                    │                     │  Planes, cylinders, spheres
                    └──────────┬──────────┘
                               │ instance_labels.npy
                               │ ransac_summary.json
                    ┌──────────▼──────────┐
            Stage 2 │   ipl label         │  Interactive GT annotation
                    │                     │  Merge segments → labeled GT
                    └──────────┬──────────┘
                               │ gt_labels.npy, gt_classes.npy
                               │ gt_metadata.json
                    ┌──────────▼──────────┐
   (optional)       │   ipl doubles       │  Dissect overlapping scans
            Stage 3 │                     │  OBB selection, split groups
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
          Stage 4a  │   ipl build-graph   │  Pipe endpoint connectivity
                    └──────────┬──────────┘
                               │ adjacency_graph.json
                    ┌──────────▼──────────┐
          Stage 4b  │   ipl review-graph  │  Accept/reject edges
                    └──────────┬──────────┘
                               │ reviewed_edges.json
                    ┌──────────▼──────────┐
            Stage 5 │  ipl describe-equip │  Free-text equipment labels
                    └─────────────────────┘
```

### Stage 1: RANSAC Segmentation (`segmentation/ransac.py`)

Automated instance segmentation. No user interaction.

1. Normal estimation (Open3D, KNN=30)
2. Multi-scale principal curvature (k=10, 20, 40) via shape operator quadratic fitting
3. Surface classification by (k1, k2) signature: flat / cylindrical / spherical / saddle / edge
4. Plane extraction: iterative RANSAC
5. Cylinder region growing: spatial + axis + radius compatibility
6. Post-assignment: 3-pass neighbor voting with normal consistency
7. Cylinder merge pass (over-segmentation cleanup)

**Inputs:** PLY point cloud
**Outputs:** `instance_labels.npy`, `ransac_summary.json`, colored PLY visualizations

### Stage 2: GT Annotation (`labeler/server.py` + `labeler/viewer.html`)

The primary interactive tool. Merge-based workflow.

- Ctrl+Click to select segments in 3D viewport
- Keyboard shortcuts (configurable via classes.yaml) to assign class labels
- Merge multiple RANSAC segments into one GT segment
- Unmerge to undo
- Split-view: point cloud (left) + GLB mesh (right)
- Context cloud: full scene subsampled at 30% opacity
- Auto-save with 500ms debounce
- SHA256 fingerprint validation on session resume (prevents silent corruption)
- Auto-backup of source labels on first run

**Inputs:** PLY, instance_labels.npy, ransac_summary.json (optional), mesh.glb (optional)
**Outputs:** gt_labels.npy, gt_classes.npy, gt_metadata.json, gt_session.json

### Stage 3: Doubles Dissection (`doubles/server.py` + `doubles/viewer.html`)

Handles segments from overlapping scan regions. Optional stage.

- 3-panel layout: segment list, 3D viewer, groups panel
- Draw oriented bounding box (click+drag, WASD/QE adjust)
- Color modes: Original / Curvature / Groups
- Assign points to physical objects within a "double" segment

**Inputs:** Pre-extracted per-segment PLY files + manifest.json
**Outputs:** splits.json

### Stage 4: Connectivity Graph (`graph/builder.py` + `graph/server.py` + `graph/viewer.html`)

Two sub-stages:

**4a — Build:** For each pipe segment, find two endpoints via PCA projection (top/bottom 10% of axis). KD-tree query for nearby process-class points within threshold distance.

**4b — Review:** Interactive accept/reject/skip per edge. 2D force-directed graph view (G key). Manual edge creation (click two segments). Fingerprint validation on resume.

**Inputs:** PLY, gt_labels.npy, gt_metadata.json
**Outputs:** adjacency_graph.json, reviewed_edges.json

### Stage 5: Equipment Description (`equipment/server.py` + `equipment/viewer.html`)

Browse equipment-class segments and annotate with free-text descriptions (valve, flange, pump, etc.). Split-view point cloud + mesh.

**Inputs:** PLY, gt_labels.npy, ransac_summary.json, adjacency_graph.json, mesh.glb (optional)
**Outputs:** equipment_descriptions.json

## Class Configuration

`classes.yaml` defines domain-specific labels:

```yaml
classes:
  pipe:
    color: [0.15, 0.8, 0.15]
    key: "1"
  tank:
    color: [1.0, 0.15, 0.15]
    key: "2"
  equipment:
    color: [0.15, 0.15, 1.0]
    key: "3"
  structural:
    color: [1.0, 0.8, 0.15]
    key: "4"
  double:
    color: [0.6, 0.15, 0.8]
    key: "5"

process_classes:
  - pipe
  - tank
  - equipment
```

Users edit this to match their domain. The `process_classes` list controls which segment types participate in graph connectivity.

## Project File System

A `project.yaml` file in the data directory auto-wires all stage inputs/outputs:

```
my-site/
├── project.yaml              # ipl init
├── classes.yaml              # copied from package, editable
├── scan.ply
├── mesh.glb
├── segmentation/             # ipl segment outputs
├── annotation/               # ipl label outputs
├── doubles/                  # ipl doubles outputs (optional)
├── graph/                    # ipl build-graph + review-graph outputs
└── equipment/                # ipl describe-equipment outputs
```

Commands simplify to:
```bash
ipl init my-site/ --points scan.ply --mesh mesh.glb
ipl segment my-site/
ipl label my-site/
ipl status my-site/
```

Stage dependencies are enforced: `ipl label my-site/` fails with a clear message if `segment` hasn't completed. Each stage stamps `project.yaml` on completion.

Legacy explicit-args mode (`--points`, `--labels`, etc.) still works for advanced use.

## Data Integrity

- **Fingerprint validation:** SHA256 of label bytes. On session resume, if the fingerprint doesn't match, the tool searches for a backup and either auto-recovers or aborts with a clear error.
- **Source backup:** First time the labeler runs, it saves `source_instance_labels.npy` so GT can always be reconstructed even if segmentation is re-run.
- **Session state:** All interactive tools auto-save on every user action. Sessions can be resumed after closing the browser.

## Input Format

- **Point cloud:** PLY with vertex properties `x, y, z, red, green, blue`
- **Mesh (optional):** GLB format for split-view overlay
- **Labels:** NumPy int32 arrays (.npy), -1 = unlabeled
- **Metadata:** JSON throughout

## Dependencies

- **Core:** numpy, plyfile, scipy, pyyaml
- **Segmentation only (optional):** open3d, scikit-learn
- **Frontend:** Three.js (loaded from CDN in HTML viewers, no build step)