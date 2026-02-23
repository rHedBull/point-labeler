# Industrial Point Labeler — Design

## Overview

A multi-stage interactive pipeline for segmenting and annotating industrial LiDAR point clouds (pipe stations, pumping stations, process plants). Extracted from a NavVis scan annotation project, generalized for any industrial point cloud.

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
                    │                     │
            Stage 2 │   ipl annotate      │  Unified annotation tool
                    │                     │  3 sub-stages in one tool:
                    │  ┌────────────────┐ │
                    │  │ 2a: Label &    │ │  Assign classes, split
                    │  │     Split      │ │  multi-object segments
                    │  │ [Finalize]     │ │
                    │  ├────────────────┤ │
                    │  │ 2b: Instance   │ │  Merge patches into
                    │  │     Merge      │ │  complete instances
                    │  │ [Finalize]     │ │
                    │  ├────────────────┤ │
                    │  │ 2c: Connection │ │  Review/create edges
                    │  │     Graph      │ │  3D left + 2D graph right
                    │  │ [Export]       │ │
                    │  └────────────────┘ │
                    └──────────┬──────────┘
                               │ patch_ids.npy, patch_classes.npy
                               │ instance_ids.npy, instance_classes.npy
                               │ reviewed_edges.json
                    ┌──────────▼──────────┐
            Stage 3 │  ipl describe-equip │  Free-text equipment labels
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

### Stage 2: Unified Annotation (`labeler/server.py` + `labeler/viewer.html`)

One tool, three sub-stages with explicit Finalize checkpoints between them. Each sub-stage saves its output independently — redoing a later stage doesn't lose earlier work.

#### Sub-stage 2a: Label & Split

Starting from RANSAC pre-segments, assign class labels and split segments that span multiple physical objects.

**Invariant:** Every output patch has exactly one class label and belongs to at most one physical object. Patches may be fragments (completeness comes in 2b).

- Ctrl+Click to select segments, Ctrl+Enter to assign class label (pipe, tank, equipment, structural)
- "Split Selected" button opens OBB splitter in a new browser tab
  - Splitter adapted from doubles_viewer: OBB box drawing, group assignment
  - Fetches full-resolution point data from server (`/split-data`)
  - POSTs results back (`/apply-split`), creates new segments, removes originals
  - Unassigned points become a remainder segment
- "Refresh" button in main tool to pick up new segments after split
- Split history recorded in session for reconstruction

**Finalize Patches** button saves:
- `patch_ids.npy` — per-point patch ID (int32, -1 for unlabeled)
- `patch_classes.npy` — per-point class ID (int32, -1 for unlabeled)
- `patch_metadata.json` — patch count, class distribution, split history

#### Sub-stage 2b: Instance Merge

After patches are finalized, merge patches that belong to the same physical object.

**Invariant:** Every output segment is one complete physical instance with a class label.

- Sidebar shows labeled patches grouped by class
- Ctrl+Click to select patches belonging to the same instance
- Ctrl+Enter to confirm merge (class inherited from patches — all must share the same class)
- Existing confirmed/unconfirmed navigation still works

**Finalize Instances** button saves:
- `instance_ids.npy` — per-point instance ID (int32)
- `instance_classes.npy` — per-point class ID (int32)
- `instance_metadata.json` — instance count, class distribution, per-instance info

Between 2b and 2c, `build_connectivity_graph.py` runs offline to generate `adjacency_graph.json`.

#### Sub-stage 2c: Connection Graph

Review and edit the connectivity graph describing how instances connect (pipe→valve→tank).

- Split-view layout: 3D point cloud (left) + 2D force-directed graph (right)
- Sidebar lists edges, click to focus in 3D
- Accept (A) / Reject (R) / Skip (S) per edge
- Ctrl+Click to select source/target for manual edge creation
- Enter to create edge with auto-computed distance

**Export Graph** button saves:
- `reviewed_edges.json` — accepted/rejected edges with distances and endpoints
- `graph_session.json` — review progress for resumption

### Stage 3: Equipment Description (`equipment/server.py` + `equipment/viewer.html`)

Browse equipment-class segments and annotate with free-text descriptions (valve, flange, pump, etc.). Split-view point cloud + mesh.

**Inputs:** PLY, instance labels, adjacency graph, mesh.glb (optional)
**Outputs:** equipment_descriptions.json

## Server Routes (Unified Annotation Tool)

| Route | Method | Sub-stage | Purpose |
|-------|--------|-----------|---------|
| `/` | GET | all | Main viewer HTML |
| `/data` | GET | all | Segment + session data |
| `/save` | POST | all | Save session state |
| `/split` | GET | 2a | Serve splitter HTML |
| `/split-data` | GET | 2a | Full-res point data for selected segments |
| `/apply-split` | POST | 2a | Apply split results |
| `/finalize-patches` | POST | 2a→2b | Save patch output, transition to instance merge |
| `/finalize-instances` | POST | 2b→2c | Save instance output, transition to graph |
| `/graph-data` | GET | 2c | Adjacency graph + review state |
| `/add-edge` | POST | 2c | Create manual edge |
| `/review-edge` | POST | 2c | Accept/reject edge |
| `/export-graph` | POST | 2c | Export reviewed graph |

## Session State

```json
{
  "stage": "label" | "instance" | "graph",
  "confirmed": [],
  "hidden_segments": [],
  "splits": [],
  "patches_finalized": false,
  "instances_finalized": false,
  "graph_reviewed": []
}
```

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
├── annotation/               # ipl annotate outputs
│   ├── gt_session.json
│   ├── patch_ids.npy         # sub-stage 2a checkpoint
│   ├── patch_classes.npy
│   ├── patch_metadata.json
│   ├── instance_ids.npy      # sub-stage 2b checkpoint
│   ├── instance_classes.npy
│   ├── instance_metadata.json
│   ├── adjacency_graph.json  # built between 2b and 2c
│   ├── reviewed_edges.json   # sub-stage 2c output
│   └── graph_session.json
└── equipment/                # ipl describe-equipment outputs
```

Commands simplify to:
```bash
ipl init my-site/ --points scan.ply --mesh mesh.glb
ipl segment my-site/
ipl annotate my-site/
ipl status my-site/
```

Stage dependencies are enforced: `ipl annotate my-site/` fails with a clear message if `segment` hasn't completed. Each stage stamps `project.yaml` on completion.

Legacy explicit-args mode (`--points`, `--labels`, etc.) still works for advanced use.

## Data Integrity

- **Fingerprint validation:** SHA256 of label bytes. On session resume, if the fingerprint doesn't match, the tool searches for a backup and either auto-recovers or aborts with a clear error.
- **Source backup:** First time the labeler runs, it saves `source_instance_labels.npy` so GT can always be reconstructed even if segmentation is re-run.
- **Session state:** All interactive tools auto-save on every user action. Sessions can be resumed after closing the browser.
- **Stage checkpoints:** Each sub-stage output is saved independently. Redoing instance merge doesn't lose patch labels. Redoing graph review doesn't lose instance definitions.

## Data Flow Summary

```
Input:
  scan.ply + instance_labels.npy (from RANSAC)

Sub-stage 2a output:
  patch_ids.npy, patch_classes.npy, patch_metadata.json

Sub-stage 2b output:
  instance_ids.npy, instance_classes.npy, instance_metadata.json

Between 2b and 2c:
  Run build_connectivity_graph.py → adjacency_graph.json

Sub-stage 2c output:
  reviewed_edges.json, graph_session.json
```

## Input Format

- **Point cloud:** PLY with vertex properties `x, y, z, red, green, blue`
- **Mesh (optional):** GLB format for split-view overlay
- **Labels:** NumPy int32 arrays (.npy), -1 = unlabeled
- **Metadata:** JSON throughout

## Dependencies

- **Core:** numpy, plyfile, scipy, pyyaml
- **Segmentation only (optional):** open3d, scikit-learn
- **Frontend:** Three.js (loaded from CDN in HTML viewers, no build step)
