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
            Stage 2 │   ipl annotate      │  Assign classes, split
                    │   Label & Split     │  multi-object segments
                    │   [Finalize]        │
                    └──────────┬──────────┘
                               │ patch_ids.npy, patch_classes.npy
                    ┌──────────▼──────────┐
            Stage 3 │   ipl annotate      │  Merge patches into
                    │   Instance Merge    │  complete instances
                    │   [Finalize]        │
                    └──────────┬──────────┘
                               │ instance_ids.npy, instance_classes.npy
                               │ adjacency_graph.json (auto-built)
                    ┌──────────▼──────────┐
            Stage 4 │   ipl annotate      │  Review/create edges
                    │   Connection Graph  │  3D left + 2D graph right
                    │   [Export]          │
                    └──────────┬──────────┘
                               │ connectivity_graph.json
                    ┌──────────▼──────────┐
            Stage 5 │  ipl describe-equip │  Free-text equipment labels
                    └─────────────────────┘
```

Stages 2–4 share a single `ipl annotate` server process with server-side transitions between them.

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

### Stage 2: Label & Split (`annotate/server.py` + `annotate/viewer.html`)

Starting from RANSAC pre-segments, assign class labels and split segments that span multiple physical objects.

**Invariant:** Every output patch has exactly one class label and belongs to at most one physical object. Patches may be fragments (completeness comes in stage 3).

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
- `patch_metadata.json` — patch count, class distribution, split history, input fingerprints

### Stage 3: Instance Merge (`annotate/server.py` + `annotate/viewer.html`)

After patches are finalized, merge patches that belong to the same physical object.

**Invariant:** Every output segment is one complete physical instance with a class label.

- Sidebar shows labeled patches grouped by class
- Ctrl+Click to select patches belonging to the same instance
- Ctrl+Enter to confirm merge (class inherited from patches — all must share the same class)
- Existing confirmed/unconfirmed navigation still works

**Finalize Instances** button saves:
- `instance_ids.npy` — per-point instance ID (int32)
- `instance_classes.npy` — per-point class ID (int32)
- `instance_metadata.json` — instance count, class distribution, per-instance info, input fingerprints

Between stages 3 and 4, `build_connectivity_graph.py` runs automatically to generate `adjacency_graph.json`.

### Stage 4: Connection Graph (`annotate/server.py` + `annotate/graph_viewer.html`)

Review and edit the connectivity graph describing how instances connect (pipe→valve→tank).

- Split-view layout: 3D point cloud (left) + 2D force-directed graph (right)
- Sidebar lists edges, click to focus in 3D
- Accept (A) / Reject (R) / Skip (S) per edge
- Ctrl+Click to select source/target for manual edge creation
- Enter to create edge with auto-computed distance

**Export Graph** button saves `connectivity_graph.json` — a clean graph containing:
- **Nodes:** all process-class instances (including isolated nodes with no connections)
- **Edges:** only accepted connections (rejected/pending edges are omitted)
- **Metadata:** input fingerprints, creation timestamp, builder threshold

Review progress is tracked separately in `graph_session.json` (not exported).

### Stage 5: Equipment Description (`equipment/server.py` + `equipment/viewer.html`)

Browse equipment-class segments and annotate with free-text descriptions (valve, flange, pump, etc.). Split-view point cloud + mesh.

**Inputs:** PLY, instance labels, connectivity graph, mesh.glb (optional)
**Outputs:** equipment_descriptions.json

## Server Routes (Unified Annotation Tool)

| Route | Method | Stage | Purpose |
|-------|--------|-------|---------|
| `/` | GET | all | Main viewer HTML |
| `/data` | GET | all | Segment + session data |
| `/save` | POST | all | Save session state |
| `/split` | GET | 2 | Serve splitter HTML |
| `/split-data` | GET | 2 | Full-res point data for selected segments |
| `/apply-split` | POST | 2 | Apply split results |
| `/finalize-patches` | POST | 2→3 | Save patch output, transition to instance merge |
| `/finalize-instances` | POST | 3→4 | Save instance output, build graph, transition to graph review |
| `/graph-data` | GET | 4 | Adjacency graph + review state |
| `/add-edge` | POST | 4 | Create manual edge |
| `/review-edge` | POST | 4 | Accept/reject edge |
| `/export-graph` | POST | 4 | Export connectivity graph |

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
├── segmentation/             # Stage 1 outputs
├── annotation/               # Stages 2–4 outputs (ipl annotate)
│   ├── gt_session.json
│   ├── patch_ids.v1.npy      # Stage 2 output (versioned)
│   ├── patch_classes.v1.npy
│   ├── patch_metadata.v1.json
│   ├── instance_ids.v1.npy   # Stage 3 output (versioned)
│   ├── instance_classes.v1.npy
│   ├── instance_metadata.v1.json
│   ├── adjacency_graph.json  # Auto-built between stages 3→4
│   ├── connectivity_graph.v1.json  # Stage 4 output (versioned)
│   └── graph_session.json    # Stage 4 review progress
└── equipment/                # Stage 5 outputs
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
- **Stage checkpoints:** Each stage's output is saved independently. Redoing stage 3 doesn't lose stage 2's patches. Redoing stage 4 doesn't lose stage 3's instances.

## Versioning

Per-stage output versioning tracks provenance and prevents data loss when re-doing earlier stages.

**Rules:**
- Each stage's metadata records SHA256 fingerprints of its input files.
- On finalize, if input fingerprints differ from the last finalized version, a new version is created (e.g. `patch_ids.v2.npy`). If inputs are unchanged, the current version is overwritten in place.
- On resume, the tool loads the latest version of each stage.
- **Staleness warnings:** If a downstream stage's recorded input fingerprints don't match the latest upstream outputs, the tool warns the user. Stale outputs are not deleted — the user decides whether to redo.

**Metadata example** (`patch_metadata.v2.json`):
```json
{
  "version": 2,
  "input_fingerprints": {
    "instance_labels.npy": "sha256:abc..."
  },
  "created": "2026-02-24T14:30:00",
  "patch_count": 142,
  "class_distribution": {"pipe": 80, "tank": 12, "equipment": 38, "structural": 12}
}
```

## Data Flow Summary

```
Input:
  scan.ply + instance_labels.npy (from RANSAC)

Stage 2 output:
  patch_ids.v{N}.npy, patch_classes.v{N}.npy, patch_metadata.v{N}.json

Stage 3 output:
  instance_ids.v{N}.npy, instance_classes.v{N}.npy, instance_metadata.v{N}.json

Between stages 3 and 4:
  Auto-run build_connectivity_graph.py → adjacency_graph.json

Stage 4 output:
  connectivity_graph.v{N}.json
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
