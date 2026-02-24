# Unified Annotate Tool — Implementation Design

## Decisions

- **Approach:** Clean-room rewrite. New `annotate/` module replaces `labeler/`, `doubles/`, `graph/` (server + viewer).
- **Stage transitions:** Server-side mode switch + page reload. Server is source of truth for stage.
- **Stage 2 workflow:** Group-and-label (select multiple segments, assign class, group becomes one patch). Same interaction as current labeler.
- **Split UX:** Port OBB drawing UI from doubles viewer, re-wire data source to fetch from main server.
- **Graph builder:** Auto-runs inline when finalizing instances (no separate CLI step).
- **Stage numbering:** Flat 1–5 (Segment, Label & Split, Instance Merge, Connection Graph, Equipment Description). No "sub-stages".
- **Versioning:** Per-stage, only on finalize when input fingerprints differ from last version. Metadata records input SHA256s.
- **Staleness:** Downstream stages warn if their input fingerprints don't match latest upstream outputs. Stale versions kept, not deleted.
- **Graph export:** `connectivity_graph.json` contains only accepted edges + all nodes (including isolated). No action field. Review progress stays in `graph_session.json`.

## File Structure

```
src/industrial_point_labeler/
├── annotate/
│   ├── __init__.py
│   ├── server.py          # Unified HTTP server, all routes
│   ├── data.py            # Segment preparation, export logic, versioning
│   ├── viewer.html        # Stages 2 (label) and 3 (instance merge)
│   ├── splitter.html      # OBB split tool (opens in new tab)
│   └── graph_viewer.html  # Stage 4 (split-view 3D + 2D graph)
├── graph/
│   └── builder.py         # Offline graph builder (imported by server for 3→4)
├── segmentation/          # Unchanged
├── equipment/             # Unchanged
├── io.py                  # Unchanged
├── config.py              # Unchanged
└── cli.py                 # Updated: label/doubles/review-graph → annotate
```

`viewer.html` handles both stages 2 and 3 (similar point-cloud + sidebar layouts). Server sends `stage` in `/data`, JS toggles controls. `graph_viewer.html` is separate (fundamentally different split-view layout). `splitter.html` opens in a new tab from stage 2.

Old `labeler/`, `doubles/`, `graph/server.py`, `graph/viewer.html` deleted after validation.

## Server Routes (`annotate/server.py`)

Single `AnnotateHandler` class. Session `stage` field drives routing.

| Route | Method | Stage | Purpose |
|-------|--------|-------|---------|
| `/` | GET | 2/3 | Serve `viewer.html` |
| `/` | GET | 4 | Serve `graph_viewer.html` |
| `/data` | GET | all | Segment data + session (content varies by stage) |
| `/save` | POST | all | Save session state |
| `/split` | GET | 2 | Serve `splitter.html` |
| `/split-data` | GET | 2 | Full-res points for selected segments |
| `/apply-split` | POST | 2 | Create new segments, remove originals |
| `/finalize-patches` | POST | 2 | Save patch outputs, advance to stage 3 |
| `/finalize-instances` | POST | 3 | Save instance outputs, run graph builder, advance to stage 4 |
| `/graph-data` | GET | 4 | Adjacency graph + review state |
| `/add-edge` | POST | 4 | Manual edge creation |
| `/review-edge` | POST | 4 | Accept/reject edge |
| `/export-graph` | POST | 4 | Export connectivity graph |
| `/mesh.glb` | GET | all | Serve mesh file |

Stage enforcement: routes that don't match current stage return 409 Conflict.

`/data` response varies by stage:
- **Stage 2 (label):** segments + context + session + class config
- **Stage 3 (instance):** same structure, session includes finalized patch info, sidebar groups by class
- **Stage 4 (graph):** segments (process-class only) + edges + graph session, with pre-cached static JSON

## Data Layer (`annotate/data.py`)

Pure functions, no server state.

**Segment preparation:**
- `prepare_segments(xyz, rgb, labels, summary=None)` — returns segments, context cloud, scene info, `point_indices_map`
- `prepare_graph_segments(xyz, rgb, labels, graph_nodes)` — segment data for graph-relevant nodes only

**Split logic:**
- `get_split_data(xyz, rgb, labels, point_indices_map, segment_ids)` — full-res points for splitter (no subsampling)
- `apply_split(labels, point_indices_map, segments, split_result)` — creates new segment IDs, updates `labels` and `point_indices_map` in place, returns new segment entries. Unassigned points become remainder segment.

**Export (with versioning):**
- `export_patches(session, labels, point_indices_map, output_dir, class_map, input_fingerprints)` — writes `patch_ids.v{N}.npy`, `patch_classes.v{N}.npy`, `patch_metadata.v{N}.json`. New version only if input fingerprints differ from last version.
- `export_instances(session, labels, point_indices_map, output_dir, class_map, input_fingerprints)` — writes `instance_ids.v{N}.npy`, `instance_classes.v{N}.npy`, `instance_metadata.v{N}.json`. Same versioning rule.
- `export_graph(edges, nodes, session, output_dir, input_fingerprints)` — writes `connectivity_graph.v{N}.json` containing only accepted edges + all nodes (including isolated). Review progress stays in `graph_session.json`.

## Frontend: `viewer.html` (stages 2 + 3)

One HTML file, JS toggles controls based on `stage` from `/data`.

**Shared UI:**
- Three.js point cloud + orbit controls
- Context cloud (grey, 50k max)
- Ctrl+Click segment selection, Ctrl+Enter to confirm with class
- Sidebar with segment list, class color legend, keyboard shortcuts (1-4)
- Auto-save on every confirm

**Stage 2 only (stage="label"):**
- "Split Selected" — opens `/split?segments=...` in new tab
- "Refresh" — re-fetches `/data` after splits
- "Finalize Patches" — POSTs `/finalize-patches`, page reloads into stage 3

**Stage 3 only (stage="instance"):**
- Sidebar groups patches by class
- Selection operates on patches
- Ctrl+Enter merges selected patches into one instance (same-class enforced)
- Confirmed/unconfirmed navigation
- "Finalize Instances" — POSTs `/finalize-instances`, page reloads into stage 4

Ported from current labeler: Three.js scene setup, camera, raycasting, segment rendering (per-segment BufferGeometry), selection highlighting, sidebar list, keyboard shortcuts, session save. Vanilla JS, no framework.

## Frontend: `splitter.html`

Port of `doubles/viewer.html`, re-wired to talk to main server.

- Fetches `/split-data?segments=12,34,56` instead of reading local PLY files
- OBB box drawing + group assignment unchanged
- POSTs to `/apply-split` with group assignments instead of `/save_splits`
- No manifest file — server provides data, receives results

## Frontend: `graph_viewer.html`

Port of `graph/viewer.html`.

- Split-view: 3D point cloud (left) + 2D force-directed graph (right)
- Fetches `/graph-data` for segments + edges + review state
- A/R/S for accept/reject/skip, Ctrl+Click for manual edge source/target
- Auto-save review progress
- "Export Graph" — POSTs `/export-graph`

## CLI

```python
commands = {
    "segment":            ("...segmentation.ransac",  "RANSAC primitive segmentation"),
    "annotate":           ("...annotate.server",      "Interactive annotation tool"),
    "describe-equipment": ("...equipment.server",     "Equipment description tool"),
}
```

`ipl annotate` accepts project-dir mode or legacy explicit args:
```
ipl annotate <project-dir>
ipl annotate --points scan.ply --labels instance_labels.npy [--summary ...] [--mesh ...] [--port 8766]
```

## Stage Resumption

On startup, `main()` reads `gt_session.json` and resumes at the saved stage:

1. **stage="label" (stage 2):** Load RANSAC segments, prepare for labeling
2. **stage="instance" (stage 3):** Load latest `patch_ids.v{N}.npy` + `patch_classes.v{N}.npy`, prepare for merge
3. **stage="graph" (stage 4):** Load latest `instance_ids.v{N}.npy` + `instance_classes.v{N}.npy` + `adjacency_graph.json`, prepare for review

Each stage loads only its required data. Fingerprint validation and source backup work the same as current labeler.

On resume, staleness checks run: if a downstream stage's recorded input fingerprints don't match the latest upstream outputs, the tool prints a warning. The user can continue or choose to redo.
