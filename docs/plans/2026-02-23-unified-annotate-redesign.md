# Unified Annotate Tool — Implementation Design

## Decisions

- **Approach:** Clean-room rewrite. New `annotate/` module replaces `labeler/`, `doubles/`, `graph/` (server + viewer).
- **Sub-stage transitions:** Server-side mode switch + page reload. Server is source of truth for stage.
- **2a workflow:** Group-and-label (select multiple segments, assign class, group becomes one patch). Same interaction as current labeler.
- **Split UX:** Port OBB drawing UI from doubles viewer, re-wire data source to fetch from main server.
- **Graph builder:** Auto-runs inline when finalizing instances (no separate CLI step).

## File Structure

```
src/industrial_point_labeler/
├── annotate/
│   ├── __init__.py
│   ├── server.py          # Unified HTTP server, all routes
│   ├── data.py            # Segment preparation, export logic
│   ├── viewer.html        # Sub-stages 2a (label) and 2b (instance merge)
│   ├── splitter.html      # OBB split tool (opens in new tab)
│   └── graph_viewer.html  # Sub-stage 2c (split-view 3D + 2D graph)
├── graph/
│   └── builder.py         # Offline graph builder (imported by server for 2b→2c)
├── segmentation/          # Unchanged
├── equipment/             # Unchanged
├── io.py                  # Unchanged
├── config.py              # Unchanged
└── cli.py                 # Updated: label/doubles/review-graph → annotate
```

`viewer.html` handles both 2a and 2b (similar point-cloud + sidebar layouts). Server sends `stage` in `/data`, JS toggles controls. `graph_viewer.html` is separate (fundamentally different split-view layout). `splitter.html` opens in a new tab from 2a.

Old `labeler/`, `doubles/`, `graph/server.py`, `graph/viewer.html` deleted after validation.

## Server Routes (`annotate/server.py`)

Single `AnnotateHandler` class. Session `stage` field drives routing.

| Route | Method | Stage | Purpose |
|-------|--------|-------|---------|
| `/` | GET | label/instance | Serve `viewer.html` |
| `/` | GET | graph | Serve `graph_viewer.html` |
| `/data` | GET | all | Segment data + session (content varies by stage) |
| `/save` | POST | all | Save session state |
| `/split` | GET | label | Serve `splitter.html` |
| `/split-data` | GET | label | Full-res points for selected segments |
| `/apply-split` | POST | label | Create new segments, remove originals |
| `/finalize-patches` | POST | label | Save patch outputs, advance to "instance" |
| `/finalize-instances` | POST | instance | Save instance outputs, run graph builder, advance to "graph" |
| `/graph-data` | GET | graph | Adjacency graph + review state |
| `/add-edge` | POST | graph | Manual edge creation |
| `/review-edge` | POST | graph | Accept/reject edge |
| `/export-graph` | POST | graph | Export reviewed edges |
| `/mesh.glb` | GET | all | Serve mesh file |

Stage enforcement: routes that don't match current stage return 409 Conflict.

`/data` response varies by stage:
- **label:** segments + context + session + class config
- **instance:** same structure, session includes finalized patch info, sidebar groups by class
- **graph:** segments (process-class only) + edges + graph session, with pre-cached static JSON

## Data Layer (`annotate/data.py`)

Pure functions, no server state.

**Segment preparation:**
- `prepare_segments(xyz, rgb, labels, summary=None)` — returns segments, context cloud, scene info, `point_indices_map`
- `prepare_graph_segments(xyz, rgb, labels, graph_nodes)` — segment data for graph-relevant nodes only

**Split logic:**
- `get_split_data(xyz, rgb, labels, point_indices_map, segment_ids)` — full-res points for splitter (no subsampling)
- `apply_split(labels, point_indices_map, segments, split_result)` — creates new segment IDs, updates `labels` and `point_indices_map` in place, returns new segment entries. Unassigned points become remainder segment.

**Export:**
- `export_patches(session, labels, point_indices_map, output_dir, class_map)` — writes `patch_ids.npy`, `patch_classes.npy`, `patch_metadata.json`
- `export_instances(session, labels, point_indices_map, output_dir, class_map)` — writes `instance_ids.npy`, `instance_classes.npy`, `instance_metadata.json`
- `export_graph(edges, session, output_dir, labels_fingerprint)` — writes `reviewed_edges.json`

## Frontend: `viewer.html` (stages 2a + 2b)

One HTML file, JS toggles controls based on `stage` from `/data`.

**Shared UI:**
- Three.js point cloud + orbit controls
- Context cloud (grey, 50k max)
- Ctrl+Click segment selection, Ctrl+Enter to confirm with class
- Sidebar with segment list, class color legend, keyboard shortcuts (1-4)
- Auto-save on every confirm

**2a-only (stage="label"):**
- "Split Selected" — opens `/split?segments=...` in new tab
- "Refresh" — re-fetches `/data` after splits
- "Finalize Patches" — POSTs `/finalize-patches`, page reloads into 2b

**2b-only (stage="instance"):**
- Sidebar groups patches by class
- Selection operates on patches
- Ctrl+Enter merges selected patches into one instance (same-class enforced)
- Confirmed/unconfirmed navigation
- "Finalize Instances" — POSTs `/finalize-instances`, page reloads into 2c

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

1. **stage="label":** Load RANSAC segments, prepare for labeling
2. **stage="instance":** Load `patch_ids.npy` + `patch_classes.npy`, prepare for merge
3. **stage="graph":** Load `instance_ids.npy` + `instance_classes.npy` + `adjacency_graph.json`, prepare for review

Each stage loads only its required data. Fingerprint validation and source backup work the same as current labeler.
