# Project Mode — Implementation Plan

## Goal

Add `project.yaml`-based pipeline control so users can run `ipl segment my-site/` instead of passing explicit file paths between stages.

## Stage Dependency Graph

```
segment ──► label ──► doubles (optional)
              │
              ├──► build-graph ──► review-graph
              │         │
              │         └──► describe-equipment
              │
              └──► describe-equipment (also needs build-graph)
```

## Stage I/O Wiring (project mode)

| Stage | Reads from | Writes to |
|-------|-----------|-----------|
| segment | `scan.ply` (project root) | `segmentation/` |
| label | `scan.ply`, `segmentation/instance_labels.npy`, `segmentation/ransac_summary.json` | `annotation/` |
| doubles | `doubles/` (pre-staged PLY files) | `doubles/splits.json` |
| build-graph | `scan.ply`, `annotation/gt_labels.npy`, `annotation/gt_metadata.json` | `graph/adjacency_graph.json` |
| review-graph | `scan.ply`, `annotation/gt_labels.npy`, `graph/adjacency_graph.json` | `graph/` |
| describe-equipment | `scan.ply`, `annotation/gt_labels.npy`, `segmentation/ransac_summary.json`, `graph/adjacency_graph.json` | `equipment/` |

All paths relative to the project directory (where `project.yaml` lives).

---

## Task 1: Add `project.py`

**Create:** `src/industrial_point_labeler/project.py`

Constants:
- `STAGE_ORDER` — ordered list of stage names
- `STAGE_DEPS` — dict mapping each stage to its required predecessors
- `STAGE_DIRS` — dict mapping stage names to subdirectory names

Functions:
- `find_project(path)` — locate `project.yaml` in a directory
- `load_project(path)` → `(project_dir, cfg)` — parse YAML
- `save_project(project_dir, cfg)` — write YAML
- `resolve_points(project_dir, cfg)` → absolute Path to point cloud
- `resolve_mesh(project_dir, cfg)` → absolute Path or None
- `resolve_config(project_dir, cfg)` → config path string or None
- `stage_dir(project_dir, stage_name)` → Path to stage output directory
- `check_deps(cfg, stage_name)` → list of missing dependency stage names
- `stamp_stage(project_dir, cfg, stage_name, **extra)` — mark stage completed in YAML

---

## Task 2: Add `ipl init`

**Modify:** `src/industrial_point_labeler/cli.py`

```
ipl init <directory> --points <path> [--mesh <path>]
```

Actions:
1. Create `<directory>/` if needed
2. Resolve `--points` path (relative to project dir if inside, absolute if outside)
3. Same for `--mesh`
4. Copy `config/classes.yaml` from the installed package → `<directory>/classes.yaml`
5. Write `project.yaml` with points, mesh, config fields + null stages
6. Create subdirectories: segmentation/, annotation/, doubles/, graph/, equipment/

---

## Task 3: Add `ipl status`

**Modify:** `src/industrial_point_labeler/cli.py`

```
ipl status [<directory>]
```

Output format:
```
my-site/ pipeline status:

  ✓ segment            1,248 instances (2026-02-23 14:30)
  ✓ label              87 GT segments (2026-02-23 15:45)
  · doubles            skipped
  ✗ build-graph        not started (requires: label ✓)
  ✗ review-graph       not started (requires: build-graph ✗)
  ✗ describe-equipment not started (requires: build-graph ✗)
```

---

## Tasks 4-8: Wire each stage to project mode

Each stage's `main()` gets the same pattern:

1. Add optional positional arg: `parser.add_argument("project_dir", nargs="?")`
2. Make explicit path args optional (not `required=True`)
3. If `project_dir` given:
   - `find_project()` → `load_project()` → resolve all paths from project.yaml
   - `check_deps()` → fail with clear message if prerequisites missing
   - Run the stage
   - `stamp_stage()` on completion
4. If no `project_dir`: require explicit args (legacy mode, same as today)

### Per-stage path resolution

**Task 4 — `segment`:**
```python
input_path  = resolve_points(project_dir, cfg)
output_path = stage_dir(project_dir, "segment")    # → segmentation/
```

**Task 5 — `label`:**
```python
points_path = resolve_points(project_dir, cfg)
labels_path = project_dir / "segmentation" / "instance_labels.npy"
summary_path = project_dir / "segmentation" / "ransac_summary.json"
resume_path  = project_dir / "annotation" / "gt_session.json"  # auto-resume if exists
mesh_path    = resolve_mesh(project_dir, cfg)
config_path  = resolve_config(project_dir, cfg)
output_dir   = project_dir / "annotation"
```

Stamp on each save (pass project info through `app_state`).

**Task 6 — `doubles`:**
```python
data_dir = project_dir / "doubles"
```

**Task 7 — `build-graph`:**
```python
points_path  = resolve_points(project_dir, cfg)
labels_path  = project_dir / "annotation" / "gt_labels.npy"
metadata_path = project_dir / "annotation" / "gt_metadata.json"
output_path  = project_dir / "graph" / "adjacency_graph.json"
config_path  = resolve_config(project_dir, cfg)
```

**Task 7 — `review-graph`:**
```python
points_path = resolve_points(project_dir, cfg)
labels_path = project_dir / "annotation" / "gt_labels.npy"
graph_path  = project_dir / "graph" / "adjacency_graph.json"
mesh_path   = resolve_mesh(project_dir, cfg)
output_dir  = project_dir / "graph"
resume      = (output_dir / "graph_session.json").exists()  # auto-resume
```

**Task 8 — `describe-equipment`:**
```python
points_path  = resolve_points(project_dir, cfg)
labels_path  = project_dir / "annotation" / "gt_labels.npy"
summary_path = project_dir / "segmentation" / "ransac_summary.json"
graph_path   = project_dir / "graph" / "adjacency_graph.json"
mesh_path    = resolve_mesh(project_dir, cfg)
output_dir   = project_dir / "equipment"  # normalized (was: summary.parent)
```

---

## Task 9: Update README

Add "Project Workflow" section:
```bash
ipl init my-site/ --points scan.ply --mesh mesh.glb
ipl segment my-site/
ipl label my-site/
ipl build-graph my-site/
ipl review-graph my-site/
ipl describe-equipment my-site/
ipl status my-site/
```

Keep current explicit-args usage as "Advanced Usage".

---

## Verification

1. `ipl init test-project/ --points /path/to/any.ply` → creates project.yaml, classes.yaml, subdirs
2. `ipl status test-project/` → all stages "not started"
3. `ipl label test-project/` → error: "Stage 'segment' not complete. Run: ipl segment test-project/"
4. `ipl segment --input scan.ply --output out/` → legacy mode still works
5. `ipl <cmd> --help` → all commands print help