# Unified Annotate Tool — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the separate labeler, doubles, and graph modules with a single unified `annotate/` module that handles all three annotation sub-stages.

**Architecture:** Clean-room rewrite. New `annotate/` package with `server.py` (unified HTTP handler), `data.py` (pure data functions), and three HTML files (`viewer.html`, `splitter.html`, `graph_viewer.html`). Old modules deleted after validation. Server-side stage transitions with page reload.

**Tech Stack:** Python stdlib `http.server`, numpy, Three.js (CDN), vanilla JS. No frameworks.

**Reference files:**
- Design doc: `docs/plans/2026-02-23-unified-annotate-redesign.md`
- Old labeler server: `src/industrial_point_labeler/labeler/server.py` (457 lines)
- Old doubles server: `src/industrial_point_labeler/doubles/server.py` (87 lines)
- Old graph server: `src/industrial_point_labeler/graph/server.py` (425 lines)
- Old graph builder: `src/industrial_point_labeler/graph/builder.py` (217 lines)
- Old labeler viewer: `src/industrial_point_labeler/labeler/viewer.html` (1302 lines)
- Old doubles viewer: `src/industrial_point_labeler/doubles/viewer.html` (908 lines)
- Old graph viewer: `src/industrial_point_labeler/graph/viewer.html` (1806 lines)

---

### Task 1: Scaffold annotate module + test infrastructure

**Files:**
- Create: `src/industrial_point_labeler/annotate/__init__.py`
- Create: `src/industrial_point_labeler/annotate/data.py` (empty)
- Create: `src/industrial_point_labeler/annotate/server.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_data.py` (empty)

**Step 1: Create the annotate package directory and empty files**

```python
# annotate/__init__.py
```

```python
# annotate/data.py
"""Segment preparation, split logic, and export functions for the annotation tool."""
```

```python
# annotate/server.py
"""Unified HTTP server for the annotation tool (stages 2a, 2b, 2c)."""
```

**Step 2: Create test infrastructure**

```python
# tests/conftest.py
"""Shared fixtures for annotation tool tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_cloud():
    """Minimal point cloud: 3 clusters of 100 points each."""
    rng = np.random.default_rng(42)
    n_per = 100
    # Cluster 0 centered at (0,0,0), cluster 1 at (5,0,0), cluster 2 at (10,0,0)
    centers = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float64)
    xyz = np.vstack([c + rng.normal(0, 0.3, (n_per, 3)) for c in centers])
    rgb = rng.integers(0, 255, (len(xyz), 3), dtype=np.uint8)
    labels = np.repeat(np.array([0, 1, 2], dtype=np.int32), n_per)
    return xyz, rgb, labels


@pytest.fixture
def sample_session():
    """Minimal session state at label stage."""
    return {
        "stage": "label",
        "confirmed": [],
        "hidden_segments": [],
        "splits": [],
        "patches_finalized": False,
        "instances_finalized": False,
        "graph_reviewed": [],
    }
```

```python
# tests/test_data.py
"""Tests for annotate/data.py."""
```

**Step 3: Verify pytest runs**

Run: `cd /home/hendrik/coding/engine/industrial-point-labeler && python -m pytest tests/ -v --co`
Expected: collects 0 tests, no import errors

**Step 4: Commit**

```bash
git add src/industrial_point_labeler/annotate/ tests/
git commit -m "scaffold: annotate module and test infrastructure"
```

---

### Task 2: `data.py` — prepare_segments

Port `prepare_segments` from `labeler/server.py:30-123` and `prepare_graph_segments` from `graph/server.py:28-95`. Extract as pure functions.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/data.py`
- Modify: `tests/test_data.py`

**Step 1: Write tests for prepare_segments**

```python
# tests/test_data.py
import numpy as np
from industrial_point_labeler.annotate.data import prepare_segments


def test_prepare_segments_basic(sample_cloud):
    xyz, rgb, labels = sample_cloud
    segments, context, center, extent, pim = prepare_segments(xyz, rgb, labels)

    assert len(segments) == 3
    assert all(s["id"] in (0, 1, 2) for s in segments)
    assert all("x" in s and "y" in s and "z" in s for s in segments)
    assert all("r" in s and "g" in s and "b" in s for s in segments)
    assert len(pim) == 3  # 3 segments in point_indices_map
    assert all(isinstance(v, np.ndarray) for v in pim.values())


def test_prepare_segments_skips_small(sample_cloud):
    xyz, rgb, labels = sample_cloud
    # Add a tiny segment (3 points) that should be skipped
    labels = labels.copy()
    labels[:3] = 99
    segments, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    seg_ids = {s["id"] for s in segments}
    assert 99 not in seg_ids


def test_prepare_segments_subsamples_large():
    rng = np.random.default_rng(42)
    n = 5000
    xyz = rng.normal(0, 1, (n, 3))
    rgb = rng.integers(0, 255, (n, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int32)
    segments, _, _, _, _ = prepare_segments(xyz, rgb, labels)
    assert len(segments) == 1
    assert len(segments[0]["x"]) == 2000  # subsampled


def test_prepare_segments_context_cloud_limit():
    rng = np.random.default_rng(42)
    n = 80000
    xyz = rng.normal(0, 1, (n, 3))
    rgb = rng.integers(0, 255, (n, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int32)
    _, context, _, _, _ = prepare_segments(xyz, rgb, labels)
    assert len(context["x"]) == 50000  # capped


def test_prepare_segments_with_summary(sample_cloud):
    xyz, rgb, labels = sample_cloud
    summary = [{"id": 0, "type": "cylinder", "radius": 0.05}]
    segments, _, _, _, _ = prepare_segments(xyz, rgb, labels, summary=summary)
    seg0 = next(s for s in segments if s["id"] == 0)
    assert seg0["type"] == "cylinder"
    assert seg0["radius"] == 0.05
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — ImportError (prepare_segments not defined)

**Step 3: Implement prepare_segments**

Port from `labeler/server.py:30-123`. Same logic, same subsampling thresholds (2000 per segment, 50000 context).

```python
# annotate/data.py
"""Segment preparation, split logic, and export functions for the annotation tool."""

import numpy as np


def prepare_segments(xyz, rgb, labels, summary=None):
    """Build per-segment data for the browser, keeping full indices server-side.

    Returns (segments, context, scene_center, scene_extent, point_indices_map).
    """
    np.random.seed(42)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids >= 0]

    summary_map = {}
    if summary:
        for p in summary:
            summary_map[p["id"]] = p

    segments = []
    point_indices_map = {}

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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: all 5 tests PASS

**Step 5: Commit**

```bash
git add src/industrial_point_labeler/annotate/data.py tests/test_data.py
git commit -m "feat(annotate): add prepare_segments to data layer"
```

---

### Task 3: `data.py` — prepare_graph_segments

Port from `graph/server.py:28-95`. Builds segment data for process-class nodes only.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/data.py`
- Modify: `tests/test_data.py`

**Step 1: Write tests**

```python
# append to tests/test_data.py
from industrial_point_labeler.annotate.data import prepare_graph_segments


def test_prepare_graph_segments_filters_by_nodes(sample_cloud):
    xyz, rgb, labels = sample_cloud
    # Only include nodes 0 and 2 in graph
    nodes = [
        {"id": 0, "label": "pipe"},
        {"id": 2, "label": "tank"},
    ]
    segments, context, center, extent = prepare_graph_segments(xyz, rgb, labels, nodes)
    seg_ids = {s["id"] for s in segments}
    assert seg_ids == {0, 2}
    assert segments[0]["label"] in ("pipe", "tank")
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_data.py::test_prepare_graph_segments_filters_by_nodes -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Append to `annotate/data.py`:

```python
def prepare_graph_segments(xyz, rgb, labels, graph_nodes):
    """Build per-segment data for process-relevant graph nodes only.

    Returns (segments, context, scene_center, scene_extent).
    """
    np.random.seed(42)

    node_map = {n["id"]: n for n in graph_nodes}
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
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add src/industrial_point_labeler/annotate/data.py tests/test_data.py
git commit -m "feat(annotate): add prepare_graph_segments"
```

---

### Task 4: `data.py` — split logic

New functions: `get_split_data` and `apply_split`. These don't exist in old code — the old doubles tool used local PLY files.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/data.py`
- Modify: `tests/test_data.py`

**Step 1: Write tests**

```python
# append to tests/test_data.py
from industrial_point_labeler.annotate.data import get_split_data, apply_split


def test_get_split_data_returns_full_res(sample_cloud):
    xyz, rgb, labels = sample_cloud
    _, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    data = get_split_data(xyz, rgb, pim, [0, 1])
    # Should return all points for segments 0 and 1 (no subsampling)
    total = len(pim[0]) + len(pim[1])
    assert len(data["x"]) == total
    assert len(data["segment_ids"]) == total


def test_apply_split_creates_new_segments(sample_cloud):
    xyz, rgb, labels = sample_cloud
    labels = labels.copy()
    segments, _, _, _, pim = prepare_segments(xyz, rgb, labels)

    # Split segment 0 into two groups: first 50 points → group A, next 50 → group B
    indices_0 = pim[0]
    split_result = {
        "original_segment_ids": [0],
        "groups": [
            {"group": "A", "point_indices": list(range(50))},
            {"group": "B", "point_indices": list(range(50, 100))},
        ],
    }

    new_segs, next_id = apply_split(labels, pim, segments, split_result, next_segment_id=100)

    # Original segment 0 removed, two new segments created
    assert 0 not in pim
    assert 100 in pim
    assert 101 in pim
    assert len(pim[100]) == 50
    assert len(pim[101]) == 50
    assert next_id == 102
    # Labels array updated
    assert np.all(labels[indices_0[:50]] == 100)
    assert np.all(labels[indices_0[50:]] == 101)


def test_apply_split_remainder(sample_cloud):
    xyz, rgb, labels = sample_cloud
    labels = labels.copy()
    segments, _, _, _, pim = prepare_segments(xyz, rgb, labels)

    # Only assign 30 of 100 points to a group — rest become remainder
    split_result = {
        "original_segment_ids": [0],
        "groups": [
            {"group": "A", "point_indices": list(range(30))},
        ],
    }

    new_segs, next_id = apply_split(labels, pim, segments, split_result, next_segment_id=100)

    assert 100 in pim  # group A
    assert 101 in pim  # remainder
    assert len(pim[100]) == 30
    assert len(pim[101]) == 70
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_data.py -k split -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Append to `annotate/data.py`:

```python
def get_split_data(xyz, rgb, point_indices_map, segment_ids):
    """Return full-resolution point data for the given segments (no subsampling).

    Returns dict with x, y, z, r, g, b, segment_ids (per-point) lists.
    """
    all_pts = []
    all_cols = []
    all_seg_ids = []

    for sid in segment_ids:
        if sid not in point_indices_map:
            continue
        indices = point_indices_map[sid]
        all_pts.append(xyz[indices])
        all_cols.append(rgb[indices])
        all_seg_ids.append(np.full(len(indices), sid, dtype=np.int32))

    if not all_pts:
        return {"x": [], "y": [], "z": [], "r": [], "g": [], "b": [], "segment_ids": []}

    pts = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    seg_ids = np.concatenate(all_seg_ids)

    return {
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "r": cols[:, 0].tolist(),
        "g": cols[:, 1].tolist(),
        "b": cols[:, 2].tolist(),
        "segment_ids": seg_ids.tolist(),
    }


def apply_split(labels, point_indices_map, segments, split_result, next_segment_id):
    """Apply a split operation: create new segments, remove originals.

    Mutates labels and point_indices_map in place.
    Returns (new_segment_entries, next_segment_id).
    """
    original_ids = split_result["original_segment_ids"]
    groups = split_result["groups"]

    # Gather all original point indices in order
    all_original_indices = []
    for sid in original_ids:
        if sid in point_indices_map:
            all_original_indices.append(point_indices_map[sid])
    if not all_original_indices:
        return [], next_segment_id
    all_original_indices = np.concatenate(all_original_indices)

    # Track which points are assigned to a group
    assigned = np.zeros(len(all_original_indices), dtype=bool)
    new_segs = []
    cur_id = next_segment_id

    for g in groups:
        pidx = np.array(g["point_indices"], dtype=np.int64)
        real_indices = all_original_indices[pidx]
        labels[real_indices] = cur_id
        point_indices_map[cur_id] = real_indices
        assigned[pidx] = True
        new_segs.append({"id": cur_id, "group": g["group"], "n_points": len(real_indices)})
        cur_id += 1

    # Remainder: unassigned points
    remainder_pidx = np.where(~assigned)[0]
    if len(remainder_pidx) > 0:
        real_indices = all_original_indices[remainder_pidx]
        labels[real_indices] = cur_id
        point_indices_map[cur_id] = real_indices
        new_segs.append({"id": cur_id, "group": "_remainder", "n_points": len(real_indices)})
        cur_id += 1

    # Remove originals
    for sid in original_ids:
        point_indices_map.pop(sid, None)
        # Remove from segments list
        for i, s in enumerate(segments):
            if s["id"] == sid:
                segments.pop(i)
                break

    return new_segs, cur_id
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: all 9 tests PASS

**Step 5: Commit**

```bash
git add src/industrial_point_labeler/annotate/data.py tests/test_data.py
git commit -m "feat(annotate): add split logic (get_split_data, apply_split)"
```

---

### Task 5: `data.py` — export functions

Three export functions: patches (2a), instances (2b), graph (2c).

**Files:**
- Modify: `src/industrial_point_labeler/annotate/data.py`
- Modify: `tests/test_data.py`

**Step 1: Write tests**

```python
# append to tests/test_data.py
from industrial_point_labeler.annotate.data import export_patches, export_instances, export_graph


def test_export_patches(sample_cloud, tmp_path):
    xyz, rgb, labels = sample_cloud
    _, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    class_map = {"pipe": 0, "tank": 1}
    session = {
        "confirmed": [
            {"gt_id": 0, "label": "pipe", "source_segment_ids": [0]},
            {"gt_id": 1, "label": "tank", "source_segment_ids": [1, 2]},
        ],
    }

    meta = export_patches(session, labels, pim, tmp_path, class_map)

    assert (tmp_path / "patch_ids.npy").exists()
    assert (tmp_path / "patch_classes.npy").exists()
    assert (tmp_path / "patch_metadata.json").exists()

    patch_ids = np.load(tmp_path / "patch_ids.npy")
    assert patch_ids.shape == (300,)
    assert set(patch_ids[labels == 0]) == {0}
    assert set(patch_ids[labels == 1]) == {1}
    assert meta["n_patches"] == 2


def test_export_instances(sample_cloud, tmp_path):
    xyz, rgb, labels = sample_cloud
    _, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    class_map = {"pipe": 0, "tank": 1}
    session = {
        "instances": [
            {"instance_id": 0, "label": "pipe", "patch_ids": [0, 1]},
            {"instance_id": 1, "label": "tank", "patch_ids": [2]},
        ],
    }

    meta = export_instances(session, labels, pim, tmp_path, class_map)

    assert (tmp_path / "instance_ids.npy").exists()
    assert (tmp_path / "instance_classes.npy").exists()
    assert (tmp_path / "instance_metadata.json").exists()
    assert meta["n_instances"] == 2


def test_export_graph(tmp_path):
    edges = [
        {"source": 0, "target": 1, "min_dist": 0.05, "n_close_pairs": 10},
    ]
    session = {
        "reviewed": [
            {"source": 0, "target": 1, "action": "accept"},
        ],
    }

    export_graph(edges, session, tmp_path, "abc123")

    assert (tmp_path / "reviewed_edges.json").exists()
    import json
    with open(tmp_path / "reviewed_edges.json") as f:
        data = json.load(f)
    assert data["summary"]["accepted"] == 1
    assert data["labels_fingerprint"] == "abc123"
```

**Step 2: Run to verify fail**

Run: `python -m pytest tests/test_data.py -k export -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Append to `annotate/data.py`:

```python
import json
from pathlib import Path


def export_patches(session, labels, point_indices_map, output_dir, class_map):
    """Save patch-level outputs from sub-stage 2a.

    Writes patch_ids.npy, patch_classes.npy, patch_metadata.json.
    Returns metadata dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = len(labels)
    patch_ids = np.full(N, -1, dtype=np.int32)
    patch_classes = np.full(N, -1, dtype=np.int32)

    confirmed = session.get("confirmed", [])
    for gt_seg in confirmed:
        gt_id = gt_seg["gt_id"]
        class_id = class_map.get(gt_seg["label"], -1)
        for src_id in gt_seg["source_segment_ids"]:
            if src_id in point_indices_map:
                idx = point_indices_map[src_id]
                patch_ids[idx] = gt_id
                patch_classes[idx] = class_id

    np.save(output_dir / "patch_ids.npy", patch_ids)
    np.save(output_dir / "patch_classes.npy", patch_classes)

    meta = {
        "n_points": N,
        "n_patches": len(confirmed),
        "n_labeled_points": int((patch_ids >= 0).sum()),
        "class_map": class_map,
        "patches": [
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
    with open(output_dir / "patch_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def export_instances(session, labels, point_indices_map, output_dir, class_map):
    """Save instance-level outputs from sub-stage 2b.

    Writes instance_ids.npy, instance_classes.npy, instance_metadata.json.
    Returns metadata dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = len(labels)
    instance_ids = np.full(N, -1, dtype=np.int32)
    instance_classes = np.full(N, -1, dtype=np.int32)

    instances = session.get("instances", [])
    for inst in instances:
        inst_id = inst["instance_id"]
        class_id = class_map.get(inst["label"], -1)
        for patch_id in inst["patch_ids"]:
            if patch_id in point_indices_map:
                idx = point_indices_map[patch_id]
                instance_ids[idx] = inst_id
                instance_classes[idx] = class_id

    np.save(output_dir / "instance_ids.npy", instance_ids)
    np.save(output_dir / "instance_classes.npy", instance_classes)

    meta = {
        "n_points": N,
        "n_instances": len(instances),
        "n_labeled_points": int((instance_ids >= 0).sum()),
        "class_map": class_map,
        "instances": [
            {
                "instance_id": inst["instance_id"],
                "label": inst["label"],
                "class_id": class_map.get(inst["label"], -1),
                "patch_ids": inst["patch_ids"],
                "n_points": int(sum(
                    len(point_indices_map.get(pid, []))
                    for pid in inst["patch_ids"]
                )),
            }
            for inst in instances
        ],
    }
    with open(output_dir / "instance_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def export_graph(edges, session, output_dir, labels_fingerprint):
    """Save reviewed connectivity graph from sub-stage 2c.

    Writes reviewed_edges.json.
    """
    import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviewed = session.get("reviewed", [])
    reviewed_map = {(r["source"], r["target"]): r["action"] for r in reviewed}

    export_edges = []
    for i, edge in enumerate(edges):
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
        "labels_fingerprint": labels_fingerprint,
    }

    with open(output_dir / "reviewed_edges.json", "w") as f:
        json.dump(output, f, indent=2)

    return output["summary"]
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: all 12 tests PASS

**Step 5: Commit**

```bash
git add src/industrial_point_labeler/annotate/data.py tests/test_data.py
git commit -m "feat(annotate): add export functions (patches, instances, graph)"
```

---

### Task 6: `server.py` — core handler and stage routing

Write the `AnnotateHandler` class with stage-aware `do_GET`/`do_POST` routing and the shared utility methods (serve HTML, serve JSON, read POST body).

**Files:**
- Modify: `src/industrial_point_labeler/annotate/server.py`

**Step 1: Implement the handler skeleton**

Reference: `labeler/server.py:172-298` for handler pattern, `graph/server.py:98-318` for graph routes.

```python
# annotate/server.py
"""Unified HTTP server for the annotation tool (stages 2a, 2b, 2c)."""

import datetime
import json
import urllib.parse
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np


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
            self._require_stage("label")
            self._serve_html("splitter.html")
        elif self.path.startswith("/split-data"):
            self._require_stage("label")
            self._serve_split_data()
        elif self.path == "/graph-data":
            self._require_stage("graph")
            self._serve_graph_data()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            self._handle_save()
        elif self.path == "/apply-split":
            self._require_stage("label")
            self._handle_apply_split()
        elif self.path == "/finalize-patches":
            self._require_stage("label")
            self._handle_finalize_patches()
        elif self.path == "/finalize-instances":
            self._require_stage("instance")
            self._handle_finalize_instances()
        elif self.path == "/add-edge":
            self._require_stage("graph")
            self._handle_add_edge()
        elif self.path == "/review-edge":
            self._require_stage("graph")
            self._handle_review_edge()
        elif self.path == "/export-graph":
            self._require_stage("graph")
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

    # --- Route handlers (stubs — implemented in subsequent tasks) ---

    def _serve_data(self):
        raise NotImplementedError

    def _serve_split_data(self):
        raise NotImplementedError

    def _serve_graph_data(self):
        raise NotImplementedError

    def _handle_save(self):
        raise NotImplementedError

    def _handle_apply_split(self):
        raise NotImplementedError

    def _handle_finalize_patches(self):
        raise NotImplementedError

    def _handle_finalize_instances(self):
        raise NotImplementedError

    def _handle_add_edge(self):
        raise NotImplementedError

    def _handle_review_edge(self):
        raise NotImplementedError

    def _handle_export_graph(self):
        raise NotImplementedError
```

**Step 2: Verify it parses**

Run: `python -c "from industrial_point_labeler.annotate.server import AnnotateHandler; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/server.py
git commit -m "feat(annotate): add handler skeleton with stage-aware routing"
```

---

### Task 7: `server.py` — /data and /save handlers

Implement the two routes that work across all stages.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/server.py`

**Step 1: Implement _serve_data**

`/data` returns different payloads based on stage. Reference: `labeler/server.py:226-243`, `graph/server.py:151-166`.

Replace the `_serve_data` stub:

```python
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
            # Use pre-cached static JSON for graph segments (avoids re-serializing MBs)
            self._serve_graph_data()
```

**Step 2: Implement _handle_save**

Replace the `_handle_save` stub. Reference: `labeler/server.py:245-281`.

```python
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
```

**Step 3: Verify it parses**

Run: `python -c "from industrial_point_labeler.annotate.server import AnnotateHandler; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/industrial_point_labeler/annotate/server.py
git commit -m "feat(annotate): implement /data and /save handlers"
```

---

### Task 8: `server.py` — split and finalize handlers

Implement `/split-data`, `/apply-split`, `/finalize-patches`, `/finalize-instances`.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/server.py`

**Step 1: Implement split handlers**

Replace stubs. These call into `data.py` functions.

```python
    def _serve_split_data(self):
        # Parse segment IDs from query string: /split-data?segments=1,2,3
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        seg_ids_str = params.get("segments", [""])[0]
        if not seg_ids_str:
            self._send_json(400, {"error": "Missing segments parameter"})
            return

        segment_ids = [int(s) for s in seg_ids_str.split(",")]
        from industrial_point_labeler.annotate.data import get_split_data

        data = get_split_data(
            self.app["xyz"], self.app["rgb"],
            self.app["point_indices_map"], segment_ids,
        )
        self._send_json(200, data)

    def _handle_apply_split(self):
        from industrial_point_labeler.annotate.data import apply_split

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
        from industrial_point_labeler.annotate.data import prepare_segments

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
```

**Step 2: Implement finalize handlers**

```python
    def _handle_finalize_patches(self):
        from industrial_point_labeler.annotate.data import export_patches

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
        from industrial_point_labeler.annotate.data import export_instances

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
        from industrial_point_labeler.annotate.data import prepare_graph_segments

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
```

**Step 3: Verify it parses**

Run: `python -c "from industrial_point_labeler.annotate.server import AnnotateHandler; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/industrial_point_labeler/annotate/server.py
git commit -m "feat(annotate): implement split and finalize handlers"
```

---

### Task 9: `server.py` — graph handlers

Implement `/graph-data`, `/add-edge`, `/review-edge`, `/export-graph`.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/server.py`

**Step 1: Implement graph route handlers**

Reference: `graph/server.py:151-318`.

Replace the stubs:

```python
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
        # Store review in session
        reviewed = self.app["session"].setdefault("graph_reviewed", [])
        reviewed.append(req)

        # Auto-save session
        out_dir = self.app["output_dir"]
        with open(out_dir / "gt_session.json", "w") as f:
            json.dump(self.app["session"], f, indent=2)

        self._send_json(200, {"ok": True})

    def _handle_export_graph(self):
        from industrial_point_labeler.annotate.data import export_graph

        summary = export_graph(
            self.app["edges"],
            self.app["session"],
            self.app["output_dir"],
            self.app["labels_fingerprint"],
        )
        self._send_json(200, {"ok": True, "summary": summary})
        print(f"  Exported graph: {summary['accepted']} accepted, "
              f"{summary['rejected']} rejected, {summary['pending']} pending")
```

**Step 2: Verify it parses**

Run: `python -c "from industrial_point_labeler.annotate.server import AnnotateHandler; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/server.py
git commit -m "feat(annotate): implement graph route handlers"
```

---

### Task 10: `server.py` — main() with startup and stage resumption

Write the `main()` function that loads data, resumes at the correct stage, and starts the server.

**Files:**
- Modify: `src/industrial_point_labeler/annotate/server.py`

**Step 1: Implement main()**

Reference: `labeler/server.py:301-457` for startup pattern with fingerprint validation.

```python
# Add to end of server.py

import argparse
import sys
import webbrowser
from functools import partial
from http.server import HTTPServer

from industrial_point_labeler.config import load_classes
from industrial_point_labeler.io import labels_fingerprint, load_ply
from industrial_point_labeler.annotate.data import (
    prepare_segments, prepare_graph_segments,
)


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
```

**Step 2: Verify it parses and --help works**

Run: `python -m industrial_point_labeler.annotate.server --help`
Expected: prints argparse help with `project_dir`, `--points`, `--labels`, etc.

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/server.py
git commit -m "feat(annotate): implement main() with startup and stage resumption"
```

---

### Task 11: `viewer.html` — label and instance merge stages

Port from `labeler/viewer.html` (1302 lines). Write a single HTML file that handles both 2a (label+split) and 2b (instance merge) stages. The `stage` field from `/data` response controls which UI elements are visible.

**Files:**
- Create: `src/industrial_point_labeler/annotate/viewer.html`

**Step 1: Port the labeler viewer**

Copy `labeler/viewer.html` as the starting point. Then make these modifications:

1. **Rename title** from "GT Annotation Tool" to "Annotation Tool"

2. **Read `stage` from `/data` response** — add to the `init()` function:
   ```javascript
   const stage = data.session.stage || 'label';
   ```

3. **Add stage-conditional toolbar controls:**
   - If `stage === 'label'`: show "Split Selected" button, "Refresh" button, "Finalize Patches" button
   - If `stage === 'instance'`: show "Finalize Instances" button
   - Both stages share the rest of the toolbar (color toggle, point size, export)

4. **"Split Selected" button** (2a only):
   ```javascript
   function splitSelected() {
       const ids = Array.from(selectedSegments).join(',');
       if (!ids) return alert('Select segments first');
       window.open('/split?segments=' + ids, '_blank');
   }
   ```

5. **"Refresh" button** (2a only): re-fetches `/data` and rebuilds the scene.

6. **"Finalize Patches" button** (2a only):
   ```javascript
   async function finalizePatches() {
       if (!confirm('Finalize patches? This advances to instance merge stage.')) return;
       const resp = await fetch('/finalize-patches', {method: 'POST'});
       const data = await resp.json();
       if (data.ok) location.reload();
   }
   ```

7. **"Finalize Instances" button** (2b only):
   ```javascript
   async function finalizeInstances() {
       if (!confirm('Finalize instances? This builds the graph and advances to graph review.')) return;
       const resp = await fetch('/finalize-instances', {method: 'POST'});
       const data = await resp.json();
       if (data.ok) location.reload();
   }
   ```

8. **Sidebar for 2b**: when `stage === 'instance'`, group the sidebar segments by class instead of showing raw segment list. The confirmed patches become the selectable units.

9. **Same-class enforcement in 2b**: when confirming a merge in instance stage, check all selected patches share the same class. If not, show an alert.

**Step 2: Verify by running the server**

Run: `python -m industrial_point_labeler.annotate.server --points <test.ply> --labels <test_labels.npy> --port 8766`
Open browser, verify:
- Stage "label": see segments, can select, toolbar shows Split/Refresh/Finalize Patches
- After finalize: reloads into "instance" stage with Finalize Instances button

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/viewer.html
git commit -m "feat(annotate): add viewer.html for label and instance stages"
```

---

### Task 12: `splitter.html` — OBB split tool

Port from `doubles/viewer.html` (908 lines). Re-wire data source from local PLY files to the annotation server.

**Files:**
- Create: `src/industrial_point_labeler/annotate/splitter.html`

**Step 1: Port the doubles viewer**

Copy `doubles/viewer.html` as starting point. Make these modifications:

1. **Data loading**: replace the manifest-based loading with a fetch to `/split-data`:
   ```javascript
   // Old: fetch('/doubles/manifest.json') then load per-segment PLY files
   // New:
   const params = new URLSearchParams(window.location.search);
   const segIds = params.get('segments');
   const resp = await fetch('/split-data?segments=' + segIds);
   const data = await resp.json();
   // data.x, data.y, data.z, data.r, data.g, data.b, data.segment_ids
   ```

2. **Save handler**: replace `/save_splits` POST with `/apply-split`:
   ```javascript
   async function saveSplits() {
       const result = {
           original_segment_ids: segmentIds,
           groups: buildGroupAssignments(),
       };
       const resp = await fetch('/apply-split', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify(result),
       });
       const data = await resp.json();
       if (data.ok) {
           alert('Split applied. Close this tab and click Refresh in the main tool.');
           window.close();
       }
   }
   ```

3. **Remove sidebar segment list** — the splitter only works on the segments passed via query params, no need for a segment picker.

4. **Keep all OBB interaction code** — box drawing, corner dragging, group assignment, WASD movement, keyboard shortcuts.

**Step 2: Verify by testing the split flow**

Run the annotation server, select segments, click "Split Selected", verify the splitter tab opens, loads point data, can draw boxes, and save applies correctly.

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/splitter.html
git commit -m "feat(annotate): add splitter.html (OBB split tool)"
```

---

### Task 13: `graph_viewer.html` — connection graph review

Port from `graph/viewer.html` (1806 lines). Minimal changes — mostly route URLs.

**Files:**
- Create: `src/industrial_point_labeler/annotate/graph_viewer.html`

**Step 1: Port the graph viewer**

Copy `graph/viewer.html` as starting point. Make these modifications:

1. **Data URL**: change `/data` fetch to `/graph-data`:
   ```javascript
   const resp = await fetch('/graph-data');
   ```

2. **Save URL**: keep `/save` (unchanged).

3. **Export URL**: change from `/export` to `/export-graph`:
   ```javascript
   const resp = await fetch('/export-graph', {method: 'POST'});
   ```

4. **Review edge URL**: add POST to `/review-edge` on each A/R/S action:
   ```javascript
   async function reviewEdge(source, target, action) {
       await fetch('/review-edge', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify({source, target, action}),
       });
   }
   ```

5. **Title**: update to "Annotation Tool — Connection Graph"

6. **Everything else stays**: split-view layout, force-directed graph, edge sidebar, keyboard shortcuts, 3D navigation.

**Step 2: Verify by testing graph stage**

Run the server with a session at "graph" stage, verify the split-view loads, edges are listed, A/R/S works, export works.

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/annotate/graph_viewer.html
git commit -m "feat(annotate): add graph_viewer.html for connection graph review"
```

---

### Task 14: Update CLI

Replace old commands with `annotate`.

**Files:**
- Modify: `src/industrial_point_labeler/cli.py`

**Step 1: Update command map**

```python
# cli.py
"""Unified CLI for the industrial point labeler pipeline."""

import sys


def main():
    commands = {
        "segment": ("industrial_point_labeler.segmentation.ransac", "RANSAC primitive segmentation"),
        "annotate": ("industrial_point_labeler.annotate.server", "Interactive annotation tool"),
        "describe-equipment": ("industrial_point_labeler.equipment.server", "Equipment description tool"),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: ipl <command> [options]\n")
        print("Industrial Point Labeler — interactive annotation toolkit\n")
        print("commands:")
        for cmd, (_, desc) in commands.items():
            print(f"  {cmd:<20s} {desc}")
        print(f"\nRun 'ipl <command> --help' for command-specific options.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    module_path, _ = commands[cmd]
    sys.argv = [f"ipl {cmd}"] + sys.argv[2:]

    import importlib
    mod = importlib.import_module(module_path)
    mod.main()


if __name__ == "__main__":
    main()
```

**Step 2: Verify**

Run: `ipl --help`
Expected: shows `segment`, `annotate`, `describe-equipment`

Run: `ipl annotate --help`
Expected: shows annotation tool argparse help

**Step 3: Commit**

```bash
git add src/industrial_point_labeler/cli.py
git commit -m "feat(cli): replace old commands with unified annotate"
```

---

### Task 15: Remove "double" class from config

The design removes the "double" concept entirely.

**Files:**
- Modify: `config/classes.yaml`

**Step 1: Remove double class**

Remove the `double` entry from `classes` and any references.

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

**Step 2: Commit**

```bash
git add config/classes.yaml
git commit -m "config: remove double class (splitting is now inline in annotate)"
```

---

### Task 16: Delete old modules

Remove `labeler/`, `doubles/`, `graph/server.py`, `graph/viewer.html`, `graph/__init__.py`. Keep `graph/builder.py`.

**Files:**
- Delete: `src/industrial_point_labeler/labeler/` (entire directory)
- Delete: `src/industrial_point_labeler/doubles/` (entire directory)
- Delete: `src/industrial_point_labeler/graph/server.py`
- Delete: `src/industrial_point_labeler/graph/viewer.html`
- Delete: `src/industrial_point_labeler/graph/__init__.py`

**Step 1: Verify new module works first**

Run the full flow: `ipl annotate --points <test.ply> --labels <test_labels.npy>`
Verify all three stages work end-to-end in the browser.

**Step 2: Delete old files**

```bash
rm -rf src/industrial_point_labeler/labeler/
rm -rf src/industrial_point_labeler/doubles/
rm src/industrial_point_labeler/graph/server.py
rm src/industrial_point_labeler/graph/viewer.html
rm src/industrial_point_labeler/graph/__init__.py
```

**Step 3: Verify nothing imports the old modules**

Run: `python -c "import industrial_point_labeler; print('OK')"`
Run: `python -m pytest tests/ -v`

**Step 4: Commit**

```bash
git add -A
git commit -m "cleanup: remove old labeler, doubles, graph modules"
```

---

### Task 17: End-to-end verification

Run through the full pipeline with real data to verify everything works.

**Steps:**

1. Start at label stage: `ipl annotate --points scan.ply --labels instance_labels.npy --summary ransac_summary.json --mesh mesh.glb`
2. In browser:
   - Select segments, assign classes, confirm
   - Click "Split Selected" on a multi-object segment → verify splitter tab opens
   - Draw OBB boxes, save → verify new segments appear after "Refresh"
   - Click "Finalize Patches" → verify page reloads into instance stage
3. In instance stage:
   - Select patches of same class, confirm merge
   - Click "Finalize Instances" → verify graph builds and page reloads into graph stage
4. In graph stage:
   - Review edges with A/R/S
   - Add manual edge with Ctrl+Click
   - Export graph
5. Verify output files exist in `annotation/`:
   - `patch_ids.npy`, `patch_classes.npy`, `patch_metadata.json`
   - `instance_ids.npy`, `instance_classes.npy`, `instance_metadata.json`
   - `adjacency_graph.json`, `reviewed_edges.json`
6. Stop server, restart → verify it resumes at the correct stage
