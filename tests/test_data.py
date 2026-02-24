"""Tests for annotate/data.py."""

import json

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


# --- prepare_segments ---


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


# --- prepare_graph_segments ---


def test_prepare_graph_segments_filters_by_nodes(sample_cloud):
    xyz, rgb, labels = sample_cloud
    nodes = [
        {"id": 0, "label": "pipe"},
        {"id": 2, "label": "tank"},
    ]
    segments, context, center, extent = prepare_graph_segments(xyz, rgb, labels, nodes)
    seg_ids = {s["id"] for s in segments}
    assert seg_ids == {0, 2}
    assert segments[0]["label"] in ("pipe", "tank")


# --- split logic ---


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


# --- export functions ---


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
    input_fingerprints = {"instance_labels.npy": "sha256:test123"}

    meta = export_patches(session, labels, pim, tmp_path, class_map, input_fingerprints)

    assert (tmp_path / "patch_ids.v1.npy").exists()
    assert (tmp_path / "patch_classes.v1.npy").exists()
    assert (tmp_path / "patch_metadata.v1.json").exists()

    patch_ids = np.load(tmp_path / "patch_ids.v1.npy")
    assert patch_ids.shape == (300,)
    assert set(patch_ids[labels == 0]) == {0}
    assert set(patch_ids[labels == 1]) == {1}
    assert meta["n_patches"] == 2
    assert meta["input_fingerprints"] == input_fingerprints


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
    input_fingerprints = {"patch_ids.v1.npy": "sha256:abc123"}

    meta = export_instances(session, labels, pim, tmp_path, class_map, input_fingerprints)

    assert (tmp_path / "instance_ids.v1.npy").exists()
    assert (tmp_path / "instance_classes.v1.npy").exists()
    assert (tmp_path / "instance_metadata.v1.json").exists()
    assert meta["n_instances"] == 2
    assert meta["input_fingerprints"] == input_fingerprints


def test_export_graph(tmp_path):
    edges = [
        {"source": 0, "target": 1, "min_dist": 0.05, "n_close_pairs": 10},
        {"source": 1, "target": 2, "min_dist": 0.12, "n_close_pairs": 3},
    ]
    nodes = [
        {"id": 0, "label": "pipe", "center": [0, 0, 0]},
        {"id": 1, "label": "tank", "center": [5, 0, 0]},
        {"id": 2, "label": "equipment", "center": [10, 0, 0]},
    ]
    session = {
        "graph_reviewed": [
            {"source": 0, "target": 1, "action": "accept"},
            {"source": 1, "target": 2, "action": "reject"},
        ],
    }
    input_fingerprints = {"instance_ids.v1.npy": "sha256:abc123"}

    result = export_graph(edges, nodes, session, tmp_path, input_fingerprints)

    assert (tmp_path / "connectivity_graph.v1.json").exists()
    with open(tmp_path / "connectivity_graph.v1.json") as f:
        data = json.load(f)
    # Only accepted edges in output
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == 0
    assert data["edges"][0]["target"] == 1
    assert "action" not in data["edges"][0]
    # All nodes present (including isolated node 2)
    assert len(data["nodes"]) == 3
    assert data["metadata"]["input_fingerprints"] == input_fingerprints


def test_export_patches_version_bump(sample_cloud, tmp_path):
    """Verify same inputs → overwrite, different inputs → version bump."""
    xyz, rgb, labels = sample_cloud
    _, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    class_map = {"pipe": 0}
    session = {"confirmed": [{"gt_id": 0, "label": "pipe", "source_segment_ids": [0]}]}
    fp1 = {"instance_labels.npy": "sha256:aaa"}

    # First export → v1
    meta1 = export_patches(session, labels, pim, tmp_path, class_map, fp1)
    assert meta1["version"] == 1

    # Same fingerprints → still v1 (overwrite)
    meta2 = export_patches(session, labels, pim, tmp_path, class_map, fp1)
    assert meta2["version"] == 1

    # Different fingerprints → v2
    fp2 = {"instance_labels.npy": "sha256:bbb"}
    meta3 = export_patches(session, labels, pim, tmp_path, class_map, fp2)
    assert meta3["version"] == 2
    assert (tmp_path / "patch_ids.v2.npy").exists()
