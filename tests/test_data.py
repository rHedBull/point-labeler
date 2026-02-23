"""Tests for annotate/data.py."""

import numpy as np
from industrial_point_labeler.annotate.data import prepare_segments


def test_prepare_segments_basic(sample_cloud):
    xyz, rgb, labels = sample_cloud
    segments, context, center, extent, pim = prepare_segments(xyz, rgb, labels)

    assert len(segments) == 3
    assert all(s["id"] in (0, 1, 2) for s in segments)
    assert all("x" in s and "y" in s and "z" in s for s in segments)
    assert all("r" in s and "g" in s and "b" in s for s in segments)
    assert len(pim) == 3
    assert all(isinstance(v, np.ndarray) for v in pim.values())


def test_prepare_segments_skips_small(sample_cloud):
    xyz, rgb, labels = sample_cloud
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
    assert len(segments[0]["x"]) == 2000


def test_prepare_segments_context_cloud_limit():
    rng = np.random.default_rng(42)
    n = 80000
    xyz = rng.normal(0, 1, (n, 3))
    rgb = rng.integers(0, 255, (n, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int32)
    _, context, _, _, _ = prepare_segments(xyz, rgb, labels)
    assert len(context["x"]) == 50000


def test_prepare_segments_with_summary(sample_cloud):
    xyz, rgb, labels = sample_cloud
    summary = [{"id": 0, "type": "cylinder", "radius": 0.05}]
    segments, _, _, _, _ = prepare_segments(xyz, rgb, labels, summary=summary)
    seg0 = next(s for s in segments if s["id"] == 0)
    assert seg0["type"] == "cylinder"
    assert seg0["radius"] == 0.05


from industrial_point_labeler.annotate.data import prepare_graph_segments


def test_prepare_graph_segments_filters_by_nodes(sample_cloud):
    xyz, rgb, labels = sample_cloud
    nodes = [
        {"id": 0, "label": "pipe"},
        {"id": 2, "label": "tank"},
    ]
    segments, context, center, extent = prepare_graph_segments(xyz, rgb, labels, nodes)
    seg_ids = {s["id"] for s in segments}
    assert seg_ids == {0, 2}
    assert all(s["label"] in ("pipe", "tank") for s in segments)


from industrial_point_labeler.annotate.data import get_split_data, apply_split


def test_get_split_data_returns_full_res(sample_cloud):
    xyz, rgb, labels = sample_cloud
    _, _, _, _, pim = prepare_segments(xyz, rgb, labels)
    data = get_split_data(xyz, rgb, pim, [0, 1])
    total = len(pim[0]) + len(pim[1])
    assert len(data["x"]) == total
    assert len(data["segment_ids"]) == total


def test_apply_split_creates_new_segments(sample_cloud):
    xyz, rgb, labels = sample_cloud
    labels = labels.copy()
    segments, _, _, _, pim = prepare_segments(xyz, rgb, labels)

    indices_0 = pim[0].copy()
    split_result = {
        "original_segment_ids": [0],
        "groups": [
            {"group": "A", "point_indices": list(range(50))},
            {"group": "B", "point_indices": list(range(50, 100))},
        ],
    }

    new_segs, next_id = apply_split(labels, pim, segments, split_result, next_segment_id=100)

    assert 0 not in pim
    assert 100 in pim
    assert 101 in pim
    assert len(pim[100]) == 50
    assert len(pim[101]) == 50
    assert next_id == 102
    assert np.all(labels[indices_0[:50]] == 100)
    assert np.all(labels[indices_0[50:]] == 101)


def test_apply_split_remainder(sample_cloud):
    xyz, rgb, labels = sample_cloud
    labels = labels.copy()
    segments, _, _, _, pim = prepare_segments(xyz, rgb, labels)

    split_result = {
        "original_segment_ids": [0],
        "groups": [
            {"group": "A", "point_indices": list(range(30))},
        ],
    }

    new_segs, next_id = apply_split(labels, pim, segments, split_result, next_segment_id=100)

    assert 100 in pim
    assert 101 in pim
    assert len(pim[100]) == 30
    assert len(pim[101]) == 70


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
