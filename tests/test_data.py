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
