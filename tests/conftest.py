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
