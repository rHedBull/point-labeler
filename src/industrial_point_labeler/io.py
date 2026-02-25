"""Shared I/O utilities for loading point clouds and labels."""

import hashlib

import numpy as np
from plyfile import PlyData


def load_ply(path):
    """Load PLY file, return (xyz, rgb) arrays. RGB defaults to gray if missing."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.column_stack([v["x"], v["y"], v["z"]])
    names = {p.name for p in v.properties}
    if {"red", "green", "blue"} <= names:
        rgb = np.column_stack([v["red"], v["green"], v["blue"]])
    else:
        rgb = np.full((len(v.data), 3), 180, dtype=np.uint8)
    return xyz, rgb


def load_ply_xyz(path):
    """Load PLY file, return xyz only."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]])


def labels_fingerprint(labels):
    """SHA256 of raw label bytes, truncated to 16 hex chars."""
    return hashlib.sha256(labels.tobytes()).hexdigest()[:16]
