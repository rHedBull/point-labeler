# Industrial Point Labeler

Interactive segmentation and annotation toolkit for industrial LiDAR point clouds.

## Install

```bash
# Recommended (using uv):
uv pip install -e ".[segmentation]"

# Or with pip:
pip install -e ".[segmentation]"
```

The `segmentation` extra pulls in Open3D and scikit-learn for RANSAC primitive fitting.

## Pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| 1 | `ipl segment` | RANSAC primitive fitting (planes, cylinders, spheres) |
| 2 | `ipl annotate` | Unified interactive annotation (label, split, instance export, graph review) |
| 3 | `ipl describe-equipment` | Annotate equipment segments with descriptions |

## Quick Start

```bash
# 1. Segment a point cloud
ipl segment --input scan.ply --output output/

# 2. Annotate interactively (opens browser at localhost:8766)
ipl annotate --points scan.ply --labels output/instance_labels.npy

# With optional RANSAC summary and custom output directory:
ipl annotate --points scan.ply --labels output/instance_labels.npy \
    --summary output/ransac_summary.json --output-dir annotation/

# Resume a previous session:
ipl annotate --points scan.ply --labels output/instance_labels.npy \
    --resume annotation/gt_session.json

# 3. Describe equipment (after annotation is finalized)
ipl describe-equipment --points scan.ply --labels annotation/gt_labels.npy \
    --summary output/ransac_summary.json --graph annotation/graph.json
```

## Configuration

Edit `config/classes.yaml` to define your own class labels, colors, and keyboard shortcuts.

## Input Format

- Point cloud: PLY with `x, y, z, red, green, blue` vertex properties
- Mesh (optional): GLB format for split-view overlay
