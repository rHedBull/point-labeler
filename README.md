# Industrial Point Labeler

Interactive segmentation and annotation toolkit for industrial LiDAR point clouds.

## Install

```bash
pip install -e .

# With RANSAC segmentation support (requires Open3D):
pip install -e ".[segmentation]"
```

## Pipeline

| Stage | Command | Description |
|-------|---------|-------------|
| 1 | `ipl segment` | RANSAC primitive fitting (planes, cylinders, spheres) |
| 2 | `ipl label` | Interactive merge-based GT annotation |
| 3 | `ipl doubles` | Dissect overlapping/double-scanned segments |
| 4a | `ipl build-graph` | Build pipe-endpoint connectivity graph |
| 4b | `ipl review-graph` | Interactive graph edge review |
| 5 | `ipl describe-equipment` | Annotate equipment segments with descriptions |

## Quick Start

```bash
# 1. Segment a point cloud
ipl segment --input scan.ply --output output/

# 2. Label the segments interactively
ipl label --points scan.ply --labels output/instance_labels.npy --output-dir gt/

# 3. Build and review connectivity graph
ipl build-graph --points scan.ply --labels gt/gt_labels.npy \
    --metadata gt/gt_metadata.json --output gt/graph.json
ipl review-graph --points scan.ply --labels gt/gt_labels.npy --graph gt/graph.json
```

## Configuration

Edit `config/classes.yaml` to define your own class labels, colors, and keyboard shortcuts.

## Input Format

- Point cloud: PLY with `x, y, z, red, green, blue` vertex properties
- Mesh (optional): GLB format for split-view overlay
