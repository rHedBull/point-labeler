# Example Data

Place your input files here:

- `scan.ply` — PLY point cloud with `x,y,z,red,green,blue` vertex properties
- `mesh.glb` — (optional) GLB mesh for split-view overlay

Then run:
    ipl segment --input scan.ply --output output/
    ipl label --points scan.ply --labels output/instance_labels.npy --output gt/
