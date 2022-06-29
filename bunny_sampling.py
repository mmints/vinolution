#!/usr/bin/env python3

import open3d as o3d

bunny = o3d.data.BunnyMesh()

print(bunny.data_root)

mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])
pcd = mesh.sample_points_uniformly(number_of_points=5000)
o3d.visualization.draw_geometries([pcd])
