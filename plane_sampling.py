#!/usr/bin/env python3

import open3d as o3d

simple_plane = o3d.io.read_triangle_mesh("meshes/simple-plane.ply")
simple_plane.compute_vertex_normals()

simple_hill = o3d.io.read_triangle_mesh("meshes/simple-hill-rinnen-combined.ply")
simple_hill.compute_vertex_normals()


o3d.visualization.draw_geometries([simple_plane])
o3d.visualization.draw_geometries([simple_hill])


simple_plane_pc = simple_plane.sample_points_uniformly(number_of_points=5000)
simple_hill_pc = simple_hill.sample_points_uniformly(number_of_points=5000)


simple_plane_pc.paint_uniform_color([1, 0.706, 0])
simple_hill_pc.paint_uniform_color([0, 0.651, 0.929])


o3d.visualization.draw_geometries([simple_hill_pc, simple_plane_pc])

