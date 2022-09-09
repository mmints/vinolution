#!/usr/bin/env python3

import open3d as o3d

hill = o3d.io.read_triangle_mesh("meshes/simple-hill.ply")
hill.compute_vertex_normals()

hill_erosion = o3d.io.read_triangle_mesh("meshes/simple-hill-erosion.ply")
hill_erosion.compute_vertex_normals()


o3d.visualization.draw_geometries([hill])
o3d.visualization.draw_geometries([hill_erosion])


hill_pc = hill.sample_points_uniformly(number_of_points=5000)
hill_erosion_pc = hill_erosion.sample_points_uniformly(number_of_points=5000)


hill_pc.paint_uniform_color([1, 0.706, 0])
hill_erosion_pc.paint_uniform_color([0, 0.651, 0.929])

o3d.visualization.draw_geometries([hill_pc, hill_erosion_pc])