#!/usr/bin/env python3

from tkinter import N
import open3d as o3d
import numpy as np

hill = o3d.io.read_triangle_mesh("../meshes/simple-hill.ply")
hill.compute_vertex_normals()

hill_erosion = o3d.io.read_triangle_mesh("../meshes/simple-hill-erosion3.ply")
hill_erosion.compute_vertex_normals()

hill_pc = hill.sample_points_uniformly(number_of_points=30000)
hill_erosion_pc = hill_erosion.sample_points_uniformly(number_of_points=30000)

hill_pc.paint_uniform_color([1, 0.706, 0])
hill_erosion_pc.paint_uniform_color([0, 0.651, 0.929])

#o3d.visualization.draw_geometries([hill_pc, hill_erosion_pc])

dists = hill_erosion_pc.compute_point_cloud_distance(hill_pc)
#dists = np.asarray(dists)
print(np.asarray(dists).size)
print(np.asarray(hill_pc.points))

difference_idx = np.argwhere(np.asarray(dists) >= 0.1)
print(np.asarray(difference_idx).size)

print(difference_idx)

difference_pc = hill_erosion_pc.select_by_index(difference_idx)
difference_pc.paint_uniform_color([1.0, 0.0, 0.0])


print("Start outlier removal with nb_neighbors")
cl, ind = difference_pc.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
print("remove_outlier: DONE!")

print(len(ind))

difference_pc = difference_pc.select_by_index(ind)

o3d.visualization.draw_geometries([hill])
o3d.visualization.draw_geometries([hill_erosion])

o3d.visualization.draw_geometries([hill_pc, hill_erosion_pc, difference_pc])
o3d.visualization.draw_geometries([hill_pc, difference_pc])

o3d.visualization.draw_geometries([hill_erosion_pc, difference_pc])
o3d.visualization.draw_geometries([hill, difference_pc])

o3d.visualization.draw_geometries([difference_pc])
