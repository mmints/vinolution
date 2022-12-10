#!/usr/bin/env python3

from tkinter import N
import open3d as o3d
import numpy as np
import tools
from change_detection import *

hill = o3d.io.read_triangle_mesh("../meshes/simple-hill.ply")
hill.compute_vertex_normals()

hill_erosion = o3d.io.read_triangle_mesh("../meshes/simple-hill-erosion3.ply")
hill_erosion.compute_vertex_normals()

hill_pc = hill.sample_points_uniformly(number_of_points=30000)
hill_erosion_pc = hill_erosion.sample_points_uniformly(number_of_points=30000)

hill_pc.paint_uniform_color([1, 0.706, 0])
hill_erosion_pc.paint_uniform_color([0, 0.651, 0.929])

dists = hill_erosion_pc.compute_point_cloud_distance(hill_pc)
#dists = np.asarray(dists)
# print(np.asarray(dists).size)
# print(np.asarray(hill_pc.points))

difference_idx = np.argwhere(np.asarray(dists) >= 0.1)
print(np.asarray(difference_idx).size)

print(difference_idx)

difference_pc = hill_erosion_pc.select_by_index(difference_idx)
difference_pc.paint_uniform_color([0.0, 1.0, 0.0])

dists2 = hill_pc.compute_point_cloud_distance(hill_erosion_pc)
#dists = np.asarray(dists)
# print(np.asarray(dists2).size)
# print(np.asarray(hill_pc.points))

difference_idx2 = np.argwhere(np.asarray(dists2) >= 0.1)
print(np.asarray(difference_idx2).size)

print(difference_idx2)

difference_pc2 = hill_pc.select_by_index(difference_idx2)
difference_pc2.paint_uniform_color([1.0, 0.0, 0.0])

print("Start outlier removal with nb_neighbors")
cl, ind = difference_pc.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
print("remove_outlier: DONE!")
difference_pc = difference_pc.select_by_index(ind)

print("Start outlier removal with nb_neighbors")
cl, ind = difference_pc2.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
print("remove_outlier: DONE!")
difference_pc2 = difference_pc2.select_by_index(ind)

# tools.draw([hill])
# tools.draw([hill_erosion])

# tools.draw([hill_pc, hill_erosion_pc, difference_pc])
# tools.draw([hill_pc, difference_pc])

# tools.draw([hill_erosion_pc, difference_pc])
# tools.draw([hill, difference_pc])

# tools.draw([difference_pc])
# tools.draw([difference_pc2])

difference_pc_out_lab = clustering(difference_pc, 0.5, 2)
difference_pc_out2_lab = clustering(difference_pc2, 0.5, 2)

tools.draw([difference_pc_out_lab])
tools.draw([difference_pc_out2_lab])