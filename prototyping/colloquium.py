#!/usr/bin/env python3

from tkinter import N
import open3d as o3d
import numpy as np
from dataclasses import dataclass
import tools
from change_detection import *

hill = o3d.io.read_triangle_mesh("../meshes/simple-hill.ply")

hill_erosion = o3d.io.read_triangle_mesh("../meshes/simple-hill-erosion3.ply")

hill_pc = hill.sample_points_uniformly(number_of_points=30000)
hill_erosion_pc = hill_erosion.sample_points_uniformly(number_of_points=30000)

hill_pc.paint_uniform_color([1, 0.706, 0])
hill_erosion_pc.paint_uniform_color([0, 0.651, 0.929])

dists = hill_erosion_pc.compute_point_cloud_distance(hill_pc)

difference_idx = np.argwhere(np.asarray(dists) >= 0.1)
print(np.asarray(difference_idx).size)

print(difference_idx)

difference_pc = hill_erosion_pc.select_by_index(difference_idx)
difference_pc.paint_uniform_color([1.0, 0.0, 0.0])

dists2 = hill_pc.compute_point_cloud_distance(hill_erosion_pc)

difference_idx2 = np.argwhere(np.asarray(dists2) >= 0.1)
print(np.asarray(difference_idx2).size)

print(difference_idx2)

difference_pc2 = hill_pc.select_by_index(difference_idx2)
difference_pc2.paint_uniform_color([0.0, 1.0, 0.0])

print("Start outlier removal with nb_neighbors")
cl, ind = difference_pc.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
print("remove_outlier: DONE!")
difference_pc_out = difference_pc.select_by_index(ind)

print("Start outlier removal with nb_neighbors")
cl, ind = difference_pc2.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
print("remove_outlier: DONE!")
difference_pc2_out = difference_pc2.select_by_index(ind)

print("Load Test Files")
# Main
pc_0 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_0.xyz")

# Changes
pc_1 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_1.xyz") # top
pc_2 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_2.xyz") # long weeds
pc_3 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_3.xyz") # short weeds
pc_4 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_4.xyz") # central
pc_5 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_5.xyz") # outer

pc_0.paint_uniform_color([0, 0, 0])
pc_1.paint_uniform_color([0, 1, 0])
pc_2.paint_uniform_color([0, 0, 1])
pc_3.paint_uniform_color([1, 1, 0])
pc_4.paint_uniform_color([1, 0, 1])
pc_5.paint_uniform_color([0, 1, 1])

# Create point clouds for evaluation
change = tools.concatenate_point_clouds([pc_1, pc_2, pc_3, pc_4, pc_5]) # All changes combined. Use for ground truth.
change.paint_uniform_color([0, 1, 0])
before =  tools.concatenate_point_clouds([pc_0,pc_2, pc_3]) # before is a reference to pc_0
after =  tools.concatenate_point_clouds([pc_0, pc_1,  pc_4, pc_5])

before.paint_uniform_color([1, 0.706, 0])
after.paint_uniform_color([0, 0.651, 0.929])

mu, sigma = 0, 0.0  # mean and standard deviation
after = tools.apply_noise(after, mu, sigma)

print("Perform change detection")
result = detect_change(after, before, 0.05)
result.paint_uniform_color([1, 0, 0])

result2 = detect_change(before, after, 0.05)
result2.paint_uniform_color([0, 1, 0])

print(change)
print(result)

o3d.visualization.draw_geometries([hill_pc])
o3d.visualization.draw_geometries([hill_erosion_pc])
o3d.visualization.draw_geometries([difference_pc])
o3d.visualization.draw_geometries([difference_pc2])
o3d.visualization.draw_geometries([difference_pc_out])
o3d.visualization.draw_geometries([difference_pc2_out])
o3d.visualization.draw_geometries([difference_pc, difference_pc2])

o3d.visualization.draw_geometries([before])
o3d.visualization.draw_geometries([after])
o3d.visualization.draw_geometries([result])
o3d.visualization.draw_geometries([result2])
o3d.visualization.draw_geometries([result, result2])