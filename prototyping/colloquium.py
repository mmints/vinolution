#!/usr/bin/env python3

from tkinter import N
import open3d as o3d
import numpy as np
from dataclasses import dataclass
import tools
from change_detection import *

def cluster(pcd, eps_in, min_points_in):
    labeled = copy.deepcopy(pcd)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(labeled.cluster_dbscan(eps = eps_in, min_points = min_points_in, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    labeled.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return labeled

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

print("Perform change detection")
result = detect_change(after, before, 0.01)
result.paint_uniform_color([1, 0, 0])

result2 = detect_change(before, after, 0.01)
result2.paint_uniform_color([0, 1, 0])


print("Start outlier removal with nb_neighbors")
cl, ind = result.remove_radius_outlier(nb_points = 10, radius = 0.05, print_progress = True)
print("remove_outlier: DONE!")
result_out = result.select_by_index(ind)

print("Start outlier removal with nb_neighbors")
cl, ind = result2.remove_radius_outlier(nb_points = 10, radius = 0.05, print_progress = True)
print("remove_outlier: DONE!")
result2_out = result2.select_by_index(ind)

difference_pc_out_lab = cluster(difference_pc_out, 0.5, 2)
difference_pc_out2_lab = cluster(difference_pc2_out, 0.5, 2)
result_lab = cluster(result_out, 0.05, 5)
result2_lab = cluster(result2_out, 0.05, 5)

src = o3d.io.read_point_cloud("../data_sets/dlr-arena_08-08-2022-downsample-x4-row.xyz")
dst = o3d.io.read_point_cloud("../data_sets/dlr-arena_06-07-2022-downsample-x4-row.xyz")
print("Start outlier removal with nb_neighbors")
cl, ind = src.remove_radius_outlier(nb_points = 10, radius = 0.1, print_progress = True)
print("remove_outlier: DONE!")
src = src.select_by_index(ind)
print("Start outlier removal with nb_neighbors")
cl, ind = dst.remove_radius_outlier(nb_points = 10, radius = 0.1, print_progress = True)
print("remove_outlier: DONE!")
dst = dst.select_by_index(ind)

src.paint_uniform_color([1, 0.706, 0])
dst.paint_uniform_color([0, 0.651, 0.929])

res_row = detect_change(dst, src, 0.15)
res_row.paint_uniform_color([1, 0, 0])
res_row2 = detect_change(src, dst, 0.15)
res_row2.paint_uniform_color([0, 1, 0])
change = tools.concatenate_point_clouds([res_row, res_row2]) # before is a reference to pc_0
print("Start outlier removal with nb_neighbors")
cl, ind = change.remove_radius_outlier(nb_points = 10, radius = 0.1, print_progress = True)
print("remove_outlier: DONE!")
change = change.select_by_index(ind)
res_row_lab = cluster(change, 0.05, 5)

# o3d.visualization.draw_geometries([hill_pc])
# o3d.visualization.draw_geometries([hill_erosion_pc])
# o3d.visualization.draw_geometries([difference_pc])
# o3d.visualization.draw_geometries([difference_pc2])
# o3d.visualization.draw_geometries([difference_pc_out])
# o3d.visualization.draw_geometries([difference_pc2_out])
# o3d.visualization.draw_geometries([difference_pc_out, difference_pc2_out])
# o3d.visualization.draw_geometries([difference_pc_out_lab, difference_pc_out2_lab])
# o3d.visualization.draw_geometries([hill_pc, difference_pc_out_lab,difference_pc_out2_lab])
# o3d.visualization.draw_geometries([hill_erosion_pc, difference_pc_out_lab,difference_pc_out2_lab])


o3d.visualization.draw_geometries([before])
o3d.visualization.draw_geometries([after])
# o3d.visualization.draw_geometries([result])
# o3d.visualization.draw_geometries([result2])
# o3d.visualization.draw_geometries([result_out])
# o3d.visualization.draw_geometries([result2_out])
o3d.visualization.draw_geometries([result_out, result2_out])
o3d.visualization.draw_geometries([result_lab, result2_lab])
o3d.visualization.draw_geometries([before, result_lab, result2_lab])
# o3d.visualization.draw_geometries([after, result_lab, result2_lab])

o3d.visualization.draw_geometries([res_row_lab,dst])
