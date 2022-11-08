from dataclasses import dataclass
import open3d as o3d
import numpy as np
import tools
from change_detection import *

if __name__ == "__main__":

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

    # Visualize all raw data
    # o3d.visualization.draw_geometries([pc_0, pc_1, pc_2, pc_3, pc_4, pc_5])

    # Create point clouds for evaluation
    change = tools.concatenate_point_clouds([pc_1, pc_2, pc_3, pc_4, pc_5]) # All changes combined. Use for ground truth.
    change.paint_uniform_color([0, 1, 0])
    before = pc_0 # before is a reference to pc_0
    after =  tools.concatenate_point_clouds([pc_0, change])
    
    # o3d.visualization.draw_geometries([before])
    # o3d.visualization.draw_geometries([after])
    # o3d.visualization.draw_geometries([change])

    mu, sigma = 0, 0.01  # mean and standard deviation
    after = tools.apply_noise(after, mu, sigma)

    print("Perform change detection")
    result = detect_change(after, before, 0.05)
    result.paint_uniform_color([1, 0, 0])

    compare = detect_change(result, change, 0.01)
    compare.paint_uniform_color([0, 0, 1])

    print(change)
    print(result)
    print(compare)

#    o3d.visualization.draw_geometries([before, result, change])
#    o3d.visualization.draw_geometries([ result, change, compare])
    o3d.visualization.draw_geometries([compare, result, change])