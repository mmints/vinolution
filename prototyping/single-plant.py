from dataclasses import dataclass
import open3d as o3d
import numpy as np
import tools
from change_detection import *

def draw(pcd_list):
    o3d.visualization.draw_geometries(pcd_list,
                                    zoom= 0.55999999999999983,
                                    front=[-0.58803903789034162, -0.79297346808868163, 0.1593837156814023 ],
                                    lookat=[ -0.062380394354546578, 0.089765625783176692, -0.34395825257148915],
                                    up=[0.018578513146899817, 0.18375989265910786, 0.98279557421630859])

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

    pc_0.paint_uniform_color([1, 0.706, 0])
    # pc_1.paint_uniform_color([0, 1, 0])
    # pc_2.paint_uniform_color([0, 0, 1])
    # pc_3.paint_uniform_color([1, 1, 0])
    # pc_4.paint_uniform_color([1, 0, 1])
    # pc_5.paint_uniform_color([0, 1, 1])

    # Visualize all raw data
    # o3d.visualization.draw_geometries([pc_0, pc_1, pc_2, pc_3, pc_4, pc_5])

    # Create point clouds for evaluation
    change = tools.concatenate_point_clouds([pc_1, pc_2, pc_3, pc_4, pc_5]) # All changes combined. Use for ground truth.
    change.paint_uniform_color([0, 1, 0])
    before = pc_0 # before is a reference to pc_0
    after =  tools.concatenate_point_clouds([pc_0, change])
    after.paint_uniform_color([0, 0.651, 0.929])

    mu, sigma = 0, 0.02  # mean and standard deviation
    after = tools.apply_noise(after, mu, sigma)

    print("Perform change detection")
    after2 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_change-no-wires.xyz") # outer
    result = detect_change(after, before, 0.01)
    result.paint_uniform_color([0, 1, 0])
    result = outlier_removal(result, 10, 0.025)
    # draw([before])
    # draw([after])
    draw([before, result])
