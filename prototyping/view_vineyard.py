#!/usr/bin/env python3

import open3d as o3d

def remove_outlier(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)  
    return pcd.select_by_index(ind)

if __name__ == "__main__":

    vine_yard_pc = o3d.io.read_point_cloud("data_sets/arena1_reduced_points-100.xyz")

    #vine_yard_pc.paint_uniform_color([0, 0.651, 0.929])

    vine_yard_pc_2 = remove_outlier(vine_yard_pc)
    
    vine_yard_pc_2.paint_uniform_color([1, 0.706, 0])
    
    radius_normal = 1.0
    vine_yard_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    vine_yard_pc_2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    o3d.visualization.draw_geometries([vine_yard_pc])
    o3d.visualization.draw_geometries([vine_yard_pc,vine_yard_pc_2])
    o3d.visualization.draw_geometries([vine_yard_pc_2])