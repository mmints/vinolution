#!/usr/bin/env python3
import copy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

if __name__ == "__main__":
    voxel_size = 0.05
    distance_threshold = 0.01
    max_iterations = 20
    max_tuples = 10

    print("Load Test Files")
    src = o3d.io.read_point_cloud("../data_sets/dlr-arena_08-08-2022-downsample-x4-4rows.xyz")
    dst = o3d.io.read_point_cloud("../data_sets/dlr-arena_06-07-2022-downsample-x4-4rows.xyz")

#    print("Start outlier removal with nb_neighbors")
#    cl, ind = src.remove_radius_outlier(nb_points = 15, radius = 0.1, print_progress = True)
#    print("remove_outlier: DONE!")
#    src = src.select_by_index(ind)
#
#    print("Start outlier removal with nb_neighbors")
#    cl, ind = dst.remove_radius_outlier(nb_points = 15, radius = 0.1, print_progress = True)
#    print("remove_outlier: DONE!")
#    dst = dst.select_by_index(ind)

    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    
    dst.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))

    src.paint_uniform_color([1, 0.706, 0])
    dst.paint_uniform_color([0, 0.651, 0.929])

    # Distance src -> dst
    print("Distance src -> dst")
    dists = src.compute_point_cloud_distance(dst)
    difference_idx = np.argwhere(np.asarray(dists) >= 0.075)
    difference_pc_src = src.select_by_index(difference_idx)
    difference_pc_src.paint_uniform_color([1.0, 0.0, 0.0])
    print("Done")

    # Distance dst -> src
    print("Distance dst -> src")
    dists = dst.compute_point_cloud_distance(src)
    difference_idx = np.argwhere(np.asarray(dists) >= 0.075)
    difference_pc_dst = dst.select_by_index(difference_idx)
    difference_pc_dst.paint_uniform_color([0.0, 1.0, 0.0])
    print("Done")

#    print("Start outlier removal with nb_neighbors SRC-DIF")
#    cl, ind = difference_pc_src.remove_radius_outlier(nb_points = 8, radius = 0.075, print_progress = True)
#    print("remove_outlier: DONE!")
#    difference_pc_src = difference_pc_src.select_by_index(ind)
#
#    print("Start outlier removal with nb_neighbors DST-DIF")
#    cl, ind = difference_pc_dst.remove_radius_outlier(nb_points = 8, radius = 0.075, print_progress = True)
#    print("remove_outlier: DONE!")
#    difference_pc_dst = difference_pc_dst.select_by_index(ind)

    # Distance difference_pc_src -> difference_pc_dst
    print("difference_pc_src -> difference_pc_dst")
    dists = difference_pc_src.compute_point_cloud_distance(difference_pc_dst)
    difference_idx = np.argwhere(np.asarray(dists) >= 0.1)
    difference_pc = difference_pc_src.select_by_index(difference_idx)
    difference_pc.paint_uniform_color([1.0, 0.0, 1.0])
    print("Done")

    print("Start outlier removal with nb_neighbors DIF")
    cl, ind = difference_pc.remove_radius_outlier(nb_points = 20, radius = 0.1, print_progress = True)
    print("remove_outlier: DONE!")
    difference_pc = difference_pc.select_by_index(ind)

    # Labeling of clusters DBSCAN [Ester1996]
    print("Create DBSCAN clusters")
    labeled = copy.deepcopy(difference_pc)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(labeled.cluster_dbscan(eps=0.1, min_points=32, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    labeled.colors = o3d.utility.Vector3dVector(colors[:, :3])

#    src_xyzrgb = o3d.io.read_point_cloud("../data_sets/mark_ma_1_0-cut-high-res.xyzrgb")
#    dst_xyzrgb = o3d.io.read_point_cloud("../data_sets/mark_ma_3_0-cut-high-res.xyzrgb")

#    o3d.visualization.draw_geometries([src_xyzrgb])
#    o3d.visualization.draw_geometries([dst_xyzrgb])

#    o3d.visualization.draw_geometries([src])
#    o3d.visualization.draw_geometries([dst])

    o3d.visualization.draw_geometries([src, dst])
    o3d.visualization.draw_geometries([difference_pc_dst])
    o3d.visualization.draw_geometries([difference_pc_src])
    o3d.visualization.draw_geometries([difference_pc, difference_pc_src, difference_pc_dst])
    o3d.visualization.draw_geometries([difference_pc])
    o3d.visualization.draw_geometries([difference_pc, dst])
    o3d.visualization.draw_geometries([labeled, dst])
    o3d.visualization.draw_geometries([labeled, src])
    o3d.visualization.draw_geometries([labeled])

#    o3d.visualization.draw_geometries([dst_xyzrgb, labeled])
