#!/usr/bin/env python3

import open3d as o3d
import numpy as np

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
    voxel_size = 0.1
    distance_threshold = 0.05
    max_iterations = 20
    max_tuples = 10

    print("Load Test Files")
    src = o3d.io.read_point_cloud("../data_sets/mark_ma_1_0-cut.xyz")
    dst = o3d.io.read_point_cloud("../data_sets/mark_ma_3_0-cut.xyz")

    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    
    dst.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))

    src.paint_uniform_color([1, 0.706, 0])
    dst.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([src])
    o3d.visualization.draw_geometries([dst])

    o3d.visualization.draw_geometries([src, dst])
    
    # print('Downsampling inputs')
    # src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    # dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    # print('Running FGR')
    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #     src_down, dst_down, src_fpfh, dst_fpfh,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=distance_threshold,
    #         iteration_number=max_iterations,
    #         maximum_tuple_count=max_tuples))

    # o3d.visualization.draw_geometries([src.transform(result.transformation), dst])

    dists = src.compute_point_cloud_distance(dst)
#dists = np.asarray(dists)
print(np.asarray(dists).size)
print(np.asarray(dst.points))

difference_idx = np.argwhere(np.asarray(dists) >= 0.015)
print(np.asarray(difference_idx).size)

print(difference_idx)

difference_pc = src.select_by_index(difference_idx)
difference_pc.paint_uniform_color([1.0, 0.0, 0.0])


# print("Start outlier removal with nb_neighbors")
# cl, ind = difference_pc.remove_radius_outlier(nb_points = 10, radius = 1.00, print_progress = True)
# print("remove_outlier: DONE!")

# print(len(ind))

# difference_pc = difference_pc.select_by_index(ind)

o3d.visualization.draw_geometries([dst, src, difference_pc])
o3d.visualization.draw_geometries([dst, difference_pc])
o3d.visualization.draw_geometries([src, difference_pc])

o3d.visualization.draw_geometries([difference_pc])
