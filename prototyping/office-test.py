#!/usr/bin/env python3

import copy
import open3d as o3d
import numpy as np

# Default values for drawing call
m_zoom = 0.02
m_front = [-0.88207343240636815, 0.23772630073755963, 0.40673414631732574]
m_lookat = [11.940487878864165, -0.58109376005067859, -1.6723159874529427]
m_up = [0.43057972625745072, 0.056464467922324971, 0.90078458201631517]


def draw(pcd_list, zoom=m_zoom, front=m_front, lookat=m_lookat, up=m_up):
    o3d.visualization.draw_geometries(pcd_list)

def estimate_normals(pcd):
    radius_normal = 1.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

def remove_outlier(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)  
    return pcd.select_by_index(ind)

def down_sampling(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)

def compute_fpfh(pcd, voxel_size, max_nn=100):
    #cd_cp = copy.deepcopy(pcd)    # Use copy to avoid adding normals to point cloud
    radius_feature = voxel_size * (10 * voxel_size)    # TODO: Same question as in normal estimation
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
    return pcd_fpfh

def fast_global_registration(source_down, target_down, source_fpfh,
                             target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5    # TODO: Same question as in normal estimation
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result.transformation

if __name__ == "__main__":

    print("Load Test Files")
    source = o3d.io.read_point_cloud("data_sets/mark_ma_1_0.xyz")
    target = o3d.io.read_point_cloud("data_sets/mark_ma_3_0.xyz")

    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    draw([source, target])

    print("Remove Outlier")
    source = remove_outlier(source)
    target = remove_outlier(target)

    # print("Down Sampling")
    voxel_size = 0.5
    source_down = down_sampling(source, voxel_size)
    target_down = down_sampling(target, voxel_size)

    print("estimate_normals")
    estimate_normals(source_down)
    estimate_normals(target_down)

    draw([source_down, target_down])

    print("Compute Fast Point Feature Histogram")
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)


    estimate_normals(source)
    estimate_normals(target)

    print("Registration")
    extrinsic_parameters = fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    source.transform(extrinsic_parameters)

    extrinsic_parameters = fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    source.transform(extrinsic_parameters)

    extrinsic_parameters = fast_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    source.transform(extrinsic_parameters)

    draw([source, target])

    # print("Start Change Detection")
    # dists = source.compute_point_cloud_distance(target)
    # #dists = np.asarray(dists)
    # print(np.asarray(dists).size)
    # print(np.asarray(target.points))

    # difference_idx = np.argwhere(np.asarray(dists) >= 0.1)
    # print(np.asarray(difference_idx).size)

    # print(difference_idx)

    # difference_pc = source.select_by_index(difference_idx)
    # difference_pc.paint_uniform_color([1.0, 0.0, 0.0])

    # print("Start outlier removal with nb_neighbors")
    # cl, ind = difference_pc.remove_radius_outlier(nb_points = 10, radius = 0.25, print_progress = True)
    # print("remove_outlier: DONE!")

    # print(len(ind))

    # difference_pc = difference_pc.select_by_index(ind)

    # draw([source, target, difference_pc])
    # draw([difference_pc])
