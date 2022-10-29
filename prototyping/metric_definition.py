#!/usr/bin/env python3

import open3d as o3d

if __name__ == "__main__":

    print("Load Test Files")
    source = o3d.io.read_point_cloud("data_sets/mark_ma_1_0.xyz")
    target = o3d.io.read_point_cloud("data_sets/mark_ma_3_0.xyz")

    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    threshold = 0.02

    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    o3d.visualization.draw_geometries([source, target])
