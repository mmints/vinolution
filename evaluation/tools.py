import open3d as o3d
import numpy as np
import copy

def concatenate_point_clouds(pc_list):
    result = np.asarray(pc_list[0].points)
    for i in range(1, len(pc_list)):
        pc_array = np.asarray(pc_list[i].points)
        result = np.concatenate((result, pc_array), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result)
    return pcd

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def get_point_count(pcd):
    return int(np.asarray(pcd.points).size/3) # Divide by 3 because it's always 3 coordinates