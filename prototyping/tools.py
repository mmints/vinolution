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

def draw(pcd_list):
    o3d.visualization.draw_geometries(pcd_list,
                                    zoom= 0.59999999999999987,
                                    front=[0.30242170034049143, -0.634062193450314, 0.711692524901128],
                                    lookat=[-0.68104510075213331, -2.9595133054757281, -2.7221508579471339],
                                    up=[-0.20611966991750472, 0.68548558515614777, 0.69830093385032266])
