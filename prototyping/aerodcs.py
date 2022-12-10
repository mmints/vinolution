import open3d as o3d
import numpy as np
from change_detection import *
import matplotlib.pyplot as plt

def draw(pcd_list):
    o3d.visualization.draw_geometries(pcd_list,
                                    zoom= 0.17999999999999999,
                                    front=[ 0.4067194420323671, -0.90866216812395595, 0.094405294836421572],
                                    lookat=[11.793754064188034, 26.149444827649589, 6.8062798758549627],
                                    up=[-0.031814686523268618, 0.089187648939673111, 0.99550659917352624])

pc_0 = o3d.io.read_point_cloud("../data_sets/aerodcs/2022-07-05.xyz")
pc_1 = o3d.io.read_point_cloud("../data_sets/aerodcs/2022-08-03.xyz")


pc_0.paint_uniform_color([1, 0.706, 0])
pc_1.paint_uniform_color([0, 0.651, 0.929])


p, n, f = full_change_detection(pc_0, pc_1, 0.1)

p.paint_uniform_color([0,1,0])
n.paint_uniform_color([1,0,0])
f.paint_uniform_color([0,0,1])

# o3d.visualization.draw_geometries([p])
# o3d.visualization.draw_geometries([n])
draw([pc_1])
draw([pc_0])
draw([pc_0])
draw([pc_1])

draw([pc_0, p])
draw([pc_1, n])

p_c = clustering(p, 0.3, 10)
n_c = clustering(n, 0.3, 10)

draw([pc_0, p_c])
draw([pc_1, n_c])
