import open3d as o3d
import numpy as np
from change_detection import *
import matplotlib.pyplot as plt

def draw(pcd_list):
    o3d.visualization.draw_geometries(pcd_list,
                                    zoom= 0.09999999999999995,
                                    front=[-0.67076420364057987, -0.74012416337666398, -0.047870720700779856 ],
                                    lookat=[ 35.053683089226155, -36.475863035733376, 10.876869940700448],
                                    up=[-0.055641512466370113, -0.01414550234284451, 0.99835060317201374])

pc_0 = o3d.io.read_point_cloud("../data_sets/dlr-arena_06-07-2022-downsample-x4-row.xyz")
pc_1 = o3d.io.read_point_cloud("../data_sets/dlr-arena_08-08-2022-downsample-x4-row.xyz")


pc_0.paint_uniform_color([1, 0.706, 0])
pc_1.paint_uniform_color([0, 0.651, 0.929])


p, n, f = full_change_detection(pc_0, pc_1, 0.1)

p.paint_uniform_color([0,1,0])
n.paint_uniform_color([1,0,0])
f.paint_uniform_color([0,0,1])

draw([pc_0])
draw([pc_1])
draw([pc_0, p])
draw([pc_1, n])

p_c = clustering(p, 0.3, 10)
n_c = clustering(n, 0.3, 10)

draw([pc_0, p_c])
draw([pc_1, n_c])
