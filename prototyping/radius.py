import open3d as o3d
import numpy as np
from change_detection import *
import matplotlib.pyplot as plt

def draw(pcd_list):
    o3d.visualization.draw_geometries(pcd_list,
                                    zoom= 0.21999999999999958,
                                    front=[-0.47022559197676966, -0.8818427591132465, -0.035231248198627295],
                                    lookat=[0.037726399477745497, 0.06456661007571099, 0.23428060275753529],
                                    up=[-0.12812699257147112, 0.028715296010989294, 0.99134197205081132])


pc_0 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_0.xyz")


s = o3d.geometry.TriangleMesh.create_sphere(radius=.1, resolution=10)
#s.paint_uniform_color([1,0,0])
#s.compute_vertex_normals()
s.translate([0.1, 0.1, 0.1])
draw([pc_0, s])