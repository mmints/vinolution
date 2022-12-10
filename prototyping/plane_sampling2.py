#!/usr/bin/env python3

from tkinter import N
import open3d as o3d
import numpy as np
from tools import * 
from change_detection import *

hill = o3d.io.read_triangle_mesh("../meshes/simple-hill.ply")
hill.compute_vertex_normals()

hill_erosion = o3d.io.read_triangle_mesh("../meshes/simple-hill-erosion3.ply")
hill_erosion.compute_vertex_normals()

hill_pc = hill.sample_points_uniformly(number_of_points=30000)
hill_erosion_pc = hill_erosion.sample_points_uniformly(number_of_points=30000)

hill_pc.paint_uniform_color([1, 0.706, 0])
hill_erosion_pc.paint_uniform_color([0, 0.651, 0.929])

draw([hill_pc, hill_erosion_pc])

positive, negative, full = full_change_detection(hill_pc, hill_erosion_pc, 0.1)

draw([positive])
draw([negative])
draw([full])
