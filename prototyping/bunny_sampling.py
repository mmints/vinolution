#!/usr/bin/env python3

import open3d as o3d

bunny = o3d.data.BunnyMesh()

print(bunny.data_root)

orig_mesh = o3d.io.read_triangle_mesh(bunny.path)
orig_mesh.compute_vertex_normals()

orig_mesh2 = o3d.io.read_triangle_mesh(bunny.path)
orig_mesh2.compute_vertex_normals()


modif_mesh = o3d.io.read_triangle_mesh("bunny_mode.ply")
modif_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([orig_mesh])
o3d.visualization.draw_geometries([modif_mesh])


orig_pcd = orig_mesh.sample_points_uniformly(number_of_points=5000)
orig_pcd2 = orig_mesh.sample_points_uniformly(number_of_points=5000)
modif_pcd = modif_mesh.sample_points_uniformly(number_of_points=5000)

orig_pcd.paint_uniform_color([1, 0.706, 0])
modif_pcd.paint_uniform_color([0, 0.651, 0.929])

#print("distance: " + str(orig_pcd.compute_point_cloud_distance(orig_pcd2)))

o3d.visualization.draw_geometries([orig_pcd, modif_pcd])

array = []

for i in orig_pcd.compute_point_cloud_distance(modif_pcd):
    if i > 0.002:
        array.append(i)

print("distance: " + str(array))


