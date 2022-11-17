import copy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def detect_change(src, dst, distance):
    out = copy.deepcopy(src)
    dists = src.compute_point_cloud_distance(dst)
    difference_idx = np.argwhere(np.asarray(dists) > distance)
    out = src.select_by_index(difference_idx)
    return out

def outlier_removal(pc, nb_points, radius):
    cl, ind = pc.remove_radius_outlier(nb_points, radius, print_progress = True)
    out = pc.select_by_index(ind)
    return out

# Labeling of clusters DBSCAN [Ester1996]
def clustering(pc, eps, min_points):
    out = copy.deepcopy(pc)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(out.cluster_dbscan(eps, min_points, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("jet")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    out.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return out