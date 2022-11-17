from dataclasses import dataclass
import open3d as o3d
import numpy as np
import tools
from change_detection import *
import quantitative as q
import matplotlib.cm as cm

if __name__ == "__main__":

    print("Load Test Files")
    # Main
    pc_0 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_0.xyz")
    
    # Changes
    pc_1 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_1.xyz") # top
    pc_2 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_2.xyz") # long weeds
    pc_3 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_3.xyz") # short weeds
    pc_4 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_4.xyz") # central
    pc_5 = o3d.io.read_point_cloud("../data_sets/rebe-mit-unkraut/rebe-mit-unkraut_5.xyz") # outer

    pc_0.paint_uniform_color([0, 0, 0])
    pc_1.paint_uniform_color([0, 1, 0])
    pc_2.paint_uniform_color([0, 0, 1])
    pc_3.paint_uniform_color([1, 1, 0])
    pc_4.paint_uniform_color([1, 0, 1])
    pc_5.paint_uniform_color([0, 1, 1])

    # Visualize all raw data
    # o3d.visualization.draw_geometries([pc_0, pc_1, pc_2, pc_3, pc_4, pc_5])

    # Create point clouds for evaluation
    ground_truth = tools.concatenate_point_clouds([pc_1, pc_2, pc_3, pc_4, pc_5]) # All changes combined. Use for ground truth.
    ground_truth.paint_uniform_color([0, 1, 0])
    before = pc_0 # before is a reference to pc_0
    after =  tools.concatenate_point_clouds([pc_0, ground_truth])
    
    # o3d.visualization.draw_geometries([before])
    # o3d.visualization.draw_geometries([after])
    # o3d.visualization.draw_geometries([change])

    mu, sigma = 0, 0.0
    completeness = []
    correctness = []
    quality = []
    f1_score = []
    sigma_ar = []
    while sigma <= 0.01:
        sigma_ar.append(sigma)
        print("Sigma: " + str(sigma))
        after = tools.apply_noise(after, mu, sigma)
        # print("Perform change detection")
        result = detect_change(after, before, 0.01) # Calculated Change


        # print("Start outlier removal with nb_neighbors")
        cl, ind = result.remove_radius_outlier(nb_points = 15, radius = 0.025, print_progress = False)
        # print("remove_outlier: DONE!")
        result = result.select_by_index(ind)

        q_res = q.quantify(ground_truth, result, 0.01)

        # print("completeness: " +  str(q_res.completeness()))
        print("correctness: " +  str(q_res.correctness()))
        # print("quality: " +  str(q_res.quality()))
        # print("f1_score: " +  str(q_res.f1_score()))

        completeness.append(q_res.completeness())
        correctness.append(q_res.correctness())
        # quality.append(q_res.quality())
        f1_score.append(q_res.f1_score())

        sigma += 0.0001

    plt.scatter(sigma_ar, completeness, cmap='viridis', label='recall')
    plt.scatter(sigma_ar, correctness, cmap='viridis', label='precision')
    # plt.scatter(sigma_ar, quality, cmap='viridis',label='quality')
    plt.scatter(sigma_ar, f1_score, cmap='viridis',label='F1 score')
    plt.plot(sigma_ar, completeness)
    plt.plot(sigma_ar, correctness)
    # plt.plot(sigma_ar, quality)
    plt.plot(sigma_ar, f1_score)
    
    plt.xlabel("Standardabweichung (m)")
    plt.ylabel("Genauigkeit (%)")
    plt.legend()
    
    # To show the plot
    plt.show()

    # o3d.visualization.draw_geometries([fp, result])
