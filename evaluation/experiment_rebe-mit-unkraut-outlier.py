from dataclasses import dataclass
import open3d as o3d
import numpy as np
import tools
from change_detection import *
from quantitative import *
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

    # Create point clouds for evaluation
    # print("Before: ", tools.get_point_count(before))
    # print("ground_truth_positive: ", tools.get_point_count(ground_truth_positive))
    # print("ground_truth_negative: ", tools.get_point_count(ground_truth_negative))
    # print("ground_truth_negative + ground_truth_positive: ", tools.get_point_count(ground_truth_positive) + tools.get_point_count(ground_truth_negative))
    # print("after: ", tools.get_point_count(after))

    results = []
    k = 10 # [5, 15, 30]  # test before plotting 
    threshold = 0.04     # 0.0 to 0.1

    sigma_runs = [0.01, 0.025, 0.5]
    for sigma in sigma_runs:
        
        # Clear point clouds for every sigma
        before = pc_0 # before is a reference to pc_0
        ground_truth_positive = tools.concatenate_point_clouds([pc_1, pc_2, pc_3, pc_4, pc_5]) # All changes combined. Use for ground truth.
        ground_truth_negative = pc_0    
        ground_truth_positive = tools.apply_noise(ground_truth_positive, 0.0, sigma)
        ground_truth_negative = tools.apply_noise(ground_truth_negative, 0.0, sigma)
        after = tools.concatenate_point_clouds([ground_truth_positive, ground_truth_negative])

        recall = []
        precision = []
        accuracy = []
        f1_score = []
        radius_ar = []

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])

        radius = 0.025        # 0.0 to 0.1
        while radius <= 0.2001:
            radius_ar.append(radius)
            print("radius: " + str(radius))

            prediction = detect_change(after, before, threshold)

            # pc = outlier_removal(prediction, k, radius) # <---------------------------------------------- EDIT K HERE!
            
            cl, ind = prediction.remove_radius_outlier(k, radius, print_progress = True)
            if (len(ind) == 0):
                out = pcl
                print("###########################################################################################################", out)
            else:
                out = prediction.select_by_index(ind)

            #o3d.visualization.draw_geometries([out])


            result = quantify(ground_truth_positive, ground_truth_negative, out)

            print("prediction count: ", tools.get_point_count(prediction))
            print("prediction count after outlier: ", tools.get_point_count(out))

            print("TP: ", result.TP)
            print("TN: ", result.TN)
            print("FP: ", result.FP)
            print("FN: ", result.FN)

            recall.append(result.recall())
            precision.append(result.precision())
            accuracy.append(result.accuracy())
            f1_score.append(result.f1_score())

            radius += 0.01

        sigma_result = [radius_ar, recall, precision, accuracy, f1_score]
        results.append(sigma_result)

    print("plot...")
    for i in range(1,5):
        plt.rcParams.update({
            'font.family' : 'Times New Roman',
            'mathtext.fontset' : 'stix',
            'font.size' : 15
        })
        plt.gcf().set_size_inches(5, 4)
        plt.grid()
        plt.xlim(0.0, 0.205) # 0.025
        plt.ylim(0.0, 1.05)
        plt.scatter(results[0][0], results[0][i], label='$\sigma$ = 0.01 m', marker='o',zorder=10)
        plt.scatter(results[1][0], results[1][i], label='$\sigma$ = 0.025 m', marker='D',zorder=11)
        plt.scatter(results[2][0], results[2][i],label='$\sigma$ = 0.05 m', marker='s',zorder=12)
        plt.plot(results[0][0], results[0][i],zorder=10)
        plt.plot(results[1][0], results[1][i],zorder=11)
        plt.plot(results[2][0], results[2][i],zorder=12)
        
        plt.xlabel("radius (m)")
        if (i == 1):
            plt.ylabel("recall")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/outlier/outlier-recall-k" + str(k) + ".pdf")
            #plt.show()
            plt.clf()

        if (i == 2):
            plt.ylabel("precision")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/outlier/outlier-precision-k" + str(k) + ".pdf")
            #plt.show()
            plt.clf()

        if (i == 3):
            plt.ylabel("accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/outlier/outlier-accuracy-k" + str(k) + ".pdf")
            #plt.show()
            plt.clf()

        if (i == 4):
            plt.ylabel("F1-score")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/outlier/outlier-f1_score-k" + str(k) + ".pdf")
            #plt.show()
            plt.clf()
