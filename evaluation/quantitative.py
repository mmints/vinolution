#!/usr/bin/env python3

'''
This script performs a quantitative analysis of the utilized Change Detection pipeline.
Calculated metrics:
    - recall = TP / (TP + FN)
    - precision = TP / (TP + FP)
    - accuracy = TP + TN / (TP + FN + FP + TN)
    - F1 = ( 2 * recall * precision) / (recall + precision)

Changes are given separated but first evaluated separated.
In a second step, the divided changes can be evaluated against calculated change clusters 
'''
from dataclasses import dataclass
import open3d as o3d
import numpy as np
import tools
from change_detection import *

@dataclass
class QuantitativeResult:
    TP: int
    FP: int
    FN: int
    TN: int

    def recall(self) -> float:
        if ((self.TP + self.FN) == 0.0):
            return 0.0
        return self.TP / (self.TP + self.FN)

    def precision(self) -> float:
        if ((self.TP + self.FP) == 0.0):
            return 0.0
        return self.TP / (self.TP + self.FP)

    def accuracy(self) -> float:
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)
    
    def f1_score(self) -> float:
        if ((self.recall() + self.precision()) == 0.0):
            return 0.0
        return ( 2 * self.recall() * self.precision()) / (self.recall() + self.precision())

def quantify(ground_truth_positive, ground_truth_negative, prediction, threshold = 0.0):
    FN_cloud = detect_change(ground_truth_positive, prediction, threshold)
    FN = tools.get_point_count(FN_cloud)

    FP_cloud = detect_change(prediction, ground_truth_positive, threshold)
    FP = tools.get_point_count(FP_cloud)

    TP_cloud = detect_change(prediction, ground_truth_negative, threshold)
    TP = tools.get_point_count(TP_cloud)


    TN = tools.get_point_count(ground_truth_positive) + tools.get_point_count(ground_truth_negative) - FN - FP - TP
    return QuantitativeResult(TP, FP, FN, TN)
