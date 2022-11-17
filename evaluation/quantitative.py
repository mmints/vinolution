#!/usr/bin/env python3

'''
This script performs a quantitative analysis of the utilized Change Detection pipeline.
Calculated metrics:
    - completeness = TP / (TP + FN)
    - correctness = TP / (TP + FP)
    - quality = TP / (TP + FN + FP)
    - F1 = ( 2 * completeness * correctness) / (completeness + correctness)

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

    def completeness(self) -> float:
        return self.TP / (self.TP + self.FN)

    def correctness(self) -> float:
        if ((self.TP + self.FP) == 0.0):
            return 0.0
        return self.TP / (self.TP + self.FP)

    def quality(self) -> float:
        return self.TP / (self.TP + self.FN + self.FP)
    
    def f1_score(self) -> float:
        if ((self.completeness() + self.correctness()) == 0.0):
            return 0.0
        return ( 2 * self.completeness() * self.correctness()) / (self.completeness() + self.correctness())

def quantify(ground_truth, pcd, threshold = 0.0):
    fn_pcd = detect_change(ground_truth, pcd, threshold)
    fn = tools.get_point_count(fn_pcd)
    tp = tools.get_point_count(ground_truth) - fn
    fp = tools.get_point_count(pcd) - tp
    return QuantitativeResult(tp, fp, fn)
