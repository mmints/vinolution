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
    TN: int
    FP: int
    FN: int

    def completeness(self) -> float:
        return self.TP / (self.TP + self.FN)

    def correctness(self) -> float:
        return self.TP / (self.TP + self.FP)

    def quality(self) -> float:
        return self.TP / (self.TP + self.FN + self.FP)
    
    def f1_score(self) -> float:
        return ( 2 * self.completeness() * self.correctness()) / (self.completeness() + self.correctness())