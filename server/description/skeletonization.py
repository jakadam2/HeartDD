import torch

from scipy.stats import multivariate_normal
from skimage.morphology import skeletonize
import cv2 as cv
import numpy as np


class HDBifurcationPDF:

    cov = [[10,0],[0,10]]

    kernels = [
            [[1,0,1],
             [0,1,0],
             [0,1,0]],

            [[0,1,0],
             [0,1,0],
             [1,0,1]],

            [[0,0,1],
             [1,1,0],
             [0,0,1]],

            [[1,0,0],
             [0,1,1],
             [1,0,0]],

            [[1,0,1],
             [0,1,0],
             [0,0,1]],

            [[0,0,1],
             [0,1,0],
             [1,0,1]],

            [[1,0,0],
             [0,1,0],
             [1,0,1]],

            [[1,0,0],
             [0,1,0],
             [1,0,1]],

            [[0,1,0],
             [1,1,0],
             [0,0,1]],
             
            [[0,1,0],
             [0,1,1],
             [1,0,0]],
             
            [[1,0,0],
             [0,1,1],
             [0,1,0]],
             
            [[0,0,1],
             [1,1,0],
             [0,1,0]]]

    def __init__(self,mask:torch.Tensor):
        self.skeletonized_mask = skeletonize(mask)
        self.skeletonized_mask = self.skeletonized_mask > 0.5
        self.bifurcation_distributions: list = self._find_bifurcations()

    def __call__(self,x:int,y:int) -> float:
        return self.predict(x,y)

    def predict(self,x:int,y:int) -> float:
        max_prob = 0
        for distribution in self.bifurcation_distributions:
            max_prob = max(max_prob,distribution.pdf([x,y]))
        return max_prob
    
    def _find_bifurcations(self) -> list:
        results = np.zeros(shape = self.skeletonized_mask.shape)
        for kernel in HDBifurcationPDF.kernels:
            results = np.any(cv.erode(self.skeletonized_mask,kernel,iterations=1),results)

        results_pdf = []
        for x,y in np.where(results == 1):
            results_pdf.append(multivariate_normal(mean=[x,y],cov=HDBifurcationPDF.cov))
        
        return results_pdf

