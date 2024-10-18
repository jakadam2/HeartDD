import torch

from scipy.stats import multivariate_normal
from skimage.morphology import skeletonize
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt


class HDBifurcationPDF:
    """
    A class to compute the probability of bifurcation points based on 
    the skeletonized representation of a binary mask.

    The class finds bifurcation points in the input mask and models 
    them using multivariate normal distributions.

    Attributes:
        skeletonized_mask (np.ndarray): The skeletonized binary mask.
        max_ratio (float): The maximum probability ratio of the distributions.
        bifurcation_distributions (list): A list of multivariate normal distributions 
            representing bifurcation points.
    """
    cov = [[20,0],[0,20]]

    kernels = [
    np.array([[1,0,1],
             [0,1,0],
             [0,1,0]],np.uint8),

    np.array([[0,1,0],
             [0,1,0],
             [1,0,1]],np.uint8),

    np.array([[0,0,1],
             [1,1,0],
             [0,0,1]],np.uint8),

    np.array([[1,0,0],
             [0,1,1],
             [1,0,0]],np.uint8),

    np.array([[1,0,1],
             [0,1,0],
             [0,0,1]],np.uint8),

    np.array([[0,0,1],
             [0,1,0],
             [1,0,1]],np.uint8),

    np.array([[1,0,0],
             [0,1,0],
             [1,0,1]],np.uint8),

    np.array([[1,0,0],
             [0,1,0],
             [1,0,1]],np.uint8),

    np.array([[0,1,0],
             [1,1,0],
             [0,0,1]],np.uint8),
             
    np.array([[0,1,0],
             [0,1,1],
             [1,0,0]],np.uint8),
             
    np.array([[1,0,0],
             [0,1,1],
             [0,1,0]],np.uint8),
             
    np.array([[0,0,1],
             [1,1,0],
             [0,1,0]],np.uint8)]

    def __init__(self,mask:torch.Tensor):
        self.skeletonized_mask = skeletonize(mask).astype(np.uint8) * 255
        self.max_ratio :float= 0.0
        self.bifurcation_distributions: list = self._find_bifurcations()
        

    def __call__(self,x:int,y:int) -> float:
        return self.predict(x,y)

    def predict(self,x:int,y:int) -> float:
        max_prob = 0
        for distribution in self.bifurcation_distributions:
            max_prob = max(max_prob,distribution.pdf([x,y]))
        return max_prob/self.max_ratio
    
    def _find_bifurcations(self) -> list:
        results = np.zeros(shape = self.skeletonized_mask.shape,dtype=bool)
        for kernel in HDBifurcationPDF.kernels:
            detections = cv.erode(self.skeletonized_mask,kernel,iterations=1) > 0
            results = detections|results
        
        coords = np.where(results == 1)
        results_pdf = []
        for x,y in zip(coords[1],coords[0]):
            distribution = multivariate_normal(mean=[x,y],cov=HDBifurcationPDF.cov)
            results_pdf.append(distribution)
            self.max_ratio = max(self.max_ratio,distribution.pdf([x,y]))

        return results_pdf
    
    def show_bifurcation_points(self):
        plt.imshow(self.skeletonized_mask, cmap='gray')
        for distribution in self.bifurcation_distributions:
            x,y  = distribution.mean
            plt.scatter(x,y, color='red', s=40)
        plt.show()
        
    def show_density_graph(self):

        x = np.linspace(0, self.skeletonized_mask.shape[1] - 1, self.skeletonized_mask.shape[1] - 1)
        y = np.linspace(0, self.skeletonized_mask.shape[0] - 1, self.skeletonized_mask.shape[0] - 1)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        max_prob = np.zeros((len(x),len(y)))
        pixel_area = (x[1] - x[0]) * (y[1] - y[0])
        

        for pdf in self.bifurcation_distributions:
            prob = pdf.pdf(pos) * pixel_area
            max_prob = np.maximum(max_prob, prob )

        plt.imshow(np.clip(0,1,max_prob/self.max_ratio),  cmap='inferno', origin='upper')
        plt.colorbar()
        plt.show()


