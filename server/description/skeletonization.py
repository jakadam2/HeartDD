import torch

from scipy.stats import multivariate_normal
from skimage.morphology import skeletonize


class HDBifurcationPDF:

    cov = [[10,0],[0,10]]

    def __init__(self,mask:torch.Tensor):
        self.skeletonized_mask = skeletonize(mask)
        self.bifurcation_distributions: list = self._find_bifurcations()

    def __call__(self,x:int,y:int) -> float:
        return self.predict(x,y)

    def predict(self,x:int,y:int) -> float:
        max_prob = 0
        for distribution in self.bifurcation_distributions:
            max_prob = max(max_prob,distribution.pdf([x,y]))
        return max_prob
    
    def _find_bifurcations(self) -> list: 
        for i in range(self.skeletonized_mask.shape[0]):
            for j in range(self.skeletonized_mask.shape[1]):
                if self._is_bifurcation(i,j):
                    self.bifurcation_distributions.append(multivariate_normal(mean = [i,j],cov = HDBifurcationPDF.cov))

    def _neighbours_8(self,x,y): 
        x_min, y_min, x_max, y_max = x-1, y-1, x+1, y+1
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, self.skeletonized_mask.shape[0]-1)
        y_max = min(y_max, self.skeletonized_mask.shape[1]-1)
        return [ self.skeletonized_mask[x_min][y], self.skeletonized_mask[x_min][y_max], self.skeletonized_mask[x][y_max], self.skeletonized_mask[x_max][y_max],  
                self.skeletonized_mask[x_max][y], self.skeletonized_mask[x_max][y_min], self.skeletonized_mask[x][y_min], self.skeletonized_mask[x_min][y_min] ]

    def _is_bifurcation(self, x, y):
        P1,P2,P3,P4,P5,P6,P7,P8 = self._neighbours_8(x, y)
        if (P8 * P2 * P6 == 1):
            return True
        elif (P8 * P6 * P4 == 1):
            return True
        elif (P6 * P4 * P2 == 1):
            return True
        elif (P4 * P2 * P8 == 1):
            return True
        elif (P8 * P2 * P5 == 1):
            return True
        elif (P8 * P6 * P3 == 1):
            return True
        elif (P6 * P4 * P1 == 1):
            return True
        elif (P4 * P2 * P7 == 1):
            return True
        elif (P6 * P1 * P3 ==1):
            return True
        elif (P4 * P7 * P1 == 1):
            return True
        elif (P2 * P5 * P7 == 1):
            return True
        elif (P8 * P5 * P3 == 1):
            return True
        return False