import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize

from hdd.server.description.characteristics import CharacteristicClassifier

class BifurcationClassifier(CharacteristicClassifier):



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
    
    sigma = 100
    
    def __init__(self, mask:np.ndarray):
        super().__init__()
        self.skeletonized_mask = self._prune_skeleton(skeletonize(mask).astype(np.uint8),min_branch_length=100)
        self._find_bifurcations()

    def predict(self, x:int, y:int) -> bool:
        return max([prob(x,y) for prob in self.bifurcations]) if len(self.bifurcations) > 0 else 0
    
    def _create_prob(self, x:int, y:int):
        def flattened_distribution(x0, y0):
            distance_squared = (x - x0)**2 + (y - y0)**2
            return 1 / (1 + (distance_squared / (BifurcationClassifier.sigma**2)))
        return flattened_distribution

    def _prune_skeleton(self, skeleton, min_branch_length=20):
        skeleton = skeleton.astype(np.uint8)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),         (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

        def count_neighbors(x, y, skeleton):
            count = 0
            for dx, dy in neighbors:
                if skeleton[y + dy, x + dx] == 1:
                    count += 1
            return count

        def find_endpoints(skeleton):
            endpoints = []
            for y in range(1, skeleton.shape[0] - 1):
                for x in range(1, skeleton.shape[1] - 1):
                    if skeleton[y, x] == 1 and count_neighbors(x, y, skeleton) == 1:
                        endpoints.append((x, y))
            return endpoints
        endpoints = find_endpoints(skeleton)

        for endpoint in endpoints:
            branch = []
            x, y = endpoint
            while True:
                branch.append((x, y))
                skeleton[y, x] = 0 
                neighbors_count = [(nx, ny) for dx, dy in neighbors
                                if skeleton[y + dy, x + dx] == 1
                                for nx, ny in [(x + dx, y + dy)]]
                
                if len(neighbors_count) == 1: 
                    x, y = neighbors_count[0]
                else:
                    break 

            if len(branch) < min_branch_length:
                for bx, by in branch:
                    skeleton[by, bx] = 0
            else:
                for bx, by in branch:
                    skeleton[by, bx] = 1

        return skeleton
    
    def _find_bifurcations(self) -> list:
       results = np.zeros(shape = self.skeletonized_mask.shape,dtype=bool)
       for kernel in BifurcationClassifier.kernels:
           detections = cv.erode(self.skeletonized_mask,kernel,iterations=1) > 0
           results = detections | results
       
       coords = np.where(results == 1)
       self.coords = coords
       probs = {self._create_prob(x,y) for x,y in zip(list(coords[0]),(coords[1]))}
       self.bifurcations = probs
       return probs