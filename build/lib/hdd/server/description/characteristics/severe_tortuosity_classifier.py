import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize

from typing import Union

from description.characteristics import CharacteristicClassifier

class SevereTortuosityClassifier(CharacteristicClassifier):

    def __init__(self, mask: np.ndarray, epsilon:int, curvature_threshold:float):
        super().__init__()
        self.mask = mask
        self.epsilon = epsilon
        self.curvature_threshold = curvature_threshold

    def predict(self) -> bool:
        return True if len(self._detect_curves()) > 1 else False
 
    def _detect_curves(self) -> set[Union[int,int]]:
        binary = self.mask.astype(np.uint8)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        curves = set()
        for contour in contours:
            approx = cv.approxPolyDP(contour, self.epsilon, True)
            for i in range(1, len(approx) - 1):
                p1 = approx[i - 1][0]
                p2 = approx[i][0]
                p3 = approx[i + 1][0]
                num = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                den = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**1.5
                if den == 0:
                    continue
                curvature = num / den
                if curvature > self.curvature_threshold:
                    curves.add(tuple(p2))
        return curves
            
