import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from sklearn.svm import SVC
from PIL import Image

from typing import Union


from description.characteristics import CharacteristicClassifier

class BluntStumpClassifier(CharacteristicClassifier):

    BOX_SIZE = 55

    def __init__(self, model_path:str):
        super().__init__()
        self.model:SVC = joblib.load(model_path)

    def predict(self, image:np.ndarray, coords:Union[int,int]):
        x,y = coords
        croped = image.crop((x, y, x + BluntStumpClassifier.BOX_SIZE, y + BluntStumpClassifier.BOX_SIZE))
        hog_features = self._extract_hog_features(croped)
        return self.model.predict_proba([hog_features])[0][1]
    @staticmethod
    def _extract_hog_features(image):
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        return features