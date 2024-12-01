import torch
import numpy as np
import cv2 as cv
from PIL import Image

import platform
import pathlib


class LesionDetector:

    def __init__(self, image, mask=None):

        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath
        self.model_path = pathlib.Path('./server/detection/checkpoints/best_new.pt')
        # Pillow Image transform to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:  # Convert only if grayscale
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        self.image = image
        self.mask = mask

    @staticmethod
    def cover_image(image, mask):
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] = (image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] * 0.3).astype(np.uint8)
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] = np.clip(
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] * 1.5, 
        0, 255
        ).astype(np.uint8)
        return image

    def detect(self):
        if self.mask is not None:
            #self.image = self.cover_image(self.image, self.mask)
            dilated_mask = cv.dilate(self.mask, kernel=np.ones((5, 5)), iterations=10)
            if len(dilated_mask) == 3: # Convert if BGR
                dilated_mask = cv.cvtColor(dilated_mask, cv.COLOR_BGR2GRAY)

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload = True)
        results = model(self.image)
        coordinates = []

        for *xyxy, conf, cls in results.xyxy[0]: 
            x1, y1, x2, y2 = map(int, xyxy)

            if self.mask is not None:
                x_mean, y_mean = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if dilated_mask[y_mean, x_mean] == 0:
                    continue

            coordinates.append({ 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 })
        
        return coordinates