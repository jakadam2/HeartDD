import numpy as np
import os
from os import path
import cv2 as cv
import re 


class MaskCoverer:

    def __init__(self,image_dir:str,mask_dir,target_dir:str) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_dir = target_dir
        pass

    @staticmethod
    def _cover_one_image(image:np.ndarray,mask:np.ndarray) -> np.ndarray:
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] = (image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] * 0.3).astype(np.uint8)
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] = np.clip(
    image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] * 1.5, 
    0, 255
).astype(np.uint8)
        return image
    
    def cover_masks(self) -> None:
        all_images = os.listdir(self.image_dir)
        for image_name in all_images:
            if not image_name.endswith('.png'):
                continue
            image = cv.imread(path.join(self.image_dir,image_name), flags= cv.IMREAD_GRAYSCALE)
            mask_name = f"{re.match(r'.*(?=_.*)', image_name).group(0)}_binmask.png"
            mask = cv.imread(path.join(self.mask_dir,mask_name), flags= cv.IMREAD_GRAYSCALE)
            cover_image = self._cover_one_image(image,mask)
            cv.imwrite(path.join(self.target_dir,image_name),cover_image)

if __name__ == '__main__':
    mask_cover = MaskCoverer('./datasets/images/raw','./datasets/images/masks','./datasets/images/covered')
    mask_cover.cover_masks()
