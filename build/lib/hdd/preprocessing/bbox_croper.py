import pandas as pd
import sys
import cv2 as cv
import numpy as np
import os
import re

BOX_SIZE = 64

class BBoxCropper:

    def __init__(self,path:str) -> None:
        self._path = path
        self._idx = 0
        self._files = os.listdir(path)
        data_file = [file for file  in self._files if re.search('.+.csv',file) is not None]
        if len(data_file) != 1:
            raise ValueError('IN FILE WITH IMAGES SHOULD BE EXACTLY ONE CSV FILE')
        self._df = pd.read_csv(f'{path}/{data_file[0]}',sep = ';')
        try:
            os.makedirs(f'{path}/marked')
        except:
            pass

    def _crop_image(self,image:np.ndarray,idx:int) -> np.ndarray:
        x,y = self._df['X'].iloc[idx],self._df['Y'].iloc[idx]
        w =self._df['WIDTH'].iloc[idx] 
        h = self._df['HEIGHT'].iloc[idx]
        cx = x + 0.5*w
        cy = y + 0.5*h
        image = skimage.exposure.equalize_adapthist(image)
        cropped = image[int(cx - 0.5*BOX_SIZE):int(cx + 0.5*BOX_SIZE), int(cy - 0.5*BOX_SIZE):int(cy + 0.5*BOX_SIZE)]
        return cropped

    def crop_images(self) -> None:
        for i in range(len(self._df)):
            image_files = [file for file in self._files if file.startswith(self._df['IMAGE_ID'].iloc[i])]
            if len(image_files) > 0:
                image = plt.imread(f'{self._path}/{image_files[0]}')
                cropped = self._crop_image(image,i)
                plt.imsave(f"{self._path}/cropped/{self._df['IMAGE_ID'].iloc[i]}.png",cropped,cmap = 'grey')

if __name__ == '__main__':
    dir_name = sys.argv[1]
    bbox_maker  = BBoxCropper(dir_name)
    bbox_maker.crop_images()