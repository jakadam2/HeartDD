import pandas as pd
import sys
import cv2 as cv
import numpy as np
import os
import re


class Annotation:

    def __init__(self,path:str) -> None:
        self._path = path
        self._files = os.listdir(path)
        self.annotations_folder = os.path.join(path, f"labels")
        os.makedirs(self.annotations_folder, exist_ok=True)
        data_file = [file for file  in self._files if re.search('.+.csv',file) is not None]
        if len(data_file) != 1:
            raise ValueError('IN FILE WITH IMAGES SHOULD BE EXACTLY ONE CSV FILE')
        self._df = pd.read_csv(f'{path}/{data_file[0]}',sep = ',')

    def get_coordinates(self, idx:int) -> np.ndarray:
        x,y = self._df['lesion_x'].iloc[idx],self._df['lesion_y'].iloc[idx]
        w =self._df['lesion_width'].iloc[idx] 
        h = self._df['lesion_height'].iloc[idx]

        return x, y, w, h

    def make_annotations(self) -> None:
        for i in range(len(self._df)):
            image_files = [file for file in self._files if file.startswith(self._df['image_id'].iloc[i])]
            for image_file in image_files:
                annotation = ""
                if self._df['lesion_x'].iloc[i] > -1:
                    # If the lesion is present, get its coordinates
                    image = cv.imread(f'{self._path}/{image_file}')
                    x, y, w, h = self.get_coordinates(i)
                    image_height, image_width = image.shape[:2]

                    x_center = abs((x + w / 2) / image_width)
                    y_center = abs((y + h / 2) / image_height)
                    width_norm = abs(w / image_width)
                    height_norm = abs(h / image_height)
                    
                    # Create annotation line in YOLO format
                    annotation = f"0 {x_center} {y_center} {width_norm} {height_norm}\n"

                file_name, _ = os.path.splitext(image_file)
                annotation_file = os.path.join(self.annotations_folder, f"{file_name}.txt")
                
                # Append the annotation to the file
                with open(annotation_file, 'a') as f:
                    f.writelines(annotation)


if __name__ == '__main__':
    dir_name = './datasets/images/covered'
    annonation  = Annotation(dir_name)
    annonation.make_annotations()


'''
USAGE:
Put csv file with description in the same directory as where you put images
Run commend: python make_yolo_annotations.py "directory path"
'''