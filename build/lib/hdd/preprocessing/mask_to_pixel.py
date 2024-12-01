import zlib
import base64
import numpy as np
import pandas as pd
import cv2 as cv
from os import path


class MaskUnpacker:

    def __init__(self,source_file:str,target_directory:str) -> None:
        self.source_file = source_file
        self.target_directory = target_directory

    @staticmethod
    def _unpack_mask(mask, shape=(512, 512)):
        """Unpack segmentation mask sent in HTTP request.

        Args:
            mask (bytes): Packed segmentation mask.

        Returns:
            np.array: Numpy array containing segmentation mask.
        """
        mask = base64.b64decode(mask)
        mask = zlib.decompress(mask)
        mask = list(mask)
        mask = np.array(mask, dtype=np.uint8)
        mask = mask.reshape(-1, *shape)
        mask = mask.squeeze()
        return mask
    
    def unpack_masks_to_dir(self) -> None:
        data = pd.read_csv(self.source_file,usecols=['image_id','segmentation'])
        for _,row in data.iterrows():
            name = row['image_id']
            try:
                mask = self._unpack_mask(row['segmentation'])
                mask[mask > 0] = 255
                
                cv.imwrite(path.join(self.target_directory,f'{name}_binmask.png'),mask)
            except:
                print(name)


if __name__ == '__main__':
    mask_unpacker = MaskUnpacker('./datasets/images/zipped_mask.csv','./datasets/images/masks')
    mask_unpacker.unpack_masks_to_dir()