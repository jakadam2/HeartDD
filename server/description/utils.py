import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

import numpy as np

import os
import pandas as pd
from PIL import Image


class ImageDataset(Dataset):
    """
    A custom dataset class for loading and processing images and their corresponding annotations.

    This dataset is designed to work with image data organized in a CSV file containing annotations.
    ------------
    Mandatory fields in the CSV:
    - IMAGE_NAME: The name of the image file.
    - X: The x-coordinate of the top-left corner of the bounding box.
    - Y: The y-coordinate of the top-left corner of the bounding box.
    - WIDTH: The width of the bounding box.
    - HEIGHT: The height of the bounding box.

    The dataset also supports optional transformations for both images and labels.

    Args:
    ----------
    BOX_SIZE : int
        Size of the square bounding box to crop around the center of the target area.
    
    _df : pd.DataFrame
        DataFrame containing image annotations filtered by required fields.
    
    img_dir : str
        Directory where the images are stored.
    
    class_names : list[str]
        List of class names corresponding to the labels in the dataset.
    
    transform : torchvision.transforms, optional
        Optional transformations to be applied to the images.
    
    target_transform : torchvision.transforms, optional
        Optional transformations to be applied to the labels.
    
    pll : torchvision.transforms.Compose
        Composed transformation pipeline for converting images to tensor format.
    
    Methods:
    -------
    __len__() -> int
        Returns the number of samples in the dataset.
    
    __getitem__(idx: int) -> tuple
        Retrieves the image and its corresponding labels for a given index.
    """
    BOX_SIZE = 55

    def __init__(self,
                annotations_file: pd.DataFrame,
                img_dir: str,
                class_names: list[str],
                transform: torchvision.transforms = None,
                target_transform: torchvision.transforms = None,
                ) -> None:
        self._df = annotations_file[['X','Y','WIDTH','HEIGHT','IMAGE_NAME'] + class_names]
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.pll = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self._df['IMAGE_NAME'].iloc[idx])
        image = Image.open(img_path)
        image = self.pll(image).to(torch.float32).squeeze()

        x,y = self._df['X'].iloc[idx],self._df['Y'].iloc[idx]
        w =self._df['WIDTH'].iloc[idx] 
        h = self._df['HEIGHT'].iloc[idx]
        cx = x + 0.5*w
        cy = y + 0.5*h

        cropped = image[int(cx - 0.5*ImageDataset.BOX_SIZE):int(cx + 0.5*ImageDataset.BOX_SIZE), int(cy - 0.5*ImageDataset.BOX_SIZE):int(cy + 0.5*ImageDataset.BOX_SIZE)].unsqueeze(0)
        labels = torch.from_numpy(self._df[self.class_names].iloc[idx].values).to(torch.float32)

        cropped = self.transform(cropped) if self.transform is not None else cropped
        labels = self.target_transform(labels) if self.target_transform is not None else labels
        
        return cropped, labels

def train_test_split_pandas(df, test_size=0.2, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    shuffled_indices = np.random.permutation(len(df))
    
    test_set_size = int(len(df) * test_size)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]
    
    return train_df, test_df


