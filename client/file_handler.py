import os
import sys
from tkinter.filedialog import askopenfilename
from pathlib import Path

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root_dir))
MASK_FILE = os.path.join(root_dir, 'base_images', 'good_df_newest.csv')

import preprocessing.mask_to_pixel as preprocessing
import pydicom as dicom
from PIL import Image
import pandas as pd


class FileHandler:
    def __init__(self, name = ""):
        self.filename = ""
        self.extension = ""


    def get_file_name(self, name=None):
        if name is None:
            name = askopenfilename()

        self.filename, self.extension = os.path.splitext(name)
        # Validate file extension
        if self.extension is None:
            raise ValueError("Incorrect file format, all files should have a .png, .jpg or .dcm extension")
        elif self.filename is None:
            raise ValueError("No file has been loaded")


    def load_file(self) -> Image:

        match self.extension:
            case ".dcm":
                # Load DICOM file and convert to PIL Image
                file = dicom.dcmread(self.filename + self.extension)
                pixel_array = file.pixel_array  # Get pixel array from the DICOM file

                # Normalize the pixel data and convert it to uint8 for display (if needed)
                pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
                pixel_array = pixel_array.astype(np.uint8)

                # Check if it's a grayscale image (2D array) or RGB-like (3D array)
                image = Image.fromarray(pixel_array)  # 'L' for grayscale
            case ".png" | ".jpg":
                # Load PNG/JPG file directly as PIL Image
                image = Image.open(self.filename + self.extension)
            case _:
                raise ValueError("Incorrect file format, all files should have a .png, .jpg or .dcm extension")

        return image


    def load_bitmask(self):
        basename = os.path.basename(self.filename)
        # Parse the image_id and frame from the filename
        image_id, frame = basename.rsplit("_", 1)
        if not image_id or not frame:
            raise ValueError("Filename format is incorrect. Expected {image_id}_{frame}.png")

        frame = int(frame)
        # Load the CSV file
        df = pd.read_csv(MASK_FILE)
        # Filter rows with the correct image_id
        image_rows = df[df['image_id'] == image_id]
        
        if image_rows.empty:
            raise ValueError(f"No entry found for image_id {image_id} in the CSV file.")
        # Check if the exact frame exists
        exact_frame_row = image_rows[image_rows['frame'] == frame]
        if not exact_frame_row.empty:
            bitmask = exact_frame_row['segmentation'].values[0]
        else:
            # Find the closest frame if exact match isn't found
            closest_frame_row = image_rows.iloc[(image_rows['frame'] - frame).abs().argsort()[:1]]
            closest_frame = closest_frame_row['frame'].values[0]
            bitmask = closest_frame_row['segmentation'].values[0]
            print(f"[CLIENT] Exact frame not found. Using closest frame {closest_frame} for image_id {image_id}.")

        mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)
        return mask