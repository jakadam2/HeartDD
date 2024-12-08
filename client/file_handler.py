import sys
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfile
import pydicom as dicom
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from config import *

sys.path.append(os.path.abspath(root_dir))
MASK_FILE = os.path.join(root_dir,
                         parser.get("DEFAULT", "bitmask_file"))
ALLOWED_EXTENSION = {".png", ".jpg", ".dcm"}

def get_file_name(name: str | None) -> (str, str):
    if name is None:
        name = askopenfilename()

    filename, extension = os.path.splitext(name)
    # Validate file extension
    if extension is None or extension not in ALLOWED_EXTENSION:
        raise NameError("Incorrect file format, all files should have a .png, .jpg or .dcm extension")
    elif filename is None:
        raise TypeError("No file has been loaded")
    return filename, extension


def load_file(filename: str, extension: str) -> Image:
    match extension:
        case ".dcm":
            # Load DICOM file and convert to PIL Image
            file = dicom.dcmread(filename + extension)
            pixel_array = file.pixel_array  # Get pixel array from the DICOM file
            if len(pixel_array) > 1:
                pixel_array = pixel_array[0]

            # Normalize the pixel data and convert it to uint8 for display (if needed)
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
            pixel_array = pixel_array.astype(np.uint8)

            # Check if it's a grayscale image (2D array) or RGB-like (3D array)
            image = Image.fromarray(pixel_array)  # 'L' for grayscale
        case ".png" | ".jpg":
            # Load PNG/JPG file directly as PIL Image
            image = Image.open(filename + extension)
        case _:
            raise RuntimeError()
    return image


def load_bitmask(filename: str):
    import preprocessing.mask_to_pixel as preprocessing
    basename = os.path.basename(filename)
    # Parse the image_id and frame from the filename
    image_id, frame = basename.rsplit("_", 1)
    if not image_id or not frame:
        raise TypeError("Filename format is incorrect. Expected {image_id}_{frame}.png")

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

    mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)
    return mask


def load_directory(dir: str | None = None):
    if dir is None:
        dir = askdirectory()
    # List all files in the directory
    filenames = next(os.walk(dir), (None, None, []))[2]

    # Filter for files with .png, .jpg, or .dcm extensions (case insensitive)
    filtered_filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSION]
    full_names = [dir + "/" + f for f in filtered_filenames]
    return full_names, dir

def save_image(filename: str, directory: str, image: Image, bounding_boxes: list[list[float]], confidence_list: list[dict[str,float]]):
    if filename is None:
        dest = asksaveasfile(defaultextension=".png").name
    else:
        output_dir = os.path.join(directory, "results")
        os.makedirs(output_dir, exist_ok=True)
        dest = os.path.join(output_dir, f"{os.path.basename(filename)}_result.png")
    # Create a copy of the image for drawing
    if isinstance(image, np.ndarray):  # If the image is already a NumPy array
        output_image = image.copy()
    else:
        output_image = np.array(image)  # Convert PIL Image to NumPy array if needed
    # Draw the bounding boxes
    for bbox, confidence in zip(bounding_boxes, confidence_list):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        label, max_prob = max(confidence.items(), key=lambda item: item[1])
        label = f"{label}"  # Format label
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    # Define the output path

    cv2.imwrite(dest, output_image)

