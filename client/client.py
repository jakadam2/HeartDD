import sys
import os

# Add the project root directory to sys.path
root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root_dir))

import preprocessing.mask_to_pixel as preprocessing
import grpc
import detection_grpc.detection_pb2_grpc as comms_grpc
import detection_grpc.detection_pb2 as comms
import pydicom as dicom
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from pydicom.data import get_testdata_file
import numpy as np
import threading
import time
import pandas as pd
import re

WIDTH, HEIGHT = 800, 800
CHUNK_SIZE = 1024
MASK_FILE = os.path.join(root_dir, 'base_images', 'good_df_newest.csv')


def process_bounding_boxes(response, scaling_ratio: float):
    # Initialize an empty list to store the rescaled bounding boxes
    bounding_boxes = []

    if response.status.success == comms.Status.SUCCESS:
        print("Bounding boxes detected:")
        for coordinates in response.coordinates_list:
            # Rescale the coordinates using the scaling_ratio
            x1 = coordinates.x1 * scaling_ratio
            y1 = coordinates.y1 * scaling_ratio
            x2 = coordinates.x2 * scaling_ratio
            y2 = coordinates.y2 * scaling_ratio

            print(f"Coordinates: x1={coordinates.x1}, y1={coordinates.y1}, "
                  f"x2={coordinates.x2}, y2={coordinates.y2}")
            # Append the rescaled bounding box to the list
            bounding_boxes.append([x1, y1, x2, y2])

    return bounding_boxes


def load_file(filename: str) -> Image:
    # Validate file extension
    filename_split = filename.rsplit(".", 1)
    if len(filename_split) < 2:
        raise ValueError("Incorrect file format, all files should have a .png, .jpg or .dcm extension")

    extension = filename_split[1].lower()  # Convert extension to lowercase for consistency
    try:
        match extension:
            case "dcm":
                # Load DICOM file and convert to PIL Image
                file = dicom.dcmread(filename)
                pixel_array = file.pixel_array  # Get pixel array from the DICOM file

                # Normalize the pixel data and convert it to uint8 for display (if needed)
                pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
                pixel_array = pixel_array.astype(np.uint8)

                # Check if it's a grayscale image (2D array) or RGB-like (3D array)
                image = Image.fromarray(pixel_array)  # 'L' for grayscale

            case "png" | "jpg":
                # Load PNG/JPG file directly as PIL Image
                image = Image.open(filename)

            case _:
                raise ValueError("Incorrect file format, all files should have a .png, .jpg or .dcm extension")

    except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            raise ValueError(message)

    return image


def load_bitmask(filename: str):
    name = os.path.basename(filename).removesuffix(".png")
    # Parse the image_id and frame from the filename
    image_id, frame = name.rsplit("_", 1)
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
        return exact_frame_row['bitmask'].values[0]

    # Find the closest frame if exact match isn't found
    closest_frame_row = image_rows.iloc[(image_rows['frame'] - frame).abs().argsort()[:1]]
    closest_frame = closest_frame_row['frame'].values[0]
    closest_bitmask = closest_frame_row['segmentation'].values[0]
    print(f"Exact frame not found. Using closest frame {closest_frame} for image_id {image_id}.")
    return closest_bitmask

def generate_detection_request(image: Image):
    try:
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        pixel_data = image_array.tobytes()

        num_chunks = len(pixel_data) // CHUNK_SIZE + (1 if len(pixel_data) % CHUNK_SIZE else 0)
        print(f"{width}x{height}| {len(image_array)} | {len(pixel_data)}")
        # Then, yield pixel data in chunks
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            yield comms.DetectionRequest(height=height, width=width, image=pixel_data[start:end])

    except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def generate_desc_request(image: Image, bbox):
    try:
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        pixel_data = image_array.tobytes()

        coordinates = comms.Coordinates(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
        num_chunks = len(pixel_data) // CHUNK_SIZE + (1 if len(pixel_data) % CHUNK_SIZE else 0)
        print(f"{width}x{height}| {len(image_array)} | {len(pixel_data)}")
        # Then, yield pixel data in chunks
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = start + CHUNK_SIZE
            yield comms.DescriptionRequest(height=height, width=width, image=pixel_data[start:end], coords = coordinates)

    except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)


def add_bounding_boxes_to_canvas(canvas, bounding_boxes):
    """Adds bounding boxes to the already displayed image."""
    for box in bounding_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


def display_image(image, canvas, image_container):
    """Displays the image immediately."""

    # Convert the numpy array to a PIL image
    # Resize image to fit window
    image = image.resize((WIDTH, HEIGHT))

    # Convert to ImageTk for Tkinter
    image_tk = ImageTk.PhotoImage(image)

    # Clear existing image, if any
    canvas.delete("all")

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    
    # Save the reference to avoid garbage collection
    image_container["image"] = image_tk


def run_client():

     # Set up gRPC connection
    channel = grpc.insecure_channel('localhost:50051')
    stub = comms_grpc.DetectionAndDescriptionStub(channel)

    filename = askopenfilename()
    image = load_file(filename)
    packed_mask = load_bitmask(filename)
    mask = preprocessing.MaskUnpacker._unpack_mask(packed_mask)
    print(mask)

    if image is None:
        print("OwO, something went howwibly bad. Sowwwwy T~T")
        return None  # Exit if no file is loaded

    # Initialize Tkinter window
    window = tk.Tk()
    window.title("DICOM Image Viewer with Bounding Boxes")

    # Set window size and create a canvas
    canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
    canvas.pack()

    # Dictionary to hold the reference to the image (to prevent garbage collection)
    image_container = {}
    # Display the image immediately
    display_image(image, canvas, image_container)

    # Start a separate thread to send the gRPC request and draw bounding boxes
    #threading.Thread(
    #    target=server_communication_handler,
    #    args=(stub, canvas, file),
    #    daemon=True
    #).start()

    # Run the Tkinter main loop
    window.mainloop()


def server_communication_handler(stub, canvas, image):
    detection_request = generate_detection_request(image)
    bounding_boxes = None
    try:
        response = stub.GetBoundingBoxes(detection_request)
        if response.status.success == comms.Status.SUCCESS:
            """Thread function to send the gRPC request and draw bounding boxes on the image."""
            # Scale the bounding boxes using the scaling ratio
            im_width, im_height = image.size
            scaling_ratio = WIDTH / im_width
            print(f"Scaling ratio: {scaling_ratio}")
            bounding_boxes = process_bounding_boxes(response, scaling_ratio)
            add_bounding_boxes_to_canvas(canvas, bounding_boxes)
        else:
            print(f"Error received from the server {response.status.err_message}") 
    except grpc.RpcError as er:
        print(f"gRPC erorr: {er}")

    if bounding_boxes is None:
        print("Bounding boxes not found ")
        return

    for bbox in bounding_boxes:
        description_request = generate_desc_request(image, bbox)     
        try: 
            response = stub.GetDescription(description_request)
            if response.status.success == comms.Status.SUCCESS:
                print(f"Bounding box ({bbox[0]},{bbox[1]})  ({bbox[2]},{bbox[3]})")
                for conf in response.confidence_list:
                    print(f"{conf.name}:{conf.confidence}")
        except grpc.RpcError as er:
            print(f"gRPC error: {er}")



if __name__ == "__main__":
    run_client()
