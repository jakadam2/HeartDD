import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
import detection_grpc.detection_pb2_grpc as comms_grpc
import detection_grpc.detection_pb2 as comms
import pydicom as dicom
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from pydicom.data import get_testdata_file
import numpy as np

WIDTH, HEIGHT = 400, 400


def display_image_with_bounding_boxes(dicom, bounding_boxes, width: int, height: int):
    window = tk.Tk()
    window.title("DICOM Image with Bounding Boxes")

    # Convert pixel data to a format that can be displayed in Tkinter
    pixel_array = dicom.pixel_array
    normalized_pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    normalized_pixel_array = normalized_pixel_array.astype(np.uint8)

    # Convert the numpy array to a PIL image
    if normalized_pixel_array.ndim == 2:  # Grayscale image
        image = Image.fromarray(normalized_pixel_array)
    else:
        # If it's a multi-channel image (RGB), handle accordingly
        image = Image.fromarray(normalized_pixel_array)

    # Resize image to fit window if needed (optional)
    image = image.resize((width, height))

    # Convert PIL image to ImageTk for Tkinter
    image_tk = ImageTk.PhotoImage(image)

    # Create a Canvas to display the image
    canvas = tk.Canvas(window, width=width, height=height)
    canvas.pack()
    # Display the image on the Canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    # Draw bounding boxes on the image
    for box in bounding_boxes:
        print(box[0], box[1], box[2], box[3])
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    window.mainloop()


def load_dicom():
    #filename = askopenfilename()
    filename = get_testdata_file("CT_small.dcm")
    extension = filename.rsplit(".", 1)[1] 
    file = None
    match extension:
        case "dcm":
            file = dicom.dcmread(filename)
        case "png" | "jpg":
            file = Image.open(filename)
        case _:
            print("Incorrect file format, files must be dicom, png or jpg")
            return None
    if file == None:
        print("OwO, something went howwibly bad. Sowwwwy T~T")
        return None 
    return file


def generate_image_request(dicom_file):
    pixel_data = dicom_file.pixel_array.tobytes()
    width = dicom_file.Columns
    height = dicom_file.Rows
    chunk_size = 1024 * 1024
    for i in range(0, len(pixel_data), chunk_size):
        chunk = pixel_data[i: i+chunk_size]
        yield comms.DetectionRequest(
            width=width,
            height=height,
            image=chunk
        )


def run_client():

    channel = grpc.insecure_channel('localhost:50051')
    stub = comms_grpc.DetectionAndDescriptionStub(channel)
    file = load_dicom()
    request = generate_image_request(file)
    
    # Stream the DICOM file's pixel data and receive the response
    response = stub.GetBoundingBoxes(request)
    # Handle the response
    bounding_boxes = None
    if response.status.success == comms.Status.SUCCESS:
        print("Bounding boxes detected:")
        bounding_boxes = process_bounding_boxes(response, (WIDTH/file.Columns))
    else:
        print("Error: Image processing failed.")
    if bounding_boxes is None:
        print("Error: Drawing bounding boexs failed")


    display_image_with_bounding_boxes(file, bounding_boxes, WIDTH, HEIGHT)


def process_bounding_boxes(response, scaling_ratio):
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



if __name__ == "__main__":
    run_client()