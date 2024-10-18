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
import threading
import time

WIDTH, HEIGHT = 400, 400

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


def generate_desc_request(dicom, bbox):
    pixel_data = dicom.pixel_array.tobytes()
    width = dicom.Columns
    height = dicom.Rows
    chunk_size = 1024 * 1024
    coordinates = comms.Coordinates(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    for i in range(0, len(pixel_data), chunk_size):
        chunk = pixel_data[i: i+chunk_size]
        yield comms.DescriptionRequest(
            coords = coordinates,
            width=width,
            height=height,
            image=chunk
        )


def add_bounding_boxes_to_canvas(canvas, bounding_boxes):
    """Adds bounding boxes to the already displayed image."""
    for box in bounding_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


def display_image(dicom, width, height, canvas, image_container):
    """Displays the image immediately."""
    pixel_array = dicom.pixel_array
    normalized_pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    normalized_pixel_array = normalized_pixel_array.astype(np.uint8)

    # Convert the numpy array to a PIL image
    if normalized_pixel_array.ndim == 2:  # Grayscale image
        image = Image.fromarray(normalized_pixel_array)
    else:
        image = Image.fromarray(normalized_pixel_array)

    # Resize image to fit window
    image = image.resize((width, height))

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

    file = load_dicom()
    if file is None:
        return  # Exit if no file is loaded

    # Initialize Tkinter window
    window = tk.Tk()
    window.title("DICOM Image Viewer with Bounding Boxes")

    # Set window size and create a canvas
    WIDTH, HEIGHT = 500, 500  # Adjust this based on your image size
    canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
    canvas.pack()

    # Dictionary to hold the reference to the image (to prevent garbage collection)
    image_container = {}
    # Display the image immediately
    display_image(file, WIDTH, HEIGHT, canvas, image_container)

    # Start a separate thread to send the gRPC request and draw bounding boxes
    threading.Thread(
        target=server_communication_handler,
        args=(stub, canvas, file),
        daemon=True
    ).start()

    # Run the Tkinter main loop
    window.mainloop()

def server_communication_handler(stub, canvas, file):
    detection_request = generate_image_request(file)
    bounding_boxes = None
    try:
        response = stub.GetBoundingBoxes(detection_request)
        if response.status.success == comms.Status.SUCCESS:
            """Thread function to send the gRPC request and draw bounding boxes on the image."""
            # Stream the DICOM file's pixel data and receive the response
            # Scale the bounding boxes using the scaling ratio
            scaling_ratio = WIDTH / file.Columns
            bounding_boxes = process_bounding_boxes(response, scaling_ratio)
            add_bounding_boxes_to_canvas(canvas, bounding_boxes)
        else:
            print("Error code: 500") 
    except grpc.RpcError as er:
        print(f"gRPC erorr: {er}")

    if bounding_boxes is None:
        print("Something went wrong line: 204")
        return

    for bbox in bounding_boxes:
        description_request = generate_desc_request(file, bbox)     
        try: 
            response = stub.GetDescription(description_request)
            if response.status.success == comms.Status.SUCCESS:
                print(f"Bounding box {bbox[0]}{bbox[1]}{bbox[2]}{bbox[3]}")
                for conf in response.confidence_list:
                    print(f"{conf.name}:{conf.confidence}")
        except grpc.RpcError as er:
            print(f"gRPC error: {er}")



if __name__ == "__main__":
    run_client()