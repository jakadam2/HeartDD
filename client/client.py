import sys
import os
from enum import Enum
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import queue
import numpy.typing as npt
import file_handler as fh
import server_communicator as sch
from testing.box_test2 import ResizableCanvasShape

WIDTH, HEIGHT = 800, 800
test_file = "/home/michal/Documents/Studia/Inzynierka/HeartDD/base_images/12aw4ack71831bocuf5j3pz235kn1v361de_33.png"

class Flag(Enum):
    LOAD = 1
    DETECT = 2
    DESCRIBE = 3
    EXIT = 4
    NOTHING = 5

class WindowController:
    def __init__(self, window, client):
        self.window = window
        self.client = client  # Reference to Client class to call application methods
        self.canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
        self.canvas.pack()

        # Initialize buttons and connect to client methods
        self.load_button = tk.Button(self.window, text="Load Image", command=self.load_action)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.detect_button = tk.Button(self.window, text="Detect Lesions", command=self.detect_action)
        self.detect_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.describe_button = tk.Button(self.window, text="Describe Lesions", command=self.describe_action)
        self.describe_button.pack(side=tk.LEFT, padx=5, pady=5)

    def display_image(self, image_tk):
        """Display image on the canvas."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.window.update()

    def display_bboxes(self, bounding_boxes):
        """Display bounding boxes on the canvas."""
        for bbox in bounding_boxes:
            ResizableCanvasShape(self.canvas, bbox)
    
    def load_action(self):
        """Action to load a file in a separate thread."""
        threading.Thread(target=self.client.load_file, daemon=True).start()

    def detect_action(self):
        """Action to detect lesions in a separate thread."""
        threading.Thread(target=self.client.request_detect, daemon=True).start()

    def describe_action(self):
        """Action to describe lesions in a separate thread."""
        threading.Thread(target=self.client.request_describe, daemon=True).start()


class Client:
    def __init__(self, window):
        self.image = None
        self.image_tk = None
        self.bitmask = None
        self.bounding_boxes = None
        self.queue = queue.Queue()
        self.shapes = []

        # Initialize WindowController for window-related tasks
        self.window_controller = WindowController(window, self)
        
        # Initialize file handler and server communicator
        self.files = fh.FileHandler()
        self.server = sch.ServerHandler()
        
        # Load file and start polling for UI updates
        self.load_file() 
        self.poll_queue()

    def poll_queue(self):
        """Poll queue for updates and perform actions based on flags."""
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        self.window_controller.display_bboxes(self.bounding_boxes)
                    case Flag.DESCRIBE:
                        self.display_confidence()  # Assuming a method to show confidence data
                    case Flag.LOAD:
                        self.window_controller.display_image(self.image_tk)
                    case _:
                        pass
        except queue.Empty:
            pass
        except Exception as ex:
            print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")
        
        # Continue polling every 100 ms
        self.window_controller.window.after(100, self.poll_queue)


    def load_file(self):
        """Load the image and bitmask, then update queue for display."""
        try:
            self.files.get_file_name(name=test_file)
            self.image = self.files.load_file()
            self.image_tk = ImageTk.PhotoImage(self.image.resize((WIDTH, HEIGHT)))
            self.bitmask = self.files.load_bitmask()
            self.queue.put(Flag.LOAD)
        except ValueError as ex:
            print(f"An error occurred while loading a file {ex.args}")


    def request_detect(self):
        """Request lesion detection from server and update queue."""
        if self.image is None:
            print("No image loaded")
            return
        self.bounding_boxes = self.server.request_detection(self.image, self.bitmask)
        self.queue.put(Flag.DETECT)


    def request_describe(self):
        """Request lesion description from server and update queue."""
        if self.bounding_boxes is None:
            print("No bounding boxes present")
            return
        for bbox in self.bounding_boxes:
            self.server.request_description(self.image, self.bitmask, bbox)
        self.queue.put(Flag.DESCRIBE)

if __name__ == "__main__":
    root = tk.Tk()
    client = Client(root)
    root.mainloop()
