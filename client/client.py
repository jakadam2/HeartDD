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
        self.load_button = tk.Button(self.window, text="Load Image", command=self.on_load)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.detect_button = tk.Button(self.window, text="Detect Lesions", command=self.on_detect)
        self.detect_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.describe_button = tk.Button(self.window, text="Describe Lesions", command=self.on_describe)
        self.describe_button.pack(side=tk.LEFT, padx=5, pady=5)

    def display_image(self, image_tk):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.window.update()

    def display_bboxes(self, bounding_boxes):
        for bbox in bounding_boxes:
            ResizableCanvasShape(self.canvas, bbox)

    def display_confidence(self):
        pass
    
    def on_load(self):
        threading.Thread(target=self.client.load_file, daemon=True).start()

    def on_detect(self):
        threading.Thread(target=self.client.request_detect, daemon=True).start()

    def on_describe(self):
        threading.Thread(target=self.client.request_describe, daemon=True).start()


class Client:
    def __init__(self, window):
        self.image = None
        self.image_tk = None
        self.bitmask = None
        self.bounding_boxes = None
        self.confidence_list = None
        self.queue = queue.Queue()
        self.shapes = []

        # Initialize WindowController for window-related tasks
        self.window_controller = WindowController(window, self)
        
        # Initialize file handler and server communicator
        self.files = fh.FileHandler()
        self.server = sch.ServerHandler()
        
        # Load file and start polling for UI updates
        self.load_file(test_file)
        self.poll_queue()

    def poll_queue(self):
        """Poll queue for updates and perform actions based on flags."""
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        self.window_controller.display_bboxes( self.scalebboxes() )
                    case Flag.DESCRIBE:
                        self.window_controller.display_confidence()  # Assuming a method to show confidence data
                    case Flag.LOAD:
                        self.window_controller.display_image(self.image_tk)
                    case _:
                        pass
        except queue.Empty:
            pass
        except Exception as ex:
            print(f"[CLIENT] An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")
        
        # Continue polling every 100 ms
        self.window_controller.window.after(100, self.poll_queue)

    def scalebboxes(self):
        img_width, img_height = self.image.size
        scaling_ratio = WIDTH / img_width
        scaled_bboxes = []
        for bbox in self.bounding_boxes:
            scaled_bboxes.append([ 
                bbox[0] * scaling_ratio,
                bbox[1] * scaling_ratio,
                bbox[2] * scaling_ratio,
                bbox[3] * scaling_ratio
            ])
        return scaled_bboxes

    def load_file(self, name=None):
        try:
            self.files.get_file_name(name)
            self.image = self.files.load_file()
            self.image_tk = ImageTk.PhotoImage(self.image.resize((WIDTH, HEIGHT)))
            self.bitmask = self.files.load_bitmask()
            self.queue.put(Flag.LOAD)
        except ValueError as ex:
            print(f"[CLIENT] An error occurred while loading a file {ex.args}")


    def request_detect(self):
        if self.image is None:
            print("[CLIENT] No image loaded")
            return
        self.bounding_boxes = self.server.request_detection(self.image, self.bitmask)
        self.queue.put(Flag.DETECT)


    def request_describe(self):
        if self.bounding_boxes is None:
            print("[CLIENT] No bounding boxes present")
            return
        self.server.request_description(self.image, self.bitmask, self.bounding_boxes)
        self.queue.put(Flag.DESCRIBE)


def start():
    root = tk.Tk()
    client = Client(root)
    root.mainloop()


if __name__ == "__main__":
    start()