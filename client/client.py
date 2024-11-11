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

WIDTH, HEIGHT = 800, 800
test_file = "/home/michal/Documents/Studia/Inzynierka/HeartDD/base_images/12aw4ack71831bocuf5j3pz235kn1v361de_33.png"

class Flag(Enum):
    LOAD = 1
    DETECT = 2
    DESCRIBE = 3
    EXIT = 4
    NOTHING = 5

class Client:
    def __init__(self, window):
        self.image = None
        self.image_tk = None
        self.bitmask = None
        self.bounding_boxes = None
        self.queue = queue.Queue()

        self.window = window
        
        # Set window size and create a canvas
        self.canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
        self.canvas.pack()
        
        # Button to trigger the load action
        self.load_button = tk.Button(self.window, text="Load Image", command=self.load_action)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.detect_button = tk.Button(self.window, text="Detect Lesions", command=self.detect_action)
        self.detect_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.describe_button = tk.Button(self.window, text="Describe Lesions", command=self.describe_action)
        self.describe_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.files = fh.FileHandler()
        self.server = sch.ServerHandler()
        self.load_file() 
        self.poll_queue()


    def poll_queue(self):
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        self.display_bboxes()
                    case Flag.DESCRIBE:
                        self.display_confidence()
                    case Flag.LOAD:
                        self.display_image()
                    case _:
                        pass 
        except queue.Empty:
            pass
        except Exception as ex:
            print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")
        self.window.after(100, self.poll_queue)
    

    def display_image(self):
        print("display called")
        # Convert to ImageTk.PhotoImage for displaying in Tkinter
        self.canvas.delete("all")  # Clear existing items on the canvas
        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        # Update the canvas to ensure the image renders
        self.window.update()


    def load_action(self):
        threading.Thread(target=self.load_file, daemon=True).start()

    def load_file(self):
        try:
            #self.files.get_file_name()
            self.files.get_file_name(name=test_file)
            self.image = self.files.load_file()
            self.image_tk = ImageTk.PhotoImage(self.image.resize((WIDTH, HEIGHT)))
            self.bitmask = self.files.load_bitmask()
            self.queue.put(Flag.LOAD)
        except ValueError as ex:
            print(f"An error occured while loading a file {ex.args}")


    def detect_action(self):
        threading.Thread(target=self.request_detect, daemon=True).start()

    def request_detect(self):
        if self.image is None:
            print("No image loaded")
            return
        self.bounding_boxes = self.server.request_detection(self.image, self.bitmask)
        self.queue.put(Flag.DETECT)

    def describe_action(self):
        threading.Thread(target=self.request_describe, daemon=True).start()

    def request_describe(self):
        if self.bounding_boxes is None:
            print("No bounding boxes present")
            return
        for bbox in self.bounding_boxes:
            self.server.request_description(self.image,self.bitmask, bbox)
        self.queue.put(Flag.DESCRIBE)


if __name__ == "__main__":
    root = tk.Tk()
    client = Client(root)
    root.mainloop()
