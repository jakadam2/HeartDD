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

class Flag(Enum):
    LOAD = 1
    DETECT = 2
    DESCRIBE = 3
    EXIT = 4
    NOTHING = 5

class Client:
    def __init__(self, window):
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
        self.load_button.pack()

        self.files = fh.FileHandler()
        self.server = sch.ServerHandler()
        
        self.poll_queue()


    def display_image(self):
        print("display called")
        # Convert to ImageTk.PhotoImage for displaying in Tkinter
        self.canvas.delete("all")  # Clear existing items on the canvas
        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        # Update the canvas to ensure the image renders
        self.canvas.update()


    def load_action(self):
        threading.Thread(target=self.load_file, daemon=True).start()

    def load_file(self):
        try:
            image = self.files.load_file().resize((WIDTH, HEIGHT))
            self.image_tk = ImageTk.PhotoImage(image)
            self.bitmask = self.files.load_bitmask()
            self.queue.put(Flag.LOAD)
        except Exception as ex:
            print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")


    def poll_queue(self):
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        request_detection()
                    case Flag.DESCRIBE:
                        request_description()
                    case Flag.LOAD:
                        self.display_image()
                    case _:
                        pass 
        except queue.Empty:
            pass
        except Exception as ex:
            print(f"An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")
        self.window.after(100, self.poll_queue)


def request_detection():
    pass

def request_description():
    pass

if __name__ == "__main__":
    root = tk.Tk()
    client = Client(root)
    root.mainloop()
