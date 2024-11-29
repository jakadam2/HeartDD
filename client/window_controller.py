from testing.box_test3 import ResizableCanvasShape
import threading
from PIL import ImageTk

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

WIDTH, HEIGHT = 700, 700


class WindowController:
    def __init__(self, client, window):
        self.window = window
        self.client = client  # Reference to Client class to call application methods

        self.window.title("Lesion Detection")
        self.window.config(bg="skyblue")

        self.image_frame = tk.Frame(self.window, width=WIDTH, height=HEIGHT, bg='white')
        self.image_frame.grid(row=0, column=1, padx=15, pady=15)
        self.canvas = tk.Canvas(self.image_frame, width=WIDTH, height=HEIGHT)
        # self.canvas.grid(row=0, column = 0)
        self.canvas.pack()

        self.confidence_frame = tk.Frame(self.window, width=400, height=500)
        self.confidence_frame.grid(row=0, column=0, padx=5, pady=5)

        self.button_frame = tk.Frame(self.window, width=250, height=250, bg='skyblue')
        self.button_frame.grid(row=1, column=0, padx=5, pady=5)

        # Initialize buttons and connect to client methods
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.on_load)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.detect_button = tk.Button(self.button_frame, text="Detect Lesions", command=self.on_detect)
        self.detect_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.describe_button = tk.Button(self.button_frame, text="Describe Lesions", command=self.on_describe)
        self.describe_button.pack(side=tk.LEFT, padx=5, pady=5)

    def display_image(self, image_tk: ImageTk):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.window.update()

    def display_bboxes(self, bounding_boxes: list[list]):
        for bbox in bounding_boxes:
            ResizableCanvasShape(self.canvas, bbox)

    def display_confidence(self, bboxes: list[list], confidence_list: dict):
        for idx, key, value in enumerate(confidence_list):
            name = tk.Entry(self.confidence_frame, text=str(key), fg="black",
                            font=("Arial", 16, 'bold'))
            value = tk.Entry(self.confidence_frame, text=str(value), fg="black",
                             font=("Arial", 16, 'bold'))
            name.grid(row=idx, column=0)
            value.grid(row=idx, column=1)

    def on_load(self):
        threading.Thread(target=self.client.load_file, daemon=True).start()

    def on_detect(self):
        threading.Thread(target=self.client.request_detect, daemon=True).start()

    def on_describe(self):
        threading.Thread(target=self.client.request_describe, daemon=True).start()
