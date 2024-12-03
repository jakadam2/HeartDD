from canvas_shape import ResizableCanvasShape
import threading
from PIL import ImageTk
from config import parser

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

CANVAS_SIZE = parser.getint("DEFAULT", "image_size")

BG_COLOR = parser.get("window", "bg_color")
SECONDARY_COLOR = parser.get("window", "sec_color")
TITLE = parser.get("window", "title")
LOAD_BUTTON = parser.get("window", "load_button_text")
DETECT_BUTTON = parser.get("window", "detect_button_text")
DESCRIBE_BUTTON = parser.get("window", "describe_button_text")
FONT = parser.get("window", "font")
FONT_SIZE = parser.getint("window", "font_size")
STYLING = parser.get("window", "font_styling")
PRECISION = parser.getint("window", "confidence_precision")
TABLE_WIDTH = 25


class WindowController:
    def __init__(self, client, window):
        self.window = window
        self.client = client  # Reference to Client class to call application methods
        self.selected_idx = 0
        self.boxes = []
        self.conf = []

        self.window.title("Lesion Detection")
        self.window.config(bg=BG_COLOR)

        self.right_frame = tk.Frame(self.window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=SECONDARY_COLOR)
        self.right_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        self.canvas = tk.Canvas(self.right_frame, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas.pack()

        self.left_frame = tk.Frame(self.window, width=400, height=CANVAS_SIZE)
        self.left_frame.pack(side=tk.LEFT, padx=5, pady=5)

        self.confidence_frame = tk.Frame(self.left_frame, width=400, height=450)
        self.confidence_frame.grid(row=0, column=0, padx=5, pady=5)

        self.button_frame = tk.Frame(self.left_frame, width=400, height=250, bg=BG_COLOR)
        self.button_frame.grid(row=1, column=0)

        # Initialize buttons and connect to client methods
        self.load_button = tk.Button(self.button_frame, text=LOAD_BUTTON, command=self.on_load)
        self.load_button.grid(row=1, column=0, padx=5, pady=5)

        self.detect_button = tk.Button(self.button_frame, text=DETECT_BUTTON, command=self.on_detect)
        self.detect_button.grid(row=1, column=1, padx=5, pady=5)

        self.describe_button = tk.Button(self.button_frame, text=DESCRIBE_BUTTON, command=self.on_describe)
        self.describe_button.grid(row=1, column=2, padx=5, pady=5)

        self.window.bind("<Delete>", self.on_delete)

    def display_image(self, image_tk: ImageTk) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.window.update()

    def display_bboxes(self, bounding_boxes: list[list[float]]) -> None:
        for shape in self.boxes:
            shape.clear()
        self.boxes.clear()
        for bbox in bounding_boxes:
            self.boxes.append(ResizableCanvasShape(self, self.canvas, bbox))

    def display_confidence(self, confidence: list[dict[str:float]]) -> None:
        if len(self.boxes) == 0:
            return

        for box in self.boxes:
            box.deselect()

        self.boxes[self.selected_idx].select()

        if not confidence:
            return
        self.conf.clear()
        print(self.selected_idx)
        idx = 0
        for key, confidence in confidence[self.selected_idx].items():
            name = tk.Entry(self.confidence_frame, width=TABLE_WIDTH, font=(FONT, FONT_SIZE, STYLING))
            value = tk.Entry(self.confidence_frame, width=10, font=(FONT, FONT_SIZE, STYLING))
            self.conf.append((name, value))
            name.grid(row=idx, column=0)
            value.grid(row=idx, column=1)
            name.insert(tk.END, key)
            value.insert(tk.END, str(round(confidence * 100, PRECISION)))
            idx += 1

    def on_load(self):
        threading.Thread(target=self.client.load_file, daemon=True).start()

    def on_detect(self):
        self.selected_idx = 0
        for name, value in self.conf:
            name.grid_remove()
            value.grid_remove()
        self.conf.clear()
        threading.Thread(target=self.client.request_detect, daemon=True).start()

    def on_describe(self):
        threading.Thread(target=self.client.request_describe, daemon=True).start()

    def on_delete(self, event):
        self.boxes[self.selected_idx].clear()
        self.boxes.pop(self.selected_idx)
        for name, value in self.conf:
            name.grid_remove()
            value.grid_remove()
        self.conf.clear()
        self.client.delete_bbox(self.selected_idx)
        self.selected_idx = 0

    def select(self, rectangle: ResizableCanvasShape):
        self.selected_idx = self.boxes.index(rectangle)
        self.client.put_confidence()

    def change_shape(self, x1: int, y1: int, x2: int, y2: int):
        self.client.change_bbox(x1, y1, x2, y2, self.selected_idx)
