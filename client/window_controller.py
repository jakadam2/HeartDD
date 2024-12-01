from canvas_shape import ResizableCanvasShape
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
        self.selected_idx = 0;
        self.boxes = []
        self.conf = [] 


        self.window.title("Lesion Detection")
        self.window.config(bg="skyblue")

        self.right_frame = tk.Frame(self.window, width=WIDTH, height=HEIGHT, bg='white')
        self.right_frame.grid(row=0, column=1, padx=15, pady=15)
        self.canvas = tk.Canvas(self.right_frame, width=WIDTH, height=HEIGHT)
        self.canvas.pack()
        

        self.left_frame = tk.Frame(self.window, width=400, height=HEIGHT)
        self.left_frame.grid(row=0, column=0, padx=5, pady=5)

        self.confidence_frame = tk.Frame(self.left_frame, width=400, height=450)
        self.confidence_frame.grid(row=0, column=0, padx=5, pady=5)

        self.button_frame = tk.Frame(self.left_frame, width=400, height=250, bg='skyblue')
        self.button_frame.grid(row=1, column=0, padx=5, pady=5)

        # Initialize buttons and connect to client methods
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.on_load)
        self.load_button.grid(row=1, column=0, padx=5, pady=5)

        self.detect_button = tk.Button(self.button_frame, text="Detect Lesions", command=self.on_detect)
        self.detect_button.grid(row = 1, column = 1, padx=5, pady=5)

        self.describe_button = tk.Button(self.button_frame, text="Describe Lesions", command=self.on_describe)
        self.describe_button.grid(row = 1, column = 2, padx=5, pady=5)

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

        for idx, entry in enumerate(confidence[self.selected_idx].entries):
            name = tk.Entry(self.confidence_frame, font=("Arial", 12, "bold"))
            value = tk.Entry(self.confidence_frame, font=("Arial",12, "bold"))
            self.conf.append((name, value))
            name.grid(row=idx, column=0)
            value.grid(row=idx, column=1)
            name.insert(tk.END, str(entry.name))
            value.insert(tk.END, str(round(entry.confidence*100, 2)))

    def on_load(self):
        threading.Thread(target=self.client.load_file, daemon=True).start()

    def on_detect(self):
        for name, value in self.conf:
            name.grid_remove()
            value.grid_remove()
        threading.Thread(target=self.client.request_detect, daemon=True).start()

    def on_describe(self):
        threading.Thread(target=self.client.request_describe, daemon=True).start()
    
    def select(self, rectangle: ResizableCanvasShape):
        self.selected_idx = self.boxes.index(rectangle)
        self.client.display_confidence()

    def move_shape(self, x1: int, y1: int, x2: int, y2: int):
        self.client.move_bbox(x1, y1, x2, y2, self.selected_idx)