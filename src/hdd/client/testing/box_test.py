import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import random

# Global variables for keeping track of the bounding box being manipulated
selected_box = None
resizing = False
move_mode = False
resize_margin = 10  # Margin size for detecting clicks near the edge
bounding_boxes = []  # List to store multiple bounding boxes


class BoundingBoxApp:
    def __init__(self, root, image_path):
        self.root = root
        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Load and display the image
        self.image = Image.open(image_path)
        self.image = self.image.resize((800, 600))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Generate random bounding boxes (3 boxes in this case)
        self.num_boxes = 3
        self.generate_bounding_boxes()

        # Bind mouse events to the canvas
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def generate_bounding_boxes(self):
        img_width, img_height = self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()

        for _ in range(self.num_boxes):
            x1 = random.randint(50, img_width // 2)
            y1 = random.randint(50, img_height // 2)
            x2 = random.randint(img_width // 2, img_width - 50)
            y2 = random.randint(img_height // 2, img_height - 50)
            box = {'rect': self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2),
                   'ix': x1, 'iy': y1, 'ex': x2, 'ey': y2}
            bounding_boxes.append(box)

    def on_mouse_down(self, event):
        global selected_box, resizing, move_mode

        for box in bounding_boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box['ix'], box['iy'], box['ex'], box['ey']

            # Check if the user clicks near the edge for resizing
            if (abs(x1 - event.x) <= resize_margin or abs(x2 - event.x) <= resize_margin) and \
               (abs(y1 - event.y) <= resize_margin or abs(y2 - event.y) <= resize_margin):
                selected_box = box
                resizing = True
                move_mode = False
                break
            # Check if the click is inside the box for moving
            elif x1 + resize_margin < event.x < x2 - resize_margin and \
                    y1 + resize_margin < event.y < y2 - resize_margin:
                selected_box = box
                resizing = False
                move_mode = True
                break

    def on_mouse_move(self, event):
        global selected_box

        if selected_box:
            if resizing:
                # Resize the box by adjusting the bottom-right corner
                selected_box['ex'] = event.x
                selected_box['ey'] = event.y
                self.update_rectangle(selected_box)
            elif move_mode:
                # Move the box by adjusting both corners
                dx = event.x - (selected_box['ix'] + (selected_box['ex'] - selected_box['ix']) // 2)
                dy = event.y - (selected_box['iy'] + (selected_box['ey'] - selected_box['iy']) // 2)
                selected_box['ix'] += dx
                selected_box['iy'] += dy
                selected_box['ex'] += dx
                selected_box['ey'] += dy
                self.update_rectangle(selected_box)

    def on_mouse_up(self, event):
        global selected_box, resizing, move_mode
        selected_box = None
        resizing = False
        move_mode = False

    def update_rectangle(self, box):
        # Update the rectangle on the canvas with new coordinates
        self.canvas.coords(box['rect'], box['ix'], box['iy'], box['ex'], box['ey'])


if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxApp(root, "image.jpg")  # Replace with the path to your image
    root.mainloop()
