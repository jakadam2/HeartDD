import os
import configparser
from tkinter import Canvas, Tk
from PIL import Image, ImageTk, ImageDraw
from config import parser


IMAGE_BOUNDARY = parser.getint("DEFAULT", "image_size")

BORDER_COLOR = parser.get("boxes", "border_color")
SELECTED_COLOR = parser.get("boxes", "selected_color")
BORDER_WIDTH = parser.getint("boxes", "border_width")
CORNER_SIZE = parser.getint("boxes", "corner_size")

class ResizableCanvasShape:
    def __init__(self, controller, canvas, bbox):
        self.controller = controller
        self.canvas = canvas
        self.x1, self.y1, self.x2, self.y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        self.image = self._create_transparent_image()

        self.rectangle = self.canvas.create_image(
            self.x1, self.y1,
            anchor="nw",
            image=ImageTk.PhotoImage(self.image),
        )
        self.border = self.canvas.create_rectangle(
            self.x1, self.y1, self.x2, self.y2,
            outline=BORDER_COLOR,
            width=BORDER_WIDTH,
        )
        self.corners = self._create_corners()  # List to store corner ovals
        self.is_resizing = False
        self.selected = False
        self.start_x = None
        self.start_y = None
        self.resize_corner = None

        self._bind_events()

    def _create_transparent_image(self):
        width = int(abs(self.x2 - self.x1))
        height = int(abs(self.y2 - self.y1))
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Fully transparent
        draw = ImageDraw.Draw(image)
        # Fill the rectangle with a slightly transparent color to allow interaction
        draw.rectangle(
            [(0, 0), (width - 1, height - 1)],
            fill=(255, 255, 255, 5),  # Minimal opacity
            outline=None,
        )
        return image

    def _rescale_image(self):
        """Rescale the image to fit the new bounding box."""
        new_width = int(self.x2 - self.x1)
        new_height = int(self.y2 - self.y1)
        resized_image = self.image.resize((new_width, new_height))
        self.canvas.itemconfig(self.rectangle, image=ImageTk.PhotoImage(resized_image))
        self.canvas.coords(self.rectangle, self.x1, self.y1)

    def _create_corners(self):
        # Create four draggable corner ovals
        return [
            self.canvas.create_oval(
                self.x1 - CORNER_SIZE, self.y1 - CORNER_SIZE,
                self.x1 + CORNER_SIZE, self.y1 + CORNER_SIZE,
                fill=BORDER_COLOR, outline=""
            ),
            self.canvas.create_oval(
                self.x2 - CORNER_SIZE, self.y1 - CORNER_SIZE,
                self.x2 + CORNER_SIZE, self.y1 + CORNER_SIZE,
                fill=BORDER_COLOR, outline=""
            ),
            self.canvas.create_oval(
                self.x1 - CORNER_SIZE, self.y2 - CORNER_SIZE,
                self.x1 + CORNER_SIZE, self.y2 + CORNER_SIZE,
                fill=BORDER_COLOR, outline=""
            ),
            self.canvas.create_oval(
                self.x2 - CORNER_SIZE, self.y2 - CORNER_SIZE,
                self.x2 + CORNER_SIZE, self.y2 + CORNER_SIZE,
                fill=BORDER_COLOR, outline=""
            ),
        ]

    def _bind_events(self):
        self.canvas.tag_bind(self.rectangle, "<Button-1>", self.on_select)
        self.canvas.tag_bind(self.rectangle, "<B1-Motion>", self.on_move)
        for corner in self.corners:
            self.canvas.tag_bind(corner, "<Button-1>", self.on_corner_select)
            self.canvas.tag_bind(corner, "<B1-Motion>", self.on_corner_resize)

    def select(self):
        self.selected = True
        self.canvas.itemconfig(self.border, outline=SELECTED_COLOR)

    def deselect(self):
        self.selected = False
        self.canvas.itemconfig(self.border, outline=BORDER_COLOR)

    def on_select(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.controller.select(self)

    def on_move(self, event):
        if not self.selected:
            return

        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.start_x = event.x
        self.start_y = event.y
        self.move(dx, dy)

    def move(self, dx, dy):
        new_x1 = self.x1 + dx
        new_y1 = self.y1 + dy
        new_x2 = self.x2 + dx
        new_y2 = self.y2 + dy

        # Prevent the rectangle from going out of boundaries
        if new_x1 < 0 or new_y1 < 0 or new_x2 > IMAGE_BOUNDARY or new_y2 > IMAGE_BOUNDARY:
            return

        self.x1, self.y1, self.x2, self.y2 = new_x1, new_y1, new_x2, new_y2
        self.canvas.move(self.rectangle, dx, dy)
        self.canvas.move(self.border, dx, dy)
        for corner in self.corners:
            self.canvas.move(corner, dx, dy)

        self.controller.change_shape(self.x1, self.y1, self.x2, self.y2)

    def on_corner_select(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.resize_corner = self.canvas.find_closest(event.x, event.y)[0]

    def on_corner_resize(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.start_x = event.x
        self.start_y = event.y

        if self.resize_corner == self.corners[0]:  # Top-left corner
            new_x1 = self.x1 + dx
            new_y1 = self.y1 + dy
            if new_x1 < 0 or new_y1 < 0 or new_x1 >= self.x2:  # Prevent shrinking too far or exceeding boundaries
                return
            self.x1, self.y1 = new_x1, new_y1
        elif self.resize_corner == self.corners[1]:  # Top-right corner
            new_x2 = self.x2 + dx
            new_y1 = self.y1 + dy
            if new_x2 > IMAGE_BOUNDARY or new_y1 < 0 or new_x2 <= self.x1:
                return
            self.x2, self.y1 = new_x2, new_y1
        elif self.resize_corner == self.corners[2]:  # Bottom-left corner
            new_x1 = self.x1 + dx
            new_y2 = self.y2 + dy
            if new_x1 < 0 or new_y2 > IMAGE_BOUNDARY or new_x1 >= self.x2:
                return
            self.x1, self.y2 = new_x1, new_y2
        elif self.resize_corner == self.corners[3]:  # Bottom-right corner
            new_x2 = self.x2 + dx
            new_y2 = self.y2 + dy
            if new_x2 > IMAGE_BOUNDARY or new_y2 > IMAGE_BOUNDARY or new_x2 <= self.x1:
                return
            self.x2, self.y2 = new_x2, new_y2

        self.controller.change_shape(self.x1, self.y1, self.x2, self.y2)
        self.redraw()

    def redraw(self):
        # Update the rectangle, border, and corners
        self._rescale_image()  # Rescale and redraw the image
        self.canvas.coords(self.border, self.x1, self.y1, self.x2, self.y2)

        # Update the positions of the corner ovals
        corner_coords = [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x1, self.y2),
            (self.x2, self.y2),
        ]
        for i, corner in enumerate(self.corners):
            x, y = corner_coords[i]
            self.canvas.coords(
                corner,
                x - CORNER_SIZE, y - CORNER_SIZE,
                x + CORNER_SIZE, y + CORNER_SIZE
            )

    def clear(self):
        self.canvas.delete(self.border)
        self.canvas.delete(self.rectangle)
        for corner in self.corners:
            self.canvas.delete(corner)
