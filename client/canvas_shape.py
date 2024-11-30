from tkinter import Canvas, Tk
from PIL import Image, ImageTk, ImageDraw

BORDER_COLOR = "red"
SELECTED_COLOR = "yellow"
BORDER_WIDTH = 3

class ResizableCanvasShape:
    def __init__(self,controller, canvas, bbox):
        """
      retialize the ResizeableCanvasShape.

        :param canvas: Tkinter canvas where the rectangle will be drawn
        :param x1, y1, x2, y2: Coordinates of the rectangle
        :param border_color: Color of the rectangle border
        :param border_width: Width of the rectangle border
        """
        self.controller = controller
        self.canvas = canvas
        self.x1, self.y1, self.x2, self.y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        self.image = self._create_transparent_image()

        self.image_tk = ImageTk.PhotoImage(self.image)
        self.rectangle = self.canvas.create_image(
            self.x1, self.y1, 
            anchor="nw", 
            image=self.image_tk, 
            tags="square"
        )
        self.border = self.canvas.create_rectangle(
            self.x1, self.y1, self.x2, self.y2, 
            outline=BORDER_COLOR, 
            width=BORDER_WIDTH, 
            tags="square"
        )
        self.is_resizing = False
        self.selected = False
        self.start_x = None
        self.start_y = None
        self.resize_side = None

        self._bind_events()


    def _create_transparent_image(self):
        """
        Create a transparent PIL image to fill the rectangle.

        :return: A PIL Image with transparency
        """
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

    def _bind_events(self):
        """
        Bind events to handle movement and resizing.
        """
        self.canvas.tag_bind(self.rectangle, "<Button-1>", self.on_select)
        #self.canvas.tag_bind(self.border, "<Button-1>", self.on_resize_start)
        self.canvas.tag_bind(self.rectangle, "<B1-Motion>", self.on_move)
        #self.canvas.tag_bind(self.border, "<B1-Motion>", self.on_resize)
        self.canvas.tag_bind(self.rectangle, "<ButtonRelease-1>", self.on_action_end)
        #self.canvas.tag_bind(self.border, "<ButtonRelease-1>", self.on_action_end)

    def select(self):
        self.selected = True
        self.canvas.itemconfig(self.border, outline = SELECTED_COLOR)
    
    def deselect(self):
        self.selected = False
        self.canvas.itemconfig(self.border, outline = BORDER_COLOR)

    def on_select(self, event):
        """
        Start moving the rectangle.
        """
        self.start_x = event.x
        self.start_y = event.y
        self.controller.select(self)

    def on_resize_start(self, event):
        """
        Start resizing the rectangle.
        """
        self.is_resizing = True
        self.start_x = event.x
        self.start_y = event.y
        x1, y1, x2, y2 = self.canvas.coords(self.border)
        self.resize_side = self._get_resize_side(event.x, event.y, x1, y1, x2, y2)

    def _get_resize_side(self, x, y, x1, y1, x2, y2):
        """
        Determine which side or corner is being resized.
        """
        if abs(x - x1) < 5:
            return "left"
        if abs(x - x2) < 5:
            return "right"
        if abs(y - y1) < 5:
            return "top"
        if abs(y - y2) < 5:
            return "bottom"
        return None

    def on_move(self, event):
        """
        Handle moving the rectangle.
        """
        if not self.selected:
            return

        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.start_x = event.x
        self.start_y = event.y
        self.move(dx, dy)

    def on_resize(self, event):
        """
        Handle resizing the rectangle.
        """
        if not self.is_resizing:
            return

        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.start_x = event.x
        self.start_y = event.y
        self._resize(dx, dy)

    def _resize(self, dx, dy):
        """
        Resize the rectangle based on the side being dragged.
        """
        if self.resize_side == "left":
            self.x1 += dx
        elif self.resize_side == "right":
            self.x2 += dx
        elif self.resize_side == "top":
            self.y1 += dy
        elif self.resize_side == "bottom":
            self.y2 += dy

        # Redraw the rectangle and border
        self._redraw()

    def move(self, dx, dy):
        """
        Move the rectangle by a given delta.
        """
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy
        self.canvas.move(self.rectangle, dx, dy)
        self.canvas.move(self.border, dx, dy)

    def _redraw(self):
        """
        Redraw the rectangle and its border.
        """
        self.canvas.delete(self.rectangle)
        self.canvas.delete(self.border)
        self.image = self._create_transparent_image()
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.rectangle = self.canvas.create_image(
            self.x1, self.y1, 
            anchor="nw", 
            image=self.image_tk
        )
        self.border = self.canvas.create_rectangle(
            self.x1, self.y1, self.x2, self.y2, 
            outline=BORDER_COLOR, 
            width=BORDER_WIDTH
        )
        self._bind_events()

    def on_action_end(self, event):
        """
        Reset state after mouse release.
        """
        self.is_resizing = False
        self.resize_side = None
    
    def clear(self):
        self.canvas.delete(self.border)
        self.canvas.delete(self.rectangle)
