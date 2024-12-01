import tkinter as tk

class ResizableCanvasShape:
    def __init__(self, canvas, bbox):
        self.canvas = canvas
        self.rect = self.canvas.create_rectangle(bbox[0],bbox[1], bbox[2], bbox[3], outline="red", width=2)
        self.selected = False
        self.offset_x = 0
        self.offset_y = 0

        # Bind mouse events to the rectangle
        self.canvas.tag_bind(self.rect, "<Button-1>", self.on_select)
        self.canvas.tag_bind(self.rect, "<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_deselect)

        # Bind the delete key to remove the rectangle if selected
        self.canvas.bind_all("<Delete>", self.delete_shape)

        # Resize handles
        self.handles = [self.create_resize_handle(x, y) for x, y in self.get_handle_positions()]
        for handle in self.handles:
            self.canvas.tag_bind(handle, "<Button-1>", self.on_select_handle)
            self.canvas.tag_bind(handle, "<B1-Motion>", self.on_resize)

        self.active_handle = None

    def create_resize_handle(self, x, y):
        return self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="black", width=1)

    def get_handle_positions(self):
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def on_select(self, event):
        # Deselect any previously selected shape
        if hasattr(self.canvas, "selected_shape") and self.canvas.selected_shape:
            self.canvas.selected_shape.deselect()

        # Select this shape
        self.canvas.selected_shape = self
        self.selected = True
        self.canvas.itemconfig(self.rect, outline="yellow")

        # Update offset based on the current cursor position
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        self.offset_x = event.x - x1
        self.offset_y = event.y - y1

    def deselect(self):
        """ Deselects the shape, resetting outline color and active handle. """
        self.selected = False
        self.canvas.itemconfig(self.rect, outline="black")
        self.active_handle = None  # Clear active handle

    def on_deselect(self, event):
        if self.selected:
            self.deselect()

    def on_drag(self, event):
        if self.selected and not self.active_handle:
            x, y = event.x - self.offset_x, event.y - self.offset_y
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            self.canvas.coords(self.rect, x, y, x + (x2 - x1), y + (y2 - y1))
            self.update_handles()

    def on_select_handle(self, event):
        if self.selected:
            self.active_handle = self.canvas.find_withtag("current")[0]

    def on_resize(self, event):
        if self.active_handle and self.selected:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            handle_index = self.handles.index(self.active_handle)
            if handle_index == 0:  # Top-left
                x1, y1 = event.x, event.y
            elif handle_index == 1:  # Top-right
                x2, y1 = event.x, event.y
            elif handle_index == 2:  # Bottom-right
                x2, y2 = event.x, event.y
            elif handle_index == 3:  # Bottom-left
                x1, y2 = event.x, event.y
            self.canvas.coords(self.rect, x1, y1, x2, y2)
            self.update_handles()

    def update_handles(self):
        positions = self.get_handle_positions()
        for handle, (x, y) in zip(self.handles, positions):
            self.canvas.coords(handle, x-5, y-5, x+5, y+5)

    def delete_shape(self, event):
        if self.selected:
            self.canvas.delete(self.rect)
            for handle in self.handles:
                self.canvas.delete(handle)
            self.selected = False
            if hasattr(self.canvas, "selected_shape"):
                self.canvas.selected_shape = None

if __name__ == "__main__":
    root = tk.Tk()
    canvas = tk.Canvas(root, width=400, height=400, bg="white")
    canvas.pack()

    # Create multiple resizable shapes on canvas
    shape1 = ResizableCanvasShape(canvas)
    shape2 = ResizableCanvasShape(canvas)
    shape3 = ResizableCanvasShape(canvas)

    # Adjust initial positions for visibility
    canvas.move(shape2.rect, 100, 100)
    canvas.move(shape3.rect, 200, 200)

    root.mainloop()
