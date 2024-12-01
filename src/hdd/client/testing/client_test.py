import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import queue

WIDTH, HEIGHT = 800, 800

class Client:
    def __init__(self, window):
        self.window = window
        self.window.title("Async Image Loader")

        # Initialize Canvas for Image
        self.canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
        self.canvas.pack()

        # Button to trigger the load action
        self.load_button = tk.Button(window, text="Load Image", command=self.load_action)
        self.load_button.pack()

        # Initialize queue for threading
        self.queue = queue.Queue()

        # Placeholder for the image
        self.image_tk = None

        # Start polling the queue
        self.poll_queue()

    def load_action(self):
        """Open a file dialog to select an image and load it in a separate thread."""
        # Open a file dialog to choose the image file
        # Start a background thread to load the image
        threading.Thread(target=self.load_image, daemon=True).start()

    def load_image(self):
        """Load the image in a separate thread and put it in the queue when done."""
        try:
            file_path = filedialog.askopenfilename()
            # Open and resize the image
            image = Image.open(file_path).resize((WIDTH, HEIGHT))
            # Convert to ImageTk.PhotoImage for Tkinter display
            self.image_tk = ImageTk.PhotoImage(image)
            # Put a success flag in the queue to signal the main thread to display the image
            self.queue.put("DISPLAY_IMAGE")
        except Exception as e:
            print(f"Error loading image: {e}")

    def poll_queue(self):
        """Poll the queue for display instructions and update the GUI accordingly."""
        try:
            # Check if there's an item in the queue
            while True:
                flag = self.queue.get_nowait()
                if flag == "DISPLAY_IMAGE":
                    # Display the loaded image on the canvas
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        except queue.Empty:
            # No items in the queue; do nothing
            pass
        # Poll the queue again after 100 ms
        self.window.after(100, self.poll_queue)

if __name__ == "__main__":
    root = tk.Tk()
    client = Client(root)
    root.mainloop()
