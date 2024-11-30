import os
import sys
from enum import Enum
from PIL import ImageTk
import queue
import file_handler as fh
import server_communicator as sch
import window_controller as wc

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

root_dir = os.path.join(os.path.dirname(__file__), "..")
TEST_FILE = os.path.join(os.path.abspath(root_dir), 'base_images', '12aw4ack71831bocuf5j3pz235kn1v361de_33.png')
WIDTH, HEIGHT = 700, 700

class Flag(Enum):
    LOAD = 1
    DETECT = 2
    DESCRIBE = 3
    EXIT = 4
    NOTHING = 5

class Client:
    def __init__(self, ip: str = "localhost", port: str = "50051"):
        self.image = None
        self.image_tk = None
        self.bitmask = None
        self.bounding_boxes = None
        self.scaled_bboxes = None
        self.confidence_list = None
        self.queue = queue.Queue()


        root = tk.Tk()
        # Initialize WindowController for window-related tasks
        self.window_controller = wc.WindowController(self, root)

        # Initialize file handler and server communicator
        self.files = fh.FileHandler()
        self.server = sch.ServerHandler(ip, port)

        # Load file and start polling for UI updates
        self.load_file(TEST_FILE)
        self.poll_queue()
        root.mainloop()

    def poll_queue(self):
        """Poll queue for updates and perform actions based on flags."""
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        self.window_controller.display_bboxes(self.scalebboxes())
                    case Flag.DESCRIBE:
                        self.display_confidence()
                    case Flag.LOAD:
                        self.window_controller.display_image(self.image_tk)
                    case _:
                        pass
        except queue.Empty:
            pass
        except Exception as ex:
            print(f"[CLIENT] An exception of type {type(ex).__name__} occurred. Arguments:\n{ex.args}")

        # Continue polling every 100 ms
        self.window_controller.window.after(100, self.poll_queue)

    def scalebboxes(self):
        img_width, img_height = self.image.size
        scaling_ratio = WIDTH / img_width
        self.scaled_bboxes = []
        for bbox in self.bounding_boxes:
            self.scaled_bboxes.append([
                bbox[0] * scaling_ratio,
                bbox[1] * scaling_ratio,
                bbox[2] * scaling_ratio,
                bbox[3] * scaling_ratio
            ])
        return self.scaled_bboxes

    def load_file(self, name: str = None):
        try:
            self.files.get_file_name(name)
            self.image = self.files.load_file()
            self.image_tk = ImageTk.PhotoImage(self.image.resize((WIDTH, HEIGHT)))
            self.bitmask = self.files.load_bitmask()
        except ValueError as ex:
            print(f"[CLIENT] An error occurred while loading a file {ex.args}")
        except FileNotFoundError:
            print(f"Couldn't find file {self.files}")
        self.queue.put(Flag.LOAD)

    def request_detect(self):
        if self.image is None:
            print("[CLIENT] No image loaded")
            return
        self.bounding_boxes = self.server.request_detection(self.image, self.bitmask)
        self.queue.put(Flag.DETECT)

    def request_describe(self):
        if self.bounding_boxes is None:
            print("[CLIENT] No bounding boxes present")
            return
        self.confidence_list = self.server.request_description(self.image, self.bitmask, self.bounding_boxes)
        for confidence in self.confidence_list:
            print(f"{confidence}")
        self.queue.put(Flag.DESCRIBE)
    
    def display_confidence(self) -> None:
        self.window_controller.display_confidence(self.confidence_list)



def start():
    if len(sys.argv) <= 2:
        Client()
    else:
        ip = sys.argv[1]
        port = sys.argv[2]
        Client(ip, port)


if __name__ == "__main__":
    start()
