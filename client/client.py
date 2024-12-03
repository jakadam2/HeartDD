from enum import Enum
from PIL import ImageTk
import queue
import file_handler as fh
import server_communicator as sch
import window_controller as wc
from error_window import ErrorWindow
from config import *
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

TEST_FILE = os.path.join(os.path.abspath(root_dir),
                            parser.get('client.test', 'test_file'))
TESTING = parser.getboolean('client.test', 'testing')
CANVAS_SIZE = parser.getint("DEFAULT", 'image_size')

class Flag(Enum):
    LOAD = 1
    DETECT = 2
    DESCRIBE = 3
    EXIT = 4
    NOTHING = 5

class Client:
    def __init__(self, window: tk.Tk):
        self.image = None
        self.image_tk = None
        self.bitmask = None
        self.bounding_boxes = None
        self.scaled_bboxes = None
        self.confidence_list = None
        self.filename = None
        self.extension = None
        # Initialize file handler and server communicator
        sch.connect()

        self.queue = None
        self.queue = queue.Queue()
        self.scaling_ratio = 1
        self.reverse_scaling_ratio = 1
        # Initialize WindowController for window-related tasks
        self.window_controller = wc.WindowController(self, window)
        if TESTING:
            # Load file and start polling for UI updates
            self.load_file(TEST_FILE)

        self.poll_queue()

    def poll_queue(self):
        try:
            while True:
                flag = self.queue.get_nowait()
                match flag:
                    case Flag.DETECT:
                        self.window_controller.display_bboxes(self.scalebboxes())
                    case Flag.DESCRIBE:
                        self.put_confidence()
                    case Flag.LOAD:
                        self.window_controller.display_image(self.image_tk)
                    case _:
                        pass
        except queue.Empty:
            pass
        except Exception as ex:
            ErrorWindow.show(type(ex).__name__ ,ex.args)
        # Continue polling every 100 ms
        self.window_controller.window.after(100, self.poll_queue)

    def scalebboxes(self):
        img_width, img_height = self.image.size
        self.scaling_ratio = CANVAS_SIZE / img_width
        self.reverse_scaling_ratio = 1 / self.scaling_ratio
        self.scaled_bboxes = []
        for bbox in self.bounding_boxes:
            self.scaled_bboxes.append([
                bbox[0] * self.scaling_ratio,
                bbox[1] * self.scaling_ratio,
                bbox[2] * self.scaling_ratio,
                bbox[3] * self.scaling_ratio
            ])
        return self.scaled_bboxes

    def load_file(self, name: str = None):
        try:
            self.filename, self.extension = fh.get_file_name(name)
            self.image = fh.load_file(self.filename, self.extension)
            self.image_tk = ImageTk.PhotoImage(self.image.resize((CANVAS_SIZE, CANVAS_SIZE)))
            self.bitmask = fh.load_bitmask(self.filename)
        except ValueError:
            ErrorWindow.show(message="An error occured while loading a file")
        except FileNotFoundError:
            ErrorWindow.show("File not found", "File not found")
        except TypeError as ex:
            ErrorWindow.show("Incorrect file format", ex.args)
        self.queue.put(Flag.LOAD)

    def request_detect(self):
        self.confidence_list = None
        if self.image is None:
            ErrorWindow.show("No image", "Please load an image you want to use")
            return
        self.bounding_boxes = sch.request_detection(self.image, self.bitmask)
        self.queue.put(Flag.DETECT)

    def request_describe(self):
        if self.image is None:
            ErrorWindow.show("No image", "Please load an image you want to use")
            return
        elif self.bounding_boxes is None:
            ErrorWindow.show("No boxes", "Please use \"Detect Lesions\" option before")
            return
        self.confidence_list = sch.request_description(self.image, self.bitmask, self.bounding_boxes)

        self.queue.put(Flag.DESCRIBE)
    
    def put_confidence(self) -> None:
        self.window_controller.display_confidence(self.confidence_list)

    def delete_bbox(self, idx: int):
        self.scaled_bboxes.pop(idx)
        self.bounding_boxes.pop(idx)
        if self.confidence_list:
            self.confidence_list.pop(idx)

    def change_bbox(self, x1: int, y1: int, x2: int, y2: int, index: int):
        #print(f"Scaling ratio {self.scaling_ratio}, Reverse scaling {reverse_scaling_ratio}")
        #print(f"Before scaling:{self.bounding_boxes[index]}")
        self.bounding_boxes[index][0] = x1 * self.reverse_scaling_ratio
        self.bounding_boxes[index][1] = y1 * self.reverse_scaling_ratio
        self.bounding_boxes[index][2] = x2 * self.reverse_scaling_ratio
        self.bounding_boxes[index][3] = y2 * self.reverse_scaling_ratio



