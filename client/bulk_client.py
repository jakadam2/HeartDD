import file_handler as fh
import server_communicator as sc
from config import *

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

TEST_DIR = os.path.join(os.path.abspath(root_dir),
                         parser.get('client.test', 'bulk_test_dir'))
TESTING = parser.getboolean('client.test', 'testing')
CANVAS_SIZE = parser.getint("DEFAULT", 'image_size')

class BulkClient:
    def __init__(self, directory: str | None = None):
        self.image = None
        self.bitmask = None
        self.bounding_boxes = None
        self.scaled_bboxes = None
        self.confidence_list = None
        self.directory = directory
        self.files = None
        self.filename = None
        self.extension = None

        sc.connect()

        if TESTING:
            self.directory = TEST_DIR

    def load_file(self, name: str | None = None):
        self.filename, self.extension = fh.get_file_name(name)
        self.image = fh.load_file(self.filename, self.extension)
        self.bitmask = fh.load_bitmask(self.filename)

    def request_detect(self):
        if self.image is None:
            print("Something went wrong no image is loaded")
            raise Exception("No image is loaded")
        self.bounding_boxes = sc.request_detection(self.image, self.bitmask)

    def request_describe(self):
        if self.bounding_boxes is None:
            raise Exception("No bounding boxes detected")
        self.confidence_list = sc.request_description(self.image, self.bitmask, self.bounding_boxes)


    def load_dir(self):
        try:
            self.files, self.directory = fh.load_directory(self.directory)
        except FileNotFoundError:
            print("Directory not found.")
        except PermissionError:
            print("Permission denied to access the directory.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def bulk_operations(self):
        while True:
            self.load_dir()
            if self.files is not None:
                break
            next_step = input("Do you wish to choose a different directory [Y/N]?")
            if next_step.lower() == "n" or next_step.lower() == "no":
                return

        next_step = input(f"{self.directory} loaded with {len(self.files)} images. Proceed with bulk analysis? [Y/N]")
        if next_step.lower() == "n" or next_step.lower() == "no":
            return

        for idx, filename in enumerate(self.files):
            try:
                print(f"[{idx + 1}/{len(self.files)}], file: {filename}")
                self.load_file(filename)
                self.request_detect()
                self.request_describe()
                fh.save_image(self.filename,
                              self.directory,
                              self.image,
                              self.bounding_boxes,
                              self.confidence_list)

            except Exception as ex:
                print(f"Something went wrong with {os.path.basename(self.filename)}, {ex.args}")







