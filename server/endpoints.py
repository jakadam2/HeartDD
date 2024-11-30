# THIS IS THE FILE USED TO IMPLEMENT ENDPOINTS TO INTERACT WITH MODELS
# REPLACE THESE FUNCTIONS WITH WHATEVER YOU NEED
from description.lesion_desriber import LesionDescriber
from detection.detect import LesionDetector

from PIL import Image
import numpy.typing as npt
from typing import Union
import random

from torchvision import transforms


def detect_bounding_boxes(image: Image, mask: npt.ArrayLike) -> list[dict[str:float]]:
    detector = LesionDetector(image, mask)
    coordinates = detector.detect()

    return coordinates
    # This function takes an image in the PIL.Image format
    # It returns positions of the bounding boxes in format
    # {x1:x1, y1:y1, x2:x2, y2:y2}


def describe_bbox(image: Image, mask: npt.ArrayLike, bboxes: list[Union[int, int]]) -> list[dict[str:float]]:
    describer = LesionDescriber()
    results = []
    pll = transforms.ToTensor()
    image_tensor = pll(image)
    mask_tensor = pll(mask)

    for bbox in bboxes:
        coords = ((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2)
        results.append(describer(image_tensor, mask_tensor, coords))

    return results

def test_bboxes():
    image_width = 512
    image_height = 512

    # Generate two random bounding boxes within the 512x512 image dimensions
    def random_box():
        x1 = round(random.uniform(0, image_width - 1), 2)
        y1 = round(random.uniform(0, image_height - 1), 2)
        x2 = round(random.uniform(x1 + 1, image_width), 2)
        y2 = round(random.uniform(y1 + 1, image_height), 2)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    boxes = []
    for _ in range(int(random.uniform(1,4))):
        boxes.append(random_box())
    

    return boxes