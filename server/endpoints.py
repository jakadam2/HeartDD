#THIS IS THE FILE USED TO IMPLEMENT ENDPOINTS TO INTERACT WITH MODELS
#REPLACE THESE FUNCTIONS WITH WHATEVER YOU NEED
from description.lesion_desriber import LesionDescriber

from PIL import Image
import random

from typing import Union

from torchvision import transforms

def detect_bounding_boxes(image: Image) -> [{}]:
    # This function takes an image in the PIL.Image format
    # It returns positions of the bounding boxes in format
    # {x1:x1, y1:y1, x2:x2, y2:y2}
    w, h = image.size
    # Returning mock bounding boxes for demonstration
    return [{'x1': 10, 'y1': 20, 'x2': random.uniform(11, w), 'y2': 60},
                {'x1': 100, 'y1': 120, 'x2': 30, 'y2': random.uniform(120, h)}]

def describe_bbox(image: Image,mask: Image, bboxes: list[Union[int,int]]) -> list[dict[str:float]]:

    describer = LesionDescriber()
    results = []
    pll = transforms.ToTensor()
    image_tensor = pll(image)
    mask_tensor = pll(mask)

    for bbox in bboxes:
        results.append(describer(image_tensor,mask_tensor,bbox))
    return results