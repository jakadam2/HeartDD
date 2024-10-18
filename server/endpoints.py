#THIS IS THE FILE USED TO IMPLEMENT ENDPOINTS TO INTERACT WITH MODELS
#REPLACE THESE FUNCTIONS WITH WHATEVER YOU NEED

from PIL import Image
import random

def detect_bounding_boxes(image: Image) -> [{}]:
    # This function takes an image in the PIL.Image format
    # It returns positions of the bounding boxes in format
    # {x1:x1, y1:y1, x2:x2, y2:y2}
        
    # Returning mock bounding boxes for demonstration
    return [{'x1': 10, 'y1': 20, 'x2': 50, 'y2': 60},
                {'x1': 100, 'y1': 120, 'x2': 30, 'y2': 40}]

def describe_bbox(image: Image, bbox: []) -> [{}]:
    # This function takes the whole image and the bbox in the format described above
    # It should return a list of confidences in format
    # {name: name, confidence: confidence}

    # Simulate detection of some objects in the image with random confidence scores
    object_names = ['Person', 'Car', 'Tree', 'Dog', 'Cat']  # Example object classes
    confidences = []
    # Randomly generate some detections (between 1 and 5)
    num_detections = random.randint(1, 5)

    for _ in range(num_detections):
        name = random.choice(object_names)  # Randomly select an object
        confidence = round(random.uniform(0.5, 1.0), 2)  # Random confidence between 0.5 and 1.0
        confidences.append({
            "name": name,
            "confidence": confidence
        })

    return confidences