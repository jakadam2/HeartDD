import torch
import numpy as np
import cv2 as cv
from PIL import Image
import preprocessing.mask_to_pixel as preprocessing
import pandas as pd
import os

class LesionDetector:

    def __init__(self, image, mask=None):
        self.model_path = './checkpoints/best_new.pt'
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:  # Convert only if grayscale
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        self.image = image
        self.mask = mask
        self.processed_image = None

    def detect(self):
        if self.mask is not None:
            dilated_mask = cv.dilate(self.mask, kernel=np.ones((5, 5)), iterations=10)
            if len(dilated_mask) == 3: # Convert if BGR
                dilated_mask = cv.cvtColor(dilated_mask, cv.COLOR_BGR2GRAY)

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        results = model(self.image)
        coordinates = []

        self.processed_image = self.image.copy()

        for *xyxy, conf, cls in results.xyxy[0]: 
            x1, y1, x2, y2 = map(int, xyxy)

            if self.mask is not None:
                x_mean, y_mean = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if dilated_mask[y_mean, x_mean] == 0:
                    continue

            coordinates.append({ 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 })

            cv.rectangle(self.processed_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        return coordinates

    def save_processed_image(self, output_path):
        if self.processed_image is not None:
            cv.imwrite(output_path, cv.cvtColor(self.processed_image, cv.COLOR_BGR2RGB))
        else:
            print("Processed image is not available. Please run detect() first.")

if __name__ == '__main__':
    #image_path = './datasets/images/raw/13c2ur549vohc0jat2wqi9t2i1_20.png'
    #image_path = './datasets/images/raw/131aedfhs6pnf1fvtvp49h4bhdjeabmt22_28.png'
    image_dir = './datasets/images/dataset2/images/test'
    mask_path = './datasets/images/good_df.csv'
    output_dir = './datasets/images/output'
    #mask_path = './datasets/images/masks/13c2ur549vohc0jat2wqi9t2i1_binmask.png'

    df = pd.read_csv(mask_path)

    for image_file in os.listdir(image_dir):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Extract image_id and frame from the filename
        image_id, frame = os.path.splitext(image_file)[0].rsplit('_', 1)
        frame = int(frame)

        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Find the best matching mask
        image_rows = df[df['image_id'] == image_id]
        if image_rows.empty:
            print(f"No mask found for image_id: {image_id}. Skipping.")
            continue

        exact_frame_row = image_rows[image_rows['frame'] == frame]
        if not exact_frame_row.empty:
            bitmask = exact_frame_row['segmentation'].values[0]
        else:
            # Use the first available mask for this image_id if no exact frame match
            closest_frame_row = image_rows.iloc[(image_rows['frame'] - frame).abs().argsort()[:1]]
            closest_frame = closest_frame_row['frame'].values[0]
            bitmask = closest_frame_row['segmentation'].values[0]
            print(f"[CLIENT] Exact frame not found. Using closest frame {closest_frame} for image_id {image_id}.")

        # Unpack the mask
        mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)

        # Perform lesion detection
        detector = LesionDetector(image, mask)
        coordinates = detector.detect()

        # Save the processed image with bounding boxes
        output_path = os.path.join(output_dir, f"processed_{image_file}")
        detector.save_processed_image(output_path)

    #image = Image.open(image_path)
    #mask = Image.open(mask_path)

    #image_rows = df[df['image_id'] == '131aedfhs6pnf1fvtvp49h4bhdjeabmt22']

    #exact_frame_row = image_rows[image_rows['frame'] == 28]

    #bitmask = exact_frame_row['segmentation'].values[0]

   # mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)

    #detector = LesionDetector(image, mask)
    #coordinates = detector.detect()
    #print("Detected coordinates:", coordinates)

    # Zapisanie obrazu z bounding boxami
    #output_path = './datasets/images/output/processed_image.png'
   # detector.save_processed_image(output_path)