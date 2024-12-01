import torch
import os
import numpy as np
import cv2 as cv
from PIL import Image
import preprocessing.mask_to_pixel as preprocessing
import pandas as pd
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class LesionDetectorRoi:

    def __init__(self, image, mask=None):
        self.model_path = './checkpoints/best_new.pt'
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:  # Convert only if grayscale
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        self.image = image
        self.mask = mask
        self.processed_image = None 
        self.skeletonized_mask = None
        self.heatmap = None

    @staticmethod
    def cover_image(image, mask):
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] = (image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) == 0] * 0.3).astype(np.uint8)
        image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] = np.clip(
            image[cv.dilate(mask, kernel=np.ones((5,5)), iterations=10) != 0] * 1.5, 
            0, 255
        ).astype(np.uint8)
        return image

    def detect(self, roi_step, roi_size):
        if self.mask is not None:
            #self.image = self.cover_image(self.image, self.mask)
            self.skeletonized_mask = skeletonize(self.mask)
            skeleton_coords = np.column_stack(np.where(self.skeletonized_mask))

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        results_heatmap = np.zeros(self.image.shape[:2], dtype=np.float32)

        self.processed_image = self.image.copy()

        # roi_step for faster detection
        for i, (y, x) in enumerate(skeleton_coords[::roi_step]):
            x_start, x_end = max(0, x - roi_size[1] // 2), min(self.image.shape[1], x + roi_size[1] // 2)
            y_start, y_end = max(0, y - roi_size[0] // 2), min(self.image.shape[0], y + roi_size[0] // 2)
            
            roi = self.image[y_start:y_end, x_start:x_end]
            results = model(roi)
            
            # Process detections for the ROI
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                # Normalize coordinates to fit in the heatmap
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                results_heatmap[y_center, x_center] += float(conf)  # Add confidence score to heatmap

                cv.rectangle(self.processed_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        # Smooth the heatmap for visualization
        self.heatmap = gaussian_filter(results_heatmap, sigma=5)

    def save_processed_image(self, output_path):
        if self.processed_image is not None:
            cv.imwrite(output_path, cv.cvtColor(self.processed_image, cv.COLOR_BGR2RGB))
        else:
            print("Processed image is not available. Please run detect() first.")

    def save_heatmap(self, output_path):
        if self.heatmap is not None:
            plt.figure(figsize=(8, 8))
            plt.title("Heatmap of Detected Lesions")
            plt.imshow(self.heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar(label="Confidence Score")
            plt.savefig(output_path)
            plt.close()
        else:
            print("Heatmap not available. Please run detect() first.")

    def save_skeleton(self, output_path, roi_step, roi_size):
        if self.skeletonized_mask is not None:
            # Copy the skeletonized mask to visualize the ROI centers
            skeleton_image = (self.skeletonized_mask * 255).astype(np.uint8)
            skeleton_image_color = cv.cvtColor(skeleton_image, cv.COLOR_GRAY2BGR)

            # Get the coordinates of ROI centers (every step along the skeleton)
            skeleton_coords = np.column_stack(np.where(self.skeletonized_mask))
            
            for y, x in skeleton_coords:
                # Add a circle at the center of each ROI
                cv.circle(skeleton_image_color, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

                # Optionally, draw the ROI rectangle (for debugging)
                #x_start, x_end = max(0, x - roi_size[1] // 2), min(self.image.shape[1], x + roi_size[1] // 2)
                #y_start, y_end = max(0, y - roi_size[0] // 2), min(self.image.shape[0], y + roi_size[0] // 2)
                #cv.rectangle(skeleton_image_color, (x_start, y_start), (x_end, y_end), color=(255, 0, 0), thickness=1)

            # Save the skeleton with ROI centers
            cv.imwrite(output_path, skeleton_image_color)
        else:
            print("Skeleton not available. Please ensure the mask has been processed.")
        
        

if __name__ == '__main__':
    #image_path = './datasets/images/raw/131aedfhs6pnf1fvtvp49h4bhdjeabmt22_28.png'
    image_dir = './datasets/images/dataset2/images/test'
    mask_path = './datasets/images/good_df.csv'
    output_dir = './datasets/images/output2'
    #mask_path = './datasets/images/masks/13c2ur549vohc0jat2wqi9t2i1_binmask.png'

    df = pd.read_csv(mask_path)

    roi_step = 10
    roi_size = (128,128)

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
            bitmask = closest_frame_row['segmentation'].values[0]
        # Unpack the mask
        mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)

        # Perform lesion detection
        detector = LesionDetectorRoi(image, mask)
        coordinates = detector.detect(roi_step, roi_size)

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

   
    #detector = LesionDetectorRoi(image, mask)
    #detector.detect(roi_step, roi_size)
    
    #output_heatmap_path = './datasets/images/output/heatmap.png'
    #detector.save_heatmap(output_heatmap_path)

    #output_skeleton_path = './datasets/images/output/skeleton.png'
    #detector.save_skeleton(output_skeleton_path,roi_step, roi_size)

    #output_path = './datasets/images/output/processed_image_roi.png'
    #detector.save_processed_image(output_path)