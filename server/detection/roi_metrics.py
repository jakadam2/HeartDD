import torch
import os
import numpy as np
import cv2 as cv
from PIL import Image
import preprocessing.mask_to_pixel as preprocessing
import pandas as pd
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

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
        self.detected_boxes = []  # Detected bounding boxes [(x1, y1, x2, y2, confidence)]

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def load_ground_truth(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                _, x_center, y_center, width, height = map(float, line.strip().split())
                image_height, image_width = self.image.shape[:2]
                x_center = abs(x_center * image_width)
                y_center = abs(y_center * image_height)
                width = abs(width * image_width)
                height = abs(height * image_height)
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                boxes.append((x1, y1, x2, y2))
        return boxes

    def detect(self, roi_step, roi_size):
        if self.mask is not None:
            skeletonized_mask = skeletonize(self.mask)
            skeleton_coords = np.column_stack(np.where(skeletonized_mask))

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)

        self.processed_image = self.image.copy()

        for (y, x) in skeleton_coords[::roi_step]:
            x_start, x_end = max(0, x - roi_size[1] // 2), min(self.image.shape[1], x + roi_size[1] // 2)
            y_start, y_end = max(0, y - roi_size[0] // 2), min(self.image.shape[0], y + roi_size[0] // 2)

            roi = self.image[y_start:y_end, x_start:x_end]
            results = model(roi)

            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                self.detected_boxes.append([x1, y1, x2, y2])

    def save_processed_image(self, output_path, ground_truth_boxes, iou_threshold):
        tp_boxes = []

        for detected_box in self.detected_boxes:
            for gt_box in ground_truth_boxes:
                iou = self.calculate_iou(detected_box, gt_box)
                if iou >= iou_threshold:
                    tp_boxes.append(detected_box)
                    cv.rectangle(self.processed_image, (detected_box[0], detected_box[1]),
                                (detected_box[2], detected_box[3]), (255, 0, 0), 2)

        cv.imwrite(output_path, cv.cvtColor(self.processed_image, cv.COLOR_BGR2RGB))
        return len(tp_boxes), len(self.detected_boxes) - len(tp_boxes)


if __name__ == '__main__':
    image_dir = './datasets/images/dataset2/images/test'
    mask_path = './datasets/images/good_df.csv'
    label_dir = './datasets/images/dataset2/labels/test'
    output_dir = './datasets/images/output2'

    df = pd.read_csv(mask_path)
    roi_step = 10
    roi_size = (90, 90)
    iou_threshold = 0.6

    os.makedirs(output_dir, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0

    for image_file in os.listdir(image_dir):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_id, frame = os.path.splitext(image_file)[0].rsplit('_', 1)
        frame = int(frame)
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        file_name, _ = os.path.splitext(image_file)
        label_path = os.path.join(label_dir, file_name + '.txt')
        if not os.path.exists(label_path):
            print(f"Ground truth not found for {image_file}. Skipping.")
            continue

        image_rows = df[df['image_id'] == image_id]
        if image_rows.empty:
            print(f"No mask found for image_id: {image_id}. Skipping.")
            continue

        exact_frame_row = image_rows[image_rows['frame'] == frame]
        if not exact_frame_row.empty:
            bitmask = exact_frame_row['segmentation'].values[0]
        else:
            closest_frame_row = image_rows.iloc[(image_rows['frame'] - frame).abs().argsort()[:1]]
            bitmask = closest_frame_row['segmentation'].values[0]

        mask = preprocessing.MaskUnpacker._unpack_mask(bitmask)

        detector = LesionDetectorRoi(image, mask)
        detector.detect(roi_step, roi_size)

        ground_truth_boxes = detector.load_ground_truth(label_path)

        output_path = os.path.join(output_dir, f"processed_{image_file}")
        tp, fp = detector.save_processed_image(output_path, ground_truth_boxes, iou_threshold)
        fn = len(ground_truth_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")