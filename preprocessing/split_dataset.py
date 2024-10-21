import os
import shutil
import random

def split_dataset(image_dir: str, label_dir: str, output_dir: str, train_ratio=0.8):
    # Create output directories for YOLO format dataset
    train_images_dir = os.path.join(output_dir, 'images/train')
    val_images_dir = os.path.join(output_dir, 'images/val')
    train_labels_dir = os.path.join(output_dir, 'labels/train')
    val_labels_dir = os.path.join(output_dir, 'labels/val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # List all images and their corresponding labels
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Copy images and labels to corresponding train/val folders
    for img_list, img_folder, lbl_folder in [(train_images, train_images_dir, train_labels_dir), (val_images, val_images_dir, val_labels_dir)]:
        for image_file in img_list:
            label_file = image_file.replace('.png', '.txt')
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(img_folder, image_file))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(lbl_folder, label_file))

    # Create data.yaml file for YOLO
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {os.path.abspath(train_images_dir)}\n")
        f.write(f"val: {os.path.abspath(val_images_dir)}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['lesion']\n")

if __name__ == '__main__':
    split_dataset('./datasets/images/covered', './datasets/images/labels', './datasets/images/dataset')