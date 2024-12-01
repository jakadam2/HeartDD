import os
import shutil
import random

def split_dataset(image_dir: str, label_dir: str, output_dir: str, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):

    # Create output directories for YOLO format dataset
    train_images_dir = os.path.join(output_dir, 'images/train')
    val_images_dir = os.path.join(output_dir, 'images/val')
    test_images_dir = os.path.join(output_dir, 'images/test')
    train_labels_dir = os.path.join(output_dir, 'labels/train')
    val_labels_dir = os.path.join(output_dir, 'labels/val')
    test_labels_dir = os.path.join(output_dir, 'labels/test')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    # List all images and shuffle
    images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    random.shuffle(images)

    # Calculate sizes for train, val, and test
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size

    # Split images into train, val, and test sets
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Copy images and labels to corresponding folders
    for img_list, img_folder, lbl_folder in [
        (train_images, train_images_dir, train_labels_dir), 
        (val_images, val_images_dir, val_labels_dir), 
        (test_images, test_images_dir, test_labels_dir)
    ]:
        for image_file in img_list:
            label_file = image_file.replace('.png', '.txt')
            shutil.copy(os.path.join(image_dir, image_file), os.path.join(img_folder, image_file))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(lbl_folder, label_file))

    # Create data.yaml file for YOLO
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {os.path.abspath(train_images_dir)}\n")
        f.write(f"val: {os.path.abspath(val_images_dir)}\n")
        f.write(f"test: {os.path.abspath(test_images_dir)}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['lesion']\n")

if __name__ == '__main__':
    split_dataset('./datasets/images/raw', './datasets/images/labels', './datasets/images/dataset2')