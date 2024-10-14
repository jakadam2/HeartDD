import os
import cv2
import numpy as np
from torchvision import transforms

class ImageAugmentor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),  # Losowe odbicie w poziomie
            transforms.RandomRotation(degrees=25),    # Losowy obrót w zakresie od -25 do 25 stopni
            transforms.RandomResizedCrop(size=(512,512), scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Losowe rozmycie
        ])

    def augment_image(self, image):
        augmented_image = self.transform(image)
        return augmented_image

    def process_images(self):
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
                img_path = os.path.join(self.input_folder, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

                if image is not None:
                    output_path = f"{os.path.splitext(filename)[0]}_aug_{0}{os.path.splitext(filename)[1]}"
                    output_path = os.path.join(self.output_folder,output_path)
                    cv2.imwrite(output_path, image)
                    for i in range(1,5):
                        augmented_image = self.augment_image(image)
                        augmented_image = augmented_image.squeeze().numpy()
                        augmented_image = (augmented_image * 255).astype(np.uint8)
                        output_path = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                        output_path = os.path.join(self.output_folder,output_path)
                        cv2.imwrite(output_path, augmented_image)
    def run(self):
        self.process_images()


if __name__ == "__main__":
    augmentor = ImageAugmentor('./datasets/images/covered','./datasets/images/augmented')
    augmentor.run()
