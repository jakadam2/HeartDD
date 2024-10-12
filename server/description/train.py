import os
import datetime

import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from layers import *
from utils import *


def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
        for batch_images, batch_targets in train_loader:
            # Przeniesienie danych na GPU, jeśli dostępne
            if torch.cuda.is_available():
                batch_images = batch_images.to('cuda')
                batch_targets = {key: value.to('cuda') for key, value in batch_targets.items()}
                model.to('cuda')

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            pbar.update(1)

    return running_loss / len(train_loader)


def train(model, train_loader, criterion, optimizer, num_epochs, run_id):
    """
    Train the model for a specified number of epochs and save weights.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        num_epochs: Number of epochs to train for.
        run_id: Unique identifier for the training run.
    """
    weights_dir = f'weights/run_{run_id}'
    os.makedirs(weights_dir, exist_ok=True)  # Tworzenie folderu dla wag, jeśli nie istnieje

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Loss: {avg_loss:.4f}")


        torch.save(model.state_dict(), os.path.join(weights_dir, f'epoch_{epoch + 1}.pth'))





if __name__ == '__main__':
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    annotations_file = pd.read_csv('./../../datasets/data.csv') 
    img_dir = './../../datasets/images'
    class_names = ['AORTO_OSTIAL_STENOSIS', 'BLUNT_STUMP', 'BRIDGING', 'HEAVY_CALCIFICATION', 'SEVERE_TORTUOSITY', 'THROMBUS']

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, class_names=class_names, transform=data_transforms)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hog_layer = HDHOGLayer()
    model = HDModel(hog=hog_layer)
    criterion = HDLoss(weights={'AORTO_OSTIAL_STENOSIS': 1.5,
                                'BLUNT_STUMP': 1.0,
                                'BRIDGING': 2.0})
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    train(model, train_loader, criterion, optimizer, num_epochs, run_id=run_id)
