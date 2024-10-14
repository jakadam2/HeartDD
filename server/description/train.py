import os
import datetime

import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from layers import *
from utils import *


def calculate_metrics(predictions, targets, threshold=0.2):
    """
    Calculates precision, recall, and F1 score based on predictions and true targets.
    
    Args:
        predictions (np.array): Predicted labels (continuous, e.g., probabilities).
        targets (np.array): True labels (binary, 0 or 1).
        threshold (float): Threshold to convert continuous predictions to binary (0 or 1).
        
    Returns:
        tuple: precision, recall, f1 score
    """
    pred_binary = (predictions >= threshold).numpy().astype(int).tolist()
    targets = targets.numpy().astype(int).tolist()
    precision = precision_score(targets, pred_binary, average='binary')
    recall = recall_score(targets, pred_binary, average='binary')
    f1 = f1_score(targets, pred_binary, average='binary')
    
    return precision, recall, f1


def evaluate_model(model,loader):
    """
    Evaluates the model on the given data loader and calculates recall, precision, and F1 score 
    for each feature (class) separately.
    
    Args:
        model: The trained model to be evaluated.
        loader: DataLoader for the evaluation data.
        
    Returns:
        DataFrame containing recall, precision, and F1 score for each feature.
    """
    model.eval()
    
    all_targets = torch.empty(size=(1,6))
    all_predictions = torch.empty(size=(1,6))

    with torch.no_grad(): 
        for inputs, targets in loader:
            inputs = inputs.to('cuda')
            outputs = model(inputs)
            all_predictions = torch.vstack((all_predictions,outputs.to('cpu')))
            all_targets = torch.vstack((all_targets,targets))

        print(all_targets.shape)
        print(all_predictions.shape)
        AORTO_OSTIAL_STENOSIS_pred = all_predictions[:,0]
        BLUNT_STUMP_pred  = all_predictions[:,1]
        BRIDGING_pred  = all_predictions[:,2]
        HEAVY_CALCIFICATION_pred  = all_predictions[:,3]
        SEVERE_TORTUOSITY_pred  = all_predictions[:,4]
        THROMBUS_pred  = all_predictions[:,5]
        
        AORTO_OSTIAL_STENOSIS_target = all_targets[:,0]
        BLUNT_STUMP_target  = all_targets[:,1]
        BRIDGING_target  = all_targets[:,2]
        HEAVY_CALCIFICATION_target  = all_targets[:,3]
        SEVERE_TORTUOSITY_target  = all_targets[:,4]
        THROMBUS_target  = all_targets[:,5]

        features = ['AORTO_OSTIAL_STENOSIS', 'BLUNT_STUMP', 'BRIDGING', 'HEAVY_CALCIFICATION', 'SEVERE_TORTUOSITY', 'THROMBUS']
        predictions_list = [AORTO_OSTIAL_STENOSIS_pred, BLUNT_STUMP_pred, BRIDGING_pred, HEAVY_CALCIFICATION_pred, SEVERE_TORTUOSITY_pred, THROMBUS_pred]
        targets_list = [AORTO_OSTIAL_STENOSIS_target, BLUNT_STUMP_target, BRIDGING_target, HEAVY_CALCIFICATION_target, SEVERE_TORTUOSITY_target, THROMBUS_target]

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred, target in zip(predictions_list, targets_list):
            precision, recall, f1 = calculate_metrics(pred, target)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        metrics_df = pd.DataFrame({
            'Feature': features,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1 Score': f1_scores
        })

        print(metrics_df)
        

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
            if torch.cuda.is_available():
                batch_images = batch_images.to('cuda')
                batch_targets = batch_targets.to('cuda')
                model.to('cuda')

            model.train()
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            pbar.update(1)

    return running_loss / len(train_loader)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, run_id):
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
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Loss: {avg_loss:.4f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.to('cuda'), targets.to('cuda'))
                val_loss += loss.item()
        print(f"Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(weights_dir, f'epoch_{epoch + 1}.pth'))


if __name__ == '__main__':
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001

    annotations_file = pd.read_csv('./../../datasets/final_df.csv') 
    img_dir = './../../datasets/images/raw'
    class_names = ['AORTO_OSTIAL_STENOSIS', 'BLUNT_STUMP', 'BRIDGING', 'HEAVY_CALCIFICATION', 'SEVERE_TORTUOSITY', 'THROMBUS']

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((55,55)),
    ])


    anno_train,anno_test = train_test_split_pandas(annotations_file)

    train_dataset = ImageDataset(anno_train, img_dir=img_dir, class_names=class_names, transform=data_transforms)
    val_dataset = ImageDataset(anno_test, img_dir=img_dir, class_names=class_names, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    hog_layer = HDHOGLayer()
    model = HDModel(hog=hog_layer)
    criterion = HDLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_id = datetime.datetime.now().strftime("%H:%M %Y-%m-%d")
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, run_id=run_id)

    hog_layer = HDHOGLayer()
    new_model = HDModel(hog=hog_layer).to('cuda')
    new_model.load_state_dict(torch.load(os.path.join(f'weights/run_{run_id}', f'epoch_{num_epochs}.pth'),weights_only=True))
    evaluate_model(new_model,val_loader)
