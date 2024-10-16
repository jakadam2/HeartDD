import torch
from torch import nn
import torch.nn.functional as F

from skimage import feature

from collections import defaultdict


class HDHOGLayer(nn.Module):
    """
    A layer that computes Histogram of Oriented Gradients (HOG) features from input tensors.

    This layer is typically used for feature extraction in image processing tasks.

    Parameters:
    ----------
    orientations : int, optional
        The number of orientation bins for the HOG feature descriptor (default is 9).
    
    pixels_per_cell : tuple[int, int], optional
        The size (height, width) of each cell in pixels (default is (8, 8)).
    
    cells_per_block : tuple[int, int], optional
        The number of cells in each block (default is (3, 3)).

    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Computes the HOG features for the input tensor.
    """

    def __init__(self, orientations: int = 9, pixels_per_cell: tuple[int, int] = (8, 8), cells_per_block: tuple[int, int] = (3, 3)) -> None:
        super(HDHOGLayer, self).__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            A batch of images as a 4D tensor with shape (batch_size, channels, height, width).
        
        Returns:
        -------
        torch.Tensor
            A tensor of HOG features for each image in the batch.
        """
        # Sprawdzenie czy tensor ma 4 wymiary: (batch_size, channels, height, width)
        assert len(x.shape) == 4, "Input tensor must be 4-dimensional (batch_size, channels, height, width)"
        
        # Upewnij się, że obrazy mają 1 kanał (dla HOG, zwykle grayscale)
        if x.shape[1] != 1:
            raise ValueError("HOG expects grayscale images with 1 channel")

        # Lista do przechowywania cech HOG dla każdego obrazu
        hog_features = []

        # Iteracja przez batch
        for i in range(x.shape[0]):
            # Weź obraz i usuń kanał (1, height, width) -> (height, width)
            img = x[i].squeeze(0).cpu().numpy()  # Przekształć do Numpy
            
            # Oblicz cechy HOG dla tego obrazu
            hog_feature = feature.hog(
                img, 
                orientations=self.orientations, 
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block, 
                transform_sqrt=True, 
                block_norm="L2",
                visualize=False,  # visualize=False, bo zwracamy tylko wektor cech
                feature_vector=True
            )
            
            # Dodaj cechy do listy
            hog_features.append(hog_feature)
        
        # Przekształć listę wyników w tensor PyTorch (batch_size, num_features)
        return torch.tensor(hog_features, dtype=torch.float32).to('cuda')


class HDClassifier(nn.Module):
    """
    A fully connected neural network classifier for classification tasks.

    This classifier consists of several linear layers with batch normalization, dropout, and ReLU activation.

    Parameters:
    ----------
    in_dim : int
        The size of the input feature vector.
    
    out_dim : int
        The size of the output (number of classes).

    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Passes the input through the network layers to obtain predictions.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(HDClassifier, self).__init__()
        self.dl1 = nn.Linear(in_features=in_dim, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dl2 = nn.Linear(in_features=512, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dl3 = nn.Linear(in_features=128, out_features=out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        features = self.dl1(features)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.relu(features)
        features = self.dl2(features)
        features = self.bn2(features)
        features = self.dropout(features)
        features = self.dl3(features)
        return features


class HDModel(nn.Module):
    """
    A model that combines HOG feature extraction and multiple classifiers for different medical conditions.

    This model utilizes HOG features as input for several classifiers targeting various medical conditions.

    Parameters:
    ----------
    hog : HDHOGLayer, optional
        An instance of the HDHOGLayer for feature extraction (default is None).

    Methods:
    -------
    forward(x: torch.Tensor) -> dict:
        Passes the input through the HOG layer and each classifier, returning a dictionary of predictions.
    """

    def __init__(self, hog: HDHOGLayer = None) -> None:
        super(HDModel, self).__init__()
        self.hog = hog
        self.AORTO_OSTIAL_STENOSIS_classifier = HDClassifier(in_dim=1296, out_dim=1)
        self.BLUNT_STUMP_classifier = HDClassifier(in_dim=1296, out_dim=1)
        self.BRIDGING_classifier = HDClassifier(in_dim=1296, out_dim=1)
        self.HEAVY_CALCIFICATION_classifier = HDClassifier(in_dim=1296, out_dim=1)
        self.SEVERE_TORTUOSITY_classifier = HDClassifier(in_dim=1296, out_dim=1)
        self.THROMBUS_classifier = HDClassifier(in_dim=1296, out_dim=1)

    def forward(self, x: torch.Tensor) -> dict:
        hog_features = self.hog(x)
        AORTO_OSTIAL_STENOSIS = self.AORTO_OSTIAL_STENOSIS_classifier(hog_features)
        BLUNT_STUMP = self.BLUNT_STUMP_classifier(hog_features)
        BRIDGING = self.BRIDGING_classifier(hog_features)
        HEAVY_CALCIFICATION = self.HEAVY_CALCIFICATION_classifier(hog_features)
        SEVERE_TORTUOSITY = self.SEVERE_TORTUOSITY_classifier(hog_features)
        THROMBUS = self.THROMBUS_classifier(hog_features)
        return torch.hstack((AORTO_OSTIAL_STENOSIS
                             ,BLUNT_STUMP
                             ,BRIDGING
                             ,HEAVY_CALCIFICATION
                             ,SEVERE_TORTUOSITY
                             ,THROMBUS))


class HDLoss(nn.Module):
    """
    Custom loss function for multi-label binary classification.

    This class implements a weighted binary cross-entropy loss, allowing 
    different weights to be assigned to each class. The weights can be used
    to emphasize or de-emphasize the importance of specific classes during 
    the training process.

    Attributes:
        loss_fn (nn.Module): The binary cross-entropy loss function 
            used for computing the loss for each class.
        weights (defaultdict): A dictionary mapping class names to their
            corresponding weights. If a class is not in the dictionary, 
            a default weight of 1.0 is assigned.

    Parameters:
        weights (dict, optional): A dictionary of weights for each class.
            If not provided, all classes will have a weight of 1.0.
            
    Example:
        >>> loss_function = HDLoss(weights={'AORTO_OSTIAL_STENOSIS': 1.5, 
        ...                                   'BLUNT_STUMP': 1.0,
        ...                                   'BRIDGING': 2.0})
        >>> loss = loss_function(outputs, targets)
    """
    steps = [1,1,1,1,1,1]

    def __init__(self) -> None:
        super(HDLoss,self).__init__()
        pos_weight = torch.tensor([8.0]).to('cuda')
        self._loss = nn.BCEWithLogitsLoss(pos_weight = pos_weight) 

    def forward(self,predicts,labels):
        cum_loss = 0

        j = 0
        labels = labels.squeeze(1)

        for i in range(labels.shape[1]):
            cum_loss += self._loss(predicts[:,j:j + HDLoss.steps[i]],labels[:,i].unsqueeze(1))
            j += HDLoss.steps[i]
        return cum_loss