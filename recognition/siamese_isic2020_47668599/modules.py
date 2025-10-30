"""
Siamese Network with ResNet50 for melanoma classification using triplet loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    """Siamese Network with shared ResNet50 feature extractor."""
    
    def __init__(self):
        super().__init__()
        self._feature_extractor = resnet50(weights=None, progress=False)
        
    def forward_once(self, image: torch.Tensor) -> torch.Tensor:
        return self._feature_extractor(image)
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.forward_once(image1),
            self.forward_once(image2),
            self.forward_once(image3)
        )


class BinaryClassifier(nn.Module):
    """Binary classifier for extracted Siamese features."""
    
    def __init__(self):
        super().__init__()
        self._layer1 = nn.Linear(1000, 500)
        self._layer2 = nn.Linear(500, 100)
        self._layer3 = nn.Linear(100, 50)
        self._layer4 = nn.Linear(50, 2)
        self._relu = nn.ReLU()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self._relu(self._layer1(features))
        x = self._relu(self._layer2(x))
        x = self._relu(self._layer3(x))
        return self._layer4(x)


class TripletLoss(nn.Module):
    """Triplet margin loss: d(a,p) - d(a,n) + margin."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        return loss.mean()


def get_model(device: str = 'cuda') -> tuple[SiameseNetwork, TripletLoss]:
    """Initialize Siamese network and triplet loss."""
    model = SiameseNetwork().to(device)
    criterion = TripletLoss(margin=1.0)
    return model, criterion
