"""
Simplified Siamese Network for melanoma classification.
Uses ResNet50 backbone without pretrained weights for triplet learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    """
    Siamese Network using ResNet50 to extract features.
    Processes triplets (anchor, positive, negative) through shared weights.
    """
    
    def __init__(self):
        """Initialize the model with ResNet50 feature extractor."""
        super().__init__()
        
        # ResNet50 without pretrained weights
        self._feature_extractor = resnet50(weights=None, progress=False)
        
    def forward_once(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image.
        
        Args:
            image: Input image tensor [batch_size, 3, H, W]
        
        Returns:
            Extracted features [batch_size, 1000]
        """
        return self._feature_extractor(image)
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        image3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for three images (anchor, positive, negative).
        
        Args:
            image1: Anchor image [batch_size, 3, H, W]
            image2: Positive image [batch_size, 3, H, W]
            image3: Negative image [batch_size, 3, H, W]
        
        Returns:
            Tuple of (anchor_features, positive_features, negative_features)
        """
        return (
            self.forward_once(image1),
            self.forward_once(image2),
            self.forward_once(image3)
        )


class BinaryClassifier(nn.Module):
    """
    Binary classifier for Siamese Network features.
    Takes extracted features and outputs binary classification.
    """
    
    def __init__(self):
        """Initialize the classifier layers."""
        super().__init__()
        
        self._layer1 = nn.Linear(1000, 500)
        self._layer2 = nn.Linear(500, 100)
        self._layer3 = nn.Linear(100, 50)
        self._layer4 = nn.Linear(50, 2)
        self._relu = nn.ReLU()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            features: Extracted features [batch_size, 1000]
        
        Returns:
            Class logits [batch_size, 2]
        """
        x = self._relu(self._layer1(features))
        x = self._relu(self._layer2(x))
        x = self._relu(self._layer3(x))
        return self._layer4(x)


class TripletLoss(nn.Module):
    """
    Triplet Loss for Siamese Network training.
    Minimizes distance between anchor-positive, maximizes anchor-negative.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor features [batch_size, feature_dim]
            positive: Positive features [batch_size, feature_dim]
            negative: Negative features [batch_size, feature_dim]
        
        Returns:
            Triplet loss (scalar)
        """
        # Distance between anchor and positive
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        
        # Distance between anchor and negative
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        return loss.mean()


def get_model(device: str = 'cuda') -> tuple[SiameseNetwork, TripletLoss]:
    """
    Create Siamese network and triplet loss.
    
    Args:
        device: Device to move model to ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, criterion)
    """
    model = SiameseNetwork().to(device)
    criterion = TripletLoss(margin=1.0)
    
    return model, criterion


# Test the module
if __name__ == "__main__":
    print("Testing Siamese Network\n")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    print("Creating model...")
    model, criterion = get_model(device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    anchor = torch.randn(batch_size, 3, 224, 224).to(device)
    positive = torch.randn(batch_size, 3, 224, 224).to(device)
    negative = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        anchor_feat, pos_feat, neg_feat = model(anchor, positive, negative)
        loss = criterion(anchor_feat, pos_feat, neg_feat)
    
    print(f"✓ Forward pass successful")
    print(f"  Anchor features: {anchor_feat.shape}")
    print(f"  Positive features: {pos_feat.shape}")
    print(f"  Negative features: {neg_feat.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test with classifier
    print("\nTesting binary classifier...")
    classifier = BinaryClassifier().to(device)
    with torch.no_grad():
        logits = classifier(anchor_feat)
    print(f"✓ Classifier successful")
    print(f"  Output shape: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
