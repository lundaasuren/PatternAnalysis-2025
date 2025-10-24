"""
Siamese Network architecture for melanoma classification.
Uses a shared CNN backbone with contrastive loss for learning embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet34_Weights, EfficientNet_B0_Weights
from typing import Tuple, Optional


class EmbeddingNetwork(nn.Module):
    """
    Embedding network that extracts features from images.
    Uses a pretrained backbone (ResNet50) with custom embedding head.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding space
            backbone: Backbone architecture ('resnet50', 'resnet34', 'efficientnet_b0')
            pretrained: If True, use ImageNet pretrained weights
            dropout: Dropout rate for regularization
        """
        super(EmbeddingNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_out_features = self.backbone.fc.in_features
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            backbone_out_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            backbone_out_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Embedding head: projects backbone features to embedding space
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract embeddings.
        
        Args:
            x: Input images [batch_size, 3, H, W]
        
        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.embedding_head(features)
        
        # L2 normalize embeddings (important for contrastive learning)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class SiameseNetwork(nn.Module):
    """
    Siamese Network for melanoma classification.
    Processes pairs of images through shared weights to learn similarity.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding space
            backbone: Backbone architecture
            pretrained: If True, use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super(SiameseNetwork, self).__init__()
        
        # Single embedding network shared by both inputs
        self.embedding_net = EmbeddingNetwork(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout
        )
        
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of images.
        
        Args:
            img1: First image [batch_size, 3, H, W]
            img2: Second image [batch_size, 3, H, W]
        
        Returns:
            Tuple of (embedding1, embedding2)
        """
        # Process both images through the same network
        embedding1 = self.embedding_net(img1)
        embedding2 = self.embedding_net(img2)
        
        return embedding1, embedding2
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a single image (used during inference).
        
        Args:
            x: Input image [batch_size, 3, H, W]
        
        Returns:
            Embedding [batch_size, embedding_dim]
        """
        return self.embedding_net(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    Pulls similar pairs together and pushes dissimilar pairs apart.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embedding [batch_size, embedding_dim]
            embedding2: Second embedding [batch_size, embedding_dim]
            label: Similarity labels [batch_size]
                   1 for similar pairs (same class)
                   0 for dissimilar pairs (different class)
        
        Returns:
            Contrastive loss (scalar)
        """
        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss formula:
        # L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2
        # where Y=1 for dissimilar, Y=0 for similar
        
        # For similar pairs (label=1): minimize distance
        loss_similar = label * torch.pow(euclidean_distance, 2)
        
        # For dissimilar pairs (label=0): maximize distance up to margin
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        # Total loss
        loss = 0.5 * torch.mean(loss_similar + loss_dissimilar)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss as an alternative to Contrastive Loss.
    Uses anchor, positive, and negative samples.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
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
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings (same class) [batch_size, embedding_dim]
            negative: Negative embeddings (different class) [batch_size, embedding_dim]
        
        Returns:
            Triplet loss (scalar)
        """
        # Distance between anchor and positive
        pos_distance = F.pairwise_distance(anchor, positive)
        
        # Distance between anchor and negative
        neg_distance = F.pairwise_distance(anchor, negative)
        
        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))
        
        return loss


class SiameseClassifier(nn.Module):
    """
    Complete Siamese Network with classification head.
    Can be used for direct binary classification.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            embedding_dim: Dimension of the embedding space
            backbone: Backbone architecture
            pretrained: If True, use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super(SiameseClassifier, self).__init__()
        
        # Siamese network for embedding
        self.siamese = SiameseNetwork(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout
        )
        
        # Classification head (optional - for direct prediction)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            img1: First image [batch_size, 3, H, W]
            img2: Second image [batch_size, 3, H, W]
            return_embeddings: If True, return embeddings instead of predictions
        
        Returns:
            Similarity predictions [batch_size, 1] or embeddings
        """
        # Get embeddings
        emb1, emb2 = self.siamese(img1, img2)
        
        if return_embeddings:
            return emb1, emb2
        
        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=1)
        
        # Predict similarity
        similarity = self.classifier(combined)
        
        return similarity


def compute_accuracy(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute accuracy for Siamese network predictions.
    
    Args:
        embedding1: First embeddings [batch_size, embedding_dim]
        embedding2: Second embeddings [batch_size, embedding_dim]
        labels: True labels [batch_size] (1=similar, 0=dissimilar)
        threshold: Distance threshold for classification
    
    Returns:
        Accuracy as float
    """
    # Compute distances
    distances = F.pairwise_distance(embedding1, embedding2)
    
    # Predict: if distance < threshold, predict similar (1), else dissimilar (0)
    predictions = (distances < threshold).float()
    
    # Compute accuracy
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    return accuracy


def get_model(
    embedding_dim: int = 256,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5,
    device: str = 'cuda'
) -> Tuple[SiameseNetwork, ContrastiveLoss]:
    """
    Factory function to create Siamese network and loss.
    
    Args:
        embedding_dim: Dimension of the embedding space
        backbone: Backbone architecture
        pretrained: If True, use ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to move model to
    
    Returns:
        Tuple of (model, criterion)
    """
    model = SiameseNetwork(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    ).to(device)
    
    criterion = ContrastiveLoss(margin=1.0)
    
    return model, criterion


# Test the module
if __name__ == "__main__":
    """
    Test script to verify the model architecture.
    """
    print("Testing Siamese Network Architecture\n")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    print("Creating model...")
    model, criterion = get_model(
        embedding_dim=256,
        backbone='resnet50',
        pretrained=True,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 224, 224).to(device)
    img2 = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # Forward pass
    with torch.no_grad():
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        accuracy = compute_accuracy(emb1, emb2, labels)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {img1.shape}")
    print(f"  Embedding 1 shape: {emb1.shape}")
    print(f"  Embedding 2 shape: {emb2.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Test single embedding extraction
    print("\nTesting single image embedding...")
    with torch.no_grad():
        single_emb = model.get_embedding(img1)
    print(f"✓ Single embedding extraction successful")
    print(f"  Embedding shape: {single_emb.shape}")
    
    # Test distance computation
    print("\nTesting distance computation...")
    distances = F.pairwise_distance(emb1, emb2)
    print(f"✓ Distance computation successful")
    print(f"  Distances: {distances.cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Architecture is ready for training.")
    print("=" * 60)