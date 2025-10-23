"""
Training script for Siamese Network on ISIC 2020 dataset.
Includes strategies for handling severe class imbalance (98% benign, 2% melanoma).
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from dataset import create_data_loaders, ISICSiameseDataset, ISICTripletDataset
from modules import get_model, compute_accuracy, ContrastiveLoss, TripletLoss
import torch.nn.functional as F


class WeightedContrastiveLoss(nn.Module):
    """
    Weighted Contrastive Loss to handle class imbalance.
    Gives higher weight to positive (melanoma) pairs.
    """
    
    def __init__(self, margin: float = 1.0, pos_weight: float = 10.0):
        """
        Args:
            margin: Margin for dissimilar pairs
            pos_weight: Weight multiplier for positive (melanoma) pairs
        """
        super(WeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_weight = pos_weight
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted contrastive loss.
        
        Args:
            embedding1: First embedding [batch_size, embedding_dim]
            embedding2: Second embedding [batch_size, embedding_dim]
            label: Similarity labels [batch_size]
                   1 for similar pairs (same class)
                   0 for dissimilar pairs (different class)
        
        Returns:
            Weighted contrastive loss (scalar)
        """
        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # For similar pairs (label=1): minimize distance
        loss_similar = label * torch.pow(euclidean_distance, 2)
        
        # For dissimilar pairs (label=0): maximize distance up to margin
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        # Apply weights (give more importance to similar pairs from minority class)
        # In our pair generation, similar pairs include melanoma-melanoma pairs
        weighted_loss_similar = self.pos_weight * loss_similar
        
        # Total loss
        loss = 0.5 * torch.mean(weighted_loss_similar + loss_dissimilar)
        
        return loss


class FocalContrastiveLoss(nn.Module):
    """
    Focal Contrastive Loss - focuses on hard examples.
    Inspired by Focal Loss for object detection.
    """
    
    def __init__(self, margin: float = 1.0, gamma: float = 2.0):
        """
        Args:
            margin: Margin for dissimilar pairs
            gamma: Focusing parameter (higher = focus more on hard examples)
        """
        super(FocalContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Standard contrastive loss components
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        # Apply focal weighting (down-weight easy examples)
        pt_similar = torch.exp(-loss_similar)
        focal_weight_similar = (1 - pt_similar) ** self.gamma
        
        pt_dissimilar = torch.exp(-loss_dissimilar)
        focal_weight_dissimilar = (1 - pt_dissimilar) ** self.gamma
        
        # Weighted loss
        loss = 0.5 * torch.mean(
            focal_weight_similar * loss_similar + 
            focal_weight_dissimilar * loss_dissimilar
        )
        
        return loss


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_triplet: bool = False
) -> tuple:
    """
    Train for one epoch.
    
    Args:
        use_triplet: If True, use triplet loss training
    
    Returns:
        Tuple of (avg_loss, avg_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for batch_idx, batch_data in enumerate(pbar):
        if use_triplet:
            # Triplet: anchor, positive, negative
            anchor, positive, negative = batch_data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = model.get_embedding(anchor)
            positive_emb = model.get_embedding(positive)
            negative_emb = model.get_embedding(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Compute accuracy (positive distance < negative distance)
            with torch.no_grad():
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
                accuracy = (pos_dist < neg_dist).float().mean().item()
        else:
            # Pair: img1, img2, labels
            img1, img2, labels = batch_data
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            
            # Compute accuracy
            with torch.no_grad():
                accuracy = compute_accuracy(emb1, emb2, labels, threshold=0.5)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (helps with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        running_accuracy += accuracy
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_triplet: bool = False
) -> tuple:
    """
    Validate the model.
    
    Args:
        use_triplet: If True, use triplet loss validation
    
    Returns:
        Tuple of (avg_loss, avg_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{total_epochs} [Val]')
    
    with torch.no_grad():
        for batch_data in pbar:
            if use_triplet:
                # Triplet: anchor, positive, negative
                anchor, positive, negative = batch_data
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                # Forward pass
                anchor_emb = model.get_embedding(anchor)
                positive_emb = model.get_embedding(positive)
                negative_emb = model.get_embedding(negative)
                
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                
                # Compute accuracy
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
                accuracy = (pos_dist < neg_dist).float().mean().item()
            else:
                # Pair: img1, img2, labels
                img1, img2, labels = batch_data
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                
                # Forward pass
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                
                # Compute accuracy
                accuracy = compute_accuracy(emb1, emb2, labels, threshold=0.5)
            
            # Update metrics
            running_loss += loss.item()
            running_accuracy += accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def plot_training_history(history: dict, save_path: str):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    history: dict,
    save_path: str
):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def train(
    metadata_path: str,
    img_dir: str,
    output_dir: str = 'outputs',
    # Model hyperparameters
    embedding_dim: int = 256,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5,
    # Training hyperparameters
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    # Class imbalance handling
    loss_type: str = 'triplet',  # 'standard', 'weighted', 'focal', 'triplet'
    pos_weight: float = 10.0,
    focal_gamma: float = 2.0,
    triplet_margin: float = 1.0,
    samples_per_class: Optional[int] = None,
    # Other settings
    num_workers: int = 4,
    random_seed: int = 42,
    save_every: int = 5
):
    """
    Main training function.
    
    Args:
        metadata_path: Path to train-metadata.csv
        img_dir: Path to image directory
        output_dir: Directory to save outputs
        embedding_dim: Dimension of embedding space
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        dropout: Dropout rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        val_split: Validation split ratio
        loss_type: Type of loss ('standard', 'weighted', 'focal')
        pos_weight: Weight for positive pairs (weighted loss)
        focal_gamma: Gamma parameter (focal loss)
        num_workers: Number of data loading workers
        random_seed: Random seed
        save_every: Save checkpoint every N epochs
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Siamese Network for Melanoma Classification")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Backbone: {backbone}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Loss Type: {loss_type}")
    if loss_type == 'triplet':
        print(f"Triplet Margin: {triplet_margin}")
        print(f"Using Class-Balanced Triplet Sampling")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"{'='*70}\n")
    
    # Create data loaders
    print("Loading data...")
    use_triplet = (loss_type == 'triplet')
    train_loader, val_loader, _ = create_data_loaders(  # We don't use test set during training
        metadata_path=metadata_path,
        img_dir=img_dir,
        val_split=val_split,
        test_split=0.1,  # 10% for test set
        batch_size=batch_size,
        num_workers=num_workers,
        random_state=random_seed,
        verbose=True,
        use_triplet=use_triplet,
        samples_per_class=samples_per_class
    )
    
    # Create model
    print(f"\nCreating model...")
    model, _ = get_model(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function based on type
    print(f"\nCreating {loss_type} loss function...")
    if loss_type == 'standard':
        criterion = ContrastiveLoss(margin=1.0)
    elif loss_type == 'weighted':
        criterion = WeightedContrastiveLoss(margin=1.0, pos_weight=pos_weight)
        print(f"  Positive pair weight: {pos_weight}")
    elif loss_type == 'focal':
        criterion = FocalContrastiveLoss(margin=1.0, gamma=focal_gamma)
        print(f"  Focal gamma: {focal_gamma}")
    elif loss_type == 'triplet':
        criterion = TripletLoss(margin=triplet_margin)
        print(f"  Triplet margin: {triplet_margin}")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs, use_triplet
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, num_epochs, use_triplet
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint periodically
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, history, checkpoint_path)
        
        print(f"{'-'*70}\n")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Siamese Network for Melanoma Classification')
    
    # Data arguments
    parser.add_argument('--metadata_path', type=str, default='data/train-metadata.csv',
                        help='Path to train-metadata.csv')
    parser.add_argument('--img_dir', type=str, default='data/train-image',
                        help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet34', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1 for 80:10:10 split)')
    
    # Class imbalance arguments
    parser.add_argument('--loss_type', type=str, default='triplet',
                        choices=['standard', 'weighted', 'focal', 'triplet'],
                        help='Type of contrastive loss')
    parser.add_argument('--pos_weight', type=float, default=10.0,
                        help='Weight for positive pairs (weighted loss)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter (focal loss)')
    parser.add_argument('--triplet_margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Number of samples per class per epoch (for triplet loss)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Train
    train(**vars(args))