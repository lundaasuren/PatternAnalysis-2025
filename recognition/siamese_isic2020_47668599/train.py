"""
Simplified training script for Siamese Network on ISIC 2020 dataset.
Trains for 100 epochs with fixed learning rate.
"""

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch.nn import TripletMarginLoss
from torch.optim import Adam

from dataset import get_dataloaders
from modules import SiameseNetwork


def process_batch(siamese, tripletloss, anchor, positive, negative, device):
    """
    Process a batch of triplets and return loss.
    
    Args:
        siamese: Siamese network model
        tripletloss: Triplet loss function
        anchor: Anchor images
        positive: Positive images
        negative: Negative images
        device: Device to run on
    
    Returns:
        Loss tensor
    """
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)
    
    anchor_result, positive_result, negative_result = siamese(anchor, positive, negative)
    return tripletloss(anchor_result, positive_result, negative_result)


def generate_loss_plot(train_loss, val_loss, output_dir='outputs'):
    """
    Plot training and validation loss.
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")


def train(
    metadata_path: str = 'data/train-metadata.csv',
    img_dir: str = 'data/train-image',
    output_dir: str = 'outputs',
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    margin: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4
):
    """
    Train Siamese Network.
    
    Args:
        metadata_path: Path to metadata CSV
        img_dir: Path to image directory
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        margin: Margin for triplet loss
        batch_size: Batch size
        num_workers: Number of data loading workers
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Training Siamese Network")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Triplet Margin: {margin}")
    print(f"Batch Size: {batch_size}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    siamese = SiameseNetwork().to(device)
    tripletloss = TripletMarginLoss(margin=margin)
    optimizer = Adam(siamese.parameters(), lr=learning_rate)
    
    total_params = sum(p.numel() for p in siamese.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training loop
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        siamese.train()
        t_loss_total = []
        
        for i, (anchor, positive, negative, label) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
            loss.backward()
            optimizer.step()
            
            t_loss_total.append(loss.item())
            
            if i % (len(train_loader) // 4) == 0 and i != len(train_loader) - 1:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.4f}")
        
        # Validation
        siamese.eval()
        v_loss_total = []
        
        with torch.no_grad():
            for i, (anchor, positive, negative, label) in enumerate(val_loader):
                loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
                v_loss_total.append(loss.item())
        
        # Record losses
        avg_train_loss = sum(t_loss_total) / len(t_loss_total)
        avg_val_loss = sum(v_loss_total) / len(v_loss_total)
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(siamese.state_dict(), save_path)
            print(f"  ✓ New best model saved! (Val Loss: {avg_val_loss:.4f})")
        
        print("-" * 70)
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print(f"\nTraining complete! It took {training_time:.2f} minutes")
    
    # Testing
    print("\nTesting the model...")
    siamese.eval()
    test_loss_total = []
    
    with torch.no_grad():
        for i, (anchor, positive, negative, label) in enumerate(test_loader):
            loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
            test_loss_total.append(loss.item())
            
            if i % (len(test_loader) // 2) == 0 and i != len(test_loader) - 1:
                print(f"Testing: Batch: {i}, Loss: {loss.item():.4f}")
    
    avg_test_loss = sum(test_loss_total) / len(test_loss_total)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(siamese.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Generate loss plot
    generate_loss_plot(train_loss, val_loss, output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    print(f"Training Time: {training_time:.2f} minutes")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    METADATA_PATH = 'data/train-metadata.csv'
    IMG_DIR = 'data/train-image'
    OUTPUT_DIR = 'outputs'
    
    # Training parameters
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    MARGIN = 1.0
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Run training
    train(
        metadata_path=METADATA_PATH,
        img_dir=IMG_DIR,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        margin=MARGIN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
