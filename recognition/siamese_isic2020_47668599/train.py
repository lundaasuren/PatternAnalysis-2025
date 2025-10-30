"""
Training script for Siamese Network on ISIC 2020 melanoma classification.
"""

import os
import time
import matplotlib.pyplot as plt
import torch
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from dataset import get_dataloaders
from modules import SiameseNetwork


def process_batch(siamese, tripletloss, anchor, positive, negative, device):
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
    anchor_out, pos_out, neg_out = siamese(anchor, positive, negative)
    return tripletloss(anchor_out, pos_out, neg_out)


def generate_loss_plot(train_loss, val_loss, output_dir='outputs'):
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
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    siamese = SiameseNetwork().to(device)
    tripletloss = TripletMarginLoss(margin=margin)
    optimizer = Adam(siamese.parameters(), lr=learning_rate)
    
    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        siamese.train()
        t_loss_total = []
        
        for anchor, positive, negative, _ in train_loader:
            optimizer.zero_grad()
            loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
            loss.backward()
            optimizer.step()
            t_loss_total.append(loss.item())
        
        siamese.eval()
        v_loss_total = []
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
                v_loss_total.append(loss.item())
        
        avg_train_loss = sum(t_loss_total) / len(t_loss_total)
        avg_val_loss = sum(v_loss_total) / len(v_loss_total)
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(siamese.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    training_time = (time.time() - start_time) / 60
    
    siamese.eval()
    test_loss_total = []
    with torch.no_grad():
        for anchor, positive, negative, _ in test_loader:
            loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
            test_loss_total.append(loss.item())
    
    avg_test_loss = sum(test_loss_total) / len(test_loss_total)
    
    torch.save(siamese.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    generate_loss_plot(train_loss, val_loss, output_dir)
    
    print(f"\nTraining Complete:")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Time: {training_time:.2f} min")


if __name__ == "__main__":
    train(
        metadata_path='data/train-metadata.csv',
        img_dir='data/train-image',
        output_dir='outputs',
        num_epochs=100,
        learning_rate=1e-3,
        margin=1.0,
        batch_size=32,
        num_workers=4
    )
