"""
Two-stage training script for Siamese Network + Binary Classifier on ISIC 2020.
Stage 1: Siamese network with triplet loss (180 epochs)
Stage 2: Binary classifier on frozen features (140 epochs)
"""

import os
import time
import matplotlib.pyplot as plt
import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_dataloaders
from modules import SiameseNetwork, BinaryClassifier


def process_batch(siamese, tripletloss, anchor, positive, negative, device):
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
    anchor_out, pos_out, neg_out = siamese(anchor, positive, negative)
    return tripletloss(anchor_out, pos_out, neg_out)


def generate_loss_plot(train_loss, val_loss, output_dir, filename, title):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def extract_features(siamese, loader, device):
    """Extract features from all images in a dataloader."""
    siamese.eval()
    all_features, all_labels = [], []
    
    with torch.no_grad():
        for anchor, _, _, label in loader:
            features = siamese.forward_once(anchor.to(device))
            all_features.append(features)
            all_labels.append(label.to(device))
    
    return all_features, all_labels


def train(
    metadata_path: str = 'data/train-metadata.csv',
    img_dir: str = 'data/train-image',
    output_dir: str = 'outputs',
    siamese_epochs: int = 180,
    classifier_epochs: int = 140,
    siamese_lr: float = 1e-4,
    classifier_lr: float = 5e-4,
    margin: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TWO-STAGE TRAINING: SIAMESE + BINARY CLASSIFIER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Stage 1 - Siamese: {siamese_epochs} epochs, LR={siamese_lr}")
    print(f"Stage 2 - Classifier: {classifier_epochs} epochs, LR={classifier_lr}")
    print("=" * 70)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # ========================================================================
    # STAGE 1: TRAIN SIAMESE NETWORK WITH TRIPLET LOSS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING SIAMESE NETWORK")
    print("=" * 70)
    
    siamese = SiameseNetwork().to(device)
    tripletloss = TripletMarginLoss(margin=margin)
    siamese_optimizer = Adam(siamese.parameters(), lr=siamese_lr)
    
    siamese_train_loss, siamese_val_loss = [], []
    best_siamese_val_loss = float('inf')
    siamese_start = time.time()
    
    for epoch in range(siamese_epochs):
        siamese.train()
        t_loss_total = []
        
        for anchor, positive, negative, _ in train_loader:
            siamese_optimizer.zero_grad()
            loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
            loss.backward()
            siamese_optimizer.step()
            t_loss_total.append(loss.item())
        
        siamese.eval()
        v_loss_total = []
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                loss = process_batch(siamese, tripletloss, anchor, positive, negative, device)
                v_loss_total.append(loss.item())
        
        avg_train_loss = sum(t_loss_total) / len(t_loss_total)
        avg_val_loss = sum(v_loss_total) / len(v_loss_total)
        siamese_train_loss.append(avg_train_loss)
        siamese_val_loss.append(avg_val_loss)
        
        if avg_val_loss < best_siamese_val_loss:
            best_siamese_val_loss = avg_val_loss
            torch.save(siamese.state_dict(), os.path.join(output_dir, 'best_siamese.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{siamese_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    siamese_time = (time.time() - siamese_start) / 60
    torch.save(siamese.state_dict(), os.path.join(output_dir, 'final_siamese.pth'))
    generate_loss_plot(siamese_train_loss, siamese_val_loss, output_dir, 
                       'siamese_loss.png', 'Siamese Network - Triplet Loss')
    
    print(f"\nStage 1 Complete: {siamese_time:.2f} min")
    print(f"Best Siamese Val Loss: {best_siamese_val_loss:.4f}")
    
    # ========================================================================
    # STAGE 2: TRAIN BINARY CLASSIFIER ON FROZEN SIAMESE FEATURES
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: EXTRACTING FEATURES & TRAINING CLASSIFIER")
    print("=" * 70)
    
    # Load best Siamese model
    siamese.load_state_dict(torch.load(os.path.join(output_dir, 'best_siamese.pth')))
    siamese.eval()
    
    # Extract features from all sets
    print("Extracting features...")
    train_features, train_labels = extract_features(siamese, train_loader, device)
    val_features, val_labels = extract_features(siamese, val_loader, device)
    test_features, test_labels = extract_features(siamese, test_loader, device)
    print(f"✓ Features extracted from {len(train_features)} train batches")
    
    # Train binary classifier
    classifier = BinaryClassifier().to(device)
    ce_loss = CrossEntropyLoss()
    classifier_optimizer = Adam(classifier.parameters(), lr=classifier_lr)
    
    classifier_train_loss, classifier_val_loss = [], []
    best_classifier_val_loss = float('inf')
    classifier_start = time.time()
    
    for epoch in range(classifier_epochs):
        classifier.train()
        t_loss_total = []
        
        for features, labels in zip(train_features, train_labels):
            classifier_optimizer.zero_grad()
            out = classifier(features)
            loss = ce_loss(out, labels)
            loss.backward()
            classifier_optimizer.step()
            t_loss_total.append(loss.item())
        
        classifier.eval()
        v_loss_total = []
        with torch.no_grad():
            for features, labels in zip(val_features, val_labels):
                out = classifier(features)
                loss = ce_loss(out, labels)
                v_loss_total.append(loss.item())
        
        avg_train_loss = sum(t_loss_total) / len(t_loss_total)
        avg_val_loss = sum(v_loss_total) / len(v_loss_total)
        classifier_train_loss.append(avg_train_loss)
        classifier_val_loss.append(avg_val_loss)
        
        if avg_val_loss < best_classifier_val_loss:
            best_classifier_val_loss = avg_val_loss
            torch.save(classifier.state_dict(), os.path.join(output_dir, 'best_classifier.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{classifier_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    classifier_time = (time.time() - classifier_start) / 60
    torch.save(classifier.state_dict(), os.path.join(output_dir, 'final_classifier.pth'))
    generate_loss_plot(classifier_train_loss, classifier_val_loss, output_dir, 
                       'classifier_loss.png', 'Binary Classifier - CrossEntropy Loss')
    
    print(f"\nStage 2 Complete: {classifier_time:.2f} min")
    print(f"Best Classifier Val Loss: {best_classifier_val_loss:.4f}")
    
    # ========================================================================
    # FINAL EVALUATION ON TEST SET
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    classifier.load_state_dict(torch.load(os.path.join(output_dir, 'best_classifier.pth')))
    classifier.eval()
    
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for features, labels in zip(test_features, test_labels):
            out = classifier(features)
            predicted = torch.argmax(out, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Final summary
    total_time = siamese_time + classifier_time
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Stage 1 - Siamese Network:")
    print(f"  Training Time: {siamese_time:.2f} min")
    print(f"  Best Val Loss: {best_siamese_val_loss:.4f}")
    print(f"\nStage 2 - Binary Classifier:")
    print(f"  Training Time: {classifier_time:.2f} min")
    print(f"  Best Val Loss: {best_classifier_val_loss:.4f}")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Sensitivity (Recall): {sensitivity:.2f}%")
    print(f"  Specificity: {specificity:.2f}%")
    print(f"\nTotal Training Time: {total_time:.2f} min")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)
    
    # Save summary to file
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Stage 1 - Siamese Network:\n")
        f.write(f"  Epochs: {siamese_epochs}\n")
        f.write(f"  Learning Rate: {siamese_lr}\n")
        f.write(f"  Training Time: {siamese_time:.2f} min\n")
        f.write(f"  Best Val Loss: {best_siamese_val_loss:.4f}\n")
        f.write(f"\nStage 2 - Binary Classifier:\n")
        f.write(f"  Epochs: {classifier_epochs}\n")
        f.write(f"  Learning Rate: {classifier_lr}\n")
        f.write(f"  Training Time: {classifier_time:.2f} min\n")
        f.write(f"  Best Val Loss: {best_classifier_val_loss:.4f}\n")
        f.write(f"\nFinal Test Results:\n")
        f.write(f"  Accuracy: {accuracy:.2f}%\n")
        f.write(f"  Sensitivity: {sensitivity:.2f}%\n")
        f.write(f"  Specificity: {specificity:.2f}%\n")
        f.write(f"  Confusion Matrix:\n")
        f.write(f"    TN={tn}, FP={fp}\n")
        f.write(f"    FN={fn}, TP={tp}\n")
        f.write(f"\nTotal Training Time: {total_time:.2f} min\n")
        f.write("=" * 70 + "\n")


if __name__ == "__main__":
    train(
        metadata_path='data/train-metadata.csv',
        img_dir='data/train-image',
        output_dir='outputs',
        siamese_epochs=180,
        classifier_epochs=140,
        siamese_lr=1e-4,
        classifier_lr=5e-4,
        margin=1.0,
        batch_size=32,
        num_workers=4
    )
