"""
Inference script for Siamese Network + Binary Classifier melanoma classification.
Evaluates on test set and reports comprehensive metrics.
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)
import numpy as np
import torch
from modules import SiameseNetwork, BinaryClassifier
from dataset import get_dataloaders


def predict(
    siamese_path: str,
    classifier_path: str,
    metadata_path: str = 'data/train-metadata.csv',
    img_dir: str = 'data/train-image',
    batch_size: int = 32,
    num_workers: int = 4,
    output_dir: str = 'outputs'
):
    """
    Perform inference using trained Siamese network + Binary classifier.
    
    Args:
        siamese_path: Path to trained Siamese network weights
        classifier_path: Path to trained classifier weights
        metadata_path: Path to metadata CSV
        img_dir: Path to image directory
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        output_dir: Directory to save results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("SIAMESE NETWORK + BINARY CLASSIFIER INFERENCE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Siamese weights: {siamese_path}")
    print(f"Classifier weights: {classifier_path}")
    print("=" * 70)
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    print(f"✓ Test set: {len(test_loader.dataset)} images")
    
    # Load models
    print("\nLoading models...")
    siamese = SiameseNetwork().to(device)
    classifier = BinaryClassifier().to(device)
    
    try:
        siamese.load_state_dict(torch.load(siamese_path, weights_only=True, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, weights_only=True, map_location=device))
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None
    
    siamese.eval()
    classifier.eval()
    
    # Extract features from Siamese network
    print("\nExtracting features from Siamese network...")
    sample_features, sample_labels = [], []
    
    with torch.no_grad():
        for anchor, _, _, label in test_loader:
            features = siamese.forward_once(anchor.to(device))
            sample_features.append(features)
            sample_labels.append(label.to(device))
    
    print(f"✓ Features extracted from {len(sample_features)} batches")
    
    # Classify with binary classifier
    print("\nClassifying with binary classifier...")
    correct, total = 0, 0
    all_labels, all_predictions, all_probabilities = [], [], []
    
    with torch.no_grad():
        for features, labels in zip(sample_features, sample_labels):
            out = classifier(features)
            predicted = torch.argmax(out, dim=1)
            probabilities = torch.softmax(out, dim=1)[:, 1]  # Probability of positive class
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)  # Sensitivity
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Calculate specificity
    tn = np.sum((all_labels == 0) & (all_predictions == 0))
    fp = np.sum((all_labels == 0) & (all_predictions == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.2f}%")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Sensitivity: {recall:.4f} (Recall)")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print("=" * 70)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=['Benign', 'Malignant'],
        digits=4
    ))
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm_display = ConfusionMatrixDisplay.from_predictions(
        all_labels, 
        all_predictions, 
        display_labels=["Benign", "Malignant"],
        cmap='Blues'
    )
    
    # Customize plot
    fig = cm_display.figure_
    fig.set_size_inches(8, 6)
    cm_display.ax_.set_title(
        f'Confusion Matrix\nAccuracy: {accuracy:.2f}% | Sensitivity: {recall*100:.2f}% | Specificity: {specificity*100:.2f}%',
        fontsize=12,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    # Save confusion matrix
    import os
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()
    
    # Save results to file
    results_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"  Accuracy:    {accuracy:.2f}%\n")
        f.write(f"  Precision:   {precision:.4f}\n")
        f.write(f"  Sensitivity: {recall:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  F1-Score:    {f1:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(
            all_labels, 
            all_predictions, 
            target_names=['Benign', 'Malignant'],
            digits=4
        ))
    
    print(f"✓ Results saved to {results_path}")
    
    return accuracy, all_labels, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with Siamese Network + Binary Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "siamese", 
        help="Path to Siamese network weights (e.g., outputs/best_siamese.pth)"
    )
    parser.add_argument(
        "classifier", 
        help="Path to binary classifier weights (e.g., outputs/best_classifier.pth)"
    )
    parser.add_argument(
        "--metadata", 
        default="data/train-metadata.csv", 
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--img_dir", 
        default="data/train-image", 
        help="Path to image directory"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict(
        siamese_path=args.siamese,
        classifier_path=args.classifier,
        metadata_path=args.metadata,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir
    )