"""
Inference script for Siamese Network + Binary Classifier melanoma classification.
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    siamese = SiameseNetwork().to(device)
    classifier = BinaryClassifier().to(device)
    
    try:
        siamese.load_state_dict(torch.load(siamese_path, weights_only=True, map_location=device))
        classifier.load_state_dict(torch.load(classifier_path, weights_only=True, map_location=device))
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None
    
    siamese.eval()
    classifier.eval()
    
    sample_features, sample_labels = [], []
    with torch.no_grad():
        for anchor, _, _, label in test_loader:
            features = siamese.forward_once(anchor.to(device))
            sample_features.append(features)
            sample_labels.append(label.to(device))
    
    correct, total = 0, 0
    all_labels, all_predictions, all_probabilities = [], [], []
    
    with torch.no_grad():
        for features, labels in zip(sample_features, sample_labels):
            out = classifier(features)
            predicted = torch.argmax(out, dim=1)
            probabilities = torch.softmax(out, dim=1)[:, 1]
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    tn = np.sum((all_labels == 0) & (all_predictions == 0))
    fp = np.sum((all_labels == 0) & (all_predictions == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    cm_display = ConfusionMatrixDisplay.from_predictions(
        all_labels, 
        all_predictions,
        labels=[0, 1],
        display_labels=["Benign", "Malignant"],
        cmap='Blues'
    )
    
    fig = cm_display.figure_
    fig.set_size_inches(8, 6)
    cm_display.ax_.set_title(
        f'Confusion Matrix\nAccuracy: {accuracy:.2f}% | Sensitivity: {recall*100:.2f}% | Specificity: {specificity*100:.2f}%',
        fontsize=12,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    results_path = os.path.join(output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("TEST SET RESULTS\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct Predictions: {correct}\n\n")
        f.write(f"Accuracy:    {accuracy:.2f}%\n")
        f.write(f"Precision:   {precision:.4f}\n")
        f.write(f"Sensitivity: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(
            all_labels, 
            all_predictions, 
            target_names=['Benign', 'Malignant'],
            digits=4,
            zero_division=0
        ))
    
    return accuracy, all_labels, all_predictions


if __name__ == "__main__":
    predict(
        siamese_path="outputs/best_siamese.pth",
        classifier_path="outputs/best_classifier.pth",
        metadata_path="data/train-metadata.csv",
        img_dir="data/train-image",
        batch_size=32,
        num_workers=4,
        output_dir="outputs"
    )