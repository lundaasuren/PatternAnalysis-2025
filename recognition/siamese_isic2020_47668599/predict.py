"""
Inference script for Siamese Network melanoma classification.
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from modules import SiameseNetwork, BinaryClassifier
from dataset import get_dataloaders


def predict(
    siamese_path: str,
    classifier_path: str,
    metadata_path: str = 'data/train-metadata.csv',
    img_dir: str = 'data/train-image',
    batch_size: int = 32,
    num_workers: int = 4
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
    siamese.load_state_dict(torch.load(siamese_path, weights_only=True, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, weights_only=True, map_location=device))
    siamese.eval()
    classifier.eval()
    
    sample_features, sample_labels = [], []
    with torch.no_grad():
        for anchor, _, _, label in test_loader:
            sample_features.append(siamese.forward_once(anchor.to(device)))
            sample_labels.append(label.to(device))
    
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for features, labels in zip(sample_features, sample_labels):
            predicted = torch.argmax(classifier(features), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu())
            all_predictions.extend(predicted.cpu())
    
    accuracy = 100 * correct / total
    
    cm_display = ConfusionMatrixDisplay.from_predictions(
        all_labels, 
        all_predictions, 
        display_labels=["Benign", "Malignant"]
    )
    cm_display.plot(colorbar=False)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy, all_labels, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Siamese Network Inference")
    parser.add_argument("siamese", help="Path to siamese network weights")
    parser.add_argument("classifier", help="Path to classifier weights")
    parser.add_argument("--metadata", default="data/train-metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--img_dir", default="data/train-image", help="Path to image directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    predict(args.siamese, args.classifier, args.metadata, args.img_dir, args.batch_size, args.num_workers)
