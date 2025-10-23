"""
Prediction and evaluation script for Siamese Network on ISIC 2020 dataset.
Loads trained model, extracts embeddings, and computes classification metrics.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier

from dataset import load_and_split_data, get_transforms
from modules import SiameseNetwork
from PIL import Image
import pandas as pd


class SingleImageDataset(Dataset):
    """
    Dataset for extracting embeddings from individual images.
    Unlike the pair/triplet datasets, this returns single images.
    """
    
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        """
        Args:
            df: DataFrame with image names and labels
            img_dir: Directory containing images
            transform: Optional transform to be applied on images
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Ensure required columns exist
        if 'isic_id' in self.df.columns and 'image_name' not in self.df.columns:
            self.df['image_name'] = self.df['isic_id']
        
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            raise ValueError("DataFrame must contain 'image_name' and 'target' columns")
    
    def _load_image(self, idx: int) -> Image.Image:
        """Load image by dataframe index."""
        img_name = self.df.iloc[idx]['image_name']
        
        # Handle different file extensions
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_name = f"{img_name}.jpg"
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, 'image', img_name),
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            img_path = possible_paths[0]
        
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single image and its label.
        
        Returns:
            Tuple of (image, label, image_name)
        """
        img = self._load_image(idx)
        label = self.df.iloc[idx]['target']
        img_name = self.df.iloc[idx]['image_name']
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_name


def load_model(
    checkpoint_path: str,
    embedding_dim: int = 256,
    backbone: str = 'resnet50',
    dropout: float = 0.5,
    device: torch.device = None
) -> SiameseNetwork:
    """
    Load trained Siamese network from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        embedding_dim: Embedding dimension
        backbone: Backbone architecture
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model architecture
    model = SiameseNetwork(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=False,  # We'll load trained weights
        dropout=dropout
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded successfully\n")
    
    return model


def extract_embeddings(
    model: SiameseNetwork,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract embeddings for all images in the dataset.
    
    Args:
        model: Trained Siamese network
        dataloader: DataLoader for images
        device: Device to run inference on
    
    Returns:
        Tuple of (embeddings, labels, image_names)
        - embeddings: [N, embedding_dim] array
        - labels: [N] array of ground truth labels
        - image_names: List of image names
    """
    embeddings_list = []
    labels_list = []
    names_list = []
    
    print("Extracting embeddings...")
    model.eval()
    
    with torch.no_grad():
        for images, labels, names in tqdm(dataloader, desc="Processing images"):
            images = images.to(device)
            
            # Extract embeddings
            emb = model.get_embedding(images)
            
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(labels.numpy())
            names_list.extend(names)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    print(f"✓ Extracted embeddings for {len(embeddings)} images")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Label distribution: {np.bincount(labels.astype(int))}\n")
    
    return embeddings, labels, names_list


def classify_with_centroids(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify images using class centroid method.
    
    Computes mean embedding (centroid) for each class from training data,
    then classifies test samples based on nearest centroid.
    
    Args:
        train_embeddings: [N_train, embedding_dim] array of training embeddings
        train_labels: [N_train] array of training labels
        test_embeddings: [N_test, embedding_dim] array of test embeddings
    
    Returns:
        Tuple of (predictions, scores)
        - predictions: [N_test] array of predicted labels
        - scores: [N_test] array of scores (distance to melanoma centroid, negated for ROC)
    """
    print("Computing class centroids from training data...")
    
    # Get unique classes
    classes = np.unique(train_labels)
    
    # Compute centroids for each class from training data
    centroids = {}
    for cls in classes:
        cls_mask = (train_labels == cls)
        cls_embeddings = train_embeddings[cls_mask]
        centroids[cls] = np.mean(cls_embeddings, axis=0)
        print(f"  Class {cls}: centroid computed from {cls_mask.sum()} training samples")
    
    # Classify test samples based on nearest centroid
    print("\nClassifying test samples based on nearest centroid...")
    predictions = []
    scores = []
    
    for emb in test_embeddings:
        # Compute distance to each centroid
        distances = {}
        for cls, centroid in centroids.items():
            dist = np.linalg.norm(emb - centroid)
            distances[cls] = dist
        
        # Predict class with minimum distance
        pred_class = min(distances, key=distances.get)
        predictions.append(pred_class)
        
        # For ROC curve, use negative distance to positive class (melanoma=1)
        # Negative because smaller distance = higher confidence
        if 1 in distances:
            score = -distances[1]  # Negate so higher score = more likely melanoma
        else:
            score = distances[0]  # If only one class, use distance to it
        scores.append(score)
    
    predictions = np.array(predictions)
    scores = np.array(scores)
    
    print("✓ Classification complete\n")
    
    return predictions, scores


def classify_with_knn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify images using k-Nearest Neighbors on embedding space.
    
    Args:
        train_embeddings: [N_train, embedding_dim] array of training embeddings
        train_labels: [N_train] array of training labels
        test_embeddings: [N_test, embedding_dim] array of test embeddings
        k: Number of neighbors to consider
    
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: [N_test] array of predicted labels
        - probabilities: [N_test] array of probability scores for positive class
    """
    print(f"Training k-NN classifier (k={k}) on training data...")
    
    # Train kNN classifier on training data
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(train_embeddings, train_labels)
    
    # Predict on test data
    print("Predicting test samples with k-NN...")
    predictions = knn.predict(test_embeddings)
    probabilities = knn.predict_proba(test_embeddings)
    
    # Get probability for positive class (melanoma=1)
    if len(knn.classes_) == 2:
        pos_class_idx = np.where(knn.classes_ == 1)[0][0]
        prob_scores = probabilities[:, pos_class_idx]
    else:
        prob_scores = probabilities[:, 0]
    
    print("✓ k-NN classification complete\n")
    
    return predictions, prob_scores


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores (for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    print("Computing metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)  # Same as sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix for specificity
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity = TN / (TN + FP)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'sensitivity': float(recall),  # Sensitivity = Recall
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    }
    
    print("✓ Metrics computed\n")
    
    return metrics, cm


def print_metrics_report(metrics: Dict[str, float], method: str):
    """
    Print formatted metrics report.
    
    Args:
        metrics: Dictionary of metric values
        method: Classification method name
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION METRICS ({method})")
    print("=" * 70)
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print("=" * 70 + "\n")


def predict(
    metadata_path: str,
    img_dir: str,
    checkpoint_path: str = 'outputs/best_model.pth',
    output_dir: str = 'outputs',
    # Model hyperparameters
    embedding_dim: int = 256,
    backbone: str = 'resnet50',
    dropout: float = 0.5,
    # Data parameters
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
    # Classification parameters
    method: str = 'centroid',  # 'centroid' or 'knn'
    knn_k: int = 5,
    # Evaluation set selection
    eval_on_test: bool = False  # If True, evaluate on test set; if False, on validation set
):
    """
    Main prediction and evaluation function.
    
    Args:
        metadata_path: Path to train-metadata.csv
        img_dir: Path to image directory
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save outputs
        embedding_dim: Embedding dimension
        backbone: Backbone architecture
        dropout: Dropout rate
        val_split: Validation split ratio
        test_split: Test split ratio
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        random_seed: Random seed
        method: Classification method ('centroid' or 'knn')
        knn_k: Number of neighbors for k-NN
        eval_on_test: If True, evaluate on test set; if False, on validation set
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    eval_set_name = "test" if eval_on_test else "validation"
    
    print("\n" + "=" * 70)
    print("SIAMESE NETWORK EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Classification method: {method}")
    print(f"Evaluation set: {eval_set_name}")
    print("=" * 70 + "\n")
    
    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        embedding_dim=embedding_dim,
        backbone=backbone,
        dropout=dropout,
        device=device
    )
    
    # Load and split data
    print("Loading data...")
    train_df, val_df, test_df = load_and_split_data(
        metadata_path=metadata_path,
        val_split=val_split,
        test_split=test_split,
        random_state=random_seed,
        stratify=True
    )
    
    # Choose evaluation set
    eval_df = test_df if eval_on_test else val_df
    
    # Create dataset for training set (to compute centroids/train kNN)
    print("\nCreating training dataset...")
    train_dataset = SingleImageDataset(
        df=train_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=224)  # No augmentation for embedding extraction
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Training dataset ready: {len(train_dataset)} images")
    
    # Create dataset for evaluation set
    print(f"\nCreating {eval_set_name} dataset...")
    eval_dataset = SingleImageDataset(
        df=eval_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=224)
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ {eval_set_name.capitalize()} dataset ready: {len(eval_dataset)} images\n")
    
    # Extract embeddings from training set
    train_embeddings, train_labels, _ = extract_embeddings(
        model=model,
        dataloader=train_loader,
        device=device
    )
    
    # Extract embeddings from evaluation set
    eval_embeddings, eval_labels, image_names = extract_embeddings(
        model=model,
        dataloader=eval_loader,
        device=device
    )
    
    # Classify evaluation data using selected method trained on training data
    if method == 'centroid':
        predictions, scores = classify_with_centroids(
            train_embeddings, train_labels, eval_embeddings
        )
    elif method == 'knn':
        predictions, scores = classify_with_knn(
            train_embeddings, train_labels, eval_embeddings, k=knn_k
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'centroid' or 'knn'")
    
    # Compute metrics on evaluation predictions
    metrics, cm = compute_metrics(eval_labels, predictions, scores)
    
    # Print report
    method_name = f"Class Centroids" if method == 'centroid' else f"k-NN (k={knn_k})"
    print_metrics_report(metrics, method_name)
    
    # Save metrics to JSON
    os.makedirs(output_dir, exist_ok=True)
    metrics_filename = f'evaluation_metrics_{method}_{eval_set_name}.json'
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    results = {
        'method': method,
        'evaluation_set': eval_set_name,
        'metrics': metrics,
        'num_train_samples': int(len(train_labels)),
        'num_eval_samples': int(len(eval_labels)),
        'train_class_distribution': {
            'benign': int(np.sum(train_labels == 0)),
            'melanoma': int(np.sum(train_labels == 1))
        },
        'eval_class_distribution': {
            'benign': int(np.sum(eval_labels == 0)),
            'melanoma': int(np.sum(eval_labels == 1))
        },
        'confusion_matrix': cm.tolist(),
        'model_config': {
            'embedding_dim': embedding_dim,
            'backbone': backbone,
            'dropout': dropout
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save predictions
    predictions_filename = f'predictions_{method}_{eval_set_name}.csv'
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df = pd.DataFrame({
        'image_name': image_names,
        'true_label': eval_labels,
        'predicted_label': predictions,
        'score': scores
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70 + "\n")
    
    return metrics, predictions, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate trained Siamese Network on validation set'
    )
    
    # Data arguments
    parser.add_argument('--metadata_path', type=str, default='data/train-metadata.csv',
                        help='Path to train-metadata.csv')
    parser.add_argument('--img_dir', type=str, default='data/train-image',
                        help='Path to image directory')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone architecture')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1 for 80:10:10 split)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio (default: 0.1 for 80:10:10 split)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    
    # Classification arguments
    parser.add_argument('--method', type=str, default='centroid',
                        choices=['centroid', 'knn'],
                        help='Classification method')
    parser.add_argument('--knn_k', type=int, default=5,
                        help='Number of neighbors for k-NN')
    
    # Evaluation set selection
    parser.add_argument('--eval_on_test', action='store_true',
                        help='Evaluate on test set instead of validation set')
    
    args = parser.parse_args()
    
    # Run prediction and evaluation
    predict(**vars(args))

