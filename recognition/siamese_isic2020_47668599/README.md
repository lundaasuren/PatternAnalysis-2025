# Siamese Network for Melanoma Classification

## Project Overview

This project implements a **Siamese Neural Network** to classify dermoscopic images from the ISIC 2020 Challenge dataset as either normal (benign) or melanoma (malignant). Siamese networks are particularly well-suited for this task as they learn to differentiate between classes by comparing pairs of images and learning a similarity metric, which is valuable in medical imaging where we want the model to learn discriminative features between subtle visual differences.

## Problem Statement

Melanoma is one of the most dangerous forms of skin cancer, and early detection is critical for successful treatment. The ISIC 2020 Kaggle Challenge provides a large-scale dataset of dermoscopic images for binary classification (normal vs melanoma). The challenge lies in:

- **Class imbalance**: The dataset contains significantly more benign samples than melanoma cases
- **Visual similarity**: Many benign lesions share visual characteristics with melanoma
- **Subtle features**: Distinguishing features can be subtle and require learning robust representations

**Goal**: Achieve a minimum accuracy of 0.8 on the test set using a Siamese network architecture.

## Dataset

- **Source**: [ISIC 2020 JPG 224x224 Resized](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-224x224-resized)
- **Format**: Pre-resized 224x224 RGB images
- **Classes**: Binary classification (0 = benign, 1 = melanoma)
- **Size**: ~33,000 training images with significant class imbalance

## How Siamese Networks Work

A Siamese network consists of two identical neural networks (sharing the same weights) that process pairs of images simultaneously. The network learns to:

1. **Extract features** from both images using a shared CNN backbone
2. **Compute embeddings** in a latent space where similar images are close together
3. **Calculate similarity** using distance metrics (e.g., Euclidean distance, cosine similarity)
4. **Classify** based on whether the pair belongs to the same class or different classes

During inference, we can compare a query image against reference examples from each class and classify based on similarity scores.

## Initial Implementation Plan

### 1. **Data Preprocessing & Augmentation** (`dataset.py`)
   - Load resized 224x224 images from the Kaggle dataset
   - Implement pair generation strategy:
     - Positive pairs: Same class (both benign or both melanoma)
     - Negative pairs: Different classes (one benign, one melanoma)
   - Handle class imbalance through strategic pair sampling
   - Apply data augmentation: rotation, flipping, color jittering, normalization

### 2. **Model Architecture** (`modules.py`)
   - **Backbone Network**: 
     - Start with ResNet-50 or EfficientNet-B0 (pre-trained on ImageNet)
     - Consider MobileNetV2 for efficiency
   - **Embedding Network**: 
     - Fully connected layers to project features to embedding space (128-512 dimensions)
     - Batch normalization and dropout for regularization
   - **Similarity/Distance Module**:
     - Euclidean distance or cosine similarity computation
     - Contrastive loss or triplet loss function

### 3. **Training Pipeline** (`train.py`)
   - Implement contrastive loss: `L = (1-Y) * D² + Y * max(margin - D, 0)²`
     - Where Y=1 for dissimilar pairs, Y=0 for similar pairs
     - D is the distance between embeddings
   - Alternative: Triplet loss with online hard negative mining
   - Training strategy:
     - Batch size: 32-64 pairs
     - Optimizer: Adam with learning rate scheduling
     - Monitor both loss and accuracy metrics
   - Data split: 80/10/10 train/validation/test
   - Plot training/validation loss and accuracy curves

### 4. **Evaluation & Prediction** (`predict.py`)
   - For classification, compare query image against:
     - Option A: k-nearest neighbors in embedding space
     - Option B: Reference set of prototypes from each class
   - Calculate metrics:
     - Accuracy, Precision, Recall, F1-score
     - ROC-AUC (important for imbalanced medical data)
     - Confusion matrix
   - Visualize:
     - Embedding space using t-SNE or UMAP
     - Example pairs with similarity scores
     - Misclassified examples for error analysis

### 5. **Expected Challenges**
   - **Class imbalance**: Use weighted sampling or focal loss
   - **Pair generation**: Ensure balanced positive/negative pairs
   - **Computational cost**: Siamese networks require processing pairs, doubling batch processing
   - **Hyperparameter tuning**: Margin for contrastive loss, embedding dimension, learning rate

### 6. **Success Metrics**
   - Primary: Test accuracy ≥ 0.8
   - Secondary: High recall for melanoma class (minimize false negatives)
   - Qualitative: Embeddings should show clear class separation

## Dependencies (Preliminary)

```
- Python 3.8+
- PyTorch 2.0+ / TensorFlow 2.x
- torchvision / tensorflow-datasets
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pillow
- albumentations (for advanced augmentation)
```

## Repository Structure

```
siamese_melanoma_classifier_[YOUR_ID]/
├── modules.py          # Siamese network architecture
├── dataset.py          # Data loading and pair generation
├── train.py           # Training script with validation
├── predict.py         # Inference and evaluation
├── utils.py           # Helper functions (optional)
├── README.md          # This file
└── results/           # Saved models, plots, and outputs
```

## References

[11] Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML deep learning workshop*, Vol. 2.