"""
Simplified dataset loader for Siamese Network training on ISIC 2020 dataset.
Generates triplets of images for contrastive learning with balanced sampling.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import random
from typing import Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split


class ISICTripletDataset(Dataset):
    """
    Dataset class for generating image triplets for Siamese network training.
    
    Triplets consist of:
    - Anchor: Reference image
    - Positive: Same class as anchor
    - Negative: Different class from anchor
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        train: bool = True
    ):
        """
        Args:
            df: DataFrame with image names and labels
            img_dir: Directory containing images
            transform: Optional transform to be applied on images
            train: If True, generate triplets dynamically
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        
        # Handle different column naming conventions
        if 'isic_id' in self.df.columns and 'image_name' not in self.df.columns:
            self.df['image_name'] = self.df['isic_id']
        
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            raise ValueError("DataFrame must contain 'image_name' (or 'isic_id') and 'target' columns")
        
        # Create label-to-indices mapping
        self.label_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            label = row['target']
            self.label_to_indices[label].append(idx)
        
        self.labels = sorted(self.label_to_indices.keys())
        
        print(f"Dataset loaded: {len(self.df)} images")
        print(f"Class distribution:")
        for label in self.labels:
            count = len(self.label_to_indices[label])
            print(f"  Class {label}: {count} images ({count/len(self.df)*100:.2f}%)")
    
    def _get_random_triplet(self) -> Tuple[int, int, int]:
        """
        Generate a random triplet.
        
        Returns:
            Tuple of (anchor_idx, positive_idx, negative_idx)
        """
        # Randomly select anchor class
        anchor_label = random.choice(self.labels)
        anchor_indices = self.label_to_indices[anchor_label]
        
        if len(anchor_indices) < 2:
            return self._get_random_triplet()
        
        # Select anchor and positive from same class
        anchor_idx, positive_idx = random.sample(anchor_indices, 2)
        
        # Select negative from different class
        negative_labels = [l for l in self.labels if l != anchor_label]
        if not negative_labels:
            raise ValueError("Need at least 2 classes for triplet loss")
        
        negative_label = random.choice(negative_labels)
        negative_idx = random.choice(self.label_to_indices[negative_label])
        
        return anchor_idx, positive_idx, negative_idx
    
    def _load_image(self, idx: int) -> Image.Image:
        """
        Load image by dataframe index.
        
        Args:
            idx: Index in the dataframe
        
        Returns:
            PIL Image
        """
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
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet of images plus a dummy label.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (anchor, positive, negative, label)
        """
        if self.train:
            # Generate random triplet
            anchor_idx, positive_idx, negative_idx = self._get_random_triplet()
        else:
            # For validation/test, use deterministic triplets based on idx
            anchor_idx = idx
            anchor_label = self.df.iloc[anchor_idx]['target']
            
            # Get positive from same class
            positive_candidates = [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
            if positive_candidates:
                positive_idx = random.choice(positive_candidates)
            else:
                positive_idx = anchor_idx
            
            # Get negative from different class
            negative_labels = [l for l in self.labels if l != anchor_label]
            negative_label = random.choice(negative_labels)
            negative_idx = random.choice(self.label_to_indices[negative_label])
        
        # Load images
        anchor = self._load_image(anchor_idx)
        positive = self._load_image(positive_idx)
        negative = self._load_image(negative_idx)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        # Add dummy label for compatibility with train.py
        label = torch.tensor(0)
        
        return anchor, positive, negative, label


def load_and_split_data(
    metadata_path: str,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load metadata and split into 70:15:15 train/val/test sets.
    
    Args:
        metadata_path: Path to train-metadata.csv
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Handle different column naming conventions
    if 'isic_id' in df.columns and 'image_name' not in df.columns:
        df['image_name'] = df['isic_id']
    
    print(f"Loaded meta {len(df)} samples")
    print(f"\nOriginal class distribution:")
    print(df['target'].value_counts())
    print(f"\nClass percentages:")
    print(df['target'].value_counts(normalize=True) * 100)
    
    # 70:15:15 split
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df['target']
    )
    
    # Second split: 50% of temp for val, 50% for test (15% each of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df['target']
    )
    
    print(f"\n{'='*70}")
    print("70:15:15 TRAIN/VAL/TEST SPLIT")
    print(f"{'='*70}")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\nTrain class distribution:")
    print(train_df['target'].value_counts())
    print(f"\nValidation class distribution:")
    print(val_df['target'].value_counts())
    print(f"\nTest class distribution:")
    print(test_df['target'].value_counts())
    
    return train_df, val_df, test_df


def get_transforms(train: bool = True, img_size: int = 224):
    """
    Get image transforms with intensity scaling to [0, 1].
    
    Args:
        train: If True, include data augmentation
        img_size: Target image size
    
    Returns:
        Composed transforms
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),  # Automatically scales to [0, 1]
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Automatically scales to [0, 1]
        ])
    
    return transform


def create_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler that undersamples the majority class.
    Sets weights to achieve balanced sampling without over-representing minority class.
    
    Args:
        df: DataFrame with 'target' column
    
    Returns:
        WeightedRandomSampler instance
    """
    # Calculate class counts
    class_counts = df['target'].value_counts().to_dict()
    total_samples = len(df)
    
    # Identify minority class (smallest count)
    minority_count = min(class_counts.values())
    
    # Set weights: minority class = 1.0, majority classes proportionally reduced
    # This ensures majority class is sampled less frequently (undersampling)
    class_weights = {}
    for label, count in class_counts.items():
        # Weight = minority_count / class_count
        # This gives weight 1.0 to minority, <1.0 to majority
        class_weights[label] = minority_count / count
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in df['target']]
    
    # Calculate effective samples per class after weighting
    effective_samples = {label: int(count * class_weights[label]) 
                        for label, count in class_counts.items()}
    
    print(f"\n{'='*70}")
    print("WEIGHTED RANDOM SAMPLER (UNDERSAMPLING)")
    print(f"{'='*70}")
    print(f"Original class counts: {class_counts}")
    print(f"Minority class count: {minority_count}")
    print(f"Class weights (undersample majority): {class_weights}")
    print(f"Effective samples per class: {effective_samples}")
    print(f"Total original samples: {total_samples}")
    print(f"Effective balanced samples: ~{minority_count * len(class_counts)}")
    print(f"{'='*70}\n")
    
    # Create sampler with replacement=True
    # num_samples = effective balanced dataset size
    num_samples = minority_count * len(class_counts)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler

def get_dataloaders(
    metadata_path: str,
    img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train/val/test sets.
    
    Args:
        metadata_path: Path to metadata CSV
        img_dir: Directory containing images
        batch_size: Batch size
        num_workers: Number of worker processes
        img_size: Image size for resizing
        random_state: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and split data
    train_df, val_df, test_df = load_and_split_data(metadata_path, random_state)
    
    # Get transforms
    train_transform = get_transforms(train=True, img_size=img_size)
    val_transform = get_transforms(train=False, img_size=img_size)
    
    # Create datasets
    train_dataset = ISICTripletDataset(train_df, img_dir, transform=train_transform, train=True)
    val_dataset = ISICTripletDataset(val_df, img_dir, transform=val_transform, train=False)
    test_dataset = ISICTripletDataset(test_df, img_dir, transform=val_transform, train=False)
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example configuration
    METADATA_PATH = "path/to/train-metadata.csv"
    IMG_DIR = "path/to/train-image/image/"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    IMG_SIZE = 224
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=METADATA_PATH,
        img_dir=IMG_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )
    
    print(f"\nDataloaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    anchor, positive, negative, label = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Anchor: {anchor.shape}")
    print(f"Positive: {positive.shape}")
    print(f"Negative: {negative.shape}")
    print(f"Value range: [{anchor.min():.2f}, {anchor.max():.2f}]")
