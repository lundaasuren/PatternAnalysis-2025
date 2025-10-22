"""
Dataset loader for Siamese Network training on ISIC 2020 dataset.
Generates pairs of images (positive and negative) for contrastive learning.
Handles train/val split from a single metadata file.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split


class ISICSiameseDataset(Dataset):
    """
    Dataset class for generating image pairs for Siamese network training.
    
    Pairs are generated as:
    - Positive pairs (label=1): Both images from the same class
    - Negative pairs (label=0): Images from different classes
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        train: bool = True,
        balance_pairs: bool = True
    ):
        """
        Args:
            df: DataFrame with image names and labels (after train/val split)
            img_dir: Directory containing images
            transform: Optional transform to be applied on images
            train: If True, generate pairs dynamically. If False, generate fixed pairs for validation
            balance_pairs: If True, ensure equal number of positive and negative pairs
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.balance_pairs = balance_pairs
        
        # Ensure required columns exist
        # Handle different column naming conventions
        if 'isic_id' in self.df.columns and 'image_name' not in self.df.columns:
            self.df['image_name'] = self.df['isic_id']
        
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            available_cols = ', '.join(self.df.columns.tolist())
            raise ValueError(f"DataFrame must contain 'image_name' (or 'isic_id') and 'target' columns. "
                           f"Available columns: {available_cols}")
        
        # Create label-to-indices mapping for efficient pair generation
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
        
        # For validation, pre-generate pairs
        if not self.train:
            self.pairs = self._generate_fixed_pairs()
            print(f"Generated {len(self.pairs)} fixed pairs for validation")
    
    def _generate_fixed_pairs(self, num_pairs_per_class: int = 500) -> List[Tuple]:
        """
        Generate fixed pairs for validation/testing.
        
        Args:
            num_pairs_per_class: Number of positive and negative pairs to generate per class
        
        Returns:
            List of tuples (idx1, idx2, label)
        """
        pairs = []
        
        # Generate positive pairs (same class)
        for label in self.labels:
            indices = self.label_to_indices[label]
            if len(indices) < 2:
                continue
            
            # Generate pairs up to the limit or all possible combinations
            max_pairs = min(num_pairs_per_class, len(indices) * (len(indices) - 1) // 2)
            pairs_generated = 0
            
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if pairs_generated >= max_pairs:
                        break
                    pairs.append((indices[i], indices[j], 1))  # label=1 for same class
                    pairs_generated += 1
                if pairs_generated >= max_pairs:
                    break
        
        # Generate negative pairs (different classes)
        num_negative = len(pairs)  # Match number of positive pairs
        for _ in range(num_negative):
            if len(self.labels) < 2:
                break
            label1, label2 = random.sample(self.labels, 2)
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            pairs.append((idx1, idx2, 0))  # label=0 for different class
        
        random.shuffle(pairs)
        return pairs
    
    def _get_random_pair(self) -> Tuple[int, int, int]:
        """
        Generate a random pair (positive or negative) on-the-fly.
        
        Returns:
            Tuple of (idx1, idx2, label)
        """
        # Randomly decide if this should be a positive or negative pair
        if random.random() < 0.5:
            # Positive pair (same class)
            label = random.choice(self.labels)
            indices = self.label_to_indices[label]
            
            # Need at least 2 images of same class
            if len(indices) < 2:
                # Fall back to negative pair
                return self._get_negative_pair()
            
            idx1, idx2 = random.sample(indices, 2)
            return idx1, idx2, 1
        else:
            # Negative pair (different classes)
            return self._get_negative_pair()
    
    def _get_negative_pair(self) -> Tuple[int, int, int]:
        """
        Generate a negative pair (different classes).
        
        Returns:
            Tuple of (idx1, idx2, label=0)
        """
        if len(self.labels) < 2:
            raise ValueError("Need at least 2 classes for negative pairs")
        
        label1, label2 = random.sample(self.labels, 2)
        idx1 = random.choice(self.label_to_indices[label1])
        idx2 = random.choice(self.label_to_indices[label2])
        return idx1, idx2, 0
    
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
        
        # Try multiple possible paths (handle nested image directory)
        possible_paths = [
            os.path.join(self.img_dir, img_name),  # Direct path
            os.path.join(self.img_dir, 'image', img_name),  # Nested in 'image' folder
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            img_path = possible_paths[0]  # Use first path for error message
        
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        if self.train:
            # For training, we can generate infinite pairs
            # Return a reasonable epoch size
            return len(self.df)
        else:
            # For validation, return number of pre-generated pairs
            return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images and their similarity label.
        
        Args:
            idx: Index (used for validation, ignored for training)
        
        Returns:
            Tuple of (img1, img2, label)
            - img1, img2: Transformed image tensors
            - label: 1 if same class (similar), 0 if different class (dissimilar)
        """
        if self.train:
            # Generate random pair on-the-fly for training
            idx1, idx2, label = self._get_random_pair()
        else:
            # Use pre-generated pairs for validation
            idx1, idx2, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(idx1)
        img2 = self._load_image(idx2)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return img1, img2, label


class ISICTripletDataset(Dataset):
    """
    Dataset class for generating image triplets for Siamese network training.
    Uses class-balanced sampling to handle severe class imbalance.
    
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
        train: bool = True,
        samples_per_class: Optional[int] = None
    ):
        """
        Args:
            df: DataFrame with image names and labels (after train/val split)
            img_dir: Directory containing images
            transform: Optional transform to be applied on images
            train: If True, generate triplets dynamically. If False, generate fixed triplets
            samples_per_class: Number of triplets per class per epoch (for class balancing)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        
        # Ensure required columns exist
        # Handle different column naming conventions
        if 'isic_id' in self.df.columns and 'image_name' not in self.df.columns:
            self.df['image_name'] = self.df['isic_id']
        
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            available_cols = ', '.join(self.df.columns.tolist())
            raise ValueError(f"DataFrame must contain 'image_name' (or 'isic_id') and 'target' columns. "
                           f"Available columns: {available_cols}")
        
        # Create label-to-indices mapping for efficient triplet generation
        self.label_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            label = row['target']
            self.label_to_indices[label].append(idx)
        
        self.labels = sorted(self.label_to_indices.keys())
        
        # Class-balanced sampling: equal triplets per class
        if samples_per_class is None:
            # Default: generate equal number of triplets for each class
            min_class_size = min(len(self.label_to_indices[label]) for label in self.labels)
            samples_per_class = min(min_class_size * 2, 500)  # Cap at 500 per class
        
        self.samples_per_class = samples_per_class
        
        print(f"Triplet Dataset loaded: {len(self.df)} images")
        print(f"Class distribution:")
        for label in self.labels:
            count = len(self.label_to_indices[label])
            print(f"  Class {label}: {count} images ({count/len(self.df)*100:.2f}%)")
        print(f"Samples per class per epoch: {samples_per_class}")
        
        # For validation, pre-generate triplets
        if not self.train:
            self.triplets = self._generate_fixed_triplets()
            print(f"Generated {len(self.triplets)} fixed triplets for validation")
    
    def _generate_fixed_triplets(self) -> List[Tuple]:
        """
        Generate fixed triplets for validation/testing with class balancing.
        
        Returns:
            List of tuples (anchor_idx, positive_idx, negative_idx)
        """
        triplets = []
        
        # Generate equal number of triplets per class
        for anchor_label in self.labels:
            anchor_indices = self.label_to_indices[anchor_label]
            if len(anchor_indices) < 2:
                continue
            
            # Get negative label
            negative_labels = [l for l in self.labels if l != anchor_label]
            if not negative_labels:
                continue
            
            for _ in range(self.samples_per_class):
                # Select anchor and positive from same class
                if len(anchor_indices) < 2:
                    break
                anchor_idx, positive_idx = random.sample(anchor_indices, 2)
                
                # Select negative from different class
                negative_label = random.choice(negative_labels)
                negative_idx = random.choice(self.label_to_indices[negative_label])
                
                triplets.append((anchor_idx, positive_idx, negative_idx))
        
        random.shuffle(triplets)
        return triplets
    
    def _get_random_triplet(self) -> Tuple[int, int, int]:
        """
        Generate a random triplet with class-balanced sampling.
        
        Returns:
            Tuple of (anchor_idx, positive_idx, negative_idx)
        """
        # Randomly select anchor class (uniform over classes, not samples)
        anchor_label = random.choice(self.labels)
        anchor_indices = self.label_to_indices[anchor_label]
        
        if len(anchor_indices) < 2:
            # Not enough samples in this class, try again
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
        
        # Try multiple possible paths (handle nested image directory)
        possible_paths = [
            os.path.join(self.img_dir, img_name),  # Direct path
            os.path.join(self.img_dir, 'image', img_name),  # Nested in 'image' folder
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            img_path = possible_paths[0]  # Use first path for error message
        
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        """Return the number of triplets in the dataset."""
        if self.train:
            # For training: class-balanced epoch size
            return len(self.labels) * self.samples_per_class
        else:
            # For validation: fixed triplets
            return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet of images.
        
        Args:
            idx: Index (used for validation, ignored for training)
        
        Returns:
            Tuple of (anchor, positive, negative)
        """
        if self.train:
            # Generate random triplet with class balancing
            anchor_idx, positive_idx, negative_idx = self._get_random_triplet()
        else:
            # Use pre-generated triplets
            anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        # Load images
        anchor = self._load_image(anchor_idx)
        positive = self._load_image(positive_idx)
        negative = self._load_image(negative_idx)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


def load_and_split_data(
    metadata_path: str,
    val_split: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load metadata and split into train and validation sets.
    
    Args:
        metadata_path: Path to train-metadata.csv
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify: If True, maintain class distribution in splits
    
    Returns:
        Tuple of (train_df, val_df)
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Handle different column naming conventions (ISIC 2020 uses 'isic_id')
    if 'isic_id' in df.columns and 'image_name' not in df.columns:
        df['image_name'] = df['isic_id']
    
    print(f"Loaded metadata: {len(df)} samples")
    print(f"\nColumns found: {', '.join(df.columns.tolist())}")
    print(f"\nOriginal class distribution:")
    print(df['target'].value_counts())
    print(f"\nClass percentages:")
    print(df['target'].value_counts(normalize=True) * 100)
    
    # Split data
    if stratify:
        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            random_state=random_state,
            stratify=df['target']
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            random_state=random_state
        )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"\nTrain class distribution:")
    print(train_df['target'].value_counts())
    print(f"\nValidation class distribution:")
    print(val_df['target'].value_counts())
    
    return train_df, val_df


def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        train: If True, include data augmentation
        img_size: Target image size
    
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    metadata_path: str,
    img_dir: str,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    random_state: int = 42,
    verbose: bool = True,
    use_triplet: bool = False,
    samples_per_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders from a single metadata file.
    
    Args:
        metadata_path: Path to train-metadata.csv
        img_dir: Directory with training images (train-image folder)
        val_split: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        img_size: Image size
        random_state: Random seed for reproducibility
        verbose: If True, print detailed statistics
        use_triplet: If True, use triplet dataset instead of pair dataset
        samples_per_class: Number of samples per class (for triplet dataset)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load and split data
    train_df, val_df = load_and_split_data(
        metadata_path=metadata_path,
        val_split=val_split,
        random_state=random_state,
        stratify=True
    )
    
    # Choose dataset type
    if use_triplet:
        dataset_class = ISICTripletDataset
        dataset_kwargs = {'samples_per_class': samples_per_class}
    else:
        dataset_class = ISICSiameseDataset
        dataset_kwargs = {}
    
    # Create datasets
    if verbose:
        print(f"\nCreating Training Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    train_dataset = dataset_class(
        df=train_df,
        img_dir=img_dir,
        transform=get_transforms(train=True, img_size=img_size),
        train=True,
        **dataset_kwargs
    )
    
    if verbose:
        print(f"\nCreating Validation Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    val_dataset = dataset_class(
        df=val_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=img_size),
        train=False,
        **dataset_kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    if verbose:
        print("\nData Loaders Ready:")
        print("-" * 60)
        print(f"Train batches per epoch: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    """
    Simple test to verify the dataset loader works correctly.
    Usage: python dataset.py
    """
    # Example usage - ADJUST THESE PATHS TO YOUR ACTUAL DATA LOCATION
    metadata_path = "data/train-metadata.csv"
    img_dir = "data/train-image"
    
    print("Testing ISIC 2020 Siamese Dataset Loader\n")
    
    try:
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            img_dir=img_dir,
            val_split=0.2,
            batch_size=32,
            num_workers=4,
            random_state=42
        )
        
        # Quick sanity check
        img1, img2, labels = next(iter(train_loader))
        print(f"\n✓ Data loader test passed!")
        print(f"  Batch shape: {img1.shape}")
        print(f"  Label distribution in batch: {labels.sum().item()}/{len(labels)} positive pairs")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nPlease verify:")
        print("  - Metadata path is correct")
        print("  - Image directory is correct")
        print("  - Required columns exist in CSV")