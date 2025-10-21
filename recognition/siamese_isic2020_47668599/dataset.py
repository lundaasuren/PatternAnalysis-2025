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
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            raise ValueError("DataFrame must contain 'image_name' and 'target' columns")
        
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
        
        img_path = os.path.join(self.img_dir, img_name)
        
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
    
    print(f"Loaded metadata: {len(df)} samples")
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
    random_state: int = 42
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
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating Training Dataset:")
    print("="*60)
    train_dataset = ISICSiameseDataset(
        df=train_df,
        img_dir=img_dir,
        transform=get_transforms(train=True, img_size=img_size),
        train=True
    )
    
    print("\n" + "="*60)
    print("Creating Validation Dataset:")
    print("="*60)
    val_dataset = ISICSiameseDataset(
        df=val_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=img_size),
        train=False
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
    
    print("\n" + "="*60)
    print("Data Loaders Created Successfully!")
    print("="*60)
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Test the dataset loader
if __name__ == "__main__":
    """
    Test script to verify the dataset loader works correctly.
    """
    import matplotlib.pyplot as plt
    
    # Example usage - ADJUST THESE PATHS TO YOUR ACTUAL DATA LOCATION
    metadata_path = "path/to/train-metadata.csv"
    img_dir = "path/to/train-image"
    
    print("="*60)
    print("Testing ISIC 2020 Siamese Dataset Loader")
    print("="*60)
    
    # Test data loading and splitting
    try:
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            img_dir=img_dir,
            val_split=0.2,
            batch_size=8,
            num_workers=2,
            random_state=42
        )
        
        print("\n" + "="*60)
        print("Testing Data Loading:")
        print("="*60)
        
        # Get a batch from training loader
        img1_batch, img2_batch, labels_batch = next(iter(train_loader))
        print(f"Train batch - Image1 shape: {img1_batch.shape}")
        print(f"Train batch - Image2 shape: {img2_batch.shape}")
        print(f"Train batch - Labels shape: {labels_batch.shape}")
        print(f"Train batch - Labels: {labels_batch.tolist()}")
        
        # Get a batch from validation loader
        img1_batch, img2_batch, labels_batch = next(iter(val_loader))
        print(f"\nVal batch - Image1 shape: {img1_batch.shape}")
        print(f"Val batch - Image2 shape: {img2_batch.shape}")
        print(f"Val batch - Labels shape: {labels_batch.shape}")
        print(f"Val batch - Labels: {labels_batch.tolist()}")
        
        # Visualize some pairs
        print("\n" + "="*60)
        print("Creating Visualization:")
        print("="*60)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Sample Image Pairs from Siamese Dataset', fontsize=16)
        
        for i in range(min(3, len(img1_batch))):
            img1 = img1_batch[i]
            img2 = img2_batch[i]
            label = labels_batch[i]
            
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img1_vis = img1 * std + mean
            img2_vis = img2 * std + mean
            
            # Clip values to [0, 1]
            img1_vis = torch.clamp(img1_vis, 0, 1)
            img2_vis = torch.clamp(img2_vis, 0, 1)
            
            # Convert to numpy and transpose
            img1_np = img1_vis.permute(1, 2, 0).numpy()
            img2_np = img2_vis.permute(1, 2, 0).numpy()
            
            # Display
            axes[i, 0].imshow(img1_np)
            axes[i, 0].set_title(f'Image 1')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img2_np)
            axes[i, 1].set_title(f'Image 2')
            axes[i, 1].axis('off')
            
            label_text = 'Similar\n(Same Class)' if label.item() == 1 else 'Dissimilar\n(Diff Class)'
            axes[i, 2].text(0.5, 0.5, label_text, 
                           ha='center', va='center', 
                           fontsize=12, fontweight='bold',
                           color='green' if label.item() == 1 else 'red')
            axes[i, 2].axis('off')
            
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_pairs.png', dpi=150, bbox_inches='tight')
        print("Sample pairs visualization saved as 'sample_pairs.png'")
        
        print("\n" + "="*60)
        print("✓ Dataset loader test completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease update the paths in the test script:")
        print("  - metadata_path: path to your train-metadata.csv")
        print("  - img_dir: path to your train-image folder")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()