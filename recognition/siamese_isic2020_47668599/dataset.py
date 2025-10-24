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
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
    patient_aware: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load metadata and split into train, validation, and test sets.
    
    IMPORTANT: Uses patient-aware splitting to prevent data leakage.
    When patient_aware=True, splits by patient_id to ensure no patient
    appears in multiple splits (train/val/test).
    
    Args:
        metadata_path: Path to train-metadata.csv
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        test_split: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        stratify: If True, maintain class distribution in splits
        patient_aware: If True, split by patient_id (no patient in multiple splits)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
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
    
    # Check if patient_id column exists
    has_patient_id = 'patient_id' in df.columns
    
    if patient_aware and has_patient_id:
        print(f"\n{'='*70}")
        print("PATIENT-AWARE SPLITTING ENABLED")
        print("Splitting by patient_id to prevent data leakage")
        print(f"{'='*70}")
        
        # Get unique patients and their majority class
        patient_info = df.groupby('patient_id').agg({
            'target': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Majority class
            'image_name': 'count'  # Number of images per patient
        }).reset_index()
        patient_info.columns = ['patient_id', 'majority_class', 'num_images']
        
        print(f"\nTotal patients: {len(patient_info)}")
        print(f"Total images: {len(df)}")
        print(f"Average images per patient: {len(df)/len(patient_info):.2f}")
        print(f"\nPatient class distribution:")
        print(patient_info['majority_class'].value_counts())
        
        # First split: separate out test patients
        train_val_size = 1.0 - test_split
        if stratify:
            train_val_patients, test_patients = train_test_split(
                patient_info,
                test_size=test_split,
                random_state=random_state,
                stratify=patient_info['majority_class']
            )
            # Second split: separate train and validation patients
            val_size_adjusted = val_split / train_val_size
            train_patients, val_patients = train_test_split(
                train_val_patients,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_patients['majority_class']
            )
        else:
            train_val_patients, test_patients = train_test_split(
                patient_info,
                test_size=test_split,
                random_state=random_state
            )
            val_size_adjusted = val_split / train_val_size
            train_patients, val_patients = train_test_split(
                train_val_patients,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        
        # Get all images for each patient set
        train_df = df[df['patient_id'].isin(train_patients['patient_id'])].copy()
        val_df = df[df['patient_id'].isin(val_patients['patient_id'])].copy()
        test_df = df[df['patient_id'].isin(test_patients['patient_id'])].copy()
        
        # Verify no patient overlap
        train_patient_set = set(train_df['patient_id'].unique())
        val_patient_set = set(val_df['patient_id'].unique())
        test_patient_set = set(test_df['patient_id'].unique())
        
        assert len(train_patient_set & val_patient_set) == 0, "Patient overlap between train and val!"
        assert len(train_patient_set & test_patient_set) == 0, "Patient overlap between train and test!"
        assert len(val_patient_set & test_patient_set) == 0, "Patient overlap between val and test!"
        
        print(f"\n✓ Patient-aware split successful - no patient overlap detected")
        print(f"\nTrain set: {len(train_patients)} patients, {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(val_patients)} patients, {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_patients)} patients, {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
        
    elif patient_aware and not has_patient_id:
        print(f"\n{'!'*70}")
        print("WARNING: patient_aware=True but no 'patient_id' column found!")
        print("Falling back to standard image-level splitting.")
        print("This may result in data leakage if multiple images are from same patient.")
        print(f"{'!'*70}\n")
        
        # Fall back to standard image-level splitting
        train_val_size = 1.0 - test_split
        if stratify:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_split,
                random_state=random_state,
                stratify=df['target']
            )
            val_size_adjusted = val_split / train_val_size
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_df['target']
            )
        else:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_split,
                random_state=random_state
            )
            val_size_adjusted = val_split / train_val_size
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        
        print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    else:
        # Standard image-level splitting (patient_aware=False)
        print(f"\n{'='*70}")
        print("STANDARD IMAGE-LEVEL SPLITTING")
        print(f"{'='*70}")
        
        train_val_size = 1.0 - test_split
        if stratify:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_split,
                random_state=random_state,
                stratify=df['target']
            )
            val_size_adjusted = val_split / train_val_size
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=train_val_df['target']
            )
        else:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_split,
                random_state=random_state
            )
            val_size_adjusted = val_split / train_val_size
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        
        print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print class distributions for all splits
    print(f"\nTrain class distribution:")
    print(train_df['target'].value_counts())
    print(f"\nValidation class distribution:")
    print(val_df['target'].value_counts())
    print(f"\nTest class distribution:")
    print(test_df['target'].value_counts())
    
    return train_df, val_df, test_df


def undersample_training_only(
    train_df: pd.DataFrame,
    random_state: int = 42,
    patient_aware: bool = True,
    target_ratio: float = 1.0
) -> pd.DataFrame:
    """
    ⚠️ CRITICAL: Undersample ONLY the training set to handle class imbalance.
    
    This is a best practice for medical ML:
    - Balance training data so model learns both classes equally
    - Keep val/test imbalanced to reflect real-world distribution
    - Prevents inflated metrics that don't generalize
    
    Args:
        train_df: Training DataFrame (ONLY training, not val/test)
        random_state: Random seed for reproducibility
        patient_aware: If True, undersample at patient level (recommended)
        target_ratio: Target ratio of minority to majority class (1.0 = balanced)
    
    Returns:
        Undersampled training DataFrame
    """
    print(f"\n{'='*70}")
    print("TRAINING-ONLY UNDERSAMPLING")
    print(f"{'='*70}")
    print("⚠️  Undersampling ONLY training set (val/test remain imbalanced)")
    print("✓  This ensures realistic evaluation on real-world distribution")
    
    # Get class counts
    class_counts = train_df['target'].value_counts().sort_index()
    print(f"\nOriginal training set distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(train_df)*100:.2f}%)")
    
    # Identify majority and minority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    
    print(f"\nMinority class: {minority_class} ({minority_count} samples)")
    print(f"Majority class: {majority_class} ({majority_count} samples)")
    
    # Calculate target count for majority class
    target_majority_count = int(minority_count / target_ratio)
    print(f"Target majority class count: {target_majority_count} (ratio {target_ratio}:1)")
    
    has_patient_id = 'patient_id' in train_df.columns
    
    if patient_aware and has_patient_id:
        print(f"\nUsing PATIENT-LEVEL undersampling (prevents data leakage)")
        print(f"  Strategy: Keep ALL patients with ANY minority images + undersample majority-only patients")
        
        # Identify patients who have at least one minority (melanoma) image
        # vs patients who have ONLY majority (benign) images
        patient_has_minority = train_df.groupby('patient_id')['target'].apply(
            lambda x: (x == minority_class).any()
        ).reset_index()
        patient_has_minority.columns = ['patient_id', 'has_minority']
        
        # Split patients into two groups
        minority_touched_patients = patient_has_minority[patient_has_minority['has_minority'] == True]['patient_id'].values
        majority_only_patients = patient_has_minority[patient_has_minority['has_minority'] == False]['patient_id'].values
        
        print(f"  Patients with ANY minority images: {len(minority_touched_patients)} patients")
        print(f"  Patients with ONLY majority images: {len(majority_only_patients)} patients")
        
        # Get all images from minority-touched patients (keep ALL their images)
        minority_touched_data = train_df[train_df['patient_id'].isin(minority_touched_patients)]
        majority_only_data = train_df[train_df['patient_id'].isin(majority_only_patients)]
        
        # Count minority and majority images from minority-touched patients
        minority_touched_class_counts = minority_touched_data['target'].value_counts()
        print(f"  Images from minority-touched patients:")
        print(f"    Class {minority_class}: {minority_touched_class_counts.get(minority_class, 0)}")
        print(f"    Class {majority_class}: {minority_touched_class_counts.get(majority_class, 0)}")
        
        # Total minority images we'll keep (all from minority-touched patients)
        total_minority_images = minority_touched_class_counts.get(minority_class, 0)
        total_majority_from_minority_touched = minority_touched_class_counts.get(majority_class, 0)
        
        # Calculate target majority images needed to achieve target_ratio
        target_total_majority = int(total_minority_images / target_ratio)
        target_majority_from_majority_only = max(0, target_total_majority - total_majority_from_minority_touched)
        
        print(f"  Target majority images needed: {target_total_majority}")
        print(f"    Already have from minority-touched: {total_majority_from_minority_touched}")
        print(f"    Need from majority-only patients: {target_majority_from_majority_only}")
        
        # Undersample majority-only patients to get the target number
        np.random.seed(random_state)
        
        if target_majority_from_majority_only > 0 and len(majority_only_patients) > 0:
            available_majority_only_images = len(majority_only_data)
            
            if target_majority_from_majority_only >= available_majority_only_images:
                # Need all majority-only images (or more than available)
                print(f"  Using ALL {available_majority_only_images} images from majority-only patients (best-effort balance)")
                sampled_majority_only_data = majority_only_data
            else:
                # Need to sample from majority-only patients
                # Calculate images per patient for each majority-only patient
                patient_image_counts = majority_only_data.groupby('patient_id').size().reset_index(name='image_count')
                patient_image_counts = patient_image_counts.sort_values('image_count')
                
                # Iteratively add patients until we reach or exceed target
                # This maintains patient integrity (all images from a patient or none)
                selected_patients = []
                cumulative_images = 0
                
                for _, row in patient_image_counts.iterrows():
                    patient_id = row['patient_id']
                    patient_images = row['image_count']
                    
                    # Add this patient if we haven't exceeded the target yet
                    # or if we're still far from the target
                    if cumulative_images < target_majority_from_majority_only:
                        selected_patients.append(patient_id)
                        cumulative_images += patient_images
                    elif cumulative_images == 0:
                        # Edge case: need at least one patient
                        selected_patients.append(patient_id)
                        cumulative_images += patient_images
                        break
                
                # If we still need more images, randomly add more patients
                if cumulative_images < target_majority_from_majority_only:
                    remaining_patients = [p for p in majority_only_patients if p not in selected_patients]
                    if len(remaining_patients) > 0:
                        # Shuffle and add remaining patients
                        np.random.shuffle(remaining_patients)
                        for patient_id in remaining_patients:
                            patient_images = len(majority_only_data[majority_only_data['patient_id'] == patient_id])
                            selected_patients.append(patient_id)
                            cumulative_images += patient_images
                            if cumulative_images >= target_majority_from_majority_only:
                                break
                
                sampled_majority_only_patients = np.array(selected_patients)
                sampled_majority_only_data = majority_only_data[
                    majority_only_data['patient_id'].isin(sampled_majority_only_patients)
                ]
                
                # Note: We keep ALL images from sampled patients to maintain patient integrity
                # This may result in slightly more images than the exact target
                print(f"  Sampled {len(sampled_majority_only_patients)} majority-only patients")
                print(f"  Got {len(sampled_majority_only_data)} images from them (target was {target_majority_from_majority_only})")
                if len(sampled_majority_only_data) > target_majority_from_majority_only:
                    print(f"  Note: Keeping {len(sampled_majority_only_data) - target_majority_from_majority_only} extra images to maintain patient integrity")
        else:
            # Don't need any images from majority-only patients
            sampled_majority_only_data = pd.DataFrame()
            print(f"  Not using any majority-only patients (minority-touched patients provide enough balance)")
        
        # Combine all data
        undersampled_df = pd.concat([minority_touched_data, sampled_majority_only_data], ignore_index=True)
        
    else:
        if patient_aware and not has_patient_id:
            print(f"\n⚠️  WARNING: patient_aware=True but no patient_id column found")
            print(f"   Using IMAGE-LEVEL undersampling (may have data leakage)")
        else:
            print(f"\nUsing IMAGE-LEVEL undersampling")
        
        # Image-level undersampling
        minority_data = train_df[train_df['target'] == minority_class]
        majority_data = train_df[train_df['target'] == majority_class]
        
        # Randomly sample from majority class
        np.random.seed(random_state)
        majority_sampled = majority_data.sample(n=target_majority_count, replace=False, random_state=random_state)
        
        # Combine
        undersampled_df = pd.concat([minority_data, majority_sampled], ignore_index=True)
    
    # Shuffle the undersampled data
    undersampled_df = undersampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Print results
    final_counts = undersampled_df['target'].value_counts().sort_index()
    print(f"\nUndersampled training set distribution:")
    for cls, count in final_counts.items():
        print(f"  Class {cls}: {count} samples ({count/len(undersampled_df)*100:.2f}%)")
    
    print(f"\nReduction: {len(train_df)} → {len(undersampled_df)} samples ({len(undersampled_df)/len(train_df)*100:.1f}%)")
    
    # Detailed verification statistics
    print(f"\n⚠️  Patient-level statistics:")
    if has_patient_id:
        final_patients = undersampled_df['patient_id'].nunique()
        print(f"   Patients in undersampled training: {final_patients}")
    print(f"   Images per class: Class 0={final_counts.get(0, 0)}, Class 1={final_counts.get(1, 0)}")
    if final_counts.get(1, 0) > 0:
        print(f"   Ratio (minority:majority): 1:{final_counts.get(0, 0)/final_counts.get(1, 0):.2f}")
    else:
        print(f"   Ratio (minority:majority): No minority samples!")
    
    print(f"\n✓ Training set balanced, val/test remain imbalanced for realistic evaluation")
    print(f"{'='*70}\n")
    
    return undersampled_df


def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Training augmentation uses ONLY geometric transforms (no color augmentation).
    This preserves color information which is critical for melanoma detection.
    
    Args:
        train: If True, include geometric data augmentation
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
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    random_state: int = 42,
    verbose: bool = True,
    use_triplet: bool = False,
    samples_per_class: Optional[int] = None,
    patient_aware: bool = True,
    undersample_training: bool = True,
    target_ratio: float = 1.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test data loaders from a single metadata file.
    
    ⚠️ CRITICAL: When undersample_training=True, ONLY the training set is balanced.
    Validation and test sets remain imbalanced to reflect real-world distribution.
    
    Args:
        metadata_path: Path to train-metadata.csv
        img_dir: Directory with training images (train-image folder)
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        img_size: Image size
        random_state: Random seed for reproducibility
        verbose: If True, print detailed statistics
        use_triplet: If True, use triplet dataset instead of pair dataset
        samples_per_class: Number of samples per class (for triplet dataset)
        patient_aware: If True, split by patient_id to prevent data leakage
        undersample_training: If True, balance ONLY training set (CRITICAL for medical ML)
        target_ratio: Target ratio for undersampling (1.0 = balanced, 0.5 = 2:1)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and split data
    train_df, val_df, test_df = load_and_split_data(
        metadata_path=metadata_path,
        val_split=val_split,
        test_split=test_split,
        random_state=random_state,
        stratify=True,
        patient_aware=patient_aware
    )
    
    # ⚠️ CRITICAL: Undersample ONLY training set (val/test remain imbalanced)
    if undersample_training:
        train_df = undersample_training_only(
            train_df=train_df,
            random_state=random_state,
            patient_aware=patient_aware,
            target_ratio=target_ratio
        )
    else:
        print(f"\n⚠️  WARNING: Training on imbalanced data (98:2 ratio)")
        print(f"   Consider setting undersample_training=True for better results\n")
    
    # Choose dataset type
    if use_triplet:
        dataset_class = ISICTripletDataset
        
        # Adjust samples_per_class to match actual training data after undersampling
        if samples_per_class is not None:
            minority_count = (train_df['target'] == 1).sum()
            majority_count = (train_df['target'] == 0).sum()
            # Set samples_per_class to the smaller class size to avoid excessive repetition
            adjusted_samples = min(minority_count, majority_count)
            
            if samples_per_class > adjusted_samples:
                print(f"\n{'='*70}")
                print(f"ADJUSTING SAMPLES_PER_CLASS")
                print(f"{'='*70}")
                print(f"⚠️  Requested samples_per_class: {samples_per_class}")
                print(f"   Training set after undersampling: {len(train_df)} images")
                print(f"     Class 0 (benign): {majority_count} images")
                print(f"     Class 1 (melanoma): {minority_count} images")
                print(f"   Adjusted samples_per_class: {adjusted_samples} (min of both classes)")
                print(f"   Epoch size will be: {2 * adjusted_samples} triplets = ~{(2 * adjusted_samples) // batch_size} batches")
                print(f"{'='*70}\n")
                samples_per_class = adjusted_samples
            
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
    
    if verbose:
        print(f"\nCreating Test Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    test_dataset = dataset_class(
        df=test_df,
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
    
    test_loader = DataLoader(
        test_dataset,
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
        print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


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
        train_loader, val_loader, test_loader = create_data_loaders(
            metadata_path=metadata_path,
            img_dir=img_dir,
            val_split=0.1,
            test_split=0.1,
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