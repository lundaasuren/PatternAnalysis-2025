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
        Uses deterministic generation to avoid heavy repetition.
        
        Args:
            num_pairs_per_class: Number of positive and negative pairs to generate per class
        
        Returns:
            List of tuples (idx1, idx2, label)
        """
        pairs = []
        
        # Use numpy random for deterministic generation
        rng = np.random.RandomState(42)
        
        # Generate positive pairs (same class)
        for label in self.labels:
            indices = self.label_to_indices[label]
            if len(indices) < 2:
                continue
            
            # Generate all possible unique pairs
            max_possible_pairs = len(indices) * (len(indices) - 1) // 2
            max_pairs = min(num_pairs_per_class, max_possible_pairs)
            
            if max_pairs < num_pairs_per_class:
                print(f"  Warning: Class {label} has only {max_possible_pairs} unique pairs, "
                      f"using all (requested {num_pairs_per_class})")
            
            pairs_generated = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if pairs_generated >= max_pairs:
                        break
                    pairs.append((indices[i], indices[j], 1))  # label=1 for same class
                    pairs_generated += 1
                if pairs_generated >= max_pairs:
                    break
        
        # Generate negative pairs (different classes) - deterministic sampling
        num_negative = len(pairs)  # Match number of positive pairs
        
        if len(self.labels) >= 2:
            # Generate negative pairs deterministically without creating all combinations in memory
            # Use systematic sampling with deterministic random seed
            negative_pairs = []
            
            # Get all indices by class for cross-class pairing
            class_indices_list = [self.label_to_indices[label] for label in self.labels]
            
            # Generate negative pairs by deterministically sampling indices
            for pair_idx in range(num_negative):
                # Deterministically select two different classes
                class_idx1 = pair_idx % len(self.labels)
                class_idx2 = (pair_idx + 1) % len(self.labels)
                if class_idx1 == class_idx2:
                    class_idx2 = (class_idx2 + 1) % len(self.labels)
                
                indices1 = class_indices_list[class_idx1]
                indices2 = class_indices_list[class_idx2]
                
                # Deterministically select indices from each class
                idx1 = indices1[rng.randint(0, len(indices1))]
                idx2 = indices2[rng.randint(0, len(indices2))]
                
                negative_pairs.append((idx1, idx2, 0))
            
            pairs.extend(negative_pairs)
        
        # Shuffle pairs deterministically
        rng.shuffle(pairs)
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
            min_class_size = min(len(self.label_to_indices[label]) for label in self.labels)

            # Goal: Use ~25% of the dataset per epoch for good diversity.
            target_epoch_coverage = 0.25
            total_images = len(self.df)
            target_triplets = int(total_images * target_epoch_coverage)
            samples_per_class = target_triplets // len(self.labels)

            # Safety net: Ensure we use at least 2x minority class (allows controlled repetition).
            samples_per_class = max(samples_per_class, min_class_size * 2)

            # Informative logging to verify the fix.
            print(f"\n{'='*70}")
            print("AUTO-CALCULATED SAMPLES_PER_CLASS")
            print(f"{'='*70}")
            print(f"Total images in training pool: {total_images:,}")
            print(f"Minority class size: {min_class_size:,}")
            print(f"Target epoch coverage: {target_epoch_coverage*100:.0f}%")
            print(f"Auto-calculated samples_per_class: {samples_per_class:,}")
            print(f"Total triplets per epoch: {samples_per_class * len(self.labels):,}")
            print(f"{'='*70}\n")
        
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
        Uses deterministic generation to avoid heavy repetition.
        
        Strategy:
        - Generate all possible unique anchor-positive pairs per class
        - Assign negatives systematically (round-robin from other classes)
        - Limit to available unique combinations to avoid excessive repetition
        
        Returns:
            List of tuples (anchor_idx, positive_idx, negative_idx)
        """
        triplets = []
        
        # Use numpy random for deterministic generation (already seeded in dataset creation)
        rng = np.random.RandomState(42)
        
        # Generate triplets per class (anchor class)
        for anchor_label in self.labels:
            anchor_indices = self.label_to_indices[anchor_label]
            if len(anchor_indices) < 2:
                continue
            
            # Get indices for negative class(es)
            negative_labels = [l for l in self.labels if l != anchor_label]
            if not negative_labels:
                continue
            
            # Generate all possible unique anchor-positive pairs for this class
            anchor_positive_pairs = []
            for i, anchor_idx in enumerate(anchor_indices):
                for positive_idx in anchor_indices[i+1:]:
                    anchor_positive_pairs.append((anchor_idx, positive_idx))
            
            # Calculate how many triplets we can reasonably generate
            max_unique_pairs = len(anchor_positive_pairs)
            target_triplets = min(self.samples_per_class, max_unique_pairs)
            
            # If we have fewer unique pairs than requested, take all
            if target_triplets < self.samples_per_class:
                print(f"  Warning: Class {anchor_label} has only {max_unique_pairs} unique pairs, "
                      f"using all (requested {self.samples_per_class})")
            
            # Sample or take all pairs (deterministic with seed)
            if target_triplets < len(anchor_positive_pairs):
                # Randomly sample without replacement
                selected_indices = rng.choice(len(anchor_positive_pairs), size=target_triplets, replace=False)
                selected_pairs = [anchor_positive_pairs[i] for i in selected_indices]
            else:
                selected_pairs = anchor_positive_pairs
            
            # Assign negatives systematically (round-robin to distribute evenly)
            negative_indices_all = []
            for neg_label in negative_labels:
                negative_indices_all.extend(self.label_to_indices[neg_label])
            
            # Shuffle negative indices for randomness (deterministic)
            rng.shuffle(negative_indices_all)
            
            # Create triplets by cycling through negatives
            for idx, (anchor_idx, positive_idx) in enumerate(selected_pairs):
                negative_idx = negative_indices_all[idx % len(negative_indices_all)]
                triplets.append((anchor_idx, positive_idx, negative_idx))
        
        # Shuffle triplets (deterministic)
        rng.shuffle(triplets)
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
    Patient-aware undersampling for training set only.
    
    Strategy:
    1. Keep ALL patients with ANY minority images (maintains patient integrity)
    2. Add majority-only patients until reaching target ratio (if possible)
    3. Accept best-effort ratio if minority-touched patients exceed target
    
    Args:
        train_df: Training dataframe with 'patient_id' and 'target' columns
        random_state: Random seed for reproducibility
        patient_aware: If True, use patient-level undersampling
        target_ratio: Target majority:minority ratio (e.g., 3.0 means 3:1)
    
    Returns:
        Undersampled training dataframe
    """
    print(f"\n{'='*70}")
    print("PATIENT-AWARE TRAINING UNDERSAMPLING")
    print(f"{'='*70}")
    
    # Identify minority and majority classes
    minority_class = train_df['target'].value_counts().idxmin()
    majority_class = train_df['target'].value_counts().idxmax()
    
    minority_count = (train_df['target'] == minority_class).sum()
    majority_count = (train_df['target'] == majority_class).sum()
    target_majority_count = int(minority_count * target_ratio)  # ✅ FIXED: multiplication not division
    
    print(f"\nOriginal training distribution:")
    print(f"  Minority (class {minority_class}): {minority_count} images")
    print(f"  Majority (class {majority_class}): {majority_count} images")
    print(f"  Target majority for ratio {target_ratio}:1 = {target_majority_count}")
    
    has_patient_id = 'patient_id' in train_df.columns
    
    if patient_aware and has_patient_id:
        # Step 1: Identify patients with ANY minority class images
        minority_patients = train_df[train_df['target'] == minority_class]['patient_id'].unique()
        print(f"\nPatients with ANY minority images: {len(minority_patients)}")
        
        # Step 2: Keep ALL images from these patients
        minority_touched_df = train_df[train_df['patient_id'].isin(minority_patients)]
        minority_touched_majority = (minority_touched_df['target'] == majority_class).sum()
        minority_touched_minority = (minority_touched_df['target'] == minority_class).sum()
        
        print(f"  Images from these patients:")
        print(f"    Minority: {minority_touched_minority}")
        print(f"    Majority: {minority_touched_majority}")
        print(f"    Total: {len(minority_touched_df)}")
        
        # Step 3: Check if we need more majority images from majority-only patients
        remaining_majority_needed = target_majority_count - minority_touched_majority
        
        if remaining_majority_needed > 0:
            print(f"\n  Additional majority images needed: {remaining_majority_needed}")
            
            # Get majority-only patients
            majority_only_patients = train_df[
                (~train_df['patient_id'].isin(minority_patients)) & 
                (train_df['target'] == majority_class)
            ]['patient_id'].unique()
            
            print(f"  Majority-only patients available: {len(majority_only_patients)}")
            
            # Sample patients (not images) to respect patient integrity
            np.random.seed(random_state)
            
            # Greedily add patients until we reach target
            sampled_patients = []
            current_majority = minority_touched_majority
            
            for patient in np.random.permutation(majority_only_patients):
                patient_images = train_df[train_df['patient_id'] == patient]
                patient_majority_count = (patient_images['target'] == majority_class).sum()
                
                if current_majority + patient_majority_count <= target_majority_count * 1.1:  # 10% tolerance
                    sampled_patients.append(patient)
                    current_majority += patient_majority_count
                    
                    if current_majority >= target_majority_count:
                        break
            
            print(f"  Sampled {len(sampled_patients)} majority-only patients")
            print(f"  Total majority images: {current_majority}")
            
            # Combine minority-touched + sampled majority-only
            majority_only_df = train_df[train_df['patient_id'].isin(sampled_patients)]
            undersampled_df = pd.concat([minority_touched_df, majority_only_df])
            
        else:
            print(f"\n⚠️  Minority-touched patients already provide {minority_touched_majority} majority images")
            print(f"   This exceeds target of {target_majority_count}")
            print(f"   Keeping all minority-touched patients (strict patient integrity)")
            undersampled_df = minority_touched_df
    
    else:
        # Fallback to image-level undersampling if no patient_id
        if patient_aware and not has_patient_id:
            print(f"\n⚠️  WARNING: patient_aware=True but no patient_id column found")
            print(f"   Using IMAGE-LEVEL undersampling (may have data leakage)")
        else:
            print(f"\nUsing IMAGE-LEVEL undersampling")
        
        minority_data = train_df[train_df['target'] == minority_class]
        majority_data = train_df[train_df['target'] == majority_class]
        
        if len(majority_data) > target_majority_count:
            majority_sampled = majority_data.sample(n=target_majority_count, replace=False, random_state=random_state)
        else:
            majority_sampled = majority_data
        
        undersampled_df = pd.concat([minority_data, majority_sampled], ignore_index=True)
    
    # Shuffle
    undersampled_df = undersampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Final statistics
    final_minority = (undersampled_df['target'] == minority_class).sum()
    final_majority = (undersampled_df['target'] == majority_class).sum()
    final_ratio = final_majority / final_minority if final_minority > 0 else 0
    
    print(f"\n{'='*70}")
    print("UNDERSAMPLING RESULTS")
    print(f"{'='*70}")
    print(f"Original: {len(train_df)} images")
    print(f"Undersampled: {len(undersampled_df)} images ({len(undersampled_df)/len(train_df)*100:.1f}%)")
    print(f"\nFinal distribution:")
    print(f"  Minority: {final_minority} ({final_minority/len(undersampled_df)*100:.1f}%)")
    print(f"  Majority: {final_majority} ({final_majority/len(undersampled_df)*100:.1f}%)")
    print(f"  Actual ratio: {final_ratio:.2f}:1")
    print(f"  Target ratio: {target_ratio:.2f}:1")
    
    if abs(final_ratio - target_ratio) / target_ratio > 0.2:  # More than 20% off
        print(f"\n⚠️  Note: Actual ratio differs from target due to patient-level constraints")
        print(f"   This is CORRECT and maintains patient data integrity")
    
    print(f"\n✓ Training pool undersampled with patient-level integrity maintained")
    print(f"  Per-epoch sampling will draw from this pool")
    print(f"  Val/test remain at original imbalanced distribution for realistic evaluation")
    print(f"{'='*70}\n")
    
    return undersampled_df


def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Training augmentation uses geometric transforms with mild color augmentation.
    Mild color jitter preserves diagnostic color information while adding diversity.
    
    Args:
        train: If True, include geometric data augmentation
        img_size: Target image size
    
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            # RandomResizedCrop for better robustness (replaces Resize)
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            # Mild ColorJitter for photometric variety (preserves diagnostic color info)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        
        # Print training data strategy summary
        minority_count_train = (train_df['target'] == 1).sum()
        majority_count_train = (train_df['target'] == 0).sum()
        print(f"{'='*70}")
        print(f"TRAINING DATA STRATEGY")
        print(f"{'='*70}")
        print(f"✓ Training pool: Undersampled to {len(train_df)} images (~{majority_count_train}:{minority_count_train} ratio)")
        print(f"  This balanced pool enables fair learning from both classes")
        print(f"✓ Per-epoch sampling: Triplet generation draws from this pool")
        if use_triplet:
            epoch_size_estimate = min(minority_count_train, majority_count_train) * 2
            print(f"  Each epoch will use ~{epoch_size_estimate} triplets (class-balanced sampling)")
        print(f"✓ Val/test: Remain at original imbalanced distribution (98:2 ratio)")
        print(f"  Realistic evaluation on real-world class imbalance")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠️  WARNING: Training on imbalanced data (98:2 ratio)")
        print(f"   Consider setting undersample_training=True for better results\n")
    
    # Choose dataset type
    if use_triplet:
        dataset_class = ISICTripletDataset
        
        # Adjust samples_per_class to match actual training data after undersampling
        train_samples_per_class = samples_per_class
        if samples_per_class is not None:
            minority_count = (train_df['target'] == 1).sum()
            majority_count = (train_df['target'] == 0).sum()
            
            # Maximum possible samples without repetition
            max_possible_samples = min(minority_count, majority_count)
            
            # Allow controlled repetition for better data utilization
            repetition_factor = 2  # Allow 2x repetition for minority class
            max_allowed_samples = minority_count * repetition_factor
            
            # Check if adjustment is needed
            if samples_per_class > minority_count:
                # Requested more samples than minority class size - repetition needed
                if samples_per_class > max_allowed_samples:
                    # Exceeds even with maximum allowed repetition
                    adjusted_samples_per_class = max_allowed_samples
                    print(f"\n{'='*70}")
                    print(f"ADJUSTING SAMPLES_PER_CLASS FOR TRAINING")
                    print(f"{'='*70}")
                    print(f"⚠️  Requested samples_per_class: {samples_per_class}")
                    print(f"   Training set after undersampling: {len(train_df)} images")
                    print(f"     Class 0 (benign): {majority_count} images")
                    print(f"     Class 1 (melanoma): {minority_count} images")
                    print(f"   Maximum allowed with {repetition_factor}x repetition: {max_allowed_samples}")
                    print(f"   Adjusted samples_per_class: {adjusted_samples_per_class}")
                    print(f"{'='*70}\n")
                else:
                    # Repetition needed but within allowed limit
                    adjusted_samples_per_class = samples_per_class
                    repetition_actual = samples_per_class / minority_count
                    print(f"\n{'='*70}")
                    print(f"CONTROLLED REPETITION ENABLED")
                    print(f"{'='*70}")
                    print(f"   Requested samples_per_class: {samples_per_class}")
                    print(f"   Minority class images: {minority_count}")
                    print(f"   Majority class images: {majority_count}")
                    print(f"🔄 Each minority image used {repetition_actual:.1f} times per epoch")
                    print(f"   This allows better batch diversity for triplet mining")
                    print(f"{'='*70}\n")
            else:
                # Requested samples_per_class is within minority class size - no repetition needed
                adjusted_samples_per_class = samples_per_class
            
            # Calculate and log effective training parameters
            train_samples_per_class = adjusted_samples_per_class
            total_triplets = train_samples_per_class * 2
            batch_count = int(np.ceil(total_triplets / batch_size))
            
            print(f"{'='*70}")
            print(f"EFFECTIVE TRAINING PARAMETERS")
            print(f"{'='*70}")
            print(f"   Samples per class: {train_samples_per_class}")
            print(f"   Total triplets per epoch: {total_triplets}")
            print(f"   Batch size: {batch_size}")
            print(f"   Batches per epoch: {batch_count}")
            print(f"{'='*70}\n")
        
        # Use adjusted samples_per_class for training only
        train_dataset_kwargs = {'samples_per_class': train_samples_per_class}
        
        # For val/test, use None to let dataset auto-calculate based on their own data
        # This preserves the imbalanced distribution without excessive repetition
        val_test_dataset_kwargs = {'samples_per_class': None}
        
        print(f"⚠️  Val/test datasets will use auto-calculated samples_per_class based on their minority class size")
        print(f"   This ensures they remain imbalanced (realistic evaluation)\n")
    else:
        dataset_class = ISICSiameseDataset
        train_dataset_kwargs = {}
        val_test_dataset_kwargs = {}
    
    # Create datasets
    if verbose:
        print(f"\nCreating Training Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    train_dataset = dataset_class(
        df=train_df,
        img_dir=img_dir,
        transform=get_transforms(train=True, img_size=img_size),
        train=True,
        **train_dataset_kwargs
    )
    
    if verbose:
        print(f"\nCreating Validation Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    val_dataset = dataset_class(
        df=val_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=img_size),
        train=False,
        **val_test_dataset_kwargs
    )
    
    if verbose:
        print(f"\nCreating Test Dataset ({'Triplet' if use_triplet else 'Pair'}):")
        print("-" * 60)
    test_dataset = dataset_class(
        df=test_df,
        img_dir=img_dir,
        transform=get_transforms(train=False, img_size=img_size),
        train=False,
        **val_test_dataset_kwargs
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