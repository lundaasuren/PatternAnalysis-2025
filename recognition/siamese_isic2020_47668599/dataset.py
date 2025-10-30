"""
Dataset loader for Siamese Network training on ISIC 2020 dataset.
Generates triplets with balanced sampling via undersampling majority class.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import random
from typing import Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split


class ISICTripletDataset(Dataset):
    """Dataset for triplet generation: anchor, positive, negative."""
    
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        
        if 'isic_id' in self.df.columns and 'image_name' not in self.df.columns:
            self.df['image_name'] = self.df['isic_id']
        
        if 'image_name' not in self.df.columns or 'target' not in self.df.columns:
            raise ValueError("DataFrame must contain 'image_name' and 'target' columns")
        
        self.label_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            self.label_to_indices[row['target']].append(idx)
        
        self.labels = sorted(self.label_to_indices.keys())
    
    def _get_random_triplet(self) -> Tuple[int, int, int]:
        anchor_label = random.choice(self.labels)
        anchor_indices = self.label_to_indices[anchor_label]
        
        if len(anchor_indices) < 2:
            return self._get_random_triplet()
        
        anchor_idx, positive_idx = random.sample(anchor_indices, 2)
        negative_label = random.choice([l for l in self.labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        
        return anchor_idx, positive_idx, negative_idx
    
    def _load_image(self, idx: int) -> Image.Image:
        img_name = self.df.iloc[idx]['image_name']
        
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_name = f"{img_name}.jpg"
        
        possible_paths = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, 'image', img_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    return Image.open(path).convert('RGB')
                except Exception:
                    pass
        
        return Image.new('RGB', (224, 224), color='black')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.train:
            anchor_idx, positive_idx, negative_idx = self._get_random_triplet()
        else:
            anchor_idx = idx
            anchor_label = self.df.iloc[anchor_idx]['target']
            
            positive_candidates = [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
            positive_idx = random.choice(positive_candidates) if positive_candidates else anchor_idx
            
            negative_label = random.choice([l for l in self.labels if l != anchor_label])
            negative_idx = random.choice(self.label_to_indices[negative_label])
        
        anchor = self._load_image(anchor_idx)
        positive = self._load_image(positive_idx)
        negative = self._load_image(negative_idx)
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative, torch.tensor(0)


def load_and_split_data(metadata_path: str, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70:15:15 stratified split."""
    df = pd.read_csv(metadata_path)
    
    if 'isic_id' in df.columns and 'image_name' not in df.columns:
        df['image_name'] = df['isic_id']
    
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_state, stratify=df['target'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state, stratify=temp_df['target'])
    
    return train_df, val_df, test_df


def get_transforms(train: bool = True, img_size: int = 224):
    """Transforms with optional augmentation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])


def create_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Undersamples majority class for balanced training."""
    class_counts = df['target'].value_counts().to_dict()
    minority_count = min(class_counts.values())
    
    class_weights = {label: minority_count / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in df['target']]
    num_samples = minority_count * len(class_counts)
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)


def get_dataloaders(
    metadata_path: str,
    img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create balanced train/val/test dataloaders."""
    train_df, val_df, test_df = load_and_split_data(metadata_path, random_state)
    
    train_transform = get_transforms(train=True, img_size=img_size)
    val_transform = get_transforms(train=False, img_size=img_size)
    
    train_dataset = ISICTripletDataset(train_df, img_dir, transform=train_transform, train=True)
    val_dataset = ISICTripletDataset(val_df, img_dir, transform=val_transform, train=False)
    test_dataset = ISICTripletDataset(test_df, img_dir, transform=val_transform, train=False)
    
    train_sampler = create_weighted_sampler(train_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader