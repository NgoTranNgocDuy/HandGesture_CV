"""
PyTorch Dataset and DataLoader for Hand Gesture Recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import json
import random
from PIL import Image

class HandGestureDataset(Dataset):
    """
    Custom PyTorch Dataset for hand gesture recognition
    Supports both image and landmark data
    """
    
    def __init__(self, dataset_root='dataset', mode='train', image_size=224, 
                 sequence_length=30, transform=None, use_augmentation=True):
        """
        Args:
            dataset_root: Root directory of dataset
            mode: 'train', 'val', or 'test'
            image_size: Size to resize images to
            sequence_length: Number of frames for LSTM
            transform: Custom transforms
            use_augmentation: Apply data augmentation
        """
        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation and (mode == 'train')
        
        # Load dataset info
        with open(self.dataset_root / 'metadata' / 'dataset_info.json', 'r') as f:
            self.info = json.load(f)
        
        self.classes = self.info['classes']
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Default transforms
        if transform is None:
            if self.use_augmentation:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load data
        self.samples = self._load_samples()
        
        # Train/val/test split
        self._split_dataset()
    
    def _load_samples(self):
        """Load all samples from dataset"""
        samples = []
        
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            
            # Get image paths
            image_dir = self.dataset_root / 'images' / class_name
            landmark_dir = self.dataset_root / 'landmarks' / class_name
            
            image_files = sorted(list(image_dir.glob('*.jpg')))
            
            for img_path in image_files:
                # Corresponding landmark file
                lm_path = landmark_dir / (img_path.stem + '.npy')
                
                if lm_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'landmark_path': str(lm_path),
                        'class': class_name,
                        'label': class_idx
                    })
        
        return samples
    
    def _split_dataset(self):
        """Split dataset into train/val/test (70/15/15)"""
        # Shuffle samples deterministically
        random.seed(42)
        random.shuffle(self.samples)
        
        n = len(self.samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if self.mode == 'train':
            self.samples = self.samples[:train_end]
        elif self.mode == 'val':
            self.samples = self.samples[train_end:val_end]
        elif self.mode == 'test':
            self.samples = self.samples[val_end:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, image_size, image_size)
            landmarks: Tensor of shape (sequence_length, 63)
            label: Integer class label
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load landmarks
        landmarks = np.load(sample['landmark_path'])  # Shape: (21, 3)
        landmarks = landmarks.flatten()  # Shape: (63,)
        
        # Create sequence by repeating (for LSTM)
        # In practice, you'd collect temporal sequences
        landmarks_seq = np.tile(landmarks, (self.sequence_length, 1))  # (seq_len, 63)
        
        # Add noise for augmentation
        if self.use_augmentation:
            noise = np.random.normal(0, 0.01, landmarks_seq.shape)
            landmarks_seq = landmarks_seq + noise
        
        landmarks_seq = torch.FloatTensor(landmarks_seq)
        label = torch.LongTensor([sample['label']])[0]
        
        return image, landmarks_seq, label


def get_data_loaders(dataset_root='dataset', batch_size=32, num_workers=4, 
                     image_size=224, sequence_length=30):
    """
    Create train, validation, and test data loaders
    
    Args:
        dataset_root: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Size to resize images to
        sequence_length: Sequence length for LSTM
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = HandGestureDataset(
        dataset_root=dataset_root,
        mode='train',
        image_size=image_size,
        sequence_length=sequence_length,
        use_augmentation=True
    )
    
    val_dataset = HandGestureDataset(
        dataset_root=dataset_root,
        mode='val',
        image_size=image_size,
        sequence_length=sequence_length,
        use_augmentation=False
    )
    
    test_dataset = HandGestureDataset(
        dataset_root=dataset_root,
        mode='test',
        image_size=image_size,
        sequence_length=sequence_length,
        use_augmentation=False
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
    
    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"  Number of classes: {train_dataset.num_classes}")
    print(f"  Batch size: {batch_size}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("="*60)
    print("Testing Hand Gesture Dataset")
    print("="*60)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=4,
        num_workers=0
    )
    
    # Get a batch
    images, landmarks, labels = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Landmarks: {landmarks.shape}")
    print(f"  Labels: {labels.shape}")
    
    print("\n" + "="*60)
    print("Dataset loaded successfully! ✓")
    print("="*60)
