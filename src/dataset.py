"""
PyTorch Dataset class for garbage classification.
Includes data augmentation pipelines for training and validation.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

# Class definitions
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_train_transforms(config: dict) -> transforms.Compose:
    """
    Get training data augmentation transforms.

    Args:
        config: Configuration dictionary

    Returns:
        Composed transforms for training
    """
    aug_config = config.get('augmentation', {})

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(aug_config.get('rotation_degrees', 15)),
        transforms.RandomHorizontalFlip(aug_config.get('horizontal_flip_prob', 0.5)),
        transforms.ColorJitter(
            brightness=aug_config.get('color_jitter', {}).get('brightness', 0.2),
            contrast=aug_config.get('color_jitter', {}).get('contrast', 0.2),
            saturation=aug_config.get('color_jitter', {}).get('saturation', 0.2),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Get validation/test data transforms.

    Returns:
        Composed transforms for validation/test
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class GarbageDataset(Dataset):
    """
    PyTorch Dataset for garbage classification.

    Expects directory structure:
    root_dir/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        classes: List[str] = CLASSES
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform to apply to images
            classes: List of class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Collect all image paths and labels
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their labels."""
        samples = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]

            for img_file in class_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    samples.append((str(img_file), class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank = torch.zeros(3, 224, 224)
            return blank, label

    def get_class_counts(self) -> dict:
        """Get the count of samples per class."""
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            counts[class_name] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        Uses inverse frequency weighting.

        Returns:
            Tensor of class weights
        """
        counts = self.get_class_counts()
        total = len(self.samples)

        weights = []
        for class_name in self.classes:
            count = counts[class_name]
            if count > 0:
                weight = total / (len(self.classes) * count)
            else:
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    config_path: str = "configs/config.yaml",
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        config_path: Path to configuration file
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = load_config(config_path)
    final_dir = config['data']['final_dir']
    batch_size = config['training']['batch_size']

    # Get transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms()

    # Create datasets
    train_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'train'),
        transform=train_transform
    )

    val_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'val'),
        transform=val_transform
    )

    test_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'test'),
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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


def main():
    """Test the dataset implementation."""
    print("Testing GarbageDataset...")

    config = load_config()
    final_dir = config['data']['final_dir']

    # Test train dataset
    train_transform = get_train_transforms(config)
    train_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'train'),
        transform=train_transform
    )

    print(f"\nTrain dataset:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Class counts: {train_dataset.get_class_counts()}")
    print(f"  Class weights: {train_dataset.get_class_weights()}")

    # Test single sample
    image, label = train_dataset[0]
    print(f"\n  Sample image shape: {image.shape}")
    print(f"  Sample label: {label} ({CLASSES[label]})")

    # Test val dataset
    val_transform = get_val_transforms()
    val_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'val'),
        transform=val_transform
    )

    print(f"\nVal dataset:")
    print(f"  Total samples: {len(val_dataset)}")
    print(f"  Class counts: {val_dataset.get_class_counts()}")

    # Test test dataset
    test_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'test'),
        transform=val_transform
    )

    print(f"\nTest dataset:")
    print(f"  Total samples: {len(test_dataset)}")
    print(f"  Class counts: {test_dataset.get_class_counts()}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders(num_workers=0)

    batch = next(iter(train_loader))
    images, labels = batch
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")

    print("\nDataset test complete!")


if __name__ == "__main__":
    main()
