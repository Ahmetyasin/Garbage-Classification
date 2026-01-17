"""
Utility functions for the garbage classification project.
"""

import os
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Class definitions - maintain this order everywhere
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (MPS for M3 Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(directory: str) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_images_in_directory(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')) -> Dict[str, int]:
    """Count images per class in a directory structure."""
    counts = {}
    dir_path = Path(directory)

    if not dir_path.exists():
        return counts

    for class_dir in dir_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            count = sum(1 for f in class_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in extensions)
            counts[class_name] = count

    return counts


def get_image_paths(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')) -> List[Tuple[str, str]]:
    """Get all image paths with their class labels from a directory structure."""
    image_paths = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return image_paths

    for class_dir in dir_path.iterdir():
        if class_dir.is_dir() and class_dir.name in CLASSES:
            class_name = class_dir.name
            for img_file in class_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in extensions:
                    image_paths.append((str(img_file), class_name))

    return image_paths


def is_valid_image(image_path: str) -> bool:
    """Check if an image file is valid and can be opened."""
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Need to reopen after verify
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False


def get_timestamp() -> str:
    """Get current timestamp string for file naming."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_dataset_stats(directory: str, title: str = "Dataset Statistics") -> None:
    """Print statistics about images in a dataset directory."""
    counts = count_images_in_directory(directory)
    total = sum(counts.values())

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Directory: {directory}")
    print(f"{'-'*50}")

    for class_name in CLASSES:
        count = counts.get(class_name, 0)
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 2)
        print(f"{class_name:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print(f"{'-'*50}")
    print(f"{'Total':12s}: {total:5d}")
    print(f"{'='*50}\n")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
