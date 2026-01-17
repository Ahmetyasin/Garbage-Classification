"""
Dataset preparation script: combines original and auto-labeled data,
then splits into train/val/test sets.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Class definitions
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_image_files(directory: str) -> list:
    """Get all image files from a directory structure organized by class."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = []

    dir_path = Path(directory)
    if not dir_path.exists():
        return image_files

    for class_name in CLASSES:
        class_dir = dir_path / class_name
        if class_dir.exists():
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append({
                        'path': str(file_path),
                        'class': class_name,
                        'source': directory
                    })

    return image_files


def combine_datasets(raw_dir: str, auto_labeled_dir: str) -> list:
    """
    Combine original and auto-labeled datasets.

    Args:
        raw_dir: Path to original Kaggle dataset
        auto_labeled_dir: Path to CLIP-labeled dataset

    Returns:
        List of all image files with metadata
    """
    all_images = []

    # Get images from original dataset
    print(f"Loading images from original dataset: {raw_dir}")
    raw_images = get_image_files(raw_dir)
    for img in raw_images:
        img['source_type'] = 'original'
    all_images.extend(raw_images)
    print(f"  Found {len(raw_images)} images")

    # Get images from auto-labeled dataset
    print(f"Loading images from auto-labeled dataset: {auto_labeled_dir}")
    auto_images = get_image_files(auto_labeled_dir)
    for img in auto_images:
        img['source_type'] = 'auto_labeled'
    all_images.extend(auto_images)
    print(f"  Found {len(auto_images)} images")

    print(f"\nTotal combined: {len(all_images)} images")

    return all_images


def stratified_split(
    images: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Perform stratified split maintaining class proportions.

    Args:
        images: List of image dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    random.seed(seed)

    # Group images by class
    class_images = defaultdict(list)
    for img in images:
        class_images[img['class']].append(img)

    train_images = []
    val_images = []
    test_images = []

    for class_name in CLASSES:
        class_list = class_images[class_name]
        if len(class_list) == 0:
            continue

        # Shuffle for randomness
        random.shuffle(class_list)

        # First split: train and remaining
        train_size = int(len(class_list) * train_ratio)
        val_size = int(len(class_list) * val_ratio)

        train = class_list[:train_size]
        remaining = class_list[train_size:]

        # Second split: val and test from remaining
        val = remaining[:val_size]
        test = remaining[val_size:]

        train_images.extend(train)
        val_images.extend(val)
        test_images.extend(test)

    return train_images, val_images, test_images


def copy_images_to_split(images: list, dest_dir: str, split_name: str) -> dict:
    """
    Copy images to the destination split directory.

    Args:
        images: List of image dictionaries
        dest_dir: Destination directory
        split_name: Name of the split (train/val/test)

    Returns:
        Dictionary with copy statistics
    """
    stats = {class_name: 0 for class_name in CLASSES}

    split_dir = Path(dest_dir) / split_name

    # Create class directories
    for class_name in CLASSES:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in tqdm(images, desc=f"Copying {split_name} images"):
        src_path = Path(img['path'])
        class_name = img['class']

        # Generate unique filename to avoid conflicts
        dest_filename = f"{img['source_type']}_{src_path.stem}{src_path.suffix}"
        dest_path = split_dir / class_name / dest_filename

        # Handle conflicts
        counter = 1
        while dest_path.exists():
            dest_filename = f"{img['source_type']}_{src_path.stem}_{counter}{src_path.suffix}"
            dest_path = split_dir / class_name / dest_filename
            counter += 1

        shutil.copy2(src_path, dest_path)
        stats[class_name] += 1

    return stats


def prepare_dataset(config_path: str = "configs/config.yaml"):
    """
    Main function to prepare the final dataset.

    Args:
        config_path: Path to configuration file
    """
    config = load_config(config_path)

    raw_dir = config['data']['raw_dir']
    auto_labeled_dir = config['data']['auto_labeled_dir']
    final_dir = config['data']['final_dir']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    seed = config.get('seed', 42)

    print("="*60)
    print("Dataset Preparation")
    print("="*60)

    # Combine datasets
    print("\n[1/3] Combining datasets...")
    all_images = combine_datasets(raw_dir, auto_labeled_dir)

    # Print class distribution before split
    print("\nClass distribution (combined):")
    class_counts = defaultdict(int)
    for img in all_images:
        class_counts[img['class']] += 1

    for class_name in CLASSES:
        print(f"  {class_name:12s}: {class_counts[class_name]:4d}")
    print(f"  {'Total':12s}: {len(all_images):4d}")

    # Stratified split
    print(f"\n[2/3] Performing stratified split ({train_split:.0%}/{val_split:.0%}/{test_split:.0%})...")
    train_images, val_images, test_images = stratified_split(
        all_images,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        seed=seed
    )

    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")

    # Clean destination directory
    final_path = Path(final_dir)
    if final_path.exists():
        print(f"\nCleaning existing directory: {final_dir}")
        shutil.rmtree(final_path)
    final_path.mkdir(parents=True, exist_ok=True)

    # Copy images to splits
    print("\n[3/3] Copying images to final directories...")
    train_stats = copy_images_to_split(train_images, final_dir, 'train')
    val_stats = copy_images_to_split(val_images, final_dir, 'val')
    test_stats = copy_images_to_split(test_images, final_dir, 'test')

    # Save split information for reproducibility
    split_info = {
        'seed': seed,
        'train_ratio': train_split,
        'val_ratio': val_split,
        'test_ratio': test_split,
        'train_count': len(train_images),
        'val_count': len(val_images),
        'test_count': len(test_images),
        'total_count': len(all_images),
        'class_distribution': {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        },
        'sources': {
            'raw_dir': raw_dir,
            'auto_labeled_dir': auto_labeled_dir
        }
    }

    split_info_path = final_path / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit information saved to: {split_info_path}")

    # Print final statistics
    print("\n" + "="*60)
    print("Final Dataset Statistics")
    print("="*60)

    print("\nTrain set:")
    for class_name in CLASSES:
        print(f"  {class_name:12s}: {train_stats[class_name]:4d}")
    print(f"  {'Total':12s}: {sum(train_stats.values()):4d}")

    print("\nValidation set:")
    for class_name in CLASSES:
        print(f"  {class_name:12s}: {val_stats[class_name]:4d}")
    print(f"  {'Total':12s}: {sum(val_stats.values()):4d}")

    print("\nTest set:")
    for class_name in CLASSES:
        print(f"  {class_name:12s}: {test_stats[class_name]:4d}")
    print(f"  {'Total':12s}: {sum(test_stats.values()):4d}")

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"Final dataset location: {final_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Prepare final dataset")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()
    prepare_dataset(args.config)


if __name__ == "__main__":
    main()
