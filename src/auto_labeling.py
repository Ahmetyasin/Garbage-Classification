"""
CLIP-based auto-labeling script for scraped images.
Uses zero-shot classification to automatically label images.
"""

import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from tqdm import tqdm
import yaml

# Class definitions
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Text prompts for CLIP zero-shot classification
TEXT_PROMPTS = [
    "a photo of cardboard",
    "a photo of a glass bottle",
    "a photo of metal can",
    "a photo of paper",
    "a photo of plastic",
    "a photo of trash"
]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Load CLIP model and processor.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, processor, device)
    """
    device = get_device()
    print(f"Loading CLIP model: {model_name}")
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    return model, processor, device


def classify_image(
    image_path: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    text_prompts: list = TEXT_PROMPTS
) -> tuple:
    """
    Classify a single image using CLIP zero-shot classification.

    Args:
        image_path: Path to the image file
        model: CLIP model
        processor: CLIP processor
        device: Device to run inference on
        text_prompts: List of text prompts for each class

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Process inputs
        inputs = processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Get predicted class
        probs_np = probs.cpu().numpy()[0]
        predicted_idx = probs_np.argmax()
        confidence = probs_np[predicted_idx]
        predicted_class = CLASSES[predicted_idx]

        return predicted_class, confidence, probs_np

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0, None


def get_image_files(directory: str) -> list:
    """Get all image files from a directory recursively."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = []

    dir_path = Path(directory)
    if not dir_path.exists():
        return image_files

    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))

    return image_files


def auto_label_images(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Auto-label all scraped images using CLIP.

    Args:
        config_path: Path to configuration file

    Returns:
        DataFrame with labeling results
    """
    config = load_config(config_path)
    scraped_dir = config['data']['scraped_dir']
    auto_labeled_dir = config['data']['auto_labeled_dir']
    model_name = config['auto_labeling']['model']
    confidence_threshold = config['auto_labeling']['confidence_threshold']

    # Load CLIP model
    model, processor, device = load_clip_model(model_name)

    # Create output directories
    for class_name in CLASSES:
        Path(auto_labeled_dir, class_name).mkdir(parents=True, exist_ok=True)

    # Get all image files from scraped directory
    image_files = get_image_files(scraped_dir)
    print(f"\nFound {len(image_files)} images to label")

    # Results storage
    results = []

    # Statistics
    accepted_per_class = {c: 0 for c in CLASSES}
    rejected_per_class = {c: 0 for c in CLASSES}
    total_accepted = 0
    total_rejected = 0

    print("\n" + "="*60)
    print("Starting Auto-Labeling with CLIP")
    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print("="*60 + "\n")

    for image_path in tqdm(image_files, desc="Labeling images"):
        # Get original folder (class from scraping)
        original_folder = Path(image_path).parent.name

        # Classify image
        predicted_class, confidence, probs = classify_image(
            image_path, model, processor, device
        )

        if predicted_class is None:
            continue

        # Record result
        result = {
            'filename': Path(image_path).name,
            'original_path': image_path,
            'original_folder': original_folder,
            'predicted_class': predicted_class,
            'confidence': confidence,
        }

        # Add probabilities for each class
        if probs is not None:
            for i, class_name in enumerate(CLASSES):
                result[f'prob_{class_name}'] = probs[i]

        results.append(result)

        # Check confidence threshold
        if confidence >= confidence_threshold:
            # Accept and copy image
            dest_path = Path(auto_labeled_dir) / predicted_class / Path(image_path).name

            # Handle filename conflicts
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = Path(auto_labeled_dir) / predicted_class / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(image_path, dest_path)
            accepted_per_class[predicted_class] += 1
            total_accepted += 1
        else:
            rejected_per_class[original_folder] += 1
            total_rejected += 1

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save CSV report
    report_path = Path(auto_labeled_dir) / "labeling_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nLabeling report saved to: {report_path}")

    # Print statistics
    print("\n" + "="*60)
    print("Auto-Labeling Statistics")
    print("="*60)

    print(f"\nAccepted (confidence >= {confidence_threshold:.0%}):")
    for class_name in CLASSES:
        print(f"  {class_name:12s}: {accepted_per_class[class_name]:4d}")
    print(f"  {'Total':12s}: {total_accepted:4d}")

    print(f"\nRejected (confidence < {confidence_threshold:.0%}):")
    for class_name in CLASSES:
        print(f"  {class_name:12s}: {rejected_per_class.get(class_name, 0):4d}")
    print(f"  {'Total':12s}: {total_rejected:4d}")

    acceptance_rate = total_accepted / (total_accepted + total_rejected) * 100 if (total_accepted + total_rejected) > 0 else 0
    print(f"\nOverall acceptance rate: {acceptance_rate:.1f}%")
    print("="*60)

    return df


def analyze_results(csv_path: str):
    """Analyze auto-labeling results from CSV file."""
    df = pd.read_csv(csv_path)

    print("\n" + "="*60)
    print("Auto-Labeling Analysis")
    print("="*60)

    # Confusion between original folder and predicted class
    print("\nOriginal Folder vs Predicted Class:")
    confusion = pd.crosstab(df['original_folder'], df['predicted_class'], margins=True)
    print(confusion)

    # Confidence distribution
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {df['confidence'].mean():.3f}")
    print(f"  Std:  {df['confidence'].std():.3f}")
    print(f"  Min:  {df['confidence'].min():.3f}")
    print(f"  Max:  {df['confidence'].max():.3f}")

    # Per-class confidence
    print("\nMean Confidence per Predicted Class:")
    for class_name in CLASSES:
        class_df = df[df['predicted_class'] == class_name]
        if len(class_df) > 0:
            print(f"  {class_name:12s}: {class_df['confidence'].mean():.3f} (n={len(class_df)})")


def main():
    parser = argparse.ArgumentParser(description="Auto-label scraped images using CLIP")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--analyze',
        type=str,
        default=None,
        help='Path to labeling report CSV to analyze (skip labeling)'
    )

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.analyze)
    else:
        df = auto_label_images(args.config)
        if len(df) > 0:
            analyze_results(Path(load_config(args.config)['data']['auto_labeled_dir']) / "labeling_report.csv")


if __name__ == "__main__":
    main()
