"""
Evaluation script for garbage classification model.
Generates metrics, confusion matrix, and visualizations.
"""

import os
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import yaml
from PIL import Image

from dataset import create_dataloaders, GarbageDataset, get_val_transforms, CLASSES, NUM_CLASSES
from model import load_model, get_device
from utils import set_seed


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Get all predictions from the model.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run on

    Returns:
        Tuple of (all_labels, all_predictions, all_probabilities)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Getting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions) * 100

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, labels=range(NUM_CLASSES)
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(NUM_CLASSES))

    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=CLASSES,
        digits=4
    )

    metrics = {
        'accuracy': accuracy,
        'per_class': {
            CLASSES[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
            for i in range(NUM_CLASSES)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        square=True
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_f1(metrics: dict, output_path: str):
    """Plot per-class F1 scores."""
    f1_scores = [metrics['per_class'][c]['f1'] for c in CLASSES]

    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(CLASSES)))
    bars = plt.bar(CLASSES, f1_scores, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{f1:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Per-Class F1 Scores', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-class F1 plot saved to: {output_path}")


def plot_training_curves(wandb_run_path: str, output_dir: str):
    """
    Plot training curves from W&B run.
    Note: This requires the wandb history to be available.
    """
    # This function would fetch data from wandb if available
    # For now, we'll create placeholder functionality
    pass


def get_sample_predictions(
    model: nn.Module,
    dataset: GarbageDataset,
    device: torch.device,
    num_per_class: int = 3
) -> dict:
    """
    Get sample correct and incorrect predictions for visualization.

    Args:
        model: Model to use
        dataset: Dataset to sample from
        device: Device to run on
        num_per_class: Number of samples per class

    Returns:
        Dictionary with correct and incorrect samples
    """
    model.eval()

    correct_samples = {c: [] for c in CLASSES}
    incorrect_samples = {c: [] for c in CLASSES}

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0, pred].item()

            true_class = CLASSES[label]
            pred_class = CLASSES[pred]

            sample_info = {
                'idx': idx,
                'path': dataset.samples[idx][0],
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': prob
            }

            if pred == label and len(correct_samples[true_class]) < num_per_class:
                correct_samples[true_class].append(sample_info)
            elif pred != label and len(incorrect_samples[true_class]) < num_per_class:
                incorrect_samples[true_class].append(sample_info)

            # Check if we have enough samples
            correct_complete = all(len(v) >= num_per_class for v in correct_samples.values())
            incorrect_complete = all(len(v) >= num_per_class for v in incorrect_samples.values())

            if correct_complete and incorrect_complete:
                break

    return {'correct': correct_samples, 'incorrect': incorrect_samples}


def plot_sample_predictions(
    samples: dict,
    output_dir: str,
    title_prefix: str = ""
):
    """Plot grid of sample predictions."""
    for pred_type in ['correct', 'incorrect']:
        type_samples = samples[pred_type]

        # Create figure
        fig, axes = plt.subplots(
            len(CLASSES), 3,
            figsize=(12, len(CLASSES) * 3)
        )

        fig.suptitle(
            f'{title_prefix}{pred_type.capitalize()} Predictions',
            fontsize=16,
            y=1.02
        )

        for row, class_name in enumerate(CLASSES):
            class_samples = type_samples[class_name]

            for col in range(3):
                ax = axes[row, col]

                if col < len(class_samples):
                    sample = class_samples[col]

                    # Load and display image
                    try:
                        img = Image.open(sample['path']).convert('RGB')
                        ax.imshow(img)
                        ax.set_title(
                            f"True: {sample['true_class']}\n"
                            f"Pred: {sample['pred_class']} ({sample['confidence']:.1%})",
                            fontsize=8
                        )
                    except Exception as e:
                        ax.text(0.5, 0.5, 'Error loading', ha='center', va='center')

                ax.axis('off')

        plt.tight_layout()

        output_path = Path(output_dir) / f"sample_{pred_type}_predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Sample {pred_type} predictions saved to: {output_path}")


def evaluate(
    model_path: str,
    config_path: str = "configs/config.yaml",
    log_to_wandb: bool = True
):
    """
    Main evaluation function.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        log_to_wandb: Whether to log results to W&B
    """
    # Load config
    config = load_config(config_path)
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device, config_path)
    model.eval()

    # Create data loaders
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(config_path, num_workers=4)

    # Get predictions
    print("\nEvaluating on test set...")
    labels, predictions, probabilities = get_predictions(model, test_loader, device)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(labels, predictions)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Test Accuracy: {metrics['accuracy']:.2f}%")

    print("\nPer-Class Metrics:")
    print("-"*60)
    print(f"{'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-"*60)

    for class_name in CLASSES:
        m = metrics['per_class'][class_name]
        print(f"{class_name:12s} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10d}")

    print("-"*60)

    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        str(output_dir / "confusion_matrix.png"),
        title="Test Set Confusion Matrix"
    )

    # Plot per-class F1
    plot_per_class_f1(
        metrics,
        str(output_dir / "per_class_f1.png")
    )

    # Get and plot sample predictions
    print("\nGenerating sample prediction visualizations...")
    final_dir = config['data']['final_dir']
    test_dataset = GarbageDataset(
        root_dir=os.path.join(final_dir, 'test'),
        transform=get_val_transforms()
    )

    samples = get_sample_predictions(model, test_dataset, device, num_per_class=3)
    plot_sample_predictions(samples, str(output_dir), title_prefix="Test Set: ")

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            'accuracy': metrics['accuracy'],
            'per_class': metrics['per_class'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Log to W&B
    if log_to_wandb and wandb.run is not None:
        # Log metrics
        wandb.log({
            'test_accuracy': metrics['accuracy'],
            **{f"test_{c}_precision": metrics['per_class'][c]['precision'] for c in CLASSES},
            **{f"test_{c}_recall": metrics['per_class'][c]['recall'] for c in CLASSES},
            **{f"test_{c}_f1": metrics['per_class'][c]['f1'] for c in CLASSES}
        })

        # Log confusion matrix as W&B plot
        wandb.log({
            'confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predictions,
                class_names=CLASSES
            )
        })

        # Log figures as artifacts
        wandb.log({
            'confusion_matrix_plot': wandb.Image(str(output_dir / "confusion_matrix.png")),
            'per_class_f1_plot': wandb.Image(str(output_dir / "per_class_f1.png")),
            'sample_correct_predictions': wandb.Image(str(output_dir / "sample_correct_predictions.png")),
            'sample_incorrect_predictions': wandb.Image(str(output_dir / "sample_incorrect_predictions.png"))
        })

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Figures saved to: {output_dir}")
    print("="*60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate garbage classifier")
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/models/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging'
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        config_path=args.config,
        log_to_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
