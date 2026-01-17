"""
Training script for garbage classification with Weights & Biases logging.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import yaml

from dataset import create_dataloaders, CLASSES, NUM_CLASSES
from model import create_model, save_model, get_device
from utils import set_seed, AverageMeter, EarlyStopping, get_timestamp


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        loss_meter.update(loss.item(), images.size(0))

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    accuracy = 100. * correct / total
    return loss_meter.avg, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    accuracy = 100. * correct / total
    return loss_meter.avg, accuracy


def train(config_path: str = "configs/config.yaml", wandb_api_key: str = None):
    """
    Main training function.

    Args:
        config_path: Path to configuration file
        wandb_api_key: Weights & Biases API key
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config['training']
    wandb_config = config['wandb']

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize W&B
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    # Create run name with timestamp
    run_name = f"mobilenetv2_{get_timestamp()}"

    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config.get('entity'),
        name=run_name,
        config={
            'architecture': training_config['model'],
            'dataset': 'garbage-classification',
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate'],
            'weight_decay': training_config['weight_decay'],
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'early_stopping_patience': training_config['early_stopping_patience'],
            'augmentation': [
                f"RandomRotation({config['augmentation']['rotation_degrees']})",
                f"RandomHorizontalFlip({config['augmentation']['horizontal_flip_prob']})",
                f"ColorJitter(brightness={config['augmentation']['color_jitter']['brightness']}, "
                f"contrast={config['augmentation']['color_jitter']['contrast']}, "
                f"saturation={config['augmentation']['color_jitter']['saturation']})",
                "RandomResizedCrop(224)"
            ],
            'seed': seed,
            'num_classes': NUM_CLASSES,
            'classes': CLASSES
        }
    )

    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config_path, num_workers=4)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config_path)
    model = model.to(device)

    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=training_config['lr_scheduler_patience'],
        factor=0.1
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_config['early_stopping_patience'],
        mode='max'
    )

    # Create output directory
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(1, training_config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{training_config['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': current_lr
        })

        # Update scheduler
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            best_model_path = output_dir / "best_model.pth"
            save_model(
                model, str(best_model_path),
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={'val_acc': val_acc, 'val_loss': val_loss}
            )
            print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")

        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Save final model with timestamp
    final_model_path = output_dir / f"model_{run_name}.pth"
    save_model(
        model, str(final_model_path),
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={'val_acc': val_acc, 'val_loss': val_loss}
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*60}")

    # Log best model as artifact
    artifact = wandb.Artifact(
        name="garbage-classifier",
        type="model",
        description=f"MobileNetV2 garbage classifier (best val acc: {best_val_acc:.2f}%)"
    )
    artifact.add_file(str(best_model_path))
    wandb.log_artifact(artifact)

    # Return paths for evaluation
    return str(best_model_path), train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train garbage classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--wandb-key',
        type=str,
        default=None,
        help='Weights & Biases API key'
    )

    args = parser.parse_args()

    # Use environment variable if not provided
    wandb_key = args.wandb_key or os.environ.get('WANDB_API_KEY')

    train(args.config, wandb_key)


if __name__ == "__main__":
    main()
