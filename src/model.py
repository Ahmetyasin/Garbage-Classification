"""
MobileNetV2 model definition for garbage classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from typing import Optional
import yaml

# Class definitions
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)


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


class GarbageClassifier(nn.Module):
    """
    MobileNetV2-based garbage classifier.

    Uses pretrained MobileNetV2 as backbone with modified classifier
    for 6-class garbage classification.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone layers
        """
        super(GarbageClassifier, self).__init__()

        # Load pretrained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace the classifier
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone layers except the classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(config_path: str = "configs/config.yaml") -> GarbageClassifier:
    """
    Create model from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized GarbageClassifier
    """
    config = load_config(config_path)
    training_config = config['training']

    model = GarbageClassifier(
        num_classes=training_config.get('num_classes', NUM_CLASSES),
        pretrained=training_config.get('pretrained', True),
        freeze_backbone=training_config.get('freeze_backbone', False)
    )

    return model


def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config_path: str = "configs/config.yaml"
) -> GarbageClassifier:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        config_path: Path to configuration file

    Returns:
        Loaded GarbageClassifier
    """
    if device is None:
        device = get_device()

    model = create_model(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def save_model(
    model: GarbageClassifier,
    path: str,
    epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    metrics: Optional[dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        path: Path to save checkpoint
        epoch: Current epoch (optional)
        optimizer: Optimizer state (optional)
        scheduler: Scheduler state (optional)
        metrics: Training metrics (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)


def main():
    """Test the model implementation."""
    print("Testing GarbageClassifier...")

    device = get_device()
    print(f"Using device: {device}")

    # Create model
    model = create_model()
    model = model.to(device)

    print(f"\nModel architecture:")
    print(f"  Backbone: MobileNetV2")
    print(f"  Classifier: Linear(1280, {NUM_CLASSES})")
    print(f"  Total parameters: {model.get_num_params():,}")
    print(f"  Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Test predictions
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1)

    print(f"\n  Sample predictions:")
    for i in range(min(batch_size, 3)):
        pred_class = CLASSES[predictions[i].item()]
        confidence = probs[i, predictions[i]].item()
        print(f"    Sample {i+1}: {pred_class} ({confidence:.2%})")

    print("\nModel test complete!")


if __name__ == "__main__":
    main()
