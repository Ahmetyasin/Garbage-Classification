#!/usr/bin/env python3
"""
Main script to run the complete garbage classification pipeline.

Usage:
    python run_pipeline.py --wandb-key YOUR_API_KEY

This script runs all steps in order:
1. Data collection (web scraping)
2. Auto-labeling with CLIP
3. Dataset preparation
4. Model training with W&B logging
5. Model evaluation
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import set_seed, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete garbage classification pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--wandb-key',
        type=str,
        required=True,
        help='Weights & Biases API key'
    )
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip web scraping step (use existing scraped data)'
    )
    parser.add_argument(
        '--skip-labeling',
        action='store_true',
        help='Skip auto-labeling step (use existing labeled data)'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip dataset preparation step'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step (only evaluate)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs/models/best_model.pth',
        help='Path to model for evaluation (when skipping training)'
    )

    args = parser.parse_args()

    # Load config and set seed
    config = load_config(args.config)
    set_seed(config.get('seed', 42))

    print("="*70)
    print("GARBAGE CLASSIFICATION PIPELINE")
    print("="*70)

    # Step 1: Data Collection (Web Scraping)
    if not args.skip_scraping:
        print("\n" + "="*70)
        print("STEP 1: Data Collection (Web Scraping)")
        print("="*70)
        from data_collection import scrape_all_classes
        scrape_all_classes(args.config)
    else:
        print("\nSkipping web scraping step...")

    # Step 2: Auto-Labeling with CLIP
    if not args.skip_labeling:
        print("\n" + "="*70)
        print("STEP 2: Auto-Labeling with CLIP")
        print("="*70)
        from auto_labeling import auto_label_images
        auto_label_images(args.config)
    else:
        print("\nSkipping auto-labeling step...")

    # Step 3: Dataset Preparation
    if not args.skip_prepare:
        print("\n" + "="*70)
        print("STEP 3: Dataset Preparation")
        print("="*70)
        from prepare_dataset import prepare_dataset
        prepare_dataset(args.config)
    else:
        print("\nSkipping dataset preparation step...")

    # Step 4: Model Training
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 4: Model Training with W&B")
        print("="*70)
        from train import train
        model_path, _, _, _ = train(args.config, args.wandb_key)
    else:
        print("\nSkipping training step...")
        model_path = args.model_path

    # Step 5: Model Evaluation
    print("\n" + "="*70)
    print("STEP 5: Model Evaluation")
    print("="*70)
    from evaluate import evaluate
    metrics = evaluate(
        model_path=model_path,
        config_path=args.config,
        log_to_wandb=not args.skip_training
    )

    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.2f}%")
    print("\nOutputs:")
    print("  - Best model: outputs/models/best_model.pth")
    print("  - Confusion matrix: outputs/figures/confusion_matrix.png")
    print("  - Per-class F1: outputs/figures/per_class_f1.png")
    print("  - Sample predictions: outputs/figures/sample_*_predictions.png")
    print("  - Metrics JSON: outputs/figures/evaluation_metrics.json")
    print("\nCheck W&B dashboard for detailed training logs and metrics!")


if __name__ == "__main__":
    main()
