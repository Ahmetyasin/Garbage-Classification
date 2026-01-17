# Garbage Classification Project

A deep learning project for automated garbage classification using MobileNetV2, featuring web scraping, CLIP-based auto-labeling, and comprehensive training pipeline.

## Overview

This project implements a 6-class garbage classification system (cardboard, glass, metal, paper, plastic, trash) using transfer learning with MobileNetV2. The pipeline includes data collection via web scraping, CLIP-based auto-labeling for additional training data, and automated training with W&B integration.

## Results

- **Test Accuracy**: 90.64%
- **Best Validation Accuracy**: 88.69% (epoch 19)
- **Training Epochs**: 24 (early stopping)

For detailed methodology and analysis, see [CS_544_Ahmet_Yasin_Aytar_Final_Report.pdf](CS_544_Ahmet_Yasin_Aytar_Final_Report.pdf)

## Project Structure

```
├── src/                        # Source code
│   ├── data_collection.py      # Web scraping script
│   ├── auto_labeling.py        # CLIP auto-labeling
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── dataset.py              # PyTorch Dataset class
│   ├── model.py                # MobileNetV2 model
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utility functions
├── configs/config.yaml         # Configuration file
├── data/final/split_info.json  # Dataset split information
├── report/                     # LaTeX report and figures
├── requirements.txt            # Python dependencies
├── run_pipeline.py             # Main pipeline script
└── CS_544_Ahmet_Yasin_Aytar_Final_Report.pdf
```

## Quick Start

### 1. Setup Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Kaggle Credentials
Create `~/.kaggle/kaggle.json` with your Kaggle API credentials.

### 3. Run Complete Pipeline
```bash
python run_pipeline.py --wandb-key YOUR_WANDB_API_KEY
```

### 4. Run Individual Steps
```bash
# Download base dataset
kaggle datasets download -d asdasdasasdas/garbage-classification -p data/raw --unzip

# Collect additional data via web scraping
python src/data_collection.py

# Auto-label scraped images using CLIP
python src/auto_labeling.py

# Prepare final dataset
python src/prepare_dataset.py

# Train model
python src/train.py --wandb-key YOUR_KEY

# Evaluate model
python src/evaluate.py --model outputs/models/best_model.pth
```

## Key Features

- **Transfer Learning**: MobileNetV2 pretrained on ImageNet
- **Data Augmentation**: Rotation, flipping, color jittering
- **CLIP Auto-Labeling**: Automated labeling of web-scraped images
- **W&B Integration**: Experiment tracking and visualization
- **Early Stopping**: Prevents overfitting with patience-based stopping

## Configuration

All hyperparameters are configurable via `configs/config.yaml`:
- Dataset splits (70% train, 15% val, 15% test)
- Training parameters (batch size, learning rate, epochs)
- Data augmentation settings
- Model architecture settings

## Notes

- Model checkpoint (~9MB) not included in repository
- Raw dataset images must be downloaded from Kaggle
- W&B logs: https://wandb.ai/aytarahmetyasin-bo-azi-i-niversitesi/garbage-classification

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- transformers (for CLIP)
- wandb
- See `requirements.txt` for complete list
