# Garbage Classification Project - Submission Package

## Project Structure

```
garbage-classification-project/
├── src/                        # Source code
│   ├── data_collection.py      # Web scraping script
│   ├── auto_labeling.py        # CLIP auto-labeling script
│   ├── prepare_dataset.py      # Dataset preparation script
│   ├── dataset.py              # PyTorch Dataset class
│   ├── model.py                # MobileNetV2 model definition
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Utility functions
├── configs/
│   └── config.yaml             # Configuration file
├── outputs/
│   ├── figures/                # Generated visualizations
│   └── models/                 # Model checkpoints (not included due to size)
├── data/
│   └── final/
│       └── split_info.json     # Dataset split information
├── report/
│   ├── main.tex                # LaTeX report
│   └── figures/                # Report figures
├── requirements.txt            # Python dependencies
├── run_pipeline.py             # Main pipeline script
└── README.md                   # Full documentation
```

## How to Run

### 1. Setup Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Kaggle Credentials
Create `~/.kaggle/kaggle.json` with your credentials.

### 3. Run Complete Pipeline
```bash
python run_pipeline.py --wandb-key YOUR_WANDB_API_KEY
```

### 4. Or Run Individual Steps
```bash
# Download dataset
kaggle datasets download -d asdasdasasdas/garbage-classification -p data/raw --unzip

# Web scraping
python src/data_collection.py

# Auto-labeling
python src/auto_labeling.py

# Prepare dataset
python src/prepare_dataset.py

# Train model
python src/train.py --wandb-key YOUR_KEY

# Evaluate
python src/evaluate.py --model outputs/models/best_model.pth
```

## Notes

- Model checkpoint (`best_model.pth`) is not included due to file size (~9MB)
- Raw dataset images are not included (download from Kaggle)
- Virtual environment (`venv/`) is not included
- W&B logs are available at: https://wandb.ai/aytarahmetyasin-bo-azi-i-niversitesi/garbage-classification

## Results

- **Test Accuracy**: 90.64%
- **Best Validation Accuracy**: 88.69% (epoch 19)
- **Training Epochs**: 24 (early stopping)
