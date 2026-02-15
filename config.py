"""
Configuration settings for the Transfer Learning MVP project.
Centralizes paths, hyperparameters, and dataset options.
"""

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# Directory where the project root lives (this file's parent)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Default directory for downloading and storing the dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Default path for the hymenoptera dataset after extraction
Hymenoptera_DATA_DIR = os.path.join(DATA_DIR, "hymenoptera_data")
# Default path to save the best model weights (feature extraction)
DEFAULT_MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "best_model.pth")
# Default path for the class names JSON (for API/CLI inference)
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.json")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# URL for the hymenoptera_data (ants vs bees) dataset
DATASET_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

# -----------------------------------------------------------------------------
# Data augmentation & preprocessing (ImageNet normalization)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Input size expected by ResNet
INPUT_SIZE = 224
RESIZE_SIZE = 256

# -----------------------------------------------------------------------------
# Training hyperparameters (MVP / Phase 1)
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # DataLoader workers; use 0 on Windows if you see multiprocessing errors

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
API_HOST = "0.0.0.0"
# Use PORT from environment (e.g. Render, Heroku) or default 5000
API_PORT = int(os.environ.get("PORT", 5000))
