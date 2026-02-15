"""
Data utilities for the Transfer Learning MVP.
Handles dataset download, transforms, and DataLoader creation.
"""

import os
import zipfile
from io import BytesIO

import requests
import torch
from torchvision import datasets, transforms

from config import (
    DATA_DIR,
    Hymenoptera_DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    RESIZE_SIZE,
    DATASET_URL,
    BATCH_SIZE,
    NUM_WORKERS,
)


def download_hymenoptera_data(dest_dir=None):
    """
    Download and extract the hymenoptera_data (ants vs bees) dataset.
    Returns the path to the extracted dataset directory.
    """
    dest_dir = dest_dir or DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)
    data_path = os.path.join(dest_dir, "hymenoptera_data")

    # If already extracted, skip download
    train_path = os.path.join(data_path, "train")
    if os.path.isdir(train_path):
        print(f"Dataset already found at {data_path}. Skipping download.")
        return data_path

    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content), "r") as zf:
        zf.extractall(dest_dir)
    print("Dataset downloaded and extracted.")
    return data_path


def get_transforms():
    """
    Return train and validation transforms as specified in the technical plan.
    Train: RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize.
    Val: Resize, CenterCrop, ToTensor, Normalize.
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }
    return data_transforms


def get_inference_transform():
    """Transform used for single-image inference (same as validation)."""
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def create_dataloaders(data_dir=None, batch_size=None, num_workers=None):
    """
    Create train and validation ImageFolder datasets and DataLoaders.
    data_dir: path to hymenoptera_data (must contain 'train' and 'val' subdirs).
    """
    data_dir = data_dir or Hymenoptera_DATA_DIR
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    num_workers = num_workers if num_workers is not None else NUM_WORKERS

    trans = get_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), trans[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=num_workers,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names
