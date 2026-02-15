"""
Transfer Learning MVP - Main training and evaluation script.
Trains a ResNet18 feature extractor on a small dataset (e.g., hymenoptera_data),
evaluates on validation set, and saves the best model.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DEFAULT_MODEL_SAVE_PATH,
    CLASS_NAMES_PATH,
    NUM_EPOCHS,
    LEARNING_RATE,
    Hymenoptera_DATA_DIR,
)
from data_utils import download_hymenoptera_data, create_dataloaders
from model_utils import create_feature_extractor


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one training epoch; return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataloader; return loss, accuracy, and collect labels/preds."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc, all_labels, all_preds


def run_training(
    data_dir=None,
    model_save_path=None,
    class_names_path=None,
    epochs=None,
    lr=None,
    batch_size=None,
):
    """
    Full training pipeline: load data, create model, train, evaluate, save best model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure dataset is available
    if data_dir is None or not os.path.isdir(os.path.join(data_dir, "train")):
        data_dir = download_hymenoptera_data()
    else:
        data_dir = os.path.abspath(data_dir)

    dataloaders, dataset_sizes, class_names = create_dataloaders(data_dir=data_dir)
    num_classes = len(class_names)
    model_save_path = model_save_path or DEFAULT_MODEL_SAVE_PATH
    class_names_path = class_names_path or CLASS_NAMES_PATH
    epochs = epochs or NUM_EPOCHS
    lr = lr or LEARNING_RATE

    # Save class names for inference (API and CLI)
    os.makedirs(os.path.dirname(os.path.abspath(class_names_path)) or ".", exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {class_names_path}")

    # Model, criterion, optimizer, scheduler
    model = create_feature_extractor(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        scheduler.step()

        val_loss, val_acc, val_labels, val_preds = evaluate(
            model, dataloaders["val"], criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved to {model_save_path}")

    # Final evaluation report
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    _, _, val_labels, val_preds = evaluate(
        model, dataloaders["val"], criterion, device
    )
    print("\n" + "=" * 40)
    print("Classification Report (validation set):")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))
    print("=" * 40)
    print(f"Best validation accuracy: {best_acc:.4f}")

    return model, class_names, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning MVP (feature extraction)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to hymenoptera_data directory (default: auto-download)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=f"Path to save best model (default: {DEFAULT_MODEL_SAVE_PATH})",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default from config)")
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        model_save_path=args.save_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
