import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import OxfordPetDataset, get_eval_transform
from models import SimpleCNN, build_resnet18


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate pet classification model")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet18"], help="Model architecture (default: cnn)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth file)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    parser.add_argument("--show-plots", action="store_true", help="Show visualization plots")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int):
    """Build model based on name."""
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_predictions(dataset, model, device, n=8):
    model.eval()
    plt.figure(figsize=(12, 6))

    for i in range(n):
        img, label = dataset[i]

        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()

        plt.subplot(2, 4, i + 1)
        # Denormalize for visualization
        img_np = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        plt.imshow(img_np)
        plt.title(f"GT: {label} | Pred: {pred}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    val_dataset = OxfordPetDataset(
        root_dir="data/raw",
        split_file="data/processed/test.txt",
        transform=get_eval_transform(args.image_size)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Get number of classes
    with open("data/processed/class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(val_dataset)}")

    # Build and load model
    model = build_model(args.model, num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    all_preds = []
    all_labels = []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total

    print(f"\nTest Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    if args.show_plots:
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix - {args.model}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        # Sample predictions
        visualize_predictions(val_dataset, model, device, n=8)


if __name__ == "__main__":
    main()