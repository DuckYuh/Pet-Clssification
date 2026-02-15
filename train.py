import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OxfordPetDataset, get_train_transform, get_eval_transform
from models import SimpleCNN, build_resnet18


def get_args():
    parser = argparse.ArgumentParser(description="Train pet classification model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="cnn", 
        choices=["cnn", "resnet18"],
        help="Model architecture to use (default: cnn)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone (resnet only)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save model")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = False):
    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return build_resnet18(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    args = get_args()
    
    # Initialize wandb
    if args.wandb:
        import wandb
        wandb.init(
            project="pet-classification",
            name= "CNN_Baseline" if args.model == "cnn" else f"ResNet18 {'(Freeze)' if args.freeze_backbone else '(Finetuned)'}",
            group="comparison",
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "image_size": args.image_size,
                "freeze_backbone": args.freeze_backbone,
            }
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_dataset = OxfordPetDataset(
        root_dir="data/raw",
        split_file="data/processed/train.txt",
        transform=get_train_transform(args.image_size)
    )

    val_dataset = OxfordPetDataset(
        root_dir="data/raw",
        split_file="data/processed/val.txt",
        transform=get_eval_transform(args.image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Get number of classes
    with open("data/processed/class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    print(f"Model: {args.model}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Build model
    model = build_model(args.model, num_classes, args.freeze_backbone)
    model = model.to(device)
    
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
        else:
            # Legacy checkpoint (state_dict only)
            model.load_state_dict(checkpoint)
            best_val_acc = 0.0
        print(f"Loaded checkpoint from {args.resume}")
    else:
        best_val_acc = 0.0
        
    save_path = args.save_path or f"{args.model}_best.pth"

    criterion = nn.CrossEntropyLoss()
    
    # Only optimize trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_per_epoch": epoch_time,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
            }, save_path)
            print(f"Saved best model to {save_path}")

    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()