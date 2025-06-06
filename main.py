# src/main.py

import os
import sys

import torch
from torch.utils.data import DataLoader

from data_loader import FungiDataset, get_transforms
from model import FungiClassifier
from train import train_model
from eval import evaluate

def main():
    # 1) Select device (MPS on Apple Silicon if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 2) Define the split data directory
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/split_mind_funga")
    )

    # 3) Prepare training dataset & loader
    train_dataset = FungiDataset(
        root_dir=os.path.join(data_dir, "train"),
        transforms=get_transforms(train=True),
        class_to_idx=None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # 0 on Mac MPS or adjust as needed
    )
    train_class_to_idx = train_dataset.class_to_idx

    # 4) Prepare validation dataset & loader (reuse class_to_idx)
    val_dataset = FungiDataset(
        root_dir=os.path.join(data_dir, "val"),
        transforms=get_transforms(train=False),
        class_to_idx=train_class_to_idx
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # 5) Initialize model, loss, optimizer
    num_classes = len(train_dataset.classes)
    model     = FungiClassifier(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 6) Train
    print("\n=== Training Model ===")
    train_model(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10
    )

    # 7) Save the trained model weights
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "resnet18_fungi.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    # 8) Evaluate on the validation set
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("\n=== Evaluating on Val Set ===")
    evaluate(model, val_loader, device, val_dataset.classes)


if __name__ == "__main__":
    main()
