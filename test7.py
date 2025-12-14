#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import pickle
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# Dataset
# ============================================================

class RetinaFeatureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.astype(np.float32) / 255.0  # simple scaling
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (28,28,3)
        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img).permute(2, 0, 1)  # CHW

        if self.labels is None:
            return img

        return img, int(self.labels[idx])


# ============================================================
# MLP model (recommended for feature maps 28×28×3)
# ============================================================

class RetinalMLP(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        in_features = 28 * 28 * 3

        self.model = nn.Sequential(
            nn.Flatten(),               # -> 2352 features

            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0
    n = 0

    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    conf = confusion_matrix(all_labels, all_preds)

    return {
        "loss": loss,
        "accuracy": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": conf,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


# ============================================================
# Inference on test.pkl
# ============================================================

def run_inference(model, test_path, output_csv, device):
    print("\n=== Running inference on test.pkl ===")

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    test_imgs = test_data["images"].astype(np.float32) / 255.0
    N = test_imgs.shape[0]
    print(f"Loaded {N} test images.")

    # Dataset sans labels
    class TestDS(Dataset):
        def __init__(self, imgs):
            self.imgs = imgs
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            img = self.imgs[idx]
            img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
            return img

    test_loader = DataLoader(TestDS(test_imgs), batch_size=64, shuffle=False)

    model.eval()
    preds = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())

    # CSV output
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for i, label in enumerate(preds, start=1):
            writer.writerow([i, label])

    print(f"Saved predictions to: {output_csv}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    # paths
    train_path = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"   # <-- change if needed
    test_path  = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"    # <-- change if needed
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Load train.pkl
    # -------------------------
    with open(train_path, "rb") as f:
        data = pickle.load(f)

    images = data["images"]       # (N,28,28,3)
    labels = data["labels"]       # (N,)

    images = np.array(images)
    labels = np.array(labels).astype(np.int64)

    print(f"Loaded {images.shape[0]} training images.")

    # -------------------------
    # Train/val split
    # -------------------------

# Flatten si labels a une dimension en trop
    if labels.ndim > 1:
        labels = labels.reshape(-1)



    train_imgs, val_imgs, train_labels, val_labels = train_test_split(images, labels,
                         test_size=0.2,
                         stratify=labels,
                         random_state=42)

    print(f"Train: {train_imgs.shape[0]}, Val: {val_imgs.shape[0]}")

    train_labels = np.array(train_labels).reshape(-1).astype(np.int64)
    val_labels   = np.array(val_labels).reshape(-1).astype(np.int64)

    # -------------------------
    # Sampler for imbalance
    # -------------------------
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # -------------------------
    # Datasets & Loaders
    # -------------------------
    train_ds = RetinaFeatureDataset(train_imgs, train_labels)
    val_ds   = RetinaFeatureDataset(val_imgs,   val_labels)

    train_loader = DataLoader(train_ds, batch_size=64,
                              sampler=sampler, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    model = RetinalMLP(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_f1 = 0
    best_path = os.path.join(output_dir, "best_mlp_f1.pth")

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(1, 41):
        print(f"\n=== Epoch {epoch}/40 ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Macro Recall: {val_metrics['macro_recall']:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")

        scheduler.step(val_metrics["macro_recall"])

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            print("Saved new best model (F1).")

    # -------------------------
    # Final Evaluation
    # -------------------------
    print("\n=== Final evaluation with best model ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    final = evaluate(model, val_loader, criterion, device)

    print("\nClassification report:")
    print(classification_report(final["y_true"], final["y_pred"],
                                digits=4, zero_division=0))

    print("Confusion matrix:")
    print(final["confusion_matrix"])

    # -------------------------
    # Test inference
    # -------------------------
    run_inference(model, test_path, "submit.csv", device)


if __name__ == "__main__":
    main()
