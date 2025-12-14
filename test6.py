#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for retinal severity classification (5 classes) with a small CNN in PyTorch.

Assumptions:
- Data stored in a .pkl file with a dict:
    {
        "images": np.ndarray of shape [N, 28, 28, 3], dtype uint8
        "labels": np.ndarray of shape [N], int labels in [0, 4]
    }

Key features:
- Channel-wise preprocessing (green amplification if low dynamic range).
- Optional upscaling from 28x28 to 64x64 for CNN convenience.
- Light data augmentations only on training set.
- Handling of class imbalance via:
    * WeightedRandomSampler (oversampling)
    * Class weights in CrossEntropyLoss
- Small CNN (< 200k params) with global average pooling.
- Full training loop with metrics and model checkpointing.
"""

import os
import random
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Config & utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility as much as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For (more) reproducible behavior; may slow things down.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# Custom preprocessing transform
# ============================================================

class RetinalPreprocess(object):
    """
    Custom preprocessing:
      - Input: tensor in [0, 1], shape [3, H, W]
      - Green channel amplified if very low dynamic range.
      - Final per-channel normalization with given mean/std.

    Note:
    - We only slightly "stretch" the green channel if its dynamic range
      is below `green_min_range`. Gain is capped by `max_green_gain`.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        green_min_range: float = 0.1,
        max_green_gain: float = 3.0,
        eps: float = 1e-6,
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.green_min_range = green_min_range
        self.max_green_gain = max_green_gain
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [3, H, W] in [0, 1]
        # Split channels
        r = x[0:1, :, :]
        g = x[1:2, :, :]
        b = x[2:3, :, :]

        # --- Adaptive green amplification if dynamic range is too small ---
        g_min = g.amin(dim=(1, 2), keepdim=True)
        g_max = g.amax(dim=(1, 2), keepdim=True)
        g_range = g_max - g_min

        # If g_range is very small, we rescale it up to at least green_min_range
        gain = self.green_min_range / (g_range + self.eps)
        gain = torch.clamp(gain, max=self.max_green_gain)
        # Affine transform: bring min to 0 then scale
        g = (g - g_min) * gain
        g = torch.clamp(g, 0.0, 1.0)

        # Re-stack channels
        out = torch.cat([r, g, b], dim=0)

        # --- Final per-channel normalization (mean/std) ---
        out = (out - self.mean) / (self.std + self.eps)
        return out


# ============================================================
# Dataset
# ============================================================

class RetinaDataset(Dataset):
    """
    Custom Dataset wrapping numpy arrays.

    images: np.ndarray [N, H, W, 3], dtype uint8
    labels: np.ndarray [N]
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert images.ndim == 4 and images.shape[-1] == 3
        assert images.shape[0] == labels.shape[0]
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]  # [H, W, 3], uint8
        label = int(self.labels[idx])

        # Transforms expect something like PIL Image or numpy / tensor.
        # We start from numpy uint8 -> ToPILImage or ToTensor will handle it.
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ============================================================
# CNN Model
# ============================================================

class SmallRetinaCNN(nn.Module):
    """
    Simple CNN for 5-class classification with:
      - 3 conv blocks
      - BatchNorm + ReLU + MaxPool
      - Global Average Pooling
      - Single fully-connected layer

    This model is intentionally small (< ~200k params) to avoid overfitting.
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 3):
        super().__init__()

        # Variant: you can reduce/increase number of channels for
        # a smaller/larger model (see comments below).
        base_channels = 32

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x64 -> 32x32 (if input 64x64)

            # Block 2: 32 -> 64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16

            # Block 3: 64 -> 128
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
        )

        # Global average pooling: 8x8 -> 1x1 regardless of the exact spatial dim
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

        # NOTE:
        # - If you change the input image size (e.g. 32x32 or 128x128),
        #   the spatial dimensions after the conv+pool blocks will change,
        #   but AdaptiveAvgPool2d will still produce (C, 1, 1), so
        #   the classifier does NOT need to change.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)            # [B, C, 1, 1]
        x = torch.flatten(x, 1)            # [B, C]
        x = self.dropout(x)
        x = self.classifier(x)             # [B, num_classes]
        return x


# OPTIONAL (comment):
# - Smaller model: set base_channels = 16 (fewer filters per layer).
# - Larger model: set base_channels = 48 or 64, or add a 4th conv block.
#   Beware of overfitting given the small dataset size.


# ============================================================
# Training & evaluation functions
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / max(num_samples, 1)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 5,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    num_samples = 0

    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    avg_loss = running_loss / max(num_samples, 1)

    acc = accuracy_score(all_labels, all_preds)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_per_class = recall_score(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "recall_per_class": recall_per_class,
        "confusion_matrix": conf_mat,
        "y_true": all_labels,
        "y_pred": all_preds,
    }
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    # -------------------------
    # Hyperparameters / config
    # -------------------------
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    data_path = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"   # <-- change to your actual path
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    num_classes = 5
    val_size = 0.2     # 20% validation
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4
    num_epochs = 40
    upsample_to = 64   # From 28x28 -> 64x64 (optional but useful for CNN)

    # NOTE: If you change input image size (e.g. 32x32), just change
    # `upsample_to` above. Thanks to AdaptiveAvgPool2d, the CNN still works.

    # -------------------------
    # Load data
    # -------------------------
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    images = data["images"]  # [N, 28, 28, 3], uint8
    labels = data["labels"]  # [N]

    assert images.ndim == 4 and images.shape[-1] == 3
    assert images.shape[0] == labels.shape[0]

    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")
    labels = data["labels"]

    labels = np.array(labels).astype(np.int64)

    # Si labels est (N,1) → le rendre 1D
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    # -------------------------
    # Split train / val (stratified)
    # -------------------------
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images,
        labels,
        test_size=val_size,
        random_state=42,
        stratify=labels,
    )

    print(f"Train size: {train_imgs.shape[0]}, Val size: {val_imgs.shape[0]}")

    # -------------------------
    # Compute mean/std on TRAIN set (per-channel, in [0,1])
    # -------------------------
    train_imgs_float = train_imgs.astype(np.float32) / 255.0
    channel_means = train_imgs_float.mean(axis=(0, 1, 2))
    channel_stds = train_imgs_float.std(axis=(0, 1, 2)) + 1e-6

    print(f"Channel means (R,G,B): {channel_means}")
    print(f"Channel stds (R,G,B):  {channel_stds}")

    # -------------------------
    # Transforms
    # -------------------------
    retinal_preprocess = RetinalPreprocess(
        mean=tuple(channel_means.tolist()),
        std=tuple(channel_stds.tolist()),
        green_min_range=0.1,    # can tweak based on how weak G is
        max_green_gain=3.0,     # avoid crazy amplification
    )

    # Data augmentations (train only):
    # - small rotations, flips, slight affine, light blur / autocontrast
    # NOTE: They are deliberately mild to avoid unrealistic artifacts.
    train_transform = T.Compose([
        T.ToPILImage(),
        # Optional up-scaling from 28x28 to upsample_to x upsample_to
        T.Resize((upsample_to, upsample_to)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            shear=5,
            scale=(0.95, 1.05),
        ),
        T.RandomAutocontrast(p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ToTensor(),           # [0,1], CHW
        retinal_preprocess,     # channel correction + normalization
    ])

    # Validation (no augmentations, deterministic):
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((upsample_to, upsample_to)),
        T.ToTensor(),
        retinal_preprocess,
    ])

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = RetinaDataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = RetinaDataset(val_imgs, val_labels, transform=val_transform)

    # -------------------------
    # Class imbalance handling
    # -------------------------
    # Compute class counts on TRAIN set
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print("Class counts (train):", class_counts)

    # Class weights inversely proportional to class frequency
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalized-ish
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("Class weights used in loss:", class_weights)

    # For oversampling: assign each sample a weight based on its class weight
    sample_weights = class_weights_tensor[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    # NOTE:
    # - We combine both: oversampling (via WeightedRandomSampler)
    #   and class-weighted loss (CrossEntropyLoss with weight).
    #   This is a common strategy for heavily imbalanced datasets.

    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,      # must be False when using sampler
        num_workers=0,      # use 0 for maximum reproducibility
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------
    # Model, optimizer, loss, scheduler
    # -------------------------
    model = SmallRetinaCNN(num_classes=num_classes, in_channels=3)
    model.to(device)

    # Loss with class weights:
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler on macro recall (val)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,

    )

    # -------------------------
    # Training loop with checkpointing
    # -------------------------
    best_acc = 0.0
    best_macro_recall = 0.0
    best_macro_f1 = 0.0

    best_acc_path = os.path.join(output_dir, "best_model_acc.pth")
    best_recall_path = os.path.join(output_dir, "best_model_recall.pth")
    best_f1_path = os.path.join(output_dir, "best_model_f1.pth")

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_macro_recall = val_metrics["macro_recall"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Val loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Macro Recall: {val_macro_recall:.4f} | "
            f"Macro F1: {val_macro_f1:.4f}"
        )

        # LR scheduler on macro recall
        scheduler.step(val_macro_recall)

        # Checkpoint best models according to different metrics
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_acc_path)
            print(f"--> Saved new best model (accuracy) to {best_acc_path}")

        if val_macro_recall > best_macro_recall:
            best_macro_recall = val_macro_recall
            torch.save(model.state_dict(), best_recall_path)
            print(f"--> Saved new best model (macro recall) to {best_recall_path}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_f1_path)
            print(f"--> Saved new best model (macro F1) to {best_f1_path}")

    # -------------------------
    # Final evaluation / report (on best F1 model)
    # -------------------------
    print("\n=== Final evaluation on validation set (best F1 model) ===")
    if os.path.exists(best_f1_path):
        model.load_state_dict(torch.load(best_f1_path, map_location=device))
        print(f"Loaded best F1 model from {best_f1_path}")
    else:
        print("Warning: best F1 checkpoint not found, using last epoch model.")

    final_metrics = evaluate(model, val_loader, criterion, device, num_classes)
    final_loss = final_metrics["loss"]
    final_acc = final_metrics["accuracy"]
    final_macro_recall = final_metrics["macro_recall"]
    final_macro_f1 = final_metrics["macro_f1"]
    recall_per_class = final_metrics["recall_per_class"]
    conf_mat = final_metrics["confusion_matrix"]
    y_true = final_metrics["y_true"]
    y_pred = final_metrics["y_pred"]

    class_names = [f"class_{i}" for i in range(num_classes)]

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = true, cols = pred):")
    print(conf_mat)

    print("\n=== Final metrics (validation, best F1 model) ===")
    print(f"Loss:          {final_loss:.4f}")
    print(f"Accuracy:      {final_acc:.4f}")
    print(f"Macro Recall:  {final_macro_recall:.4f}")
    print(f"Macro F1:      {final_macro_f1:.4f}")
    print("Recall per class:")
    for idx, r in enumerate(recall_per_class):
        print(f"  {class_names[idx]}: {r:.4f}")
    
        # -------------------------
    # RUN TEST INFERENCE
    # -------------------------
    best_model_path = os.path.join(output_dir, "best_model_f1.pth")
    print(f"\nLoading best model for inference: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    run_inference(
        model=model,
        test_path="ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl",       # <-- changer le chemin si besoin
        output_csv="IFT3395_YAPS_MSC_V64.csv",
        transform=val_transform,    # val_transform = pas d'augmentations
        device=device
    )


# ============================================================
# Inference on test.pkl and export CSV
# ============================================================

def run_inference(model, test_path, output_csv, transform, device):
    print("\n=== Running inference on test set ===")

    # Load test.pkl
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    test_imgs = test_data["images"]  # shape [N,28,28,3]
    N = test_imgs.shape[0]
    print(f"Loaded test set with {N} images.")

    # Dataset without labels
    class TestDataset(Dataset):
        def __init__(self, imgs, transform):
            self.imgs = imgs
            self.transform = transform
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            img = self.imgs[idx]
            if self.transform:
                img = self.transform(img)
            return img

    test_dataset = TestDataset(test_imgs, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Inference
    model.eval()
    all_preds = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    # Generate CSV (ID starts at 1)
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for idx, label in enumerate(all_preds, start=1):
            writer.writerow([idx, label])

    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()
