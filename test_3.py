import os
import pickle
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# ================================================
# 1. CONFIG
# ================================================

DATA_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"
BATCH_SIZE = 64
NUM_CLASSES = 5
NUM_EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 3e-4
VAL_SIZE = 0.15
RANDOM_STATE = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    ("cuda" if torch.cuda.is_available() else "cpu")
)

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_STATE)

# ================================================
# 2. CHARGEMENT DES DONNÉES
# ================================================

def load_data_plk(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if "images" not in data or "labels" not in data:
        raise ValueError("Format de train.pkl incorrect")

    images = data["images"].astype(np.uint8)   # très important : uint8 ici
    labels = np.array(data["labels"], dtype=np.int64)

    return images, labels


images, labels = load_data_plk(DATA_PATH)
labels = labels.reshape(-1).astype(np.int64)

assert images.shape[1:] == (28, 28, 3)

print("\nImages:", images.shape)
print("Labels:", labels.shape)

# ================================================
# 3. TRANSFORMS + PREPROCESS
# ================================================

augment = transforms.Compose([
    
    #transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3),
    transforms.RandomAutocontrast(p=0.3),
    transforms.GaussianBlur(3, 0.3),
    transforms.RandomRotation(degrees=3),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees=0, shear=5),
])
"""
def preprocess_retina(img_tensor):
    r = img_tensor[0]
    b = img_tensor[2]

    r_n = (r - r.min()) / (r.max() - r.min() + 1e-6)
    diff = r - b
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    contrast = torch.pow(r_n, 2.0)

    return torch.stack([r_n, diff_n, contrast], dim=0)
"""
def preprocess_retina(img_tensor):
    R = img_tensor[0]
    G = img_tensor[1]
    B = img_tensor[2]

    # Normalisation canal par canal
    def normalize_channel(c):
        return (c - c.min()) / (c.max() - c.min() + 1e-6)

    Rn = normalize_channel(R)
    Gn = normalize_channel(G)
    Bn = normalize_channel(B)

    # Correction du canal vert si trop faible
    g_range = G.max() - G.min()
    if g_range < 0.15:  # seuil empirique
        # amplification progressive (pas brutale)
        gain = 0.3 / (g_range + 1e-6)   # gain adaptatif
        Gn = torch.clamp(Gn * gain, 0, 1)

    # Reconstruction RGB corrigé
    return torch.stack([Rn, Gn, Bn], dim=0)


train_transform = transforms.Compose([
    transforms.ToTensor(),  
    
    transforms.Lambda(preprocess_retina),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

# ================================================
# 4. DATASET
# ================================================

class RetinaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # uint8 (28,28,3)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, int(self.labels[idx])

# ================================================
# 5. TRAIN/VAL SPLIT
# ================================================

train_idx, val_idx = train_test_split(
    np.arange(len(images)),
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=labels
)

train_images, train_labels = images[train_idx], labels[train_idx]
val_images, val_labels = images[val_idx], labels[val_idx]

# ================================================
# 6. OVERSAMPLING (UNIQUEMENT TRAIN)
# ================================================

def oversample_class(images, labels, target_class, factor):
    new_imgs = []
    new_lbls = []

    idxs = np.where(labels == target_class)[0]

    for idx in idxs:
        img_uint8 = images[idx]

        for _ in range(factor - 1):
            pil = Image.fromarray(img_uint8)
            aug = augment(pil)
            new_imgs.append(np.array(aug))
            new_lbls.append(target_class)

    return new_imgs, new_lbls


oversample_factor = {0: 1, 1: 4, 2: 2, 3: 2, 4: 6}

all_new_imgs, all_new_lbls = [], []

for c, f in oversample_factor.items():
    if f > 1:
        imgs_new, lbls_new = oversample_class(train_images, train_labels, c, f)
        all_new_imgs.extend(imgs_new)
        all_new_lbls.extend(lbls_new)

if len(all_new_imgs) > 0:
    all_new_imgs = np.stack(all_new_imgs)
    all_new_lbls = np.array(all_new_lbls)

    train_images = np.concatenate([train_images, all_new_imgs], axis=0)
    train_labels = np.concatenate([train_labels, all_new_lbls], axis=0)

print("Train distribution:", Counter(train_labels.tolist()))

# ================================================
# 7. LOADERS
# ================================================

train_dataset = RetinaDataset(train_images, train_labels, train_transform)
val_dataset   = RetinaDataset(val_images, val_labels, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================================================
# 8. CNN
# ================================================

class TinyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
model = TinyCNN().to(DEVICE)

from collections import Counter
counter = Counter(train_labels.tolist())


weights = torch.tensor([1.0 / (counter[c] ** 0.5) for c in range(NUM_CLASSES)], dtype=torch.float32)
weights /= weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ================================================
# 9. TRAINING / EVAL LOOPS
# (identiques à ton code, je peux les nettoyer si tu veux)
# ================================================

from torchview import draw_graph

# ==========================
# 8. Boucles d'entraînement / évaluation
# ==========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_recall_macro = recall_score(all_targets, all_preds, average="macro")
    epoch_f1_macro = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_recall_macro, epoch_f1_macro


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_recall_macro = recall_score(all_targets, all_preds, average="macro")
    epoch_f1_macro = f1_score(all_targets, all_preds, average="macro")

    # détails supplémentaires
    class_recalls = recall_score(all_targets, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "recall_macro": epoch_recall_macro,
        "f1_macro": epoch_f1_macro,
        "class_recalls": class_recalls,
        "confusion_matrix": cm,
        "y_true": all_targets,
        "y_pred": all_preds,
    }


# ==========================
# 9. Entraînement complet
# ==========================

best_val_recall = 0.0
best_state_dict = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=4
)

# Variables pour suivre les meilleurs modèles
best_val_acc = 0.0
best_val_recall = 0.0
best_val_f1 = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_acc, train_recall, train_f1 = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )

    val_metrics = evaluate(model, val_loader, criterion, DEVICE)

    val_acc = val_metrics["acc"]
    val_recall = val_metrics["recall_macro"]
    val_f1 = val_metrics["f1_macro"]

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Recall(macro): {train_recall:.4f} | F1(macro): {train_f1:.4f}")
    print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_acc:.4f} | Recall(macro): {val_recall:.4f} | F1(macro): {val_f1:.4f}")

    # ===============================
    # 1️⃣ Sauvegarder meilleur ACC
    # ===============================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_1.pth")
        print(f"  Nouveau meilleur ACC: {best_val_acc:.4f} → modèle sauvegardé.")

    # ====================================
    # 2️⃣ Sauvegarder meilleur Recall Macro
    # ====================================
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), "best_model_1.pth")
        print(f" Nouveau meilleur Recall Macro: {best_val_recall:.4f} → modèle sauvegardé.")

    # ===============================
    # 3️⃣ Sauvegarder meilleur F1 Macro
    # ===============================
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model_1.pth")
        print(f"  Nouveau meilleur F1 Macro: {best_val_f1:.4f} → modèle sauvegardé.")

    # Scheduler
    scheduler.step(val_recall)


# ==========================
# 10. Rapport final sur la validation
# ==========================

final_metrics = evaluate(model, val_loader, criterion, DEVICE)
y_true = final_metrics["y_true"]
y_pred = final_metrics["y_pred"]

print("\n=== Résultats finaux sur le set de validation ===")
print(f"Loss: {final_metrics['loss']:.4f}")
print(f"Accuracy: {final_metrics['acc']:.4f}")
print(f"Macro Recall: {final_metrics['recall_macro']:.4f}")
print(f"Macro F1: {final_metrics['f1_macro']:.4f}")
print("\nRecall par classe:", final_metrics["class_recalls"])
print("\nMatrice de confusion:\n", final_metrics["confusion_matrix"])

print("\nClassification report détaillé :")
print(classification_report(y_true, y_pred, digits=4))

graph = draw_graph(
    TinyCNN(),
    input_size=(1, 3, 28, 28),
)

graph.visual_graph.render(filename="tinycnn_torchview", format="png")


# ==========================
# INFÉRENCE SUR TEST.PKL
# ==========================

TEST_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"

with open(TEST_PATH, "rb") as f:
    test_data = pickle.load(f)

test_images = test_data["images"].astype(np.uint8)
print("Test images shape:", test_images.shape)

# === mêmes transforms que validation ===
test_transform = transforms.Compose([
    # transforms.Resize((64, 64)),   # <-- ajouter si le training l’utilise
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

test_dataset = TestDataset(test_images, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# === Charger le meilleur modèle ===
MODEL_PATH = "best_model_recall.pth"  # ou acc / f1
model = TinyCNN(num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Inférence ===
all_preds = []

with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

all_preds = np.array(all_preds)
print("Predictions:", all_preds.shape)

# === Export CSV ===
df = pd.DataFrame({
    "ID": np.arange(len(all_preds)),
    "Label": all_preds
})

df = pd.DataFrame({
    "ID": np.arange(1, len(all_preds)+1),
    "Label": all_preds
})

df.to_csv("IFT3396_YAPS_MCS_V62.csv", index=False)
print("IFT3395_YAPS_MCS_V61.csv généré avec succès!")
