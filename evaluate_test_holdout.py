###############################################
# evaluate_holdout_loop.py — Évaluation de 15 modèles
###############################################

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
BATCH_SIZE = 64
NUM_MODELS = 5  # Nombre de modèles à tester (best_model_1.pth à best_model_15.pth)

print(f"Device utilisé : {DEVICE}")

# ==========================================
# 2. PRÉPROCESSING (IDENTIQUE)
# ==========================================
def preprocess_retina(img_tensor):
    r = img_tensor[0]
    b = img_tensor[2]
    eps = 1e-6

    r_n = (r - r.min()) / (r.max() - r.min() + eps)
    diff = r - b
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + eps)
    contrast = torch.pow(r_n, 2.0)

    return torch.stack([r_n, diff_n, contrast], dim=0)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])

# ==========================================
# 3. DATASET
# ==========================================
class RetinaDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_arr = self.images[idx]
        img_pil = Image.fromarray(img_arr.astype("uint8"))
        return self.transform(img_pil), int(self.labels[idx])

# ==========================================
# 4. CHARGEMENT DONNÉES
# ==========================================
print("\nChargement du test_holdout...")
with open("test_holdout.pkl", "rb") as f:
    holdout_raw = pickle.load(f)

X_holdout = holdout_raw["images"]
y_holdout = holdout_raw["labels"].reshape(-1)
holdout_ds = RetinaDataset(X_holdout, y_holdout, transform=val_transform)
holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f"Taille du holdout : {len(X_holdout)}")

# ==========================================
# 5. MODÈLE (SimpleNet)
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1); self.bn1 = nn.BatchNorm2d(64); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1); self.bn2 = nn.BatchNorm2d(128); self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1); self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1); self.bn4 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.fc(self.gap(x))

# ==========================================
# 6. BOUCLE D'ÉVALUATION
# ==========================================
accuracies = []
bal_accuracies = []

# Pour l'Ensemble : on va sommer les probas de tous les modèles
ensemble_logits = torch.zeros(len(X_holdout), 5).to(DEVICE)

print("\n" + "="*50)
print(f"DÉMARRAGE ÉVALUATION SUR {NUM_MODELS} MODÈLES")
print("="*50)

model = SimpleNet(num_classes=5).to(DEVICE)

for i in range(1, NUM_MODELS + 1):
    model_path = f"best_model_{i}.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modèle introuvable : {model_path}")
        continue

    # Charger poids
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    current_preds = []
    current_targets = []
    
    # Inférence pour ce modèle
    batch_start = 0
    with torch.no_grad():
        for imgs, labels in holdout_loader:
            imgs = imgs.to(DEVICE)
            
            # Prediction individuelle
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            current_preds.extend(preds.cpu().numpy())
            current_targets.extend(labels.numpy())
            
            # Accumulation pour l'ensemble (Soft Voting)
            batch_size = imgs.size(0)
            ensemble_logits[batch_start : batch_start + batch_size] += probs
            batch_start += batch_size

    # Calcul scores individuels
    acc = accuracy_score(current_targets, current_preds)
    bal_acc = balanced_accuracy_score(current_targets, current_preds)
    
    accuracies.append(acc)
    bal_accuracies.append(bal_acc)
    
    print(f"Modèle {i:02d} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f}")

# ==========================================
# 7. RÉSULTATS GLOBAUX
# ==========================================
print("\n" + "-"*50)
print("RÉSUMÉ STATISTIQUE (INDIVIDUELS)")
print("-"*50)
print(f"Accuracy Moyenne  : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Bal. Acc Moyenne  : {np.mean(bal_accuracies):.4f} ± {np.std(bal_accuracies):.4f}")
print(f"Meilleur Bal. Acc : {np.max(bal_accuracies):.4f} (Modèle #{np.argmax(bal_accuracies)+1})")

# ==========================================
# 8. RÉSULTAT DE L'ENSEMBLE (FUSION)
# ==========================================
ensemble_preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy()
ens_acc = accuracy_score(y_holdout, ensemble_preds)
ens_bal_acc = balanced_accuracy_score(y_holdout, ensemble_preds)

print("\n" + "="*50)
print("🏆 RÉSULTAT DE L'ENSEMBLE (VOTE)")
print("="*50)
print(f"Ensemble Accuracy      : {ens_acc:.4f}")
print(f"Ensemble Balanced Acc  : {ens_bal_acc:.4f}")

print("\nClassification Report (Ensemble) :")
print(classification_report(y_holdout, ensemble_preds, zero_division=0))

print("Confusion Matrix (Ensemble) :")
print(confusion_matrix(y_holdout, ensemble_preds))