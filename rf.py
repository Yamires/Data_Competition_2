import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURATION SÉCURISÉE
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
SEED = 42

# Chemins CRITIQUES pour éviter les fuites
# On entraîne le RF sur les données augmentées (pour la robustesse)
TRAIN_PKL = "train_aug.pkl" 
# On valide le RF sur les données de validation pures (pour la vérité)
EVAL_PKL  = "val.pkl"
# On garde le chemin du modèle CNN
PTH_PATH  = "best_model_4.pth" 

print(f"Device : {DEVICE}")
print(f"Train sur : {TRAIN_PKL}")
print(f"Eval sur  : {EVAL_PKL}")

# ==========================================
# 2. PRÉPROCESSING (Toujours identique)
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])

# ==========================================
# 3. DATASET & MODÈLE CNN
# ==========================================
class RetinaDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx].astype('uint8'))
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, y

class SimpleNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1); self.bn1 = nn.BatchNorm2d(64); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1); self.bn2 = nn.BatchNorm2d(128); self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1); self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1); self.bn4 = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # --- CORRECTION ICI : On remet la vraie couche pour que le chargement marche ---
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        return self.fc(x)

# ==========================================
# 4. EXTRACTION DES FEATURES
# ==========================================
def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict): return data['images'], data['labels']
    return data[0], data[1]

X_train, y_train = load_data(TRAIN_PKL)
X_eval, y_eval = load_data(EVAL_PKL)

train_loader = DataLoader(RetinaDataset(X_train, y_train, transform), batch_size=BATCH_SIZE, shuffle=False)
eval_loader  = DataLoader(RetinaDataset(X_eval, y_eval, transform), batch_size=BATCH_SIZE, shuffle=False)

# Chargement du CNN
cnn = SimpleNet().to(DEVICE)
cnn.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
cnn.eval()

def get_embeddings(loader):
    embeddings = []
    targets = []
    print("Extraction en cours...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = cnn(x)
            out = out.view(out.size(0), -1)
            embeddings.append(out.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(embeddings), np.concatenate(targets)

print("--- Features Train (Augmenté) ---")
X_train_emb, y_train_emb = get_embeddings(train_loader)
print("--- Features Eval (Propre) ---")
X_eval_emb, y_eval_emb = get_embeddings(eval_loader)

# ==========================================
# 5. RANDOM FOREST & RÉSULTATS
# ==========================================
print("\nEntraînement Random Forest sur features augmentées...")
rf = RandomForestClassifier(
    n_estimators=1000,
    class_weight='balanced',
    n_jobs=-1,
    random_state=SEED
)
rf.fit(X_train_emb, y_train_emb)

print("Évaluation sur le set de validation...")
val_preds = rf.predict(X_eval_emb)

acc = accuracy_score(y_eval_emb, val_preds)
bal_acc = balanced_accuracy_score(y_eval_emb, val_preds)

print("\n" + "="*40)
print(f"RÉSULTATS HYBRIDE (CNN + RF)")
print("="*40)
print(f"Validation Accuracy : {acc:.4f}")
print(f"Validation Bal. Acc : {bal_acc:.4f}")
print("\nMatrice de Confusion :")
print(confusion_matrix(y_eval_emb, val_preds))
print("\nRapport :")
print(classification_report(y_eval_emb, val_preds))