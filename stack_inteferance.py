import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
TEST_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"

# Liste de tes 5 meilleurs modèles entraînés avec des seeds différents
MODEL_PATHS = [
    "best_model_1.pth","best_model_2.pth","best_model_3.pth","best_model_4.pth","best_model_5.pth",

]

print(f"Device : {DEVICE}")
print(f"Ensemble de {len(MODEL_PATHS)} modèles.")


def preprocess_retina(img_tensor):
    r = img_tensor[0]
    b = img_tensor[2]
    eps = 1e-6
    
    r_n = (r - r.min()) / (r.max() - r.min() + eps)
    diff = r - b
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + eps)
    contrast = torch.pow(r_n, 2.0)

    return torch.stack([r_n, diff_n, contrast], dim=0)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])

# ==========================================
# 3. DATASET & MODEL
# ==========================================
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_pil = Image.fromarray(self.images[idx].astype("uint8"))
        return self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)

class SimpleNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # ... (Copier exactement ton architecture SimpleNet ici) ...
        # Pour faire court, je reprends la structure des couches:
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
# 4. CHARGEMENT DES DONNÉES
# ==========================================
with open(TEST_PATH, "rb") as f:
    test_data = pickle.load(f)
X_test = test_data["images"] if isinstance(test_data, dict) else test_data
test_loader = DataLoader(TestDataset(X_test, transform=test_transform), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. CHARGEMENT DES 5 MODÈLES EN MÉMOIRE
# ==========================================
models = []
for path in MODEL_PATHS:
    try:
        m = SimpleNet(num_classes=5).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
        print(f"✓ Chargé : {path}")
    except Exception as e:
        print(f"Erreur sur {path}: {e}")

# ==========================================
# 6. INFÉRENCE PAR ENSEMBLE (SOFT VOTING)
# ==========================================
final_preds = []

print("Démarrage de l'ensemble...")
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        
        # On va stocker la somme des probabilités pour ce batch
        # Forme : [Batch_Size, 5 Classes]
        batch_probs_sum = torch.zeros(imgs.size(0), 5).to(DEVICE)
        
        for model in models:
            logits = model(imgs)
            # IMPORTANT : On transforme les logits en probabilités (0.0 à 1.0)
            probs = F.softmax(logits, dim=1) 
            batch_probs_sum += probs
            
        # On fait la moyenne (diviser par 5)
        avg_probs = batch_probs_sum / len(models)
        
        # On prend la classe qui a la plus haute probabilité moyenne
        preds = torch.argmax(avg_probs, dim=1)
        final_preds.extend(preds.cpu().numpy())

# ==========================================
# 7. SAUVEGARDE
# ==========================================
df = pd.DataFrame({"ID": np.arange(1, len(final_preds) + 1), "Label": final_preds})
df.to_csv("IFT3365_YAPS_MCS_V62.csv", index=False)
print("✅ Fichier 'submission_ensemble_5seeds.csv' généré avec succès.")