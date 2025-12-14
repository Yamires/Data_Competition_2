###############################################
# inference.py — Génération de soumission CSV
###############################################

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

###########################################
# 1. CONFIGURATION
###########################################

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Device utilisé : {DEVICE}")

BATCH_SIZE = 64
TEST_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"  # Chemin vers vos données de test
MODEL_PATH = "best_model.pth" # Chemin vers votre meilleur modèle sauvegardé

###########################################
# 2. PRÉPROCESSING (DOIT ÊTRE IDENTIQUE)
###########################################
def preprocess_retina(img_tensor):
    """
    Même logique que evaluate_holdout.py
    """
    r = img_tensor[0]
    b = img_tensor[2]
    eps = 1e-6

    # 1. Structure (Rouge)
    r_min, r_max = r.min(), r.max()
    r_n = (r - r_min) / (r_max - r_min + eps)

    # 2. Texture (Diff)
    diff = r - b
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + eps)

    # 3. Contraste (Gamma)
    contrast = torch.pow(r_n, 2.0)

    return torch.stack([r_n, diff_n, contrast], dim=0)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])

###########################################
# 3. DATASET DE TEST (SANS LABELS)
###########################################
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        img_pil = Image.fromarray(img_arr.astype("uint8"))
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        # On ne retourne que l'image (pas de label en test)
        return img_tensor

###########################################
# 4. ARCHITECTURE DU MODÈLE (SimpleNet)
###########################################
class SimpleNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2) 

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x)
        return self.fc(x)

###########################################
# 5. EXÉCUTION
###########################################

print(f"Chargement des données depuis {TEST_PATH}...")
with open(TEST_PATH, "rb") as f:
    test_data = pickle.load(f)

# Adaptation selon la structure de votre pickle (dictionnaire ou array direct)
if isinstance(test_data, dict):
    X_test = test_data["images"]
else:
    X_test = test_data

print(f"Nombre d'images à prédire : {len(X_test)}")

# Création du DataLoader
test_ds = TestDataset(X_test, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Chargement du modèle
print(f"Chargement du modèle depuis {MODEL_PATH}...")
model = SimpleNet(num_classes=5).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    print(f"ERREUR : Impossible de charger les poids. Vérifiez que {MODEL_PATH} existe.")
    raise e

model.eval()

# Boucle d'inférence
predictions = []
print("Démarrage de l'inférence...")

with torch.no_grad():
    for i, imgs in enumerate(test_loader):
        imgs = imgs.to(DEVICE)
        
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        
        if (i+1) % 10 == 0:
            print(f"Batch {i+1} traité...")

###########################################
# 6. SAUVEGARDE CSV
###########################################

# Création du DataFrame
# Attention : Les IDs Kaggle commencent souvent à 1, pas 0.
df = pd.DataFrame({
    "ID": np.arange(1, len(predictions) + 1), 
    "Label": predictions
})

output_filename = "IFT3395_YAPS_MSC_V58.csv"
df.to_csv(output_filename, index=False)

print(f"\n✅ Terminé ! Fichier généré : {output_filename}")
print(df.head())