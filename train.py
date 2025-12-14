###############################################
# train.py — Entraînement + Validation
###############################################

import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

###########################################
# 1. CONFIG
###########################################

SEED = 0
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-4
EARLY_STOP_PATIENCE = 20

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(SEED)
print(f"Device utilisé : {DEVICE}\n")


###########################################
# 2. PRÉPROCESSING IMAGES
###########################################

def preprocess_retina(img_tensor):

    r = img_tensor[0]
    b = img_tensor[2]

    eps = 1e-6

    r_min, r_max = r.min(), r.max()
    r_n = (r - r_min) / (r_max - r_min + eps)


    diff = r - b
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + eps)

    contrast = torch.pow(r_n, 2.0) 
    
    return torch.stack([r_n, diff_n, contrast], dim=0)


###########################################
# 3. DATASET
###########################################

class RetinaDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, is_test=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_arr = self.images[idx]
        img_pil = Image.fromarray(img_arr.astype('uint8'))

        img_tensor = self.transform(img_pil)

        if self.is_test:
            return img_tensor

        label = int(self.labels[idx])
        return img_tensor, label


###########################################
# 4. MODELE CNN
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
        # Pas de pool ici
        
        x = F.relu(self.bn4(self.conv4(x)))
        # Pas de pool ici
        
        x = self.gap(x)
        return self.fc(x)


###########################################
# 5. CHARGER TRAIN_AUG + VAL
###########################################

print("Chargement des données...")

with open("train_aug.pkl", "rb") as f:
    train_raw = pickle.load(f)

with open("val.pkl", "rb") as f:
    val_raw = pickle.load(f)

X_train = train_raw["images"].astype(np.float32)
y_train = train_raw["labels"].reshape(-1)

X_val = val_raw["images"].astype(np.float32)
y_val = val_raw["labels"].reshape(-1)

print(f"Train_aug : {X_train.shape} | Val : {X_val.shape}\n")


###########################################
# 6. TRANSFORMATIONS
###########################################

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 0.25)),

    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(preprocess_retina),
    transforms.Normalize([0.5]*3, [0.25]*3)
])


###########################################
# 7. DATA LOADERS
###########################################

train_ds = RetinaDataset(X_train, y_train, transform=train_transform)
val_ds   = RetinaDataset(X_val,   y_val,   transform=val_transform)

# Weighted sampler
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
sample_weights = class_weights[y_train_tensor]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


###########################################
# 8. MODEL, LOSS, OPTIM, SCHEDULER
###########################################

model = SimpleNet(num_classes=5).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
)

###########################################
# 9. TRAINING LOOP AVEC EARLY STOPPING
###########################################

best_bal_acc = 0.0
best_val_loss = float("inf")
patience_counter = 0
MIN_DELTA = 1e-3  # amélioration minimale sur la loss pour reset la patience

for epoch in range(EPOCHS):

    #######################################
    # TRAIN
    #######################################
    model.train()
    train_loss, train_correct = 0.0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == lbls).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_ds)

    #######################################
    # VALIDATION
    #######################################
    model.eval()
    preds, targets = [], []
    val_loss = 0.0

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item()

            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(lbls.cpu().numpy())

    val_loss /= len(val_loader)
    bal_acc = balanced_accuracy_score(targets, preds)
    val_acc = accuracy_score(targets, preds)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"TrLoss={train_loss:.3f} | ValLoss={val_loss:.3f} | "
        f"TrAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | BalAcc={bal_acc:.3f}"
    )

    # Scheduler sur la loss (plus stable)
    scheduler.step(val_loss)

    #######################################
    # SAVE BEST MODEL (sur métrique cible)
    #######################################
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        torch.save(model.state_dict(), "best_model_1.pth")
        print("  ✓ Nouveau meilleur modèle (BalAcc) sauvegardé")

    #######################################
    # EARLY STOP (sur val_loss)
    #######################################
    # on considère que la loss doit VRAIMENT s'améliorer
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"⛔ Early stopping activé à l'epoch {epoch+1}")
            break

print("\n===== FIN DE L'ENTRAÎNEMENT =====")
print("Meilleure Balanced Accuracy :", best_bal_acc)


#######################################
# ÉVALUATION FINALE SUR LA VALIDATION
#######################################

# Charger le meilleur modèle sauvegardé
model.load_state_dict(torch.load("best_model_15.pth", map_location=DEVICE))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(lbls.cpu().numpy())

# Convertir en numpy arrays
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# === MÉTRIQUES ===
acc = accuracy_score(all_targets, all_preds)
bal_acc = balanced_accuracy_score(all_targets, all_preds)
report = classification_report(all_targets, all_preds, zero_division=0)
cm = confusion_matrix(all_targets, all_preds)

print("\n=========== MÉTRIQUES FINALES SUR VAL ===========")
print(f"Accuracy            : {acc:.4f}")
print(f"Balanced Accuracy   : {bal_acc:.4f}")

print("\nClassification Report :")
print(report)

print("\nConfusion Matrix :")
print(cm)
