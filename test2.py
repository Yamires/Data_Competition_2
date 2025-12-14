import pickle
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 64
EPOCHS_PER_FOLD = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 12
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Choix du modèle: 'mobilenet_v3_small', 'efficientnet_b0', 'resnet18'
MODEL_NAME = 'efficientnet_b0'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(SEED)
print(f"Device: {DEVICE}")
print(f"Modèle: {MODEL_NAME}\n")

# ==========================================
# DATASET
# ==========================================
def enhance_channels(img_tensor):
    r = img_tensor[0]
    b = img_tensor[2]
    diff = (r - b) + 0.5
    diff = torch.clamp(diff, 0, 1)
    return torch.stack([r, diff, b], dim=0)

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
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        if self.is_test:
            return img_tensor
        
        label = self.labels[idx]
        if hasattr(label, 'item'):
            label = int(label.item())
        else:
            label = int(label)
            
        return img_tensor, label

# ==========================================
# MODÈLE AVEC TRANSFER LEARNING
# ==========================================
def create_model(model_name='mobilenet_v3_small', num_classes=5, pretrained=True):
    """
    Crée un modèle avec transfer learning
    """
    if model_name == 'mobilenet_v3_small':
        # MobileNetV3-Small (recommandé pour 28x28)
        model = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modifier la première conv pour accepter 28x28
        # (par défaut attend 224x224 mais fonctionne avec 28x28)
        
        # Remplacer le classifier
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet18':
        # ResNet18
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Modèle '{model_name}' non supporté")
    
    return model

# ==========================================
# FOCAL LOSS (pour classes difficiles)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# CHARGEMENT DONNÉES
# ==========================================
print("Chargement des données...")
TRAIN_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl" 
TEST_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"

with open(TRAIN_PATH, "rb") as f:
    train_data_raw = pickle.load(f)

X_all = train_data_raw["images"].astype(np.float32)
y_all = train_data_raw["labels"].reshape(-1)

# Stats
X_tmp = X_all / 255.0
IR_MEAN = X_tmp.mean(axis=(0, 1, 2)).tolist()
IR_STD = X_tmp.std(axis=(0, 1, 2)).tolist()
print(f"Stats -> Mean: {[f'{m:.3f}' for m in IR_MEAN]}, Std: {[f'{s:.3f}' for s in IR_STD]}")

unique, counts = np.unique(y_all, return_counts=True)
print(f"Distribution: {dict(zip(unique, counts))}\n")

# TRANSFORMS pour Transfer Learning
# Note: Les modèles pré-entraînés attendent des stats ImageNet, mais on peut utiliser les nôtres
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Légère upscale pour aider les features
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(20),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Lambda(enhance_channels),
    transforms.Normalize(IR_MEAN, IR_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(enhance_channels),
    transforms.Normalize(IR_MEAN, IR_STD)
])

# ==========================================
# CROSS-VALIDATION
# ==========================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_scores = []
fold_reports = []
model_paths = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    
    train_ds = RetinaDataset(X_train, y_train, transform=train_transform)
    val_ds = RetinaDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Créer le modèle
    model = create_model(MODEL_NAME, num_classes=5, pretrained=True).to(DEVICE)
    
    # Optimizer avec différentiel learning rate
    # Features pré-entraînées: LR plus bas
    # Classifier: LR normal
    if MODEL_NAME == 'mobilenet_v3_small':
        params = [
            {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
        ]
    elif MODEL_NAME == 'efficientnet_b0':
        params = [
            {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
        ]
    elif MODEL_NAME == 'resnet18':
        params = [
            {'params': list(model.parameters())[:-2], 'lr': LEARNING_RATE * 0.1},
            {'params': list(model.parameters())[-2:], 'lr': LEARNING_RATE}
        ]
    
    optimizer = optim.Adam(params, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-7
    )
    
    # Loss avec class weights boostés pour classe 3 et 4
    from sklearn.utils.class_weight import compute_class_weight
    loss_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    loss_weights = torch.FloatTensor(loss_weights).to(DEVICE)

    
    # Utilise Focal Loss pour mieux gérer les classes difficiles
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    
    # Training
    best_bal_acc = 0.0
    best_model_name = f"model_fold_{fold}_{MODEL_NAME}.pth"
    patience_counter = 0
    
    for epoch in range(EPOCHS_PER_FOLD):
        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)
        
        # VALIDATION
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(lbls.cpu().numpy())
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_bal_acc = balanced_accuracy_score(all_targets, all_preds)
        val_acc = accuracy_score(all_targets, all_preds)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Ep {epoch+1:02d} | TrL: {avg_train:.3f} TrAcc: {train_acc:.3f} | "
                  f"VL: {avg_val:.3f} VAcc: {val_acc:.3f} BalAcc: {val_bal_acc:.4f}")
        
        scheduler.step(val_bal_acc)
        
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_name)
            
            if epoch >= 5:
                report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
                c3_recall = report.get('3', {}).get('recall', 0)
                c4_recall = report.get('4', {}).get('recall', 0)
                print(f"  ✓ Best | BalAcc: {best_bal_acc:.4f} | C3: {c3_recall:.2f} C4: {c4_recall:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stop @ epoch {epoch+1}")
                break
    
    # Évaluation finale + matrice de confusion
    model.load_state_dict(torch.load(best_model_name, map_location=DEVICE))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())
    
    final_report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\n{'─'*60}")
    print(f"FOLD {fold+1} FINAL")
    print(f"{'─'*60}")
    print(f"Balanced Acc: {best_bal_acc:.4f} | Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
    
    print("\nPer-Class Recall:")
    for cls in range(5):
        recall = final_report.get(str(cls), {}).get('recall', 0)
        support = final_report.get(str(cls), {}).get('support', 0)
        print(f"  Class {cls}: {recall:.3f} (n={int(support)})")
    
    print("\nMatrice de confusion:")
    print("      ", " ".join([f"P{i}" for i in range(5)]))
    for i, row in enumerate(cm):
        print(f"True{i}:", "  ".join([f"{v:3d}" for v in row]))
    
    # Analyse classe 3
    if 3 in all_targets:
        class_3_mask = np.array(all_targets) == 3
        class_3_preds = np.array(all_preds)[class_3_mask]
        pred_dist = np.bincount(class_3_preds, minlength=5)
        print(f"\nClasse 3 prédite comme:")
        for i, count in enumerate(pred_dist):
            if count > 0:
                print(f"  → Classe {i}: {count} ({count/len(class_3_preds)*100:.1f}%)")
    print()
    
    fold_scores.append(best_bal_acc)
    fold_reports.append(final_report)
    model_paths.append(best_model_name)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# RÉSUMÉ CV
# ==========================================
print(f"{'='*60}")
print("RÉSUMÉ CROSS-VALIDATION")
print(f"{'='*60}")
print(f"Balanced Acc: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"Scores: {[f'{s:.4f}' for s in fold_scores]}\n")

avg_recalls = defaultdict(list)
for report in fold_reports:
    for cls in range(5):
        recall = report.get(str(cls), {}).get('recall', 0)
        avg_recalls[cls].append(recall)

print("Recall moyen par classe:")
for cls in range(5):
    recalls = avg_recalls[cls]
    print(f"  Classe {cls}: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

# ==========================================
# INFERENCE
# ==========================================
print(f"\n{'='*60}")
print("GÉNÉRATION SOUMISSION")
print(f"{'='*60}")

with open(TEST_PATH, "rb") as f:
    test_data = pickle.load(f)

test_ds = RetinaDataset(test_data['images'], labels=None, transform=val_transform, is_test=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

models_ensemble = []
for path in model_paths:
    m = create_model(MODEL_NAME, num_classes=5, pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    models_ensemble.append(m)

final_preds = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        ensemble_logits = torch.zeros(imgs.size(0), 5).to(DEVICE)
        
        for m in models_ensemble:
            # TTA: Normal + Flip H + Flip V
            out_norm = m(imgs)
            out_h = m(torch.flip(imgs, [3]))
            out_v = m(torch.flip(imgs, [2]))
            ensemble_logits += (out_norm + out_h + out_v) / 3.0
        
        ensemble_logits /= len(models_ensemble)
        _, predicted = torch.max(ensemble_logits, 1)
        final_preds.extend(predicted.cpu().numpy())

df = pd.DataFrame({"ID": np.arange(1, len(final_preds) + 1), "Label": final_preds})
filename = f"submission_{MODEL_NAME}_kfold.csv"
df.to_csv(filename, index=False)
print(f"✓ Fichier '{filename}' généré!")