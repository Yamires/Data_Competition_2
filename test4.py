"""
Script complet PyTorch pour classification de sévérité sur images de rétine 28x28x3.

Architecture:
- Prétraitement adapté aux spécificités des canaux RGB (vert faible, bleu bruité)
- Augmentation de données modérée pour limiter l'overfitting
- CNN léger (~100k params) avec GlobalAvgPool pour réduire les paramètres
- Gestion du déséquilibre par class weights + légère augmentation de sampling
- Métriques de validation: accuracy, macro recall, macro F1
- Sauvegarde automatique des meilleurs modèles selon chaque métrique
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    accuracy_score
)
from collections import Counter
import copy

# ============================================================================
# GESTION DE LA REPRODUCTIBILITÉ
# ============================================================================
def set_seed(seed=42):
    """
    Fixe tous les seeds pour assurer la reproductibilité complète.
    
    Note: Sur GPU, certaines opérations restent non-déterministes pour des raisons
    de performance. Pour un déterminisme complet (au prix de la vitesse), 
    décommenter les lignes torch.backends.cudnn ci-dessous.
    """
    # Python random
    import random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Pour multi-GPU
    
    # PyTorch backends
    # Pour un déterminisme complet (plus lent) :
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Pour un bon compromis (reproductibilité + performance) :
    torch.backends.cudnn.benchmark = True  # Active pour vitesse
    torch.use_deterministic_algorithms(False)  # Permet certaines optimisations
    
    # Variables d'environnement pour opérations déterministes
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Pour certaines ops CUDA
    
    print(f"Seed fixé à {seed} pour reproductibilité")


# Appeler au tout début du script, avant CONFIGURATION
SEED = 42
set_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Device
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
print(f"Using device: {device}")

# Hyperparamètres

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Régularisation L2
NUM_EPOCHS = 150
PATIENCE = 20  # Early stopping patience
INPUT_SIZE = 32  # Upscale de 28x28 à 32x32 pour meilleures convolutions
NUM_CLASSES = 5

# Chemins
DATA_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"  # À ajuster selon votre fichier
BEST_MODEL_ACC = "best_model_acc.pth"
BEST_MODEL_RECALL = "best_model_recall.pth"
BEST_MODEL_F1 = "best_model_f1.pth"


# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    images = data["images"].astype(np.uint8)
    labels = np.array(data["labels"], dtype=np.int64)

    # Fix critique : labels doivent être 1D
    labels = labels.reshape(-1)

    print(f"Distribution des classes: {Counter(labels)}")
    return images, labels



def split_data(images, labels, val_ratio=0.15, seed=42):
    """Split train/val stratifié"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_ratio, 
        stratify=labels, random_state=seed
    )
    
    print(f"Train: {len(X_train)} images, Val: {len(X_val)} images")
    print(f"Train distribution: {Counter(y_train)}")
    print(f"Val distribution: {Counter(y_val)}")
    
    return X_train, X_val, y_train, y_val


# ============================================================================
# 2. PRÉTRAITEMENT ET AUGMENTATION
# ============================================================================
class AdaptiveChannelNormalization:
    """
    Normalisation adaptative des canaux RGB:
    - Canal vert: amplification si dynamique trop faible
    - Canal bleu: léger lissage pour réduire le bruit
    - Canal rouge: normalisation standard
    """
    def __init__(self, green_boost_threshold=0.1):
        self.green_boost_threshold = green_boost_threshold
    
    def __call__(self, img):
        """
        Input: Tensor [C, H, W] en float [0, 1]
        Output: Tensor [C, H, W] normalisé
        """
        # Séparation des canaux
        r, g, b = img[0], img[1], img[2]
        
        # Canal VERT: amplification si faible dynamique
        g_std = g.std()
        if g_std < self.green_boost_threshold:
            # Amplification adaptative: normalisation + boost de contraste
            g_min, g_max = g.min(), g.max()
            if g_max > g_min:
                g = (g - g_min) / (g_max - g_min)  # Normalisation [0, 1]
                g = torch.clamp(g * 1.5, 0, 1)  # Boost modéré
        
        # Canal BLEU: léger lissage gaussien pour réduire le bruit
        # On applique un petit blur via une convolution simple
        b_smooth = self._smooth_channel(b.unsqueeze(0).unsqueeze(0))
        b = b_smooth.squeeze()
        
        # Reconstruction
        img_corrected = torch.stack([r, g, b], dim=0)
        
        # Normalisation finale canal par canal
        # (permet à chaque canal de contribuer équitablement)
        for c in range(3):
            channel = img_corrected[c]
            mean, std = channel.mean(), channel.std()
            if std > 1e-6:
                img_corrected[c] = (channel - mean) / std
        
        return img_corrected
    
    def _smooth_channel(self, channel):
        """Applique un léger lissage gaussien"""
        # Kernel gaussien 3x3 simple
        kernel = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        
        # Padding et convolution
        padded = F.pad(channel, (1, 1, 1, 1), mode='reflect')
        smoothed = F.conv2d(padded, kernel)
        
        return smoothed


# Transforms pour TRAIN (avec augmentation)
train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize(INPUT_SIZE),  # Upscale 28→32
    
    # Augmentations géométriques modérées
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),  # Rotations légères
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    
    # Augmentations de couleur/contraste légères
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
    
    T.ToTensor(),  # Conversion en [C, H, W] float [0, 1]
    AdaptiveChannelNormalization(green_boost_threshold=0.1),
])

# Transforms pour VALIDATION (sans augmentation)
val_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
    AdaptiveChannelNormalization(green_boost_threshold=0.1),
])


# ============================================================================
# 3. DATASET PERSONNALISÉ
# ============================================================================
class RetinaDataset(Dataset):
    """Dataset pour images de rétine avec transforms"""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: ndarray [N, 28, 28, 3] uint8
            labels: ndarray [N] int
            transform: torchvision transforms
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # [28, 28, 3] uint8
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# 4. GESTION DU DÉSÉQUILIBRE
# ============================================================================
def create_weighted_sampler(labels):
    """
    Crée un WeightedRandomSampler pour oversampling des classes minoritaires.
    Stratégie: donner plus de poids aux classes rares, mais pas trop pour éviter
    l'overfitting sur quelques exemples.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Calcul des poids inversement proportionnels aux fréquences
    class_weights = {cls: total_samples / (num_classes * count) 
                     for cls, count in class_counts.items()}
    
    # Atténuation des poids pour éviter un oversampling trop agressif
    # On prend la racine carrée pour adoucir les poids
    class_weights = {cls: w**0.5 for cls, w in class_weights.items()}
    
    # Attribution des poids à chaque sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"Class weights pour sampler: {class_weights}")
    
    return sampler


def compute_class_weights(labels, num_classes):
    """
    Calcule les class weights pour nn.CrossEntropyLoss.
    Stratégie combinée: on utilise à la fois le sampler ET les class weights
    dans la loss pour une gestion plus équilibrée du déséquilibre.
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Weights inversement proportionnels
    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)  # Éviter division par 0
        weights[cls] = total_samples / (num_classes * count)
    
    # Normalisation douce (racine carrée)
    weights = weights**0.5
    weights = weights / weights.sum() * num_classes  # Re-normalisation
    
    print(f"Class weights pour loss: {weights.numpy()}")
    
    return weights


# ============================================================================
# 5. MODÈLE CNN
# ============================================================================
class RetinaCNN(nn.Module):
    """
    CNN léger pour classification de rétine.
    
    Architecture:
    - 3 blocs Conv -> BatchNorm -> ReLU -> MaxPool
    - GlobalAvgPool pour réduire les paramètres
    - Dropout pour régularisation
    - ~100k paramètres
    
    Variantes possibles (commentées):
    - Plus petit: réduire les filtres (32, 64, 128) -> ~50k params
    - Plus grand: ajouter un 4ème bloc + plus de filtres -> ~200k params
    
    Ajustement pour tailles d'entrée:
    - 28x28 -> INPUT_SIZE=28: fonctionne mais convolutions moins efficaces
    - 32x32 -> INPUT_SIZE=32: bon compromis (actuel)
    - 64x64 -> INPUT_SIZE=64: ajouter un bloc Conv ou réduire le stride
    """
    
    def __init__(self, num_classes=5, input_size=32, dropout=0.4):
        super(RetinaCNN, self).__init__()
        
        # Bloc 1: 3 -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32->16 ou 28->14
        
        # Bloc 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16->8 ou 14->7
        
        # Bloc 3: 128 -> 256 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8->4 ou 7->3
        
        # Global Average Pooling (remplace Flatten + FC large)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # -> [batch, 256, 1, 1]
        
        # Dropout pour régularisation
        self.dropout = nn.Dropout(dropout)
        
        # Classifier final
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Bloc 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Bloc 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_avg_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)     # [batch, 256]
        
        # Dropout + classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Variante PLUS PETITE (~50k params) - décommenter si besoin
"""
class RetinaCNNSmall(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4):
        super(RetinaCNNSmall, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
"""


# Variante PLUS GRANDE (~200k params) - décommenter si besoin
"""
class RetinaCNNLarge(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super(RetinaCNNLarge, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 4ème bloc supplémentaire
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
"""


def count_parameters(model):
    """Compte le nombre de paramètres du modèle"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return total, trainable


# ============================================================================
# 6. FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Entraîne le modèle pour une epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, num_classes=5):
    """
    Évalue le modèle sur un loader.
    
    Returns:
        loss: float
        accuracy: float (%)
        macro_recall: float
        macro_f1: float
        per_class_recall: list[float]
        confusion_mat: ndarray
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métriques
    total = len(all_labels)
    loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Recall par classe
    per_class_recall = recall_score(all_labels, all_preds, average=None, 
                                     labels=list(range(num_classes)), zero_division=0)
    
    # Matrice de confusion
    confusion_mat = confusion_matrix(all_labels, all_preds, 
                                     labels=list(range(num_classes)))
    
    return loss, accuracy, macro_recall, macro_f1, per_class_recall, confusion_mat


# ============================================================================
# 7. BOUCLE D'ENTRAÎNEMENT COMPLÈTE
# ============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs, num_classes):
    """
    Entraîne le modèle avec early stopping et sauvegarde des meilleurs modèles.
    """
    best_acc = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_model_acc_state = None
    best_model_recall_state = None
    best_model_f1_state = None
    patience_counter = 0
    
    print("\n" + "="*70)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*70)
    
    for epoch in range(num_epochs):
        # Entraînement
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, 
                                                  criterion, device)
        
        # Validation
        val_loss, val_acc, val_recall, val_f1, val_per_class_recall, _ = \
            evaluate(model, val_loader, criterion, device, num_classes)
        
        # Scheduler step (sur macro recall)
        scheduler.step(val_recall)
        
        # Affichage
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
              f"Recall: {val_recall:.4f} F1: {val_f1:.4f}")
        
        # Sauvegarde du meilleur modèle selon accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_acc_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  → Nouveau meilleur Accuracy: {best_acc:.2f}%")
        
        # Sauvegarde du meilleur modèle selon macro recall
        if val_recall > best_recall:
            best_recall = val_recall
            best_model_recall_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  → Nouveau meilleur Macro Recall: {best_recall:.4f}")
        
        # Sauvegarde du meilleur modèle selon macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_f1_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  → Nouveau meilleur Macro F1: {best_f1:.4f}")
        
        # Early stopping
        if val_recall <= best_recall and val_f1 <= best_f1 and val_acc <= best_acc:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping après {epoch+1} epochs (patience={PATIENCE})")
            break
    
    # Sauvegarde des modèles
    if best_model_acc_state:
        torch.save(best_model_acc_state, BEST_MODEL_ACC)
        print(f"\nModèle sauvegardé: {BEST_MODEL_ACC} (Acc: {best_acc:.2f}%)")
    
    if best_model_recall_state:
        torch.save(best_model_recall_state, BEST_MODEL_RECALL)
        print(f"Modèle sauvegardé: {BEST_MODEL_RECALL} (Recall: {best_recall:.4f})")
    
    if best_model_f1_state:
        torch.save(best_model_f1_state, BEST_MODEL_F1)
        print(f"Modèle sauvegardé: {BEST_MODEL_F1} (F1: {best_f1:.4f})")
    
    return best_model_recall_state  # On retourne le meilleur selon recall


# ============================================================================
# 8. RAPPORT FINAL
# ============================================================================
def print_final_report(model, val_loader, criterion, device, num_classes):
    """Affiche un rapport complet sur le set de validation"""
    print("\n" + "="*70)
    print("RAPPORT FINAL SUR LE SET DE VALIDATION")
    print("="*70)
    
    # Évaluation
    val_loss, val_acc, val_recall, val_f1, val_per_class_recall, conf_mat = \
        evaluate(model, val_loader, criterion, device, num_classes)
    
    # Métriques globales
    print(f"\nMÉTRIQUES GLOBALES:")
    print(f"  Loss:          {val_loss:.4f}")
    print(f"  Accuracy:      {val_acc:.2f}%")
    print(f"  Macro Recall:  {val_recall:.4f}")
    print(f"  Macro F1:      {val_f1:.4f}")
    
    # Recall par classe
    print(f"\nRECALL PAR CLASSE:")
    for i, recall in enumerate(val_per_class_recall):
        print(f"  Classe {i}: {recall:.4f}")
    
    # Matrice de confusion
    print(f"\nMATRICE DE CONFUSION:")
    print(conf_mat)
    
    # Classification report sklearn
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print(f"\nCLASSIFICATION REPORT (sklearn):")
    print(classification_report(all_labels, all_preds, 
                                target_names=[f"Classe {i}" for i in range(num_classes)],
                                zero_division=0))
    
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================
def main():
    # 1. Chargement des données
    print("Chargement des données...")
    images, labels = load_data(DATA_PATH)
    
    # 2. Split train/val
    X_train, X_val, y_train, y_val = split_data(images, labels)
    
    # 3. Création des datasets
    train_dataset = RetinaDataset(X_train, y_train, transform=train_transforms)
    val_dataset = RetinaDataset(X_val, y_val, transform=val_transforms)
    
    # 4. Gestion du déséquilibre
    print("\nGestion du déséquilibre des classes...")
    train_sampler = create_weighted_sampler(y_train)
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    
    # 5. DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler,  # Utilise le WeightedRandomSampler
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 6. Modèle
    print("\nCréation du modèle...")
    model = RetinaCNN(num_classes=NUM_CLASSES, input_size=INPUT_SIZE, dropout=0.4)
    model = model.to(device)
    count_parameters(model)
    
    # 7. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # 8. Entraînement
    best_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler, device, NUM_EPOCHS, NUM_CLASSES
    )
    
    # 9. Rapport final avec le meilleur modèle (selon macro recall)
    model.load_state_dict(best_state)
    print_final_report(model, val_loader, criterion, device, NUM_CLASSES)


if __name__ == "__main__":
    main()