import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score

# ==========================================
# 1. CONFIGURATION & CLASSES (Global Scope)
# ==========================================
# Ces définitions doivent rester en haut, accessibles aux workers

CONFIG = {
    'data_path': 'ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl',
    'img_size': 28,
    'batch_size': 64,
    'epochs': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'num_classes': 5,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
}

class RetinaSpecificPreprocess(object):
    def __call__(self, img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        g_min, g_max = g.min(), g.max()
        if g_max > g_min:
            g = (g - g_min) / (g_max - g_min) * 255.0
        g = g.astype(np.uint8)
        img_processed = np.stack([r, g, b], axis=2)
        return img_processed.astype(np.uint8)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RetinaDataset(Dataset):
    def __init__(self, images, labels, transform=None, specific_preprocess=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.specific_preprocess = specific_preprocess

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.specific_preprocess:
            img = self.specific_preprocess(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

class SimpleRetinaCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleRetinaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': epoch_loss, 'accuracy': acc, 'macro_recall': macro_recall,
        'macro_f1': macro_f1, 'per_class_recall': per_class_recall,
        'confusion_matrix': cm, 'all_labels': all_labels, 'all_preds': all_preds
    }

def load_data():
    if not os.path.exists(CONFIG['data_path']):
        print("⚠️ Fichier .pkl non trouvé. Génération de données factices...")
        N = 1080
        X = np.random.randint(0, 255, (N, 28, 28, 3), dtype=np.uint8)
        probs = np.array([486, 128, 206, 194, 66])
        probs = probs / probs.sum()
        y = np.random.choice(5, N, p=probs)
    else:
        with open(CONFIG['data_path'], 'rb') as f:
            data = pickle.load(f)
            X = data['images']
            y = data['labels']
    return X, y

# ==========================================
# 2. MAIN EXECUTION (PROTECTED)
# ==========================================

def main():
    # Tout le code exécutable doit être ici
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    print(f"Utilisation du device : {CONFIG['device']}")

    # Chargement
    X_all, y_all = load_data()
    
    # --- CORRECTION NUMPY ARRAY ---
    # On aplatit les labels immédiatement pour éviter les problèmes
    y_all = np.array(y_all).reshape(-1)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=CONFIG['seed']
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Datasets & Loaders
    preproc = RetinaSpecificPreprocess()
    train_dataset = RetinaDataset(X_train, y_train, transform=train_transforms, specific_preprocess=preproc)
    val_dataset = RetinaDataset(X_val, y_val, transform=val_transforms, specific_preprocess=preproc)

    # Note: num_workers=2 est sûr MAINTENANT qu'on est dans le bloc if __name__ == '__main__'
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # Modèle
    model = SimpleRetinaCNN(num_classes=CONFIG['num_classes']).to(CONFIG['device'])

    # --- CORRECTION POIDS (NUMPY) ---
    classes_present, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(classes_present, counts))
    
    total_samples = len(y_train)
    class_weights = []
    for i in range(CONFIG['num_classes']):
        count = class_counts.get(i, 0)
        weight = total_samples / (CONFIG['num_classes'] * max(count, 1))
        class_weights.append(weight)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(CONFIG['device'])
    print(f"Poids des classes : {np.round(class_weights, 2)}")

    # Optimisation
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Boucle d'entraînement
    best_recall = 0.0

    print("\nDébut de l'entraînement...")
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])
        
        scheduler.step(val_metrics['macro_recall'])
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"Recall: {val_metrics['macro_recall']:.4f}")
        
        if val_metrics['macro_recall'] > best_recall:
            best_recall = val_metrics['macro_recall']
            torch.save(model.state_dict(), 'best_model_recall.pth')

    print("\nEntraînement terminé. Résultats finaux sur best_model_recall.pth :")
    model.load_state_dict(torch.load('best_model_recall.pth', map_location=CONFIG['device'], weights_only=True))
    final_metrics = evaluate(model, val_loader, criterion, CONFIG['device'])
    
    print(classification_report(final_metrics['all_labels'], final_metrics['all_preds']))


# C'EST LA PARTIE LA PLUS IMPORTANTE :
if __name__ == '__main__':
    # Sur macOS, ceci empêche le crash multiprocessing
    main()