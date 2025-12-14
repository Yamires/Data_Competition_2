import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
# ================================================
# 1. CONFIGURATION & DEVICE
# ================================================
DATA_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Utilisation du device : {DEVICE}")

# Pour la reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# ================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ================================================
def load_data_formatted(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    images = data["images"] # (N, 28, 28, 3) uint8
    labels = np.array(data["labels"], dtype=np.int64).reshape(-1)
    
    # 1. Normalisation (0-255 -> 0-1) et Conversion float32 (requis pour Sklearn/Torch)
    images = images.astype(np.float32) / 255.0
    
    # 2. Transposition pour PyTorch : (N, H, W, C) -> (N, C, H, W)
    # Scikit-Learn mange du Numpy, mais Skorch le convertira en Tensor PyTorch correctement
    images = np.transpose(images, (0, 3, 1, 2))
    
    return images, labels

print("Chargement des données...")
X, y = load_data_formatted(DATA_PATH)
print(f"X shape: {X.shape} (N, C, H, W)")
print(f"y shape: {y.shape}")

# Calcul des poids pour le déséquilibre des classes
class_counts = np.bincount(y)
# Poids inversement proportionnels à la fréquence
weights = 1.0 / class_counts
weights = weights / weights.sum()
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Poids des classes calculés : {weights}")

# ================================================
# 3. DÉFINITION DU MODÈLE (Avec hyperparamètres)
# ================================================
class TinyCNN(nn.Module):
    # On ajoute dropout_rate et hidden_size comme arguments pour qu'ils soient "tunables"
    def __init__(self, num_classes=5, dropout_rate=0.5, hidden_size=128):
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
            nn.Dropout(dropout_rate),         # Paramètre variable
            nn.Linear(128, hidden_size),      # Paramètre variable
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ================================================
# 4. CRÉATION DU WRAPPER SKORCH
# ================================================
# NeuralNetClassifier rend le modèle PyTorch compatible avec l'API .fit() de Sklearn

net = NeuralNetClassifier(
    module=TinyCNN,
    
    # Paramètres liés à la Loss et l'Optimiseur
    criterion=nn.CrossEntropyLoss,
    criterion__weight=weights_tensor, # On passe les poids ici
    optimizer=torch.optim.Adam,
    
    # Paramètres d'entraînement par défaut (seront écrasés par le RandomSearch si présents dans la grille)
    max_epochs=20,
    batch_size=64,
    device=DEVICE,
    
    # Gestion du learning rate et Early Stopping
    callbacks=[
        ('early_stop', EarlyStopping(patience=5, load_best=True)),
        ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', mode='min', monitor='valid_loss', patience=2))
    ],
    
    # Affichage
    verbose=0, # Mettre à 1 pour voir chaque époque, 0 pour garder la sortie propre
    
    # IMPORTANT : Skorch gère lui-même un split de validation interne (20% par défaut)
    # pour le monitoring (EarlyStopping).
    train_split=ValidSplit(0.2, stratified=True)
)

# ================================================
# 5. CONFIGURATION DU RANDOMIZED SEARCH
# ================================================

# Notez la syntaxe : 'module__nom_param' pour le modèle, 'optimizer__nom_param' pour l'optim
params = {
    'lr': [3e-4, 5e-4, 1e-4],                 # Learning Rate
    'batch_size': [64],                   # Taille de lot
    'module__dropout_rate': [0.2, 0.25, 0.3],  # Hyperparamètre du TinyCNN
    'module__hidden_size': [32, 64, 128],    # Hyperparamètre du TinyCNN
    'optimizer__weight_decay': [3e-4, 1e-3],  # Régularisation L2
    'max_epochs': [15, 20, 25]                    # Durée d'entraînement
}

rs = RandomizedSearchCV(
    estimator=net,
    param_distributions=params,
    n_iter=5,           # Nombre de combinaisons à tester (augmenter si vous avez du temps)
    cv=3,               # Validation croisée (3 folds)
    scoring='accuracy', # Métrique à maximiser
    verbose=2,          # Niveau de détail dans la console
    n_jobs=1,           # Toujours 1 quand on utilise le GPU avec PyTorch
    random_state=42
)

# ================================================
# 6. LANCEMENT DE LA RECHERCHE
# ================================================
print("Lancement de la recherche d'hyperparamètres...")
# Attention : X doit être float32 et y int64
rs.fit(X, y)

print("\n========================================")
print(f"Meilleur Score (Accuracy) : {rs.best_score_:.4f}")
print("Meilleurs Paramètres :")
for param, val in rs.best_params_.items():
    print(f"  - {param}: {val}")
print("========================================\n")

# ================================================
# 7. SAUVEGARDE ET UTILISATION DU MEILLEUR MODÈLE
# ================================================

# Récupérer le meilleur estimateur (c'est un objet NeuralNetClassifier entraîné)
best_net = rs.best_estimator_

# Sauvegarder les poids PyTorch du meilleur modèle
best_net.save_params(f_params='best_model_params.pt')
print("Meilleurs paramètres sauvegardés dans 'best_model_params.pt'")

# Sauvegarder l'objet complet (y compris l'historique d'entraînement)
with open('best_skorch_model.pkl', 'wb') as f:
    pickle.dump(best_net, f)

# --- Exemple d'inférence ---
# Si vous avez des données de test X_test (format N, C, H, W)
# preds = best_net.predict(X_test)