import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Chemins vers les fichiers
train_path = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"
test_path  = "ift-3395-6390-kaggle-2-competition-fall-2025/test_data.pkl"

# Charger les données
with open(train_path, "rb") as f:
    train_data = pickle.load(f)
with open(test_path, "rb") as f:
    test_data = pickle.load(f)

X = train_data["images"]
y = train_data["labels"].reshape(-1)
X_test = test_data["images"]

# --- Split train/validation avant augmentation ---
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Rééchantillonnage simple pour les classes 1,2,3,4 (uniquement sur le train) ---
classes_to_augment = [1, 2, 3, 4]
X_extra, y_extra = [], []
for cls in classes_to_augment:
    idx = np.where(y_tr == cls)[0]
    X_extra.append(X_tr[idx])
    y_extra.append(y_tr[idx])

X_tr = np.concatenate([X_tr] + X_extra)
y_tr = np.concatenate([y_tr] + y_extra)

# --- Fonction pour extraire des features avec cercles concentriques ---
def extract_features(img_array, n_bins=8, n_circles=3):
    h, w, _ = img_array.shape
    cy, cx = h//2, w//2
    Y, X_coord = np.ogrid[:h, :w]
    radius = np.sqrt((X_coord - cx)**2 + (Y - cy)**2)
    max_radius = radius.max()
    
    features = []
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    features.extend([r.mean(), g.mean(), b.mean()])
    lum = (0.299*r + 0.587*g + 0.114*b).mean()
    features.append(lum)
    
    for i in range(n_circles):
        mask = (radius >= i*max_radius/n_circles) & (radius < (i+1)*max_radius/n_circles)
        for ch in [r, g, b]:
            vals = ch[mask]
            if len(vals) == 0:
                vals = np.array([0])
            features.extend([vals.mean(), vals.var(), vals.min(), vals.max()])
            hist, _ = np.histogram(vals, bins=n_bins, range=(0,255))
            features.extend(hist / (vals.size if vals.size>0 else 1))
    return np.array(features)

# --- Extraire features ---
X_tr_feat  = np.array([extract_features(img) for img in X_tr])
X_val_feat = np.array([extract_features(img) for img in X_val])
X_test_feat = np.array([extract_features(img) for img in X_test])

# --- RandomForest avec class_weight='balanced' ---
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_tr_feat, y_tr)

# --- Évaluer sur la validation ---
pred_val = clf.predict(X_val_feat)
print("=== Classification report sur la validation ===")
print(classification_report(y_val, pred_val, digits=3))

# --- Nombre de prédictions par classe sur validation ---
unique, counts = np.unique(pred_val, return_counts=True)
print("\nNombre de prédictions par classe sur validation :")
for u, c in zip(unique, counts):
    print(f"Classe {u}: {c} images")

# --- Prédire le test ---
pred_test = clf.predict(X_test_feat)

# Nombre de prédictions par classe sur test
unique_test, counts_test = np.unique(pred_test, return_counts=True)
print("\nNombre de prédictions par classe sur test :")
for u, c in zip(unique_test, counts_test):
    print(f"Classe {u}: {c} images")

# --- Créer le CSV pour soumission ---
results_test = [{"ID": idx, "Label": int(label)} for idx, label in enumerate(pred_test, start=1)]
df_test = pd.DataFrame(results_test)
df_test.to_csv("manual_classification_no_leak_test.csv", index=False)
print("\nCSV test créé avec succès !")
