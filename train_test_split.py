##########################################################
# split_train_val_test.py — Split 70% / 20% / 10%
##########################################################
# 85291, 37406, 19582, 64037, 92154
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# ======== 1. Charger le dataset original ========
SEED = 0
TRAIN_PATH = "ift-3395-6390-kaggle-2-competition-fall-2025/train_data.pkl"

with open(TRAIN_PATH, "rb") as f:
    data_raw = pickle.load(f)

X = data_raw["images"]
y = data_raw["labels"].reshape(-1)

print("Dataset complet :", X.shape, y.shape)


# ======== 2. Split 1 : train (70%) + temp (30%) ========

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.20,          # 30% qui seront re-split ensuite
    stratify=y,
    random_state=SEED
)

print("Train 70% :", X_train.shape, y_train.shape)
print("Temp 30%  :", X_temp.shape, y_temp.shape)


# ======== 3. Split 2 : temp (30%) → val (20%) + test_holdout (10%) ========

# 20% val = 20/30 = 0.666...
# 10% test = 10/30 = 0.333...

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=1/2,           # 1/3 of 30% = 10% of dataset
    stratify=y_temp,
    random_state=SEED
)


print("Val 20%      :", X_val.shape, y_val.shape)
print("Test 10% (holdout) :", X_test.shape, y_test.shape)


# ======== 4. Sauvegarde ========

with open("train.pkl", "wb") as f:
    pickle.dump({"images": X_train, "labels": y_train}, f)

with open("val.pkl", "wb") as f:
    pickle.dump({"images": X_val, "labels": y_val}, f)

with open("test_holdout.pkl", "wb") as f:
    pickle.dump({"images": X_test, "labels": y_test}, f)


print("\n✓ Fichiers générés avec succès :")
print("  - train.pkl      (70%)")
print("  - val.pkl        (20%)")
print("  - test_holdout.pkl (10%)")
