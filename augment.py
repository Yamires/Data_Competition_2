import pickle 
import numpy as np
from PIL import Image, ImageEnhance
import random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# Charger les données
TRAIN_PATH = "train.pkl" 

with open(TRAIN_PATH, "rb") as f:
    train_data_raw = pickle.load(f)

X_all = train_data_raw["images"].astype(np.float32)
y_all = train_data_raw["labels"].reshape(-1)


# -----------------------
#  Fonction d'augmentation photométrique ONLY
# -----------------------
def augment(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    variations = []

    # --------------------------
    # Contraste agressif
    # --------------------------
    enhancer_contrast = ImageEnhance.Contrast(img)
    img_contrast = enhancer_contrast.enhance(np.random.uniform(0.5, 1.5))
    variations.append(np.array(img_contrast))

    # --------------------------
    # Saturation agressive
    # --------------------------
    enhancer_color = ImageEnhance.Color(img)
    img_saturation = enhancer_color.enhance(np.random.uniform(0.6, 1.4))
    variations.append(np.array(img_saturation))

    # --------------------------
    # Bruit gaussien fort
    # --------------------------
    sigma = np.random.uniform(5, 20)
    noise = np.random.normal(0, sigma, img_array.shape)
    img_noise = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    variations.append(img_noise)

    # --------------------------
    # Brightness jitter (modéré)
    # --------------------------
    enhancer_brightness = ImageEnhance.Brightness(img)
    img_bright = enhancer_brightness.enhance(np.random.uniform(0.7, 1.3))
    variations.append(np.array(img_bright))

    # --------------------------
    # Cutout léger
    # --------------------------
    arr = img_array.copy()
    h2, w2, c = arr.shape
    cut = np.random.randint(3, 7)
    y = np.random.randint(0, h2 - cut)
    x = np.random.randint(0, w2 - cut)
    arr[y:y+cut, x:x+cut] = 0
    variations.append(arr)

    # Sélection finale (1 à 3 augmentations)
    k = random.choice([1, 2, 3, 4])
    return random.sample(variations, k)


# -----------------------
#  Trouver classes minoritaires
# -----------------------
def minority_classes(labels):
    classes, counts = np.unique(labels, return_counts=True)
    mean_count = np.mean(counts)
    minority = classes[counts < mean_count]
    return list(minority)

minorities = minority_classes(y_all)
print("Classes minoritaires :", minorities)


# -----------------------
#  Génération des augmentations
# -----------------------
aug_data = []
aug_labels = []

for img, label in zip(X_all, y_all):
    if label in minorities:
        new_imgs = augment(img)
        aug_data.extend(new_imgs)
        aug_labels.extend([label] * len(new_imgs))


# -----------------------
#  Concat final + sauvegarde
# -----------------------
final_data = np.concatenate([X_all, np.array(aug_data)])
final_labels = np.concatenate([y_all, np.array(aug_labels)])

with open("train_aug.pkl", "wb") as f:
    pickle.dump({"images": final_data, "labels": final_labels}, f)

print("✓ Nouveau dataset photométriquement augmenté sauvegardé dans train_aug.pkl")
print("Final shape :", final_data.shape, final_labels.shape)
