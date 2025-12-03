def extract_features_full(img):
    """
    Version COMPLÈTE: 337 features optimisées
    Pour rétinopathie diabétique sur images 28x28x3
    """
    features = []
    
    # Convert to float
    img = img.astype(np.float32)
    
    # Grayscale
    gray = img.mean(axis=2)
    

    for c in range(3):
        channel = img[:, :, c]
        # Percentiles groupés pour performance
        percs = np.percentile(channel, [10, 25, 50, 75, 90])
        features.extend([
            channel.mean(),
            channel.std(),
            percs[0],  # p10
            percs[1],  # p25
            percs[2],  # p50 (médiane)
            percs[3],  # p75
            percs[4],  # p90
        ])

    gray_percs = np.percentile(gray, [10, 25, 50, 75, 90])
    features.extend([
        gray.mean(),
        gray.std(),
        gray.min(),
        gray.max(),
        gray_percs[0],  # p10
        gray_percs[1],  # p25
        gray_percs[2],
        gray_percs[3],  # p75
        gray_percs[4],  # p90
    ])

    H, W = gray.shape
    h2, w2 = H//2, W//2
    
    quadrants = [
        gray[:h2, :w2],   # Top-left
        gray[:h2, w2:],   # Top-right
        gray[h2:, :w2],   # Bottom-left
        gray[h2:, w2:]    # Bottom-right
    ]
    
    for quad in quadrants:
        features.extend([
            quad.mean(),
            quad.std(),
            quad.min(),
            quad.max()
        ])

    features.append(gray.max() - gray.min())
    
    features.append(gray.std() / (gray.mean() + 1e-8))
    
    hist_entr, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    hist_entr = hist_entr / (hist_entr.sum() + 1e-8)
    entropy = -np.sum(hist_entr * np.log2(hist_entr + 1e-8))
    features.append(entropy)
    
    skew = ((gray - gray.mean())**3).mean() / (gray.std()**3 + 1e-8)
    features.append(skew)
    

    means = img.mean(axis=(0, 1))  
    stds = img.std(axis=(0, 1))    
    
    features.extend([
        means[0] / (means[1] + 1e-8),  # R/G ratio
        means[0] / (means[2] + 1e-8),  # R/B ratio
        means[1] / (means[2] + 1e-8),  # G/B ratio
        stds[0] / (stds[1] + 1e-8),    # R_std/G_std
        stds[0] / (stds[2] + 1e-8),    # R_std/B_std
        stds[1] / (stds[2] + 1e-8),    # G_std/B_std
    ])

    H, W = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    lbp_img = np.zeros((H, W), dtype=np.uint8)
    
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for idx, (dy, dx) in enumerate(offsets):
        neigh = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        lbp_img |= ((neigh >= gray).astype(np.uint8) << idx)
    
    hist_lbp, _ = np.histogram(lbp_img, bins=256, range=(0, 256))
    features.extend(hist_lbp.tolist())
 
    hist_intensity, _ = np.histogram(gray, bins=16, range=(0, 256))
    features.extend(hist_intensity.tolist())

    return np.array(features, dtype=np.float32)



import numpy as np

def extract_features(img):
    """
    Version légère: ~100 features au lieu de 337
    Garde les plus importantes
    """
    features = []
    img = img.astype(np.float32)
    gray = img.mean(axis=2)
    
    # 1. RGB stats (18)
    for c in range(3):
        channel = img[:, :, c]
        features.append(channel.mean())
        features.append(channel.std())
        features.append(np.percentile(channel, 10))
        features.append(np.percentile(channel, 50))
        features.append(np.percentile(channel, 90))
        features.append(channel.max())
    
    # 2. Grayscale (6)
    features.append(gray.mean())
    features.append(gray.std())
    features.append(np.percentile(gray, 25))
    features.append(np.percentile(gray, 75))
    features.append(gray.min())
    features.append(gray.max())
    
    # 3. Edges (3)
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    features.extend([gx, gy, np.sqrt(gx**2 + gy**2)])
    
    # 4. Quadrants (8)
    H, W = gray.shape
    h2, w2 = H//2, W//2
    for quad in [gray[:h2, :w2], gray[:h2, w2:], gray[h2:, :w2], gray[h2:, w2:]]:
        features.append(quad.mean())
        features.append(quad.std())
    
    # 5. Color ratios (3)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    features.append(R.mean() / (G.mean() + 1e-8))
    features.append(R.mean() / (B.mean() + 1e-8))
    features.append(G.mean() / (B.mean() + 1e-8))
    
    # 6. LBP (64 bins au lieu de 256)
    def lbp(gray):
        H, W = gray.shape
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        lbp_img = np.zeros((H, W), dtype=np.uint8)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for idx, (dy, dx) in enumerate(offsets):
            neigh = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
            bit = (neigh >= gray).astype(np.uint8)
            lbp_img |= (bit << idx)
        return lbp_img
    
    lbp_img = lbp(gray)
    hist_lbp = np.histogram(lbp_img, bins=64, range=(0, 256))[0]
    features.extend(hist_lbp.tolist())
    
    
    return np.array(features, dtype=np.float32)


import numpy as np
import itertools

def polynomial_expansion_degree3(X):
    N, d = X.shape
    features = []

    # Degree 1
    features.append(X)

    # Degree 2
    for i, j in itertools.combinations_with_replacement(range(d), 2):
        features.append((X[:, i] * X[:, j]).reshape(N, 1))
    
    # Degree 3
    for i, j, k in itertools.combinations_with_replacement(range(d), 3):
        features.append((X[:, i] * X[:, j] * X[:, k]).reshape(N, 1))

    return np.hstack(features)


class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, reg=0.0, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.W = 0.01 * np.random.randn(input_dim, num_classes).astype(np.float32)
        self.reg = reg  # L2
        self.b = np.zeros(num_classes)
        self.seed = seed

    def _softmax(self, scores):
        # scores: (N, K)
        scores = scores - scores.max(axis=1, keepdims=True) 
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    def loss(self, X, y, sample_weights=None):
        N = X.shape[0]

        scores = X @ self.W + self.b
        probs = self._softmax(scores)

        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-12)

        if sample_weights is None:
            sample_weights = np.ones(N)

        loss = np.sum(sample_weights * correct_logprobs) / N
        loss += 0.5 * self.reg * np.sum(self.W**2)

        return loss, probs
    
    def grad(self, X, y, probs, sample_weights=None):
        N = X.shape[0]

        if sample_weights is None:
            sample_weights = np.ones(N)

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores *= sample_weights[:, None]
        dscores /= N

        dW = X.T @ dscores + self.reg * self.W
        db = dscores.sum(axis=0)
        return dW, db
            

    def fit(self, X, y, lr=1e-4, n_steps=1000, sample_weights=None, verbose=True):
        losses = []
        for step in range(n_steps):
            
            loss, probs = self.loss(X, y, sample_weights)
            dW, db = self.grad(X, y, probs, sample_weights)

            self.W -= lr * dW
            self.b -= lr * db

            losses.append(loss)

            if verbose and step % 100 == 0:
                print(f"Step {step}, loss = {loss:.4f}")

        return losses


    def predict(self, X):
        """Retourne la classe prédite pour chaque échantillon de X."""
        # Calculer les scores bruts
        scores = X @ self.W + self.b
        
        # Obtenir les probabilités via Softmax
        probs = self._softmax(scores)
        
        # La prédiction est la classe avec la probabilité maximale
        return np.argmax(probs, axis=1)

    def predict_proba(self, X): # Ajout optionnel pour le post-traitement
        """Retourne les probabilités pour chaque classe."""
        scores = X @ self.W + self.b
        return self._softmax(scores)

    import numpy as np

def random_flip(img):
    return np.fliplr(img)

def random_brightness(img):
    factor = 0.5 + np.random.rand()  # between 0.5 and 1.5
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def random_noise(img):
    noise = np.random.normal(0, 10, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def random_shift(img, max_shift=2):
    """Shift image by a few pixels using NumPy roll."""
    dx = np.random.randint(-max_shift, max_shift+1)
    dy = np.random.randint(-max_shift, max_shift+1)
    shifted = np.roll(img, dx, axis=0)
    shifted = np.roll(shifted, dy, axis=1)
    return shifted

def augment_image(img):
    """Apply one random augmentation with pure NumPy."""
    ops = [random_flip, random_brightness, random_noise, random_shift]
    op = np.random.choice(ops)
    return op(img)



def PCA_(X, k=30):

    X_centered = X - X.mean(axis=0)
    C = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[idx][:k]  # k = 30 par ex.
    X_reduced = X_centered @ eigvecs.T
    return X_reduced