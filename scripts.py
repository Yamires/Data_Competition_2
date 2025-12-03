import numpy as np


def train_test_split(*arrays, test_size=0.2, train_size=None, random_state=None, shuffle=True):
    arrays = list(arrays)
    n_samples = arrays[0].shape[0]

    if train_size is None:
        n_train = int(n_samples * (1 - test_size))
    else:
        n_train = int(n_samples * train_size)

    if random_state is not None:
        np.random.seed(random_state)

    perm = np.random.permutation(n_samples)
    
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return result

class StandardScaler:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
    def transform(self, X):
        return (X - self.mu) / self.sigma
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm)
    real_pos = cm.sum(axis=1)
    recall = np.where(real_pos > 0, TP / real_pos, 0.0)
    return recall.mean()


def recall_per_class(cm):
    """
    cm : matrice de confusion (numpy array KxK)
    retourne un vecteur de recall par classe
    """
    TP = np.diag(cm)
    real_pos = cm.sum(axis=1)   # total de vrais échantillons par classe
    
    # recall par classe (évite division par zéro)
    recall = np.where(real_pos > 0, TP / real_pos, 0.0)
    return recall

def hybrid_kernel(x, X, sigma=5, degree=3, coef=1, alpha=0.5, beta=0.5):
    """
    x : shape (d,)
    X : shape (N, d)
    returns : shape (N,)
    """
    # RBF kernel (vectorized)
    diff = X - x
    rbf = np.exp(-np.sum(diff * diff, axis=1) / (2 * sigma**2))
    
    # Polynomial kernel (vectorized)
    poly = (X @ x + coef) ** degree
    
    # Combine
    return ( alpha * rbf) * (beta * poly)

    
    return alpha * rbf + beta * poly

def rbf_kernel(x, X, sigma=1.0):
    """Calcule le noyau RBF entre x et X."""
    return np.exp(-np.sum((X - x)**2, axis=1) / (2 * sigma**2))


class KernelPerceptron:
    def __init__(self, kernel_fn, n_classes, sigma=1.0, learning_rate=1.0, sample_weights=None, lam=1.0):
        self.kernel_fn = kernel_fn
        self.n_classes = n_classes
        self.sigma = sigma
        self.sample_weights = sample_weights
        self.lr= learning_rate
        self.lam=lam 

    def fit(self, X, y, max_epochs=10):
        """Entraînement multiclasse One-vs-Rest."""
        N = X.shape[0]
        self.X = X
        self.classes = np.unique(y)

        self.class_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_y = y
        self.alpha = np.zeros((self.n_classes, N))

        if self.sample_weights is None:
            self.sample_weights = np.ones(N)
        
        # Precompute Gram matrix
        self.K = np.array([self.kernel_fn(X[i], X, self.sigma) for i in range(N)])
        
        for c in self.classes:
            print(f"Training classifier for class {c}...")
            idx = self.class_idx[c]
            y_bin = np.where(y == c, 1, -1)
            count, i, n_iter = 0, 0, 0
            
            while count < N and n_iter < max_epochs * N:
                self.alpha[idx] *= (1 - self.lr * self.lam)

                scores = np.dot(self.alpha[idx] * y_bin, self.K[i])
 
                if scores * y_bin[i] <= 0:
                    self.alpha[idx][i] += self.lr * self.sample_weights[i]
                    count = 0
                else:
                    count += 1
                i = (i + 1) % N
                n_iter += 1
        print("Training done.")

    def predict(self, X_test):
        """Retourne la classe prédite pour chaque x_test."""
        return self.classes[np.argmax(self.predict_scores(X_test), axis=1)]

    def predict_scores(self, X_test):
        """Retourne la classe prédite pour chaque x_test."""
        scores = np.zeros((len(X_test), self.n_classes))
        for i, x in enumerate(X_test):
            k = self.kernel_fn(x, self.X, self.sigma)
            for c in self.classes:
                idx = self.class_idx[c]
                y_bin = np.where(self.train_y == c, 1, -1)
                scores[i, idx] = np.sum(self.alpha[idx] * y_bin * k)
        return scores
    

def rbf_kernel_svm(X,Y, sigma):
    dists = np.sum(X**2, axis=1)[:,None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
    return np.exp(-dists / (2*sigma**2))

class SVM:
    def __init__(self, sigma=1.0, max_iter=5, sample_weights=None):
        self.sigma = sigma
        self.max_iter = max_iter
        self.sample_weights = sample_weights 

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.K = rbf_kernel_svm(X,X,self.sigma)

        N = X.shape[0]

        self.alpha = {}

        for c in self.classes: 
            print(f"train class {c} vs rest")
            y_bin = np.where(y == c, 1, -1)
            alpha_c = np.zeros(N)

            for it in range(self.max_iter):
                for i in range(N):
                    f = np.sum(alpha_c * y_bin * self.K[:, i])
                    if y_bin[i] * f <= 0:
                        alpha_c[i] += self.sample_weights[i]

            self.alpha[c] = alpha_c

    def decision_function(self, X_test):
        K_test = rbf_kernel_svm(X_test, self.X, self.sigma)

        scores = [] 
        for c in self.classes: 
            y_bin = np.where(self.y == c,1,-1)
            s = K_test @ (self.alpha[c]*y_bin)
            scores.append(s)

        return np.vstack(scores).T
    
    def predict(self, X_test):
        scores = self.decision_function(X_test)
        return self.classes[np.argmax(scores, axis=1)]
    


    def remove_highly_correlated_features(X_train, X_val, threshold=0.95):
        """
        Supprime les caractéristiques dont la corrélation absolue avec une autre
        caractéristique précédente dépasse le seuil.
        """
        # 1. Calculer la matrice de corrélation (features x features)
        # rowvar=False car les colonnes sont les features
        corr_matrix = np.corrcoef(X_train, rowvar=False)
        
        # 2. Prendre la valeur absolue
        corr_matrix = np.abs(corr_matrix)
        
        # 3. Sélectionner le triangle supérieur de la matrice (sans la diagonale)
        # k=1 exclut la diagonale (qui est toujours 1)
        upper_tri = np.triu(corr_matrix, k=1)
        
        # 4. Trouver les colonnes à supprimer
        # np.where renvoie les indices où la condition est vraie
        to_drop_indices = np.where(upper_tri > threshold)[1]
        
        # On veut des indices uniques
        to_drop_unique = np.unique(to_drop_indices)
        
        # 5. Garder seulement les indices NON présents dans to_drop
        all_indices = np.arange(X_train.shape[1])
        keep_indices = np.setdiff1d(all_indices, to_drop_unique)
        
        print(f"Caractéristiques originales: {X_train.shape[1]}")
        print(f"Caractéristiques supprimées (redondantes): {len(to_drop_unique)}")
        print(f"Caractéristiques restantes: {len(keep_indices)}")
        
        return X_train[:, keep_indices], X_val[:, keep_indices], keep_indices
    
    def remove_low_var_features(X_train, X_val, threshold=1e-5):

        variances = np.var(X_train, axis=0)
        
        high_var_indices = np.where(variances > threshold)[0]

        return X_train[:, high_var_indices], X_val[:, high_var_indices]
    
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
            

    def fit(self, X, y, lr=1e-4, n_steps=1000, sample_weights=None, 
                decay_rate=0.999, verbose=True): # <-- AJOUT de decay_rate
            
            losses = []
            
            # Initialisation du taux d'apprentissage courant (lr_current)
            lr_current = lr 

            for step in range(n_steps):
                
                loss, probs = self.loss(X, y, sample_weights)
                dW, db = self.grad(X, y, probs, sample_weights)

                # Mise à jour des poids avec le taux d'apprentissage actuel
                self.W -= lr_current * dW
                self.b -= lr_current * db

                # AJUSTEMENT CLÉ : Décroissance exponentielle du taux d'apprentissage
                lr_current = lr * (decay_rate ** step) # lr_current diminue légèrement à chaque pas

                losses.append(loss)

                if verbose and step % 100 == 0:
                    print(f"Step {step}, lr_current = {lr_current:.6f}, loss = {loss:.4f}")

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