import numpy as np

# Séparation stratifiée des données en ensembles d'entraînement et test
def train_test_split(*arrays, test_size=0.2, train_size=None,
                     random_state=None, shuffle=True):
    # Conversion des entrées en liste
    arrays = list(arrays)

    y = arrays[-1]
    n_samples = y.shape[0]

    # Calcul du nombre d'échantillons d'entraînement et de test
    if train_size is None:
        n_train = int(round(n_samples * (1 - test_size)))
    else:
        n_train = int(round(n_samples * train_size))
    n_test = n_samples - n_train

    # Initialisation du générateur aléatoire
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    # Identification des classes et de leur fréquence
    classes, counts = np.unique(y, return_counts=True)

    # Calcul du nombre d'échantillons de test par classe 
    frac_test = n_test / n_samples
    n_test_per_class = np.rint(counts * frac_test).astype(int)

    # Ajustement pour garantir le bon nombre total d'échantillons de test
    diff = n_test - n_test_per_class.sum()
    if diff != 0:
        raw = counts * frac_test
        err = raw - np.rint(raw)
        order = np.argsort(err)

        if diff > 0:
            pick = order[:diff]
            n_test_per_class[pick] += 1
        else:
            pick = order[::-1][:(-diff)]
            for i in pick:
                if n_test_per_class[i] > 0:
                    n_test_per_class[i] -= 1

    train_idx, test_idx = [], []

    # Séparation des indices pour chaque classe
    for cls, n_cls_test in zip(classes, n_test_per_class):
        cls_idx = np.where(y == cls)[0]
        if shuffle:
            rng.shuffle(cls_idx)

        test_part = cls_idx[:n_cls_test]
        train_part = cls_idx[n_cls_test:]

        test_idx.append(test_part)
        train_idx.append(train_part)

    # Fusion des indices de toutes les classes
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    # Mélange final des indices
    if shuffle:
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    # Application des indices aux tableaux 
    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return result


# Standardisation des données 
class StandardScaler:
    def fit(self, X):
        # Calcul de la moyenne et de l'écart-type par caractéristique
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8

    def transform(self, X):
        # Application de la standardisation
        return (X - self.mu) / self.sigma

    def fit_transform(self, X):
        # Apprentissage + transformation
        self.fit(X)
        return self.transform(X)


# Métriques d'évaluation
def accuracy(y_true, y_pred):
    # Prédictions correctes
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, num_classes=None):
    # Matrice de confusion
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def balanced_accuracy(y_true, y_pred):
    # Calcul du balanced accuracy 
    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm)
    real_pos = cm.sum(axis=1)
    recall = np.where(real_pos > 0, TP / real_pos, 0.0)
    return recall.mean()


def recall_per_class(cm):
    # Calcul du rappel pour chaque classe
    TP = np.diag(cm)
    real_pos = cm.sum(axis=1)
    recall = np.where(real_pos > 0, TP / real_pos, 0.0)
    return recall


# Noyau RBF (Gaussian)
def rbf_kernel(x, X, sigma=1.0):
    """Calcule le noyau RBF entre un point x et un ensemble X."""
    return np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * sigma ** 2))


# Perceptron à noyau multiclasse (One-vs-Rest)
class KernelPerceptron:
    def __init__(self, kernel_fn, n_classes, sigma=1.0,
                 learning_rate=1.0, sample_weights=None, lam=0.0):
        self.kernel_fn = kernel_fn
        self.n_classes = n_classes
        self.sigma = sigma
        self.sample_weights = sample_weights
        self.lr = learning_rate
        self.lam = lam

    def fit(self, X, y, max_epochs=10):
        """Entraînement du perceptron à noyau en One-vs-Rest."""
        N = X.shape[0]
        self.X = X
        self.train_y = y

        # Identification des classes
        self.classes = np.unique(y)
        self.class_idx = {c: i for i, c in enumerate(self.classes)}

        # Coefficients alpha pour chaque classe
        self.alpha = np.zeros((self.n_classes, N))

        # Initialisation des poids d'échantillons
        if self.sample_weights is None:
            self.sample_weights = np.ones(N)

        # Pré-calcul de la matrice de gram
        self.K = np.array([self.kernel_fn(X[i], X, self.sigma) for i in range(N)])

        # Entraînement pour chaque classe (One-vs-Rest)
        for c in self.classes:
            idx = self.class_idx[c]
            y_bin = np.where(y == c, 1, -1)

            count, i, n_iter = 0, 0, 0
            while count < N and n_iter < max_epochs * N:
                # Terme de régularisation 
                self.alpha[idx] *= (1 - self.lr * self.lam)

                # Score du perceptron
                score = np.dot(self.alpha[idx] * y_bin, self.K[i])

                # Mise à jour en cas d'erreur
                if score * y_bin[i] <= 0:
                    self.alpha[idx][i] += self.lr * self.sample_weights[i]
                    count = 0
                else:
                    count += 1

                i = (i + 1) % N
                n_iter += 1

    def predict(self, X_test):
        # Prédiction finale 
        return self.classes[np.argmax(self.predict_scores(X_test), axis=1)]

    def predict_scores(self, X_test):
        # Calcul des scores pour chaque classe
        scores = np.zeros((len(X_test), self.n_classes))
        for i, x in enumerate(X_test):
            k = self.kernel_fn(x, self.X, self.sigma)
            for c in self.classes:
                idx = self.class_idx[c]
                y_bin = np.where(self.train_y == c, 1, -1)
                scores[i, idx] = np.sum(self.alpha[idx] * y_bin * k)
        return scores
