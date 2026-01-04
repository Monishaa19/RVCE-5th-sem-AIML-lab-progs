import numpy as np

class NaiveBayes:
    def __init__(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = X_c.mean(axis=0)
            self.vars[idx, :] = X_c.var(axis=0) + 1e-9  # numerical stability
            self.priors[idx] = X_c.shape[0] / n_samples

    def pdf(self, x, class_idx):
        mean = self.means[class_idx]
        var = self.vars[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict_one(self, x):
        posteriors = []

        for idx in range(len(self.classes)):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.pdf(x, idx)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
