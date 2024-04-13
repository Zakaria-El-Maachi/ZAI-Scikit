from sklearn_base import Predictor
import numpy as np

class LinearRegression(Predictor):

    def __init__(self, n_iter = 5000, alpha = 0.01) -> None:
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.alpha = alpha

    def fit(self, X, y):
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            pred = X@self.w + self.b
            self.w -= self.alpha * (pred-y)@X / len(y)
            self.b -= self.alpha * np.sum(pred-y) / len(y)

    def predict(self, X):
        return X@self.w + self.b