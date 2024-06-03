from sklearn_base import Predictor
import numpy as np

class LinearRegression(Predictor):
    """
    Linear Regression model using gradient descent optimization.

    Parameters:
    n_iter (int): Number of iterations for the gradient descent optimization. Default is 5000.
    alpha (float): Learning rate for the gradient descent optimization. Default is 0.01.

    Attributes:
    w (numpy.ndarray): Weights of the linear regression model.
    b (float): Bias term of the linear regression model.

    Methods:
    fit(X, y): Fit the linear regression model to the training data.
    predict(X): Predict the target values for the given input data.
    """

    def __init__(self, n_iter=5000, alpha=0.01) -> None:
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        X (numpy.ndarray): Training data, shape (n_samples, n_features).
        y (numpy.ndarray): Target values, shape (n_samples,).

        Returns:
        None
        """
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iter):
            pred = X @ self.w + self.b
            self.w -= self.alpha * (pred - y) @ X / len(y)
            self.b -= self.alpha * np.sum(pred - y) / len(y)

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters:
        X (numpy.ndarray): Input data, shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Predicted target values, shape (n_samples,).
        """
        return X @ self.w + self.b
