from sklearn_base import Predictor, Classifier
from utilities import sigmoid
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


class LogisticRegression(Classifier):
    """
    Logistic Regression model using gradient descent optimization.

    Parameters:
    learning_rate (float): Learning rate for the gradient descent optimization. Default is 0.01.
    num_iterations (int): Number of iterations for the gradient descent optimization. Default is 5000.

    Attributes:
    model_weights (numpy.ndarray): Weights of the logistic regression model.
    model_bias (float): Bias term of the logistic regression model.
    classes (dict): Dictionary mapping class labels to indices.

    Methods:
    fit(feature_matrix, target_vector): Fit the logistic regression model to the training data.
    predict_sample_proba(sample_features): Predict the class probabilities for a single sample.
    """

    def __init__(self, learning_rate=0.01, num_iterations=5000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model_weights = None
        self.model_bias = None

    def fit(self, feature_matrix, target_vector):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        feature_matrix (numpy.ndarray): Training data, shape (n_samples, n_features).
        target_vector (numpy.ndarray): Target values, shape (n_samples,).

        Returns:
        None
        """
        num_samples, num_features = feature_matrix.shape
        self.model_weights = np.zeros(num_features)
        self.model_bias = 0
        self.classes = {0: 0, 1: 1}

        for _ in range(self.num_iterations):
            linear_model = np.dot(feature_matrix, self.model_weights) + self.model_bias
            probability_predictions = sigmoid(linear_model)

            weight_gradient = (1 / num_samples) * np.dot(feature_matrix.T, (probability_predictions - target_vector))
            bias_gradient = (1 / num_samples) * np.sum(probability_predictions - target_vector)

            self.model_weights -= self.learning_rate * weight_gradient
            self.model_bias -= self.learning_rate * bias_gradient

    def predict_sample_proba(self, sample_features):
        """
        Predict the class probabilities for a single sample.

        Parameters:
        sample_features (numpy.ndarray): Input features for the sample, shape (n_features,).

        Returns:
        list: List containing the probabilities of the sample belonging to class 0 and class 1.
        """
        linear_model = np.dot(sample_features, self.model_weights) + self.model_bias
        probability = sigmoid(linear_model)
        return [1 - probability, probability]
    



class SVM(Classifier):
    """
    Support Vector Machine (SVM) classifier.

    Parameters:
    learning_rate (float): Learning rate for gradient descent.
    lambda_param (float): Regularization parameter.
    n_iters (int): Number of iterations for training.

    Attributes:
    weights (ndarray): Weights of the model.
    bias (float): Bias term of the model.
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=5000):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, feature_matrix, target_vector):
        """
        Fit the SVM model to the training data.

        Args:
        feature_matrix (ndarray): Training data.
        target_vector (ndarray): Target values.
        """
        num_samples, num_features = feature_matrix.shape

        transformed_target = np.where(target_vector <= 0, -1, 1)

        # Initialize weights
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for idx, sample in enumerate(feature_matrix):
                if transformed_target[idx] * (np.dot(sample, self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(sample, transformed_target[idx]))
                    self.bias -= self.learning_rate * transformed_target[idx]

    def predict_sample_proba(self, sample_features):
        """
        Predict the probability for a single sample.

        Args:
        sample_features (ndarray): The input features for a single sample.

        Returns:
        list: The probabilities of the sample belonging to each class.
        """
        approx = np.dot(sample_features, self.weights) - self.bias
        decision_value = np.sign(approx)
        probability = (decision_value + 1) / 2  # Convert to probability (0 to 1)
        return [1 - probability, probability]

    def predict(self, feature_matrix):
        """
        Predict the class labels for the input data.

        Args:
        feature_matrix (ndarray): The input data.

        Returns:
        ndarray: The predicted class labels.
        """
        approx = np.dot(feature_matrix, self.weights) - self.bias
        return np.where(approx >= 0, 1, 0)
