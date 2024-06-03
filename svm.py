import numpy as np
from sklearn_base import Classifier


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
