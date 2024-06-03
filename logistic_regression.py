import numpy as np
from sklearn_base import Classifier
from utilities import sigmoid

class LogisticRegression(Classifier):

    def __init__(self, learning_rate=0.01, num_iterations=5000):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model_weights = None
        self.model_bias = None

    def fit(self, feature_matrix, target_vector):
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
        linear_model = np.dot(sample_features, self.model_weights) + self.model_bias
        probability = sigmoid(linear_model)
        return [1 - probability, probability]