from sklearn_base import Classifier
import numpy as np
from metrics import mse, mae, probability_distribution

class KNN(Classifier):
    """
    K-Nearest Neighbors (KNN) classifier.

    Parameters:
    k (int): Number of neighbors to use.
    distance (str): Distance metric to use ('euclidean' or 'manhattan'). Default is 'euclidean'.
    """

    distanceFuncs = {'euclidean': mse, 'manhattan': mae}

    def __init__(self, k, distance='euclidean') -> None:
        super().__init__()
        self.k = k
        self.distance = distance
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fit the KNN model using the provided training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        self.X = X
        self.y = y
        self.classes = {v: i for i, v in enumerate(np.unique(y))}

    def predict_sample_proba(self, sample):
        """
        Predict the class probabilities for a single sample.

        Parameters:
        sample (numpy.ndarray): The input sample.

        Returns:
        numpy.ndarray: Predicted class probabilities for the sample.
        """
        metric = self.distanceFuncs[self.distance]
        arr = [metric(i, sample) for i in self.X]
        return probability_distribution(self.y[np.argsort(arr)][:self.k], self.classes)
