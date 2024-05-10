from sklearn_base import Classifier
import numpy as np
from metrics import mse, mae, probability_distribution

class KNN(Classifier):

    distanceFuncs = {'euclidean':mse, 'manhattan':mae}

    def __init__(self, k, distance = 'euclidean') -> None:
        super().__init__()
        self.k = k
        self.distance = distance
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = {v:i for i, v in enumerate(np.unique(y))}
    
    def predict_sample_proba(self, sample):
        metric = self.distanceFuncs[self.distance]
        arr = [metric(i, sample) for i in self.X]
        return probability_distribution(self.y[np.argsort(arr)][:self.k], self.classes)