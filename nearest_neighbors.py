from sklearn_base import Classifier
import numpy as np
from metrics import mse, mae, probabilityDistribution

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

    def predictProba(self, X):
        return np.vectorize(self.helperProba, signature='(n)->(k)')(X)
    
    def helperProba(self, sample):
        metric = self.distanceFuncs[self.distance]
        arr = [metric(i, sample) for i in self.X]
        return probabilityDistribution(self.y[np.argsort(arr)][:self.k], self.classes)