from metrics import mode, probabilityDistribution
from math import sqrt
import numpy as np
from sklearn_base import Classifier
from decision_tree import DecisionTreeClassifier
from joblib import Parallel, delayed


class RandomForestClassifier(Classifier):
    
    def __init__(self, n_estimators = 100, criterion='entropy', min_sample_split=2, max_depth=100, n_features=None) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = None
        self.classes = None
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):
        self.trees = []
        classes = np.unique(y)
        self.classes = {classes[i]:i for i in classes}
        def build_tree():
            tree = DecisionTreeClassifier(criterion=self.criterion, min_sample_split=self.min_sample_split, max_depth=self.max_depth, n_features=self.n_features)
            sampleSize, n_features = X.shape
            indices = np.random.choice(sampleSize, sampleSize, replace=True)
            features = np.random.choice(n_features, max(2, round(np.sqrt(n_features))), replace=False)
            tree.fit(X[indices][:, features], y[indices])
            return tree
        self.trees = Parallel(n_jobs=-1)(delayed(build_tree)() for _ in range(self.n_estimators))

    
    def predictProba(self, X):
        y_pred = np.swapaxes([tree.predict(X) for tree in self.trees], 0, 1)
        return np.array([probabilityDistribution(predictions, self.classes) for predictions in y_pred])