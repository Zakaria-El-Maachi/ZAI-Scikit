from metrics import mode, probability_distribution
from math import sqrt
import numpy as np
from sklearn_base import Classifier
from decision_tree import DecisionTreeClassifier
from joblib import Parallel, delayed


class RandomForestClassifier(Classifier):
    
    def __init__(self, n_estimators = 100, criterion='entropy', min_sample_split=2, max_depth=100) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = None
        self.classes = None
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = None

    def fit(self, X, y):
        self.trees = []
        classes = np.unique(y)
        self.classes = {classes[i]:i for i in classes}
        self.n_features = []
        def build_tree():
            tree = DecisionTreeClassifier(criterion=self.criterion, min_sample_split=self.min_sample_split, max_depth=self.max_depth, classes=self.classes)
            sampleSize, n_features = X.shape
            indices = np.random.choice(sampleSize, sampleSize, replace=True)
            features = np.random.choice(n_features, min(n_features, max(2, round(np.sqrt(n_features)))), replace=False)
            tree.fit(X[indices][:, features], y[indices])
            return tree, features
        trees_and_features = Parallel(n_jobs=-1)(delayed(build_tree)() for _ in range(self.n_estimators))

        # Unpack the results into separate lists
        self.trees, self.n_features = zip(*trees_and_features)

    
    def predict_sample_proba(self, sample):
        # Collect predictions from all decision trees
        predictions = [tree.predict([sample[self.n_features[i]]])[0] for i, tree in enumerate(self.trees)]
        # Calculate the probability distribution using the predictions
        return probability_distribution(predictions, self.classes)