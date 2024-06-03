from metrics import mode, probability_distribution
from math import sqrt
import numpy as np
from sklearn_base import Classifier, Predictor
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from joblib import Parallel, delayed


class RandomForestClassifier(Classifier):
    """
    Random Forest Classifier.

    Parameters:
    n_estimators (int): Number of trees in the forest. Default is 100.
    criterion (str): The function to measure the quality of a split ('entropy' or 'gini'). Default is 'entropy'.
    min_sample_split (int): The minimum number of samples required to split an internal node. Default is 2.
    max_depth (int): The maximum depth of the tree. Default is 100.
    """

    def __init__(self, n_estimators=100, criterion='entropy', min_sample_split=2, max_depth=100) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.trees = None
        self.classes = None
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = None

    def fit(self, X, y):
        """
        Fit the Random Forest model using the provided training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        self.trees = []
        classes = np.unique(y)
        self.classes = {classes[i]: i for i in classes}
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
        """
        Predict the class probabilities for a single sample.

        Parameters:
        sample (numpy.ndarray): The input sample.

        Returns:
        numpy.ndarray: Predicted class probabilities for the sample.
        """
        # Collect predictions from all decision trees
        predictions = [tree.predict([sample[self.n_features[i]]])[0] for i, tree in enumerate(self.trees)]
        # Calculate the probability distribution using the predictions
        return probability_distribution(predictions, self.classes)



class RandomForestRegressor(Predictor):
    """
    Random Forest Regressor.

    Parameters:
    n_estimators (int): Number of trees in the forest. Default is 100.
    max_depth (int): The maximum depth of the tree. Default is 100.
    min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
    """
    
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        """
        Fit the Random Forest model using the provided training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_indices = []

        def build_tree():
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            selected_features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
            tree.fit(X[bootstrap_indices][:, selected_features], y[bootstrap_indices])
            return tree, selected_features

        trees_and_features = Parallel(n_jobs=-1)(delayed(build_tree)() for _ in range(self.n_estimators))

        self.trees, self.feature_indices = zip(*trees_and_features)

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted target values for the input data.
        """
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i, (tree, features) in enumerate(zip(self.trees, self.feature_indices)):
            predictions[:, i] = tree.predict(X[:, features])

        return np.mean(predictions, axis=1)