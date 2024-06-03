from sklearn_base import Classifier
import numpy as np

class DecisionTreeRegressor:
    """
    A simple decision tree regressor for binary splits.

    Parameters:
    max_depth (int): Maximum depth of the tree.
    min_samples_split (int): Minimum number of samples required to split an internal node.
    """
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree regressor to the data.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        depth (int): Current depth of the tree.

        Returns:
        node (tuple or float): The built tree node.
        """
        if len(y) < self.min_samples_split or depth == self.max_depth:
            return np.mean(y)
        best_idx, best_thr = self._best_split(X, y)
        if best_idx is None:
            return np.mean(y)
        left = X[:, best_idx] <= best_thr
        right = X[:, best_idx] > best_thr
        left_tree = self._build_tree(X[left], y[left], depth+1)
        right_tree = self._build_tree(X[right], y[right], depth+1)
        return (best_idx, best_thr, left_tree, right_tree)

    def predict(self, X):
        """
        Predict target values for the given data.

        Parameters:
        X (array-like): Input data.

        Returns:
        predictions (array): Predicted target values.
        """
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, inputs, node):
        """
        Recursively predict the target value for a single sample.

        Parameters:
        inputs (array-like): Input sample.
        node (tuple or float): Current node of the tree.

        Returns:
        prediction (float): Predicted target value.
        """
        if not isinstance(node, tuple):
            return node
        else:
            feature, threshold, left, right = node
            if inputs[feature] <= threshold:
                return self._predict(inputs, left)
            else:
                return self._predict(inputs, right)

    def _best_split(self, X, y):
        """
        Find the best split for the data.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.

        Returns:
        best_idx (int): Index of the best feature to split on.
        best_thr (float): Best threshold value to split on.
        """
        best_mse = np.inf
        best_idx, best_thr = None, None
        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left = y[X[:, idx] <= thr]
                right = y[X[:, idx] > thr]
                mse = self._calculate_mse(left, right)
                if mse < best_mse:
                    best_mse = mse
                    best_idx, best_thr = idx, thr
        return best_idx, best_thr

    def _calculate_mse(self, left, right):
        """
        Calculate the mean squared error for the given split.

        Parameters:
        left (array-like): Left split of the target values.
        right (array-like): Right split of the target values.

        Returns:
        mse (float): Mean squared error for the split.
        """
        if len(left) == 0 or len(right) == 0:
            return np.inf
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        left_mse = np.mean((left - left_mean) ** 2)
        right_mse = np.mean((right - right_mean) ** 2)
        return len(left) / (len(left) + len(right)) * left_mse + len(right) / (len(left) + len(right)) * right_mse

class GradientBoostingClassifier(Classifier):
    """
    Gradient Boosting Classifier.

    Parameters:
    n_estimators (int): The number of boosting stages to perform.
    learning_rate (float): Learning rate shrinks the contribution of each tree.
    max_depth (int): Maximum depth of the individual regression estimators.
    min_samples_split (int): Minimum number of samples required to split an internal node.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.models = []

    def fit(self, X, y):
        """
        Fit the Gradient Boosting model.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values (binary classification).
        """
        # Initial predictions are the log-odds of the positive class
        self.init_pred = np.log(np.sum(y) / (len(y) - np.sum(y)))
        F = np.full(y.shape, self.init_pred)
        for _ in range(self.n_estimators):
            residuals = -1 * (2 * y - 1) / (1 + np.exp((2 * y - 1) * F))
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.models.append(tree)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters:
        X (array-like): Input data.

        Returns:
        proba (array): Predicted class probabilities.
        """
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.models:
            F += self.learning_rate * tree.predict(X)
        proba = 1 / (1 + np.exp(-2 * F))
        return np.vstack([1-proba, proba]).T

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters:
        X (array-like): Input data.

        Returns:
        predictions (array): Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_sample_proba(self, sample):
        """
        Predict class probabilities for a single sample.

        Parameters:
        sample (array-like): Input sample.

        Returns:
        proba (array): Predicted class probabilities for the sample.
        """
        F = self.init_pred
        for tree in self.models:
            F += self.learning_rate * tree.predict(sample.reshape(1, -1))
        proba = 1 / (1 + np.exp(-2 * F))
        return np.array([1-proba, proba])
