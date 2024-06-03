from sklearn_base import Classifier
import numpy as np
from decision_tree import DecisionTreeRegressor

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



class DecisionStump:
    """
    A weak learner in the form of a decision stump used in the AdaBoost algorithm.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None  # The weight of this stump in the final decision

    def fit(self, X, y, sample_weights):
        """
        Fit the decision stump to the data.
        
        Args:
        - X: Features dataset.
        - y: Target values.
        - sample_weights: Array of weights for each sample.
        """
        n_samples, n_features = X.shape
        best_error = float("inf")

        # Iterate over all features and their possible thresholds
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = self._make_predictions(X[:, feature], threshold, polarity)
                    error = np.sum(sample_weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.polarity = polarity
                        self.feature_index = feature
                        self.threshold = threshold

    def _make_predictions(self, features, threshold, polarity):
        
        """
        Make predictions based on feature values, threshold and polarity.
        """
        if polarity == 1:
            return np.where(features < threshold, -1, 1)
        else:
            return np.where(features > threshold, -1, 1)

    def predict(self, X):
        """
        Predict the labels for a dataset.
        
        Args:
        - X: Dataset to predict.
        
        Returns:
        - Array of predictions.
        """
        features = X[:, self.feature_index]
        return self._make_predictions(features, self.threshold, self.polarity)




class AdaBoostClassifier(Classifier):
    """
    AdaBoost classifier that uses Decision Stumps as weak learners.
    """
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        """
        Fit the AdaBoost model.
        
        Args:
        - X: Features dataset.
        - y: Target values.
        """
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1 / n_samples)
        self.clfs = []

        for _ in range(self.n_clf):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)
            misclassified = predictions != y
            error = np.dot(weights, misclassified) / weights.sum()
            
            if error > 0.5:
                break

            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))
            stump.alpha = alpha
            self.clfs.append(stump)

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= weights.sum()

    def predict(self, X):
        """
        Predict the class labels for the given samples.
        
        Args:
        - X: Dataset to predict.
        
        Returns:
        - Predicted class labels.
        """
        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Args:
        - X: Dataset to predict.
        
        Returns:
        - Probability estimates for the positive class.
        """

        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sum(clf_preds, axis=0)
        prob_pos = np.exp(y_pred) / (np.exp(y_pred) + np.exp(-y_pred))
        return np.vstack((1 - prob_pos, prob_pos)).T

    def predict_sample_proba(self, sample):
        """
        Predict the class probabilities for a single sample.
        
        Args:
        - sample: A single data point.
        
        Returns:
        - Probability estimates for the positive class.
        """
        sample = np.array(sample).reshape(1, -1)  # Ensure sample is 2D
        return self.predict_proba(sample)[0]
