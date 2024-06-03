import numpy as np
from sklearn_base import Classifier

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None  # The weight of this stump in the final decision

    def fit(self, X, y, sample_weights):
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
        if polarity == 1:
            return np.where(features < threshold, -1, 1)
        else:
            return np.where(features > threshold, -1, 1)

    def predict(self, X):
        features = X[:, self.feature_index]
        return self._make_predictions(features, self.threshold, self.polarity)




class AdaBoostClassifier(Classifier):
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
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
        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return y_pred

    def predict_proba(self, X):
        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])
        y_pred = np.sum(clf_preds, axis=0)
        prob_pos = np.exp(y_pred) / (np.exp(y_pred) + np.exp(-y_pred))
        return np.vstack((1 - prob_pos, prob_pos)).T

    def predict_sample_proba(self, sample):
        sample = np.array(sample).reshape(1, -1)  # Ensure sample is 2D
        return self.predict_proba(sample)[0]
