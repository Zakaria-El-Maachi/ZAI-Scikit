import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

def f_classif(X, y):
    """
    Compute the ANOVA F-value for each feature.

    Parameters:
    X (array-like): The training input samples.
    y (array-like): The target values (class labels).

    Returns:
    f_scores (array): The computed F-values for each feature.
    p_values (array): P-values are returned as NaN for compatibility.
    """
    mean_y = np.mean(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    f_scores = np.zeros(n_features)
    
    for i in range(n_features):
        numerator = 0
        denominator = 0
        overall_mean = np.mean(X[:, i])
        
        for cls in classes:
            cls_mask = (y == cls)
            cls_mean = np.mean(X[cls_mask, i])
            cls_size = np.sum(cls_mask)
            numerator += cls_size * (cls_mean - overall_mean) ** 2
            denominator += np.sum((X[cls_mask, i] - cls_mean) ** 2)
        
        f_scores[i] = (numerator / (len(classes) - 1)) / (denominator / (len(y) - len(classes)))
    
    return f_scores, np.array([np.nan] * n_features)

class SelectKBest:
    """
    Select features according to the k highest scores.

    Parameters:
    score_func (callable): Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues).
    k (int): The number of top features to select.
    """
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        """
        Run score function on (X, y) and get the appropriate features.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).
        """
        scores, _ = self.score_func(X, y)
        self.scores_ = scores
        self.indices_ = np.argsort(scores)[-self.k:]
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters:
        X (array-like): The training input samples.

        Returns:
        X_reduced (array): The array of selected features.
        """
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).

        Returns:
        X_reduced (array): The array of selected features.
        """
        self.fit(X, y)
        return self.transform(X)

class SelectFromModel:
    """
    Meta-transformer for selecting features based on importance weights.

    Parameters:
    model (estimator): The base estimator from which the transformer is built.
    threshold (string, float): The threshold value to use for feature selection.
    """
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the SelectFromModel meta-transformer.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).
        """
        self.model = clone(self.model).fit(X, y)
        self.importances_ = self.model.feature_importances_
        if isinstance(self.threshold, str) and self.threshold == "mean":
            self.threshold_ = self.importances_.mean()
        else:
            self.threshold_ = self.threshold
        self.indices_ = np.where(self.importances_ >= self.threshold_)[0]
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters:
        X (array-like): The training input samples.

        Returns:
        X_reduced (array): The array of selected features.
        """
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).

        Returns:
        X_reduced (array): The array of selected features.
        """
        self.fit(X, y)
        return self.transform(X)
    
class RFE:
    """
    Feature ranking with recursive feature elimination.

    Parameters:
    model (estimator): A supervised learning estimator with a fit method that updates a coef_ or feature_importances_ attribute.
    n_features_to_select (int, optional): The number of features to select.
    """
    def __init__(self, model, n_features_to_select=None):
        self.model = model
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        """
        Fit the RFE model.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).
        """
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            self.n_features_to_select = n_features // 2

        self.ranking_ = np.ones(n_features, dtype=int)
        remaining_features = list(range(n_features))
        while len(remaining_features) > self.n_features_to_select:
            self.model.fit(X[:, remaining_features], y)
            importances = self.model.feature_importances_
            min_index = np.argmin(importances)
            del remaining_features[min_index]
            self.ranking_[remaining_features] += 1

        self.indices_ = remaining_features
        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters:
        X (array-like): The training input samples.

        Returns:
        X_reduced (array): The array of selected features.
        """
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The training input samples.
        y (array-like): The target values (class labels).

        Returns:
        X_reduced (array): The array of selected features.
        """
        self.fit(X, y)
        return self.transform(X)
