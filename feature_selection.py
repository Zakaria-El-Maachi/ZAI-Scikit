import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

def f_classif(X, y):
    """Compute the ANOVA F-value for the provided sample."""
    # Determine the mean of each class
    mean_y = np.mean(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    f_scores = np.zeros(n_features)
    
    # For each feature calculate the F-score
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
        
        # Calculate the F-value for the feature
        f_scores[i] = (numerator / (len(classes) - 1)) / (denominator / (len(y) - len(classes)))
    
    return f_scores, np.array([np.nan] * n_features)  # Return p-values as NaN for compatibility


class SelectKBest:
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.indices_ = None

    def fit(self, X, y):
        scores, _ = self.score_func(X, y)
        self.scores_ = scores
        self.indices_ = np.argsort(scores)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)



class SelectFromModel:
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold  # Could be a float or a string (e.g., "mean")

    def fit(self, X, y):
        self.model = clone(self.model).fit(X, y)
        self.importances_ = self.model.feature_importances_
        if isinstance(self.threshold, str) and self.threshold == "mean":
            self.threshold_ = self.importances_.mean()
        else:
            self.threshold_ = self.threshold
        self.indices_ = np.where(self.importances_ >= self.threshold_)[0]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
class RFE:
    def __init__(self, model, n_features_to_select=None):
        self.model = model
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
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
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

