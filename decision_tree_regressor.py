import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)
        best_idx, best_thr = self._best_split(X, y)
        if best_idx is None:
            return np.mean(y)
        left = X[:, best_idx] <= best_thr
        right = X[:, best_idx] > best_thr
        left_tree = self._build_tree(X[left], y[left], depth + 1)
        right_tree = self._build_tree(X[right], y[right], depth + 1)
        return (best_idx, best_thr, left_tree, right_tree)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left_tree, right_tree = tree
        if x[feature_idx] <= threshold:
            return self._predict(x, left_tree)
        else:
            return self._predict(x, right_tree)

    def _best_split(self, X, y):
        m, n = X.shape
        if m < 2:
            return None, None
        best_mse = np.inf
        best_idx, best_thr = None, None
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            left_y = []
            right_y = list(classes)
            for i in range(1, m):  # possible split positions
                left_y.append(right_y.pop(0))
                if thresholds[i] == thresholds[i - 1]:
                    continue
                mse = len(left_y) * np.var(left_y) + len(right_y) * np.var(right_y)
                if mse < best_mse:
                    best_mse = mse
                    best_idx, best_thr = idx, (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
