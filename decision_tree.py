import numpy as np
from metrics import *
from sklearn_base import Classifier

class DecisionStump:

    criterionMetric = {'entropy':entropy, 'gini': gini}

    def __init__(self, criterion='entropy', feature=None, threshold=None, classes = None) -> None:
        self.criterion = criterion
        self.feature = feature
        self.threshold = threshold
        self.classes = classes
        self.left = None
        self.right = None
        self.probaDistribution = None

    def is_leaf(self):
        return self.left == None and self.right == None

    def counting_map(self, target, empty = False):
        if empty:
            return {val:0 for val in np.unique(target)}
        uniqueValues, count = np.unique(target, return_counts=True)
        return dict(zip(uniqueValues, count))
    
    def calculate_proba(self, target):
        self.probaDistribution = probability_distribution(target, self.classes)

    def fit(self, data, target):
        data = data.copy()
        metricFunc = self.criterionMetric[self.criterion]
        _, classCounts = np.unique(target, return_counts=True)
        basePerformance = metricFunc(classCounts, proba=False)
        bestGain = 0

        leftMap = self.counting_map(target, empty=True)
        rightMap = self.counting_map(target)
        
        for feature in range(data.shape[1]):
            sortedIndices = np.argsort(data[:, feature])
            data = data[sortedIndices]
            target = target[sortedIndices]
            left_map = leftMap.copy()
            right_map = rightMap.copy()
            counter = 0
            leftNumber = 0
            rightNumber = len(target)
            thresholds = np.unique(data[:, feature])
            for index, t in enumerate(thresholds):
                while counter < len(data):
                    if data[counter, feature] > t:
                        break
                    left_map[target[counter]] += 1
                    leftNumber += 1
                    right_map[target[counter]] -= 1
                    rightNumber -= 1
                    counter += 1
                performance = (metricFunc(left_map, proba=False) * leftNumber + metricFunc(right_map, proba=False) * rightNumber) / len(target)
                gain = basePerformance - performance
                if gain > bestGain:
                    bestGain = gain
                    if(index+1 < len(thresholds)):
                        self.threshold = (t + thresholds[index+1])/2
                    else : 
                        self.threshold = t
                    self.feature = feature

    def predict_sample_proba(self, X):
        return self.probaDistribution





class DecisionTreeClassifier(Classifier):

    def __init__(self, criterion='entropy', min_sample_split=2, max_depth=100, n_features=None, classes=None):
        super().__init__()
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = None
        self.root = None
        self.classes = classes

    def fit(self, X, y):
        if(self.classes == None):
            classes = np.unique(y)
            self.classes = {classes[i]:i for i in range(len(classes))}
        self.root = self._growTree(X, y)


    def _growTree(self, X, y, depth = 0):
        samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if(depth >= self.max_depth or n_labels == 1 or samples < self.min_sample_split):
            leafNode = DecisionStump(classes=self.classes)
            leafNode.calculate_proba(y)
            return leafNode


        # Find the best split
        node = DecisionStump(criterion=self.criterion)
        node.fit(X, y)

        if node.feature is None:
            node.classes = self.classes
            node.calculate_proba(y)
            return node

        # Create Child Nodes
        leftIndices = np.where(X[:, node.feature] <= node.threshold)
        Xleft, yleft = X[leftIndices], y[leftIndices]
        node.left = self._growTree(Xleft, yleft, depth+1)

        rightIndices = np.where(X[:, node.feature] > node.threshold)
        Xright, yright = X[rightIndices], y[rightIndices]
        node.right = self._growTree(Xright, yright, depth+1)

        return node


    def predict_sample_proba(self, sample):
        curNode = self.root
        depth = 0
        print("prediction of tree")
        while not curNode.is_leaf():
            print(depth)
            depth += 1
            if sample[curNode.feature] <= curNode.threshold:
                curNode = curNode.left
            else:
                curNode = curNode.right
        return curNode.probaDistribution


class DecisionStumpRegressor:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None

    def fit(self, data, target):
        best_mse = np.inf
        best_feature = None
        best_threshold = None

        for feature in range(data.shape[1]):
            thresholds = np.unique(data[:, feature])
            for t in thresholds:
                left_indices = data[:, feature] <= t
                right_indices = data[:, feature] > t
                if len(left_indices) < 2 or len(right_indices) < 2:
                    continue

                left_mse = np.var(target[left_indices]) * np.sum(left_indices)
                right_mse = np.var(target[right_indices]) * np.sum(right_indices)
                mse = left_mse + right_mse

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = t

        self.feature = best_feature
        self.threshold = best_threshold

    def predict_sample(self, sample):
        if self.is_leaf():
            return self.value
        if sample[self.feature] <= self.threshold:
            return self.left.predict_sample(sample)
        else:
            return self.right.predict_sample(sample)




class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) == 0:
            leaf_value = np.mean(y)
            return DecisionStumpRegressor(value=leaf_value)

        node = DecisionStumpRegressor()
        node.fit(X, y)
        if node.feature is None:
            leaf_value = np.mean(y)
            return DecisionStumpRegressor(value=leaf_value)

        left_indices = X[:, node.feature] <= node.threshold
        right_indices = X[:, node.feature] > node.threshold
        node.left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return node

    def predict(self, X):
        return np.array([self.root.predict_sample(sample) for sample in X])
