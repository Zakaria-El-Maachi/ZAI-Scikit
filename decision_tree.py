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

    def isLeaf(self):
        return self.left == None and self.right == None

    def countingMap(self, target, empty = False):
        if empty:
            return {val:0 for val in np.unique(target)}
        uniqueValues, count = np.unique(target, return_counts=True)
        return dict(zip(uniqueValues, count))
    
    def calculateProba(self, target):
        self.probaDistribution = probabilityDistribution(target, self.classes)

    def fit(self, data, target):
        data = data.copy()
        metricFunc = self.criterionMetric[self.criterion]
        _, classCounts = np.unique(target, return_counts=True)
        basePerformance = metricFunc(classCounts, proba=False)
        bestGain = 0

        leftMap = self.countingMap(target, empty=True)
        rightMap = self.countingMap(target)
        
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
                    self.threshold = (t + thresholds[index+1])/2
                    self.feature = feature

    def predictProba(self, X):
        return self.probaDistribution





class DecisionTreeClassifier(Classifier):

    def __init__(self, criterion='entropy', min_sample_split=2, max_depth=100, n_features=None):
        super().__init__()
        self.criterion = criterion
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        classes = np.unique(y)
        self.classes = {classes[i]:i for i in range(len(classes))}
        self.root = self._growTree(X, y)


    def _growTree(self, X, y, depth = 0):
        samples, _ = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if(depth >= self.max_depth or n_labels == 1 or samples < self.min_sample_split):
            leafNode = DecisionStump(classes=self.classes)
            leafNode.calculateProba(y)
            return leafNode


        # Find the best split
        node = DecisionStump(criterion=self.criterion)
        node.fit(X, y)

        if node.feature is None:
            node.classes = self.classes
            node.calculateProba(y)
            return node

        # Create Child Nodes
        leftIndices = np.where(X[:, node.feature] <= node.threshold)
        Xleft, yleft = X[leftIndices], y[leftIndices]
        node.left = self._growTree(Xleft, yleft, depth+1)

        rightIndices = np.where(X[:, node.feature] > node.threshold)
        Xright, yright = X[rightIndices], y[rightIndices]
        node.right = self._growTree(Xright, yright, depth+1)

        return node


    def predictProba(self, X):
        vectorizedTraversal = np.vectorize(self._traverseTree, signature='(n)->(k)')
        return vectorizedTraversal(X)

    def _traverseTree(self, sample):
        curNode = self.root
        while not curNode.isLeaf():
            if sample[curNode.feature] <= curNode.threshold:
                curNode = curNode.left
            else:
                curNode = curNode.right
        return curNode.probaDistribution
