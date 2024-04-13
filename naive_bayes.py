import numpy as np
from sklearn_base import Classifier
from metrics import probabilityDistribution, accuracy

class NaiveBayes(Classifier):

    def __init__(self, laplace = 1, priors = None, classProba = None) -> None:
        super().__init__()
        self.laplace = laplace
        self.priors = priors
        self.classes = None
        self.classProba = classProba
        self.featureValues = None

    def fit(self, X, y):
        if self.priors is not None:
            return
        self.priors = {}
        total = X.shape[0]
        classes, classCounts = np.unique(y, return_counts=True)
        self.classes = dict(zip(classes, [i for i in range(len(classes))]))
        self.classProba = dict(zip(classes, classCounts/total))
        self.featureValues = {}

        for feature in range(X.shape[1]):
            featureValues = np.unique(X[:, feature])
            self.featureValues[feature] = dict(zip(featureValues,[i for i in range(len(featureValues))]))

        for cls in classes:
            self.priors[cls] = {}
            for feature in range(X.shape[1]):
                subset = X[y == cls, feature]
                self.priors[cls][feature] = probabilityDistribution(subset, self.featureValues[feature], laplaceSmoothing = self.laplace)
                
    
    def predictProba(self, X):
        return np.vectorize(self.predictHelper, signature='(n)->(k)')(X)
        
    def predictHelper(self, sample):
        def predict(sample, cls):
            probas = np.array([self.priors[cls][feature][self.featureValues[feature][sample[feature]]] for feature in range(len(sample))] + [self.classProba[cls]])
            return np.sum(np.log(probas))
        return np.array([predict(sample, cls) for cls in self.classes])