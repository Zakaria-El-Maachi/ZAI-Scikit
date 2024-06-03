import numpy as np
from sklearn_base import Classifier
from metrics import probability_distribution, accuracy

class NaiveBayes(Classifier):
    """
    Naive Bayes classifier for categorical features.

    Parameters:
    laplace (float): Smoothing parameter. Default is 1.
    priors (dict): Predefined priors. Default is None.
    classProba (dict): Class probabilities. Default is None.
    """

    def __init__(self, laplace=1, priors=None, classProba=None) -> None:
        super().__init__()
        self.laplace = laplace
        self.priors = priors
        self.classes = None
        self.classProba = classProba
        self.featureValues = None

    def fit(self, X, y):
        """
        Fit the Naive Bayes model according to the given training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        if self.priors is not None:
            return
        self.priors = {}
        total = X.shape[0]
        classes, classCounts = np.unique(y, return_counts=True)
        self.classes = dict(zip(classes, [i for i in range(len(classes))]))
        self.classProba = dict(zip(classes, classCounts / total))
        self.featureValues = {}

        for feature in range(X.shape[1]):
            featureValues = np.unique(X[:, feature])
            self.featureValues[feature] = dict(zip(featureValues, [i for i in range(len(featureValues))]))

        for cls in classes:
            self.priors[cls] = {}
            for feature in range(X.shape[1]):
                subset = X[y == cls, feature]
                self.priors[cls][feature] = probability_distribution(subset, self.featureValues[feature], laplaceSmoothing=self.laplace)

    def predict_sample_proba(self, sample):
        """
        Predict the class probabilities for a single sample.

        Parameters:
        sample (numpy.ndarray): The input sample.

        Returns:
        numpy.ndarray: Predicted class probabilities for the sample.
        """
        def predict(sample, cls):
            probas = np.array([self.priors[cls][feature][self.featureValues[feature][sample[feature]]] for feature in range(len(sample))] + [self.classProba[cls]])
            return np.sum(np.log(probas))
        return np.array([predict(sample, cls) for cls in self.classes])


class GaussianNaiveBayes(Classifier):
    """
    Gaussian Naive Bayes classifier for continuous features.
    """

    def __init__(self):
        super().__init__()
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model according to the given training data.

        Parameters:
        X (numpy.ndarray): Training features.
        y (numpy.ndarray): Training labels.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters:
        X (numpy.ndarray): Test features.

        Returns:
        numpy.ndarray: Predicted class labels for X.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Parameters:
        x (numpy.ndarray): The input sample.

        Returns:
        int: Predicted class label.
        """
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a given class and sample.

        Parameters:
        class_idx (int): Index of the class.
        x (numpy.ndarray): The input sample.

        Returns:
        numpy.ndarray: Probability density values.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_sample_proba(self, x):
        """
        Predict the class probabilities for a single sample.

        Parameters:
        x (numpy.ndarray): The input sample.

        Returns:
        numpy.ndarray: Predicted class probabilities for the sample.
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(np.exp(posterior))
        return np.array(posteriors) / np.sum(posteriors)
