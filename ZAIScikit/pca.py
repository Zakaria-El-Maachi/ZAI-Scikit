import numpy as np
from sklearn_base import Transformer

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    n_components (int): Number of principal components to keep.

    Attributes:
    components (ndarray): Principal components.
    mean_vector (ndarray): Mean of each feature in the training data.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean_vector = None

    def fit(self, data_matrix, y=None):
        """
        Fit the PCA model to the training data.

        Args:
        data_matrix (ndarray): Training data.
        """
        # Mean centering
        self.mean_vector = np.mean(data_matrix, axis=0)
        centered_data = data_matrix - self.mean_vector

        # Compute covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components].T

    def transform(self, data_matrix):
        """
        Transform the data to the new space defined by the principal components.

        Args:
        data_matrix (ndarray): The data to transform.

        Returns:
        ndarray: Transformed data.
        """
        centered_data = data_matrix - self.mean_vector
        return np.dot(centered_data, self.components.T)
