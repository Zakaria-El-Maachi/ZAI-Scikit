import numpy as np
from sklearn_base import Transformer
from utilities import euclidean_distance, chebyshev_distance, manhattan_distance, cosine_distance

class KMeans(Transformer):
    """
    K-Means clustering algorithm.

    Parameters:
    num_clusters (int): Number of clusters to form.
    max_iterations (int): Maximum number of iterations for clustering.

    Attributes:
    clusters (list): Indices of samples in each cluster.
    centroids (ndarray): Centroids of the clusters.
    """

    def __init__(self, num_clusters=5, max_iterations=100, distance_metric='euclidean'):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric

        # The centers (mean vector) for each cluster
        self.centroids = []

        # Mapping distance metrics to functions
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'chebyshev': chebyshev_distance,
            'cosine': cosine_distance
        }

    def fit(self, data, y=None):
        self.data = data
        self.num_samples, self.num_features = data.shape

        # Initialize centroids
        random_sample_indices = np.random.choice(self.num_samples, self.num_clusters, replace=False)
        self.centroids = [self.data[idx] for idx in random_sample_indices]

        # Optimize clusters
        for _ in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            old_centroids = self.centroids
            self.centroids = self._compute_centroids(clusters)

            if self._is_converged(old_centroids, self.centroids):
                break
    
    def transform(self, data, y=None):
        num_samples, _ = data.shape
        cluster_labels = np.zeros(num_samples)

        for i, sample in enumerate(data):
            # Find the index of the closest centroid for the current sample
            closest_centroid_idx = self._closest_centroid(sample, self.centroids)
            cluster_labels[i] = closest_centroid_idx

        return cluster_labels


    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.num_clusters)]
        for idx, sample in enumerate(self.data):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [self.distance_functions[self.distance_metric](sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _compute_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.num_clusters, self.num_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        # Distances between old and new centroids, for all centroids
        distances = [self.distance_functions[self.distance_metric](old_centroids[i], new_centroids[i]) for i in range(self.num_clusters)]
        return sum(distances) == 0
