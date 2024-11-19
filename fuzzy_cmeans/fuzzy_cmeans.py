import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist


class FuzzyCMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, m=2, max_iter=150, tol=1e-4, random_state=None):
        """
        Parameters:
        n_clusters (int): Number of clusters.
        m (float): Fuzziness coefficient (m > 1). Typical value is 2.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        random_state (int): Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """
        Fit the Fuzzy C-Means model to the data.
        """
        n_samples, n_features = X.shape
        self.X_ = X

        # Initialize cluster centers using K-Means++ strategy
        self.cluster_centers_ = self._kmeans_plusplus_init(X)

        # Initialize fuzzy membership matrix randomly
        self.U_ = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        
        for iteration in range(self.max_iter):
            U_prev = self.U_.copy()
            
            # Update cluster centers
            self.cluster_centers_ = self._compute_centers(X)
            
            # Update membership matrix
            self.U_ = self._update_memberships(X)
            
            # Convergence check
            if np.linalg.norm(self.U_ - U_prev) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        """
        Predict hard cluster labels based on fuzzy membership.
        """
        memberships = self._compute_memberships(X)
        return memberships.argmax(axis=1)
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and return hard cluster labels.
        """
        self.fit(X, y)
        return self.predict(X)
    
    def _kmeans_plusplus_init(self, X):
        """
        Initialize cluster centers using K-Means++ algorithm.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Step 1: Randomly choose the first cluster center
        centers = []
        initial_idx = np.random.choice(n_samples)
        centers.append(X[initial_idx])
        
        for _ in range(1, self.n_clusters):
            # Step 2: Compute distances to the nearest cluster center
            distances = np.min(cdist(X, np.array(centers)), axis=1)
            
            # Step 3: Compute probabilities proportional to distances
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            # Step 4: Choose the next cluster center based on probabilities
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(X[next_idx])
        
        return np.array(centers)
    
    def _compute_centers(self, X):
        """
        Compute cluster centers based on the current membership matrix.
        """
        um = self.U_ ** self.m  # Fuzzy memberships raised to the power of m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
    
    def _update_memberships(self, X):
        """
        Update the fuzzy membership matrix based on the current cluster centers.
        """
        distances = cdist(X, self.cluster_centers_, metric="euclidean")
        distances = np.fmax(distances, 1e-10)  # Avoid division by zero
        inv_distances = 1.0 / distances
        power = 2 / (self.m - 1)
        return inv_distances ** power / np.sum(inv_distances ** power, axis=1, keepdims=True)
    
    def _compute_memberships(self, X):
        """
        Compute membership values for new data points.
        """
        distances = cdist(X, self.cluster_centers_, metric="euclidean")
        distances = np.fmax(distances, 1e-10)
        inv_distances = 1.0 / distances
        power = 2 / (self.m - 1)
        return inv_distances ** power / np.sum(inv_distances ** power, axis=1, keepdims=True)
