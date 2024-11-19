import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist


class FuzzyCMeans(BaseEstimator, ClusterMixin):
    """
    Fuzzy C-Means clustering.

    Fuzzy C-Means is a clustering algorithm where each data point belongs to a cluster
    with a degree of membership. It is particularly useful when data points can belong
    to multiple clusters simultaneously.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.

    m : float, default=2
        Fuzziness coefficient. Must be greater than 1. Higher values lead to fuzzier clusters.

    max_iter : int, default=150
        Maximum number of iterations of the algorithm for convergence.

    tol : float, default=1e-4
        Convergence tolerance. The algorithm stops when the change in membership matrix
        is less than this value.

    random_state : int or None, default=None
        Determines the random number generation for initialization.
        Use an int to make the randomness deterministic.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    U_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy membership matrix, where `U_[i, j]` is the degree of membership
        of sample `i` to cluster `j`.

    X_ : ndarray of shape (n_samples, n_features)
        The input data.

    Examples
    --------
    >>> from fuzzy_cmeans import FuzzyCMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]])
    >>> fcm = FuzzyCMeans(n_clusters=2, m=2, random_state=42)
    >>> fcm.fit(X)
    FuzzyCMeans(...)
    >>> fcm.cluster_centers_
    array([[ 2.,  3.],
           [11., 12.]])
    >>> fcm.predict(X)
    array([0, 0, 0, 1, 1, 1])
    """

    def __init__(self, n_clusters=3, m=2, max_iter=150, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Compute Fuzzy C-Means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
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
        Predict hard cluster labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each data point.
        """
        memberships = self._compute_memberships(X)
        return memberships.argmax(axis=1)

    def fit_predict(self, X, y=None):
        """
        Compute Fuzzy C-Means clustering and return hard cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each data point.
        """
        self.fit(X, y)
        return self.predict(X)

    def _kmeans_plusplus_init(self, X):
        """
        Initialize cluster centers using the K-Means++ algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initialized cluster centers.
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Updated cluster centers.
        """
        um = self.U_ ** self.m  # Fuzzy memberships raised to the power of m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def _update_memberships(self, X):
        """
        Update the fuzzy membership matrix based on the current cluster centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        memberships : ndarray of shape (n_samples, n_clusters)
            Updated membership matrix.
        """
        distances = cdist(X, self.cluster_centers_, metric="euclidean")
        distances = np.fmax(distances, 1e-10)  # Avoid division by zero
        inv_distances = 1.0 / distances
        power = 2 / (self.m - 1)
        return inv_distances ** power / np.sum(inv_distances ** power, axis=1, keepdims=True)

    def _compute_memberships(self, X):
        """
        Compute membership values for new data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data for membership computation.

        Returns
        -------
        memberships : ndarray of shape (n_samples, n_clusters)
            Membership values for each data point.
        """
        distances = cdist(X, self.cluster_centers_, metric="euclidean")
        distances = np.fmax(distances, 1e-10)
        inv_distances = 1.0 / distances
        power = 2 / (self.m - 1)
        return inv_distances ** power / np.sum(inv_distances ** power, axis=1, keepdims=True)
