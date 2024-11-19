Enhanced Fuzzy C-Means Algorithm
================================

Introduction
------------
Fuzzy C-Means (FCM) is a clustering algorithm where data points belong to clusters with varying degrees of membership. Unlike hard clustering methods (e.g., k-means), FCM allows partial membership, making it suitable for data with overlapping clusters.

Mathematical Overview
----------------------
The Fuzzy C-Means algorithm minimizes the following objective function:

.. math::

   J_m = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \| x_i - c_j \|^2

Where:

- :math:`N`: Number of data points.
- :math:`C`: Number of clusters.
- :math:`u_{ij}`: Membership degree of data point :math:`x_i` in cluster :math:`j`.
- :math:`m > 1`: Fuzziness coefficient.
- :math:`\| x_i - c_j \|`: Euclidean distance between :math:`x_i` and cluster center :math:`c_j`.

Algorithm Steps
^^^^^^^^^^^^^^^

1. **Initialize**: Randomly initialize cluster centers and membership matrix.

2. **Update Cluster Centers**:

   .. math::

      c_j = \frac{\sum_{i=1}^{N} u_{ij}^m x_i}{\sum_{i=1}^{N} u_{ij}^m}

3. **Update Membership Matrix**:

   .. math::

      u_{ij} = \frac{1}{\sum_{k=1}^{C} \left( \frac{\| x_i - c_j \|}{\| x_i - c_k \|} \right)^{\frac{2}{m-1}}}

4. **Convergence Check**: Stop if:

   .. math::

      \| U^{(t)} - U^{(t-1)} \| < \epsilon

Python Implementation
---------------------
The following Python implementation captures the FCM algorithm using a reusable class.

Code Overview
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from sklearn.base import BaseEstimator, ClusterMixin
   from scipy.spatial.distance import cdist

   class FuzzyCMeans(BaseEstimator, ClusterMixin):
       """
       Fuzzy C-Means clustering algorithm.
       """

Parameters
^^^^^^^^^^
- ``n_clusters``: Number of clusters (:math:`C`).
- ``m``: Fuzziness coefficient (:math:`m`).
- ``max_iter``: Maximum number of iterations.
- ``tol``: Convergence threshold (:math:`\epsilon`).
- ``random_state``: Seed for reproducibility.

Initialization
^^^^^^^^^^^^^^
The constructor initializes parameters:

.. code-block:: python

   def __init__(self, n_clusters=3, m=2, max_iter=150, tol=1e-4, random_state=None):
       self.n_clusters = n_clusters
       self.m = m
       self.max_iter = max_iter
       self.tol = tol
       self.random_state = random_state

Fitting the Model
^^^^^^^^^^^^^^^^^
The `fit` method executes the FCM algorithm:

1. Initialize cluster centers with K-Means++.
2. Randomly initialize the membership matrix.
3. Iteratively update cluster centers and membership matrix until convergence.

.. code-block:: python

   def fit(self, X, y=None):
       n_samples, n_features = X.shape
       self.X_ = X

       # Step 1: Initialize cluster centers
       self.cluster_centers_ = self._kmeans_plusplus_init(X)

       # Step 2: Initialize membership matrix
       self.U_ = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)

       for iteration in range(self.max_iter):
           U_prev = self.U_.copy()

           # Step 3: Update cluster centers
           self.cluster_centers_ = self._compute_centers(X)

           # Step 4: Update membership matrix
           self.U_ = self._update_memberships(X)

           # Step 5: Check for convergence
           if np.linalg.norm(self.U_ - U_prev) < self.tol:
               break

       return self

Helper Methods
^^^^^^^^^^^^^^

1. **Initialize Cluster Centers**:
   Cluster centers are initialized using K-Means++ to improve convergence speed.

   .. code-block:: python

      def _kmeans_plusplus_init(self, X):
          np.random.seed(self.random_state)
          centers = []
          initial_idx = np.random.choice(n_samples)
          centers.append(X[initial_idx])

          for _ in range(1, self.n_clusters):
              distances = np.min(cdist(X, np.array(centers)), axis=1)
              probabilities = distances ** 2 / np.sum(distances ** 2)
              next_idx = np.random.choice(n_samples, p=probabilities)
              centers.append(X[next_idx])

          return np.array(centers)

2. **Update Cluster Centers**:
   Update the cluster centers based on the current membership matrix:

   .. math::

      c_j = \frac{\sum_{i=1}^{N} u_{ij}^m x_i}{\sum_{i=1}^{N} u_{ij}^m}

   .. code-block:: python

      def _compute_centers(self, X):
          um = self.U_ ** self.m
          return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

3. **Update Membership Matrix**:
   Update membership values using:

   .. math::

      u_{ij} = \frac{1}{\sum_{k=1}^{C} \left( \frac{\| x_i - c_j \|}{\| x_i - c_k \|} \right)^{\frac{2}{m-1}}}

   .. code-block:: python

      def _update_memberships(self, X):
          distances = cdist(X, self.cluster_centers_, metric="euclidean")
          distances = np.fmax(distances, 1e-10)
          inv_distances = 1.0 / distances
          power = 2 / (self.m - 1)
          return inv_distances ** power / np.sum(inv_distances ** power, axis=1, keepdims=True)

4. **Prediction**:
   Assign hard cluster labels based on the maximum membership degree:

   .. code-block:: python

      def predict(self, X):
          memberships = self._compute_memberships(X)
          return memberships.argmax(axis=1)

Applications
------------
- **Image Segmentation**: Identify meaningful regions in images (e.g., medical imaging).
- **Pattern Recognition**: Classify overlapping data (e.g., handwriting recognition).
- **Data Mining**: Extract insights from datasets with fuzzy boundaries.

References
----------
1. J.C. Bezdek, "Pattern Recognition with Fuzzy Objective Function Algorithms," Springer, 2013.
2. T.J. Ross, "Fuzzy Logic with Engineering Applications," Wiley, 2010.
3. Duda, R.O., et al., "Pattern Classification," 2nd Edition, Wiley-Interscience, 2000.
