# FuzzyCMeans: A Fuzzy C-Means Clustering Extension for Scikit-Learn

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

`FuzzyCMeans` is a Scikit-Learn-compatible implementation of the **Fuzzy C-Means** clustering algorithm. This algorithm extends traditional clustering techniques by allowing data points to belong to multiple clusters with varying degrees of membership, making it ideal for datasets with overlapping or ambiguous cluster boundaries.

This package is designed to seamlessly integrate with the Scikit-Learn ecosystem, offering a familiar API and compatibility with pipelines, transformers, and other utilities.

---

## Features

- **Fuzzy Membership**: Assigns data points to clusters with degrees of membership, offering more flexibility than hard clustering methods.
- **Scikit-Learn Compatible**: Implements the `BaseEstimator` and `ClusterMixin` interfaces for easy integration.
- **K-Means++ Initialization**: Utilizes the robust K-Means++ algorithm for initializing cluster centers.
- **Customizable Parameters**: Adjust the number of clusters, fuzziness coefficient, convergence tolerance, and maximum iterations.
- **Reproducibility**: Supports setting a random seed for reproducible results.

---

## Installation

### Using pip
```bash
pip install scikit-fuzzy-c-means
```

### Manual Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/fuzzy-c-means
cd fuzzy-c-means
pip install -r requirements.txt
```

---

## Quick Start

### Import and Initialize
```python
from fuzzy_c_means import FuzzyCMeans
import numpy as np

# Sample dataset
X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

# Initialize the FuzzyCMeans model
fcm = FuzzyCMeans(n_clusters=2, m=2, max_iter=150, tol=1e-4, random_state=42)
```

### Fit and Predict
```python
# Fit the model to the data
fcm.fit(X)

# Predict cluster labels
labels = fcm.predict(X)

# View cluster centers
print("Cluster Centers:")
print(fcm.cluster_centers_)
```

---

## Parameters

| Parameter     | Type   | Description                                                                                 | Default   |
|---------------|--------|---------------------------------------------------------------------------------------------|-----------|
| `n_clusters`  | `int`  | Number of clusters to find.                                                                 | `3`       |
| `m`           | `float`| Fuzziness coefficient. Larger values lead to fuzzier clusters (`m > 1`).                     | `2`       |
| `max_iter`    | `int`  | Maximum number of iterations for the algorithm to converge.                                 | `150`     |
| `tol`         | `float`| Convergence tolerance. Stops iteration when changes are smaller than this value.            | `1e-4`    |
| `random_state`| `int`  | Random seed for reproducibility.                                                            | `None`    |

---

## Methods

### `fit(X, y=None)`
Fits the Fuzzy C-Means model to the input data `X`.

### `predict(X)`
Predicts hard cluster labels for the input data `X` based on fuzzy memberships.

### `fit_predict(X, y=None)`
Fits the model to `X` and returns the predicted hard cluster labels.

### `cluster_centers_`
Returns the cluster centers after fitting the model.

---

## Use Cases

1. **Image Segmentation**: Separate regions of an image into clusters based on color, texture, or intensity.
2. **Market Segmentation**: Group customers into fuzzy clusters based on purchasing behavior and demographics.
3. **Anomaly Detection**: Identify data points with low membership to any cluster as potential outliers.
4. **Genomics**: Analyze genetic data to classify genes or species with ambiguous relationships.

---

## Advantages

- **Soft Clustering**: Unlike K-Means, Fuzzy C-Means allows for nuanced assignments of data points to multiple clusters.
- **Customizable Fuzziness**: Control the degree of fuzziness with the `m` parameter.
- **Robust Initialization**: Reduces the impact of poor initializations with K-Means++.

---

## Requirements

- Python 3.9 or higher
- NumPy
- SciPy
- Scikit-Learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Feel free to submit a pull request or report issues in the [GitHub repository](https://github.com/your-repo/fuzzy-c-means).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Keywords

- Fuzzy C-Means
- Soft Clustering
- Machine Learning
- Scikit-Learn
- Python
- Clustering Algorithm
- Data Science

---

## References

- Bezdek, J.C. "Pattern Recognition with Fuzzy Objective Function Algorithms." Springer, 1981.
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

---

## Acknowledgments

This implementation is inspired by the principles of fuzzy clustering and integrates seamlessly with Scikit-Learn for ease of use in real-world machine learning pipelines.
