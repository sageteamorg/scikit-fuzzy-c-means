from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)

# Visualize the dataset
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_true, palette="deep", legend=None)
plt.title("Generated Data with True Cluster Labels")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
