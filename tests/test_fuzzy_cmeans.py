import pytest
import numpy as np
from sklearn.datasets import make_blobs
from fuzzy_cmeans import FuzzyCMeans

# Fixtures for reusable test data
@pytest.fixture
def sample_data():
    # Generate a simple dataset with 3 clusters
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X

@pytest.fixture
def fcm_model():
    # Provide a default FuzzyCMeans instance
    return FuzzyCMeans(n_clusters=3, m=2, random_state=42)

# Test Initialization
def test_initialization():
    model = FuzzyCMeans(n_clusters=3, m=2, max_iter=100, tol=1e-4, random_state=42)
    assert model.n_clusters == 3
    assert model.m == 2
    assert model.max_iter == 100
    assert model.tol == 1e-4
    assert model.random_state == 42

# Test Fitting
def test_fit(sample_data, fcm_model):
    fcm_model.fit(sample_data)
    assert fcm_model.cluster_centers_.shape == (3, sample_data.shape[1])
    assert fcm_model.U_.shape == (sample_data.shape[0], 3)
    assert np.allclose(fcm_model.U_.sum(axis=1), 1)  # Memberships sum to 1

# Test Predict
def test_predict(sample_data, fcm_model):
    fcm_model.fit(sample_data)
    labels = fcm_model.predict(sample_data)
    assert labels.shape == (sample_data.shape[0],)
    assert labels.dtype == int

# Test Fit and Predict Combined
def test_fit_predict(sample_data, fcm_model):
    labels = fcm_model.fit_predict(sample_data)
    assert labels.shape == (sample_data.shape[0],)
    assert labels.dtype == int

# Test K-Means++ Initialization
def test_kmeans_plusplus_init(sample_data, fcm_model):
    centers = fcm_model._kmeans_plusplus_init(sample_data)
    assert centers.shape == (3, sample_data.shape[1])
    assert np.all(np.isfinite(centers))

# Test Convergence
def test_convergence(sample_data, fcm_model):
    fcm_model.fit(sample_data)
    assert fcm_model.max_iter > 0
    assert hasattr(fcm_model, "cluster_centers_")
    assert hasattr(fcm_model, "U_")

# Test Membership Updates
def test_membership_updates(sample_data, fcm_model):
    fcm_model.fit(sample_data)
    memberships = fcm_model._update_memberships(sample_data)
    assert memberships.shape == (sample_data.shape[0], fcm_model.n_clusters)
    assert np.allclose(memberships.sum(axis=1), 1)

# Test Edge Case: Single Cluster
def test_single_cluster(sample_data):
    model = FuzzyCMeans(n_clusters=1, m=2, random_state=42)
    model.fit(sample_data)
    assert model.cluster_centers_.shape == (1, sample_data.shape[1])
    assert model.U_.shape == (sample_data.shape[0], 1)
    assert np.allclose(model.U_, 1)  # All points belong fully to one cluster

# Test Edge Case: High Fuzziness
def test_high_fuzziness(sample_data):
    model = FuzzyCMeans(n_clusters=3, m=10, random_state=42)
    model.fit(sample_data)
    memberships = model.U_
    assert memberships.shape == (sample_data.shape[0], 3)
    assert np.allclose(memberships.sum(axis=1), 1)

# Test Edge Case: Few Data Points
def test_few_data_points():
    X = np.array([[0, 0], [1, 1]])
    model = FuzzyCMeans(n_clusters=2, m=2, random_state=42)
    model.fit(X)
    assert model.cluster_centers_.shape == (2, 2)
    assert model.U_.shape == (2, 2)
    assert np.allclose(model.U_.sum(axis=1), 1)

# Test Tolerance Impact
def test_tolerance_impact(sample_data):
    model_1 = FuzzyCMeans(n_clusters=3, tol=1e-4, random_state=42)
    model_2 = FuzzyCMeans(n_clusters=3, tol=1e-2, random_state=42)
    model_1.fit(sample_data)
    model_2.fit(sample_data)
    assert model_1.cluster_centers_.shape == model_2.cluster_centers_.shape

# Test Different Random States
def test_random_state(sample_data):
    model_1 = FuzzyCMeans(n_clusters=3, random_state=42)
    model_2 = FuzzyCMeans(n_clusters=3, random_state=42)
    model_3 = FuzzyCMeans(n_clusters=3, random_state=7)
    model_1.fit(sample_data)
    model_2.fit(sample_data)
    model_3.fit(sample_data)
    assert np.allclose(model_1.cluster_centers_, model_2.cluster_centers_)
    assert not np.allclose(model_1.cluster_centers_, model_3.cluster_centers_)

# Test Invalid Parameters
def test_invalid_parameters():
    with pytest.raises(ValueError):
        FuzzyCMeans(n_clusters=0)
    with pytest.raises(ValueError):
        FuzzyCMeans(m=1)  # m must be > 1
    with pytest.raises(ValueError):
        FuzzyCMeans(max_iter=0)

