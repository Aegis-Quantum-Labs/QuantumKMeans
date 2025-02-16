import numpy as np
from .utils import quantum_random_init

def quantum_kmeans(X, k, max_iter=100, tol=1e-4, random_state=None):
    """
    Quantum-inspired k-means clustering.

    Parameters:
    - X : np.ndarray
        Data points as a 2D array of shape (n_samples, n_features).
    - k : int
        Number of clusters.
    - max_iter : int
        Maximum number of iterations.
    - tol : float
        Tolerance for convergence.
    - random_state : int or None
        Seed for random initialization.

    Returns:
    - labels : np.ndarray
        Cluster assignment for each data point.
    - centroids : np.ndarray
        Final centroid positions.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Quantum-inspired random initialization of centroids
    centroids = quantum_random_init(X, k)

    for iteration in range(max_iter):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Assign each sample to the nearest centroid
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(k)])
        # Check for convergence
        shift = np.linalg.norm(centroids - new_centroids)
        if shift < tol:
            print(f"Converged after {iteration+1} iterations.")
            break
        centroids = new_centroids

    return labels, centroids