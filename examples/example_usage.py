import numpy as np
import matplotlib.pyplot as plt
from quantum_kmeans.kmeans import quantum_kmeans
from sklearn.datasets import make_blobs

# Generate synthetic 2D data (e.g., blobs)
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Run quantum-inspired k-means clustering
labels, centroids = quantum_kmeans(X, k=3, random_state=42)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')
plt.title("Quantum-Inspired k-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()