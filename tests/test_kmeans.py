import numpy as np
import unittest
from quantum_kmeans.kmeans import quantum_kmeans

class TestQuantumKMeans(unittest.TestCase):
    def test_convergence(self):
        # Create a simple dataset: two clusters in 2D
        X = np.vstack([
            np.random.normal(loc=0.0, scale=0.5, size=(50, 2)),
            np.random.normal(loc=5.0, scale=0.5, size=(50, 2))
        ])
        labels, centroids = quantum_kmeans(X, k=2, random_state=123)
        self.assertEqual(len(centroids), 2)
        # Expect the centroids to be roughly around 0.0 and 5.0 in one dimension
        self.assertTrue(np.abs(np.mean(centroids[:, 0]) - 2.5) < 2.0)

if __name__ == '__main__':
    unittest.main()