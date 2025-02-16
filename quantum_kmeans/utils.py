import numpy as np

def quantum_random_init(X, k):
    """
    Quantum-inspired random initialization for k-means.
    
    This function selects k random samples from X, simulating the 
    idea of quantum superposition for initial state selection.
    
    Parameters:
    - X : np.ndarray
        Data points as a 2D array.
    - k : int
        Number of clusters.
    
    Returns:
    - centroids : np.ndarray
        Initial centroids.
    """
    n_samples = X.shape[0]
    # Using np.random.choice to simulate the probabilistic nature
    indices = np.random.choice(n_samples, size=k, replace=False)
    return X[indices]