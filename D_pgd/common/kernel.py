"""Dense RBF dual matrix for the small-data correctness experiment."""

import numpy as np


def rbf_dual_matrix(X, y, gamma):
    """Return Q[i,j] = y[i] * y[j] * exp(-gamma * ||X[i]-X[j]||²)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or not len(y):
        raise ValueError("Expected X with shape (n, d) and y with shape (n,), n > 0")
    if not np.isfinite(X).all() or not np.isin(y, (-1, 1)).all():
        raise ValueError("Expected finite features and labels in {-1, +1}")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")

    Q = np.empty((len(y), len(y)), dtype=np.float64)
    for i, sample in enumerate(X):
        delta = X - sample
        distances_sq = np.einsum("ij,ij->i", delta, delta)
        Q[i] = y[i] * y * np.exp(-gamma * distances_sq)
    return Q
