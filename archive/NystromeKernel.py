import numpy as np

def rbf_kernel_approximation(X, D, gamma):
    """
    Approximate RBF kernel using Random Fourier Features.
    
    Parameters:
    X : ndarray
        Data matrix of shape (n_samples, n_features).
    D : int
        Number of random features.
    gamma : float
        Parameter for the RBF kernel (gamma = 1 / (2 * sigma^2)).
    
    Returns:
    Z : ndarray
        Transformed feature matrix of shape (n_samples, D).
    """
    n_samples, n_features = X.shape
    omega = np.sqrt(2 * gamma) * np.random.randn(n_features, D)
    b = 2 * np.pi * np.random.rand(D)

    print(len(omega))
    print(len(b))
    
    
    Z = np.sqrt(2.0 / D) * np.cos(X @ omega + b)
    return Z

# Example usage
X = np.random.randn(1000, 50)  # 1000 samples, 50 features
D = 200  # Number of random features
gamma = 1.0 / 50  # RBF kernel parameter

# Compute the random Fourier feature map
Z = rbf_kernel_approximation(X, D, gamma)

# Approximate kernel matrix
K_approx = Z @ Z.T

K_approx