"""
Backend-agnostic implementation of Principal Component Analysis (PCA).

This module provides a PCA implementation using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Optional, Dict, Any, Union, Tuple
from math import log

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.ops.linearalg import svd
def _svd_flip(u, v):
    """Sign correction for SVD to ensure deterministic output.
    
    Adjusts the signs of the columns of u and rows of v such that
    the loadings in v are always positive.
    
    Args:
        u: Left singular vectors
        v: Right singular vectors (transposed)
        
    Returns:
        u_adjusted, v_adjusted: Adjusted singular vectors
    """
    # Columns of u, rows of v
    max_abs_cols = ops.argmax(ops.abs(u), axis=0)
    signs = ops.sign(ops.gather(u, max_abs_cols, axis=0))
    u = ops.multiply(u, signs)
    v = ops.multiply(v, signs[:, ops.newaxis])
    return u, v


def _find_ncomponents(
    n_components: Optional[Union[int, float, str]],
    n_samples: int,
    n_features: int,
    explained_variance: Any,
    explained_variance_ratio: Any = None,
) -> int:
    """Find the number of components to keep.
    
    Args:
        n_components: Number of components specified by the user
        n_samples: Number of samples
        n_features: Number of features
        explained_variance: Explained variance of each component
        explained_variance_ratio: Explained variance ratio of each component
        
    Returns:
        Number of components to keep
    """
    if n_components is None:
        n_components = min(n_samples, n_features)
    elif isinstance(n_components, float) and 0 < n_components < 1.0:
        # Compute number of components that explain at least n_components of variance
        ratio_cumsum = ops.cumsum(explained_variance_ratio)
        n_components = ops.add(ops.sum(ops.less(ratio_cumsum, n_components)), 1)
    elif n_components == 'mle':
        # Minka's MLE for selecting number of components
        n_components = _infer_dimensions(explained_variance, n_samples)
    
    # Ensure n_components is an integer and within bounds
    n_components = int(n_components)
    n_components = min(n_components, min(n_samples, n_features))
    
    return n_components


def _infer_dimensions(explained_variance, n_samples):
    """Infer the dimensions using Minka's MLE.
    
    Args:
        explained_variance: Explained variance of each component
        n_samples: Number of samples
        
    Returns:
        Number of components to keep
    """
    # Implementation of Minka's MLE for dimensionality selection
    n_components = explained_variance.shape[0]
    ll = tensor.zeros((n_components,))
    
    for i in range(n_components):
        if i < n_components - 1:
            sigma2 = ops.mean(explained_variance[i+1:])
            if sigma2 > 0:
                ll = ops.tensor_scatter_nd_update(
                    ll,
                    [[i]],
                    [ops.multiply(
                        ops.multiply(-0.5, n_samples),
                        ops.add(
                            ops.add(
                                ops.sum(ops.log(explained_variance[:i+1])),
                                ops.multiply(n_components - i - 1, ops.log(sigma2))
                            ),
                            ops.add(
                                ops.divide(n_components - i - 1, n_components - i),
                                ops.add(
                                    ops.divide(ops.sum(explained_variance[:i+1]), sigma2),
                                    ops.divide(ops.multiply(n_components - i - 1, sigma2), sigma2)
                                )
                            )
                        )
                    )]
                )
        else:
            ll = ops.tensor_scatter_nd_update(
                ll,
                [[i]],
                [ops.multiply(-0.5, ops.multiply(n_samples, ops.sum(ops.log(explained_variance))))]
            )
    
    return ops.add(ops.argmax(ll), 1)


def _randomized_svd(
    X: Any,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: Union[int, str] = 'auto',
    power_iteration_normalizer: str = 'auto',
    random_state: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Randomized SVD implementation using ops abstraction layer.
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of components to extract
        n_oversamples: Additional number of random vectors for more stable approximation
        n_iter: Number of power iterations
        power_iteration_normalizer: Normalization method for power iterations
        random_state: Random seed
        
    Returns:
        U, S, V: Left singular vectors, singular values, right singular vectors
    """
    n_samples, n_features = ops.shape(X)
    
    # Set random seed if provided
    if random_state is not None:
        ops.set_seed(random_state)
    
    # Handle n_iter parameter
    if n_iter == 'auto':
        # Heuristic: set n_iter based on matrix size
        if min(n_samples, n_features) <= 10:
            n_iter = 7
        else:
            n_iter = 4
    
    # Step 1: Sample random vectors
    n_random = min(n_components + n_oversamples, min(n_samples, n_features))
    Q = ops.random_normal((n_features, n_random))
    
    # Step 2: Compute Y = X * Q
    Y = ops.matmul(X, Q)
    
    # Step 3: Perform power iterations to increase accuracy
    for _ in range(n_iter):
        if power_iteration_normalizer == 'auto' or power_iteration_normalizer == 'QR':
            Q, _ = ops.qr(Y)
        elif power_iteration_normalizer == 'LU':
            # LU normalization not directly available in ops, use QR instead
            Q, _ = ops.qr(Y)
        else:  # 'none'
            Q = Y
        
        # Project X onto Q
        Y = ops.matmul(X, ops.matmul(ops.transpose(X), Q))
        
        if power_iteration_normalizer == 'auto' or power_iteration_normalizer == 'QR':
            Q, _ = ops.qr(Y)
        elif power_iteration_normalizer == 'LU':
            Q, _ = ops.qr(Y)
        else:  # 'none'
            Q = Y
    
    # Step 4: Compute QR decomposition of Y
    Q, _ = ops.qr(Y)
    
    # Step 5: Project X onto Q
    B = ops.matmul(ops.transpose(Q), X)
    
    # Step 6: Compute SVD of the small matrix B
    Uhat, S, V = ops.svd(B)
    U = ops.matmul(Q, Uhat)
    
    return U, S, V


class PCA:
    """Principal Component Analysis (PCA) implementation using ops abstraction layer.
    
    This implementation is backend-agnostic and works with all backends (NumPy, PyTorch, MLX).
    It implements the PCAInterface from ember_ml.features.interfaces.
    """
    
    def __init__(self):
        """Initialize PCA."""
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.noise_variance_ = None
        self.whiten_ = False
    
    def fit(
        self,
        X: Any,
        n_components: Optional[Union[int, float, str]] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> "PCA":
        """
        Fit the PCA model.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            n_components: Number of components to keep
            whiten: Whether to whiten the data
            center: Whether to center the data
            svd_solver: SVD solver to use
            
        Returns:
            Self
        """
        X_tensor = tensor.convert_to_tensor(X)
        self.n_samples_, self.n_features_ = tensor.shape(X_tensor)
        self.whiten_ = whiten
        
        # Choose SVD solver
        if svd_solver == "auto":
            if max(self.n_samples_, self.n_features_) <= 500:
                svd_solver = "full"
            elif n_components is not None and ops.less(
                n_components, 
                ops.multiply(0.8, tensor.convert_to_tensor(min(self.n_samples_, self.n_features_)))
            ):
                svd_solver = "randomized"
            else:
                svd_solver = "full"
        
        # Center data
        if center:
            self.mean_ = ops.mean(X_tensor, axis=0)
            X_centered = ops.subtract(X_tensor, self.mean_)
        else:
            self.mean_ = tensor.zeros((self.n_features_,))
            X_centered = X_tensor
        
        # Perform SVD
        if svd_solver == "full":
            U, S, V = svd(X_centered)
            # Explained variance
            explained_variance = ops.divide(ops.square(S), ops.subtract(self.n_samples_, 1))
            total_var = ops.sum(explained_variance)
            explained_variance_ratio = ops.divide(explained_variance, total_var)
        elif svd_solver == "randomized":
            if n_components is None:
                n_components = min(self.n_samples_, self.n_features_)
            elif not isinstance(n_components, int):
                raise ValueError("Randomized SVD only supports integer number of components")
            
            U, S, V = _randomized_svd(
                X_centered,
                n_components=n_components,
                n_oversamples=10,
                n_iter=7,
                power_iteration_normalizer='auto',
                random_state=None,
            )
            # Explained variance
            explained_variance = ops.divide(ops.square(S), ops.subtract(self.n_samples_, 1))
            total_var = ops.divide(ops.sum(ops.square(X_centered)), ops.subtract(self.n_samples_, 1))
            explained_variance_ratio = ops.divide(explained_variance, total_var)
        elif svd_solver == "covariance_eigh":
            # Compute covariance matrix
            cov = ops.divide(
                ops.matmul(ops.transpose(X_centered), X_centered),
                ops.subtract(self.n_samples_, 1)
            )
            # Eigendecomposition
            eigenvals, eigenvecs = ops.eigh(cov)
            # Sort in descending order
            idx = ops.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            # Fix numerical errors
            eigenvals = ops.clip(eigenvals, min_value=0.0)
            # Compute equivalent variables to full SVD output
            explained_variance = eigenvals
            total_var = ops.sum(explained_variance)
            explained_variance_ratio = ops.divide(explained_variance, total_var)
            S = ops.sqrt(ops.multiply(eigenvals, ops.subtract(self.n_samples_, 1)))
            V = ops.transpose(eigenvecs)
            U = None  # Not needed
        else:
            raise ValueError(f"Unrecognized svd_solver='{svd_solver}'")
        
        # Flip signs for deterministic output
        if U is not None and V is not None:
            U, V = _svd_flip(U, V)
        
        # Determine number of components
        self.n_components_ = _find_ncomponents(
            n_components=n_components,
            n_samples=self.n_samples_,
            n_features=self.n_features_,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
        )
        
        # Store results
        self.components_ = V[:self.n_components_]
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        
        # Compute noise variance
        if ops.less(self.n_components_, min(self.n_samples_, self.n_features_)):
            self.noise_variance_ = ops.mean(explained_variance[self.n_components_:])
        else:
            self.noise_variance_ = 0.0
        
        return self
    
    def transform(self, X: Any) -> Any:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            X_new: Transformed values of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        X_centered = ops.subtract(X_tensor, self.mean_)
        X_transformed = ops.matmul(X_centered, ops.transpose(self.components_))
        
        if self.whiten_:
            # Avoid division by zero
            eps = 1e-8  # Small constant to avoid division by zero
            scale = ops.sqrt(ops.clip(self.explained_variance_, min_value=eps))
            X_transformed = ops.divide(X_transformed, scale)
        
        return X_transformed
    
    def fit_transform(
        self,
        X: Any,
        n_components: Optional[Union[int, float, str]] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> Any:
        """
        Fit the model and apply dimensionality reduction.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            n_components: Number of components to keep
            whiten: Whether to whiten the data
            center: Whether to center the data
            svd_solver: SVD solver to use
            
        Returns:
            X_new: Transformed values of shape (n_samples, n_components)
        """
        self.fit(
            X,
            n_components=n_components,
            whiten=whiten,
            center=center,
            svd_solver=svd_solver,
        )
        return self.transform(X)
    
    def inverse_transform(self, X: Any) -> Any:
        """
        Transform data back to its original space.
        
        Args:
            X: Input data of shape (n_samples, n_components)
            
        Returns:
            X_original: Original data of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        X_tensor = tensor.convert_to_tensor(X)
        
        if self.whiten_:
            # Avoid division by zero
            eps = 1e-8  # Small constant to avoid division by zero
            scale = ops.sqrt(ops.clip(self.explained_variance_, min_value=eps))
            X_unwhitened = ops.multiply(X_tensor, scale)
        else:
            X_unwhitened = X_tensor
        
        X_original = ops.matmul(X_unwhitened, self.components_)
        X_original = ops.add(X_original, self.mean_)
        
        return X_original