"""
NumPy implementation of feature operations for ember_ml.

This module provides NumPy implementations of feature extraction and transformation operations.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from ember_ml.backend.numpy.types import TensorLike

def pca(
    X: TensorLike,
    n_components: Optional[int] = None,
    *,
    whiten: bool = False,
    center: bool = True,
    svd_solver: str = "auto",
) -> Dict[str, Any]:
    """
    Principal Component Analysis (PCA) using NumPy.
    
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        n_components: Number of components to keep. If None, all components are kept.
        whiten: When True, the components are divided by the singular values
            and multiplied by the square root of n_samples.
        center: When True, X will be centered before computation.
        svd_solver: SVD solver to use:
            - 'auto': Auto-select based on data shape
            - 'full': Use full SVD
            - 'randomized': Use randomized SVD
    
    Returns:
        Dictionary containing:
        - components: Principal axes in feature space (n_components, n_features)
        - explained_variance: Amount of variance explained by each component
        - explained_variance_ratio: Percentage of variance explained by each component
        - mean: Per-feature empirical mean, used for centering
        - singular_values: Singular values corresponding to each component
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    X_tensor = tensor.convert_to_tensor(X)
    n_samples, n_features = X_tensor.shape
    
    # Handle n_components
    if n_components is None:
        n_components = min(n_samples, n_features)
    elif n_components > min(n_samples, n_features):
        n_components = min(n_samples, n_features)
    
    # Center data
    if center:
        mean = np.mean(X_tensor, axis=0)
        X_centered = X_tensor - mean
    else:
        mean = np.zeros(n_features)
        X_centered = X_tensor
    
    # Choose SVD solver using numpy operations to avoid direct Python operators
    if svd_solver == "auto":
        max_dim = np.max(np.array(X_tensor.shape))
        min_dim = np.min(np.array(X_tensor.shape))
        
        if max_dim <= 500:
            svd_solver = "full"
        elif n_components is not None and n_components < np.multiply(np.array(0.8), min_dim).item():
            svd_solver = "randomized"
        else:
            svd_solver = "full"
    
    # Perform SVD
    if svd_solver == "full":
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    elif svd_solver == "randomized":
        from sklearn.utils.extmath import randomized_svd
        # Ensure n_components is not None for randomized_svd
        n_components_value = n_components if n_components is not None else min(X_tensor.shape)
        U, S, Vt = randomized_svd(
            X_centered, n_components=int(n_components_value), random_state=42
        )
    else:
        raise ValueError(f"Unrecognized svd_solver='{svd_solver}'")
    
    # Get variance explained by singular values
    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_var
    
    # Truncate to n_components
    components = Vt[:n_components]
    explained_variance = explained_variance[:n_components]
    explained_variance_ratio = explained_variance_ratio[:n_components]
    singular_values = S[:n_components]
    
    return {
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "mean": mean,
        "singular_values": singular_values,
    }


def transform(
    X: TensorLike,
    components: Any,
    mean: Optional[Any] = None,
    *,
    whiten: bool = False,
    explained_variance: Optional[Any] = None,
) -> Any:
    """
    Apply dimensionality reduction to X using NumPy.
    
    X is projected on the components previously extracted.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        components: Principal axes in feature space (n_components, n_features)
        mean: Per-feature empirical mean, used for centering. If None, no centering is performed.
        whiten: When True, the components are divided by the singular values
            and multiplied by the square root of n_samples.
        explained_variance: Variance of each component, used for whitening
    
    Returns:
        X_new: Transformed values of shape (n_samples, n_components)
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    X_tensor = tensor.convert_to_tensor(X)
    components_tensor = tensor.convert_to_tensor(components)
    
    # Center data
    if mean is not None:
        mean_tensor = tensor.convert_to_tensor(mean)
        X_centered = X_tensor - mean_tensor
    else:
        X_centered = X_tensor
    
    # Project data
    X_transformed = np.dot(X_centered, components_tensor.T)
    
    # Whiten if requested
    if whiten:
        if explained_variance is None:
            raise ValueError("explained_variance must be provided when whiten=True")
        explained_variance_tensor = tensor.convert_to_tensor(explained_variance)
        # Avoid division by zero
        eps = np.finfo(X_transformed.dtype).eps
        scale = np.sqrt(explained_variance_tensor.clip(eps))
        X_transformed /= scale
    
    return X_transformed


def inverse_transform(
    X: TensorLike,
    components: Any,
    mean: Optional[Any] = None,
    *,
    whiten: bool = False,
    explained_variance: Optional[Any] = None,
) -> Any:
    """
    Transform data back to its original space using NumPy.
    
    Args:
        X: Input data of shape (n_samples, n_components)
        components: Principal axes in feature space (n_components, n_features)
        mean: Per-feature empirical mean, used for centering. If None, no centering is performed.
        whiten: When True, the components are divided by the singular values
            and multiplied by the square root of n_samples.
        explained_variance: Variance of each component, used for whitening
    
    Returns:
        X_original: Original data of shape (n_samples, n_features)
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    X_tensor = tensor.convert_to_tensor(X)
    components_tensor = tensor.convert_to_tensor(components)
    
    # Unwhiten if needed
    if whiten:
        if explained_variance is None:
            raise ValueError("explained_variance must be provided when whiten=True")
        explained_variance_tensor = tensor.convert_to_tensor(explained_variance)
        # Avoid division by zero
        eps = np.finfo(X_tensor.dtype).eps
        scale = np.sqrt(explained_variance_tensor.clip(eps))
        X_unwhitened = X_tensor * scale
    else:
        X_unwhitened = X_tensor
    
    # Project back to original space
    X_original = np.dot(X_unwhitened, components_tensor)
    
    # Add mean if provided
    if mean is not None:
        mean_tensor = tensor.convert_to_tensor(mean)
        X_original += mean_tensor
    
    return X_original


def standardize(
    X: TensorLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    axis: int = 0,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Standardize features by removing the mean and scaling to unit variance using NumPy.
    
    Args:
        X: Input data
        with_mean: If True, center the data before scaling
        with_std: If True, scale the data to unit variance
        axis: Axis along which to standardize
    
    Returns:
        Tuple containing:
        - X_scaled: Standardized data
        - mean: Mean values used for centering (None if with_mean=False)
        - std: Standard deviation values used for scaling (None if with_std=False)
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    X_tensor = tensor.convert_to_tensor(X)
    
    mean = None
    std = None
    
    # Center data
    if with_mean:
        mean = np.mean(X_tensor, axis=axis, keepdims=True)
        X_centered = X_tensor - mean
    else:
        X_centered = X_tensor
    
    # Scale data
    if with_std:
        std = np.std(X_tensor, axis=axis, keepdims=True)
        # Avoid division by zero
        eps = np.finfo(X_tensor.dtype).eps
        std_clipped = np.maximum(std, eps)
        X_scaled = X_centered / std_clipped
    else:
        X_scaled = X_centered
    
    # Remove singleton dimensions if not keeping dims
    if mean is not None and mean.shape[axis] == 1:
        mean = np.squeeze(mean, axis=axis)
    if std is not None and std.shape[axis] == 1:
        std = np.squeeze(std, axis=axis)
    
    return X_scaled, mean, std


def normalize(
    X: TensorLike,
    *,
    norm: str = "l2",
    axis: int = 1,
) -> Any:
    """
    Scale input vectors individually to unit norm using NumPy.
    
    Args:
        X: Input data
        norm: The norm to use:
            - 'l1': Sum of absolute values
            - 'l2': Square root of sum of squares
            - 'max': Maximum absolute value
        axis: Axis along which to normalize
    
    Returns:
        X_normalized: Normalized data
    """
    from ember_ml.backend.numpy.tensor import NumpyTensor
    tensor = NumpyTensor()
    X_tensor = tensor.convert_to_tensor(X)
    
    if norm == "l1":
        norms = np.sum(np.abs(X_tensor), axis=axis, keepdims=True)
    elif norm == "l2":
        norms = np.sqrt(np.sum(X_tensor ** 2, axis=axis, keepdims=True))
    elif norm == "max":
        norms = np.max(np.abs(X_tensor), axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    
    # Avoid division by zero
    eps = np.finfo(X_tensor.dtype).eps
    norms_clipped = np.maximum(norms, eps)
    
    return X_tensor / norms_clipped


# Removed NumpyFeatureOps class as it's redundant with standalone functions