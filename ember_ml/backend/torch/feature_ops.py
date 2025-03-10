"""
PyTorch implementation of feature operations for ember_ml.

This module provides PyTorch implementations of feature extraction and transformation operations.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from ember_ml.backend.torch.tensor_ops import convert_to_tensor


def pca(
    X: Any,
    n_components: Optional[int] = None,
    *,
    whiten: bool = False,
    center: bool = True,
    svd_solver: str = "auto",
) -> Dict[str, Any]:
    """
    Principal Component Analysis (PCA) using PyTorch.
    
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
    X_tensor = convert_to_tensor(X)
    n_samples, n_features = X_tensor.shape
    
    # Handle n_components
    if n_components is None:
        n_components = min(n_samples, n_features)
    elif n_components > min(n_samples, n_features):
        n_components = min(n_samples, n_features)
    
    # Center data
    if center:
        mean = torch.mean(X_tensor, dim=0)
        X_centered = X_tensor - mean
    else:
        mean = torch.zeros(n_features, device=X_tensor.device, dtype=X_tensor.dtype)
        X_centered = X_tensor
    
    # Choose SVD solver
    if svd_solver == "auto":
        if max(X_tensor.shape) <= 500:
            svd_solver = "full"
        elif n_components < 0.8 * min(X_tensor.shape):
            svd_solver = "randomized"
        else:
            svd_solver = "full"
    
    # Perform SVD
    if svd_solver == "full":
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        # PyTorch returns V, not V^T, so we need to transpose
        Vt = Vt.T
    elif svd_solver == "randomized":
        # Implement randomized SVD for PyTorch
        # This is a simplified version of sklearn's randomized_svd
        n_random = min(n_components + 10, min(n_samples, n_features))
        Q = torch.randn(n_features, n_random, device=X_tensor.device, dtype=X_tensor.dtype)
        Q, _ = torch.linalg.qr(X_centered @ Q)
        
        # Project X onto Q
        B = Q.T @ X_centered
        
        # SVD of the small matrix B
        Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat
        # PyTorch returns V, not V^T, so we need to transpose
        Vt = Vt.T
    else:
        raise ValueError(f"Unrecognized svd_solver='{svd_solver}'")
    
    # Get variance explained by singular values
    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = torch.sum(explained_variance)
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
    X: Any,
    components: Any,
    mean: Optional[Any] = None,
    *,
    whiten: bool = False,
    explained_variance: Optional[Any] = None,
) -> Any:
    """
    Apply dimensionality reduction to X using PyTorch.
    
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
    X_tensor = convert_to_tensor(X)
    components_tensor = convert_to_tensor(components)
    
    # Ensure components are on the same device as X
    if components_tensor.device != X_tensor.device:
        components_tensor = components_tensor.to(X_tensor.device)
    
    # Center data
    if mean is not None:
        mean_tensor = convert_to_tensor(mean)
        if mean_tensor.device != X_tensor.device:
            mean_tensor = mean_tensor.to(X_tensor.device)
        X_centered = X_tensor - mean_tensor
    else:
        X_centered = X_tensor
    
    # Project data
    X_transformed = X_centered @ components_tensor.T
    
    # Whiten if requested
    if whiten:
        if explained_variance is None:
            raise ValueError("explained_variance must be provided when whiten=True")
        explained_variance_tensor = convert_to_tensor(explained_variance)
        if explained_variance_tensor.device != X_tensor.device:
            explained_variance_tensor = explained_variance_tensor.to(X_tensor.device)
        # Avoid division by zero
        eps = torch.finfo(X_transformed.dtype).eps
        scale = torch.sqrt(torch.clamp(explained_variance_tensor, min=eps))
        X_transformed = X_transformed / scale
    
    return X_transformed


def inverse_transform(
    X: Any,
    components: Any,
    mean: Optional[Any] = None,
    *,
    whiten: bool = False,
    explained_variance: Optional[Any] = None,
) -> Any:
    """
    Transform data back to its original space using PyTorch.
    
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
    X_tensor = convert_to_tensor(X)
    components_tensor = convert_to_tensor(components)
    
    # Ensure components are on the same device as X
    if components_tensor.device != X_tensor.device:
        components_tensor = components_tensor.to(X_tensor.device)
    
    # Unwhiten if needed
    if whiten:
        if explained_variance is None:
            raise ValueError("explained_variance must be provided when whiten=True")
        explained_variance_tensor = convert_to_tensor(explained_variance)
        if explained_variance_tensor.device != X_tensor.device:
            explained_variance_tensor = explained_variance_tensor.to(X_tensor.device)
        # Avoid division by zero
        eps = torch.finfo(X_tensor.dtype).eps
        scale = torch.sqrt(torch.clamp(explained_variance_tensor, min=eps))
        X_unwhitened = X_tensor * scale
    else:
        X_unwhitened = X_tensor
    
    # Project back to original space
    X_original = X_unwhitened @ components_tensor
    
    # Add mean if provided
    if mean is not None:
        mean_tensor = convert_to_tensor(mean)
        if mean_tensor.device != X_tensor.device:
            mean_tensor = mean_tensor.to(X_tensor.device)
        X_original = X_original + mean_tensor
    
    return X_original


def standardize(
    X: Any,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    axis: int = 0,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Standardize features by removing the mean and scaling to unit variance using PyTorch.
    
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
    X_tensor = convert_to_tensor(X)
    
    mean = None
    std = None
    
    # Center data
    if with_mean:
        mean = torch.mean(X_tensor, dim=axis, keepdim=True)
        X_centered = X_tensor - mean
    else:
        X_centered = X_tensor
    
    # Scale data
    if with_std:
        std = torch.std(X_tensor, dim=axis, keepdim=True, unbiased=True)
        # Avoid division by zero
        eps = torch.finfo(X_tensor.dtype).eps
        std_clipped = torch.clamp(std, min=eps)
        X_scaled = X_centered / std_clipped
    else:
        X_scaled = X_centered
    
    # Remove singleton dimensions if not keeping dims
    if mean is not None and mean.shape[axis] == 1:
        mean = torch.squeeze(mean, dim=axis)
    if std is not None and std.shape[axis] == 1:
        std = torch.squeeze(std, dim=axis)
    
    return X_scaled, mean, std


def normalize(
    X: Any,
    *,
    norm: str = "l2",
    axis: int = 1,
) -> Any:
    """
    Scale input vectors individually to unit norm using PyTorch.
    
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
    X_tensor = convert_to_tensor(X)
    
    if norm == "l1":
        norms = torch.sum(torch.abs(X_tensor), dim=axis, keepdim=True)
    elif norm == "l2":
        norms = torch.sqrt(torch.sum(X_tensor ** 2, dim=axis, keepdim=True))
    elif norm == "max":
        norms = torch.max(torch.abs(X_tensor), dim=axis, keepdim=True)[0]
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    
    # Avoid division by zero
    eps = torch.finfo(X_tensor.dtype).eps
    norms_clipped = torch.clamp(norms, min=eps)
    
    return X_tensor / norms_clipped


class TorchFeatureOps:
    """PyTorch implementation of feature operations."""
    
    def pca(
        self,
        X,
        n_components=None,
        *,
        whiten=False,
        center=True,
        svd_solver="auto",
    ):
        """Principal Component Analysis (PCA)."""
        return pca(
            X,
            n_components=n_components,
            whiten=whiten,
            center=center,
            svd_solver=svd_solver,
        )
    
    def transform(
        self,
        X,
        components,
        mean=None,
        *,
        whiten=False,
        explained_variance=None,
    ):
        """Apply dimensionality reduction to X."""
        return transform(
            X,
            components,
            mean=mean,
            whiten=whiten,
            explained_variance=explained_variance,
        )
    
    def inverse_transform(
        self,
        X,
        components,
        mean=None,
        *,
        whiten=False,
        explained_variance=None,
    ):
        """Transform data back to its original space."""
        return inverse_transform(
            X,
            components,
            mean=mean,
            whiten=whiten,
            explained_variance=explained_variance,
        )
    
    def standardize(
        self,
        X,
        *,
        with_mean=True,
        with_std=True,
        axis=0,
    ):
        """Standardize features by removing the mean and scaling to unit variance."""
        return standardize(
            X,
            with_mean=with_mean,
            with_std=with_std,
            axis=axis,
        )
    
    def normalize(
        self,
        X,
        *,
        norm="l2",
        axis=1,
    ):
        """Scale input vectors individually to unit norm."""
        return normalize(
            X,
            norm=norm,
            axis=axis,
        )