"""
Feature operations interface.

This module defines the abstract interface for feature extraction and transformation operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, Any, Tuple, Dict


class FeatureOps(ABC):
    """Abstract interface for feature operations."""
    
    @abstractmethod
    def pca(
        self,
        X: Any,
        n_components: Optional[int] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> Dict[str, Any]:
        """
        Principal Component Analysis (PCA).
        
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
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Any,
        components: Any,
        mean: Optional[Any] = None,
        *,
        whiten: bool = False,
        explained_variance: Optional[Any] = None,
    ) -> Any:
        """
        Apply dimensionality reduction to X.
        
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
        pass
    
    @abstractmethod
    def inverse_transform(
        self,
        X: Any,
        components: Any,
        mean: Optional[Any] = None,
        *,
        whiten: bool = False,
        explained_variance: Optional[Any] = None,
    ) -> Any:
        """
        Transform data back to its original space.
        
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
        pass
    
    @abstractmethod
    def standardize(
        self,
        X: Any,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = 0,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """
        Standardize features by removing the mean and scaling to unit variance.
        
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
        pass
    
    @abstractmethod
    def normalize(
        self,
        X: Any,
        *,
        norm: str = "l2",
        axis: int = 1,
    ) -> Any:
        """
        Scale input vectors individually to unit norm.
        
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
        pass