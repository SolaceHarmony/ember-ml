"""
Feature extraction and transformation interfaces.

This module defines the abstract interfaces for feature extraction and transformation operations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union

from ember_ml.features.interfaces.tensor_features import TensorFeaturesInterface


class PCAInterface(ABC):
    """Abstract interface for Principal Component Analysis (PCA)."""
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        n_components: Optional[Union[int, float, str]] = None,
        *,
        whiten: bool = False,
        center: bool = True,
        svd_solver: str = "auto",
    ) -> Any:
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
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Any,
    ) -> Any:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            X_new: Transformed values of shape (n_samples, n_components)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def inverse_transform(
        self,
        X: Any,
    ) -> Any:
        """
        Transform data back to its original space.
        
        Args:
            X: Input data of shape (n_samples, n_components)
            
        Returns:
            X_original: Original data of shape (n_samples, n_features)
        """
        pass


class StandardizeInterface(ABC):
    """Abstract interface for standardization."""
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = 0,
    ) -> Any:
        """
        Compute the mean and std to be used for standardization.
        
        Args:
            X: Input data
            with_mean: Whether to center the data
            with_std: Whether to scale the data
            axis: Axis along which to standardize
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Any,
    ) -> Any:
        """
        Standardize data.
        
        Args:
            X: Input data
            
        Returns:
            X_scaled: Standardized data
        """
        pass
    
    @abstractmethod
    def fit_transform(
        self,
        X: Any,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        axis: int = 0,
    ) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input data
            with_mean: Whether to center the data
            with_std: Whether to scale the data
            axis: Axis along which to standardize
            
        Returns:
            X_scaled: Standardized data
        """
        pass
    
    @abstractmethod
    def inverse_transform(
        self,
        X: Any,
    ) -> Any:
        """
        Scale back the data to the original representation.
        
        Args:
            X: Input data
            
        Returns:
            X_original: Original data
        """
        pass


class NormalizeInterface(ABC):
    """Abstract interface for normalization."""
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        *,
        norm: str = "l2",
        axis: int = 1,
    ) -> Any:
        """
        Compute the norm to be used for normalization.
        
        Args:
            X: Input data
            norm: The norm to use
            axis: Axis along which to normalize
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Any,
    ) -> Any:
        """
        Normalize data.
        
        Args:
            X: Input data
            
        Returns:
            X_normalized: Normalized data
        """
        pass
    
    @abstractmethod
    def fit_transform(
        self,
        X: Any,
        *,
        norm: str = "l2",
        axis: int = 1,
    ) -> Any:
        """
        Fit to data, then transform it.
        
        Args:
            X: Input data
            norm: The norm to use
            axis: Axis along which to normalize
            
        Returns:
            X_normalized: Normalized data
        """
        pass


__all__ = [
    'PCAInterface',
    'StandardizeInterface',
    'NormalizeInterface',
    'TensorFeaturesInterface',
]