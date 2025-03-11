"""
Loss operations interface.

This module defines the abstract interface for loss operations used in machine learning models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Sequence

class LossOps(ABC):
    """Abstract interface for loss operations."""
    
    @abstractmethod
    def mean_squared_error(self, y_true: Any, y_pred: Any, axis: Optional[Union[int, Sequence[int]]] = None, 
                          keepdims: bool = False) -> Any:
        """
        Compute the mean squared error between y_true and y_pred.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Mean squared error
        """
        pass
    
    @abstractmethod
    def mean_absolute_error(self, y_true: Any, y_pred: Any, axis: Optional[Union[int, Sequence[int]]] = None, 
                           keepdims: bool = False) -> Any:
        """
        Compute the mean absolute error between y_true and y_pred.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Mean absolute error
        """
        pass
    
    @abstractmethod
    def binary_crossentropy(self, y_true: Any, y_pred: Any, from_logits: bool = False, 
                           axis: Optional[Union[int, Sequence[int]]] = None, 
                           keepdims: bool = False) -> Any:
        """
        Compute the binary crossentropy loss between y_true and y_pred.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            from_logits: Whether y_pred is expected to be a logits tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Binary crossentropy loss
        """
        pass
    
    @abstractmethod
    def categorical_crossentropy(self, y_true: Any, y_pred: Any, from_logits: bool = False, 
                                axis: Optional[Union[int, Sequence[int]]] = None, 
                                keepdims: bool = False) -> Any:
        """
        Compute the categorical crossentropy loss between y_true and y_pred.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            from_logits: Whether y_pred is expected to be a logits tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Categorical crossentropy loss
        """
        pass
    
    @abstractmethod
    def sparse_categorical_crossentropy(self, y_true: Any, y_pred: Any, from_logits: bool = False, 
                                       axis: Optional[Union[int, Sequence[int]]] = None, 
                                       keepdims: bool = False) -> Any:
        """
        Compute the sparse categorical crossentropy loss between y_true and y_pred.
        
        Args:
            y_true: Ground truth values (integer indices)
            y_pred: Predicted values
            from_logits: Whether y_pred is expected to be a logits tensor
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Sparse categorical crossentropy loss
        """
        pass
    
    @abstractmethod
    def huber_loss(self, y_true: Any, y_pred: Any, delta: float = 1.0, 
                  axis: Optional[Union[int, Sequence[int]]] = None, 
                  keepdims: bool = False) -> Any:
        """
        Compute the Huber loss between y_true and y_pred.
        
        The Huber loss is a loss function that is less sensitive to outliers than MSE.
        It's quadratic for small errors and linear for large errors.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            delta: Threshold at which to change from quadratic to linear
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Huber loss
        """
        pass
    
    @abstractmethod
    def log_cosh_loss(self, y_true: Any, y_pred: Any, 
                     axis: Optional[Union[int, Sequence[int]]] = None, 
                     keepdims: bool = False) -> Any:
        """
        Compute the logarithm of the hyperbolic cosine of the prediction error.
        
        log(cosh(x)) is approximately equal to (x^2 / 2) for small x and
        to abs(x) - log(2) for large x. This means that log_cosh works
        mostly like the mean squared error, but will not be so strongly affected
        by the occasional wildly incorrect prediction.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            axis: Axis or axes along which to compute the mean
            keepdims: Whether to keep the reduced dimensions
            
        Returns:
            Log-cosh loss
        """
        pass