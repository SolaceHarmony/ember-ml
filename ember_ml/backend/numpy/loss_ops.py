"""
NumPy implementation of loss operations.

This module provides NumPy implementations of loss operations.
"""

import numpy as np
from typing import Any, Optional, Union, Sequence, Tuple

from ember_ml.backend.numpy.tensor.tensor import NumpyTensor

# Create a tensor instance for tensor operations
_tensor_ops = NumpyTensor()

class NumpyLossOps:
    """NumPy implementation of loss operations."""
    
    def mean_squared_error(self, y_true: Any, y_pred: Any, 
                          axis: Optional[Union[int, Sequence[int]]] = None, 
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        squared_error = np.square(y_true - y_pred)
        return np.mean(squared_error, axis=axis, keepdims=keepdims)
    
    def mean_absolute_error(self, y_true: Any, y_pred: Any, 
                           axis: Optional[Union[int, Sequence[int]]] = None, 
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        absolute_error = np.abs(y_true - y_pred)
        return np.mean(absolute_error, axis=axis, keepdims=keepdims)
    
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        if from_logits:
            # Apply sigmoid to convert logits to probabilities
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Clip to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary crossentropy formula: -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
        bce = -y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1.0 - y_pred)
        
        return np.mean(bce, axis=axis, keepdims=keepdims)
    
    def categorical_crossentropy(self, y_true: Any, y_pred: Any, from_logits: bool = False, 
                                axis: Optional[Union[int, Sequence[int]]] = -1, 
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        if from_logits:
            # Apply softmax to convert logits to probabilities
            y_pred_max = np.max(y_pred, axis=axis, keepdims=True)
            y_pred_exp = np.exp(y_pred - y_pred_max)
            y_pred = y_pred_exp / np.sum(y_pred_exp, axis=axis, keepdims=True)
        
        # Clip to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0)
        
        # Categorical crossentropy formula: -sum(y_true * log(y_pred))
        cce = -np.sum(y_true * np.log(y_pred), axis=axis, keepdims=True)
        
        if not keepdims and axis is not None:
            cce = np.squeeze(cce, axis=axis)
            
        return cce
    
    def sparse_categorical_crossentropy(self, y_true: Any, y_pred: Any, from_logits: bool = False, 
                                       axis: Optional[Union[int, Sequence[int]]] = -1, 
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        # Convert sparse labels to one-hot encoding
        if len(y_true.shape) == len(y_pred.shape) - 1:
            num_classes = y_pred.shape[-1]
            y_true_one_hot = np.zeros_like(y_pred)
            if len(y_pred.shape) == 2:
                y_true_one_hot[np.arange(len(y_true)), y_true.astype(np.int32)] = 1
            else:
                # Handle batched inputs with more dimensions
                indices = []
                for i in range(len(y_pred.shape) - 1):
                    if i < len(y_true.shape):
                        indices.append(np.arange(y_pred.shape[i]))
                    else:
                        indices.append(np.zeros(y_pred.shape[i], dtype=np.int32))
                indices.append(y_true.astype(np.int32))
                y_true_one_hot[tuple(np.meshgrid(*indices, indexing='ij'))] = 1
            
            # Use categorical crossentropy with one-hot encoded labels
            return self.categorical_crossentropy(y_true_one_hot, y_pred, from_logits, axis, keepdims)
        else:
            # If y_true is already one-hot encoded, use categorical crossentropy directly
            return self.categorical_crossentropy(y_true, y_pred, from_logits, axis, keepdims)
    
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        error = y_true - y_pred
        abs_error = np.abs(error)
        
        # Quadratic part (for errors <= delta)
        quadratic = 0.5 * np.square(error)
        # Linear part (for errors > delta)
        linear = delta * (abs_error - 0.5 * delta)
        
        # Combine quadratic and linear parts
        loss = np.where(abs_error <= delta, quadratic, linear)
        
        return np.mean(loss, axis=axis, keepdims=keepdims)
    
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
        y_true = _tensor_ops.convert_to_tensor(y_true)
        y_pred = _tensor_ops.convert_to_tensor(y_pred)
        
        error = y_pred - y_true
        
        # Compute log(cosh(error))
        # Use a numerically stable formula to avoid overflow
        log_cosh = error + np.log(1 + np.exp(-2 * error)) - np.log(2)
        
        return np.mean(log_cosh, axis=axis, keepdims=keepdims)