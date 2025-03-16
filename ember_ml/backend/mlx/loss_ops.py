"""
MLX implementation of loss operations.

This module provides MLX implementations of loss operations.
"""

import mlx.core as mx
from typing import Any, Optional, Union, Sequence, cast

import mlx.core as mx
from typing import Any, Optional, Union, Sequence, cast

from ember_ml.backend.mlx.tensor.tensor import MLXTensor

# Create a tensor instance for tensor operations
_tensor_ops = MLXTensor()

# Import scatter function from the appropriate module
# This is a placeholder - you'll need to update this with the correct import
def scatter(src, indices, dim_size, aggr="add", axis=-1):
    """Placeholder for scatter function."""
    # Implementation would go here
    pass

class MLXLossOps:
    """MLX implementation of loss operations."""
    
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
        
        squared_error = mx.square(y_true - y_pred)
        return mx.mean(squared_error, axis=axis, keepdims=keepdims)
    
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
        
        absolute_error = mx.abs(y_true - y_pred)
        return mx.mean(absolute_error, axis=axis, keepdims=keepdims)
    
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
            y_pred = mx.sigmoid(y_pred)
        
        # Clip to avoid log(0)
        epsilon = 1e-7
        y_pred = mx.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary crossentropy formula: -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
        bce = -y_true * mx.log(y_pred) - (1.0 - y_true) * mx.log(1.0 - y_pred)
        
        return mx.mean(bce, axis=axis, keepdims=keepdims)
    
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
            if axis is not None:
                axis_val = cast(int, axis)
                y_pred = mx.softmax(y_pred, axis=axis_val)
            else:
                y_pred = mx.softmax(y_pred, axis=-1)
        
        # Clip to avoid log(0)
        epsilon = 1e-7
        y_pred = mx.clip(y_pred, epsilon, 1.0)
        
        # Categorical crossentropy formula: -sum(y_true * log(y_pred))
        if axis is not None:
            axis_val = cast(int, axis)
            cce = -mx.sum(y_true * mx.log(y_pred), axis=axis_val, keepdims=keepdims)
        else:
            cce = -mx.sum(y_true * mx.log(y_pred), axis=-1, keepdims=keepdims)
            
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
            num_classes = y_pred.shape[-1] if axis is None else y_pred.shape[cast(int, axis)]
            
            # Convert to one-hot encoding
            y_true_one_hot = mx.zeros_like(y_pred)
            
            # Create indices for one-hot encoding
            if axis is None or axis == -1:
                # Handle the common case where class dimension is the last dimension
                indices = []
                for i in range(len(y_pred.shape) - 1):
                    indices.append(mx.arange(y_pred.shape[i]))
                indices.append(y_true)
                
                # Set the one-hot values
                # Create empty tensor for output
                y_true_one_hot = scatter(mx.ones_like(y_true), indices, y_pred.shape[-1], aggr="add", axis=-1)
            else:
                # Handle the case where class dimension is not the last dimension
                axis_val = cast(int, axis)
                indices = []
                for i in range(len(y_pred.shape)):
                    if i == axis_val:
                        indices.append(y_true)
                    else:
                        indices.append(mx.arange(y_pred.shape[i]))
                
                # Set the one-hot values
                # Create empty tensor for output with the correct shape
                y_true_one_hot = scatter(mx.ones_like(y_true), indices, y_pred.shape[axis_val], aggr="add", axis=axis_val)
            
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
        abs_error = mx.abs(error)
        
        # Quadratic part (for errors <= delta)
        quadratic = 0.5 * mx.square(error)
        # Linear part (for errors > delta)
        linear = delta * (abs_error - 0.5 * delta)
        
        # Combine quadratic and linear parts
        loss = mx.where(abs_error <= delta, quadratic, linear)
        
        return mx.mean(loss, axis=axis, keepdims=keepdims)
    
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
        log_cosh = error + mx.log(1 + mx.exp(-2 * error)) - mx.log(mx.array(2.0))
        
        return mx.mean(log_cosh, axis=axis, keepdims=keepdims)