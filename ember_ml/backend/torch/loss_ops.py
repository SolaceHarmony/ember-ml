"""
PyTorch implementation of loss operations.

This module provides PyTorch implementations of loss operations.
"""

import torch
import torch.nn.functional as F
from typing import Any, Optional, Union, Sequence, Tuple, List, cast

from ember_ml.backend.torch.tensor.tensor import TorchTensor

# Create a tensor instance for tensor operations
_tensor_ops = TorchTensor()

class TorchLossOps:
    """PyTorch implementation of loss operations."""
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        squared_error = torch.square(y_true - y_pred)
        
        # Handle axis parameter for PyTorch
        if axis is None:
            return torch.mean(squared_error)
        elif isinstance(axis, (list, tuple)):
            # For multiple axes, reduce one by one
            result = squared_error
            for ax in sorted(cast(Sequence[int], axis), reverse=True):
                result = torch.mean(result, dim=ax, keepdim=keepdims)
            return result
        else:
            return torch.mean(squared_error, dim=cast(int, axis), keepdim=keepdims)
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        absolute_error = torch.abs(y_true - y_pred)
        
        # Handle axis parameter for PyTorch
        if axis is None:
            return torch.mean(absolute_error)
        elif isinstance(axis, (list, tuple)):
            # For multiple axes, reduce one by one
            result = absolute_error
            for ax in sorted(cast(Sequence[int], axis), reverse=True):
                result = torch.mean(result, dim=ax, keepdim=keepdims)
            return result
        else:
            return torch.mean(absolute_error, dim=cast(int, axis), keepdim=keepdims)
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        # Use PyTorch's binary cross entropy function
        if from_logits:
            bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        else:
            bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        
        # Handle axis parameter for PyTorch
        if axis is None:
            return torch.mean(bce)
        elif isinstance(axis, (list, tuple)):
            # For multiple axes, reduce one by one
            result = bce
            for ax in sorted(cast(Sequence[int], axis), reverse=True):
                result = torch.mean(result, dim=ax, keepdim=keepdims)
            return result
        else:
            return torch.mean(bce, dim=cast(int, axis), keepdim=keepdims)
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        # Use PyTorch's cross entropy function
        if from_logits:
            # For cross_entropy, target should be class indices, not one-hot
            if y_true.dim() == y_pred.dim():
                # If y_true is one-hot encoded, convert to class indices
                if axis is not None:
                    y_true = torch.argmax(y_true, dim=cast(int, axis))
                else:
                    y_true = torch.argmax(y_true, dim=-1)
            
            # PyTorch expects class dim to be 1 for cross_entropy
            if axis != 1 and y_pred.dim() > 1:
                # Move class dim to position 1
                if axis is not None:
                    axis_val = cast(int, axis)
                    dims = list(range(y_pred.dim()))
                    dims.remove(axis_val)
                    dims.insert(1, axis_val)
                    y_pred = y_pred.permute(dims)
                
            cce = F.cross_entropy(y_pred, y_true.long(), reduction='none')
        else:
            # If not from logits, we need to compute cross entropy manually
            # Clip to avoid log(0)
            epsilon = 1e-7
            y_pred = torch.clamp(y_pred, epsilon, 1.0)
            
            # Categorical crossentropy formula: -sum(y_true * log(y_pred))
            if axis is not None:
                cce = -torch.sum(y_true * torch.log(y_pred), dim=cast(int, axis), keepdim=keepdims)
            else:
                cce = -torch.sum(y_true * torch.log(y_pred), dim=-1, keepdim=keepdims)
            
        if axis is not None and axis != -1 and not keepdims:
            # If we're not keeping dims and axis is not -1, we need to squeeze
            cce = torch.squeeze(cce, dim=cast(int, axis))
            
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        # PyTorch's cross_entropy expects class indices, which is what sparse categorical uses
        if from_logits:
            # PyTorch expects class dim to be 1 for cross_entropy
            if axis != 1 and y_pred.dim() > 1 and axis is not None:
                # Move class dim to position 1
                axis_val = cast(int, axis)
                dims = list(range(y_pred.dim()))
                dims.remove(axis_val)
                dims.insert(1, axis_val)
                y_pred = y_pred.permute(dims)
                
            return F.cross_entropy(y_pred, y_true.long(), reduction='none')
        else:
            # If not from logits, we need to compute log softmax first
            if axis is not None:
                log_softmax = F.log_softmax(y_pred, dim=cast(int, axis))
            else:
                log_softmax = F.log_softmax(y_pred, dim=-1)
            
            # Gather the values corresponding to the target classes
            if y_true.dim() == y_pred.dim() - 1:
                # Create indices for gather
                if axis is not None:
                    axis_val = cast(int, axis)
                    indices = y_true.unsqueeze(axis_val)
                    gathered = torch.gather(log_softmax, axis_val, indices)
                else:
                    indices = y_true.unsqueeze(-1)
                    gathered = torch.gather(log_softmax, -1, indices)
                return -gathered
            else:
                # If y_true is already one-hot encoded, use categorical crossentropy
                return self.categorical_crossentropy(y_true, y_pred, from_logits=False, 
                                                   axis=axis, keepdims=keepdims)
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        # Use PyTorch's smooth_l1_loss which is equivalent to Huber loss
        # with delta=1.0, and scale it appropriately for other delta values
        if delta == 1.0:
            huber = F.smooth_l1_loss(y_pred, y_true, reduction='none')
        else:
            error = y_true - y_pred
            abs_error = torch.abs(error)
            
            # Quadratic part (for errors <= delta)
            quadratic = 0.5 * torch.square(error)
            # Linear part (for errors > delta)
            linear = delta * (abs_error - 0.5 * delta)
            
            # Combine quadratic and linear parts
            huber = torch.where(abs_error <= delta, quadratic, linear)
        
        # Handle axis parameter for PyTorch
        if axis is None:
            return torch.mean(huber)
        elif isinstance(axis, (list, tuple)):
            # For multiple axes, reduce one by one
            result = huber
            for ax in sorted(cast(Sequence[int], axis), reverse=True):
                result = torch.mean(result, dim=ax, keepdim=keepdims)
            return result
        else:
            return torch.mean(huber, dim=cast(int, axis), keepdim=keepdims)
    
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
        y_true = _tensor_tensor.convert_to_tensor(y_true)
        y_pred = _tensor_tensor.convert_to_tensor(y_pred)
        
        error = y_pred - y_true
        
        # Compute log(cosh(error))
        # Use a numerically stable formula to avoid overflow
        log_cosh = error + torch.log(1 + torch.exp(-2 * error)) - torch.log(torch.tensor(2.0))
        
        # Handle axis parameter for PyTorch
        if axis is None:
            return torch.mean(log_cosh)
        elif isinstance(axis, (list, tuple)):
            # For multiple axes, reduce one by one
            result = log_cosh
            for ax in sorted(cast(Sequence[int], axis), reverse=True):
                result = torch.mean(result, dim=ax, keepdim=keepdims)
            return result
        else:
            return torch.mean(log_cosh, dim=cast(int, axis), keepdim=keepdims)