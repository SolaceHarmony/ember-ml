"""
Backend-agnostic implementation of tensor feature operations.

This module provides tensor feature operations using the ops abstraction layer,
making it compatible with all backends (NumPy, PyTorch, MLX).
"""

from typing import Any, Optional, Union, Sequence, List, Tuple, cast

from ember_ml import ops
from ember_ml.features.interfaces.tensor_features import TensorFeaturesInterface
from ember_ml.nn import tensor

class TensorFeatures(TensorFeaturesInterface):
    """Backend-agnostic implementation of tensor feature operations."""
    
    def one_hot(
        self,
        indices: Any,
        num_classes: int,
        *,
        axis: int = -1,
        dtype: Any = None
    ) -> Any:
        """
        Create a one-hot tensor.
        
        Args:
            indices: A tensor of indices.
            num_classes: The number of classes.
            axis: The axis to place the one-hot dimension.
            dtype: The data type of the output tensor.
            
        Returns:
            A tensor with one-hot encoding.
        """
        # Convert indices to tensor if needed
        indices = tensor.convert_to_tensor(indices)
        
        # Get the shape of the output tensor
        shape = list(tensor.shape(indices))
        
        # Handle negative axis
        if axis < 0:
            axis = len(shape) + axis + 1
        else:
            axis = axis
        
        # Insert the num_classes dimension at the specified axis
        shape.insert(axis, num_classes)
        
        # Create a tensor of zeros with the output shape
        if dtype is None:
            dtype = tensor.float32
        
        one_hot_tensor = tensor.zeros(shape, dtype=dtype)
        
        # Create a tensor of ones for updates
        updates = tensor.ones([1], dtype=dtype)
        
        # For each index, update the corresponding position in the one-hot tensor
        for i in range(tensor.shape(indices)[0]):
            # Get the index for this sample
            idx = indices[i]
            
            # Create the slice for this update
            slices = [slice(i, i+1)] * len(shape)
            slices[axis] = slice(idx, idx+1)
            
            # Update the tensor
            one_hot_tensor = tensor.slice_update(one_hot_tensor, slices, updates)
        
        return one_hot_tensor
    
    def scatter(
        self,
        tensor: Any,
        indices: Any,
        updates: Any,
        *,
        axis: int = 0
    ) -> Any:
        """
        Scatter updates into a tensor according to indices.
        
        Args:
            tensor: The tensor to update.
            indices: The indices where updates will be scattered.
            updates: The values to scatter.
            axis: The axis along which to scatter.
            
        Returns:
            The updated tensor.
        """
        # Convert inputs to tensors
        tensor = tensor.convert_to_tensor(tensor)
        indices = tensor.convert_to_tensor(indices)
        updates = tensor.convert_to_tensor(updates)
        
        # Create a copy of the tensor to update
        result = tensor.copy(tensor)
        
        # For each index, update the corresponding position in the tensor
        for i in range(tensor.shape(indices)[0]):
            # Get the index for this update
            idx = indices[i]
            
            # Create the slice for this update
            slices = [slice(None)] * len(tensor.shape(tensor))
            slices[axis] = slice(idx, idx+1)
            
            # Get the update value
            update = updates[i:i+1]
            
            # Update the tensor
            result = tensor.slice_update(result, slices, update)
        
        return result