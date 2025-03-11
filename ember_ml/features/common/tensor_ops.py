"""
Common tensor operations implementations.

This module provides backend-agnostic implementations of tensor operations
using the ops abstraction layer.
"""

from typing import Any, Optional, Union, Sequence, List, Tuple, cast

import numpy as np

from ember_ml import ops
from ember_ml.features.interfaces.tensor_ops import TensorOpsInterface


class TensorOps(TensorOpsInterface):
    """Backend-agnostic implementation of tensor operations."""
    
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
        indices = ops.convert_to_tensor(indices)
        
        # Get the shape of the output tensor
        shape = list(ops.shape(indices))
        
        # Handle negative axis
        if axis < 0:
            axis = len(shape) + axis + 1
        else:
            axis = axis
        
        # Insert the num_classes dimension at the specified axis
        shape.insert(axis, num_classes)
        
        # Create a tensor of zeros with the output shape
        if dtype is None:
            dtype = ops.float32
        
        one_hot_tensor = ops.zeros(shape, dtype=dtype)
        
        # Create indices for scatter
        ndims = len(shape)
        scatter_indices = []
        
        for i in range(ndims):
            if i == axis:
                # This is the one-hot dimension
                scatter_indices.append(ops.reshape(indices, [-1]))
            else:
                # Create indices for other dimensions
                idx = i if i < axis else i - 1
                dim_size = shape[i]
                
                # Create a range for this dimension
                dim_indices = ops.arange(dim_size)
                
                # Reshape to broadcast with other dimensions
                reshape_shape = [1] * (ndims - 1)
                reshape_shape[idx] = dim_size
                dim_indices = ops.reshape(dim_indices, reshape_shape)
                
                # Tile to match the shape of indices
                tile_shape = list(ops.shape(indices))
                tile_shape[idx] = 1
                dim_indices = ops.tile(dim_indices, tile_shape)
                
                # Flatten for scatter
                scatter_indices.append(ops.reshape(dim_indices, [-1]))
        
        # Create updates (all ones)
        updates = ops.ones([ops.shape(indices)[0]], dtype=dtype)
        
        # Use scatter to set the one-hot values
        return self.scatter(one_hot_tensor, scatter_indices, updates, axis=axis)
    
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
        tensor = ops.convert_to_tensor(tensor)
        indices = ops.convert_to_tensor(indices)
        updates = ops.convert_to_tensor(updates)
        
        # Create a copy of the tensor to update
        result = ops.copy(tensor)
        
        # Handle different backends
        backend = ops.get_ops()
        
        if backend == 'numpy':
            # For NumPy backend, use advanced indexing
            if isinstance(indices, list):
                # Multiple indices for different dimensions
                idx = tuple(indices)
                result[idx] = updates
            else:
                # Single index for the specified axis
                # Create a tuple of slices for indexing
                idx = [slice(None)] * len(ops.shape(tensor))
                idx[axis] = indices
                result[tuple(idx)] = updates
        
        elif backend == 'torch':
            # For PyTorch backend, use index_put_
            import torch
            
            if isinstance(indices, list):
                # Multiple indices for different dimensions
                idx = tuple(indices)
                result = torch.index_put(result, idx, updates)
            else:
                # Single index for the specified axis
                # Create a tuple of slices for indexing
                idx = [slice(None)] * len(ops.shape(tensor))
                idx[axis] = indices
                result = torch.index_put(result, tuple(idx), updates)
        
        elif backend == 'mlx':
            # For MLX backend, implement scatter manually
            # MLX doesn't have a direct scatter operation
            
            # Convert to numpy, perform scatter, and convert back
            tensor_np = ops.to_numpy(tensor)
            indices_np = ops.to_numpy(indices)
            updates_np = ops.to_numpy(updates)
            
            if isinstance(indices, list):
                # Multiple indices for different dimensions
                idx = tuple(indices_np)
                tensor_np[idx] = updates_np
            else:
                # Single index for the specified axis
                # Create a tuple of slices for indexing
                idx = [slice(None)] * len(tensor_np.shape)
                idx[axis] = indices_np
                tensor_np[tuple(idx)] = updates_np
            
            # Convert back to MLX tensor
            result = ops.convert_to_tensor(tensor_np)
        
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return result