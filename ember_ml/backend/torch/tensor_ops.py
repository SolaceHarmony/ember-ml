"""
PyTorch tensor operations for EmberHarmony.

This module provides PyTorch implementations of tensor operations.
"""

import torch
from typing import Optional, Union, Tuple, List, Any, Sequence, Dict

# Type aliases
ArrayLike = Union[torch.Tensor, list, tuple, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[torch.dtype, str, None]

# Import from config
from ember_ml.backend.torch.config import DEFAULT_DEVICE


def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to a PyTorch tensor.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        PyTorch tensor representation of the input
    
    Raises:
        TypeError: If x is a tensor from another backend
    """
    # Check if x is a tensor from another backend
    if hasattr(x, '__class__') and 'Tensor' in x.__class__.__name__ and not isinstance(x, torch.Tensor):
        raise TypeError(f"Cannot convert tensor of type {type(x)} to PyTorch tensor. "
                        f"Use the appropriate backend for this tensor type.")
    
    # Create tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.tensor(x, dtype=dtype)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the specified shape
    """
    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the specified shape
    """
    return torch.ones(shape, dtype=dtype, device=device)


def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of zeros with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    return torch.zeros_like(x_tensor, dtype=dtype, device=device)


def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Tensor of ones with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    return torch.ones_like(x_tensor, dtype=dtype, device=device)


def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device to place the tensor on
        
    Returns:
        Identity matrix of shape (n, m)
    """
    # Handle the case where m is None
    if m is None:
        return torch.eye(n, dtype=dtype, device=device)
    else:
        return torch.eye(n, m=m, dtype=dtype, device=device)


def reshape(x: ArrayLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    return convert_to_tensor(x).reshape(shape)


def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> torch.Tensor:
    """
    Permute the dimensions of a tensor.
    
    Args:
        x: Input tensor
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        return x_tensor.transpose(-2, -1)
    
    # Convert to PyTorch's permute format
    return x_tensor.permute(*axes)


def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> torch.Tensor:
    """
    Insert new axes into a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where new axes should be inserted
        
    Returns:
        Tensor with expanded dimensions
    """
    x_tensor = convert_to_tensor(x)
    
    if isinstance(axis, (list, tuple)):
        # Handle multiple axes
        result = x_tensor
        # Sort axes in ascending order to avoid dimension shifting
        for ax in sorted(axis):
            result = torch.unsqueeze(result, dim=ax)
        return result
    
    # Handle single axis
    return torch.unsqueeze(x_tensor, dim=axis)


def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Concatenate tensors along a specified axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    return torch.cat([convert_to_tensor(arr) for arr in arrays], dim=axis)


def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> torch.Tensor:
    """
    Stack tensors along a new axis.
    
    Args:
        arrays: Sequence of tensors
        axis: Axis along which to stack
        
    Returns:
        Stacked tensor
    """
    return torch.stack([convert_to_tensor(arr) for arr in arrays], dim=axis)


def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[torch.Tensor]:
    """
    Split a tensor into sub-tensors.
    
    Args:
        x: Input tensor
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-tensors
    """
    x_tensor = convert_to_tensor(x)
    if isinstance(num_or_size_splits, int):
        # Calculate split size using torch operations
        # Use torch.div with rounding_mode='trunc' instead of // operator
        split_size = torch.div(torch.tensor(x_tensor.shape[axis]),
                              torch.tensor(num_or_size_splits),
                              rounding_mode='trunc').item()
        return torch.split(x_tensor, split_size, dim=axis)
    return torch.split(x_tensor, num_or_size_splits, dim=axis)


def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> torch.Tensor:
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where dimensions should be removed
        
    Returns:
        Tensor with squeezed dimensions
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return torch.squeeze(x_tensor)
    if isinstance(axis, (list, tuple)):
        result = x_tensor
        for ax in sorted(axis, reverse=True):
            result = torch.squeeze(result, ax)
        return result
    return torch.squeeze(x_tensor, axis)


def tile(x: ArrayLike, reps: Sequence[int]) -> torch.Tensor:
    """
    Construct a tensor by tiling a given tensor.
    
    Args:
        x: Input tensor
        reps: Number of repetitions along each dimension
        
    Returns:
        Tiled tensor
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.repeat(*reps)


def gather(x: ArrayLike, indices: Any, axis: int = 0) -> torch.Tensor:
    """
    Gather slices from a tensor along an axis.
    
    Args:
        x: Input tensor
        indices: Indices of slices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered tensor
    """
    x_tensor = convert_to_tensor(x)
    indices_tensor = convert_to_tensor(indices)
    return torch.gather(x_tensor, axis, indices_tensor)


def tensor_scatter_nd_update(array: ArrayLike, indices: ArrayLike, updates: ArrayLike) -> torch.Tensor:
    """
    Updates elements of a tensor at specified indices with given values.
    
    Args:
        array: The tensor to be updated.
        indices: An array of indices, where each row represents
                 the index of an element to be updated.
        updates: An array of update values, with the same
                 length as the number of rows in indices.
        
    Returns:
        A new tensor with the updates applied.
    """
    array_tensor = convert_to_tensor(array)
    indices_tensor = convert_to_tensor(indices).long()  # Ensure indices are long type
    updates_tensor = convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    output_tensor = array_tensor.clone()
    
    # For each dimension, create a tensor of indices for scatter_
    if indices_tensor.dim() == 2:
        # Get the number of dimensions in the indices
        ndim = indices_tensor.shape[1]
        
        # For each dimension, create a scatter operation
        for dim in range(ndim):
            # Extract indices for this dimension
            dim_indices = indices_tensor[:, dim].view(-1, 1)
            
            # Create a tensor of the same shape as updates for scatter_
            src = updates_tensor.clone()
            
            # Use scatter_ to update the tensor
            # We need to create a new tensor for each dimension because scatter_ modifies in-place
            temp_tensor = output_tensor.clone()
            temp_tensor.scatter_(dim, dim_indices, src)
            output_tensor = temp_tensor
    else:
        # For 1D indices, use a simpler approach
        dim = 0  # Default dimension
        output_tensor.scatter_(dim, indices_tensor, updates_tensor)
    
    return output_tensor


def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Shape of the tensor
    """
    return tuple(convert_to_tensor(x).shape)


def dtype(x: ArrayLike) -> torch.dtype:
    """
    Get the data type of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Data type of the tensor
    """
    return convert_to_tensor(x).dtype


def cast(x: ArrayLike, dtype: DType) -> torch.Tensor:
    """
    Cast a tensor to a different data type.
    
    Args:
        x: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with the target data type
    """
    return convert_to_tensor(x).to(dtype)


def copy(x: ArrayLike) -> torch.Tensor:
    """
    Create a copy of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Copy of the tensor
    """
    return convert_to_tensor(x).clone()


def to_numpy(x: ArrayLike) -> Any:
    """
    Convert a tensor to a NumPy array.
    
    This function is an exception to the general backend purity rules as it's specifically
    designed to convert tensors to NumPy arrays for use with plotting libraries and other
    external tools that require NumPy arrays.
    
    Args:
        x: Input tensor
        
    Returns:
        NumPy array
    """
    x_tensor = convert_to_tensor(x)
    
    # Move to CPU if on another device
    if x_tensor.device.type != 'cpu':
        x_tensor = x_tensor.cpu()
    
    # EMBERLINT: IGNORE - Direct NumPy usage is allowed in this function as an exception
    # Convert to NumPy using PyTorch's native method
    if x_tensor.requires_grad:
        # First detach the tensor to remove the gradient tracking
        detached_tensor = x_tensor.detach()
        # EMBERLINT: IGNORE - Direct tensor.numpy() usage is allowed here
        import numpy as np
        return np.array(detached_tensor.cpu().detach())
    else:
        # EMBERLINT: IGNORE - Direct tensor.numpy() usage is allowed here
        import numpy as np
        return np.array(x_tensor.cpu().detach())


def var(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> torch.Tensor:
    """
    Compute the variance of a tensor along specified axes.
    
    Args:
        x: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    x_tensor = convert_to_tensor(x)
    
    if axis is None:
        return torch.var(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        # PyTorch doesn't support multiple axes directly, so we need to do it sequentially
        result = x_tensor
        # Sort axes in descending order to avoid dimension shifting
        for ax in sorted(axis, reverse=True):
            result = torch.var(result, dim=ax, keepdim=keepdims)
        return result
    
    return torch.var(x_tensor, dim=axis, keepdim=keepdims)


def full(shape: Shape, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value.
    
    Args:
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        Tensor filled with the specified value
    """
    if device is None:
        device = DEFAULT_DEVICE
    return torch.full(shape, fill_value, dtype=dtype, device=device)


def full_like(x: ArrayLike, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value with the same shape as the input.
    
    Args:
        x: Input tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        Tensor filled with the specified value with the same shape as x
    """
    if device is None:
        device = DEFAULT_DEVICE
    x_tensor = convert_to_tensor(x)
    return torch.full_like(x_tensor, fill_value, dtype=dtype, device=device)


def linspace(start: float, stop: float, num: int, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of the interval
        stop: End of the interval
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device
        
    Returns:
        Tensor with evenly spaced values
    """
    if device is None:
        device = DEFAULT_DEVICE
    return torch.linspace(start, stop, num, dtype=dtype, device=device)


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        Tensor with evenly spaced values
    """
    if device is None:
        device = DEFAULT_DEVICE
    if stop is None:
        # If only one argument is provided, it's the stop value
        return torch.arange(start=0, end=start, step=step, dtype=dtype, device=device)
    return torch.arange(start=start, end=stop, step=step, dtype=dtype, device=device)


class TorchTensorOps:
    """PyTorch implementation of tensor operations."""
    
    def zeros(self, shape, dtype=None, device=None):
        """Create a tensor of zeros."""
        return zeros(shape, dtype=dtype, device=device)
    
    def ones(self, shape, dtype=None, device=None):
        """Create a tensor of ones."""
        return ones(shape, dtype=dtype, device=device)
    
    def zeros_like(self, x, dtype=None, device=None):
        """Create a tensor of zeros with the same shape as the input."""
        return zeros_like(x, dtype=dtype, device=device)
    
    def ones_like(self, x, dtype=None, device=None):
        """Create a tensor of ones with the same shape as the input."""
        return ones_like(x, dtype=dtype, device=device)
    
    def eye(self, n, m=None, dtype=None, device=None):
        """Create an identity matrix."""
        return eye(n, m=m, dtype=dtype, device=device)
    
    def arange(self, start, stop=None, step=1, dtype=None, device=None):
        """Create a tensor with evenly spaced values within a given interval."""
        return arange(start, stop=stop, step=step, dtype=dtype, device=device)
    
    def linspace(self, start, stop, num, dtype=None, device=None):
        """Create a tensor with evenly spaced values within a given interval."""
        return linspace(start, stop, num, dtype=dtype, device=device)
    
    def full(self, shape, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value."""
        return full(shape, fill_value, dtype=dtype, device=device)
    
    def full_like(self, x, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value with the same shape as the input."""
        return full_like(x, fill_value, dtype=dtype, device=device)
    
    def reshape(self, x, shape):
        """Reshape a tensor to a new shape."""
        return reshape(x, shape)
    
    def transpose(self, x, axes=None):
        """Permute the dimensions of a tensor."""
        return transpose(x, axes=axes)
    
    def concatenate(self, tensors, axis=0):
        """Concatenate tensors along a specified axis."""
        return concatenate(tensors, axis=axis)
    
    def stack(self, tensors, axis=0):
        """Stack tensors along a new axis."""
        return stack(tensors, axis=axis)
    
    def split(self, x, num_or_size_splits, axis=0):
        """Split a tensor into sub-tensors."""
        return split(x, num_or_size_splits, axis=axis)
    
    def expand_dims(self, x, axis):
        """Insert new axes into a tensor's shape."""
        return expand_dims(x, axis)
    
    def squeeze(self, x, axis=None):
        """Remove single-dimensional entries from a tensor's shape."""
        return squeeze(x, axis=axis)
    
    def tile(self, x, reps):
        """Construct a tensor by tiling a given tensor."""
        return tile(x, reps)
    
    def gather(self, x, indices, axis=0):
        """Gather slices from a tensor along an axis."""
        return gather(x, indices, axis=axis)
    
    def tensor_scatter_nd_update(self, array, indices, updates):
        """Update elements of a tensor at specified indices with given values."""
        return tensor_scatter_nd_update(array, indices, updates)
    
    def convert_to_tensor(self, x, dtype=None, device=None):
        """Convert input to a tensor."""
        return convert_to_tensor(x, dtype=dtype, device=device)
    
    def shape(self, x):
        """Get the shape of a tensor."""
        return shape(x)
    
    def dtype(self, x):
        """Get the data type of a tensor."""
        return dtype(x)
    
    def cast(self, x, dtype):
        """Cast a tensor to a different data type."""
        return cast(x, dtype)
    
    def copy(self, x):
        """Create a copy of a tensor."""
        return copy(x)
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of a tensor along specified axes."""
        return var(x, axis=axis, keepdims=keepdims)
    
    def to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        return to_numpy(x)