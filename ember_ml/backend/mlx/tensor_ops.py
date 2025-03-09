"""
MLX implementation of tensor operations.

This module provides MLX implementations of tensor operations.
"""

import mlx.core as mx
from typing import Union, Sequence, Optional, Tuple, Any, List

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Any

def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Convert input to an MLX array.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Ignored for MLX backend (always uses Metal on Apple Silicon)
        
    Returns:
        MLX array representation of the input
    
    Raises:
        TypeError: If x is a tensor from another backend
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    # Check if x is a tensor from another backend
    if hasattr(x, '__class__') and 'Tensor' in x.__class__.__name__ and not isinstance(x, mx.array):
        raise TypeError(f"Cannot convert tensor of type {type(x)} to MLX array. "
                        f"Use the appropriate backend for this tensor type.")
    
    if isinstance(x, mx.array):
        array = x
    else:
        array = mx.array(x, dtype=dtype)
    
    # MLX doesn't support explicit device placement as it automatically
    # uses the most efficient device (Metal on Apple Silicon)
    
    return array

def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros with the specified shape
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    return mx.zeros(shape, dtype=dtype)

def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones.
    
    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones with the specified shape
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    return mx.ones(shape, dtype=dtype)

def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros with the same shape as the input.
    
    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of zeros with the same shape as x
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    x_array = convert_to_tensor(x)
    # MLX zeros_like doesn't accept dtype parameter
    if dtype is None:
        return mx.zeros_like(x_array)
    else:
        # Create zeros with the same shape but specified dtype
        return mx.zeros(x_array.shape, dtype=dtype)

def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones with the same shape as the input.
    
    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array of ones with the same shape as x
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    x_array = convert_to_tensor(x)
    # MLX ones_like doesn't accept dtype parameter
    if dtype is None:
        return mx.ones_like(x_array)
    else:
        # Create ones with the same shape but specified dtype
        return mx.ones(x_array.shape, dtype=dtype)

def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX identity matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX identity matrix of shape (n, m)
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    if m is None:
        m = n
    return mx.eye(n, m, dtype=dtype)

def reshape(x: ArrayLike, shape: Shape) -> mx.array:
    """
    Reshape an MLX array to a new shape.
    
    Args:
        x: Input array
        shape: New shape
        
    Returns:
        Reshaped MLX array
    """
    # Ensure shape is a sequence
    if isinstance(shape, int):
        shape = (shape,)
    return mx.reshape(convert_to_tensor(x), shape)

def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> mx.array:
    """
    Permute the dimensions of an MLX array.
    
    Args:
        x: Input array
        axes: Optional permutation of dimensions
        
    Returns:
        Transposed MLX array
    """
    x_array = convert_to_tensor(x)
    
    if axes is None:
        # Default transpose behavior (swap last two dimensions)
        ndim = len(x_array.shape)
        if ndim <= 1:
            return x_array
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
    
    return mx.transpose(x_array, axes)

def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> mx.array:
    """
    Concatenate MLX arrays along a specified axis.
    
    Args:
        arrays: Sequence of arrays
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated MLX array
    """
    return mx.concatenate([convert_to_tensor(arr) for arr in arrays], axis=axis)

def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> mx.array:
    """
    Stack MLX arrays along a new axis.
    
    Args:
        arrays: Sequence of arrays
        axis: Axis along which to stack
        
    Returns:
        Stacked MLX array
    """
    return mx.stack([convert_to_tensor(arr) for arr in arrays], axis=axis)

def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]], axis: int = 0) -> List[mx.array]:
    """
    Split an MLX array into sub-arrays.
    
    Args:
        x: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split
        
    Returns:
        List of sub-arrays
    """
    x_array = convert_to_tensor(x)
    # MLX split returns an array or a tuple of arrays
    result = mx.split(x_array, num_or_size_splits, axis=axis)
    
    # Convert to list if it's not already a list
    if isinstance(result, list):
        return result
    elif isinstance(result, tuple):
        return list(result)
    else:
        # If it's a single array, return a list with that array
        return [result]

def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> mx.array:
    """
    Insert new axes into an MLX array's shape.
    
    Args:
        x: Input array
        axis: Position(s) where new axes should be inserted
        
    Returns:
        MLX array with expanded dimensions
    """
    x_array = convert_to_tensor(x)
    
    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis):
            x_array = mx.expand_dims(x_array, ax)
        return x_array
    
    return mx.expand_dims(x_array, axis)

def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> mx.array:
    """
    Remove single-dimensional entries from an MLX array's shape.
    
    Args:
        x: Input array
        axis: Position(s) where dimensions should be removed
        
    Returns:
        MLX array with squeezed dimensions
    """
    x_array = convert_to_tensor(x)
    
    if axis is None:
        return mx.squeeze(x_array)
    
    if isinstance(axis, (list, tuple)):
        for ax in sorted(axis, reverse=True):  # Squeeze from highest dim to lowest
            x_array = mx.squeeze(x_array, ax)
        return x_array
    
    return mx.squeeze(x_array, axis)

def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of an MLX array.
    
    Args:
        x: Input array
        
    Returns:
        Shape of the array
    """
    return convert_to_tensor(x).shape

def dtype(x: ArrayLike) -> Any:
    """
    Get the data type of an MLX array.
    
    Args:
        x: Input array
        
    Returns:
        Data type of the array
    """
    return convert_to_tensor(x).dtype

def cast(x: ArrayLike, dtype: DType) -> mx.array:
    """
    Cast an MLX array to a different data type.
    
    Args:
        x: Input array
        dtype: Target data type
        
    Returns:
        MLX array with the target data type
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    return mx.array(convert_to_tensor(x), dtype=dtype)

def copy(x: ArrayLike) -> mx.array:
    """
    Create a copy of an MLX array.
    
    Args:
        x: Input array
        
    Returns:
        Copy of the array
    """
    # MLX arrays are immutable, so we can just convert to a new array
    return convert_to_tensor(x)

def to_numpy(x: ArrayLike) -> Any:
    """
    Convert a tensor to a list representation that can be converted to NumPy.
    
    Args:
        x: Input array
        
    Returns:
        List representation of the array
    """
    x_array = convert_to_tensor(x)
    return x_array.tolist()


def tensor_scatter_nd_update(array: ArrayLike, indices: ArrayLike, updates: ArrayLike) -> mx.array:
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
    # Create a copy of the tensor
    array_tensor = convert_to_tensor(array)
    indices_tensor = convert_to_tensor(indices)
    updates_tensor = convert_to_tensor(updates)
    
    # Ensure indices are integers
    indices_tensor = mx.array(indices_tensor, dtype=mx.int32)
    
    # Create a copy of the tensor
    updated_tensor = mx.array(array_tensor)
    
    # Iterate over the indices and apply updates
    for i in range(indices_tensor.shape[0]):
        # Extract indices for this update
        idx = []
        for j in range(indices_tensor.shape[1]):
            # Get each dimension's index value
            idx.append(indices_tensor[i, j].item())
        
        # Apply the update directly using tuple indexing
        updated_tensor[tuple(idx)] = updates_tensor[i]
    
    return updated_tensor


def full(shape: Shape, fill_value: Union[float, int], dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value.
    
    Args:
        shape: Shape of the array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array filled with the specified value
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    return mx.full(shape, fill_value, dtype=dtype)


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    if stop is None:
        # If only one argument is provided, it's the stop value
        return mx.arange(start=0, stop=start, step=step, dtype=dtype)
    return mx.arange(start=start, stop=stop, step=step, dtype=dtype)


class MLXTensorOps:
    """MLX implementation of tensor operations."""
    
    def __init__(self):
        """Initialize MLX tensor operations."""
        self._default_device = 'mps'  # Metal Performance Shaders
        self._current_seed = None
    
    def zeros(self, shape, dtype=None, device=None):
        """Create a tensor of zeros."""
        return zeros(shape, dtype=dtype, device=device)
    
    def ones(self, shape, dtype=None, device=None):
        """Create a tensor of ones."""
        return ones(shape, dtype=dtype, device=device)
    
    def full(self, shape, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value."""
        return full(shape, fill_value, dtype=dtype, device=device)
    
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
    
    def to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        return to_numpy(x)
    
    def tensor_scatter_nd_update(self, array, indices, updates):
        """Update elements of a tensor at specified indices with given values."""
        return tensor_scatter_nd_update(array, indices, updates)
    
    def get_default_device(self):
        """Get the default device for tensor operations."""
        return self._default_device
    
    def set_default_device(self, device):
        """Set the default device for tensor operations."""
        self._default_device = device
    
    def synchronize(self, device=None):
        """Synchronize the specified device."""
        # MLX handles synchronization automatically
        pass
    
    def get_seed(self):
        """Get the current random seed."""
        return self._current_seed