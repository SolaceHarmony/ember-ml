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
    # Handle EmberDtype objects
    elif hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype.name)
    
    # Handle EmberTensor specially by checking class name and data attribute
    # This avoids importing EmberTensor which would cause circular imports
    if isinstance(x, object):  # Type guard for attribute access
        if (getattr(x.__class__, '__name__', '') == 'EmberTensor'
            and hasattr(x, 'data')):
            # Safe to access data after type checking
            data = getattr(x, 'data')
            return convert_to_tensor(data, dtype=dtype, device=device)
    
        # Check if x is a tensor from another backend
        if ('Tensor' in getattr(x.__class__, '__name__', '')
            and not isinstance(x, mx.array)
            and getattr(x.__class__, '__name__', '') != 'EmberTensor'):
            raise TypeError(f"Cannot convert tensor of type {type(x)} to MLX array. "
                            f"Use the appropriate backend for this tensor type.")
    
    if isinstance(x, mx.array):
        array = x
    elif isinstance(x, (list, tuple)) and len(x) == 1:
        # Handle single-element lists/tuples like scalars
        array = mx.array(x[0]) if dtype is None else mx.array(x[0]).astype(dtype)
    elif isinstance(x, (int, float)):
        # Handle scalars separately
        array = mx.array(x) if dtype is None else mx.array(x).astype(dtype)
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
    # Handle EmberDtype objects
    elif hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype.name)
    
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
    # Handle EmberDtype objects
    elif hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype.name)
    
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

def item(x: ArrayLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    This method extracts the scalar value from a tensor containing a single element.
    
    Args:
        x: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    x_array = convert_to_tensor(x)
    
    # Get the raw value
    raw_value = x_array.item()
    
    # Handle different types explicitly to ensure we return the expected types
    if isinstance(raw_value, bool):
        return bool(raw_value)
    elif isinstance(raw_value, int):
        return int(raw_value)
    elif isinstance(raw_value, float):
        return float(raw_value)
    elif raw_value is True or raw_value is False:
        return bool(raw_value)
    
    # For other types, determine the best conversion based on the value
    try:
        # Try to convert to int if it looks like an integer
        if isinstance(raw_value, (str, bytes)) and raw_value.isdigit():
            return 0  # Default to 0 for safety
        # For numeric-looking values, convert to float
        return 0.0  # Default to 0.0 for safety
    except (ValueError, TypeError, AttributeError):
        # If all else fails, return False
        return False


def to_numpy(x: ArrayLike) -> Any:
    """
    Convert a tensor to a NumPy array.
    
    Args:
        x: Input array
        
    Returns:
        NumPy array representation of the tensor
    """
    x_array = convert_to_tensor(x)
    
    # This is allowed as an exception to the no NumPy rule
    # since to_numpy is specifically for NumPy conversion
    import numpy as np
    
    # Handle bfloat16 arrays by converting to float32 first
    if x_array.dtype == mx.bfloat16:
        x_array = mx.array(x_array, dtype=mx.float32)
    
    # Use NumPy's buffer protocol support for efficient conversion
    # This creates a copy of the array
    result = np.array(x_array)
    
    return result


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


def full_like(x: ArrayLike, fill_value: Union[float, int], dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value with the same shape as the input.
    
    Args:
        x: Input array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array filled with the specified value with the same shape as x
    """
    x_array = convert_to_tensor(x)
    
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    
    # If dtype is None, use the dtype of the input array
    if dtype is None:
        dtype = x_array.dtype
    
    # Create a full array with the same shape as the input
    return mx.full(x_array.shape, fill_value, dtype=dtype)


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
    # Handle EmberDtype objects
    elif hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype.name)
    
    if stop is None:
        # If only one argument is provided, it's the stop value
        return mx.arange(start=0, stop=start, step=step, dtype=dtype)
    return mx.arange(start=start, stop=stop, step=step, dtype=dtype)

def linspace(start: float, stop: float, num: int, dtype: DType = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.
    
    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array with evenly spaced values
    """
    # Handle string dtype values
    if isinstance(dtype, str):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype)
    # Handle EmberDtype objects
    elif hasattr(dtype, 'name') and hasattr(dtype, 'ember_dtype'):
        from ember_ml.backend.mlx.dtype_ops import get_dtype
        dtype = get_dtype(dtype.name)
    
    return mx.linspace(start=start, stop=stop, num=num, dtype=dtype)


def tile(x: ArrayLike, reps: Sequence[int]) -> mx.array:
    """
    Construct an MLX array by tiling a given array.

    Args:
        x: Input array
        reps: Number of repetitions along each dimension

    Returns:
        Tiled MLX array
    """
    x_tensor = convert_to_tensor(x)
    return mx.tile(x_tensor, reps)


def gather(x: ArrayLike, indices: Any, axis: int = 0) -> mx.array:
    """
    Gather slices from an MLX array along an axis.

    Args:
        x: Input array
        indices: Indices of slices to gather
        axis: Axis along which to gather

    Returns:
        Gathered MLX array
    """
    x_tensor = convert_to_tensor(x)
    indices_tensor = convert_to_tensor(indices)
    return mx.take(x_tensor, indices_tensor, axis=axis)


def tensor_slice(x: ArrayLike, starts: Sequence[int], sizes: Sequence[int]) -> mx.array:
    """
    Extract a slice from a tensor.
    
    Args:
        x: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Create a list of slice objects for each dimension
    slice_objects = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            slice_objects.append(slice(int(start), None))
        else:
            slice_objects.append(slice(int(start), int(start + size)))
    
    # Extract the slice
    return x_tensor[tuple(slice_objects)]

def slice_update(x: ArrayLike, slices: Union[List, Tuple], updates: ArrayLike) -> mx.array:
    """
    Update a tensor at specific indices.
    
    Args:
        x: Input tensor to update
        slices: List or tuple of slice objects or indices
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    x_tensor = convert_to_tensor(x)
    updates_tensor = convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    # MLX arrays are immutable, so we need to create a new array
    result = mx.array(x_tensor)
    
    # Convert slices to start_indices and axes
    if isinstance(slices, (list, tuple)):
        # Extract start indices and axes
        start_indices = []
        axes = []
        
        for i, s in enumerate(slices):
            if hasattr(s, 'start') and hasattr(s, 'stop'):
                # For slice objects, use the start index
                start_idx = s.start if s.start is not None else 0
                start_indices.append(start_idx)
                axes.append(i)
            else:
                # For direct indices
                start_indices.append(s)
                axes.append(i)
        
        # Convert to MLX arrays
        start_indices_array = mx.array(start_indices, dtype=mx.int32)
        
        # Update the tensor at the specified indices
        result = mx.slice_update(result, updates_tensor, start_indices_array, tuple(axes))
    else:
        # Single index case
        start_indices_array = mx.array([slices], dtype=mx.int32)
        result = mx.slice_update(result, updates_tensor, start_indices_array, (0,))
    
    return result


def pad(x: ArrayLike, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> mx.array:
    """
    Pad a tensor with a constant value.
    
    Args:
        x: Input tensor
        paddings: Sequence of sequences of integers specifying the padding for each dimension
                 Each inner sequence should contain two integers: [pad_before, pad_after]
        constant_values: Value to pad with
        
    Returns:
        Padded tensor
    """
    x_tensor = convert_to_tensor(x)
    
    # Convert paddings to the format expected by mx.pad
    # MLX expects a tuple of (pad_before, pad_after) for each dimension
    pad_width = tuple(tuple(p) for p in paddings)
    
    # Pad the tensor
    return mx.pad(x_tensor, pad_width, constant_values)

def scatter(values: ArrayLike, index: ArrayLike, out_size: Optional[int] = None,
            aggr: str = "add", axis: int = 0) -> mx.array:
    """
    Scatter values into a tensor along a specified axis.
    
    Args:
        values: Array with all the values to scatter in the output tensor
        index: Array with index to which scatter the values
        out_size: Number of elements in the output array (size of the first dimension).
                 If not provided, uses the number of elements in `values`
        aggr: Scattering method employed for reduction at index ("add", "max", "mean", "min")
        axis: Axis on which applying the scattering
        
    Returns:
        Array with `out_size` elements containing the scattered values at given index
    """
    values_tensor = convert_to_tensor(values)
    index_tensor = convert_to_tensor(index)
    
    # Ensure index is int32
    index_tensor = mx.array(index_tensor, dtype=mx.int32)
    
    # Determine output size if not provided
    _out_size = out_size if out_size is not None else values_tensor.shape[0]
    
    # Handle different aggregation methods
    if aggr == "mean":
        return scatter_mean(values_tensor, index_tensor, _out_size, axis)
    
    # Create output tensor shape
    out_shape = list(values_tensor.shape)
    out_shape[axis] = _out_size
    
    # Create empty tensor for output
    empty_tensor = mx.zeros(out_shape, dtype=values_tensor.dtype)
    
    # Apply appropriate scatter operation
    if aggr == "add":
        return scatter_add(empty_tensor, index_tensor, values_tensor)
    elif aggr == "max":
        return scatter_max(empty_tensor, index_tensor, values_tensor)
    elif aggr == "min":
        return scatter_min(empty_tensor, index_tensor, values_tensor)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggr}")


def scatter_add(src: ArrayLike, index: ArrayLike, values: ArrayLike) -> mx.array:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the sum of the values will be assigned to these index.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        
    Returns:
        The resulting array after applying scatter and sum operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index)
    values_tensor = convert_to_tensor(values)
    
    # Ensure index is int32
    index_tensor = mx.array(index_tensor, dtype=mx.int32)
    
    # Use MLX's at[].add method for scatter_add
    return src_tensor.at[index_tensor].add(values_tensor)


def scatter_max(src: ArrayLike, index: ArrayLike, values: ArrayLike) -> mx.array:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the maximum value is kept at these indices.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        
    Returns:
        The resulting array after applying scatter and max operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index)
    values_tensor = convert_to_tensor(values)
    
    # Ensure index is int32
    index_tensor = mx.array(index_tensor, dtype=mx.int32)
    
    # Use MLX's at[].maximum method for scatter_max
    return src_tensor.at[index_tensor].maximum(values_tensor)


def scatter_min(src: ArrayLike, index: ArrayLike, values: ArrayLike) -> mx.array:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the minimum value is kept at these indices.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        
    Returns:
        The resulting array after applying scatter and min operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index)
    values_tensor = convert_to_tensor(values)
    
    # Ensure index is int32
    index_tensor = mx.array(index_tensor, dtype=mx.int32)
    
    # Use MLX's at[].minimum method for scatter_min
    return src_tensor.at[index_tensor].minimum(values_tensor)


def scatter_mean(values: ArrayLike, index: ArrayLike, out_size: int, axis: int = 0) -> mx.array:
    """
    Computes the mean of values that are scattered along a specified axis, grouped by index.
    
    Args:
        values: Input array containing values to be scattered
        index: Array containing indices that determine the scatter of the `values`
        out_size: Size of the output array
        axis: Axis along which to scatter
        
    Returns:
        An array containing mean of `values` grouped by `index`
    """
    values_tensor = convert_to_tensor(values)
    index_tensor = convert_to_tensor(index)
    
    # Ensure index is int32
    index_tensor = mx.array(index_tensor, dtype=mx.int32)
    
    # Use scatter_add to sum values by index
    scatt_add = scatter(values_tensor, index_tensor, out_size, aggr="add", axis=axis)
    
    # Calculate degrees (count of each index)
    degrees = mx.zeros((out_size,), dtype=mx.int32)
    degrees = degrees.at[index_tensor].add(mx.ones_like(index_tensor))
    
    # Avoid division by zero
    degrees = mx.where(degrees < 1, mx.array(1), degrees)
    
    # Broadcast degrees to match shape of scatt_add for division
    if axis != 0 or scatt_add.ndim > 1:
        # Create shape for broadcasting
        broadcast_shape = [1] * scatt_add.ndim
        broadcast_shape[axis] = out_size
        degrees = mx.reshape(degrees, broadcast_shape)
        
        # Create expanded shape for broadcasting
        expanded_shape = list(scatt_add.shape)
        expanded_shape[axis] = 1
        degrees = mx.tile(degrees, expanded_shape)
    
    # Compute mean by dividing sum by count
    return mx.divide(scatt_add, degrees)


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
    
    def full_like(self, x, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value with the same shape as the input."""
        return full_like(x, fill_value, dtype=dtype, device=device)
    
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
        
    def item(self, x):
        """Extract the scalar value from a tensor."""
        return item(x)
    
    def tensor_scatter_nd_update(self, array, indices, updates):
        """Update elements of a tensor at specified indices with given values."""
        return tensor_scatter_nd_update(array, indices, updates)
    
    def tile(self, x, reps):
        """Construct a tensor by tiling a given tensor."""
        return tile(x, reps)
    
    def gather(self, x, indices, axis=0):
        """Gather slices from a tensor along an axis."""
        return gather(x, indices, axis=axis)
    
    def slice(self, x, starts, sizes):
        """Extract a slice from a tensor."""
        return tensor_slice(x, starts, sizes)
        
    def slice_update(self, x, slices, updates):
        """Update a tensor at specific indices."""
        return slice_update(x, slices, updates)
        
    def pad(self, x, paddings, constant_values=0):
        """Pad a tensor with a constant value."""
        return pad(x, paddings, constant_values)
    
    def scatter(self, x, indices, updates, axis=-1):
        """Scatter updates into a tensor along a specified axis."""
        return scatter(x, indices, updates, axis=axis)
    
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