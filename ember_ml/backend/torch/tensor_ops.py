"""
PyTorch tensor operations for ember_ml.

This module provides PyTorch implementations of tensor operations
with automatic device selection for optimal performance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Any, Sequence

# Type aliases
ArrayLike = Union[torch.Tensor, list, tuple, float, int]
Shape = Union[int, Sequence[int]]
DType = Union[torch.dtype, str, None]

# Import from config and dtype_ops
from ember_ml.backend.torch.config import DEFAULT_DEVICE
from ember_ml.backend.torch.dtype_ops import from_dtype_str as ember_dtype_to_torch


def _prepare_tensor_args(shape: Optional[Shape] = None, dtype: DType = None, device: Optional[str] = None):
    """
    Helper function to prepare arguments for tensor creation functions.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device
        
    Returns:
        Tuple of (shape, dtype, device) with proper conversions applied
    """
    # Convert dtype to PyTorch dtype
    torch_dtype = ember_dtype_to_torch(dtype)
    
    # Handle shape conversion for torch
    shape_tuple = None
    if shape is not None:
        if isinstance(shape, int):
            shape_tuple = (shape,)
        else:
            shape_tuple = tuple(shape) if not isinstance(shape, tuple) else shape  # type: ignore
    
    # Use the specified device or the default device
    target_device = device or DEFAULT_DEVICE
    
    return shape_tuple, torch_dtype, target_device


def convert_to_tensor(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Convert input to a PyTorch tensor with automatic device selection.
    
    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses DEFAULT_DEVICE)
        
    Returns:
        PyTorch tensor representation of the input
    
    Raises:
        TypeError: If x is a tensor from another backend
    """
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
            and not isinstance(x, torch.Tensor)
            and getattr(x.__class__, '__name__', '') != 'EmberTensor'):
            raise TypeError(f"Cannot convert tensor of type {type(x)} to PyTorch tensor. "
                            f"Use the appropriate backend for this tensor type.")
    
    # Convert dtype and get device
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    # Create tensor
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, (list, tuple)) and len(x) == 1:
        # Handle single-element lists/tuples like scalars
        tensor = torch.tensor(x[0]).to(torch_dtype)
    elif isinstance(x, (int, float)):
        # Handle scalars separately to avoid TypeError with dtype
        tensor = torch.tensor(x).to(torch_dtype)
    else:
        tensor = torch.tensor(x, dtype=torch_dtype)

    # Only move to device if it's different from current device
    if tensor.device.type != target_device:
        try:
            tensor = tensor.to(target_device)
        except RuntimeError:
            # Fallback if device is not available
            print(f"Warning: Failed to move tensor to {target_device}, falling back to default device")
            tensor = tensor.to(DEFAULT_DEVICE)
    
    return tensor


def zeros(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with automatic device selection.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses DEFAULT_DEVICE)
        
    Returns:
        Tensor of zeros with the specified shape
    """
    shape_tuple, torch_dtype, target_device = _prepare_tensor_args(shape, dtype, device)
    
    if shape_tuple is None:
        raise ValueError("Shape cannot be None for zeros operation")
    
    try:
        return torch.zeros(shape_tuple, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.zeros(shape_tuple, dtype=torch_dtype, device=DEFAULT_DEVICE)


def ones(shape: Shape, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with automatic device selection.
    
    Args:
        shape: Shape of the tensor
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses DEFAULT_DEVICE)
        
    Returns:
        Tensor of ones with the specified shape
    """
    shape_tuple, torch_dtype, target_device = _prepare_tensor_args(shape, dtype, device)
    
    if shape_tuple is None:
        raise ValueError("Shape cannot be None for ones operation")
    
    try:
        return torch.ones(shape_tuple, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.ones(shape_tuple, dtype=torch_dtype, device=DEFAULT_DEVICE)


def zeros_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of zeros with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses same as input or DEFAULT_DEVICE)
        
    Returns:
        Tensor of zeros with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        return torch.zeros_like(x_tensor, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.zeros_like(x_tensor, dtype=torch_dtype, device=DEFAULT_DEVICE)


def ones_like(x: ArrayLike, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor of ones with the same shape as the input.
    
    Args:
        x: Input tensor
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses same as input or DEFAULT_DEVICE)
        
    Returns:
        Tensor of ones with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        return torch.ones_like(x_tensor, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.ones_like(x_tensor, dtype=torch_dtype, device=DEFAULT_DEVICE)


def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create an identity matrix with automatic device selection.
    
    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device to place the tensor on (if None, uses DEFAULT_DEVICE)
        
    Returns:
        Identity matrix of shape (n, m)
    """
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        # Handle the case where m is None
        if m is None:
            return torch.eye(n, dtype=torch_dtype, device=target_device)
        else:
            return torch.eye(n, m=m, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        if m is None:
            return torch.eye(n, dtype=torch_dtype, device=DEFAULT_DEVICE)
        else:
            return torch.eye(n, m=m, dtype=torch_dtype, device=DEFAULT_DEVICE)


def reshape(x: ArrayLike, shape: Shape) -> torch.Tensor:
    """
    Reshape a tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    shape_tuple, _, _ = _prepare_tensor_args(shape)
    if shape_tuple is None:
        raise ValueError("Shape cannot be None for reshape operation")
    return convert_to_tensor(x).reshape(shape_tuple)


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


def expand_dims(x: ArrayLike, axis: Union[int, List[int], Tuple[int, ...]]) -> torch.Tensor:
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
            # Convert to int explicitly since PyTorch expects an int
            result = result.unsqueeze(int(ax))
        return result
    
    # Handle single axis - convert to int explicitly
    return x_tensor.unsqueeze(int(axis))


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
        return list(torch.split(x_tensor, int(split_size), dim=axis))
    
    # Convert to list if it's a sequence but not a list
    if not isinstance(num_or_size_splits, list):
        num_or_size_splits = list(num_or_size_splits)
        
    return list(torch.split(x_tensor, num_or_size_splits, dim=axis))


def squeeze(x: ArrayLike, axis: Optional[Union[int, List[int], Tuple[int, ...]]] = None) -> torch.Tensor:
    """
    Remove single-dimensional entries from a tensor's shape.
    
    Args:
        x: Input tensor
        axis: Position(s) where dimensions should be removed
        
    Returns:
        Tensor with squeezed dimensions
    """
    x_tensor = convert_to_tensor(x)
    
    # PyTorch 2.0+ uses dim parameter for squeeze
    if axis is None:
        return torch.squeeze(x_tensor)
    
    if isinstance(axis, (list, tuple)):
        result = x_tensor
        for ax in sorted(axis, reverse=True):  # Squeeze from highest dim to lowest
            # Convert to int and use as dim parameter
            dim_val = int(ax)
            result = result.squeeze(dim=dim_val)
        return result
    
    # Convert to int and use as dim parameter
    dim_val = int(axis)
    return x_tensor.squeeze(dim=dim_val)


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
    torch_dtype = ember_dtype_to_torch(dtype)
    return convert_to_tensor(x).to(torch_dtype)


def copy(x: ArrayLike) -> torch.Tensor:
    """
    Create a copy of a tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Copy of the tensor
    """
    return convert_to_tensor(x).clone()


def item(x: ArrayLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    This method extracts the scalar value from a tensor containing a single element.
    
    Args:
        x: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.item()


def slice(x: ArrayLike, starts: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
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
    
    if not starts:
        raise ValueError("starts parameter cannot be empty")
    
    if not sizes:
        raise ValueError("sizes parameter cannot be empty")
    
    if len(starts) != len(sizes):
        raise ValueError(f"starts and sizes must have the same length, got {len(starts)} and {len(sizes)}")
    
    # Create a list of slice objects for each dimension
    slices = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            # Use Python's slice syntax directly
            slices.append(slice(start, None))  # type: ignore
        else:
            # Use Python's slice syntax directly
            slices.append(slice(start, start + size))  # type: ignore
    
    # Extract the slice
    return x_tensor[tuple(slices)]


def slice_update(x: ArrayLike, slices: Union[List, Tuple], updates: ArrayLike) -> torch.Tensor:
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
    result = x_tensor.clone()
    
    # Update the tensor at the specified indices
    result[tuple(slices)] = updates_tensor
    
    return result


def pad(x: ArrayLike, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> torch.Tensor:
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
    
    # Convert paddings to the format expected by torch.nn.functional.pad
    # PyTorch expects (pad_left, pad_right, pad_top, pad_bottom, ...)
    # We need to reverse the order of dimensions and flatten the pairs
    torch_paddings: List[int] = []
    for padding in reversed(paddings):
        torch_paddings.extend(padding)
    
    # Pad the tensor
    return F.pad(x_tensor, torch_paddings, mode='constant', value=constant_values)


def scatter(values: ArrayLike, index: ArrayLike, out_size: Optional[int] = None, 
            aggr: str = "add", axis: int = 0) -> torch.Tensor:
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
    index_tensor = convert_to_tensor(index).long()  # Ensure index is long type
    
    # Determine output size if not provided
    _out_size = out_size if out_size is not None else values_tensor.shape[0]
    
    # Handle different aggregation methods
    if aggr == "mean":
        return scatter_mean(values_tensor, index_tensor, _out_size, axis)
    
    # Create output tensor shape
    out_shape = list(values_tensor.shape)
    out_shape[axis] = _out_size
    
    # Create empty tensor for output
    empty_tensor = torch.zeros(out_shape, dtype=values_tensor.dtype, device=values_tensor.device)
    
    # Apply appropriate scatter operation
    if aggr == "add":
        return scatter_add(empty_tensor, index_tensor, values_tensor, axis)
    elif aggr == "max":
        return scatter_max(empty_tensor, index_tensor, values_tensor, axis)
    elif aggr == "min":
        return scatter_min(empty_tensor, index_tensor, values_tensor, axis)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggr}")


def scatter_add(src: ArrayLike, index: ArrayLike, values: ArrayLike, axis: int = 0) -> torch.Tensor:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the sum of the values will be assigned to these index.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        axis: Axis along which to scatter
        
    Returns:
        The resulting array after applying scatter and sum operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index).long()  # Ensure index is long type
    values_tensor = convert_to_tensor(values)
    
    # Use PyTorch's scatter_add_ method
    result = src_tensor.clone()
    return result.scatter_add(axis, index_tensor, values_tensor)


def scatter_max(src: ArrayLike, index: ArrayLike, values: ArrayLike, axis: int = 0) -> torch.Tensor:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the maximum value is kept at these indices.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        axis: Axis along which to scatter
        
    Returns:
        The resulting array after applying scatter and max operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index).long()  # Ensure index is long type
    values_tensor = convert_to_tensor(values)
    
    # Use PyTorch's scatter_reduce_ method with 'amax' reduction
    result = src_tensor.clone()
    # PyTorch 1.12+ supports scatter_reduce_ with 'amax'
    try:
        return result.scatter_reduce_(axis, index_tensor, values_tensor, reduce='amax')
    except (AttributeError, RuntimeError):
        # Fallback for older PyTorch versions
        # We need to handle this manually
        for i in range(index_tensor.shape[0]):
            idx = index_tensor[i]
            val = values_tensor[i]
            # Create a tensor for indexing
            idx_tensor = torch.zeros_like(result, dtype=torch.bool)
            idx_tensor.index_fill_(axis, idx.view(-1), True)
            # Update with maximum values
            result = torch.where(idx_tensor, torch.maximum(result, val), result)
        return result


def scatter_min(src: ArrayLike, index: ArrayLike, values: ArrayLike, axis: int = 0) -> torch.Tensor:
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present,
    the minimum value is kept at these indices.
    
    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'
        values: Input array containing values to be scattered
        axis: Axis along which to scatter
        
    Returns:
        The resulting array after applying scatter and min operations
    """
    src_tensor = convert_to_tensor(src)
    index_tensor = convert_to_tensor(index).long()  # Ensure index is long type
    values_tensor = convert_to_tensor(values)
    
    # Use PyTorch's scatter_reduce_ method with 'amin' reduction
    result = src_tensor.clone()
    # PyTorch 1.12+ supports scatter_reduce_ with 'amin'
    try:
        return result.scatter_reduce_(axis, index_tensor, values_tensor, reduce='amin')
    except (AttributeError, RuntimeError):
        # Fallback for older PyTorch versions
        # We need to handle this manually
        for i in range(index_tensor.shape[0]):
            idx = index_tensor[i]
            val = values_tensor[i]
            # Create a tensor for indexing
            idx_tensor = torch.zeros_like(result, dtype=torch.bool)
            idx_tensor.index_fill_(axis, idx.view(-1), True)
            # Update with minimum values
            result = torch.where(idx_tensor, torch.minimum(result, val), result)
        return result


def scatter_mean(values: ArrayLike, index: ArrayLike, out_size: int, axis: int = 0) -> torch.Tensor:
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
    index_tensor = convert_to_tensor(index).long()  # Ensure index is long type
    
    # Use scatter_add to sum values by index
    out_shape = list(values_tensor.shape)
    out_shape[axis] = out_size
    
    # Create empty tensors for sum and count
    sum_tensor = torch.zeros(out_shape, dtype=values_tensor.dtype, device=values_tensor.device)
    count_tensor = torch.zeros(out_size, dtype=torch.long, device=values_tensor.device)
    
    # Scatter add the values
    sum_result = scatter_add(sum_tensor, index_tensor, values_tensor, axis)
    
    # Count occurrences of each index
    ones = torch.ones_like(index_tensor)
    count_result = scatter_add(count_tensor, index_tensor, ones, 0)
    
    # Avoid division by zero
    count_result = torch.clamp(count_result, min=1)
    
    # Broadcast count_result to match the shape of sum_result for division
    if axis != 0 or sum_result.ndim > 1:
        # Create shape for broadcasting
        broadcast_shape = [1] * sum_result.ndim
        broadcast_shape[axis] = out_size
        count_broadcast = count_result.view(broadcast_shape)
        
        # Create expanded shape for broadcasting
        expanded_shape = list(sum_result.shape)
        expanded_shape[axis] = 1
        count_broadcast = count_broadcast.expand(sum_result.shape)
    else:
        count_broadcast = count_result
    
    # Compute mean by dividing sum by count
    return sum_result / count_broadcast


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


def var(x: ArrayLike, axis: Optional[Union[int, List[int], Tuple[int, ...]]] = None, keepdims: bool = False) -> torch.Tensor:
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
            result = torch.var(result, dim=int(ax), keepdim=keepdims)
        return result
    
    return torch.var(x_tensor, dim=int(axis), keepdim=keepdims)


def full(shape: Shape, fill_value: Union[float, int], dtype: Optional[DType] = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor filled with a scalar value with automatic device selection.
    
    Args:
        shape: Shape of the tensor
        fill_value: Value to fill the tensor with
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        Tensor filled with the specified value
    """
    shape_tuple, torch_dtype, target_device = _prepare_tensor_args(shape, dtype, device)
    
    if shape_tuple is None:
        raise ValueError("Shape cannot be None for full operation")
    
    try:
        return torch.full(shape_tuple, fill_value, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.full(shape_tuple, fill_value, dtype=torch_dtype, device=DEFAULT_DEVICE)


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
    x_tensor = convert_to_tensor(x)
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        return torch.full_like(x_tensor, fill_value, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.full_like(x_tensor, fill_value, dtype=torch_dtype, device=DEFAULT_DEVICE)


def linspace(start: float, stop: float, num: int, dtype: DType = None, device: Optional[str] = None) -> torch.Tensor:
    """
    Create a tensor with evenly spaced values within a given interval.
    
    Args:
        start: Start of the interval
        stop: End of the interval
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device (defaults to DEFAULT_DEVICE if None)
        
    Returns:
        Tensor with evenly spaced values
    """
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        return torch.linspace(start, stop, num, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        return torch.linspace(start, stop, num, dtype=torch_dtype, device=DEFAULT_DEVICE)


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
    _, torch_dtype, target_device = _prepare_tensor_args(dtype=dtype, device=device)
    
    try:
        if stop is None:
            # If only one argument is provided, it's the stop value
            return torch.arange(start=0, end=start, step=step, dtype=torch_dtype, device=target_device)
        return torch.arange(start=start, end=stop, step=step, dtype=torch_dtype, device=target_device)
    except RuntimeError:
        # Fallback if device is not available
        print(f"Warning: Failed to create tensor on {target_device}, falling back to default device")
        if stop is None:
            return torch.arange(start=0, end=start, step=step, dtype=torch_dtype, device=DEFAULT_DEVICE)
        return torch.arange(start=start, end=stop, step=step, dtype=torch_dtype, device=DEFAULT_DEVICE)


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
        
    def item(self, x):
        """Extract the scalar value from a tensor."""
        return item(x)
    
    def slice(self, x, starts, sizes):
        """Extract a slice from a tensor."""
        if not starts or not sizes:
            raise ValueError("starts and sizes parameters cannot be empty")
        # Use the fully qualified name to avoid confusion with Python's built-in slice
        from ember_ml.backend.torch.tensor_ops import slice as tensor_slice
        return tensor_slice(x, starts, sizes)
        
    def slice_update(self, x, slices, updates):
        """Update a tensor at specific indices."""
        return slice_update(x, slices, updates)
        
    def pad(self, x, paddings, constant_values=0):
        """Pad a tensor with a constant value."""
        return pad(x, paddings, constant_values)
    
    def scatter(self, values, index, out_size=None, aggr="add", axis=0):
        """Scatter values into a tensor along a specified axis."""
        return scatter(values, index, out_size, aggr, axis)
    
    def scatter_add(self, src, index, values, axis=0):
        """Scatter and add values at specified indices."""
        return scatter_add(src, index, values, axis)
    
    def scatter_max(self, src, index, values, axis=0):
        """Scatter and take maximum values at specified indices."""
        return scatter_max(src, index, values, axis)
    
    def scatter_min(self, src, index, values, axis=0):
        """Scatter and take minimum values at specified indices."""
        return scatter_min(src, index, values, axis)
    
    def scatter_mean(self, values, index, out_size, axis=0):
        """Scatter and compute mean values at specified indices."""
        return scatter_mean(values, index, out_size, axis)