"""
NumPy tensor operations for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Any, Sequence, Type

# Import psutil if available
try:
    import psutil  # type: ignore
except ImportError:
    # Fallback for when psutil is not available
    # We'll handle this case in the memory_info function
    psutil = None  # type: ignore

# Type aliases
ArrayLike = Union[np.ndarray, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Union[np.dtype, str, None]


def convert_to_tensor(x: ArrayLike, dtype: DType = None,
                      device: Optional[str] = None) -> np.ndarray:
    """
    Convert input to a NumPy array.

    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array representation of the input

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
            and not isinstance(x, np.ndarray)
            and getattr(x.__class__, '__name__', '') != 'EmberTensor'):
            raise TypeError(
                f"Cannot convert tensor of type {type(x)} to NumPy array. "
                f"Use the appropriate backend for this tensor type."
            )

    return np.asarray(x, dtype=dtype)


def zeros(shape: Shape, dtype: DType = None,
          device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of zeros.

    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array of zeros with the specified shape
    """
    return np.zeros(shape, dtype=dtype)


def ones(shape: Shape, dtype: DType = None,
         device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of ones.

    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array of ones with the specified shape
    """
    return np.ones(shape, dtype=dtype)


def zeros_like(x: ArrayLike, dtype: DType = None,
               device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of zeros with the same shape as the input.

    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array of zeros with the same shape as x
    """
    return np.zeros_like(x, dtype=dtype)


def ones_like(x: ArrayLike, dtype: DType = None,
              device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array of ones with the same shape as the input.

    Args:
        x: Input array
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array of ones with the same shape as x
    """
    return np.ones_like(x, dtype=dtype)


def eye(n: int, m: Optional[int] = None, dtype: DType = None,
        device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy identity matrix.

    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy identity matrix of shape (n, m)
    """
    return np.eye(n, m, dtype=dtype)


def reshape(x: ArrayLike, shape: Shape) -> np.ndarray:
    """
    Reshape a NumPy array to a new shape.

    Args:
        x: Input array
        shape: New shape

    Returns:
        Reshaped NumPy array
    """
    return np.reshape(x, shape)


def transpose(x: ArrayLike,
              axes: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Permute the dimensions of a NumPy array.

    Args:
        x: Input array
        axes: Optional permutation of dimensions

    Returns:
        Transposed NumPy array
    """
    return np.transpose(x, axes)


def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> np.ndarray:
    """
    Insert new axes into a NumPy array's shape.

    Args:
        x: Input array
        axis: Position(s) where new axes should be inserted

    Returns:
        NumPy array with expanded dimensions
    """
    if isinstance(axis, (list, tuple)):
        result = convert_to_tensor(x)
        for ax in sorted(axis):
            result = np.expand_dims(result, ax)
        return result
    return np.expand_dims(x, axis)


def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """
    Concatenate NumPy arrays along a specified axis.

    Args:
        arrays: Sequence of arrays
        axis: Axis along which to concatenate

    Returns:
        Concatenated NumPy array
    """
    tensors = [convert_to_tensor(arr) for arr in arrays]
    return np.concatenate(tensors, axis=axis)


def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> np.ndarray:
    """
    Stack NumPy arrays along a new axis.

    Args:
        arrays: Sequence of arrays
        axis: Axis along which to stack

    Returns:
        Stacked NumPy array
    """
    return np.stack([convert_to_tensor(arr) for arr in arrays], axis=axis)


def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]],
          axis: int = 0) -> List[np.ndarray]:
    """
    Split a NumPy array into sub-arrays.

    Args:
        x: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split

    Returns:
        List of sub-arrays
    """
    return np.split(x, num_or_size_splits, axis=axis)


def squeeze(x: ArrayLike,
            axis: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
    """
    Remove single-dimensional entries from a NumPy array's shape.

    Args:
        x: Input array
        axis: Position(s) where dimensions should be removed

    Returns:
        NumPy array with squeezed dimensions
    """
    return np.squeeze(x, axis=axis)


def tile(x: ArrayLike, reps: Sequence[int]) -> np.ndarray:
    """
    Construct a NumPy array by tiling a given array.

    Args:
        x: Input array
        reps: Number of repetitions along each dimension

    Returns:
        Tiled NumPy array
    """
    return np.tile(x, reps)


def gather(x: ArrayLike, indices: Any, axis: int = 0) -> np.ndarray:
    """
    Gather slices from a NumPy array along an axis.

    Args:
        x: Input array
        indices: Indices of slices to gather
        axis: Axis along which to gather

    Returns:
        Gathered NumPy array
    """
    x_tensor = convert_to_tensor(x)
    indices_tensor = convert_to_tensor(indices)

    # Create a list of slice objects for each dimension
    slices = []
    for i in range(x_tensor.ndim):
        if i == axis:
            # Convert indices_tensor to a list for proper indexing
            if hasattr(indices_tensor, 'tolist'):
                slices.append(indices_tensor.tolist())
            else:
                slices.append(indices_tensor)
        else:
            slices.append(slice(None, None))  # type: ignore

    return x_tensor[tuple(slices)]


def tensor_scatter_nd_update(array: ArrayLike, indices: ArrayLike, updates: ArrayLike) -> np.ndarray:
    """
    Updates elements of an array at specified indices with given values.

    Args:
        array: The array to be updated.
        indices: An array of indices, where each row represents
                 the index of an element to be updated.
        updates: An array of update values, with the same
                 length as the number of rows in indices.

    Returns:
        A new array with the updates applied.
    """
    array_tensor = convert_to_tensor(array)
    indices_tensor = convert_to_tensor(indices)
    updates_tensor = convert_to_tensor(updates)
    
    output_array = array_tensor.copy()
    for i in range(indices_tensor.shape[0]):
        output_array[tuple(indices_tensor[i])] = updates_tensor[i]
    
    return output_array


def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of a NumPy array.

    Args:
        x: Input array

    Returns:
        Shape of the array
    """
    return convert_to_tensor(x).shape


def dtype(x: ArrayLike) -> np.dtype:
    """
    Get the data type of a NumPy array.

    Args:
        x: Input array

    Returns:
        Data type of the array
    """
    return convert_to_tensor(x).dtype


def cast(x: ArrayLike, dtype: DType) -> np.ndarray:
    """
    Cast a NumPy array to a different data type.

    Args:
        x: Input array
        dtype: Target data type

    Returns:
        NumPy array with the target data type
    """
    return convert_to_tensor(x).astype(dtype)


def copy(x: ArrayLike) -> np.ndarray:
    """
    Create a copy of a NumPy array.

    Args:
        x: Input array

    Returns:
        Copy of the array
    """
    return convert_to_tensor(x).copy()


def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.

    Args:
        x: Input array

    Returns:
        NumPy array
    """
    return convert_to_tensor(x)


def full(shape: Shape, fill_value: Union[float, int],
         dtype: Optional[DType] = None,
         device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array filled with a scalar value.

    Args:
        shape: Shape of the array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array filled with the specified value
    """
    return np.full(shape, fill_value, dtype=dtype)


def full_like(x: ArrayLike, fill_value: Union[float, int],
              dtype: Optional[DType] = None,
              device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array filled with a scalar value.

    Args:
        x: Input array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array filled with the specified value with the same shape as x
    """
    return np.full_like(x, fill_value, dtype=dtype)


def linspace(start: float, stop: float, num: int,
             dtype: Optional[DType] = None,
             device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array with evenly spaced values
    """
    return np.linspace(start, stop, num, dtype=dtype)


def arange(start: int, stop: Optional[int] = None, step: int = 1,
           dtype: Optional[DType] = None,
           device: Optional[str] = None) -> np.ndarray:
    """
    Create a NumPy array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Ignored for NumPy backend

    Returns:
        NumPy array with evenly spaced values
    """
    if stop is None:
        # If only one argument is provided, it's the stop value
        return np.arange(0, start, step, dtype=dtype)
    return np.arange(start, stop, step, dtype=dtype)


def pi() -> np.ndarray:
    """
    Return the mathematical constant pi.

    Returns:
        The value of pi as a NumPy array
    """
    return np.array(np.pi)


def sign(x: ArrayLike) -> np.ndarray:
    """
    Compute the sign of a NumPy array element-wise.

    Args:
        x: Input array

    Returns:
        Element-wise sign
    """
    return np.sign(x)


def from_numpy_dtype(dtype: Type) -> Type:
    """
    Convert a NumPy data type to a NumPy data type.

    Args:
        dtype: The NumPy data type to convert

    Returns:
        The corresponding NumPy data type
    """
    return dtype


def get_dtype(name: str) -> Type:
    """
    Get a data type by name.

    Args:
        name: The name of the data type

    Returns:
        The corresponding NumPy data type
    """
    if name == 'float32':
        return np.float32
    elif name == 'float64':
        return np.float64
    elif name == 'int32':
        return np.int32
    elif name == 'int64':
        return np.int64
    elif name == 'bool' or name == 'bool_':
        return np.bool_
    elif name == 'int8':
        return np.int8
    elif name == 'int16':
        return np.int16
    elif name == 'uint8':
        return np.uint8
    elif name == 'uint16':
        return np.uint16
    elif name == 'uint32':
        return np.uint32
    elif name == 'uint64':
        return np.uint64
    elif name == 'float16':
        return np.float16
    else:
        raise ValueError(f"Unknown data type: {name}")


def to_numpy_dtype(dtype: Any) -> Type:
    """
    Convert a data type to a NumPy data type.

    Args:
        dtype: The data type to convert

    Returns:
        The corresponding NumPy data type
    """
    if isinstance(dtype, type) and hasattr(np, dtype.__name__):
        return dtype
    elif isinstance(dtype, str):
        return get_dtype(dtype)
    else:
        raise ValueError(f"Cannot convert {dtype} to NumPy data type")


def memory_info(device: Optional[str] = None) -> dict:
    """
    Get memory information for the specified device.

    Args:
        device: Device to get memory information for (default: current device)

    Returns:
        Dictionary containing memory information
    """
    if device is not None and device != 'cpu':
        raise ValueError(
            f"NumPy backend only supports 'cpu' device, got {device}"
        )

    # Get system memory information
    if psutil is not None:
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'percent': mem.percent
        }
    else:
        # Fallback when psutil is not available
        return {
            'total': 0,
            'available': 0,
            'used': 0,
            'percent': 0
        }


def is_available(device_type: str) -> bool:
    """
    Check if a device type is available.

    Args:
        device_type: Device type to check

    Returns:
        True if the device type is available, False otherwise
    """
    return device_type == 'cpu'


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
    return x_array.item()


def slice(x: ArrayLike, starts: Sequence[int], sizes: Sequence[int]) -> np.ndarray:
    """
    Extract a slice from a tensor.
    
    Args:
        x: Input tensor
        starts: Starting indices for each dimension
        sizes: Size of the slice in each dimension. A value of -1 means "all remaining elements in this dimension"
        
    Returns:
        Sliced tensor
    """
    x_array = convert_to_tensor(x)
    
    # Create a list of slice objects for each dimension
    slices = []
    for i, (start, size) in enumerate(zip(starts, sizes)):
        if size == -1:
            # -1 means "all remaining elements in this dimension"
            # Use Python's built-in slice constructor directly with type ignore
            slices.append(slice(start, None))  # type: ignore
        else:
            # Use Python's built-in slice constructor directly with type ignore
            slices.append(slice(start, start + size))  # type: ignore
    
    # Extract the slice
    return x_array[tuple(slices)]

def slice_update(x: ArrayLike, slices: Union[List, Tuple], updates: ArrayLike) -> np.ndarray:
    """
    Update a tensor at specific indices.
    
    Args:
        x: Input tensor to update
        slices: List or tuple of slice objects or indices
        updates: Values to insert at the specified indices
        
    Returns:
        Updated tensor
    """
    x_array = convert_to_tensor(x)
    updates_array = convert_to_tensor(updates)
    
    # Create a copy of the input tensor
    result = x_array.copy()
    
    # Update the tensor at the specified indices
    result[tuple(slices)] = updates_array
    
    return result


def sort(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """
    Sort a tensor along a specified axis.
    
    Args:
        x: Input tensor
        axis: Axis along which to sort
        
    Returns:
        Sorted tensor
    """
    x_tensor = convert_to_tensor(x)
    return np.sort(x_tensor, axis=axis)


def pad(x: ArrayLike, paddings: Sequence[Sequence[int]], constant_values: Union[int, float] = 0) -> np.ndarray:
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
    x_array = convert_to_tensor(x)
    
    # Convert paddings to the format expected by np.pad
    # NumPy expects a tuple of (pad_before, pad_after) for each dimension
    pad_width = tuple(tuple(p) for p in paddings)
    
    # Pad the tensor
    return np.pad(x_array, pad_width, mode='constant', constant_values=constant_values)


class NumpyTensorOps:
    """NumPy implementation of tensor operations."""

    def __init__(self):
        """Initialize NumPy tensor operations."""
        self._default_device = 'cpu'
        self._current_seed = None

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
        """Create a tensor with evenly spaced values."""
        return arange(start, stop=stop, step=step, dtype=dtype, device=device)

    def linspace(self, start, stop, num, dtype=None, device=None):
        """Create a tensor with evenly spaced values."""
        return linspace(start, stop, num, dtype=dtype, device=device)

    def full(self, shape, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value."""
        return full(shape, fill_value, dtype=dtype, device=device)

    def full_like(self, x, fill_value, dtype=None, device=None):
        """Create a tensor filled with a scalar value."""
        return full_like(x, fill_value, dtype=dtype, device=device)

    def reshape(self, x, shape):
        """Reshape a tensor to a new shape."""
        return reshape(x, shape)

    def transpose(self, x, axes=None):
        """Permute the dimensions of a tensor."""
        return transpose(x, axes=axes)

    def concatenate(self, tensors, axis=0):
        """Concatenate tensors along an axis."""
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
        """Update elements of an array at specified indices with given values."""
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

    def to_numpy(self, x):
        """Convert a tensor to a NumPy array."""
        return to_numpy(x)

    def pi(self):
        """Return the mathematical constant pi."""
        return pi()

    def sign(self, x):
        """Compute the sign of a tensor element-wise."""
        return sign(x)

    def from_numpy_dtype(self, dtype):
        """Convert a NumPy data type to a NumPy data type."""
        return from_numpy_dtype(dtype)

    def get_dtype(self, name):
        """Get a data type by name."""
        return get_dtype(name)

    def to_numpy_dtype(self, dtype):
        """Convert a data type to a NumPy data type."""
        return to_numpy_dtype(dtype)

    def memory_info(self, device=None):
        """Get memory information for the specified device."""
        return memory_info(device)

    def is_available(self, device_type):
        """Check if a device type is available."""
        return is_available(device_type)

    def get_default_device(self):
        """Get the default device for tensor operations."""
        return self._default_device

    def set_default_device(self, device):
        """Set the default device for tensor operations."""
        if device != 'cpu':
            raise ValueError(
                f"NumPy backend only supports 'cpu' device, got {device}"
            )

        self._default_device = device

    def synchronize(self, device=None):
        """Synchronize the specified device."""
        # NumPy is synchronous, so this is a no-op
        pass

    def get_seed(self):
        """Get the current random seed."""
        return self._current_seed

    # Data type properties
    def float16(self):
        """Get the float16 data type."""
        return np.float16

    def float32(self):
        """Get the float32 data type."""
        return np.float32

    def float64(self):
        """Get the float64 data type."""
        return np.float64

    def int8(self):
        """Get the int8 data type."""
        return np.int8

    def int16(self):
        """Get the int16 data type."""
        return np.int16

    def int32(self):
        """Get the int32 data type."""
        return np.int32

    def int64(self):
        """Get the int64 data type."""
        return np.int64

    def uint8(self):
        """Get the uint8 data type."""
        return np.uint8

    def uint16(self):
        """Get the uint16 data type."""
        return np.uint16

    def uint32(self):
        """Get the uint32 data type."""
        return np.uint32

    def uint64(self):
        """Get the uint64 data type."""
        return np.uint64

    def bool_(self):
        """Get the boolean data type."""
        return np.bool_
        
    def item(self, x):
        """Extract the scalar value from a tensor."""
        return item(x)
    
    def slice(self, x, starts, sizes):
        """Extract a slice from a tensor."""
        return slice(x, starts, sizes)
        
    def slice_update(self, x, slices, updates):
        """Update a tensor at specific indices."""
        return slice_update(x, slices, updates)
        
    def pad(self, x, paddings, constant_values=0):
        """Pad a tensor with a constant value."""
        return pad(x, paddings, constant_values)

    def sort(self, x, axis=-1):
        """Sort a tensor along a specified axis."""
        return sort(x, axis=axis)
