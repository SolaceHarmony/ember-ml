"""
MLX backend for emberharmony.

This module provides MLX implementations of the emberharmony backend interface,
optimized for Apple Silicon (M1/M2/M3) processors.
"""

import numpy as np
import mlx.core as mx
import math
from typing import Union, Sequence, Optional, Tuple, Any, List, Dict, Type

# Clear the MLX memory cache when the module is imported
# This ensures the MLX backend is in a clean state
mx.metal.clear_cache()

# Type aliases
ArrayLike = Union[mx.array, float, int, list, tuple]
Shape = Union[int, Sequence[int]]
DType = Any

# Backend information
__version__ = getattr(mx, "__version__", "unknown")
has_gpu = False  # MLX uses Metal on Apple Silicon, not CUDA
has_mps = True   # MLX is optimized for Metal Performance Shaders
default_float_type = mx.float32

# Define power function
power = None  # Will be set to pow after it's defined


# Data type operations
def bool_():
    """Get the boolean data type."""
    return getattr(mx, "bool_", np.bool_)


def float16():
    """Get the float16 data type."""
    return mx.float16


def float32():
    """Get the float32 data type."""
    return mx.float32


def float64():
    """Get the float64 data type."""
    # MLX doesn't support float64, use float32 instead
    return mx.float32


def int8():
    """Get the int8 data type."""
    return mx.int8


def int16():
    """Get the int16 data type."""
    return mx.int16


def int32():
    """Get the int32 data type."""
    return mx.int32


def int64():
    """Get the int64 data type."""
    # MLX doesn't support int64, use int32 instead
    return mx.int32


def uint8():
    """Get the uint8 data type."""
    return mx.uint8


def uint16():
    """Get the uint16 data type."""
    return mx.uint16


def uint32():
    """Get the uint32 data type."""
    return mx.uint32


def uint64():
    """Get the uint64 data type."""
    # MLX doesn't support uint64, use uint32 instead
    return mx.uint32


def get_dtype(name: str) -> Any:
    """
    Get a data type by name.

    Args:
        name: The name of the data type

    Returns:
        The corresponding MLX data type
    """
    if name == 'float32':
        return mx.float32
    elif name == 'float64':
        # MLX doesn't support float64, use float32 instead
        return mx.float32
    elif name == 'int32':
        return mx.int32
    elif name == 'int64':
        # MLX doesn't support int64, use int32 instead
        return mx.int32
    elif name == 'bool' or name == 'bool_':
        return bool_()
    elif name == 'int8':
        return mx.int8
    elif name == 'int16':
        return mx.int16
    elif name == 'uint8':
        return mx.uint8
    elif name == 'uint16':
        return mx.uint16
    elif name == 'uint32':
        return mx.uint32
    elif name == 'uint64':
        # MLX doesn't support uint64, use uint32 instead
        return mx.uint32
    elif name == 'float16':
        return mx.float16
    else:
        raise ValueError(f"Unknown data type: {name}")


def from_numpy_dtype(dtype: Type) -> Any:
    """
    Convert a NumPy data type to an MLX data type.

    Args:
        dtype: The NumPy data type to convert

    Returns:
        The corresponding MLX data type
    """
    if dtype == np.float32:
        return mx.float32
    elif dtype == np.float64:
        # MLX doesn't support float64, use float32 instead
        return mx.float32
    elif dtype == np.int32:
        return mx.int32
    elif dtype == np.int64:
        # MLX doesn't support int64, use int32 instead
        return mx.int32
    elif dtype == np.bool_:
        return bool_()
    elif dtype == np.int8:
        return mx.int8
    elif dtype == np.int16:
        return mx.int16
    elif dtype == np.uint8:
        return mx.uint8
    elif dtype == np.uint16:
        return mx.uint16
    elif dtype == np.uint32:
        return mx.uint32
    elif dtype == np.uint64:
        # MLX doesn't support uint64, use uint32 instead
        return mx.uint32
    elif dtype == np.float16:
        return mx.float16
    else:
        raise ValueError(f"Cannot convert {dtype} to MLX data type")


def to_numpy_dtype(dtype: Any) -> Type:
    """
    Convert an MLX data type to a NumPy data type.

    Args:
        dtype: The MLX data type to convert

    Returns:
        The corresponding NumPy data type
    """
    if dtype == mx.float32:
        return np.float32
    elif dtype == mx.int32:
        return np.int32
    elif dtype == bool_():
        return np.bool_
    elif dtype == mx.int8:
        return np.int8
    elif dtype == mx.int16:
        return np.int16
    elif dtype == mx.uint8:
        return np.uint8
    elif dtype == mx.uint16:
        return np.uint16
    elif dtype == mx.uint32:
        return np.uint32
    elif dtype == mx.float16:
        return np.float16
    elif isinstance(dtype, str):
        return to_numpy_dtype(get_dtype(dtype))
    else:
        raise ValueError(f"Cannot convert {dtype} to NumPy data type")


# Tensor operations
def convert_to_tensor(x: Any, dtype: Optional[DType] = None,
                      device: Optional[str] = None) -> mx.array:
    """
    Convert input to an MLX array.

    Args:
        x: Input data (array, tensor, scalar)
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array representation of the input
    """
    # Handle EmberTensor objects
    if hasattr(x, '__class__') and x.__class__.__name__ == 'EmberTensor':
        # Extract the underlying data from the EmberTensor
        return convert_to_tensor(x.data, dtype=dtype, device=device)

    # If already an MLX array, just handle dtype conversion if needed
    if isinstance(x, mx.array):
        if dtype is not None:
            if dtype == int or dtype == 'int32':
                return x.astype(mx.int32)
            elif dtype == float or dtype == 'float32':
                return x.astype(mx.float32)
            elif dtype == 'float64':
                # MLX doesn't support float64, use float32 instead
                return x.astype(mx.float32)
            # If we can't determine the dtype, return the array as is
            return x
        return x

    # For all other types, try to convert to MLX array
    try:
        # For MLX, we need to pass a valid MLX dtype or None
        if dtype is not None:
            if dtype == int or dtype == 'int32':
                return mx.array(x, dtype=mx.int32)
            elif dtype == float or dtype == 'float32':
                return mx.array(x, dtype=mx.float32)
            elif dtype == 'float64':
                # MLX doesn't support float64, use float32 instead
                return mx.array(x, dtype=mx.float32)
            # If we can't determine the dtype, use the default
            return mx.array(x)
        else:
            # Use the default dtype
            return mx.array(x)
    except Exception as e:
        raise TypeError(f"Cannot convert {type(x)} to MLX array: {str(e)}")


def zeros(shape: Shape, dtype: Optional[DType] = None,
          device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros.

    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array of zeros with the specified shape
    """
    # For MLX, we need to pass a valid MLX dtype or None
    # If dtype is None, MLX will use the default dtype (float32)
    if dtype is not None:
        if dtype == int or dtype == 'int32':
            return mx.zeros(shape, dtype=mx.int32)
        elif dtype == float or dtype == 'float32':
            return mx.zeros(shape, dtype=mx.float32)
        elif dtype == 'float64':
            # MLX doesn't support float64, use float32 instead
            return mx.zeros(shape, dtype=mx.float32)

        # If we can't determine the dtype, use the default
        return mx.zeros(shape)
    else:
        # Use the default dtype
        return mx.zeros(shape)


def ones(shape: Shape, dtype: Optional[DType] = None,
         device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones.

    Args:
        shape: Shape of the array
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array of ones with the specified shape
    """
    # For MLX, we need to pass a valid MLX dtype or None
    # If dtype is None, MLX will use the default dtype (float32)
    if dtype is not None:
        if dtype == int or dtype == 'int32':
            return mx.ones(shape, dtype=mx.int32)
        elif dtype == float or dtype == 'float32':
            return mx.ones(shape, dtype=mx.float32)
        elif dtype == 'float64':
            # MLX doesn't support float64, use float32 instead
            return mx.ones(shape, dtype=mx.float32)

        # If we can't determine the dtype, use the default
        return mx.ones(shape)
    else:
        # Use the default dtype
        return mx.ones(shape)


def zeros_like(x: ArrayLike, dtype: Optional[DType] = None,
               device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of zeros with the same shape as the input.

    Args:
        x: Input array
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array of zeros with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    result = mx.zeros_like(x_tensor)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def ones_like(x: ArrayLike, dtype: Optional[DType] = None,
              device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array of ones with the same shape as the input.

    Args:
        x: Input array
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array of ones with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    result = mx.ones_like(x_tensor)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def eye(n: int, m: Optional[int] = None, dtype: Optional[DType] = None,
        device: Optional[str] = None) -> mx.array:
    """
    Create an MLX identity matrix.

    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX identity matrix of shape (n, m)
    """
    if m is None:
        m = n
    return mx.eye(n, m, dtype=dtype)


def full(shape: Shape, fill_value: Union[float, int],
         dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value.

    Args:
        shape: Shape of the array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array filled with the specified value
    """
    return mx.full(shape, fill_value, dtype=dtype)


def full_like(x: ArrayLike, fill_value: Union[float, int],
              dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array filled with a scalar value with the same shape as the input.

    Args:
        x: Input array
        fill_value: Value to fill the array with
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array filled with the specified value with the same shape as x
    """
    x_tensor = convert_to_tensor(x)
    shape = x_tensor.shape
    return full(shape, fill_value, dtype=dtype, device=device)


def arange(start: int, stop: Optional[int] = None, step: int = 1,
           dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (exclusive)
        step: Spacing between values
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with evenly spaced values
    """
    if stop is None:
        # If only one argument is provided, it's the stop value
        return mx.arange(0, start, step, dtype=dtype)
    return mx.arange(start, stop, step, dtype=dtype)


def linspace(start: float, stop: float, num: int,
             dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Create an MLX array with evenly spaced values within a given interval.

    Args:
        start: Start of interval (inclusive)
        stop: End of interval (inclusive)
        num: Number of values to generate
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with evenly spaced values
    """
    return mx.linspace(start, stop, num, dtype=dtype)


def reshape(x: ArrayLike, shape: Sequence[int]) -> mx.array:
    """
    Reshape an MLX array to a new shape.

    Args:
        x: Input array
        shape: New shape

    Returns:
        Reshaped MLX array
    """
    x_tensor = convert_to_tensor(x)
    return mx.reshape(x_tensor, shape)


def transpose(x: ArrayLike, axes: Optional[Sequence[int]] = None) -> mx.array:
    """
    Permute the dimensions of an MLX array.

    Args:
        x: Input array
        axes: Optional permutation of dimensions

    Returns:
        Transposed MLX array
    """
    x_tensor = convert_to_tensor(x)
    if axes is None:
        # For 3D or higher tensors, swap the last two dimensions
        if len(x_tensor.shape) >= 2:
            axes = list(range(len(x_tensor.shape)))
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            return x_tensor
    return mx.transpose(x_tensor, axes)


def concatenate(tensors: Sequence[ArrayLike], axis: int = 0) -> mx.array:
    """
    Concatenate MLX arrays along a specified axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to concatenate

    Returns:
        Concatenated MLX array
    """
    return mx.concatenate([convert_to_tensor(t) for t in tensors], axis=axis)


def stack(tensors: Sequence[ArrayLike], axis: int = 0) -> mx.array:
    """
    Stack MLX arrays along a new axis.

    Args:
        tensors: Sequence of arrays
        axis: Axis along which to stack

    Returns:
        Stacked MLX array
    """
    return mx.stack([convert_to_tensor(t) for t in tensors], axis=axis)


def split(x: ArrayLike, num_or_size_splits: Union[int, Sequence[int]],
          axis: int = 0) -> List[mx.array]:
    """
    Split an MLX array into sub-arrays.

    Args:
        x: Input array
        num_or_size_splits: Number of splits or sizes of each split
        axis: Axis along which to split

    Returns:
        List of sub-arrays
    """
    x_tensor = convert_to_tensor(x)
    if isinstance(num_or_size_splits, int):
        # MLX doesn't have a direct split function like NumPy or PyTorch
        # So we need to implement it manually
        total_size = x_tensor.shape[axis]
        section_size = mx.floor_divide(total_size, mx.array(num_or_size_splits))
        indices = []
        for i in range(1, num_or_size_splits):
            idx = mx.multiply(section_size, mx.array(i))
            # Convert to Python int without using int() directly on the mx.array
            idx_value = idx.item()
            indices.append(idx_value)
        return list(mx.split(x_tensor, indices, axis=axis))
    else:
        # For size_splits, we need to calculate the indices
        indices = []
        current_idx = mx.array(0)
        for size in num_or_size_splits[:-1]:
            current_idx = mx.add(current_idx, mx.array(size))
            # Convert to Python int without using int() directly on the mx.array
            idx_value = current_idx.item()
            indices.append(idx_value)
        return list(mx.split(x_tensor, indices, axis=axis))


def expand_dims(x: ArrayLike, axis: Union[int, Sequence[int]]) -> mx.array:
    """
    Insert new axes into an MLX array's shape.

    Args:
        x: Input array
        axis: Position(s) where new axes should be inserted

    Returns:
        MLX array with expanded dimensions
    """
    x_tensor = convert_to_tensor(x)
    if isinstance(axis, (list, tuple)):
        result = x_tensor
        for ax in sorted(axis):
            result = mx.expand_dims(result, ax)
        return result
    return mx.expand_dims(x_tensor, axis)


def squeeze(x: ArrayLike, axis: Optional[Union[int, Sequence[int]]] = None) -> mx.array:
    """
    Remove single-dimensional entries from an MLX array's shape.

    Args:
        x: Input array
        axis: Position(s) where dimensions should be removed

    Returns:
        MLX array with squeezed dimensions
    """
    x_tensor = convert_to_tensor(x)
    if axis is None:
        return mx.squeeze(x_tensor)
    if isinstance(axis, (list, tuple)):
        result = x_tensor
        for ax in sorted(axis, reverse=True):
            result = mx.squeeze(result, ax)
        return result
    return mx.squeeze(x_tensor, axis)


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


def copy(x: ArrayLike) -> mx.array:
    """
    Create a copy of an MLX array.

    Args:
        x: Input array

    Returns:
        Copy of the array
    """
    x_tensor = convert_to_tensor(x)
    # MLX arrays are immutable, so creating a new array with the same data is a copy
    return mx.array(x_tensor)


def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert an MLX array to a NumPy array.

    Args:
        x: Input array

    Returns:
        NumPy array representation of the input
    """
    x_tensor = convert_to_tensor(x)
    # Always return a NumPy array, not just a list
    if isinstance(x_tensor, mx.array):
        # Use tolist() to convert to Python list, then create NumPy array
        return np.asarray(x_tensor.tolist())
    else:
        # For non-MLX arrays, convert directly to NumPy
        return np.asarray(x_tensor)


def cast(x: ArrayLike, dtype: DType) -> mx.array:
    """
    Cast an MLX array to a different data type.

    Args:
        x: Input array
        dtype: Target data type

    Returns:
        MLX array with the target data type
    """
    x_tensor = convert_to_tensor(x)

    # For MLX, we need to pass a valid MLX dtype
    if dtype is not None:
        if dtype == int or dtype == 'int32':
            return x_tensor.astype(mx.int32)
        elif dtype == float or dtype == 'float32':
            return x_tensor.astype(mx.float32)
        elif dtype == 'float64':
            # MLX doesn't support float64, use float32 instead
            return x_tensor.astype(mx.float32)

        # If we can't determine the dtype, return the tensor as is
        return x_tensor
    else:
        # If no dtype is specified, return the tensor as is
        return x_tensor


def shape(x: ArrayLike) -> Tuple[int, ...]:
    """
    Get the shape of an MLX array.

    Args:
        x: Input array

    Returns:
        Shape of the array
    """
    x_tensor = convert_to_tensor(x)
    return x_tensor.shape


def dtype(x: ArrayLike) -> type:
    """
    Get the data type of an MLX array.

    Args:
        x: Input array

    Returns:
        Data type of the array
    """
    x_tensor = convert_to_tensor(x)
    return to_numpy_dtype(x_tensor.dtype)


# Math operations
def add(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Add two MLX arrays element-wise.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Element-wise sum of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.add(x_tensor, y_tensor)


def subtract(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Subtract two MLX arrays element-wise.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Element-wise difference of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.subtract(x_tensor, y_tensor)


def multiply(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Multiply two MLX arrays element-wise.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Element-wise product of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.multiply(x_tensor, y_tensor)


def divide(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Divide two MLX arrays element-wise.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Element-wise quotient of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.divide(x_tensor, y_tensor)


def dot(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the dot product of two MLX arrays.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Dot product of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    # MLX doesn't have a direct dot function, so we use matmul
    # This is a simplification that might need adjustment
    return mx.matmul(x_tensor, y_tensor)


def matmul(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute the matrix product of two MLX arrays.

    Args:
        x: First input array
        y: Second input array

    Returns:
        Matrix product of x and y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.matmul(x_tensor, y_tensor)


def pow(x: ArrayLike, y: ArrayLike) -> mx.array:
    """
    Compute x raised to the power of y.

    Args:
        x: Base
        y: Exponent

    Returns:
        x raised to the power of y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.power(x_tensor, y_tensor)


def abs(x: ArrayLike) -> mx.array:
    """
    Compute the absolute value of an MLX array.

    Args:
        x: Input array

    Returns:
        Absolute value of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.abs(x_tensor)


def exp(x: ArrayLike) -> mx.array:
    """
    Compute the exponential of an MLX array.

    Args:
        x: Input array

    Returns:
        Exponential of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.exp(x_tensor)


def log(x: ArrayLike) -> mx.array:
    """
    Compute the natural logarithm of an MLX array.

    Args:
        x: Input array

    Returns:
        Natural logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.log(x_tensor)


def log10(x: ArrayLike) -> mx.array:
    """
    Compute the base-10 logarithm of an MLX array.

    Args:
        x: Input array

    Returns:
        Base-10 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.log10(x_tensor)


def log2(x: ArrayLike) -> mx.array:
    """
    Compute the base-2 logarithm of an MLX array.

    Args:
        x: Input array

    Returns:
        Base-2 logarithm of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.log2(x_tensor)


def sqrt(x: ArrayLike) -> mx.array:
    """
    Compute the square root of an MLX array.

    Args:
        x: Input array

    Returns:
        Square root of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.sqrt(x_tensor)


def square(x: ArrayLike) -> mx.array:
    """
    Compute the square of an MLX array.

    Args:
        x: Input array

    Returns:
        Square of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.square(x_tensor)


def sin(x: ArrayLike) -> mx.array:
    """
    Compute the sine of an MLX array.

    Args:
        x: Input array

    Returns:
        Sine of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.sin(x_tensor)


def cos(x: ArrayLike) -> mx.array:
    """
    Compute the cosine of an MLX array.

    Args:
        x: Input array

    Returns:
        Cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.cos(x_tensor)


def tan(x: ArrayLike) -> mx.array:
    """
    Compute the tangent of an MLX array.

    Args:
        x: Input array

    Returns:
        Tangent of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.tan(x_tensor)


def sinh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic sine of an MLX array.

    Args:
        x: Input array

    Returns:
        Hyperbolic sine of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.sinh(x_tensor)


def cosh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic cosine of an MLX array.

    Args:
        x: Input array

    Returns:
        Hyperbolic cosine of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.cosh(x_tensor)


def tanh(x: ArrayLike) -> mx.array:
    """
    Compute the hyperbolic tangent of an MLX array.

    Args:
        x: Input array

    Returns:
        Hyperbolic tangent of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.tanh(x_tensor)


def sigmoid(x: ArrayLike) -> mx.array:
    """
    Compute the sigmoid of an MLX array.

    Args:
        x: Input array

    Returns:
        Sigmoid of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.sigmoid(x_tensor)


def relu(x: ArrayLike) -> mx.array:
    """
    Compute the rectified linear unit of an MLX array.

    Args:
        x: Input array

    Returns:
        ReLU of x
    """
    x_tensor = convert_to_tensor(x)
    # Use mx.maximum instead of importing mlx.nn
    return mx.maximum(x_tensor, mx.array(0))


def softmax(x: ArrayLike, axis: int = -1) -> mx.array:
    """
    Compute the softmax of an MLX array.

    Args:
        x: Input array
        axis: Axis along which to compute the softmax

    Returns:
        Softmax of x
    """
    x_tensor = convert_to_tensor(x)
    # Convert to float32 if needed
    if x_tensor.dtype not in [mx.float16, mx.float32]:
        x_tensor = x_tensor.astype(mx.float32)

    # Implement softmax directly without importing mlx.nn
    # Subtract max for numerical stability
    x_max = mx.max(x_tensor, axis=axis, keepdims=True)
    exp_x = mx.exp(mx.subtract(x_tensor, x_max))
    return mx.divide(exp_x, mx.sum(exp_x, axis=axis, keepdims=True))


def clip(
    x: ArrayLike,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> mx.array:
    """
    Clip the values of an MLX array to the specified range.

    Args:
        x: Input array
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped array
    """
    x_tensor = convert_to_tensor(x)
    return mx.clip(x_tensor, min_val, max_val)


def sign(x: ArrayLike) -> mx.array:
    """
    Compute the sign of an MLX array.

    Args:
        x: Input array

    Returns:
        Sign of x
    """
    x_tensor = convert_to_tensor(x)
    return mx.sign(x_tensor)


def sum(
    x: ArrayLike,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False
) -> mx.array:
    """
    Compute the sum of an MLX array along the specified axis.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the sum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Sum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    return mx.sum(x_tensor, axis=axis, keepdims=keepdims)


def mean(
    x: ArrayLike,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False
) -> mx.array:
    """
    Compute the mean of an MLX array along the specified axis.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the mean
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Mean of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    return mx.mean(x_tensor, axis=axis, keepdims=keepdims)


def max(
    x: ArrayLike,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False
) -> mx.array:
    """
    Compute the maximum of an MLX array along the specified axis.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the maximum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Maximum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    return mx.max(x_tensor, axis=axis, keepdims=keepdims)


def min(
    x: ArrayLike,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False
) -> mx.array:
    """
    Compute the minimum of an MLX array along the specified axis.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the minimum
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Minimum of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    return mx.min(x_tensor, axis=axis, keepdims=keepdims)


def var(
    x: ArrayLike,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False
) -> mx.array:
    """
    Compute the variance of an MLX array along the specified axis.

    Args:
        x: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Variance of x along the specified axis
    """
    x_tensor = convert_to_tensor(x)
    return mx.var(x_tensor, axis=axis, keepdims=keepdims)


def pi() -> mx.array:
    """
    Return the mathematical constant pi.

    Returns:
        The value of pi as an MLX array
    """
    # Use a direct value instead of np.pi
    return mx.array(3.141592653589793)


# Comparison operations
def equal(x: Any, y: Any) -> mx.array:
    """
    Check if two tensors are equal element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x == y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.equal(x_tensor, y_tensor)


def not_equal(x: Any, y: Any) -> mx.array:
    """
    Check if two tensors are not equal element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x != y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.not_equal(x_tensor, y_tensor)


def less(x: Any, y: Any) -> mx.array:
    """
    Check if one tensor is less than another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x < y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.less(x_tensor, y_tensor)


def less_equal(x: Any, y: Any) -> mx.array:
    """
    Check if one tensor is less than or equal to another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x <= y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.less_equal(x_tensor, y_tensor)


def greater(x: Any, y: Any) -> mx.array:
    """
    Check if one tensor is greater than another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x > y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.greater(x_tensor, y_tensor)


def greater_equal(x: Any, y: Any) -> mx.array:
    """
    Check if one tensor is greater than or equal to another element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x >= y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.greater_equal(x_tensor, y_tensor)


def logical_and(x: Any, y: Any) -> mx.array:
    """
    Compute the logical AND of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x AND y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.logical_and(x_tensor, y_tensor)


def logical_or(x: Any, y: Any) -> mx.array:
    """
    Compute the logical OR of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x OR y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    return mx.logical_or(x_tensor, y_tensor)


def logical_not(x: Any) -> mx.array:
    """
    Compute the logical NOT of a tensor element-wise.

    Args:
        x: Input tensor

    Returns:
        Boolean MLX array with True where NOT x
    """
    x_tensor = convert_to_tensor(x)
    return mx.logical_not(x_tensor)


def logical_xor(x: Any, y: Any) -> mx.array:
    """
    Compute the logical XOR of two tensors element-wise.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Boolean MLX array with True where x XOR y
    """
    x_tensor = convert_to_tensor(x)
    y_tensor = convert_to_tensor(y)
    # MLX doesn't have a direct logical_xor function, so we implement it using other operations
    return mx.logical_or(
        mx.logical_and(x_tensor, mx.logical_not(y_tensor)),
        mx.logical_and(mx.logical_not(x_tensor), y_tensor)
    )


# Random operations
def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed: Random seed
    """
    mx.random.seed(seed)


def set_random_seed(seed: int) -> None:
    """
    Alias for set_seed for backward compatibility.

    Args:
        seed: Random seed
    """
    set_seed(seed)


def get_seed() -> Optional[int]:
    """
    Get the current random seed.

    Returns:
        Current random seed
    """
    # MLX doesn't provide a way to get the current seed
    return None


def random_normal(shape: Sequence[int], mean: float = 0.0, stddev: float = 1.0,
                  dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a normal distribution.

    Args:
        shape: Shape of the output array
        mean: Mean of the normal distribution
        stddev: Standard deviation of the normal distribution
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from a normal distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    # MLX's normal function takes scale (stddev) as parameter
    result = mx.random.normal(shape=shape, loc=mean, scale=stddev)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def random_uniform(shape: Sequence[int], minval: float = 0.0, maxval: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a uniform distribution.

    Args:
        shape: Shape of the output array
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from a uniform distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    # MLX's uniform function takes low and high as parameters
    result = mx.random.uniform(shape=shape, low=minval, high=maxval)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def random_binomial(shape: Sequence[int], p: float = 0.5,
                    dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a binomial distribution.

    Args:
        shape: Shape of the output array
        p: Probability of success
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from a binomial distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    # MLX doesn't have a direct binomial function, so we'll use uniform and threshold
    uniform = mx.random.uniform(shape=shape)
    result = mx.array(uniform < p, dtype=mx.int32)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def random_exponential(shape: Sequence[int], scale: float = 1.0,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from an exponential distribution.

    Args:
        shape: Shape of the output array
        scale: Scale parameter
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from an exponential distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    # Generate uniform random values
    u = mx.random.uniform(shape=shape)

    # Transform to exponential distribution
    # Exponential distribution: f(x) = (1/scale) * exp(-x/scale)
    # Can be sampled by taking -scale * ln(U) where U is uniform(0,1)
    # Avoid log(0) by using 1-u instead of u
    scale_tensor = mx.array(scale)
    result = mx.multiply(mx.negative(scale_tensor), mx.log(mx.subtract(mx.array(1.0), u)))

    if dtype is not None:
        result = result.astype(dtype)

    return result


def random_gamma(shape: Sequence[int], alpha: float = 1.0, beta: float = 1.0,
                 dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a gamma distribution.

    Args:
        shape: Shape of the output array
        alpha: Shape parameter
        beta: Scale parameter
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from a gamma distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")

    # For alpha = 1, gamma is equivalent to exponential
    if alpha == 1.0:
        return random_exponential(shape, scale=beta, dtype=dtype, device=device)

    # For integer alpha, we can use the sum of exponentials
    if isinstance(alpha, int) and alpha > 1:
        result = mx.zeros(shape)
        for _ in range(alpha):
            result = mx.add(result, random_exponential(shape, scale=beta, dtype=None, device=device))

        if dtype is not None:
            result = result.astype(dtype)

        return result

    # For non-integer alpha, we use the Marsaglia and Tsang method
    # This is a simplified version that works for alpha > 1
    # For alpha < 1, we would need a more complex algorithm
    d = mx.subtract(mx.array(alpha), mx.divide(mx.array(1.0), mx.array(3.0)))
    c = mx.divide(mx.array(1.0), mx.sqrt(mx.multiply(mx.array(9.0), d)))

    result = mx.zeros(shape)
    valid_samples = mx.zeros(shape, dtype=bool_())

    # Keep generating until all samples are valid
    while not mx.all(valid_samples):
        # Generate standard normal samples
        z = mx.random.normal(shape=shape)

        # Calculate v = (1 + c*z)^3
        v = mx.power(mx.add(mx.array(1.0), mx.multiply(c, z)), mx.array(3.0))

        # Filter out invalid samples (v <= 0)
        v_valid = mx.greater(v, mx.array(0.0))

        # Calculate log acceptance ratio
        u = mx.random.uniform(shape=shape)
        log_accept = mx.add(
            mx.add(
                mx.multiply(mx.array(0.5), mx.square(z)),
                d
            ),
            mx.subtract(
                mx.negative(mx.multiply(d, v)),
                mx.multiply(d, mx.log(v))
            )
        )

        # Accept samples where log(u) < log_accept
        accept = mx.less(mx.log(u), log_accept)

        # Update valid samples and result
        new_valid = mx.logical_and(
            mx.logical_and(v_valid, accept),
            mx.logical_not(valid_samples)
        )
        result = mx.where(new_valid, mx.multiply(mx.multiply(d, v), beta), result)
        valid_samples = mx.logical_or(valid_samples, new_valid)

    if dtype is not None:
        result = result.astype(dtype)

    return result


def random_poisson(shape: Sequence[int], lam: float = 1.0,
                   dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Generate random values from a Poisson distribution.

    Args:
        shape: Shape of the output array
        lam: Rate parameter
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random values from a Poisson distribution
    """
    # Convert shape to sequence if it's an int
    if isinstance(shape, int):
        shape = (shape,)

    # Convert lambda to MLX array if it's a scalar
    if isinstance(lam, (int, float)):
        lam_array = mx.full(shape, lam)
    else:
        lam_array = convert_to_tensor(lam)

    # Initialize counts and time accumulators
    counts = mx.zeros(shape, dtype=mx.int32)
    times = mx.zeros(shape)

    # Generate exponential waiting times until exceeding 1.0
    # This is based on the fact that Poisson process events have
    # exponentially distributed inter-arrival times
    while not mx.all(mx.greater_equal(times, mx.array(1.0))):
        # Generate exponential random variables with rate lambda
        exp_samples = mx.divide(
            mx.negative(mx.log(mx.random.uniform(shape=shape))),
            lam_array
        )
        # Add to accumulated times
        new_times = mx.add(times, exp_samples)
        # Increment counts where we haven't exceeded 1.0 yet
        counts = mx.where(
            mx.less(new_times, mx.array(1.0)),
            mx.add(counts, mx.array(1)),
            counts
        )
        times = new_times

    if dtype is not None and dtype != mx.int32:
        counts = counts.astype(dtype)

    return counts


def random_permutation(x: Union[int, Any]) -> mx.array:
    """
    Generate a random permutation of integers from 0 to x-1 if x is an integer,
    or randomly permute the elements of x along the first axis if x is an array.

    Args:
        x: If an integer, randomly permute integers from 0 to x-1.
           If an array, randomly permute along the first axis.

    Returns:
        MLX array with random permutation
    """
    if isinstance(x, int):
        # If x is an integer, generate a permutation of integers from 0 to x-1
        return mx.random.permutation(x)
    else:
        # If x is an array, permute along the first axis
        x_tensor = convert_to_tensor(x)
        return mx.random.permutation(x_tensor)


def random_categorical(logits: Any, num_samples: int,
                       dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Draw samples from a categorical distribution.

    Args:
        logits: 2D tensor with unnormalized log probabilities
        num_samples: Number of samples to draw
        dtype: Optional data type
        device: Optional device

    Returns:
        MLX array with random categorical values
    """
    # Convert to MLX array if needed
    logits_tensor = convert_to_tensor(logits)

    # MLX's categorical function takes axis and shape parameters
    result = mx.random.categorical(logits=logits_tensor, num_samples=num_samples)

    if dtype is not None:
        result = result.astype(dtype)

    return result


def shuffle(x: Any) -> mx.array:
    """
    Randomly shuffle an MLX array along the first dimension.

    Args:
        x: Input array

    Returns:
        Shuffled MLX array
    """
    x_tensor = convert_to_tensor(x)

    # Get the shape of the tensor
    shape = x_tensor.shape

    # If the tensor is empty or has only one element, return it as is
    if shape[0] <= 1:
        return x_tensor

    # Generate random indices
    indices = mx.random.permutation(shape[0])

    # Gather along the first dimension
    return mx.take(x_tensor, indices, axis=0)


# Device operations
def get_device(x: ArrayLike) -> str:
    """
    Get the device of an MLX array.

    Args:
        x: Input array

    Returns:
        Device of the array
    """
    # MLX doesn't have explicit device tracking like PyTorch
    # It automatically uses the best available device
    if mx.metal.is_available():
        return 'gpu'
    return 'cpu'


def get_default_device() -> str:
    """
    Get the default device for MLX operations.

    Returns:
        Default device
    """
    if mx.metal.is_available():
        return 'gpu'
    return 'cpu'


def set_default_device(device: str) -> None:
    """
    Set the default device for MLX operations.

    Args:
        device: Default device
    """
    # MLX doesn't support setting a default device
    pass


def get_available_devices() -> List[str]:
    """
    Get a list of available devices.

    Returns:
        List of available devices
    """
    devices = ['cpu']
    if mx.metal.is_available():
        devices.append('gpu')
    return devices


def is_available(device: str) -> bool:
    """
    Check if the specified device is available.

    Args:
        device: Device to check

    Returns:
        True if the device is available, False otherwise
    """
    if device == 'cpu':
        return True
    elif device == 'gpu':
        return mx.metal.is_available()
    return False


def to_device(x: ArrayLike, device: str) -> mx.array:
    """
    Move an MLX array to the specified device.

    Args:
        x: Input array
        device: Target device

    Returns:
        MLX array on the target device
    """
    # MLX handles device placement automatically
    return convert_to_tensor(x)


def synchronize(device: Optional[str] = None) -> None:
    """
    Synchronize the specified device.

    Args:
        device: Target device
    """
    # MLX handles synchronization automatically
    pass


def memory_info(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory information for the specified device.

    Args:
        device: Target device

    Returns:
        Dictionary with memory information
    """
    # MLX doesn't provide memory information
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}


def memory_usage(device: Optional[str] = None) -> Dict[str, int]:
    """
    Get memory usage information for the specified device.

    Args:
        device: Target device

    Returns:
        Dictionary with memory usage information
    """
    # MLX doesn't provide memory usage information
    return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}


# Solver operations
def solve(a: Any, b: Any) -> Any:
    """
    Solve a linear system of equations Ax = b for x using MLX backend.

    Args:
        a: Coefficient matrix A
        b: Right-hand side vector or matrix b

    Returns:
        Solution to the system of equations

    Notes:
        Uses MLX's linalg.solve which requires a to be square and of full-rank.
    """
    # Convert inputs to MLX arrays
    a_array = convert_to_tensor(a)
    b_array = convert_to_tensor(b)

    # Use MLX's linalg.solve
    try:
        return mx.linalg.solve(a_array, b_array)
    except (AttributeError, ValueError):
        # Fallback to custom implementation if mx.linalg.solve is not available
        # or if there's a value error
        raise NotImplementedError("MLX backend does not support solve operation")


# Set power function
power = pow

# Class implementations for ops module
class MLXMathOps:
    """MLX implementation of math operations."""
    
    @property
    def pi(self):
        """Return the mathematical constant pi."""
        return mx.array(math.pi)
    
    def add(self, x, y):
        """Add two arrays element-wise."""
        return add(x, y)
    
    def subtract(self, x, y):
        """Subtract two arrays element-wise."""
        return subtract(x, y)
    
    def multiply(self, x, y):
        """Multiply two arrays element-wise."""
        return multiply(x, y)
    
    def divide(self, x, y):
        """Divide two arrays element-wise."""
        return divide(x, y)
    
    def dot(self, x, y):
        """Compute the dot product of two arrays."""
        return dot(x, y)
    
    def matmul(self, x, y):
        """Compute the matrix product of two arrays."""
        return matmul(x, y)
    
    def mean(self, x, axis=None, keepdims=False):
        """Compute the mean of an array along specified axes."""
        return mean(x, axis=axis, keepdims=keepdims)
    
    def sum(self, x, axis=None, keepdims=False):
        """Compute the sum of an array along specified axes."""
        return sum(x, axis=axis, keepdims=keepdims)
    
    def max(self, x, axis=None, keepdims=False):
        """Compute the maximum of an array along specified axes."""
        return max(x, axis=axis, keepdims=keepdims)
    
    def min(self, x, axis=None, keepdims=False):
        """Compute the minimum of an array along specified axes."""
        return min(x, axis=axis, keepdims=keepdims)
    
    def exp(self, x):
        """Compute the exponential of an array."""
        return exp(x)
    
    def log(self, x):
        """Compute the natural logarithm of an array."""
        return log(x)
    
    def log10(self, x):
        """Compute the base-10 logarithm of an array."""
        return log10(x)
    
    def log2(self, x):
        """Compute the base-2 logarithm of an array."""
        return log2(x)
    
    def pow(self, x, y):
        """Compute x raised to the power of y."""
        return pow(x, y)
    
    def sqrt(self, x):
        """Compute the square root of an array."""
        return sqrt(x)
    
    def square(self, x):
        """Compute the square of an array."""
        return square(x)
    
    def abs(self, x):
        """Compute the absolute value of an array."""
        return abs(x)
    
    def sign(self, x):
        """Compute the sign of an array."""
        return sign(x)
    
    def sin(self, x):
        """Compute the sine of an array."""
        return sin(x)
    
    def cos(self, x):
        """Compute the cosine of an array."""
        return cos(x)
    
    def tan(self, x):
        """Compute the tangent of an array."""
        return tan(x)
    
    def sinh(self, x):
        """Compute the hyperbolic sine of an array."""
        return sinh(x)
    
    def cosh(self, x):
        """Compute the hyperbolic cosine of an array."""
        return cosh(x)
    
    def tanh(self, x):
        """Compute the hyperbolic tangent of an array."""
        return tanh(x)
    
    def sigmoid(self, x):
        """Compute the sigmoid of an array."""
        return sigmoid(x)
    
    def relu(self, x):
        """Compute the rectified linear unit of an array."""
        return relu(x)
    
    def softmax(self, x, axis=-1):
        """Compute the softmax of an array."""
        return softmax(x, axis=axis)
    
    def clip(self, x, min_val=None, max_val=None):
        """Clip the values of an array."""
        return clip(x, min_val=min_val, max_val=max_val)
    
    def var(self, x, axis=None, keepdims=False):
        """Compute the variance of an array."""
        # This function might not be directly available in the module
        # So we'll use the global var function
        return var(x, axis=axis, keepdims=keepdims)


class MLXTensorOps:
    """MLX implementation of tensor operations."""
    
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


class MLXRandomOps:
    """MLX implementation of random operations."""
    
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, device=None):
        """Generate random values from a normal distribution."""
        return random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, device=device)
    
    def random_uniform(self, shape, minval=0.0, maxval=1.0, dtype=None, device=None):
        """Generate random values from a uniform distribution."""
        return random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, device=device)
    
    def random_binomial(self, shape, p=0.5, dtype=None, device=None):
        """Generate random values from a binomial distribution."""
        return random_binomial(shape, p=p, dtype=dtype, device=device)
    
    def random_permutation(self, x):
        """Generate a random permutation."""
        return random_permutation(x)
    
    def set_random_seed(self, seed):
        """Set the random seed for reproducibility."""
        return set_seed(seed)
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        return set_seed(seed)


class MLXComparisonOps:
    """MLX implementation of comparison operations."""
    
    def equal(self, x, y):
        """Check if two tensors are equal element-wise."""
        return equal(x, y)
    
    def not_equal(self, x, y):
        """Check if two tensors are not equal element-wise."""
        return not_equal(x, y)
    
    def less(self, x, y):
        """Check if one tensor is less than another element-wise."""
        return less(x, y)
    
    def less_equal(self, x, y):
        """Check if one tensor is less than or equal to another element-wise."""
        return less_equal(x, y)
    
    def greater(self, x, y):
        """Check if one tensor is greater than another element-wise."""
        return greater(x, y)
    
    def greater_equal(self, x, y):
        """Check if one tensor is greater than or equal to another element-wise."""
        return greater_equal(x, y)
    
    def logical_and(self, x, y):
        """Compute the logical AND of two tensors element-wise."""
        return logical_and(x, y)
    
    def logical_or(self, x, y):
        """Compute the logical OR of two tensors element-wise."""
        return logical_or(x, y)
    
    def logical_not(self, x):
        """Compute the logical NOT of a tensor element-wise."""
        return logical_not(x)
    
    def logical_xor(self, x, y):
        """Compute the logical XOR of two tensors element-wise."""
        return logical_xor(x, y)


class MLXDeviceOps:
    """MLX implementation of device operations."""
    
    def to_device(self, x, device):
        """Move a tensor to the specified device."""
        return to_device(x, device)
    
    def get_device(self, x):
        """Get the device of a tensor."""
        return get_device(x)
    
    def get_available_devices(self):
        """Get a list of available devices."""
        return get_available_devices()
    
    def memory_usage(self, device=None):
        """Get memory usage information for the specified device."""
        return memory_usage(device)


class MLXDTypeOps:
    """MLX implementation of data type operations."""
    
    def get_dtype(self, name):
        """Get a data type by name."""
        return get_dtype(name)
    
    def to_numpy_dtype(self, dtype):
        """Convert an MLX data type to a NumPy data type."""
        return to_numpy_dtype(dtype)
    
    def from_numpy_dtype(self, dtype):
        """Convert a NumPy data type to an MLX data type."""
        return from_numpy_dtype(dtype)


class MLXSolverOps:
    """MLX implementation of solver operations."""
    
    def solve(self, a, b):
        """Solve a linear system of equations."""
        return solve(a, b)
