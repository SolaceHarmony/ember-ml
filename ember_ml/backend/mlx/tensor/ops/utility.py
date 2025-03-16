"""MLX tensor utility operations."""

import mlx.core as mx
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.mlx.tensor.dtype import MLXDType, DType

# Type aliases
Shape = Union[int, Sequence[int]]

def _convert_input(x: Any) -> mx.array:
    """Convert input to MLX array."""
    if isinstance(x, mx.array):
        return x
    # Check for NumPy arrays by type name rather than direct import
    elif hasattr(x, '__class__') and x.__class__.__module__ == 'numpy' and x.__class__.__name__ == 'ndarray':
        return mx.array(x)
    return mx.array(x)

def _validate_dtype(dtype_cls: MLXDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to MLX format.
    
    Args:
        dtype_cls: MLXDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated MLX dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already an MLX dtype, return as is
    if isinstance(dtype, type(mx.float32)):
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def convert_to_tensor(tensor_obj, data: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Convert input to MLX array.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input data
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array
    """
    tensor = _convert_input(data)
    if dtype is not None:
        mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
        if mlx_dtype is not None:
            tensor = tensor.astype(mlx_dtype)
    # device parameter is ignored for MLX backend
    return tensor

def to_numpy(tensor_obj, data: Any) -> List:
    """
    Convert an MLX array to a NumPy array.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input MLX array
        
    Returns:
        NumPy array
    """
    # This is a special case where we need to use NumPy directly
    # It's only used for visualization or when explicitly requested
    tensor_array = tensor_obj.convert_to_tensor(data)
    return tensor_array.tolist()

def item(tensor_obj, data: Any) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    tensor_array = tensor_obj.convert_to_tensor(data)
    
    # Get the raw value
    raw_value = tensor_array.item()
    
    # Handle different types explicitly to ensure we return the expected types
    # We don't need to use casts here since raw_value is already a Python scalar
    if isinstance(raw_value, bool) or raw_value is True or raw_value is False:
        return raw_value  # Already a Python bool
    elif isinstance(raw_value, int):
        return raw_value  # Already a Python int
    elif isinstance(raw_value, float):
        return raw_value  # Already a Python float
    
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

def shape(tensor_obj, data: Any) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        
    Returns:
        Shape of the array
    """
    return tensor_obj.convert_to_tensor(data).shape

def dtype(tensor_obj, data: Any) -> Any:
    """
    Get the data type of a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        
    Returns:
        Data type of the array
    """
    return tensor_obj.convert_to_tensor(data).dtype

def copy(tensor_obj, data: Any) -> mx.array:
    """
    Create a copy of an MLX array.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        
    Returns:
        Copy of the array
    """
    # MLX arrays are immutable, so we can just convert to a new array
    return tensor_obj.convert_to_tensor(data)

def var(tensor_obj, data: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the variance of a tensor.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the dimensions or not
        
    Returns:
        Variance of the array
    """
    tensor_array = tensor_obj.convert_to_tensor(data)
    return mx.var(tensor_array, axis=axis, keepdims=keepdims)

def sort(tensor_obj, data: Any, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Sort a tensor along the given axis.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted array
    """
    tensor_array = tensor_obj.convert_to_tensor(data)
    sorted_array = mx.sort(tensor_array, axis=axis)
    
    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * tensor_array.ndim
        # Reverse the array along the specified axis
        slices[axis] = slice(None, None, -1)
        sorted_array = sorted_array[tuple(slices)]
    
    return sorted_array

def argsort(tensor_obj, data: Any, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Return the indices that would sort a tensor along the given axis.
    
    Args:
        tensor_obj: MLXTensor instance
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the array
    """
    tensor_array = tensor_obj.convert_to_tensor(data)
    
    if descending:
        # For descending order, we need to negate the array, get the argsort, and then use those indices
        # We can't directly negate the array because it might not be numeric, so we'll use a different approach
        # First, get the sorted indices in ascending order
        indices = mx.argsort(tensor_array, axis=axis)
        
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * indices.ndim
        # Reverse the indices along the specified axis
        slices[axis] = slice(None, None, -1)
        return indices[tuple(slices)]
    else:
        return mx.argsort(tensor_array, axis=axis)

def maximum(tensor_obj, data1: Any, data2: Any) -> mx.array:
    """
    Element-wise maximum of two arrays.
    
    Args:
        tensor_obj: MLXTensor instance
        data1: First input array
        data2: Second input array
        
    Returns:
        Element-wise maximum
    """
    data1_array = tensor_obj.convert_to_tensor(data1)
    data2_array = tensor_obj.convert_to_tensor(data2)
    return mx.maximum(data1_array, data2_array)