"""MLX tensor utility operations."""

from typing import Any, Optional, Sequence, Tuple, Union

import mlx.core as mx

from ember_ml.backend.mlx.tensor.dtype import MLXDType
from ember_ml.backend.mlx.types import TensorLike, DType

DTypeHandler = MLXDType()

def _convert_input(x: TensorLike) -> Any:
    """
    Convert input to MLX array.
    
    Handles various input types:
    - MLX arrays (returned as-is)
    - NumPy arrays (converted to MLX arrays)
    - MLXTensor objects (extract underlying data)
    - EmberTensor objects (extract underlying data)
    - Python scalars (int, float, bool)
    - Python sequences (list, tuple)
       
    Special handling for:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors
    
    Args:
        x: Input data to convert
        
    Returns:
        MLX array
        
    Raises:
        ValueError: If the input cannot be converted to an MLX array
    """
    # Already an MLX array - check by type and module
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__module__') and
        x.__class__.__module__ == 'mlx.core' and
        x.__class__.__name__ == 'array'):
        return x
        
    # Handle MLXTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'MLXTensor'):
        return x._tensor

    # Handle EmberTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'EmberTensor'):
        if hasattr(x, '_tensor'):
            from ember_ml.nn.tensor.common.dtypes import EmberDType
            if isinstance(x._dtype, EmberDType):
                dtype_from_ember = x._dtype._backend_dtype
                if dtype_from_ember is not None:
                    x._tensor = x._tensor.astype(dtype_from_ember)
            return x._tensor
        else:
          raise ValueError(f"EmberTensor does not have a '_tensor' attribute: {x}")

    # Check for NumPy arrays by type name rather than direct import
    if (hasattr(x, '__class__') and
        x.__class__.__module__ == 'numpy' and
        x.__class__.__name__ == 'ndarray'):
        return mx.array(x)
        
    # Handle Python scalars (0D tensors)
    if isinstance(x, (int, float, bool)):
        try:
            return mx.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert scalar {x} to MLX array: {e}")
    
    # Handle Python sequences (potential 1D or higher tensors) recursively
    if isinstance(x, (list, tuple)):
       try:
            # Check if it's a nested sequence (2D or higher)
            if x and isinstance(x[0], (list, tuple)):
                # Handle potential jagged arrays by ensuring consistent dimensions
                shapes = [len(item) for item in x if isinstance(item, (list, tuple))]
                if len(set(shapes)) > 1:
                    # Jagged array - warn but proceed
                    import warnings
                    warnings.warn(f"Converting jagged array with inconsistent shapes: {shapes}")
            return mx.array([_convert_input(item) for item in x])
       except Exception as e:
            raise ValueError(f"Cannot convert sequence {type(x)} to MLX array: {e}")

    # Handle MLX scalar types
    if mx.isscalar(x):
        return mx.array(x)

    # For any other type, reject it
    raise ValueError(f"Cannot convert {type(x)} to MLX array. Only int, float, bool, list, tuple, numpy.ndarray, MLXTensor, and EmberTensor are supported.")
def convert_to_mlx_tensor(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> mx.array:
    """
    Convert input to MLX array.
    
    Handles various input types with special attention to dimensionality:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors
    
    Args:
        data: Input data
        dtype: Optional data type
        device: Ignored for MLX backend
        
    Returns:
        MLX array
    """
    tensor = _convert_input(data)
    if dtype is not None:
        mlx_dtype = DTypeHandler.validate_dtype(dtype)
        if mlx_dtype is not None:
            tensor = tensor.astype(mlx_dtype)
    
    # Ensure proper dimensionality
    # If data is a scalar but we need a 0-dim tensor, reshape accordingly
    if isinstance(data, (int, float, bool)) and tensor.ndim > 0:
        tensor = mx.reshape(tensor, ())
        
    return tensor

def to_numpy(data: TensorLike) -> Any:
    """
    Convert an MLX array to a NumPy array.
    
    IMPORTANT: This function is provided ONLY for visualization/plotting libraries 
    that specifically require NumPy arrays. It should NOT be used for general tensor 
    conversions or operations. Ember ML has a zero backend design where EmberTensor 
    relies entirely on the selected backend for representation.
    
    Args:
        data: Input MLX array
        
    Returns:
        NumPy array
    """
    # This is a special case where we need to use NumPy directly
    # It's only used for visualization or when explicitly requested
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_data = Tensor.convert_to_tensor(data)
    import numpy as np
    return np.array(tensor_data)

def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    Args:
        data: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
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

def shape(data: TensorLike) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        data: Input array
        
    Returns:
        Shape of the array
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    return Tensor.convert_to_tensor(data).shape

def dtype(data: TensorLike) -> Any:
    """
    Get the data type of a tensor.
    
    Args:
        data: Input array
        
    Returns:
        Data type of the array
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    # Return the dtype property directly to maintain backward compatibility
    # with code that accesses the property directly
    return Tensor.convert_to_tensor(data).dtype

def copy(data: TensorLike) -> mx.array:
    """
    Create a copy of an MLX array.
    
    Args:
        data: Input array
        
    Returns:
        Copy of the array
    """
    # MLX arrays are immutable, so we can just convert to a new array
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    return Tensor.convert_to_tensor(data)

def var(data: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> mx.array:
    """
    Compute the variance of a tensor.
    
    Args:
        data: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the dimensions or not
        
    Returns:
        Variance of the array
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    return mx.var(tensor_array, axis=axis, keepdims=keepdims)

def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted array
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    sorted_array = mx.sort(tensor_array, axis=axis)
    
    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * tensor_array.ndim
        # Reverse the array along the specified axis
        slices[axis] = slice(None, None, -1)
        sorted_array = sorted_array[tuple(slices)]
    
    return sorted_array

def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> mx.array:
    """
    Return the indices that would sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the array
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
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

def maximum(data1: TensorLike, data2: TensorLike) -> mx.array:
    """
    Element-wise maximum of two arrays.
    
    Args:
        data1: First input array
        data2: Second input array
        
    Returns:
        Element-wise maximum
    """
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    Tensor = MLXTensor()
    data1_array = Tensor.convert_to_tensor(data1)
    data2_array = Tensor.convert_to_tensor(data2)
    return mx.maximum(data1_array, data2_array)