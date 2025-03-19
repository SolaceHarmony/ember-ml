"""NumPy tensor utility operations."""

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from ember_ml.backend.numpy.tensor.dtype import NumpyDType
from ember_ml.backend.numpy.types import TensorLike, DType

DTypeHandler = NumpyDType()

def _convert_input(x: TensorLike) -> Any:
    """
    Convert input to NumPy array.
    
    Handles various input types:
    - NumPy arrays (returned as-is)
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
        NumPy array
        
    Raises:
        ValueError: If the input cannot be converted to a NumPy array
    """
    # Already a NumPy array
    if isinstance(x, np.ndarray):
        return x
        
    # Handle EmberTensor objects
    if (hasattr(x, '__class__') and
        hasattr(x.__class__, '__name__') and
        x.__class__.__name__ == 'EmberTensor'):
        from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
        if isinstance(x, EmberTensor):
            # Extract the underlying tensor data
            if hasattr(x, '_tensor') and isinstance(x._tensor, np.ndarray):
                return np.array(x._tensor)
            else:
                ValueError(f"EmberTensor does not have a '_tensor' attribute: {x}")
        else:
            raise ValueError(f"Unknown type: {type(x)}")
    
    # Handle Python scalars (0D tensors)
    if isinstance(x, (int, float, bool)):
        try:
            return np.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert scalar {type(x)} to NumPy array: {e}")
    
    # Handle Python sequences (potential 1D or higher tensors)
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
            return np.array(x)
        except Exception as e:
            raise ValueError(f"Cannot convert sequence {type(x)} to NumPy array: {e}")
    
    # Handle NumPy scalar types
    if np.isscalar(x):
        return np.array(x)
        
    # For any other type, reject it
    raise ValueError(f"Cannot convert {type(x)} to NumPy array. Only int, float, bool, list, tuple, numpy scalar types, and numpy.ndarray are supported.")



def convert_to_numpy_tensor(data: TensorLike, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Convert input to NumPy array.
    
    Handles various input types with special attention to dimensionality:
    - 0D tensors (scalars)
    - 1D tensors (vectors)
    - 2D tensors (matrices)
    - Higher dimensional tensors
    
    Args:
        data: Input data
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array
    """
    tensor = _convert_input(data)
    if dtype is not None:
        numpy_dtype = DTypeHandler.validate_dtype(dtype)
        if numpy_dtype is not None:
            tensor = tensor.astype(numpy_dtype)
    
    # Ensure proper dimensionality
    # If data is a scalar but we need a 0-dim tensor, reshape accordingly
    if isinstance(data, (int, float, bool)) and tensor.ndim > 0:
        tensor = np.reshape(tensor, ())
        
    return tensor

def to_numpy(data: TensorLike) -> Any:
    """
    Convert a NumPy array to a NumPy array.
    
    IMPORTANT: This function is provided ONLY for visualization/plotting libraries
    that specifically require NumPy arrays. It should NOT be used for general tensor
    conversions or operations. Ember ML has a zero backend design where EmberTensor
    relies entirely on the selected backend for representation.
    
    Args:
        data: Input NumPy array
        
    Returns:
        NumPy array
    """
    # For NumPy, this is a no-op since we're already using NumPy arrays
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_data = Tensor.convert_to_tensor(data)
    return tensor_data

def item(data: TensorLike) -> Union[int, float, bool]:
    """
    Extract the scalar value from a tensor.
    
    Args:
        data: Input tensor containing a single element
        
    Returns:
        Standard Python scalar (int, float, or bool)
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    # Get the raw value
    raw_value = tensor_array.item()
    
    # Handle different types explicitly to ensure we return the expected types
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
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    return Tensor.convert_to_tensor(data).shape

def dtype(data: TensorLike) -> Any:
    """
    Get the data type of a tensor.
    
    Args:
        data: Input array
        
    Returns:
        Data type of the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    # Return the dtype property directly to maintain backward compatibility
    # with code that accesses the property directly
    return Tensor.convert_to_tensor(data).dtype

def copy(data: TensorLike) -> np.ndarray:
    """
    Create a copy of a NumPy array.
    
    Args:
        data: Input array
        
    Returns:
        Copy of the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_np = Tensor.convert_to_tensor(data)
    return tensor_np.copy()

def var(data: TensorLike, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the variance of a tensor.
    
    Args:
        data: Input array
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the dimensions or not
        
    Returns:
        Variance of the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    return np.var(tensor_array, axis=axis, keepdims=keepdims)

def sort(data: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    # Sort the tensor
    sorted_array = np.sort(tensor_array, axis=axis)
    
    if descending:
        # Create a list of slice objects for each dimension
        slices = [slice(None)] * tensor_array.ndim
        # Reverse the array along the specified axis
        slices[axis] = slice(None, None, -1)
        sorted_array = sorted_array[tuple(slices)]
    
    return sorted_array

def argsort(data: TensorLike, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Return the indices that would sort a tensor along the given axis.
    
    Args:
        data: Input array
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the array
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    tensor_array = Tensor.convert_to_tensor(data)
    
    if descending:
        # For descending order, we need to negate the array, get the argsort, and then use those indices
        indices = np.argsort(-tensor_array, axis=axis)
        return indices
    else:
        return np.argsort(tensor_array, axis=axis)

def maximum(data1: TensorLike, data2: TensorLike) -> np.ndarray:
    """
    Element-wise maximum of two arrays.
    
    Args:
        data1: First input array
        data2: Second input array
        
    Returns:
        Element-wise maximum
    """
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    Tensor = NumpyTensor()
    data1_array = Tensor.convert_to_tensor(data1)
    data2_array = Tensor.convert_to_tensor(data2)
    return np.maximum(data1_array, data2_array)

# Alias for backward compatibility
convert_to_tensor = convert_to_numpy_tensor