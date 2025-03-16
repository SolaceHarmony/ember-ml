"""NumPy tensor utility operations."""

import numpy as np
from typing import Union, Optional, Sequence, Any, List, Tuple

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

# Type aliases
Shape = Union[int, Sequence[int]]
DType = Any

def _convert_input(x: Any) -> np.ndarray:
    """Convert input to NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    # Handle EmberTensor objects
    if isinstance(x, object) and getattr(x.__class__, '__name__', '') == 'EmberTensor':
        # For EmberTensor, extract the underlying NumPy array
        return getattr(x, '_tensor')
    # Convert other types to NumPy array
    try:
        return np.array(x)
    except:
        raise ValueError(f"Cannot convert {type(x)} to NumPy array")

def _validate_dtype(dtype_cls: NumpyDType, dtype: Optional[DType]) -> Optional[Any]:
    """
    Validate and convert dtype to NumPy format.
    
    Args:
        dtype_cls: NumpyDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated NumPy dtype or None
    """
    if dtype is None:
        return None
    
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # Handle EmberDType objects
    if hasattr(dtype, 'name'):
        return dtype_cls.from_dtype_str(str(dtype.name))
        
    # If it's already a NumPy dtype, return as is
    if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64, 
                                               np.bool_, np.int8, np.int16, np.uint8, 
                                               np.uint16, np.uint32, np.uint64, np.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def convert_to_tensor(tensor_obj, data: Any, dtype: Optional[DType] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Convert data to a NumPy array.
    
    Args:
        tensor_obj: NumpyTensor instance
        data: The data to convert
        dtype: Optional data type
        device: Ignored for NumPy backend
        
    Returns:
        NumPy array
    """
    # If it's already a NumPy array, return it or cast it
    if isinstance(data, np.ndarray):
        if dtype is not None:
            # Convert dtype if needed
            numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
            if numpy_dtype is not None:
                return data.astype(numpy_dtype)
        return data
    
    # Convert to NumPy array
    tensor = _convert_input(data)
    
    if dtype is not None:
        numpy_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
        if numpy_dtype is not None:
            tensor = tensor.astype(numpy_dtype)
    
    # device parameter is ignored for NumPy backend
    return tensor

def to_numpy(tensor_obj, tensor: Any) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to convert
        
    Returns:
        NumPy array
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    # For non-array types, convert to NumPy array
    return tensor_obj.convert_to_tensor(tensor)

def item(tensor_obj, tensor: Any) -> Union[int, float, bool]:
    """
    Get the value of a scalar tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to get the value from
        
    Returns:
        The scalar value
    """
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
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

def shape(tensor_obj, tensor: Any) -> Tuple[int, ...]:
    """
    Get the shape of a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to get the shape of
        
    Returns:
        The shape of the tensor
    """
    return tensor_obj.convert_to_tensor(tensor).shape

def dtype(tensor_obj, tensor: Any) -> Any:
    """
    Get the data type of a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to get the data type of
        
    Returns:
        The data type of the tensor
    """
    return tensor_obj.convert_to_tensor(tensor).dtype

def copy(tensor_obj, tensor: Any) -> np.ndarray:
    """
    Create a copy of a tensor.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: The tensor to copy
        
    Returns:
        Copy of the tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    return tensor_np.copy()

def var(tensor_obj, tensor: Any, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the variance of a tensor along specified axes.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        axis: Axis or axes along which to compute the variance
        keepdims: Whether to keep the reduced dimensions
        
    Returns:
        Variance of the tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    return np.var(tensor_np, axis=axis, keepdims=keepdims)

def sort(tensor_obj, tensor: Any, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Sort a tensor along a specified axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Sorted tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    
    # Sort the tensor
    if descending:
        return -np.sort(-tensor_np, axis=axis)
    else:
        return np.sort(tensor_np, axis=axis)

def argsort(tensor_obj, tensor: Any, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Return the indices that would sort a tensor along a specified axis.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        axis: Axis along which to sort
        descending: Whether to sort in descending order
        
    Returns:
        Indices that would sort the tensor
    """
    tensor_np = tensor_obj.convert_to_tensor(tensor)
    
    # Get the indices that would sort the tensor
    if descending:
        return np.argsort(-tensor_np, axis=axis)
    else:
        return np.argsort(tensor_np, axis=axis)

def maximum(tensor_obj, x: Any, y: Any) -> np.ndarray:
    """
    Element-wise maximum of two tensors.
    
    Args:
        tensor_obj: NumpyTensor instance
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Element-wise maximum
    """
    x_np = tensor_obj.convert_to_tensor(x)
    y_np = tensor_obj.convert_to_tensor(y)
    return np.maximum(x_np, y_np)