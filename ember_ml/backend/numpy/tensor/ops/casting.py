"""NumPy tensor casting operations."""

import numpy as np
from typing import Any, Optional
from ember_ml.backend.numpy.tensor.dtype import NumpyDType
from ember_ml.backend.numpy.types import DType, TensorLike

def _validate_dtype(dtype_cls: NumpyDType, dtype: DType) -> Optional[Any]:
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
        
    # If it's already a NumPy dtype, return as is
    if isinstance(dtype, np.dtype) or dtype in [np.float32, np.float64, np.int32, np.int64,
                                              np.bool_, np.int8, np.int16, np.uint8,
                                              np.uint16, np.uint32, np.uint64, np.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def cast(tensor: TensorLike, dtype: DType) -> np.ndarray:
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Import NumpyTensor lazily to avoid circular import
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    tensor_obj = NumpyTensor()
    
    # Get the tensor array from the tensor object
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
    # Validate the dtype
    numpy_dtype = _validate_dtype(NumpyDType(), dtype)
    
    # If numpy_dtype is None, return the tensor as is
    if numpy_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor_array.astype(numpy_dtype)