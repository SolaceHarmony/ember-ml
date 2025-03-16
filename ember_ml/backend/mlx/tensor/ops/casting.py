"""MLX tensor casting operations."""

import mlx.core as mx
from typing import Any, Optional

from ember_ml.backend.mlx.tensor.dtype import MLXDType, DType

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

def cast(tensor_obj, tensor, dtype):
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor_obj: MLXTensor instance
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Get the tensor array from the tensor object
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
    # Validate the dtype
    mlx_dtype = _validate_dtype(tensor_obj._dtype_cls, dtype)
    
    # If mlx_dtype is None, return the tensor as is
    if mlx_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor_array.astype(mlx_dtype)