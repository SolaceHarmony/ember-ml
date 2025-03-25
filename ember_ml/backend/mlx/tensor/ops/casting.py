"""MLX tensor casting operations."""

import mlx.core
from typing import Any, Optional
from ember_ml.backend.mlx.tensor.dtype import MLXDType
from ember_ml.backend.mlx.types import DType, TensorLike

def _validate_dtype(dtype_cls: MLXDType, dtype: DType) -> Optional[Any]:
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
    
    # EmberDType handling
    if (hasattr(dtype, '__class__') and
        hasattr(dtype.__class__, '__name__') and
        dtype.__class__.__name__ == 'EmberDType'):
        from ember_ml.nn.tensor.common.dtypes import EmberDType
        if isinstance(dtype, EmberDType):
            dtype_from_ember = dtype._backend_dtype
            if dtype_from_ember is not None:
                return dtype_from_ember
            
    # Handle string dtypes
    if isinstance(dtype, str):
        return dtype_cls.from_dtype_str(dtype)
        
    # If it's already an MLX dtype, return as is
    if isinstance(dtype, mlx.core.Dtype) or dtype in [mlx.core.float32, mlx.core.float64, mlx.core.int32, mlx.core.int64,
                                                  mlx.core.bool_, mlx.core.int8, mlx.core.int16, mlx.core.uint8,
                                                  mlx.core.uint16, mlx.core.uint32, mlx.core.uint64, mlx.core.float16]:
        return dtype
    
    raise ValueError(f"Invalid dtype: {dtype}")

def cast(tensor: TensorLike, dtype: DType) -> mlx.core.array:
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Import MLXTensor lazily to avoid circular import
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor
    tensor_obj = MLXTensor()
    
    # Get the tensor array from the tensor object
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
    # Validate the dtype
    mlx_dtype = _validate_dtype(MLXDType(), dtype)
    
    # If mlx_dtype is None, return the tensor as is
    if mlx_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor_array.astype(mlx_dtype)