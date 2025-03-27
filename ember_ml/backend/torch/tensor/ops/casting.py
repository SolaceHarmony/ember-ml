"""PyTorch tensor casting operations."""

import torch
from typing import Any, Optional
from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.types import DType, TensorLike

def _validate_dtype(dtype_cls: TorchDType, dtype: DType) -> Optional[Any]:
    """
    Validate and convert dtype to Torch format.
    
    Args:
        dtype_cls: TorchDType instance for conversions
        dtype: Input dtype to validate
        
    Returns:
        Validated Torch dtype or None
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
    if isinstance(dtype, torch.dtype) or dtype in [torch.float32, torch.float64, torch.int32, torch.int64,
                                                  torch.bool, torch.int8, torch.int16, torch.uint8,
                                                  torch.float16]:
        return dtype
        
    raise ValueError(f"Invalid dtype: {dtype}")

def cast(tensor: TensorLike, dtype: DType) -> torch.Tensor:
    """
    Cast a tensor to a new data type.
    
    Args:
        tensor: Input tensor
        dtype: Target data type
        
    Returns:
        Tensor with new data type
    """
    # Import TorchTensor lazily to avoid circular import
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    tensor_obj = TorchTensor()
    
    # Get the tensor array from the tensor object
    tensor_array = tensor_obj.convert_to_tensor(tensor)
    
    # Validate the dtype
    torch_dtype = _validate_dtype(TorchDType(), dtype)
    
    # If mlx_dtype is None, return the tensor as is
    if torch_dtype is None:
        return tensor_array
        
    # Cast the tensor to the new dtype
    return tensor.to(torch_dtype)