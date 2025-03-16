"""PyTorch tensor casting operations."""

import torch
from typing import Any

from ember_ml.backend.torch.tensor.dtype import TorchDType

def cast(tensor_obj, tensor, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        tensor_obj: TorchTensor instance
        tensor: Input tensor
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    torch_dtype = TorchDType().from_dtype_str(dtype)
    return tensor.to(torch_dtype)