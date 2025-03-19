"""PyTorch tensor casting operations."""

import torch
from typing import Any, Optional

from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.tensor.ops.utility import convert_to_tensor

# Type aliases
TensorLike = Any
DType = Any

def cast(data: TensorLike, dtype: DType) -> torch.Tensor:
    """
    Cast a tensor to a different data type.
    
    Args:
        data: Input tensor
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    tensor = convert_to_tensor(data)
    
    torch_dtype = TorchDType().from_dtype_str(dtype)
    return tensor.to(torch_dtype)