"""NumPy tensor casting operations."""

import numpy as np
from typing import Any

from ember_ml.backend.numpy.tensor.dtype import NumpyDType

def cast(tensor_obj, tensor, dtype):
    """
    Cast a tensor to a different data type.
    
    Args:
        tensor_obj: NumpyTensor instance
        tensor: Input tensor
        dtype: The target data type
        
    Returns:
        Cast tensor
    """
    if not isinstance(tensor, np.ndarray):
        tensor = tensor_obj.convert_to_tensor(tensor)
    
    numpy_dtype = NumpyDType().from_dtype_str(dtype)
    return tensor.astype(numpy_dtype)