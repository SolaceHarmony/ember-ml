"""
NumPy inverse linear algebra operations for ember_ml.

This module provides NumPy operations.
"""

import numpy as np

from ember_ml.backend.numpy.tensor import NumpyDType
# Import from tensor_ops
from ember_ml.backend.numpy.types import TensorLike

dtype_obj = NumpyDType()

def inv(a: TensorLike) -> np.ndarray:
    """
    Compute the inverse of a square matrix.
    
    Args:
        a: Input square matrix
        
    Returns:
        Inverse of the matrix
    """
    # Convert input to NumPy array
    from ember_ml.backend.numpy.tensor import NumpyTensor
    Tensor = NumpyTensor()
    a_array = Tensor.convert_to_tensor(a)
    
    # Use NumPy's built-in inv function
    return np.linalg.inv(a_array)