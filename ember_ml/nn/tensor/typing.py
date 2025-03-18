"""Type definitions for tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import Union, Optional, Sequence, Any, List, Tuple, TypeVar, TYPE_CHECKING

# Import EmberTensor for type annotations
from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
from ember_ml.nn.tensor.common.dtypes import EmberDType

# Conditionally import backend types for type checking only
if TYPE_CHECKING:
    # These imports are only used for type checking and not at runtime
    import numpy as np
    import torch
    import mlx.core
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    from ember_ml.backend.torch.tensor.tensor import TorchTensor
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor

# Standard type aliases for general tensor-like inputs
# This covers all possible input types that can be converted to a tensor
TensorLike = Optional[Union[
    int, float, bool, list, tuple, EmberTensor,
    'np.ndarray' if TYPE_CHECKING else Any,
    'torch.Tensor' if TYPE_CHECKING else Any,
    'mlx.core.array' if TYPE_CHECKING else Any,
    'NumpyTensor' if TYPE_CHECKING else Any,
    'TorchTensor' if TYPE_CHECKING else Any,
    'MLXTensor' if TYPE_CHECKING else Any
]]

# Dimension-specific tensor types
Scalar = Union[int, float, bool, EmberTensor]  # 0D tensors
Vector = Union[List[Union[int, float, bool]], Tuple[Union[int, float, bool], ...], EmberTensor]  # 1D tensors
Matrix = Union[List[List[Union[int, float, bool]]], EmberTensor]  # 2D tensors

# Shape definitions
Shape = Union[int, Sequence[int]]
ShapeLike = Union[int, Tuple[int, ...], List[int]]

# Dtype definitions
DType = Optional[Union[str, EmberDType, Any]]  # Any covers backend-specific dtype objects
Device = Optional[str]

# Type variable for generic functions
T = TypeVar('T')