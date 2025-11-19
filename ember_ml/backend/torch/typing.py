"""
Type definitions for tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

from typing import Union, Optional, Sequence, Any, List, Tuple, TypeVar, TYPE_CHECKING
import torch
from ember_ml.backend.torch.tensor.dtype import TorchDType

# Conditionally import types for type checking only
if TYPE_CHECKING:
    from torch import Tensor
    from ember_ml import tensor
    from ember_ml.nn.tensor.common.dtypes import EmberDType
    from ember_ml.backend.torch.tensor.tensor import TorchTensor

AllTensorLike = Optional[Union[
    'Tensor' if TYPE_CHECKING else Any,
    'TorchTensor' if TYPE_CHECKING else Any,
    'EmberTensor' if TYPE_CHECKING else Any,
]]

AllDType = Optional[Union[
    'torch.dtype' if TYPE_CHECKING else Any,
    'EmberDType' if TYPE_CHECKING else Any,
    'TorchDType' if TYPE_CHECKING else Any,
]]

# Standard type aliases for general tensor-like inputs
TensorLike = Optional[Union[
    int, float, bool, list, tuple,
    AllTensorLike
]]

DType = Optional[Union[str,
    AllDType,
    Any]]  # Any covers backend-specific dtype objects

Scalar = Union[int, float]  # 0D tensors

# Dimension-specific tensor types
ScalarLike = Union[int, float, bool,
    AllTensorLike,
    Any
]  # 0D tensors

VectorLike = Union[List[Union[int, float, bool]], Tuple[Union[int, float, bool], ...],
    AllTensorLike,
    Any
]  # 1D tensors

MatrixLike = Union[List[List[Union[int, float, bool]]],
    AllTensorLike,
    Any
]  # 2D tensors

# Shape definitions
Shape = Sequence[int]
ShapeLike = Union[int, Shape]

# Device definitions
Device = Optional[str]

# Type variable for generic functions
T = TypeVar('T')
