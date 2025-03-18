"""
MLX backend configuration for ember_ml.

This module provides configuration settings for the MLX backend.
"""

import mlx.core as mx
from typing import Union, Optional, Sequence, Any, List, Tuple, TypeVar, TYPE_CHECKING
from ember_ml.backend.mlx.tensor.dtype import MLXDType

# Default device for MLX operations
DEFAULT_DEVICE = mx.default_device().type

# Default data type for MLX operations
DEFAULT_DTYPE = MLXDType().float32

"""Type definitions for tensor operations.

This module provides standard type aliases for tensor operations in the Ember ML framework.
These type aliases ensure consistent type annotations across the codebase and
help with static type checking.
"""

# Conditionally import backend types for type checking only
if TYPE_CHECKING:
    # These imports are only used for type checking and not at runtime
    import numpy
    import mlx.core
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.nn.tensor.common.dtypes import EmberDType
    from ember_ml.backend.mlx.tensor.tensor import MLXTensor

AllTensorLike = Optional[Union[
    'numpy.ndarray' if TYPE_CHECKING else Any,
    'mlx.core.array' if TYPE_CHECKING else Any,
    'MLXTensor' if TYPE_CHECKING else Any,
    'EmberTensor' if TYPE_CHECKING else Any,
]]

AllDType = Optional[Union[
    'mlx.core.Dtype' if TYPE_CHECKING else Any,
    'numpy.dtype' if TYPE_CHECKING else Any,
    'EmberDType' if TYPE_CHECKING else Any,
    'MLXDType' if TYPE_CHECKING else Any,
]]
# Standard type aliases for general tensor-like inputs
# This covers all possible input types that can be converted to a tensor
TensorLike = Optional[Union[
    int, float, bool, list, tuple, 
    AllTensorLike
]]

DType = Optional[Union[str, 
    AllDType,
    Any]]  # Any covers backend-specific dtype objects
Scalar = Union[int, float]  # 0D tensors

# Dimension-specific tensor types
ScalarLike = Union[int,float,bool, 
    AllTensorLike,
    Any
    ]  # 0D tensors

VectorLike = Union[List[Union[int, float, bool]],Tuple[Union[int, float, bool], ...], 
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

# Dtype definitions
Device = Optional[str]

# Type variable for generic functions
T = TypeVar('T')