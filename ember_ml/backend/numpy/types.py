"""NumPy type definitions for ember_ml."""

from typing import Any, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

# Basic type aliases
type Numeric = Union[int, float]

# Type definitions for NumPy dtypes
type DTypeStr = str
type DTypeClass = Union[np.dtype, str, None]

# Type alias for dtype arguments that maintains compatibility
# with both NumPy's dtype system and tensor.py's DType
type DType = Any  # Using Any for maximum compatibility

# NumPy array types
type NumpyArray = np.ndarray
type ArrayLike = Union[NumpyArray, Numeric, List[Any], Tuple[Any, ...]]

type TensorTypes = Any
# Conditional imports
if TYPE_CHECKING:
    import numpy as np
    from ember_ml.nn.tensor.common.ember_tensor import EmberTensor
    from ember_ml.backend.numpy.tensor.tensor import NumpyTensor
    
    type TensorTypes = Union[
        np.ndarray,
        NumpyTensor,
        EmberTensor
    ]

# Main type definitions
type TensorLike = Optional[Union[
    Numeric,
    bool,
    List[Any],
    Tuple[Any, ...],
    TensorTypes
]]

# Shape types
type Shape = Sequence[int]
type ShapeType = Union[int, Tuple[int, ...], List[int]]
type ShapeLike = Union[int, List[int], Tuple[int, ...], Shape]

# Dimension types
type DimSize = Union[int, NumpyArray]
type Axis = Optional[Union[int, Sequence[int]]]

# Scalar types
type ScalarLike = Optional[Union[
    Numeric,
    bool,
    NumpyArray,
    TensorTypes
]]

# Device type
type Device = Optional[str]

# Index types
type IndexType = Union[int, Sequence[int], NumpyArray]
type Indices = Union[Sequence[int], NumpyArray]