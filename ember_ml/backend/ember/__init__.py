"""
Ember backend for ember_ml.

This module provides Ember tensor implementations of tensor operations.
"""

# Import tensor operations
from ember_ml.backend.ember.tensor_ops import (
    EmberBackendTensorOps,
    convert_to_tensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    eye,
    reshape,
    transpose,
    expand_dims,
    concatenate,
    stack,
    split,
    squeeze,
    tile,
    gather,
    tensor_scatter_nd_update,
    shape,
    dtype,
    cast,
    copy,
    to_numpy,
    full,
    full_like,
    linspace,
    arange,
    item,
    slice,
    slice_update,
    sort,
    pad,
    var,
    argsort
)


# Import math operations
from ember_ml.backend.ember.math_ops import (
    EmberBackendMathOps,
    add,
    subtract,
    multiply,
    divide,
    pi
)


# Import dtype operations
from ember_ml.backend.ember.dtype_ops import (
    EmberBackendDTypeOps,
    get_dtype,
    to_dtype_str,
    from_dtype_str,
    float32,
    float64,
    float16,
    int32,
    int64,
    int16,
    int8,
    uint8,
    uint16,
    uint32,
    uint64,
    bool_
)

# Import configuration
from ember_ml.backend.ember.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    _current_seed
)