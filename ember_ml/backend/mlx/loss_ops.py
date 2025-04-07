"""MLX backend implementation for loss operations."""

from typing import Any, Optional, Union, Sequence
import mlx.core as mx
import numpy as np # For epsilon, potentially other constants

from ember_ml.backend.mlx.types import TensorLike

# Epsilon for numerical stability
EPSILON = 1e-7

# Helper function (module level)
def _reduce_loss(loss: mx.array,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 keepdims: bool = False) -> mx.array:
    """Helper to apply reduction (mean) to loss tensor."""
    # MLX's mean function handles None axis and keepdims directly
    return mx.mean(loss, axis=axis, keepdims=keepdims)

# --- Standalone Loss Functions ---

def mean_squared_error(y_true: TensorLike, y_pred: TensorLike,
                       axis: Optional[Union[int, Sequence[int]]] = None,
                       keepdims: bool = False) -> mx.array:
    """MLX implementation of mean squared error."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)
    squared_diff = mx.square(y_pred_arr - y_true_arr)
    return _reduce_loss(squared_diff, axis=axis, keepdims=keepdims)

def mean_absolute_error(y_true: TensorLike, y_pred: TensorLike,
                         axis: Optional[Union[int, Sequence[int]]] = None,
                         keepdims: bool = False) -> mx.array:
    """MLX implementation of mean absolute error."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)
    abs_diff = mx.abs(y_pred_arr - y_true_arr)
    return _reduce_loss(abs_diff, axis=axis, keepdims=keepdims)

def binary_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                        from_logits: bool = False,
                        axis: Optional[Union[int, Sequence[int]]] = None,
                        keepdims: bool = False) -> mx.array:
    """MLX implementation of binary crossentropy."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)

    if from_logits:
        # Stable implementation: max(logits, 0) - logits * y_true + log(1 + exp(-abs(logits)))
        max_val = mx.maximum(y_pred_arr, 0)
        log_exp_term = mx.log(1 + mx.exp(-mx.abs(y_pred_arr)))
        loss = max_val - y_pred_arr * y_true_arr + log_exp_term
    else:
        # Clip predictions for numerical stability
        y_pred_arr = mx.clip(y_pred_arr, EPSILON, 1.0 - EPSILON)
        loss = - (y_true_arr * mx.log(y_pred_arr) +
                  (1.0 - y_true_arr) * mx.log(1.0 - y_pred_arr))

    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                             from_logits: bool = False,
                             axis: Optional[Union[int, Sequence[int]]] = None,
                             keepdims: bool = False) -> mx.array:
    """MLX implementation of categorical crossentropy."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)

    if from_logits:
        log_probs = mx.log_softmax(y_pred_arr, axis=-1)
    else:
        y_pred_arr = mx.clip(y_pred_arr, EPSILON, 1.0 - EPSILON)
        log_probs = mx.log(y_pred_arr)

    cce = -mx.sum(y_true_arr * log_probs, axis=-1)
    return _reduce_loss(cce, axis=axis, keepdims=keepdims)

def sparse_categorical_crossentropy(y_true: TensorLike, y_pred: TensorLike,
                                    from_logits: bool = False,
                                    axis: Optional[Union[int, Sequence[int]]] = None,
                                    keepdims: bool = False) -> mx.array:
    """MLX implementation of sparse categorical crossentropy."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_int = convert_to_mlx_tensor(data=y_true).astype(mx.int32)
    y_pred_logits = convert_to_mlx_tensor(data=y_pred)

    if not from_logits:
         y_pred_logits = mx.clip(y_pred_logits, EPSILON, 1.0 - EPSILON)
         y_pred_logits = mx.log(y_pred_logits)
    
    probs = mx.softmax(y_pred_logits, axis=-1)
    log_probs = mx.log(probs)
    y_true_int_expanded = mx.expand_dims(y_true_int, axis=-1)
    neg_log_likelihood = -mx.take_along_axis(log_probs, y_true_int_expanded, axis=-1)
    loss = mx.squeeze(neg_log_likelihood, axis=-1)

    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def huber_loss(y_true: TensorLike, y_pred: TensorLike, delta: float = 1.0,
               axis: Optional[Union[int, Sequence[int]]] = None,
               keepdims: bool = False) -> mx.array:
    """MLX implementation of Huber loss."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)
    error = y_pred_arr - y_true_arr
    abs_error = mx.abs(error)
    quadratic = mx.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * mx.square(quadratic) + delta * linear
    return _reduce_loss(loss, axis=axis, keepdims=keepdims)

def log_cosh_loss(y_true: TensorLike, y_pred: TensorLike,
                  axis: Optional[Union[int, Sequence[int]]] = None,
                  keepdims: bool = False) -> mx.array:
    """MLX implementation of log-cosh loss."""
    from ember_ml.backend.mlx.tensor.ops.utility import convert_to_mlx_tensor # Lazy load functional
    y_true_arr = convert_to_mlx_tensor(data=y_true)
    y_pred_arr = convert_to_mlx_tensor(data=y_pred)
    error = y_pred_arr - y_true_arr
    logcosh = mx.logaddexp(error, -error) - mx.log(mx.array(2.0, dtype=error.dtype))
    return _reduce_loss(logcosh, axis=axis, keepdims=keepdims)

# Removed MLXLossOps class
