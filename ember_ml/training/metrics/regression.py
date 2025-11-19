"""
Regression metrics for the ember_ml library.

This module provides metrics utilities for regression tasks.
"""

from typing import Dict
from ember_ml import tensor, stats
from ember_ml.types import TensorLike
from ember_ml import ops

try:  # Optional dependency
    from sklearn.metrics import r2_score
except ModuleNotFoundError:  # pragma: no cover - fallback path
    r2_score = None

def regression_metrics(y_true: TensorLike, y_pred: TensorLike) -> Dict[str, TensorLike]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse_value = ops.mse(y_true, y_pred)
    rmse_value = ops.sqrt(mse_value)
    mae_value = ops.mean_absolute_error(y_true, y_pred)

    if r2_score is None:
        y_true_tensor = tensor.convert_to_tensor(y_true)
        y_pred_tensor = tensor.convert_to_tensor(y_pred)
        residual = ops.subtract(y_true_tensor, y_pred_tensor)
        ss_res = stats.sum(ops.multiply(residual, residual))
        mean_true = stats.mean(y_true_tensor)
        diff = ops.subtract(y_true_tensor, mean_true)
        ss_tot = stats.sum(ops.multiply(diff, diff))
        r2_value = ops.where(
            ops.greater(ss_tot, tensor.convert_to_tensor(0.0)),
            ops.subtract(
                tensor.convert_to_tensor(1.0),
                ops.divide(ss_res, ss_tot)
            ),
            tensor.convert_to_tensor(0.0)
        )
    else:
        r2_value = r2_score(y_true, y_pred)

    return {
        'mse': mse_value,
        'rmse': rmse_value,
        'mae': mae_value,
        'r2': r2_value
    }
