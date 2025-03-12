"""
Interfaces for operations.

This module defines the abstract interfaces for operations that abstract
machine learning library tensor and scalar operations.
"""

from ember_ml.ops.interfaces.tensor_ops import TensorOps
from ember_ml.ops.interfaces.math_ops import MathOps
from ember_ml.ops.interfaces.device_ops import DeviceOps
from ember_ml.ops.interfaces.random_ops import RandomOps
from ember_ml.ops.interfaces.comparison_ops import ComparisonOps
from ember_ml.ops.interfaces.dtype_ops import DTypeOps
from ember_ml.ops.interfaces.solver_ops import SolverOps
from ember_ml.ops.interfaces.io_ops import IOOps
from ember_ml.ops.interfaces.loss_ops import LossOps
from ember_ml.ops.interfaces.vector_ops import VectorOps

__all__ = [
    'TensorOps',
    'MathOps',
    'DeviceOps',
    'RandomOps',
    'ComparisonOps',
    'DTypeOps',
    'SolverOps',
    'IOOps',
    'LossOps',
    'VectorOps',
]