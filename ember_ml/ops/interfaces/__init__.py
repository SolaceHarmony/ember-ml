"""
Interfaces for operations.

This module defines the abstract interfaces for operations that abstract
machine learning library scalar operations. Tensor operations have been moved to ember_ml.nn.tensor.
"""

from ember_ml.ops.interfaces.math_ops import MathOps
from ember_ml.ops.interfaces.device_ops import DeviceOps
from ember_ml.ops.interfaces.comparison_ops import ComparisonOps
from ember_ml.ops.interfaces.solver_ops import SolverOps
from ember_ml.ops.interfaces.io_ops import IOOps
from ember_ml.ops.interfaces.loss_ops import LossOps
from ember_ml.ops.interfaces.vector_ops import VectorOps

__all__ = [
    'MathOps',
    'DeviceOps',
    'ComparisonOps',
    'SolverOps',
    'IOOps',
    'LossOps',
    'VectorOps',
]