"""
MLX backend for EmberHarmony.

This module provides MLX implementations of tensor operations.
"""

# Import all operations from the MLX backend modules
from ember_ml.backend.mlx.config import *
from ember_ml.backend.mlx.tensor_ops import *
from ember_ml.backend.mlx.math_ops import *
from ember_ml.backend.mlx.random_ops import *
from ember_ml.backend.mlx.comparison_ops import *
from ember_ml.backend.mlx.device_ops import *
from ember_ml.backend.mlx.dtype_ops import *
from ember_ml.backend.mlx.solver_ops import *

# Import MLX Ops classes
from ember_ml.backend.mlx.tensor_ops import MLXTensorOps
from ember_ml.backend.mlx.math_ops import MLXMathOps
from ember_ml.backend.mlx.random_ops import MLXRandomOps
from ember_ml.backend.mlx.comparison_ops import MLXComparisonOps
from ember_ml.backend.mlx.device_ops import MLXDeviceOps
from ember_ml.backend.mlx.dtype_ops import MLXDTypeOps
from ember_ml.backend.mlx.solver_ops import MLXSolverOps