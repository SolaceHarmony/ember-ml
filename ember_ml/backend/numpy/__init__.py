"""
NumPy backend for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

# Import all operations from the NumPy backend modules
from ember_ml.backend.numpy.config import *
from ember_ml.backend.numpy.tensor_ops import *
from ember_ml.backend.numpy.math_ops import *
from ember_ml.backend.numpy.random_ops import *
from ember_ml.backend.numpy.comparison_ops import *
from ember_ml.backend.numpy.device_ops import *
from ember_ml.backend.numpy.dtype_ops import *
from ember_ml.backend.numpy.solver_ops import *
from ember_ml.backend.numpy.io_ops import *

# Import NumPy Ops classes
from ember_ml.backend.numpy.tensor_ops import NumpyTensorOps
from ember_ml.backend.numpy.math_ops import NumpyMathOps
from ember_ml.backend.numpy.random_ops import NumpyRandomOps
from ember_ml.backend.numpy.comparison_ops import NumpyComparisonOps
from ember_ml.backend.numpy.device_ops import NumpyDeviceOps
from ember_ml.backend.numpy.dtype_ops import NumpyDTypeOps
from ember_ml.backend.numpy.solver_ops import NumpySolverOps
from ember_ml.backend.numpy.io_ops import NumPyIOOps