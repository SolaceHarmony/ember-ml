"""
NumPy backend for ember_ml.

This module provides NumPy implementations of tensor operations.
"""

# Import tensor classes
from ember_ml.backend.numpy.tensor import NumpyDType, NumpyTensor

# Import all operations from the NumPy backend modules
from ember_ml.backend.numpy.config import *
from ember_ml.backend.numpy.math_ops import *
from ember_ml.backend.numpy.comparison_ops import *
from ember_ml.backend.numpy.device_ops import *
from ember_ml.backend.numpy.linearalg.solvers_ops import *
from ember_ml.backend.numpy.io_ops import *
from ember_ml.backend.numpy.activations.ops import *
from ember_ml.backend.numpy.loss_ops import *
from ember_ml.backend.numpy.vector_ops import *
from ember_ml.backend.numpy.stats import *
from ember_ml.backend.numpy.bitwise import *

from . import linearalg  # re-export module namespace
from . import stats  # re-export module namespace
from . import bitwise  # re-export module namespace