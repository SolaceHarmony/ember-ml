"""
NumPy backend implementation for BizarroMath arbitrary-precision arithmetic.

This module houses the NumPy-based implementations of MegaNumber and MegaBinary,
providing the foundation for arbitrary-precision and binary wave computations
within the NumPy backend.
"""

from ember_ml.backend.numpy.bizarromath.mega_binary import NumpyMegaBinary, InterferenceMode
# Import classes from submodules
from ember_ml.backend.numpy.bizarromath.mega_number import NumpyMegaNumber

# Define what gets exported when 'from . import *' is used
__all__ = [
    'NumpyMegaNumber',
    'NumpyMegaBinary',
    'InterferenceMode',
]