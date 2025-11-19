"""
MLX backend implementation for BizarroMath arbitrary-precision arithmetic.

This module houses the MLX-based implementations of MegaNumber and MegaBinary,
providing the foundation for arbitrary-precision and binary wave computations
within the MLX backend.
"""

from .mega_binary import MLXMegaBinary, InterferenceMode
# Import classes from submodules
from .mega_number import MLXMegaNumber

# Define what gets exported when 'from . import *' is used
__all__ = [
    'MLXMegaNumber',
    'MLXMegaBinary',
    'InterferenceMode',
]