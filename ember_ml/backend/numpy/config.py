"""
NumPy backend configuration for EmberHarmony.

This module provides configuration settings for the NumPy backend.
"""

import numpy as np

# Default device for NumPy operations
DEFAULT_DEVICE = 'cpu'

# Default data type for NumPy operations
DEFAULT_DTYPE = np.float32

# Current random seed
_current_seed = None