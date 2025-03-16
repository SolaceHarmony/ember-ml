"""
PyTorch backend configuration for ember_ml.

This module provides configuration information for the PyTorch backend,
with automatic device selection for optimal performance.
"""

import torch
from ember_ml.backend.torch.tensor.dtype import TorchDType

# Backend information
__version__ = torch.__version__
has_gpu = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Default data type for PyTorch operations
DEFAULT_DTYPE = TorchDType().float32

# Determine the best available device
if has_gpu:
    DEFAULT_DEVICE = 'cuda'
elif has_mps:
    DEFAULT_DEVICE = 'mps'
else:
    DEFAULT_DEVICE = 'cpu'

# Default device storage
_default_device = DEFAULT_DEVICE