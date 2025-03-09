"""
PyTorch backend configuration for EmberHarmony.

This module provides configuration information for the PyTorch backend.
"""

import torch

# Backend information
__version__ = torch.__version__
has_gpu = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
default_float_type = torch.float32

# Determine the best available device
if has_gpu:
    DEFAULT_DEVICE = 'cuda'
elif has_mps:
    DEFAULT_DEVICE = 'mps'
else:
    DEFAULT_DEVICE = 'cpu'

# Default device storage
_default_device = DEFAULT_DEVICE