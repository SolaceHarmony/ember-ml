"""
PyTorch backend configuration for ember_ml.

This module provides configuration settings for the PyTorch backend,
with automatic device selection for optimal performance.
"""

import torch
from ember_ml.backend.torch.tensor.dtype import TorchDType
from ember_ml.backend.torch.typing import Device

# Backend information
__version__ = torch.__version__
has_gpu = torch.cuda.is_available()
has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Default settings
DEFAULT_DTYPE = TorchDType().float32
DEFAULT_DEVICE = 'cuda' if has_gpu else 'mps' if has_mps else 'cpu'
_default_device = DEFAULT_DEVICE
