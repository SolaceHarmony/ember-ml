"""
PyTorch optimizations module.

This module provides optimizations for PyTorch,
including tensor optimizations and backend comparisons.
"""

from ember_ml.backend.torch.optimizations.pytorch_tensor_optimization import *
from ember_ml.backend.torch.optimizations.torch_mps_demo import *
from ember_ml.backend.torch.optimizations.compare_torch_backends import *
from ember_ml.backend.torch.optimizations.compare_torch_backends_improved import *

__all__ = [
    'pytorch_tensor_optimization',
    'torch_mps_demo',
    'compare_torch_backends',
    'compare_torch_backends_improved',
]
