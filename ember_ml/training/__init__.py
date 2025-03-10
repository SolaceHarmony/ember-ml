"""
Training module for ember_ml.

This module provides backend-agnostic implementations of training components
that work with any backend (NumPy, PyTorch, MLX).
"""

from ember_ml.training.optimizer import Optimizer, SGD, Adam
from ember_ml.training.loss import Loss, MSELoss, CrossEntropyLoss

__all__ = [
    'optimizer',
    'loss',
    'Optimizer',
    'SGD',
    'Adam',
    'Loss',
    'MSELoss',
    'CrossEntropyLoss',
]