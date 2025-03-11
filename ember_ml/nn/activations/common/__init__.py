"""
Common activation function implementations.

This module provides backend-agnostic implementations of activation functions
using the ops abstraction layer.
"""

from ember_ml.nn.activations.common.tanh import Tanh
from ember_ml.nn.activations.common.softmax import Softmax
from ember_ml.nn.activations.common.dropout import Dropout

__all__ = [
    'Tanh',
    'Softmax',
    'Dropout',
]