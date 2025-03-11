"""
Common container implementations.

This module provides backend-agnostic implementations of container operations
using the ops abstraction layer.
"""

from ember_ml.nn.container.common.dense import Dense
from ember_ml.nn.container.common.dropout import Dropout
from ember_ml.nn.container.common.batch_normalization import BatchNormalization
from ember_ml.nn.container.common.sequential import Sequential

__all__ = [
    'Dense',
    'Dropout',
    'BatchNormalization',
    'Sequential',
]