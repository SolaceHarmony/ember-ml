"""
Modules for neural network components.

This module provides various neural network components that can be used
to build complex neural networks.
"""

from ember_ml.nn.modules.module import Module, Parameter
from ember_ml.nn.modules.base_module import BaseModule
from ember_ml.nn.modules.ncp import NCP
from ember_ml.nn.modules.auto_ncp import AutoNCP

__all__ = [
    'Module',
    'Parameter',
    'BaseModule',
    'NCP',
    'AutoNCP',
]