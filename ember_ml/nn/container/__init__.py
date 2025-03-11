"""
Container modules for ember_ml.

This module provides backend-agnostic implementations of container modules
that work with any backend (NumPy, PyTorch, MLX).
"""

import os
import importlib
from typing import Optional, Dict, Any, Type, List

# Import interfaces
from ember_ml.nn.container.interfaces import DenseInterface, DropoutInterface, BatchNormalizationInterface, SequentialInterface

# Use backend directly
from ember_ml.backend import get_backend, set_backend, get_backend_module

_CURRENT_INSTANCES = {}

def get_container():
    """Get the current container implementation name."""
    return get_backend()

def set_container(container_name: str):
    """Set the current container implementation."""
    global _CURRENT_INSTANCES
    
    # Set the backend
    set_backend(container_name)
    
    # Clear instances
    _CURRENT_INSTANCES = {}

def _load_container_module():
    """Load the current container module."""
    try:
        return get_backend_module()
    except (ImportError, ModuleNotFoundError):
        # If backend-specific implementation not found, use common implementation
        return importlib.import_module('ember_ml.nn.container.common')

def _get_container_instance(container_class: Type):
    """Get an instance of the specified container class."""
    global _CURRENT_INSTANCES
    
    if container_class not in _CURRENT_INSTANCES:
        try:
            module = _load_container_module()
            class_name = container_class.__name__[:-9]  # Remove 'Interface' suffix
            container_class_impl = getattr(module, class_name)
            _CURRENT_INSTANCES[container_class] = container_class_impl
        except (ImportError, AttributeError):
            # If backend-specific implementation not found, use common implementation
            common_module = importlib.import_module('ember_ml.nn.container.common')
            class_name = container_class.__name__[:-9]  # Remove 'Interface' suffix
            container_class_impl = getattr(common_module, class_name)
            _CURRENT_INSTANCES[container_class] = container_class_impl
    
    return _CURRENT_INSTANCES[container_class]

# Direct access to operations
# Dense operations
Dense = lambda units, activation=None, use_bias=True: _get_container_instance(DenseInterface)(units=units, activation=activation, use_bias=use_bias)

# Dropout operations
Dropout = lambda rate, seed=None: _get_container_instance(DropoutInterface)(rate=rate, seed=seed)

# BatchNormalization operations
BatchNormalization = lambda axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True: _get_container_instance(BatchNormalizationInterface)(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale)

# Sequential operations
Sequential = lambda layers=None: _get_container_instance(SequentialInterface)(layers=layers)

# Export all functions and classes
__all__ = [
    # Classes
    'DenseInterface',
    'DropoutInterface',
    'BatchNormalizationInterface',
    'SequentialInterface',
    
    # Functions
    'get_container',
    'set_container',
    
    # Operations
    'Dense',
    'Dropout',
    'BatchNormalization',
    'Sequential',
]