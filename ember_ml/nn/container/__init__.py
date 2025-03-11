"""
Container modules for ember_ml.

This module provides backend-agnostic implementations of container modules
that work with any backend (NumPy, PyTorch, MLX).
"""

import os
import importlib
from typing import Optional, Dict, Any, Type, List

# Import interfaces
from ember_ml.nn.container.interfaces import ContainerInterfaces

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

# Import directly from common implementation
from ember_ml.nn.container.common import Linear, Dropout, Sequential

# Export all functions and classes
__all__ = [
    # Classes
    'ContainerInterfaces',
    
    # Functions
    'get_container',
    'set_container',
    
    # Operations
    'Linear',
    'Dropout',
    'Sequential',
]