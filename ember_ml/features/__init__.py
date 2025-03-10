"""
Features module.

This module provides feature extraction and transformation operations that abstract
machine learning library implementations.
"""

import os
import importlib
from typing import Optional, Dict, Any, Type

# Import interfaces
from ember_ml.features.interfaces import PCAInterface, StandardizeInterface, NormalizeInterface

# Use backend directly
from ember_ml.backend import get_backend, set_backend, get_backend_module

_CURRENT_INSTANCES = {}

def get_features():
    """Get the current features implementation name."""
    return get_backend()

def set_features(features_name: str):
    """Set the current features implementation."""
    global _CURRENT_INSTANCES
    
    # Set the backend
    set_backend(features_name)
    
    # Clear instances
    _CURRENT_INSTANCES = {}

def _load_features_module():
    """Load the current features module."""
    try:
        return get_backend_module()
    except (ImportError, ModuleNotFoundError):
        # If backend-specific implementation not found, use common implementation
        return importlib.import_module('ember_ml.features.common')

def _get_features_instance(features_class: Type):
    """Get an instance of the specified features class."""
    global _CURRENT_INSTANCES
    
    if features_class not in _CURRENT_INSTANCES:
        try:
            module = _load_features_module()
            
            # Get the backend directly
            backend = get_backend()
            
            # Get the class name prefix based on the current implementation
            if backend == 'numpy':
                class_name_prefix = 'Numpy'
            elif backend == 'torch':
                class_name_prefix = 'Torch'
            elif backend == 'mlx':
                class_name_prefix = 'MLX'
            else:
                raise ValueError(f"Unknown features implementation: {backend}")
            
            # Get the class name
            class_name = f"{class_name_prefix}{features_class.__name__[:-9]}"  # Remove 'Interface' suffix
            
            # Get the class and create an instance
            features_class_impl = getattr(module, class_name)
            _CURRENT_INSTANCES[features_class] = features_class_impl()
        except (ImportError, AttributeError):
            # If backend-specific implementation not found, use common implementation
            common_module = importlib.import_module('ember_ml.features.common')
            class_name = features_class.__name__[:-9]  # Remove 'Interface' suffix
            features_class_impl = getattr(common_module, class_name)
            _CURRENT_INSTANCES[features_class] = features_class_impl()
    
    return _CURRENT_INSTANCES[features_class]

# Convenience functions
def pca_features() -> PCAInterface:
    """Get PCA features."""
    return _get_features_instance(PCAInterface)

def standardize_features() -> StandardizeInterface:
    """Get standardize features."""
    return _get_features_instance(StandardizeInterface)

def normalize_features() -> NormalizeInterface:
    """Get normalize features."""
    return _get_features_instance(NormalizeInterface)

# Direct access to operations
# PCA operations
fit = lambda *args, **kwargs: pca_features().fit(*args, **kwargs)
transform = lambda *args, **kwargs: pca_features().transform(*args, **kwargs)
fit_transform = lambda *args, **kwargs: pca_features().fit_transform(*args, **kwargs)
inverse_transform = lambda *args, **kwargs: pca_features().inverse_transform(*args, **kwargs)

# Export all functions and classes
__all__ = [
    # Classes
    'PCAInterface',
    'StandardizeInterface',
    'NormalizeInterface',
    
    # Functions
    'get_features',
    'set_features',
    'pca_features',
    'standardize_features',
    'normalize_features',
    
    # Operations
    'fit',
    'transform',
    'fit_transform',
    'inverse_transform',
]
