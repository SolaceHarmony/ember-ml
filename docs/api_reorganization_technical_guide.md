# Ember ML API Reorganization: Technical Implementation Guide

This guide provides detailed technical instructions for implementing the Ember ML API reorganization. It supplements the high-level plan in `api_reorganization_plan.md` with specific code changes and implementation details.

## Table of Contents

1. [Registry System Enhancements](#registry-system-enhancements)
2. [Top-Level API Implementation](#top-level-api-implementation)
3. [Backend Implementation Changes](#backend-implementation-changes)
4. [Function Categorization](#function-categorization)
5. [Module Implementation Details](#module-implementation-details)
6. [Testing Strategy](#testing-strategy)
7. [Backward Compatibility](#backward-compatibility)

## Registry System Enhancements

### Enhanced Function Registry

The registry system will be enhanced to support function categories:

```python
# ember_ml/backend/registry.py

class EnhancedBackendRegistry(BackendRegistry):
    """Enhanced registry with category support."""
    
    def _initialize(self):
        """Initialize the registry with category support."""
        super()._initialize()
        self._function_categories = {}  # Maps category to function names
        
    def register_function(self, name, func, categories=None):
        """Register a function with optional categories.
        
        Args:
            name: Function name
            func: Function implementation
            categories: Optional list of categories
        """
        self._registered_functions[name] = func
        
        # Register with categories
        if categories:
            for category in categories:
                if category not in self._function_categories:
                    self._function_categories[category] = set()
                self._function_categories[category].add(name)
    
    def get_category_functions(self, category):
        """Get all functions in a category.
        
        Args:
            category: Category name
            
        Returns:
            Dict of function name to function for the category
        """
        if category not in self._function_categories:
            return {}
            
        result = {}
        for name in self._function_categories[category]:
            if name in self._registered_functions:
                result[name] = self._registered_functions[name]
        return result
```

### Registration Decorator

Create a new decorator for registering functions with categories:

```python
# ember_ml/backend/registry.py

def register_op(name, categories=None):
    """Decorator for registering backend operations.
    
    Args:
        name: Name of the operation
        categories: Optional list of categories
    """
    def decorator(func):
        registry = EnhancedBackendRegistry()
        registry.register_function(name, func, categories)
        return func
    return decorator
```

## Top-Level API Implementation

### Main Module Update

Update the main module to expose all operations at the top level:

```python
# ember_ml/tensor.py

from ember_ml.backend import get_backend, auto_select_backend, set_backend

# Import core tensor functions
from ember_ml._tensor_ops import (
    array, ones, zeros, eye, arange, linspace,
    zeros_like, ones_like, full, full_like
)

# Import common operations
from ember_ml._ops import (
    add, subtract, multiply, divide, matmul,
    reshape, transpose, concatenate, stack, split
)

# Import submodules
from ember_ml import linalg
from ember_ml import stats
from ember_ml import random
from ember_ml import activations
from ember_ml import nn

# Re-export common submodule functions
from ember_ml.linalg import svd, qr, inv, eigh
from ember_ml.stats import mean, std, var, median
from ember_ml.random import normal, uniform, seed
from ember_ml.activations import relu, sigmoid, tanh, softmax

# Constants
from ember_ml._ops import pi, e
```

### Tensor Operations Module

Create a module for tensor operations:

```python
# ember_ml/_tensor_ops.py

from ember_ml.backend import get_backend_module

def _get_tensor_ops_module():
    """Get the tensor ops module from the current backend."""
    backend = get_backend_module()
    return backend.tensor_ops

def array(data, dtype=None):
    """Create a tensor array.

    Args:
        data: Input data
        dtype: Optional data type

    Returns:
        Backend tensor
    """
    tensor_ops = _get_tensor_ops_module()
    return tensor_ops.array(data, dtype)

def ones(shape, dtype=None):
    """Create a tensor of ones.

    Args:
        shape: Shape of the tensor
        dtype: Optional data type

    Returns:
        Backend tensor of ones
    """
    tensor_ops = _get_tensor_ops_module()
    return tensor_ops.ones(shape, dtype)

# Additional tensor creation functions...
```

## Backend Implementation Changes

### Backend Structure Changes

Flatten the backend directory structure:

```
ember_ml/backend/
├── __init__.py
├── registry.py
├── numpy/
│   ├── __init__.py
│   ├── tensor_ops.py       # (Moved from tensor/)
│   ├── math_ops.py
│   ├── linalg_ops.py
│   └── stats_ops.py
├── torch/
│   └── ...
└── mlx/
    └── ...
```

### Backend Function Registration

Register backend functions with categories:

```python
# ember_ml/backend/numpy/tensor_ops.py

from ember_ml.backend.registry import register_op

@register_op("array", categories=["tensor"])
def array(data, dtype=None):
    """Create a NumPy array."""
    import numpy as np
    return np.array(data, dtype=dtype)

@register_op("ones", categories=["tensor"])
def ones(shape, dtype=None):
    """Create a NumPy array of ones."""
    import numpy as np
    return np.ones(shape, dtype=dtype)
```

## Function Categorization

### Category Module Implementation

Create a module for each category:

```python
# ember_ml/linalg/tensor.py

from ember_ml.backend.registry import EnhancedBackendRegistry

# Get registry instance
_registry = EnhancedBackendRegistry()

# Get all linear algebra functions
_linalg_functions = _registry.get_category_functions("linalg")

# Add them to the module namespace
globals().update(_linalg_functions)

# Define what's accessible via import
__all__ = list(_linalg_functions.keys())
```

### Example Categories

Here are the primary function categories:

1. **Tensor Operations** (Moved to top level)
   - Creation: `array`, `ones`, `zeros`, etc.
   - Manipulation: `reshape`, `transpose`, etc.

2. **Math Operations** (Moved to top level)
   - Basic: `add`, `subtract`, `multiply`, `divide`
   - Advanced: `matmul`, `pow`, etc.

3. **Linear Algebra** (`linalg` module)
   - Decomposition: `svd`, `qr`, `eigh`
   - Solutions: `solve`, `inv`
   - Norms: `norm`

4. **Statistics** (`stats` module)
   - Descriptive: `mean`, `std`, `var`
   - Distributions: `normal`, `uniform`

5. **Random** (`random` module)
   - Generators: `normal`, `uniform`, `binomial`
   - Utilities: `seed`, `permutation`

6. **Activations** (`activations` module)
   - Functions: `relu`, `sigmoid`, `tanh`

## Module Implementation Details

### Linear Algebra Module

```python
# ember_ml/linalg/tensor.py

"""
Linear algebra functions for Ember ML.

This module provides linear algebra operations such as matrix decomposition,
eigenvalue problems, and linear systems.
"""

from ember_ml.backend.registry import EnhancedBackendRegistry

# Get registry instance
_registry = EnhancedBackendRegistry()

# Get all linear algebra functions
_linalg_functions = _registry.get_category_functions("linalg")

# Add them to the module namespace
globals().update(_linalg_functions)

# Define what's accessible via import
__all__ = list(_linalg_functions.keys())
```

### Stats Module

```python
# ember_ml/stats/tensor.py

"""
Statistical functions for Ember ML.

This module provides statistical operations such as mean, variance,
standard deviation, and other descriptive statistics.
"""

from ember_ml.backend.registry import EnhancedBackendRegistry

# Get registry instance
_registry = EnhancedBackendRegistry()

# Get all statistics functions
_stats_functions = _registry.get_category_functions("stats")

# Add them to the module namespace
globals().update(_stats_functions)

# Define what's accessible via import
__all__ = list(_stats_functions.keys())
```

### Activations Module

```python
# ember_ml/activations/tensor.py

"""
Activation functions for Ember ML.

This module provides neural network activation functions such as
ReLU, sigmoid, and tanh.
"""

from ember_ml.backend.registry import EnhancedBackendRegistry

# Get registry instance
_registry = EnhancedBackendRegistry()

# Get all activation functions
_activations_functions = _registry.get_category_functions("activations")

# Add them to the module namespace
globals().update(_activations_functions)

# Define what's accessible via import
__all__ = list(_activations_functions.keys())
```

## Testing Strategy

### Test Structure

Update tests to follow the new API structure:

```python
# tests/test_tensor_ops.py

import ember_ml as em
import numpy as np

def test_array_creation():
    """Test array creation."""
    x = em.array([1, 2, 3])
    assert x.shape == (3,)
    
def test_ones():
    """Test ones creation."""
    x = em.ones((2, 3))
    assert x.shape == (2, 3)
    assert np.all(em.to_numpy(x) == 1)
```

### Test Coverage

Ensure test coverage for:

1. Top-level functions
2. Category-specific functions
3. Backend switching
4. Edge cases and error handling

## Backward Compatibility

### Compatibility Layer

Create a compatibility layer to transition users:

```python
# ember_ml/nn/tensor/tensor.py

import warnings

warnings.warn(
    "The ember_ml.nn.tensor module is deprecated. "
    "Please use the top-level tensor functions (ember_ml.array, etc.) instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from top-level for backward compatibility
from ember_ml import array, ones, zeros, eye, arange, linspace
from ember_ml import reshape, transpose, concatenate, stack, split
```

### Deprecated Function Warnings

Add warnings to deprecated functions:

```python
# ember_ml/nn/tensor/common/tensor.py

def array(data, dtype=None):
    """Create a tensor array (DEPRECATED)."""
    warnings.warn(
        "tensor.array() is deprecated. Please use ember_ml.array() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ember_ml import array as new_array
    return new_array(data, dtype)
```

## Migration Timeline

1. **Phase 1 (Immediate)**: Release with new API, backward compatibility
2. **Phase 2 (3 months)**: Mark old API as deprecated with warnings
3. **Phase 3 (6 months)**: Remove old API paths

This technical implementation guide provides specific code templates and structures for implementing the API reorganization. Each section could be expanded with more detailed code examples as needed during implementation.
