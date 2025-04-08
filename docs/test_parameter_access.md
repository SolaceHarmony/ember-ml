# Test Cases for Parameter Access Issue

This document provides test cases to demonstrate and verify the Parameter access issue in Ember ML. These tests can be used to confirm the issue exists and to verify that the proposed fix resolves it.

## Test Setup

Create a new file `tests/test_parameter_access.py` with the following content:

```python
"""
Tests for Parameter access in Modules.
"""

import pytest
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.base_module import BaseModule, Parameter
from ember_ml.backend import set_backend, get_backend


class SimpleModule(BaseModule):
    """Simple module for testing parameter access."""
    
    def __init__(self):
        super().__init__()
        # Initialize parameters to None
        self._param1 = None
        self._param2 = None
    
    def forward(self, x):
        # Use parameters directly
        return ops.matmul(x, self._param1) + self._param2
    
    # Helper methods to demonstrate different ways of setting parameters
    def set_param1_normal(self, param):
        """Set param1 using normal assignment."""
        self._param1 = param
    
    def set_param2_direct(self, param):
        """Set param2 using object.__setattr__."""
        object.__setattr__(self, '_param2', param)


def test_parameter_access_normal_assignment():
    """Test parameter access with normal assignment."""
    module = SimpleModule()
    param = Parameter(tensor.ones((10, 10)))
    
    # Set parameter using normal assignment
    module.set_param1_normal(param)
    
    # Test direct attribute access
    print(f"Direct access: {module._param1}")
    
    # Test parameter collection access
    param_dict = dict(module.named_parameters())
    print(f"Dictionary access: {param_dict.get('_param1')}")
    
    # Check if they're the same object
    assert module._param1 is param_dict.get('_param1'), \
        "Parameter should be accessible both directly and via parameter collection"


def test_parameter_access_direct_assignment():
    """Test parameter access with direct assignment."""
    module = SimpleModule()
    param = Parameter(tensor.ones((10, 10)))
    
    # Set parameter using direct assignment
    module.set_param2_direct(param)
    
    # Test direct attribute access
    print(f"Direct access: {module._param2}")
    
    # Test parameter collection access
    param_dict = dict(module.named_parameters())
    print(f"Dictionary access: {param_dict.get('_param2')}")
    
    # Check if they're the same object
    assert module._param2 is param_dict.get('_param2'), \
        "Parameter should be accessible both directly and via parameter collection"


def test_parameter_update_propagation():
    """Test that updates to parameters propagate correctly."""
    module = SimpleModule()
    param1 = Parameter(tensor.ones((10, 10)))
    param2 = Parameter(tensor.ones((10, 10)))
    
    # Set parameters
    module.set_param1_normal(param1)
    module.set_param2_direct(param2)
    
    # Update param1 via direct access
    module._param1.data = tensor.zeros((10, 10))
    
    # Update param2 via parameter collection
    param_dict = dict(module.named_parameters())
    param_dict['_param2'].data = tensor.zeros((10, 10))
    
    # Check both access methods reflect the updates
    assert tensor.all_equal(module._param1.data, tensor.zeros((10, 10))), \
        "Direct update to param1 should be reflected"
    
    assert tensor.all_equal(module._param2.data, tensor.zeros((10, 10))), \
        "Update via parameter collection to param2 should be reflected"


@pytest.mark.parametrize("backend", ["numpy", "torch", "mlx"])
def test_parameter_access_cross_backend(backend):
    """Test parameter access across different backends."""
    # Skip if backend is not available
    try:
        set_backend(backend)
    except ImportError:
        pytest.skip(f"Backend {backend} not available")
    
    # Create module and parameters
    module = SimpleModule()
    param1 = Parameter(tensor.ones((10, 10)))
    param2 = Parameter(tensor.ones((10, 10)))
    
    # Set parameters
    module.set_param1_normal(param1)
    module.set_param2_direct(param2)
    
    # Check direct access
    assert module._param1 is not None, \
        f"Parameter should be accessible directly with {backend} backend"
    assert module._param2 is not None, \
        f"Parameter should be accessible directly with {backend} backend"
    
    # Check parameter collection
    param_dict = dict(module.named_parameters())
    assert param_dict.get('_param1') is not None, \
        f"Parameter should be in collection with {backend} backend"
    assert param_dict.get('_param2') is not None, \
        f"Parameter should be in collection with {backend} backend"
    
    # Check if they're the same object
    assert module._param1 is param_dict.get('_param1'), \
        f"Parameter should be the same object with {backend} backend"
    assert module._param2 is param_dict.get('_param2'), \
        f"Parameter should be the same object with {backend} backend"
    
    # Reset backend
    set_backend('numpy')
```

## Expected Results Before Fix

When running these tests before applying the fix:

1. `test_parameter_access_normal_assignment` will fail with the MLX backend because `module._param1` will be `None` even though the parameter is correctly registered in the parameter collection.

2. `test_parameter_access_direct_assignment` will pass with all backends because it uses `object.__setattr__` to bypass the custom attribute setting logic.

3. `test_parameter_update_propagation` will fail with the MLX backend because updates to `module._param1` won't propagate correctly.

4. `test_parameter_access_cross_backend` will fail for the MLX backend because parameters won't be accessible directly.

## Expected Results After Fix

After applying the fix to the `__setattr__` method:

1. All tests should pass with all backends.

2. Parameters should be accessible both directly as attributes and via the parameter collection.

3. Updates to parameters should propagate correctly regardless of how they're accessed.

## Running the Tests

To run these tests:

```bash
pytest -xvs tests/test_parameter_access.py
```

To run with a specific backend:

```bash
pytest -xvs tests/test_parameter_access.py::test_parameter_access_cross_backend[mlx]
```

These tests will help verify that the proposed fix resolves the parameter access issue consistently across all backends.