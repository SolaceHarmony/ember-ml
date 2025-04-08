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
        # Also register the parameter in the parameters dictionary
        self.register_parameter('_param2', param)
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
    assert ops.all(ops.equal(module._param1.data, tensor.zeros((10, 10)))), \
        "Direct update to param1 should be reflected"
    
    assert ops.all(ops.equal(module._param2.data, tensor.zeros((10, 10)))), \
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