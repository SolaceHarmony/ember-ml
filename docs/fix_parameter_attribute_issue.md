# Fixing the Parameter Attribute Issue in Ember ML

## Problem Description

There's an issue in the Ember ML framework where Parameter objects assigned to attributes in Module classes are not being properly set, particularly with the MLX backend. This happens because of how the `__setattr__` method in the `BaseModule` class works:

1. When a Parameter is assigned to an attribute (e.g., `self._kernel = param`), the `__setattr__` method registers it in the `_parameters` dictionary but doesn't set the actual attribute.
2. Since the attribute was previously initialized to `None` in `__init__`, when attempting to access it later, Python finds the `None` value directly and doesn't use `__getattr__` to look up the Parameter in the `_parameters` dictionary.

## Current Implementation

Here's the current implementation of `__setattr__` in the `BaseModule` class:

```python
def __setattr__(self, name, value):
    """Set an attribute on the module."""
    if isinstance(value, Parameter):
        self.register_parameter(name, value)
    elif isinstance(value, BaseModule):
        self.add_module(name, value)
    else:
        object.__setattr__(self, name, value)
```

And the `register_parameter` method:

```python
def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    """
    Register a parameter with the module.
    
    Args:
        name: Name of the parameter
        param: Parameter to register, or None to remove
    """
    if param is None:
        self._parameters.pop(name, None)
    else:
        self._parameters[name] = param
```

## Proposed Fix

To make the `__setattr__` method more robust, we need to ensure that it both registers the Parameter in the `_parameters` dictionary AND sets the actual attribute. Here's the proposed fix:

```python
def __setattr__(self, name, value):
    """Set an attribute on the module."""
    if isinstance(value, Parameter):
        # Register the parameter in the _parameters dictionary
        self.register_parameter(name, value)
        # ALSO set the actual attribute to enable direct attribute access
        object.__setattr__(self, name, value)
    elif isinstance(value, BaseModule):
        self.add_module(name, value)
        # Similarly, also set the actual attribute for modules
        object.__setattr__(self, name, value)
    else:
        object.__setattr__(self, name, value)
```

## Explanation of the Fix

This fix modifies the `__setattr__` method to do both:

1. Register Parameters in the `_parameters` dictionary (for parameter collection, gradients, etc.)
2. Set the actual instance attribute (for direct attribute access)

With this fix, both `self._parameters[name]` and `self.name` will refer to the same Parameter object, ensuring consistent behavior regardless of how the attribute is accessed.

## Benefits of This Approach

1. **Backward Compatibility**: This change maintains the existing parameter registration system while fixing the attribute access issue.
2. **Cross-Backend Consistency**: The fix ensures consistent behavior across all backends, including MLX.
3. **Intuitive Behavior**: Attributes behave as expected - setting an attribute actually sets that attribute.
4. **No Workarounds Needed**: Modules no longer need to use `object.__setattr__` to bypass the custom attribute setting logic.

## Implementation Plan

1. Modify the `__setattr__` method in `ember_ml/nn/modules/base_module.py` with the proposed fix.
2. Write tests to verify that Parameter attributes are properly set and accessible.
3. Test with all supported backends (NumPy, PyTorch, MLX) to ensure consistent behavior.
4. Update documentation to clarify how the parameter registration system works.

## Testing Strategy

To test this fix, we should:

1. Create a simple module with Parameter attributes
2. Verify that both direct attribute access and parameter collection return the same objects
3. Test with all supported backends
4. Ensure that parameter updates affect both the attribute and the collected parameter

This approach will make the Ember ML parameter system more robust and intuitive, eliminating the need for workarounds.