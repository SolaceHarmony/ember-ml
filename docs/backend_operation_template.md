# Backend Operation Implementation Template

This document provides a template and guidelines for implementing operations across different backends in the EmberHarmony framework.

## Implementation Template

When implementing a new operation or adding a missing operation to a backend, follow this template:

```python
def operation_name(x, *args, **kwargs):
    """
    Operation description.
    
    Parameters
    ----------
    x : tensor
        Input tensor.
    arg1 : type
        Description of arg1.
    arg2 : type, optional
        Description of arg2. Default: value.
    
    Returns
    -------
    tensor
        Description of return value.
    
    Notes
    -----
    Any implementation notes, including differences between backends.
    
    Examples
    --------
    >>> x = ops.ones((3, 3))
    >>> result = ops.operation_name(x, arg1, arg2=value)
    """
    # Implementation for the specific backend
    pass
```

## Implementation Guidelines

### 1. Function Signature

- Keep function signatures consistent across backends
- Use the same parameter names and default values
- Document all parameters clearly

### 2. Type Handling

- Handle different input types consistently
- Convert inputs to tensors when appropriate
- Return tensors of the appropriate type

### 3. Error Handling

- Check input shapes and types
- Provide clear error messages
- Handle edge cases consistently

### 4. Performance Considerations

- Optimize for the specific backend
- Use backend-specific optimizations when available
- Consider memory usage and computational efficiency

## Example Implementation

Here's an example of implementing the `softplus` operation across different backends:

### NumPy Backend

```python
def softplus(x):
    """
    Softplus activation function: log(1 + exp(x)).
    
    Parameters
    ----------
    x : tensor
        Input tensor.
    
    Returns
    -------
    tensor
        Softplus of the input tensor.
    """
    return np.log1p(np.exp(x))
```

### PyTorch Backend

```python
def softplus(x):
    """
    Softplus activation function: log(1 + exp(x)).
    
    Parameters
    ----------
    x : tensor
        Input tensor.
    
    Returns
    -------
    tensor
        Softplus of the input tensor.
    """
    return torch.nn.functional.softplus(x)
```

### MLX Backend

```python
def softplus(x):
    """
    Softplus activation function: log(1 + exp(x)).
    
    Parameters
    ----------
    x : tensor
        Input tensor.
    
    Returns
    -------
    tensor
        Softplus of the input tensor.
    """
    return mx.log(mx.add(1.0, mx.exp(x)))
```

## Testing

After implementing an operation, add tests to verify consistent behavior across backends:

1. Add the operation to `OPERATIONS_TO_TEST` in `utils/test_backend_consistency.py`
2. Run the consistency test to verify the implementation
3. Add specific tests to the appropriate test files

## Documentation

Update the documentation to reflect the new operation:

1. Add the operation to the API reference
2. Include examples of usage
3. Document any backend-specific considerations