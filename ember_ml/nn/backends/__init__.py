"""
Backend implementations for neural network components.

This module provides backend-specific implementations of neural network
components for different backends (NumPy, PyTorch, MLX).
"""

from ember_ml.nn.backends.torch_backend import TorchParameter, TorchModule
from ember_ml.nn.backends.torch_layers import (
    TorchLinear, TorchReLU, TorchSigmoid, 
    TorchTanh, TorchSoftmax, TorchSequential
)
from ember_ml.nn.backends.torch_loss import (
    TorchLoss, TorchMSELoss, TorchCrossEntropyLoss,
    TorchBCELoss, TorchBCEWithLogitsLoss
)

# Dictionary mapping backend names to their implementations
BACKEND_IMPLEMENTATIONS = {
    'torch': {
        'Parameter': TorchParameter,
        'Module': TorchModule,
        'Linear': TorchLinear,
        'ReLU': TorchReLU,
        'Sigmoid': TorchSigmoid,
        'Tanh': TorchTanh,
        'Softmax': TorchSoftmax,
        'Sequential': TorchSequential,
        'MSELoss': TorchMSELoss,
        'CrossEntropyLoss': TorchCrossEntropyLoss,
        'BCELoss': TorchBCELoss,
        'BCEWithLogitsLoss': TorchBCEWithLogitsLoss
    }
    # Add other backends (numpy, mlx) here as they are implemented
}

def get_implementation(backend_name, class_name):
    """
    Get the backend-specific implementation of a neural network component.
    
    Args:
        backend_name: Name of the backend ('torch', 'numpy', 'mlx')
        class_name: Name of the class to get
        
    Returns:
        Backend-specific implementation of the class
        
    Raises:
        ValueError: If the backend or class is not implemented
    """
    if backend_name not in BACKEND_IMPLEMENTATIONS:
        raise ValueError(f"Backend '{backend_name}' is not implemented")
    
    backend_classes = BACKEND_IMPLEMENTATIONS[backend_name]
    if class_name not in backend_classes:
        raise ValueError(f"Class '{class_name}' is not implemented for backend '{backend_name}'")
    
    return backend_classes[class_name]