"""
Factory functions for creating neural network components.

This module provides factory functions that create the appropriate implementation
of neural network components based on the current backend.
"""

from ember_ml import backend as K
from ember_ml.nn.backends import get_implementation

def get_current_backend_name():
    """
    Get the name of the current backend.
    
    Returns:
        Name of the current backend ('torch', 'numpy', 'mlx')
    """
    # Check the backend module's attributes to determine its type
    if hasattr(K, 'torch') or hasattr(K, '__name__') and 'torch' in K.__name__:
        return 'torch'
    elif hasattr(K, 'numpy') or hasattr(K, '__name__') and 'numpy' in K.__name__:
        return 'numpy'
    elif hasattr(K, 'mlx') or hasattr(K, '__name__') and 'mlx' in K.__name__:
        return 'torch'  # Default to torch for MLX for now
    
    # If we can't determine the backend, default to torch
    return 'torch'

def create_module(class_name, *args, **kwargs):
    """
    Create a neural network module with the current backend.
    
    Args:
        class_name: Name of the class to create
        *args: Positional arguments to pass to the constructor
        **kwargs: Keyword arguments to pass to the constructor
        
    Returns:
        Instance of the requested class with the current backend
    """
    backend_name = get_current_backend_name()
    implementation = get_implementation(backend_name, class_name)
    return implementation(*args, **kwargs)

# Factory functions for specific module types

def Parameter(data, requires_grad=True):
    """
    Create a Parameter with the current backend.
    
    Args:
        data: Initial data for the parameter
        requires_grad: Whether the parameter requires gradients
        
    Returns:
        Parameter instance with the current backend
    """
    return create_module('Parameter', data, requires_grad)

def Linear(in_features, out_features, bias=True, device=None, dtype=None):
    """
    Create a Linear layer with the current backend.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias
        device: Device to place the parameters on
        dtype: Data type of the parameters
        
    Returns:
        Linear layer instance with the current backend
    """
    return create_module('Linear', in_features, out_features, bias, device, dtype)

def ReLU(inplace=False):
    """
    Create a ReLU activation with the current backend.
    
    Args:
        inplace: If True, modify the input tensor in-place (not supported in all backends)
        
    Returns:
        ReLU activation instance with the current backend
    """
    return create_module('ReLU', inplace)

def Sigmoid():
    """
    Create a Sigmoid activation with the current backend.
    
    Returns:
        Sigmoid activation instance with the current backend
    """
    return create_module('Sigmoid')

def Tanh():
    """
    Create a Tanh activation with the current backend.
    
    Returns:
        Tanh activation instance with the current backend
    """
    return create_module('Tanh')

def Softmax(dim=-1):
    """
    Create a Softmax activation with the current backend.
    
    Args:
        dim: Dimension along which Softmax will be computed (default: -1)
        
    Returns:
        Softmax activation instance with the current backend
    """
    return create_module('Softmax', dim)

def Sequential(*args):
    """
    Create a Sequential container with the current backend.
    
    Args:
        *args: Modules to add to the container
        
    Returns:
        Sequential container instance with the current backend
    """
    return create_module('Sequential', *args)

def MSELoss(reduction='mean'):
    """
    Create a Mean Squared Error loss with the current backend.
    
    Args:
        reduction: Specifies the reduction to apply to the output
        
    Returns:
        MSELoss instance with the current backend
    """
    return create_module('MSELoss', reduction=reduction)

def CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0):
    """
    Create a Cross Entropy loss with the current backend.
    
    Args:
        weight: Manual rescaling weight given to each class
        reduction: Specifies the reduction to apply to the output
        ignore_index: Specifies a target value that is ignored
        label_smoothing: Float in [0.0, 1.0], specifies the amount of smoothing
        
    Returns:
        CrossEntropyLoss instance with the current backend
    """
    return create_module('CrossEntropyLoss', weight, reduction, ignore_index, label_smoothing)

def BCELoss(weight=None, reduction='mean'):
    """
    Create a Binary Cross Entropy loss with the current backend.
    
    Args:
        weight: Manual rescaling weight given to the loss of each batch element
        reduction: Specifies the reduction to apply to the output
        
    Returns:
        BCELoss instance with the current backend
    """
    return create_module('BCELoss', weight, reduction)

def BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None):
    """
    Create a Binary Cross Entropy with Logits loss with the current backend.
    
    Args:
        weight: Manual rescaling weight given to the loss of each batch element
        reduction: Specifies the reduction to apply to the output
        pos_weight: Weight of positive examples
        
    Returns:
        BCEWithLogitsLoss instance with the current backend
    """
    return create_module('BCEWithLogitsLoss', weight, reduction, pos_weight)