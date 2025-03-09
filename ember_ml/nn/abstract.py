"""
Abstract base classes for neural network components.

This module provides abstract base classes that define the interfaces for
neural network components. Concrete implementations for different backends
(NumPy, PyTorch, MLX) will inherit from these classes.
"""

import abc
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Set, Tuple, Union, Any, List

class AbstractParameter(abc.ABC):
    """
    Abstract base class for parameters in neural networks.
    
    Parameters are tensors that require gradients and are updated during
    the optimization process.
    """
    
    @abc.abstractmethod
    def __init__(self, data, requires_grad=True):
        """
        Initialize a parameter with data.
        
        Args:
            data: Initial data for the parameter
            requires_grad: Whether the parameter requires gradients
        """
        pass
    
    @property
    @abc.abstractmethod
    def data(self):
        """Get the parameter data."""
        pass
    
    @data.setter
    @abc.abstractmethod
    def data(self, value):
        """Set the parameter data."""
        pass
    
    @property
    @abc.abstractmethod
    def grad(self):
        """Get the parameter gradient."""
        pass
    
    @grad.setter
    @abc.abstractmethod
    def grad(self, value):
        """Set the parameter gradient."""
        pass
    
    @property
    @abc.abstractmethod
    def requires_grad(self):
        """Get whether the parameter requires gradients."""
        pass
    
    @requires_grad.setter
    @abc.abstractmethod
    def requires_grad(self, value):
        """Set whether the parameter requires gradients."""
        pass

class AbstractModule(abc.ABC):
    """
    Abstract base class for all neural network modules.
    
    All custom modules should subclass this class and override the forward method.
    """
    
    @abc.abstractmethod
    def __init__(self):
        """Initialize the module."""
        pass
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the computation performed at every call.
        
        This method should be overridden by all subclasses.
        """
        pass
    
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Call the module on inputs.
        
        This method calls the forward method and handles any pre/post processing.
        """
        pass
    
    @abc.abstractmethod
    def register_parameter(self, name: str, param: Optional['AbstractParameter']) -> None:
        """
        Register a parameter with the module.
        
        Args:
            name: Name of the parameter
            param: Parameter to register, or None to remove
        """
        pass
    
    @abc.abstractmethod
    def register_buffer(self, name: str, buffer: Any) -> None:
        """
        Register a buffer with the module.
        
        Buffers are tensors that are not considered parameters but are part of the
        module's state, such as running means in batch normalization.
        
        Args:
            name: Name of the buffer
            buffer: Buffer to register, or None to remove
        """
        pass
    
    @abc.abstractmethod
    def add_module(self, name: str, module: Optional['AbstractModule']) -> None:
        """
        Register a submodule with the module.
        
        Args:
            name: Name of the submodule
            module: Module to register, or None to remove
        """
        pass
    
    @abc.abstractmethod
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, 'AbstractParameter']]:
        """
        Return an iterator over module parameters, yielding both the name and the parameter.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, yield parameters of submodules
            
        Yields:
            (name, parameter) pairs
        """
        pass
    
    @abc.abstractmethod
    def parameters(self, recurse: bool = True) -> Iterator['AbstractParameter']:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, yield parameters of submodules
            
        Yields:
            Module parameters
        """
        pass
    
    @abc.abstractmethod
    def train(self, mode: bool = True) -> 'AbstractModule':
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def eval(self) -> 'AbstractModule':
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def to(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> 'AbstractModule':
        """
        Move and/or cast the parameters and buffers.
        
        Args:
            device: Device to move parameters and buffers to
            dtype: Data type to cast parameters and buffers to
            
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        pass

class AbstractLinear(AbstractModule):
    """
    Abstract base class for linear layers.
    
    Applies a linear transformation to the incoming data: y = x @ W.T + b
    """
    
    @abc.abstractmethod
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[Any] = None
    ):
        """
        Initialize a linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            device: Device to place the parameters on
            dtype: Data type of the parameters
        """
        pass

class AbstractActivation(AbstractModule):
    """
    Abstract base class for activation functions.
    """
    pass

class AbstractReLU(AbstractActivation):
    """
    Abstract base class for ReLU activation.
    
    Applies the Rectified Linear Unit (ReLU) function element-wise:
    ReLU(x) = max(0, x)
    """
    
    @abc.abstractmethod
    def __init__(self, inplace=False):
        """
        Initialize a ReLU activation.
        
        Args:
            inplace: If True, modify the input tensor in-place (not supported in all backends)
        """
        pass

class AbstractSigmoid(AbstractActivation):
    """
    Abstract base class for Sigmoid activation.
    
    Applies the Sigmoid function element-wise:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    pass

class AbstractTanh(AbstractActivation):
    """
    Abstract base class for Tanh activation.
    
    Applies the Hyperbolic Tangent (Tanh) function element-wise:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    pass

class AbstractSoftmax(AbstractActivation):
    """
    Abstract base class for Softmax activation.
    
    Applies the Softmax function to an n-dimensional input tensor.
    
    The Softmax function is defined as:
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    
    @abc.abstractmethod
    def __init__(self, dim=-1):
        """
        Initialize a Softmax activation.
        
        Args:
            dim: Dimension along which Softmax will be computed (default: -1)
        """
        pass

class AbstractSequential(AbstractModule):
    """
    Abstract base class for sequential containers.
    
    A sequential container that runs modules in the order they were added.
    """
    
    @abc.abstractmethod
    def __init__(self, *args):
        """
        Initialize a Sequential container.
        
        Args:
            *args: Modules to add to the container
        """
        pass
    
    @abc.abstractmethod
    def append(self, module):
        """
        Append a module to the end of the container.
        
        Args:
            module: Module to append
            
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx):
        """
        Get a module or slice of modules from the container.
        
        Args:
            idx: Index or slice
            
        Returns:
            Module or Sequential container with sliced modules
        """
        pass
    
    @abc.abstractmethod
    def __len__(self):
        """Return the number of modules in the container."""
        pass