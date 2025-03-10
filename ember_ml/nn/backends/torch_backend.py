"""
PyTorch backend implementation for neural network components.

This module provides PyTorch-specific implementations of the abstract neural
network components defined in ember_ml.nn.abstract.
"""

import torch
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Set, Tuple, Union, Any, List
import platform

from ember_ml.nn.abstract import (
    AbstractParameter, AbstractModule, AbstractLinear,
    AbstractReLU, AbstractSigmoid, AbstractTanh, AbstractSoftmax,
    AbstractSequential
)

# Check if Metal is available on macOS
def is_metal_available():
    """Check if Metal is available on the system."""
    return (platform.system() == "Darwin" and 
            hasattr(torch.backends, "mps") and 
            torch.backends.mps.is_available())

# Get the default device based on availability
def get_default_device():
    """Get the default device based on what's available."""
    if torch.cuda.is_available():
        return "cuda"
    elif is_metal_available():
        return "mps"  # Metal Performance Shaders device
    else:
        return "cpu"

class TorchParameter(AbstractParameter):
    """
    PyTorch implementation of a parameter.
    """
    
    def __init__(self, data, requires_grad=True, device=None):
        """
        Initialize a parameter with data.
        
        Args:
            data: Initial data for the parameter
            requires_grad: Whether the parameter requires gradients
            device: Device to place the parameter on (default: auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            device = get_default_device()
            
        if isinstance(data, torch.Tensor):
            self._data = data.to(device=device)
        else:
            self._data = torch.tensor(data, device=device)
        
        self._data.requires_grad_(requires_grad)
    
    @property
    def data(self):
        """Get the parameter data."""
        return self._data
    
    @data.setter
    def data(self, value):
        """Set the parameter data."""
        if isinstance(value, torch.Tensor):
            self._data = value.to(device=self._data.device, dtype=self._data.dtype)
        else:
            self._data = torch.tensor(value, device=self._data.device, dtype=self._data.dtype)
        
        self._data.requires_grad_(self._data.requires_grad)
    
    @property
    def grad(self):
        """Get the parameter gradient."""
        return self._data.grad
    
    @grad.setter
    def grad(self, value):
        """Set the parameter gradient."""
        self._data.grad = value
    
    @property
    def requires_grad(self):
        """Get whether the parameter requires gradients."""
        return self._data.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value):
        """Set whether the parameter requires gradients."""
        self._data.requires_grad_(value)
    
    def __repr__(self):
        return f"TorchParameter(shape={tuple(self._data.shape)}, dtype={self._data.dtype}, device={self._data.device})"

class TorchModule(AbstractModule):
    """
    PyTorch implementation of a module.
    """
    
    def __init__(self):
        """Initialize the module."""
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()
        self.training = True
    
    def forward(self, *args, **kwargs):
        """
        Define the computation performed at every call.
        
        This method should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def __call__(self, *args, **kwargs):
        """
        Call the module on inputs.
        
        This method calls the forward method and handles any pre/post processing.
        """
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name: str, param: Optional[TorchParameter]) -> None:
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
    
    def register_buffer(self, name: str, buffer: Any) -> None:
        """
        Register a buffer with the module.
        
        Buffers are tensors that are not considered parameters but are part of the
        module's state, such as running means in batch normalization.
        
        Args:
            name: Name of the buffer
            buffer: Buffer to register, or None to remove
        """
        if buffer is None:
            self._buffers.pop(name, None)
        else:
            if isinstance(buffer, torch.Tensor):
                self._buffers[name] = buffer
            else:
                device = get_default_device()
                self._buffers[name] = torch.tensor(buffer, device=device)
    
    def add_module(self, name: str, module: Optional['TorchModule']) -> None:
        """
        Register a submodule with the module.
        
        Args:
            name: Name of the submodule
            module: Module to register, or None to remove
        """
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, TorchParameter]]:
        """
        Return an iterator over module parameters, yielding both the name and the parameter.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, yield parameters of submodules
            
        Yields:
            (name, parameter) pairs
        """
        for name, param in self._parameters.items():
            yield prefix + ('.' if prefix else '') + name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, param in module.named_parameters(submodule_prefix, recurse):
                    yield name, param
    
    def parameters(self, recurse: bool = True) -> Iterator[TorchParameter]:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, yield parameters of submodules
            
        Yields:
            Module parameters
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param
    
    def train(self, mode: bool = True) -> 'TorchModule':
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'TorchModule':
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)
    
    def to(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> 'TorchModule':
        """
        Move and/or cast the parameters and buffers.
        
        Args:
            device: Device to move parameters and buffers to
            dtype: Data type to cast parameters and buffers to
            
        Returns:
            self
        """
        # Handle Metal device if specified as "metal" or "mps"
        if device in ["metal", "mps"] and not is_metal_available():
            print("Warning: Metal requested but not available. Falling back to CPU.")
            device = "cpu"
        
        # Auto-detect device if not specified
        if device is None:
            device = get_default_device()
        
        for param in self.parameters():
            if dtype is not None:
                param.data = param.data.to(dtype=dtype)
            if device is not None:
                param.data = param.data.to(device=device)
        
        for key, buf in self._buffers.items():
            if dtype is not None:
                self._buffers[key] = buf.to(dtype=dtype)
            if device is not None:
                self._buffers[key] = buf.to(device=device)
        
        return self
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def __getattr__(self, name):
        """
        Get an attribute from the module.
        
        This method is called when the default attribute lookup fails.
        It looks for the attribute in _parameters, _buffers, and _modules.
        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """
        Set an attribute on the module.
        
        This method handles special cases for parameters, modules, and buffers.
        """
        # Handle parameters
        if isinstance(value, TorchParameter):
            self.register_parameter(name, value)
        # Handle modules
        elif isinstance(value, TorchModule):
            self.add_module(name, value)
        # Handle buffers (tensors that are not parameters)
        elif isinstance(value, torch.Tensor) and name not in ['training']:
            self.register_buffer(name, value)
        # Handle normal attributes
        else:
            object.__setattr__(self, name, value)
    
    def __repr__(self):
        """Return a string representation of the module."""
        lines = [self.__class__.__name__ + '(']
        
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '  ' + mod_str.replace('\n', '\n  ')
            lines.append('(' + name + '): ' + mod_str)
        
        for name, param in self._parameters.items():
            lines.append('(' + name + '): ' + repr(param))
        
        lines.append(')')
        return '\n'.join(lines)