"""
Base Module class for neural network components.

This module provides the foundation for building neural network components
that work with any backend (NumPy, PyTorch, MLX).

The BaseModule class is the base class for all neural network modules in ember_ml.
"""

import inspect
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Set, Tuple, Union, Any, List

from ember_ml import ops
from ember_ml.nn import tensor

class Parameter:
    """
    A special kind of tensor that represents a trainable parameter.
    
    Parameters are tensors that require gradients and are updated during
    the optimization process.
    """
    def __init__(self, data, requires_grad=True):
        """
        Initialize a parameter with data.
        
        Args:
            data: Initial data for the parameter
            requires_grad: Whether the parameter requires gradients
        """
        self.data = tensor.convert_to_tensor(data)
        self.requires_grad = requires_grad
        self.grad = None
    
    def __repr__(self):
        shape_val = tensor.shape(self.data)
        try:
            dtype_val = tensor.dtype(self.data)
            return f"Parameter(shape={shape_val}, dtype={dtype_val})"
        except:
            return f"Parameter(shape={shape_val})"

class BaseModule:
    """
    Base class for all neural network modules in ember_ml.
    
    All custom modules should subclass this class and override the forward method.
    This class provides the foundation for building neural network components
    that work with any backend (NumPy, PyTorch, MLX).
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
            self._buffers[name] = tensor.convert_to_tensor(buffer)
    
    def add_module(self, name: str, module: Optional['BaseModule']) -> None:
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
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
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
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, yield parameters of submodules
            
        Yields:
            Module parameters
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Any]]:
        """
        Return an iterator over module buffers, yielding both the name and the buffer.
        
        Args:
            prefix: Prefix to prepend to buffer names
            recurse: If True, yield buffers of submodules
            
        Yields:
            (name, buffer) pairs
        """
        for name, buf in self._buffers.items():
            yield prefix + ('.' if prefix else '') + name, buf
        
        if recurse:
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, buf in module.named_buffers(submodule_prefix, recurse):
                    yield name, buf
    
    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        """
        Return an iterator over module buffers.
        
        Args:
            recurse: If True, yield buffers of submodules
            
        Yields:
            Module buffers
        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf
    
    def named_modules(self, prefix: str = '', memo: Optional[Set['BaseModule']] = None) -> Iterator[Tuple[str, 'BaseModule']]:
        """
        Return an iterator over all modules in the network, yielding both the name and the module.
        
        Args:
            prefix: Prefix to prepend to module names
            memo: Set of modules already yielded
            
        Yields:
            (name, module) pairs
        """
        if memo is None:
            memo = set()
        
        if self not in memo:
            memo.add(self)
            yield prefix, self
            
            for module_name, module in self._modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + module_name
                for name, mod in module.named_modules(submodule_prefix, memo):
                    yield name, mod
    
    def modules(self) -> Iterator['BaseModule']:
        """
        Return an iterator over all modules in the network.
        
        Yields:
            Modules in the network
        """
        for _, module in self.named_modules():
            yield module
    
    def train(self, mode: bool = True) -> 'BaseModule':
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        self.training = mode
        for module in self.modules():
            module.training = mode
        return self
    
    def eval(self) -> 'BaseModule':
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)
    
    def to(self, device: Optional[str] = None, dtype: Optional[Any] = None) -> 'BaseModule':
        """
        Move and/or cast the parameters and buffers.
        
        Args:
            device: Device to move parameters and buffers to
            dtype: Data type to cast parameters and buffers to
            
        Returns:
            self
        """
        for param in self.parameters():
            if dtype is not None:
                param.data = tensor.cast(param.data, dtype)
            if device is not None:
                param.data = ops.to_device(param.data, device)
        
        for key, buf in self._buffers.items():
            if dtype is not None:
                self._buffers[key] = tensor.cast(buf, dtype)
            if device is not None:
                self._buffers[key] = ops.to_device(buf, device)
        
        return self
    
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
    
    def __setattr__(self, name, value):
        """Set an attribute on the module."""
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, BaseModule):
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the module."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = tensor.zeros_like(param.grad)