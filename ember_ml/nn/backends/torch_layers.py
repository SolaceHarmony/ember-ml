"""
PyTorch backend implementations for neural network layers.

This module provides PyTorch-specific implementations of neural network layers
such as Linear, activation functions, etc.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Any

from ember_ml.nn.abstract import (
    AbstractLinear, AbstractReLU, AbstractSigmoid, 
    AbstractTanh, AbstractSoftmax, AbstractSequential
)
from ember_ml.nn.backends.torch_backend import (
    TorchModule, TorchParameter, get_default_device
)

class TorchLinear(TorchModule, AbstractLinear):
    """
    PyTorch implementation of a linear layer.
    
    Applies a linear transformation to the incoming data: y = x @ W.T + b
    """
    
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
            device: Device to place the parameters on (default: auto-detect)
            dtype: Data type of the parameters
        """
        TorchModule.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias  # Store whether we have a bias
        
        # Auto-detect device if not specified
        if device is None:
            device = get_default_device()
        
        # Initialize weights using Kaiming initialization (He initialization)
        # This is a good default for layers followed by ReLU
        std = (2.0 / in_features) ** 0.5
        weight_data = torch.randn(out_features, in_features, dtype=dtype, device=device) * std
        
        # Register the weight parameter
        self.register_parameter('weight', TorchParameter(weight_data, device=device))
        
        if bias:
            # Initialize bias to zeros
            bias_data = torch.zeros(out_features, dtype=dtype, device=device)
            self.register_parameter('bias', TorchParameter(bias_data, device=device))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device='cpu')
            if hasattr(self.weight, 'data') and hasattr(self.weight.data, 'device'):
                x = x.to(device=self.weight.data.device, dtype=self.weight.data.dtype)
        
        # Get the weight and bias from the parameters dictionary
        weight = self.weight.data if hasattr(self.weight, 'data') else self.weight
        bias = self.bias.data if self.bias is not None and hasattr(self.bias, 'data') else self.bias
        
        # Compute the linear transformation
        output = F.linear(x, weight, bias)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias}"
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias})"

class TorchReLU(TorchModule, AbstractReLU):
    """
    PyTorch implementation of ReLU activation.
    
    Applies the Rectified Linear Unit (ReLU) function element-wise:
    ReLU(x) = max(0, x)
    """
    
    def __init__(self, inplace=False):
        """
        Initialize a ReLU activation.
        
        Args:
            inplace: If True, modify the input tensor in-place
        """
        TorchModule.__init__(self)
        self.inplace = inplace
    
    def forward(self, x):
        """
        Forward pass of the ReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU activation applied
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=get_default_device())
        
        return F.relu(x, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"inplace={self.inplace}"

class TorchSigmoid(TorchModule, AbstractSigmoid):
    """
    PyTorch implementation of Sigmoid activation.
    
    Applies the Sigmoid function element-wise:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        """Initialize a Sigmoid activation."""
        TorchModule.__init__(self)
    
    def forward(self, x):
        """
        Forward pass of the Sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Sigmoid activation applied
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=get_default_device())
        
        return torch.sigmoid(x)

class TorchTanh(TorchModule, AbstractTanh):
    """
    PyTorch implementation of Tanh activation.
    
    Applies the Hyperbolic Tangent (Tanh) function element-wise:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def __init__(self):
        """Initialize a Tanh activation."""
        TorchModule.__init__(self)
    
    def forward(self, x):
        """
        Forward pass of the Tanh activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Tanh activation applied
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=get_default_device())
        
        return torch.tanh(x)

class TorchSoftmax(TorchModule, AbstractSoftmax):
    """
    PyTorch implementation of Softmax activation.
    
    Applies the Softmax function to an n-dimensional input tensor.
    
    The Softmax function is defined as:
    Softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    
    def __init__(self, dim=-1):
        """
        Initialize a Softmax activation.
        
        Args:
            dim: Dimension along which Softmax will be computed (default: -1)
        """
        TorchModule.__init__(self)
        self.dim = dim
    
    def forward(self, x):
        """
        Forward pass of the Softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with Softmax activation applied
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=get_default_device())
        
        return F.softmax(x, dim=self.dim)
    
    def extra_repr(self) -> str:
        """Return a string with extra information."""
        return f"dim={self.dim}"

class TorchSequential(TorchModule, AbstractSequential):
    """
    PyTorch implementation of a sequential container.
    
    A sequential container that runs modules in the order they were added.
    """
    
    def __init__(self, *args):
        """
        Initialize a Sequential container.
        
        Args:
            *args: Modules to add to the container
        """
        TorchModule.__init__(self)
        
        if len(args) == 1 and isinstance(args[0], dict):
            # If a single dict is provided, use it as the module dict
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # Add modules in order
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    
    def forward(self, x):
        """
        Forward pass through all modules in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through all modules
        """
        for module in self._modules.values():
            x = module(x)
        return x
    
    def append(self, module):
        """
        Append a module to the end of the container.
        
        Args:
            module: Module to append
            
        Returns:
            self
        """
        self.add_module(str(len(self)), module)
        return self
    
    def __getitem__(self, idx):
        """
        Get a module or slice of modules from the container.
        
        Args:
            idx: Index or slice
            
        Returns:
            Module or Sequential container with sliced modules
        """
        if isinstance(idx, slice):
            # Return a new Sequential container with the sliced modules
            return TorchSequential(dict(list(self._modules.items())[idx]))
        else:
            # Convert to integer index if needed
            if not isinstance(idx, int):
                idx = int(idx)
            
            # Handle negative indices
            if idx < 0:
                idx += len(self)
            
            # Check bounds
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for Sequential with length {len(self)}")
            
            # Return the module at the specified index
            return list(self._modules.values())[idx]
    
    def __len__(self):
        """Return the number of modules in the container."""
        return len(self._modules)