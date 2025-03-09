"""
Sequential container module for emberharmony.

This module provides a backend-agnostic implementation of the Sequential container
that works with any backend (NumPy, PyTorch, MLX).
"""

from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple, Union, Any, Sequence

from ember_ml.nn.modules import Module

class Sequential(Module):
    """
    A sequential container that runs modules in the order they were added.
    
    Example:
        ```
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )
        ```
        
        or
        
        ```
        model = Sequential()
        model.add_module('fc1', Linear(10, 20))
        model.add_module('relu', ReLU())
        model.add_module('fc2', Linear(20, 1))
        ```
    """
    
    def __init__(self, *args):
        """
        Initialize a Sequential container.
        
        Args:
            *args: Modules to add to the container
        """
        super().__init__()
        
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # If a single OrderedDict is provided, use it as the module dict
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
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
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
    
    def __iter__(self):
        """Return an iterator over the modules in the container."""
        return iter(self._modules.values())
    
    def __repr__(self):
        """Return a string representation of the container."""
        if not self._modules:
            return "Sequential()"
        
        module_str = ",\n  ".join(repr(module) for module in self._modules.values())
        return f"Sequential(\n  {module_str}\n)"