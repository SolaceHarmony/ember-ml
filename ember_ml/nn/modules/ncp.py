"""
Neural Circuit Policy (NCP) module.

This module provides the NCP class, which implements a neural circuit policy
using a wiring configuration.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List

from ember_ml import ops
from ember_ml.nn.modules.base_module import BaseModule as Module, Parameter
from ember_ml.nn.wirings.wiring import Wiring

class NCP(Module):
    """
    Neural Circuit Policy (NCP) module.
    
    This module implements a neural circuit policy using a wiring configuration.
    It consists of a recurrent neural network with a specific connectivity pattern
    defined by the wiring configuration.
    """
    
    def __init__(
        self,
        wiring: Wiring,
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        dtype: Optional[Any] = None,
    ):
        """
        Initialize an NCP module.
        
        Args:
            wiring: Wiring configuration
            activation: Activation function to use
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias weights
            dtype: Data type for the weights
        """
        super().__init__()
        
        self.wiring = wiring
        self.activation_name = activation
        self.activation = ops.get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.dtype = dtype
        
        # Get masks from wiring
        self.input_mask = ops.convert_to_tensor(self.wiring.get_input_mask())
        self.recurrent_mask = ops.convert_to_tensor(self.wiring.get_recurrent_mask())
        self.output_mask = ops.convert_to_tensor(self.wiring.get_output_mask())
        
        # Initialize weights
        self._kernel = Parameter(
            self._initialize_tensor(
                (self.wiring.input_dim, self.wiring.units),
                self.kernel_initializer
            )
        )
        
        self._recurrent_kernel = Parameter(
            self._initialize_tensor(
                (self.wiring.units, self.wiring.units),
                self.recurrent_initializer
            )
        )
        
        if self.use_bias:
            self._bias = Parameter(
                self._initialize_tensor(
                    (self.wiring.units,),
                    self.bias_initializer
                )
            )
        else:
            self._bias = None
        
        # Initialize state
        self.state = ops.zeros((1, self.wiring.units))
    
    @property
    def kernel(self):
        """Get the kernel parameter."""
        return self._kernel.data
    
    @property
    def recurrent_kernel(self):
        """Get the recurrent kernel parameter."""
        return self._recurrent_kernel.data
    
    @property
    def bias(self):
        """Get the bias parameter."""
        if self.use_bias and self._bias is not None:
            return self._bias.data
        return None
    
    def _initialize_tensor(self, shape, initializer):
        """Initialize a tensor with the specified shape and initializer."""
        if initializer == "glorot_uniform":
            # Glorot uniform initialization
            fan_in = shape[0] if len(shape) >= 1 else 1
            fan_out = shape[1] if len(shape) >= 2 else 1
            limit = np.sqrt(6 / (fan_in + fan_out))
            return ops.random_uniform(shape, -limit, limit, dtype=self.dtype)
        elif initializer == "orthogonal":
            # Orthogonal initialization
            if len(shape) < 2:
                raise ValueError("Orthogonal initialization requires at least 2 dimensions")
            # Generate a random matrix
            a = ops.random_normal(shape, dtype=self.dtype)
            # Compute the QR factorization
            q, r = np.linalg.qr(ops.to_numpy(a))
            # Make Q uniform according to https://arxiv.org/pdf/1312.6120.pdf
            d = np.diag(r)
            ph = np.sign(d)
            q *= ph
            return ops.convert_to_tensor(q)
        elif initializer == "zeros":
            # Zeros initialization
            return ops.zeros(shape, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
    
    def forward(
        self,
        inputs: Any,
        state: Optional[Any] = None,
        return_state: bool = False
    ) -> Union[Any, Tuple[Any, Any]]:
        """
        Forward pass of the NCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        if state is None:
            state = self.state
            
        # Apply input mask
        masked_inputs = ops.multiply(inputs, self.input_mask)
        
        # Apply recurrent mask
        masked_state = ops.matmul(state, self.recurrent_mask)
        
        # Compute new state
        new_state = ops.matmul(masked_inputs, self.kernel)
        if self.use_bias:
            new_state = ops.add(new_state, self.bias)
        new_state = ops.add(new_state, ops.matmul(masked_state, self.recurrent_kernel))
        new_state = self.activation(new_state)
        
        # Compute output - only include motor neurons
        # The output mask is a binary mask that selects only the motor neurons
        masked_output = ops.multiply(new_state, self.output_mask)
        
        # Extract only the motor neurons (first output_dim neurons)
        output = masked_output[:, :self.wiring.output_dim]
        
        # Update state
        self.state = new_state
        
        if return_state:
            return output, new_state
        else:
            return output
    
    def reset_state(self) -> None:
        """
        Reset the state of the NCP module.
        """
        self.state = ops.zeros((1, self.wiring.units))
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP module.
        
        Returns:
            Dictionary containing the configuration
        """
        config = {
            "wiring": self.wiring.get_config(),
            "wiring_class": self.wiring.__class__.__name__,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "dtype": self.dtype
        }
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCP':
        """
        Create an NCP module from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP module
        """
        wiring_config = config.pop("wiring")
        wiring_class = config.pop("wiring_class")
        
        # Import the wiring class
        if wiring_class == "NCPWiring":
            from ember_ml.nn.wirings.ncp_wiring import NCPWiring
            wiring_class = NCPWiring
        else:
            from ember_ml.nn.wirings import wiring as wiring_module
            wiring_class = getattr(wiring_module, wiring_class)
        
        # Create the wiring
        wiring = wiring_class.from_config(wiring_config)
        
        # Create the NCP module
        return cls(wiring=wiring, **config)