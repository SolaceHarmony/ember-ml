# Neural Circuit Policies (NCPs) in EmberHarmony

## Introduction

Neural Circuit Policies (NCPs) are a biologically-inspired neural network architecture that models the connectivity patterns of biological neural circuits. This document provides a comprehensive overview of the NCP integration in EmberHarmony, including its architecture, components, and usage.

## Background

Neural Circuit Policies were introduced in the paper [Neural Circuit Policies Enabling Auditable Autonomy](https://www.nature.com/articles/s42256-020-00237-3) by Lechner et al. They are designed to mimic the connectivity patterns of biological neural circuits, which can lead to more interpretable and auditable neural networks.

The key insight behind NCPs is that by constraining the connectivity of the network to follow biologically-inspired patterns, we can create neural networks that are more interpretable and auditable, while still maintaining high performance.

## Architecture

The NCP integration in EmberHarmony consists of the following components:

### Wiring System

The wiring system defines the connectivity patterns between neurons in the network. It consists of the following components:

- **Wiring**: Base class for all wiring configurations
- **FullyConnectedWiring**: All neurons are connected to all other neurons
- **RandomWiring**: Connections between neurons are randomly generated
- **NCPWiring**: Neurons are divided into sensory, inter, and motor neurons

### NCP Module

The NCP module implements a neural circuit policy using a wiring configuration. It consists of a recurrent neural network with a specific connectivity pattern defined by the wiring configuration.

### AutoNCP Module

The AutoNCP module is a convenience wrapper around the NCP module that automatically configures the wiring based on the number of units and outputs.

## Implementation Details

### Wiring System

The wiring system is implemented in the `emberharmony.nn.wirings` package. Each wiring configuration is a subclass of the `Wiring` base class, which defines the interface for all wiring configurations.

The wiring configurations use binary masks to define the connectivity patterns between neurons. These masks are used to constrain the connectivity of the network during the forward pass.

### NCP Module

The NCP module is implemented in the `emberharmony.nn.modules.ncp` module. It uses the wiring configurations to create a recurrent neural network with a specific connectivity pattern.

The NCP module applies the wiring masks during the forward pass to constrain the connectivity of the network. This ensures that the network follows the specified connectivity pattern.

### AutoNCP Module

The AutoNCP module is implemented in the `emberharmony.nn.modules.auto_ncp` module. It automatically configures the wiring based on the number of units and outputs, making it easier to create neural circuit policies.

## Backend Agnosticism

The NCP implementation in EmberHarmony is backend-agnostic, meaning it can work with different backends (NumPy, PyTorch, MLX) without any code changes. This is achieved through the use of the `ops` module, which provides a unified interface for tensor operations across different backends.

## Usage Examples

### Creating a Wiring Configuration

```python
from emberharmony.nn.wirings import NCPWiring

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)
```

### Creating an NCP Module

```python
from emberharmony.nn.wirings import NCPWiring
from emberharmony.nn.modules import NCP

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)

# Create an NCP model
model = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros"
)
```

### Creating an AutoNCP Module

```python
from emberharmony.nn.modules import AutoNCP

# Create an AutoNCP model
model = AutoNCP(
    units=20,
    output_size=5,
    sparsity_level=0.5,
    seed=42,
    activation="tanh",
    use_bias=True
)
```

### Training an NCP Model

```python
import numpy as np
from emberharmony import ops
from emberharmony.nn.wirings import NCPWiring
from emberharmony.nn.modules import NCP

# Create a simple dataset
X = ops.reshape(ops.linspace(0, 2 * np.pi, 100), (-1, 1))
y = ops.sin(X)

# Convert to numpy for splitting
X_np = ops.to_numpy(X)
y_np = ops.to_numpy(y)

# Split into train and test sets
X_train, X_test = X_np[:80], X_np[80:]
y_train, y_test = y_np[:80], y_np[80:]

# Create a wiring configuration
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=1,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)

# Create an NCP model
model = NCP(
    wiring=wiring,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros"
)

# Train the model
learning_rate = 0.01
epochs = 100
batch_size = 16

for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Train in batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        model.reset_state()
        y_pred = model(ops.convert_to_tensor(X_batch))
        
        # Compute loss
        loss = ops.mean(ops.square(y_pred - ops.convert_to_tensor(y_batch)))
        
        # Compute gradients
        params = list(model.parameters())
        grads = ops.gradients(loss, params)
        
        # Update parameters
        for param, grad in zip(params, grads):
            param.data = ops.subtract(param.data, ops.multiply(ops.convert_to_tensor(learning_rate), grad))
        
        epoch_loss += ops.to_numpy(loss)
    
    epoch_loss /= (len(X_train) // batch_size)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")

# Evaluate the model
model.reset_state()
y_pred = ops.to_numpy(model(ops.convert_to_tensor(X_test)))
test_loss = np.mean(np.square(y_pred - y_test))
print(f"Test Loss: {test_loss:.6f}")
```

## Advanced Usage

### Custom Wiring Configurations

You can create custom wiring configurations by subclassing the `Wiring` base class and implementing the `build` method:

```python
from emberharmony.nn.wirings import Wiring
import numpy as np

class CustomWiring(Wiring):
    """
    Custom wiring configuration.
    """
    
    def __init__(self, units, output_dim=None, sparsity_level=0.5, seed=None):
        """
        Initialize a custom wiring configuration.
        
        Args:
            units: Number of units in the circuit
            output_dim: Number of output dimensions (default: units)
            sparsity_level: Sparsity level for the connections (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(units, output_dim, sparsity_level, seed)
    
    def build(self):
        """
        Build the custom wiring configuration.
        
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Create custom masks
        input_mask = np.ones(self.input_dim, dtype=np.int32)
        recurrent_mask = np.ones((self.units, self.units), dtype=np.int32)
        output_mask = np.ones(self.units, dtype=np.int32)
        
        # Apply custom connectivity pattern
        # ...
        
        return input_mask, recurrent_mask, output_mask
```

### Custom NCP Modules

You can create custom NCP modules by subclassing the `NCP` class and overriding the `forward` method:

```python
from emberharmony.nn.modules import NCP
from emberharmony import ops

class CustomNCP(NCP):
    """
    Custom Neural Circuit Policy module.
    """
    
    def __init__(self, wiring, activation="tanh", use_bias=True, **kwargs):
        """
        Initialize a custom NCP module.
        
        Args:
            wiring: Wiring configuration
            activation: Activation function to use
            use_bias: Whether to use bias
            **kwargs: Additional arguments
        """
        super().__init__(wiring, activation, use_bias, **kwargs)
    
    def forward(self, inputs, state=None, return_state=False):
        """
        Forward pass of the custom NCP module.
        
        Args:
            inputs: Input tensor
            state: Optional state tensor
            return_state: Whether to return the state
            
        Returns:
            Output tensor, or tuple of (output, state) if return_state is True
        """
        # Custom forward pass
        # ...
        
        # Call the parent forward method
        return super().forward(inputs, state, return_state)
```

## Module Organization

The NCP implementation in EmberHarmony is organized as follows:

- **emberharmony.nn.wirings**: Contains the wiring configurations
  - `Wiring`: Base class for all wiring configurations
  - `FullyConnectedWiring`: All neurons are connected to all other neurons
  - `RandomWiring`: Connections between neurons are randomly generated
  - `NCPWiring`: Neurons are divided into sensory, inter, and motor neurons

- **emberharmony.nn.modules**: Contains the NCP and AutoNCP modules
  - `NCP`: Implements a neural circuit policy using a wiring configuration
  - `AutoNCP`: A convenience wrapper around the NCP module

Note that while there is a top-level `emberharmony.wirings` module, the NCP wiring configurations are specifically located in `emberharmony.nn.wirings` to maintain a clear separation between neural network components and other types of wirings.

## Conclusion

The NCP integration in EmberHarmony provides a flexible and powerful way to create biologically-inspired neural networks. By constraining the connectivity of the network to follow biologically-inspired patterns, we can create neural networks that are more interpretable and auditable, while still maintaining high performance.

## References

- [Neural Circuit Policies Enabling Auditable Autonomy](https://www.nature.com/articles/s42256-020-00237-3) by Lechner et al.
- [NCPS: Neural Circuit Policy Search](https://github.com/mlech26l/ncps) - Original implementation