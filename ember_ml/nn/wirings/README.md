# Neural Network Wirings

This module provides neural network specific wiring implementations for ember_ml, with a focus on Neural Circuit Policies (NCPs) and other biologically-inspired connectivity patterns.

## Overview

Wirings define the connectivity patterns between neurons in a neural network. They are used to constrain the connectivity of the network during the forward pass, which can lead to more interpretable and auditable neural networks.

## Available Wirings

### Base Wiring

The `Wiring` class is the base class for all wiring configurations. It defines the interface for all wiring configurations and provides common functionality.

```python
from ember_ml.nn.wirings import Wiring

# Create a custom wiring
class CustomWiring(Wiring):
    def __init__(self, units, output_dim=None, sparsity_level=0.5, seed=None):
        super().__init__(units, output_dim, sparsity_level, seed)
    
    def build(self):
        # Implement custom wiring logic
        return input_mask, recurrent_mask, output_mask
```

### Neural Circuit Policy (NCP) Wiring

The `NCPWiring` class implements the wiring configuration for Neural Circuit Policies. It divides neurons into sensory, inter, and motor neurons, and defines the connectivity patterns between them.

```python
from ember_ml.nn.wirings import NCPWiring

# Create an NCP wiring
wiring = NCPWiring(
    inter_neurons=10,
    motor_neurons=5,
    sensory_neurons=0,
    sparsity_level=0.5,
    seed=42
)
```

### Fully Connected Wiring

The `FullyConnectedWiring` class implements a fully connected wiring configuration, where all neurons are connected to all other neurons.

```python
from ember_ml.nn.wirings import FullyConnectedWiring

# Create a fully connected wiring
wiring = FullyConnectedWiring(
    units=20,
    output_size=5
)
```

### Random Wiring

The `RandomWiring` class implements a randomly generated wiring configuration, where connections between neurons are randomly generated based on a sparsity level.

```python
from ember_ml.nn.wirings import RandomWiring

# Create a random wiring
wiring = RandomWiring(
    units=20,
    output_size=5,
    sparsity_level=0.5,
    seed=42
)
```

### Auto NCP Wiring

The `AutoNCPWiring` class is a convenience wrapper around the `NCPWiring` class that automatically configures the wiring based on the number of units and outputs.

```python
from ember_ml.nn.wirings import AutoNCPWiring

# Create an auto NCP wiring
wiring = AutoNCPWiring(
    units=20,
    output_size=5,
    sparsity_level=0.5,
    seed=42
)
```

## Usage with NCP Modules

These wirings are designed to be used with the NCP modules in `ember_ml.nn.modules`. Here's an example of how to use them together:

```python
from ember_ml.nn.wirings import NCPWiring
from ember_ml.nn.modules import NCP

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

## Implementation Details

The wiring configurations use binary masks to define the connectivity patterns between neurons. These masks are used to constrain the connectivity of the network during the forward pass.

- **Input Mask**: Defines which neurons receive input from the input layer
- **Recurrent Mask**: Defines the connectivity between neurons in the recurrent layer
- **Output Mask**: Defines which neurons contribute to the output

For more detailed information on Neural Circuit Policies and their wiring configurations, see the [Neural Circuit Policies documentation](../../docs/neural_circuit_policies.md).

## Relationship with ember_ml.wirings

It's important to understand the distinction between this module (`ember_ml.nn.wirings`) and the general wirings module (`ember_ml.wirings`):

- **ember_ml.nn.wirings**: Contains neural network specific wiring implementations, such as Neural Circuit Policy (NCP) wirings, fully connected wirings, and random wirings. These are specifically designed for use with neural network components in the `ember_ml.nn` module.

- **ember_ml.wirings**: Contains general connectivity utilities and patterns that can be used in various contexts, not just neural networks. This includes graph-based connectivity, general network topologies, and utility functions for working with connectivity patterns.