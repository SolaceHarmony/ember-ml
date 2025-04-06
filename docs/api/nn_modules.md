# Neural Network Modules (nn.modules)

The `ember_ml.nn.modules` package provides a comprehensive set of backend-agnostic neural network modules for building machine learning models. These modules follow a consistent API across different backends and are designed to be composable and extensible.

## Importing

```python
from ember_ml.nn import modules
```

## Base Classes

### Module

`Module` is the base class for all neural network modules in Ember ML. It provides common functionality like parameter management, forward pass, and state tracking.

```python
from ember_ml.nn.modules import Module

class MyModule(Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters
        
    def forward(self, x):
        # Implement forward pass
        return x
```

### Parameter

`Parameter` represents a trainable parameter in a neural network module.

```python
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn import tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(tensor.random_normal((in_features, out_features)))
        self.bias = Parameter(tensor.zeros((out_features,)))
        
    def forward(self, x):
        return tensor.matmul(x, self.weight) + self.bias
```

### BaseModule

`BaseModule` extends `Module` with additional functionality for building more complex neural network modules.

### ModuleCell

`ModuleCell` is the base class for recurrent neural network cells.

### ModuleWiredCell

`ModuleWiredCell` extends `ModuleCell` with neuron map capabilities, allowing for custom connectivity patterns between neurons.

## Core Modules

### Dense

`Dense` implements a fully connected layer, also known as a linear or dense layer.

```python
from ember_ml.nn.modules import Dense
from ember_ml.nn import tensor

# Create a dense layer
layer = Dense(in_features=10, out_features=5, activation='relu')

# Forward pass
x = tensor.random_normal((32, 10))  # Batch of 32 samples with 10 features each
y = layer(x)  # Shape: (32, 5)
```

### NCP (Neural Circuit Policy)

`NCP` implements a neural circuit policy, a type of neural network with custom connectivity patterns.

```python
from ember_ml.nn.modules import NCP
from ember_ml.nn.modules.wiring import NCPMap

# Create a wiring configuration
wiring = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)

# Create an NCP
ncp = NCP(wiring=wiring, activation='tanh')

# Forward pass
x = tensor.random_normal((32, 8))  # Batch of 32 samples with 8 features each
y = ncp(x)  # Shape: (32, 3)
```

### AutoNCP

`AutoNCP` provides a convenient way to create an NCP with automatic wiring configuration.

```python
from ember_ml.nn.modules import AutoNCP

# Create an AutoNCP
auto_ncp = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5,
    seed=42
)

# Forward pass
x = tensor.random_normal((32, 16))  # Batch of 32 samples with 16 features each
y = auto_ncp(x)  # Shape: (32, 10)
```

## Activation Functions

The following activation functions are available:

| Activation | Description |
|------------|-------------|
| `ReLU` | Rectified Linear Unit activation |
| `Tanh` | Hyperbolic Tangent activation |
| `Sigmoid` | Sigmoid activation |
| `Softmax` | Softmax activation |
| `Softplus` | Softplus activation |
| `LeCunTanh` | LeCun Tanh activation |
| `Dropout` | Dropout regularization |

```python
from ember_ml.nn.modules import Dense, ReLU, Dropout
from ember_ml.nn import tensor

# Create a dense layer with ReLU activation
layer1 = Dense(in_features=10, out_features=5)
activation = ReLU()
dropout = Dropout(rate=0.2)

# Forward pass
x = tensor.random_normal((32, 10))
y = layer1(x)
y = activation(y)
y = dropout(y, training=True)
```

## Neuron Maps (Wiring)

Neuron maps (formerly called wirings) define the connectivity patterns between neurons in a neural network.

### NeuronMap

`NeuronMap` is the base class for all neuron maps.

### NCPMap

`NCPMap` implements a Neural Circuit Policy connectivity pattern.

```python
from ember_ml.nn.modules.wiring import NCPMap

# Create an NCP wiring configuration
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=3,
    sensory_neurons=8,
    seed=42
)
```

### FullyConnectedMap

`FullyConnectedMap` implements a fully connected connectivity pattern.

```python
from ember_ml.nn.modules.wiring import FullyConnectedMap

# Create a fully connected wiring configuration
neuron_map = FullyConnectedMap(
    units=10,
    output_size=5,
    input_size=8
)
```

### RandomMap

`RandomMap` implements a random connectivity pattern.

```python
from ember_ml.nn.modules.wiring import RandomMap

# Create a random wiring configuration
neuron_map = RandomMap(
    units=10,
    output_size=5,
    input_size=8,
    sparsity_level=0.5,
    seed=42
)
```

## Recurrent Neural Networks (RNN)

The `modules` package provides various recurrent neural network implementations.

### RNN

`RNN` implements a basic recurrent neural network.

```python
from ember_ml.nn.modules import RNN
from ember_ml.nn import tensor

# Create an RNN
rnn = RNN(
    input_size=10,
    hidden_size=20,
    activation='tanh'
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = rnn(x)  # y: (32, 5, 20), h: (32, 20)
```

### LSTM

`LSTM` implements a Long Short-Term Memory network.

```python
from ember_ml.nn.modules import LSTM
from ember_ml.nn import tensor

# Create an LSTM
lstm = LSTM(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, (h, c) = lstm(x)  # y: (32, 5, 20), h: (32, 20), c: (32, 20)
```

### GRU

`GRU` implements a Gated Recurrent Unit network.

```python
from ember_ml.nn.modules import GRU
from ember_ml.nn import tensor

# Create a GRU
gru = GRU(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = gru(x)  # y: (32, 5, 20), h: (32, 20)
```

### CfC and WiredCfCCell

`CfC` implements a Closed-form Continuous-time network, and `WiredCfCCell` adds neuron map capabilities.

```python
from ember_ml.nn.modules import CfC, WiredCfCCell
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a CfC
cfc = CfC(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = cfc(x)  # y: (32, 5, 20), h: (32, 20)

# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10,
    seed=42
)

# Create a WiredCfCCell
wired_cfc_cell = WiredCfCCell(
    neuron_map=neuron_map,
    mixed_memory=True
)

# Forward pass (cell level)
x_t = tensor.random_normal((32, 10))  # Single time step
h_prev = tensor.random_normal((32, 20))
h_next = wired_cfc_cell(x_t, h_prev)
```

### LTC and LTCCell

`LTC` implements a Liquid Time-Constant network, and `LTCCell` is the cell implementation.

```python
from ember_ml.nn.modules import LTC, LTCCell
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create an LTC
ltc = LTC(
    input_size=10,
    hidden_size=20
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = ltc(x)  # y: (32, 5, 20), h: (32, 20)

# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10,
    seed=42
)

# Create an LTCCell
ltc_cell = LTCCell(
    neuron_map=neuron_map,
    input_mapping='affine'
)

# Forward pass (cell level)
x_t = tensor.random_normal((32, 10))  # Single time step
h_prev = tensor.random_normal((32, 20))
h_next = ltc_cell(x_t, h_prev)
```

## Stride-Aware Modules

Stride-aware modules are specialized for processing temporal data with variable strides.

### StrideAware

`StrideAware` is the base class for stride-aware modules.

### StrideAwareCell

`StrideAwareCell` is the base class for stride-aware cells.

### StrideAwareCfC

`StrideAwareCfC` implements a stride-aware Closed-form Continuous-time network.

```python
from ember_ml.nn.modules import StrideAwareCfC
from ember_ml.nn import tensor

# Create a StrideAwareCfC
stride_cfc = StrideAwareCfC(
    input_size=10,
    hidden_size=20,
    stride_lengths=[1, 2, 4]
)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y, h = stride_cfc(x)  # y: (32, 5, 20), h: (32, 20)
```

### StrideAwareWiredCfCCell

`StrideAwareWiredCfCCell` implements a stride-aware Closed-form Continuous-time cell with neuron map capabilities.

```python
from ember_ml.nn.modules import StrideAwareWiredCfCCell
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn import tensor

# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=10,
    command_neurons=5,
    motor_neurons=5,
    sensory_neurons=10,
    seed=42
)

# Create a StrideAwareWiredCfCCell
stride_wired_cfc_cell = StrideAwareWiredCfCCell(
    neuron_map=neuron_map,
    stride_length=4,
    backbone_layers=2
)

# Forward pass (cell level)
x_t = tensor.random_normal((32, 10))  # Single time step
h_prev = tensor.random_normal((32, 20))
h_next = stride_wired_cfc_cell(x_t, h_prev)
```

## Building Complex Models

You can combine these modules to build complex neural network architectures:

```python
from ember_ml.nn.modules import Module, Dense, LSTM, Dropout
from ember_ml.nn import tensor

class SequenceClassifier(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dropout = Dropout(rate=0.2)
        self.dense = Dense(in_features=hidden_size, out_features=num_classes, activation='softmax')
        
    def forward(self, x, training=False):
        # x shape: (batch_size, sequence_length, input_size)
        y, (h, _) = self.lstm(x)
        # Use the last hidden state
        h = self.dropout(h, training=training)
        # Pass through the dense layer
        output = self.dense(h)
        return output

# Create a sequence classifier
model = SequenceClassifier(input_size=10, hidden_size=20, num_classes=5)

# Forward pass
x = tensor.random_normal((32, 5, 10))  # Batch of 32 sequences of length 5 with 10 features each
y = model(x, training=True)  # Shape: (32, 5)
```

## Backend Support

All modules are backend-agnostic and work with any backend (NumPy, PyTorch, MLX) using the backend abstraction layer.

```python
from ember_ml.nn.modules import Dense
from ember_ml.backend import set_backend

# Use NumPy backend
set_backend('numpy')
dense_numpy = Dense(in_features=10, out_features=5)

# Use PyTorch backend
set_backend('torch')
dense_torch = Dense(in_features=10, out_features=5)

# Use MLX backend
set_backend('mlx')
dense_mlx = Dense(in_features=10, out_features=5)
```

## Implementation Details

The neural network modules are implemented using a layered architecture:

1. **Base Classes**: Provide common functionality for all modules
2. **Core Modules**: Implement basic neural network components
3. **Advanced Modules**: Implement specialized neural network architectures

This architecture allows Ember ML to provide a consistent API across different backends while still leveraging the unique capabilities of each backend.

## Additional Resources

For more detailed information on specific modules, see the following resources:

- [RNN Modules Documentation](nn_modules_rnn.md): Detailed documentation on recurrent neural network modules
- [Neuron Maps Documentation](nn_modules_wiring.md): Detailed documentation on neuron maps
- [Tensor Module Documentation](nn_tensor.md): Documentation on tensor operations used by the modules