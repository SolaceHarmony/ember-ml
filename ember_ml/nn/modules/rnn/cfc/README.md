# Closed-form Continuous-time (CfC) Neural Networks

This module provides implementations of Closed-form Continuous-time (CfC) neural networks for the ember_ml framework. CfC networks are a type of recurrent neural network that operate in continuous time, making them well-suited for modeling dynamical systems and time series data.

## Overview

The CfC implementation in ember_ml includes:

1. **CfCCell**: A basic CfC cell implementation
2. **WiredCfCCell**: A CfC cell with custom wiring (e.g., Neural Circuit Policies)
3. **CfC**: A wrapper layer for CfC cells
4. **StrideAwareCfC**: A CfC implementation with awareness of different stride lengths

## Usage

### Basic CfC

```python
from ember_ml.nn.cfc import CfC

# Create a CfC layer with 32 units
cfc_layer = CfC(
    units=32,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    return_sequences=True,
    mixed_memory=True
)
```

### Wired CfC with Neural Circuit Policies

```python
from ember_ml.nn.wirings import AutoNCP
from ember_ml.nn.cfc import CfC

# Create an AutoNCP wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

# Create a CfC layer with the wiring
cfc_layer = CfC(
    wiring,
    return_sequences=True,
    mixed_memory=True
)
```

### Stride-Aware CfC

```python
from ember_ml.nn.wirings import AutoNCP
from ember_ml.nn.cfc import StrideAwareCfC, StrideAwareWiredCfCCell

# Create an AutoNCP wiring
wiring = AutoNCP(
    units=64,
    output_size=10,
    sparsity_level=0.5
)

# Create a stride-aware CfC cell
cell = StrideAwareWiredCfCCell(
    wiring=wiring,
    stride_length=2,
    time_scale_factor=1.0
)

# Create a stride-aware CfC layer
cfc_layer = StrideAwareCfC(
    cell=cell,
    return_sequences=True,
    mixed_memory=True
)
```

## Parameters

### CfCCell

- `units`: Number of units in the cell
- `time_scale_factor`: Factor to scale the time constant
- `activation`: Activation function for the output
- `recurrent_activation`: Activation function for the recurrent step
- `use_bias`: Whether to use bias
- `kernel_initializer`: Initializer for the kernel weights
- `recurrent_initializer`: Initializer for the recurrent weights
- `bias_initializer`: Initializer for the bias
- `mixed_memory`: Whether to use mixed memory

### WiredCfCCell

- `wiring`: Wiring configuration (e.g., AutoNCP)
- `time_scale_factor`: Factor to scale the time constant
- `activation`: Activation function for the output
- `recurrent_activation`: Activation function for the recurrent step
- `use_bias`: Whether to use bias
- `kernel_initializer`: Initializer for the kernel weights
- `recurrent_initializer`: Initializer for the recurrent weights
- `bias_initializer`: Initializer for the bias
- `mixed_memory`: Whether to use mixed memory

### CfC

- `cell_or_units`: CfCCell, WiredCfCCell, Wiring, or number of units
- `return_sequences`: Whether to return the full sequence or just the last output
- `return_state`: Whether to return the final state
- `go_backwards`: Whether to process the sequence backwards
- `mixed_memory`: Whether to use mixed memory

### StrideAwareCfC

- `cell`: StrideAwareCfCCell or StrideAwareWiredCfCCell instance
- `return_sequences`: Whether to return the full sequence or just the last output
- `return_state`: Whether to return the final state
- `go_backwards`: Whether to process the sequence backwards
- `stateful`: Whether to reuse the last state for the next batch
- `unroll`: Whether to unroll the RNN or use symbolic loop
- `mixed_memory`: Whether to use mixed memory for different strides

## Examples

See the `examples/cfc_example.py` file for a complete example of using CfC networks for time series prediction.

## References

- [Neural Circuit Policies Enabling Auditable Autonomy](https://www.nature.com/articles/s42256-020-00237-3) by Lechner et al.
- [NCPS: Neural Circuit Policy Search](https://github.com/mlech26l/ncps) - Original implementation