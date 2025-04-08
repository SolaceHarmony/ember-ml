# Current State Analysis of Ember ML RNN Architecture

## 1. File Structure and Organization

### Core Files
- `ember_ml/nn/modules/wiring/neuron_map.py`: Base NeuronMap class
- `ember_ml/nn/modules/wiring/ncp_map.py`: NCPMap implementation
- `ember_ml/nn/modules/wiring/fully_connected_map.py`: FullyConnectedMap implementation
- `ember_ml/nn/modules/module_cell.py`: ModuleCell base class
- `ember_ml/nn/modules/module_wired_cell.py`: ModuleWiredCell base class
- `ember_ml/nn/modules/rnn/ltc_cell.py`: LTCCell implementation
- `ember_ml/nn/modules/rnn/ltc.py`: LTC layer implementation
- `ember_ml/nn/modules/rnn/cfc_cell.py`: CfCCell implementation
- `ember_ml/nn/modules/rnn/wired_cfc_cell.py`: WiredCfCCell implementation
- `ember_ml/nn/modules/rnn/cfc.py`: CfC layer implementation

### Documentation Files
- `docs/nn_architecture_refactor_plan.md`: Refactoring plan
- `docs/ember_ml_rnn_structure.md`: RNN structure documentation
- `docs/neuron_model_vs_role.md`: Neuron model vs. role explanation
- `docs/original_ncps_structure.md`: Original NCPS structure

## 2. Class Hierarchy

```
Module (BaseModule)
├── ModuleCell
│   ├── RNNCell
│   ├── LSTMCell
│   ├── GRUCell
│   ├── CfCCell
│   └── ModuleWiredCell
│       ├── LTCCell
│       └── WiredCfCCell
│
NeuronMap
├── NCPMap
├── FullyConnectedMap
└── RandomMap
```

## 3. Key Classes and Their Parameters

### NeuronMap Base Class
```python
def __init__(
    self, 
    units: int, 
    output_dim: Optional[int] = None, 
    input_dim: Optional[int] = None,
    sparsity_level: float = 0.5, 
    seed: Optional[int] = None
)
```

### NCPMap Class
```python
def __init__(
    self,
    inter_neurons: int,
    command_neurons: int,
    motor_neurons: int,
    sensory_neurons: int = 0,
    sparsity_level: float = 0.5,
    seed: Optional[int] = None,
    sensory_to_inter_sparsity: Optional[float] = None,
    sensory_to_motor_sparsity: Optional[float] = None,
    inter_to_inter_sparsity: Optional[float] = None,
    inter_to_motor_sparsity: Optional[float] = None,
    motor_to_motor_sparsity: Optional[float] = None,
    motor_to_inter_sparsity: Optional[float] = None,
    units: Optional[int] = None,
    output_dim: Optional[int] = None,
    input_dim: Optional[int] = None,
)
```

### ModuleCell Base Class
```python
def __init__(
    self,
    input_size: int,
    hidden_size: int,
    activation: str = "tanh",
    use_bias: bool = True,
    **kwargs
)
```

### ModuleWiredCell Base Class
```python
def __init__(
    self,
    neuron_map: Union[NeuronMap, Dict[str, Any]],
    mode: str = "default",
    **kwargs
)
```

### LTCCell Class
```python
def __init__(
    self,
    neuron_map: NeuronMap,
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6,
    epsilon=1e-8,
    implicit_param_constraints=False,
    **kwargs
)
```

### CfCCell Class
```python
def __init__(
    self,
    input_size: int,
    hidden_size: int,
    mode: str = "default",
    sparsity_mask = None,
    time_scale_factor: float = 1.0,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    recurrent_initializer: str = "orthogonal",
    bias_initializer: str = "zeros",
    mixed_memory: bool = False,
    **kwargs
)
```

### WiredCfCCell Class
```python
def __init__(
    self,
    neuron_map: NeuronMap,
    time_scale_factor: float = 1.0,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    recurrent_initializer: str = "orthogonal",
    bias_initializer: str = "zeros",
    mixed_memory: bool = False,
    input_size: Optional[int] = None,
    mode: str = "default",
    **kwargs
)
```

### CfC Layer Class
```python
def __init__(
    self,
    cell_or_map,
    return_sequences: bool = False,
    return_state: bool = False,
    go_backwards: bool = False,
    mixed_memory: bool = False,
    **kwargs
)
```

### LTC Layer Class
```python
def __init__(
    self,
    neuron_map,
    return_sequences: bool = True,
    return_state: bool = False,
    **kwargs
)
```

## 4. Current Usage Patterns

### Creating and Using LTC
```python
# Create a NeuronMap
neuron_map = NCPMap(
    inter_neurons=40,
    command_neurons=20,
    motor_neurons=10,
    sensory_neurons=8,
    sparsity_level=0.5,
    seed=42
)

# Create an LTC layer
ltc_layer = LTC(
    neuron_map=neuron_map,
    return_sequences=True,
    return_state=False
)

# Forward pass
x = tensor.random_normal((32, 10, 8))
y = ltc_layer(x)
```

### Creating and Using CfC
```python
# Option 1: Using a cell
cell = CfCCell(
    input_size=8,
    hidden_size=64,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    mixed_memory=False
)

cfc_layer = CfC(
    cell_or_map=cell,
    return_sequences=True,
    return_state=False
)

# Option 2: Using a NeuronMap
neuron_map = NCPMap(
    inter_neurons=40,
    command_neurons=20,
    motor_neurons=10,
    sensory_neurons=8,
    sparsity_level=0.5,
    seed=42
)

cfc_layer = CfC(
    cell_or_map=neuron_map,
    return_sequences=True,
    return_state=False,
    mixed_memory=False
)

# Forward pass
x = tensor.random_normal((32, 10, 8))
y = cfc_layer(x)
```

## 5. Key Architectural Issues

1. **Hybrid Cell/Map Approach**: CfC layer accepts either cells or maps, creating inconsistency
2. **Duplicated Parameters**: Cell-specific parameters (time_scale_factor, activation) duplicated across implementations
3. **Inconsistent Initialization**: Some components use build-at-init, others use deferred build
4. **Parameter Fragmentation**: Related parameters spread across different classes
5. **Inheritance Complexity**: Deep inheritance chain with overlapping responsibilities