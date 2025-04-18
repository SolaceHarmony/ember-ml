# Quantum-Inspired Neural Networks (nn.modules.rnn)

The `ember_ml.nn.modules.rnn` package includes quantum-inspired neural network modules that combine principles from quantum computing with classical neural networks. These modules are designed to leverage quantum-inspired dynamics for enhanced temporal processing capabilities.

## Importing

```python
from ember_ml.nn.modules.rnn import LQNet, CTRQNet
```

## Core Concepts

Quantum-inspired neural networks use classical hardware to emulate certain aspects of quantum computing, such as:

1. **Superposition**: Representing multiple states simultaneously
2. **Entanglement**: Creating correlations between different parts of the system
3. **Interference**: Allowing probability amplitudes to interfere constructively or destructively

These networks use stochastic-quantum mapping to emulate quantum effects and b-symplectic structures to preserve geometric integrity during evolution.

## Liquid Quantum Neural Network (LQNet)

`LQNet` is a quantum-inspired recurrent neural network that combines liquid neural networks with quantum computing concepts using classical hardware.

### Basic Usage

```python
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import LQNet
from ember_ml.nn import tensor

# Create a neuron map for connectivity
neuron_map = NCPMap(
    inter_neurons=32,
    command_neurons=16,
    motor_neurons=8,
    sensory_neurons=10,
    seed=42
)

# Create LQNet model
lqnet = LQNet(
    neuron_map=neuron_map,
    nu_0=1.0,
    beta=0.1,
    noise_scale=0.05,
    return_sequences=True,
    return_state=False,
    batch_first=True
)

# Forward pass
inputs = tensor.random_normal((32, 100, 10))  # (batch_size, seq_length, input_dim)
outputs = lqnet(inputs)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `neuron_map` | `NeuronMap` | NeuronMap instance defining connectivity |
| `nu_0` | `float` | Base viscosity parameter (default: 1.0) |
| `beta` | `float` | Energy scaling parameter (default: 0.1) |
| `noise_scale` | `float` | Scale of the stochastic noise (default: 0.1) |
| `return_sequences` | `bool` | Whether to return the full sequence or just the last output (default: True) |
| `return_state` | `bool` | Whether to return the final state (default: False) |
| `batch_first` | `bool` | Whether the batch or time dimension is the first (0-th) dimension (default: True) |

### Methods

| Method | Description |
|--------|-------------|
| `forward(inputs, initial_state=None, time_deltas=None)` | Forward pass through the layer |
| `reset_state(batch_size=1)` | Reset the layer state |

### Key Components

#### B-Symplectic Structure

LQNet uses b-symplectic structures to preserve geometric integrity during evolution. This is implemented through b-Poisson brackets that maintain quantum correlations and support adaptive feedback.

#### Stochastic-Quantum Mapping

The stochastic-quantum mapping emulates quantum effects on classical hardware. It uses stochastic differential equations to model quantum superposition, interference, and entanglement.

#### Boltzmann-Modulated Viscosity

The viscosity of the system is modulated by a Boltzmann factor based on the energy of the state. This allows the system to adapt its dynamics based on the current state.

## Continuous-Time Recurrent Quantum Neural Network (CTRQNet)

`CTRQNet` extends LQNet with continuous-time dynamics and enhanced quantum-inspired features. It adds time-scale modulation and harmonic embedding for improved temporal processing.

### Basic Usage

```python
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import CTRQNet
from ember_ml.nn import tensor

# Create a neuron map for connectivity
neuron_map = NCPMap(
    inter_neurons=32,
    command_neurons=16,
    motor_neurons=8,
    sensory_neurons=10,
    seed=42
)

# Create CTRQNet model
ctrqnet = CTRQNet(
    neuron_map=neuron_map,
    nu_0=1.0,
    beta=0.1,
    noise_scale=0.05,
    time_scale_factor=1.0,
    use_harmonic_embedding=True,
    return_sequences=True,
    return_state=False,
    batch_first=True
)

# Forward pass
inputs = tensor.random_normal((32, 100, 10))  # (batch_size, seq_length, input_dim)
outputs = ctrqnet(inputs)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `neuron_map` | `NeuronMap` | NeuronMap instance defining connectivity |
| `nu_0` | `float` | Base viscosity parameter (default: 1.0) |
| `beta` | `float` | Energy scaling parameter (default: 0.1) |
| `noise_scale` | `float` | Scale of the stochastic noise (default: 0.1) |
| `time_scale_factor` | `float` | Factor to scale the time constant (default: 1.0) |
| `use_harmonic_embedding` | `bool` | Whether to use harmonic embedding (default: True) |
| `return_sequences` | `bool` | Whether to return the full sequence or just the last output (default: True) |
| `return_state` | `bool` | Whether to return the final state (default: False) |
| `batch_first` | `bool` | Whether the batch or time dimension is the first (0-th) dimension (default: True) |

### Methods

| Method | Description |
|--------|-------------|
| `forward(inputs, initial_state=None, time_deltas=None)` | Forward pass through the layer |
| `reset_state(batch_size=1)` | Reset the layer state |

### Key Components

#### Harmonic Embedding

CTRQNet can use harmonic embedding to convert static token embeddings into time-evolving waveforms. Each token is encoded as a harmonic oscillation with learned parameters.

#### Time-Scale Modulation

The time-scale parameter modulates the decay rate of the hidden state, allowing the network to adapt its temporal dynamics based on the input.

## Integration with NeuronMap

Both LQNet and CTRQNet use NeuronMap for defining connectivity patterns. This allows for flexible and customizable network architectures.

```python
from ember_ml.nn.modules.wiring import NCPMap, RandomMap, FullyConnectedMap
from ember_ml.nn.modules.rnn import LQNet

# Using NCPMap
ncp_map = NCPMap(
    inter_neurons=32,
    command_neurons=16,
    motor_neurons=8,
    sensory_neurons=10,
    seed=42
)
lqnet_ncp = LQNet(neuron_map=ncp_map)

# Using RandomMap
random_map = RandomMap(
    units=56,  # 32 + 16 + 8
    output_size=8,
    input_size=10,
    sparsity_level=0.5,
    seed=42
)
lqnet_random = LQNet(neuron_map=random_map)

# Using FullyConnectedMap
fc_map = FullyConnectedMap(
    units=56,
    output_size=8,
    input_size=10
)
lqnet_fc = LQNet(neuron_map=fc_map)
```

## Example: Time Series Prediction

```python
from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules.wiring import NCPMap
from ember_ml.nn.modules.rnn import CTRQNet

# Set the backend to MLX for better performance
ops.set_backend('mlx')

# Generate sine wave data
def generate_sine_wave(seq_length, num_samples, freq=0.1, noise=0.1):
    # Create time points from 0 to 10
    x = tensor.linspace(0.0, 10.0, seq_length)
    data = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random phase between 0 and 2π
        # Use a fixed phase for each sample to avoid random_uniform issues
        phase = 2.0 * ops.pi * (float(_) / float(num_samples))
        
        # Generate sine wave: sin(2π * freq * x + phase)
        sine_wave = ops.sin(ops.add(
            ops.multiply(2.0 * ops.pi * freq, x),
            phase
        ))
        
        # Add noise to create noisy sine wave
        noise_tensor = tensor.random_normal(tensor.shape(sine_wave), stddev=noise)
        noisy_sine = ops.add(sine_wave, noise_tensor)
        
        # Reshape to add feature dimension
        noisy_sine_reshaped = tensor.reshape(noisy_sine, (-1, 1))
        sine_wave_reshaped = tensor.reshape(sine_wave, (-1, 1))
        
        # Use the noisy sine as input and clean sine as target
        data.append(noisy_sine_reshaped)
        targets.append(sine_wave_reshaped)
    
    # Stack along batch dimension
    return tensor.stack(data), tensor.stack(targets)

# Generate data
seq_length = 100
num_samples = 32
input_dim = 1
hidden_dim = 32

X, y = generate_sine_wave(seq_length, num_samples)

# No need to convert to tensors as they are already tensors
X_tensor = X
y_tensor = y

# Create NeuronMap
neuron_map = NCPMap(
    inter_neurons=hidden_dim // 2,
    command_neurons=hidden_dim // 4,
    motor_neurons=hidden_dim // 4,
    sensory_neurons=input_dim,
    seed=42
)

# Create CTRQNet model
ctrqnet = CTRQNet(
    neuron_map=neuron_map,
    nu_0=1.0,
    beta=0.1,
    noise_scale=0.05,
    time_scale_factor=1.0,
    use_harmonic_embedding=True,
    return_sequences=True,
    return_state=False,
    batch_first=True
)

# Forward pass
outputs = ctrqnet(X_tensor)

# Calculate MSE loss using the mse function
mse = ops.mse(y_tensor, outputs)
print(f"MSE: {mse}")
```

## References

1. Barandes, J. A., & Kagan, D. (2020). Measurement and Quantum Dynamics in the Minimal Modal Interpretation of Quantum Theory. Foundations of Physics, 50(10), 1189-1218.
2. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2020). Liquid Time-constant Networks. arXiv preprint arXiv:2006.04439.
3. Markidis, S. (2021). The Old and the New: Can Quantum Computing Become a Reality? ACM Computing Surveys, 54(8), 1-36.

## See Also

- [Neural Network Modules](nn_modules.md): Documentation on base neural network modules
- [RNN Modules Documentation](nn_modules_rnn.md): Documentation on recurrent neural network modules
- [Neuron Maps Documentation](nn_modules_wiring.md): Documentation on neuron maps for defining connectivity patterns