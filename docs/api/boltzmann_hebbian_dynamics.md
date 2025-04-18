# Boltzmann-Hebbian Dynamics

## Overview

The Boltzmann-Hebbian Dynamics framework represents a novel approach to neural network learning that balances stochastic exploration with deterministic stability. This framework creates a natural tension between chaos and order, allowing neural networks to explore widely during early training while gradually stabilizing around strong connections as learning progresses.

## Core Principles

1. **Stochastic Exploration**: Boltzmann distribution enables probabilistic activation of neurons
2. **Deterministic Stability**: Hebbian learning strengthens connections between frequently co-firing neurons
3. **Dynamic Balance**: The system naturally shifts from exploration to stability over time
4. **Continuous Adaptation**: Even in mature networks, some level of exploration remains possible

## Mathematical Foundation

### Standard Boltzmann Distribution

The standard Boltzmann distribution determines the probability of a neuron activating based on its energy:

$$P(x) = \frac{e^{-\frac{E(x)}{kT}}}{\sum e^{-\frac{E(x^{\prime})}{kT}}}$$

Where:
- $E(x)$ is the energy of the connection (lower energy = more likely to activate)
- $kT$ is the temperature parameter (higher temperature = more exploration)

### Hebbian Learning

Hebbian learning strengthens connections between neurons that frequently fire together:

$$W_{ij} = W_{ij} + \eta \cdot (A_i \cdot A_j)$$

Where:
- $W_{ij}$ is the weight between neurons $i$ and $j$
- $\eta$ is the learning rate
- $A_i$ and $A_j$ are the activations of neurons $i$ and $j$

### Modified Boltzmann-Hebbian Equation

The key innovation is modifying the energy term in the Boltzmann equation to incorporate Hebbian weights:

$$E^{\prime}(x) = E(x) - \beta \cdot W(x)$$

Where:
- $\beta$ is a scaling factor that determines how much the Hebbian weight influences energy
- $W(x)$ is the Hebbian weight of the connection
- $E^{\prime}(x)$ is the modified energy term

The updated Boltzmann equation becomes:

$$P(x) = \frac{e^{-\frac{E(x) - \beta \cdot W(x)}{kT}}}{\sum e^{-\frac{E(x^{\prime}) - \beta \cdot W(x^{\prime})}{kT}}}$$

## Developmental Stages

The Boltzmann-Hebbian Dynamics framework naturally creates a developmental progression in neural networks:

### 1. Early Development: High Exploration

- High temperature ($T$) and low Hebbian influence ($\beta$)
- Network explores widely, testing many potential connections
- Forms weak connections between neurons that fire together
- Behavior is highly stochastic and unpredictable

### 2. Middle Development: Balancing Chaos and Order

- Moderate temperature ($T$) and increasing Hebbian influence ($\beta$)
- Network prioritizes stronger connections (learned pathways)
- Occasionally explores weaker connections (discovering new patterns)
- Behavior becomes more predictable but maintains flexibility

### 3. Mature Development: Stabilized Learning

- Low temperature ($T$) and high Hebbian influence ($\beta$)
- Network relies on well-learned pathways for most processing
- Rarely explores, but retains flexibility for novel situations
- Behavior is mostly deterministic but can adapt when necessary

## Implementation Design

### BoltzmannHebbianLayer Class

```python
class BoltzmannHebbianLayer(Module):
    def __init__(
        self,
        units: int,
        input_dim: int,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.999,
        hebbian_factor: float = 0.1,
        hebbian_growth: float = 1.001,
        learning_rate: float = 0.01,
        **kwargs
    ):
        """
        Initialize a Boltzmann-Hebbian Layer.
        
        Args:
            units: Number of neurons in the layer
            input_dim: Dimension of input features
            initial_temperature: Initial temperature for Boltzmann distribution
            min_temperature: Minimum temperature (prevents complete determinism)
            temperature_decay: Rate at which temperature decreases
            hebbian_factor: Initial scaling factor for Hebbian weights
            hebbian_growth: Rate at which Hebbian influence increases
            learning_rate: Learning rate for Hebbian updates
        """
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.hebbian_factor = hebbian_factor
        self.hebbian_growth = hebbian_growth
        self.learning_rate = learning_rate
        
        # Current temperature and hebbian factor
        self.temperature = initial_temperature
        self.current_hebbian_factor = hebbian_factor
        
        # Initialize weights
        self.weights = Parameter(tensor.random_normal((input_dim, units)))
        
        # Initialize Hebbian weights (separate from main weights)
        self.hebbian_weights = Parameter(tensor.zeros((input_dim, units)))
        
        # Previous activations for Hebbian updates
        self.prev_input = None
        self.prev_output = None
        
        # Training step counter
        self.steps = 0
    
    def update_parameters(self):
        """Update temperature and Hebbian factor based on training progress."""
        # Decay temperature
        self.temperature = ops.maximum(
            ops.multiply(self.temperature, tensor.convert_to_tensor(self.temperature_decay)),
            tensor.convert_to_tensor(self.min_temperature)
        )
        
        # Increase Hebbian factor
        self.current_hebbian_factor = ops.multiply(
            self.current_hebbian_factor,
            tensor.convert_to_tensor(self.hebbian_growth)
        )
        
        # Increment step counter
        self.steps += 1
    
    def update_hebbian_weights(self, inputs, outputs):
        """Update Hebbian weights based on co-activation."""
        # Skip if no previous activations
        if self.prev_input is None or self.prev_output is None:
            self.prev_input = inputs
            self.prev_output = outputs
            return
        
        # Compute outer product of input and output activations
        for i in range(tensor.shape(inputs)[0]):  # For each batch item
            input_i = tensor.reshape(inputs[i], (-1, 1))  # [input_dim, 1]
            output_i = tensor.reshape(outputs[i], (1, -1))  # [1, units]
            
            # Outer product: [input_dim, units]
            outer_product = ops.matmul(input_i, output_i)
            
            # Update Hebbian weights
            hebbian_update = ops.multiply(
                tensor.convert_to_tensor(self.learning_rate),
                outer_product
            )
            self.hebbian_weights = ops.add(self.hebbian_weights, hebbian_update)
        
        # Store current activations for next update
        self.prev_input = inputs
        self.prev_output = outputs
    
    def forward(self, inputs, training=False):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_dim]
            training: Whether the layer is in training mode
            
        Returns:
            Output tensor of shape [batch_size, units]
        """
        # Compute raw activations (energy)
        raw_activations = ops.matmul(inputs, self.weights)  # [batch_size, units]
        
        # Modify energy with Hebbian weights
        hebbian_activations = ops.matmul(inputs, self.hebbian_weights)  # [batch_size, units]
        modified_energy = ops.subtract(
            raw_activations,
            ops.multiply(
                tensor.convert_to_tensor(self.current_hebbian_factor),
                hebbian_activations
            )
        )
        
        # Apply Boltzmann distribution
        boltzmann_weights = ops.exp(ops.divide(
            ops.negative(modified_energy),
            tensor.convert_to_tensor(self.temperature)
        ))
        boltzmann_weights = ops.divide(
            boltzmann_weights,
            ops.sum(boltzmann_weights, axis=-1, keepdims=True)
        )
        
        # In training mode, use stochastic sampling
        if training:
            # Sample from Boltzmann distribution
            # Note: This is a simplified version; actual sampling would be more complex
            random_uniform = tensor.random_uniform(tensor.shape(boltzmann_weights))
            cumulative_probs = ops.cumsum(boltzmann_weights, axis=-1)
            samples = tensor.cast(
                ops.less(random_uniform, cumulative_probs),
                dtype=tensor.float32
            )
            # Take the first true value in each row
            diff = ops.subtract(
                samples,
                tensor.concat([tensor.zeros_like(samples[:, :1]), samples[:, :-1]], axis=1)
            )
            outputs = ops.multiply(diff, tensor.convert_to_tensor(1.0))
        else:
            # In inference mode, use the most probable activation
            outputs = boltzmann_weights
        
        # Update parameters and weights if in training mode
        if training:
            self.update_parameters()
            self.update_hebbian_weights(inputs, outputs)
        
        return outputs
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'initial_temperature': self.initial_temperature,
            'min_temperature': self.min_temperature,
            'temperature_decay': self.temperature_decay,
            'hebbian_factor': self.hebbian_factor,
            'hebbian_growth': self.hebbian_growth,
            'learning_rate': self.learning_rate
        })
        return config
```

### BoltzmannHebbianNetwork Class

```python
class BoltzmannHebbianNetwork(Module):
    def __init__(
        self,
        layer_units: List[int],
        input_dim: int,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.999,
        hebbian_factor: float = 0.1,
        hebbian_growth: float = 1.001,
        learning_rate: float = 0.01,
        **kwargs
    ):
        """
        Initialize a Boltzmann-Hebbian Network.
        
        Args:
            layer_units: List of units for each layer
            input_dim: Dimension of input features
            initial_temperature: Initial temperature for Boltzmann distribution
            min_temperature: Minimum temperature (prevents complete determinism)
            temperature_decay: Rate at which temperature decreases
            hebbian_factor: Initial scaling factor for Hebbian weights
            hebbian_growth: Rate at which Hebbian influence increases
            learning_rate: Learning rate for Hebbian updates
        """
        super().__init__(**kwargs)
        self.layer_units = layer_units
        self.input_dim = input_dim
        
        # Create layers
        self.layers = []
        prev_dim = input_dim
        for i, units in enumerate(layer_units):
            # Temperature decreases for higher layers
            layer_temp = initial_temperature * (0.9 ** i)
            
            layer = BoltzmannHebbianLayer(
                units=units,
                input_dim=prev_dim,
                initial_temperature=layer_temp,
                min_temperature=min_temperature,
                temperature_decay=temperature_decay,
                hebbian_factor=hebbian_factor,
                hebbian_growth=hebbian_growth,
                learning_rate=learning_rate
            )
            self.layers.append(layer)
            prev_dim = units
    
    def forward(self, inputs, training=False):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_dim]
            training: Whether the network is in training mode
            
        Returns:
            Output tensor of shape [batch_size, layer_units[-1]]
        """
        x = inputs
        layer_outputs = []
        
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x, training=training)
            layer_outputs.append(x)
        
        return {
            'output': x,
            'layer_outputs': layer_outputs
        }
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'layer_units': self.layer_units,
            'input_dim': self.input_dim
        })
        return config
```

## Integration with Ember ML

The Boltzmann-Hebbian Dynamics framework can be integrated with other Ember ML components:

```python
class BoltzmannHebbianModel(Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        layer_units: List[int] = [64, 32, 16],
        initial_temperature: float = 1.0,
        **kwargs
    ):
        """
        Initialize Boltzmann-Hebbian Model.
        
        Args:
            input_dim: Input dimension
            embedding_dim: Embedding dimension
            layer_units: Units for each Boltzmann-Hebbian layer
            initial_temperature: Initial temperature for Boltzmann distribution
        """
        super().__init__(**kwargs)
        
        # Embedding layer
        self.embedding = Dense(embedding_dim)
        
        # Boltzmann-Hebbian network
        self.bh_network = BoltzmannHebbianNetwork(
            layer_units=layer_units,
            input_dim=embedding_dim,
            initial_temperature=initial_temperature
        )
        
        # Output layer
        self.output_layer = Dense(1)
    
    def forward(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether the model is in training mode
        """
        # Generate embeddings
        embeddings = self.embedding(inputs)
        
        # Process through Boltzmann-Hebbian network
        bh_outputs = self.bh_network(embeddings, training=training)
        
        # Generate final output
        output = self.output_layer(bh_outputs['output'])
        
        return {
            'embeddings': embeddings,
            'bh_outputs': bh_outputs,
            'output': output
        }
```

## Applications

### Anomaly Detection

The Boltzmann-Hebbian Dynamics framework is particularly well-suited for anomaly detection:

1. **Early Training**:
   - Network explores all possible connections, treating all logs as equally probable
   - Hebbian reinforcement strengthens connections for frequently co-occurring patterns

2. **Later Training**:
   - Strong patterns dominate (e.g., routine logs)
   - Rare patterns (anomalies) still activate probabilistically, ensuring the network can discover them

3. **Inference**:
   - Network prioritizes learned pathways but remains flexible enough to adapt to new anomalies
   - Anomalies can be detected by measuring the "surprise" (energy difference) between expected and actual activations

### Example: Log Anomaly Detection

```python
# Example: Log anomaly detection with Boltzmann-Hebbian Dynamics
def detect_anomalies(logs, model, threshold=0.9):
    """
    Detect anomalies in logs using Boltzmann-Hebbian model.
    
    Args:
        logs: Log entries to analyze
        model: Trained Boltzmann-Hebbian model
        threshold: Anomaly threshold (higher = more sensitive)
        
    Returns:
        List of anomalies with their scores
    """
    # Preprocess logs
    log_embeddings = preprocess_logs(logs)
    
    # Get model predictions
    predictions = model(log_embeddings, training=False)
    
    # Compute energy (negative log probability)
    energy = ops.negative(ops.log(predictions['output']))
    
    # Normalize energy to [0, 1]
    normalized_energy = ops.divide(
        energy,
        ops.maximum(ops.max(energy), tensor.convert_to_tensor(1.0))
    )
    
    # Detect anomalies
    anomalies = []
    for i, (log, score) in enumerate(zip(logs, normalized_energy)):
        if score > threshold:
            anomalies.append({
                'log': log,
                'score': score,
                'index': i
            })
    
    return anomalies
```

### Reinforcement Learning

The framework can also be applied to reinforcement learning:

1. **Exploration Phase**:
   - High temperature encourages exploration of the state-action space
   - Hebbian connections begin to form for successful action sequences

2. **Exploitation Phase**:
   - Lower temperature and stronger Hebbian connections favor actions that led to rewards
   - Some exploration continues, allowing discovery of better strategies

3. **Adaptive Behavior**:
   - If the environment changes, the system can adapt by temporarily increasing temperature
   - Hebbian connections will gradually shift to reflect the new optimal policy

## Visualization and Analysis

### Temperature and Hebbian Factor Evolution

```python
def visualize_parameter_evolution(model, steps=1000):
    """Visualize the evolution of temperature and Hebbian factor over training."""
    import matplotlib.pyplot as plt
    
    # Initialize arrays to store values
    temperatures = []
    hebbian_factors = []
    
    # Create a copy of the model's initial parameters
    layer = model.layers[0]
    temp = layer.temperature
    hf = layer.current_hebbian_factor
    
    # Simulate parameter evolution
    for _ in range(steps):
        temperatures.append(temp)
        hebbian_factors.append(hf)
        
        # Update parameters
        temp = max(temp * layer.temperature_decay, layer.min_temperature)
        hf = hf * layer.hebbian_growth
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot temperature
    plt.subplot(1, 2, 1)
    plt.plot(temperatures)
    plt.title("Temperature Evolution")
    plt.xlabel("Training Steps")
    plt.ylabel("Temperature")
    plt.grid(True)
    
    # Plot Hebbian factor
    plt.subplot(1, 2, 2)
    plt.plot(hebbian_factors)
    plt.title("Hebbian Factor Evolution")
    plt.xlabel("Training Steps")
    plt.ylabel("Hebbian Factor")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Activation Probability Distribution

```python
def visualize_activation_distribution(model, inputs, steps=[0, 100, 500, 1000]):
    """Visualize how activation probabilities evolve during training."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a copy of the model for simulation
    sim_model = copy.deepcopy(model)
    
    # Initialize figure
    plt.figure(figsize=(15, 10))
    
    # For each step, compute and visualize activation probabilities
    for i, step in enumerate(steps):
        # Fast-forward model parameters to desired step
        for layer in sim_model.layers:
            layer.temperature = layer.initial_temperature * (layer.temperature_decay ** step)
            layer.current_hebbian_factor = layer.hebbian_factor * (layer.hebbian_growth ** step)
        
        # Get model predictions
        predictions = sim_model(inputs, training=False)
        
        # Get activation probabilities from first layer
        probs = predictions['layer_outputs'][0].numpy()
        
        # Plot histogram of probabilities
        plt.subplot(2, 2, i+1)
        plt.hist(probs.flatten(), bins=50, alpha=0.7)
        plt.title(f"Step {step}: Activation Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Biological Parallels

The Boltzmann-Hebbian Dynamics framework has strong parallels to biological neural systems:

1. **Synaptic Plasticity**:
   - Hebbian learning mirrors how synapses strengthen with repeated use
   - The framework's balance between stability and flexibility resembles how the brain maintains established pathways while adapting to new information

2. **Developmental Stages**:
   - The progression from high exploration to stable behavior mirrors human cognitive development
   - Infants and children exhibit more exploratory behavior, while adults rely more on established patterns

3. **Neurochemical Modulation**:
   - The temperature parameter is analogous to how neurotransmitters like dopamine modulate neural activity
   - Higher dopamine levels (analogous to higher temperature) promote exploratory behavior

## Conclusion

The Boltzmann-Hebbian Dynamics framework represents a powerful approach to neural network learning that balances exploration and stability. By modifying the Boltzmann energy equation to incorporate Hebbian weights, this framework creates a system that:

1. Explores widely during early training
2. Gradually stabilizes around strong connections
3. Maintains some capacity for exploration even in mature networks
4. Adapts naturally to changing environments

This approach has applications in anomaly detection, reinforcement learning, and any domain where balancing learned patterns with novel exploration is crucial. The framework's biological parallels suggest it may capture important aspects of how natural intelligence develops and adapts over time.

## References

1. Hebb, D.O. (1949). The Organization of Behavior: A Neuropsychological Theory.
2. Hinton, G.E., & Sejnowski, T.J. (1986). Learning and Relearning in Boltzmann Machines.
3. Friston, K. (2010). The free-energy principle: a unified brain theory?
4. Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2020). Liquid Time-constant Networks.

## See Also

- [Grand Unified Cognitive Equation (GUCE)](guce.md): A theoretical framework that incorporates Boltzmann-Hebbian dynamics
- [Abacus Neural Architecture](abacus_neural_architecture.md): A neural architecture that can be combined with Boltzmann-Hebbian dynamics
- [Training Module](training.md): Documentation on training and evaluation in Ember ML