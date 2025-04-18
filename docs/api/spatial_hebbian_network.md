# Spatial Hebbian Network

## Overview

The Spatial Hebbian Network represents a novel approach to neural network architecture that more closely mimics the biological brain's development and organization. Unlike traditional layered networks, this architecture organizes neurons in a three-dimensional space where connectivity is determined by spatial proximity and co-activation patterns. This approach allows for more organic growth, localized specialization, and emergent hierarchical structures.

## Core Principles

1. **Spatial Organization**: Neurons exist in a 3D coordinate space rather than discrete layers
2. **Proximity-Based Connectivity**: Neurons connect primarily to nearby neurons
3. **Hebbian Learning with Spatial Constraints**: Connections strengthen between co-firing neurons that are spatially proximate
4. **Boltzmann Exploration in Spatial Context**: Stochastic activation spreads to nearby neurons
5. **Organic Growth**: The network expands from a small "seed" region as needed

## Mathematical Foundation

### Spatial Representation

Each neuron is assigned a position in 3D space:

$$p_i = (x_i, y_i, z_i)$$

Where:
- $p_i$ is the position of neuron $i$
- $(x_i, y_i, z_i)$ are the coordinates in 3D space

### Spatial Distance

The distance between neurons determines their connectivity potential:

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}$$

### Proximity-Based Connection Probability

The probability of a connection existing between two neurons decreases with distance:

$$P_{conn}(i, j) = e^{-\lambda \cdot d_{ij}}$$

Where:
- $\lambda$ is a distance decay parameter

### Spatially-Constrained Hebbian Learning

Connection weights are updated based on co-activation and spatial proximity:

$$\Delta W_{ij} = \eta \cdot A_i \cdot A_j \cdot e^{-\lambda \cdot d_{ij}}$$

Where:
- $\eta$ is the learning rate
- $A_i$ and $A_j$ are the activations of neurons $i$ and $j$
- $e^{-\lambda \cdot d_{ij}}$ is the spatial constraint factor

### Spatial Boltzmann Activation

The probability of a neuron activating is influenced by both its energy and the activation of nearby neurons:

$$P(A_i = 1) = \frac{e^{-\frac{E_i - \beta \sum_j W_{ij} \cdot A_j \cdot e^{-\lambda \cdot d_{ij}}}{kT}}}{\sum e^{-\frac{E_i - \beta \sum_j W_{ij} \cdot A_j \cdot e^{-\lambda \cdot d_{ij}}}{kT}}}$$

Where:
- $E_i$ is the energy of neuron $i$
- $\beta$ is the Hebbian influence factor
- $kT$ is the temperature parameter
- $\sum_j W_{ij} \cdot A_j \cdot e^{-\lambda \cdot d_{ij}}$ represents the weighted influence of nearby active neurons

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Spatial Hebbian Network                         │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       3D Neuron Space                            │
│                                                                  │
│    ●       ●                      ●                              │
│        ●       ●             ●        ●                          │
│  ●         ●        ●    ●       ●       ●                       │
│     ●  ●      ●  ●     ●    ●  ●    ●                           │
│  ●     ●   ●     ●  ●     ●    ●      ●                         │
│    ●       ●  ●      ●       ●     ●                            │
│       ●        ●         ●      ●                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Spatial Growth Manager                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ember ML Integration Layer                    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Design

### SpatialNeuron Class

```python
class SpatialNeuron(Module):
    def __init__(
        self,
        position: Tuple[float, float, float],
        connection_radius: float = 2.0,
        activation_function: str = 'tanh',
        **kwargs
    ):
        """
        Initialize a Spatial Neuron.
        
        Args:
            position: 3D coordinates (x, y, z)
            connection_radius: Maximum distance for connections
            activation_function: Activation function to use
        """
        super().__init__(**kwargs)
        self.position = position
        self.connection_radius = connection_radius
        self.activation_function = get_activation(activation_function)
        
        # Neuron state
        self.activation = 0.0
        self.energy = 0.0
        
        # Connections (will be populated by the network)
        self.connections = {}  # {neuron_id: weight}
    
    def compute_distance(self, other_neuron):
        """Compute Euclidean distance to another neuron."""
        x1, y1, z1 = self.position
        x2, y2, z2 = other_neuron.position
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    def can_connect_to(self, other_neuron):
        """Determine if this neuron can connect to another based on distance."""
        return self.compute_distance(other_neuron) <= self.connection_radius
    
    def add_connection(self, neuron_id, weight=0.1):
        """Add or update a connection to another neuron."""
        self.connections[neuron_id] = weight
    
    def update_connection(self, neuron_id, delta_weight):
        """Update connection weight using Hebbian learning."""
        if neuron_id in self.connections:
            self.connections[neuron_id] += delta_weight
            # Ensure weight stays in reasonable range
            self.connections[neuron_id] = max(0.0, min(1.0, self.connections[neuron_id]))
    
    def compute_input(self, neuron_activations):
        """Compute weighted input from connected neurons."""
        total_input = 0.0
        for neuron_id, weight in self.connections.items():
            if neuron_id in neuron_activations:
                total_input += weight * neuron_activations[neuron_id]
        return total_input
    
    def update_activation(self, input_value):
        """Update neuron activation based on input."""
        self.activation = self.activation_function(input_value)
        return self.activation
    
    def compute_energy(self):
        """Compute energy level of the neuron."""
        # Simple energy model: higher activation = lower energy
        self.energy = -self.activation
        return self.energy
```

### SpatialHebbianNetwork Class

```python
class SpatialHebbianNetwork(Module):
    def __init__(
        self,
        initial_neurons: int = 10,
        max_neurons: int = 1000,
        space_dimensions: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        connection_radius: float = 2.0,
        distance_decay: float = 0.5,
        learning_rate: float = 0.01,
        temperature: float = 1.0,
        hebbian_factor: float = 0.5,
        growth_threshold: float = 0.8,
        **kwargs
    ):
        """
        Initialize a Spatial Hebbian Network.
        
        Args:
            initial_neurons: Number of neurons to start with
            max_neurons: Maximum number of neurons allowed
            space_dimensions: Size of 3D space (x, y, z)
            connection_radius: Maximum distance for connections
            distance_decay: Decay factor for distance in connection probability
            learning_rate: Learning rate for Hebbian updates
            temperature: Temperature for Boltzmann activation
            hebbian_factor: Factor for Hebbian influence in activation
            growth_threshold: Activation threshold for adding new neurons
        """
        super().__init__(**kwargs)
        self.initial_neurons = initial_neurons
        self.max_neurons = max_neurons
        self.space_dimensions = space_dimensions
        self.connection_radius = connection_radius
        self.distance_decay = distance_decay
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.hebbian_factor = hebbian_factor
        self.growth_threshold = growth_threshold
        
        # Initialize neurons
        self.neurons = {}  # {neuron_id: SpatialNeuron}
        self.neuron_activations = {}  # {neuron_id: activation}
        
        # Create initial "seed" neurons at the center of the space
        self._create_seed_neurons()
        
        # Input and output mappings
        self.input_neurons = []  # List of neuron IDs for input
        self.output_neurons = []  # List of neuron IDs for output
    
    def _create_seed_neurons(self):
        """Create initial seed neurons at the center of the space."""
        center = (
            self.space_dimensions[0] / 2,
            self.space_dimensions[1] / 2,
            self.space_dimensions[2] / 2
        )
        
        # Create neurons in a small sphere around the center
        for i in range(self.initial_neurons):
            # Random position in a small sphere
            angle1 = 2 * np.pi * np.random.random()
            angle2 = np.pi * np.random.random()
            radius = 0.5 * np.random.random()  # Small radius for seed cluster
            
            x = center[0] + radius * np.sin(angle2) * np.cos(angle1)
            y = center[1] + radius * np.sin(angle2) * np.sin(angle1)
            z = center[2] + radius * np.cos(angle2)
            
            # Create neuron
            neuron = SpatialNeuron(
                position=(x, y, z),
                connection_radius=self.connection_radius
            )
            
            # Add to network
            neuron_id = len(self.neurons)
            self.neurons[neuron_id] = neuron
            self.neuron_activations[neuron_id] = 0.0
            
            # First few neurons are input neurons
            if i < self.initial_neurons // 3:
                self.input_neurons.append(neuron_id)
            # Last few neurons are output neurons
            elif i >= 2 * self.initial_neurons // 3:
                self.output_neurons.append(neuron_id)
        
        # Create initial connections
        self._create_initial_connections()
