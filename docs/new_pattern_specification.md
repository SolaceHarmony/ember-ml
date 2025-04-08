# New Pattern Specification for Pure Wired Architecture

## 1. Proposed Class Hierarchy

```
Module (BaseModule)
├── RNN Layers (direct implementation)
│   ├── LTC
│   ├── CfC
│   ├── LSTM
│   └── GRU
│
NeuronMap
├── NCPMap (enhanced with cell parameters)
├── FullyConnectedMap
└── RandomMap
```

## 2. Enhanced NCPMap Implementation

```python
class NCPMap(NeuronMap):
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int,
        motor_neurons: int,
        sensory_neurons: int = 0,
        sparsity_level: float = 0.5,
        seed: Optional[int] = None,
        # Add cell-specific parameters
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        mode: str = "default",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        mixed_memory: bool = False,
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        # Keep existing sparsity parameters
        sensory_to_inter_sparsity: Optional[float] = None,
        sensory_to_motor_sparsity: Optional[float] = None,
        inter_to_inter_sparsity: Optional[float] = None,
        inter_to_motor_sparsity: Optional[float] = None,
        motor_to_motor_sparsity: Optional[float] = None,
        motor_to_inter_sparsity: Optional[float] = None,
        units: Optional[int] = None,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
    ):
        # Calculate units if not provided
        if units is None:
            units = inter_neurons + command_neurons + motor_neurons
        
        # Call parent init with basic structural parameters
        super().__init__(
            units=units,
            output_dim=output_dim if output_dim is not None else motor_neurons,
            input_dim=input_dim if input_dim is not None else sensory_neurons if sensory_neurons > 0 else None,
            sparsity_level=sparsity_level,
            seed=seed
        )
        
        # Store NCP-specific structural parameters
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_neurons = sensory_neurons
        
        # Store cell-specific parameters
        self.time_scale_factor = time_scale_factor
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.mode = mode
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.mixed_memory = mixed_memory
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        self.implicit_param_constraints = implicit_param_constraints
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        
        # Store sparsity parameters
        self.sensory_to_inter_sparsity = sensory_to_inter_sparsity or sparsity_level
        self.sensory_to_motor_sparsity = sensory_to_motor_sparsity or sparsity_level
        self.inter_to_inter_sparsity = inter_to_inter_sparsity or sparsity_level
        self.inter_to_motor_sparsity = inter_to_motor_sparsity or sparsity_level
        self.motor_to_motor_sparsity = motor_to_motor_sparsity or sparsity_level
        self.motor_to_inter_sparsity = motor_to_inter_sparsity or sparsity_level
    
    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the NCP map."""
        config = super().get_config()
        config.update({
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons,
            "motor_neurons": self.motor_neurons,
            "sensory_neurons": self.sensory_neurons,
            # Add cell-specific parameters
            "time_scale_factor": self.time_scale_factor,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "mode": self.mode,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "mixed_memory": self.mixed_memory,
            "ode_unfolds": self.ode_unfolds,
            "epsilon": self.epsilon,
            "implicit_param_constraints": self.implicit_param_constraints,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            # Existing sparsity parameters
            "sensory_to_inter_sparsity": self.sensory_to_inter_sparsity,
            "sensory_to_motor_sparsity": self.sensory_to_motor_sparsity,
            "inter_to_inter_sparsity": self.inter_to_inter_sparsity,
            "inter_to_motor_sparsity": self.inter_to_motor_sparsity,
            "motor_to_motor_sparsity": self.motor_to_motor_sparsity,
            "motor_to_inter_sparsity": self.motor_to_inter_sparsity
        })
        return config
```

## 3. Refactored CfC Layer Implementation

```python
class CfC(Module):
    def __init__(
        self,
        neuron_map: NCPMap,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
        
        # Store the neuron map
        self.neuron_map = neuron_map
        
        # Store layer-specific parameters
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        
        # Initialize parameters
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False
    
    def build(self, input_shape):
        """Build the CfC layer."""
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize parameters
        self.kernel = Parameter(tensor.zeros((input_dim, units * 4)))
        self.recurrent_kernel = Parameter(tensor.zeros((units, units * 4)))
        
        # Initialize bias if needed
        if self.neuron_map.use_bias:
            self.bias = Parameter(tensor.zeros((units * 4,)))
        
        # Initialize weights
        if self.neuron_map.kernel_initializer == "glorot_uniform":
            self.kernel.data = glorot_uniform((input_dim, units * 4))
        
        if self.neuron_map.recurrent_initializer == "orthogonal":
            self.recurrent_kernel.data = orthogonal((units, units * 4))
        
        # Mark as built
        self.built = True
    
    def forward(self, inputs, initial_state=None):
        """Forward pass through the CfC layer."""
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get input shape
        input_shape = tensor.shape(inputs)
        batch_size, time_steps, _ = input_shape
        
        # Create initial state if not provided
        if initial_state is None:
            h0 = tensor.zeros((batch_size, self.neuron_map.units))
            t0 = tensor.zeros((batch_size, self.neuron_map.units))
            initial_state = [h0, t0]
        
        # Process sequence
        outputs = []
        states = initial_state
        
        # Get parameters from neuron_map
        time_scale_factor = self.neuron_map.time_scale_factor
        activation_fn = get_activation(self.neuron_map.activation)
        rec_activation_fn = get_activation(self.neuron_map.recurrent_activation)
        
        # Process sequence in reverse if go_backwards is True
        time_indices = range(time_steps - 1, -1, -1) if self.go_backwards else range(time_steps)
        
        # Process each time step
        for t in time_indices:
            # Get current input
            x_t = inputs[:, t]
            
            # Project input
            z = ops.matmul(x_t, self.kernel)
            z = ops.add(z, ops.matmul(states[0], self.recurrent_kernel))
            if self.neuron_map.use_bias:
                z = ops.add(z, self.bias)
            
            # Split into gates
            z_chunks = tensor.split(z, 4, axis=-1)
            z_i, z_f, z_o, z_c = z_chunks
            
            # Apply activations
            i = rec_activation_fn(z_i)  # Input gate
            f = rec_activation_fn(z_f)  # Forget gate
            o = rec_activation_fn(z_o)  # Output gate
            c = activation_fn(z_c)      # Cell input
            
            # Apply time scaling
            decay = ops.exp(ops.divide(-1.0, time_scale_factor))
            
            # Update state
            t_next = ops.add(ops.multiply(f, states[1]), ops.multiply(i, c))
            h_next = ops.multiply(o, activation_fn(ops.add(
                ops.multiply(decay, states[0]),
                ops.multiply(ops.subtract(tensor.ones_like(decay), decay), t_next)
            )))
            
            # Store output and update state
            outputs.append(h_next)
            states = [h_next, t_next]
        
        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1)
        else:
            outputs_tensor = outputs[-1]
        
        # Return outputs and states if requested
        if self.return_state:
            return outputs_tensor, states
        else:
            return outputs_tensor
    
    def get_config(self):
        """Returns the configuration of the CfC layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "go_backwards": self.go_backwards
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Creates a CfC layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)
```

## 4. Refactored LTC Layer Implementation

```python
class LTC(Module):
    def __init__(
        self,
        neuron_map: NCPMap,
        return_sequences: bool = True,
        return_state: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate neuron_map type
        if not isinstance(neuron_map, NeuronMap):
            raise TypeError("neuron_map must be a NeuronMap instance")
        
        # Store the neuron map
        self.neuron_map = neuron_map
        
        # Store layer-specific parameters
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Initialize parameters
        self.gleak = None
        self.vleak = None
        self.cm = None
        self.sigma = None
        self.mu = None
        self.w = None
        self.erev = None
        self.sensory_sigma = None
        self.sensory_mu = None
        self.sensory_w = None
        self.sensory_erev = None
        self.sparsity_mask = None
        self.sensory_sparsity_mask = None
        self.built = False
    
    def build(self, input_shape):
        """Build the LTC layer."""
        # Get input dimension
        input_dim = input_shape[-1]
        
        # Build the neuron map if not already built
        if not self.neuron_map.is_built():
            self.neuron_map.build(input_dim)
        
        # Get dimensions from neuron map
        units = self.neuron_map.units
        
        # Initialize parameters based on LTC-specific logic
        # (Implementation details would go here)
        
        # Mark as built
        self.built = True
    
    def forward(self, inputs, initial_state=None):
        """Forward pass through the LTC layer."""
        # Build if not already built
        if not self.built:
            self.build(tensor.shape(inputs))
        
        # Get input shape
        input_shape = tensor.shape(inputs)
        batch_size, time_steps, _ = input_shape
        
        # Create initial state if not provided
        if initial_state is None:
            initial_state = tensor.zeros((batch_size, self.neuron_map.units))
        
        # Process sequence
        outputs = []
        state = initial_state
        
        # Get parameters from neuron_map
        ode_unfolds = self.neuron_map.ode_unfolds
        epsilon = self.neuron_map.epsilon
        implicit_param_constraints = self.neuron_map.implicit_param_constraints
        input_mapping = self.neuron_map.input_mapping
        output_mapping = self.neuron_map.output_mapping
        
        # Process each time step
        for t in range(time_steps):
            # Get current input
            x_t = inputs[:, t]
            
            # Map inputs based on input_mapping
            if input_mapping in ["affine", "linear"]:
                # Apply input mapping
                pass
            
            # Solve ODE
            next_state = self._ode_solver(x_t, state, 1.0)
            
            # Map outputs based on output_mapping
            if output_mapping in ["affine", "linear"]:
                # Apply output mapping
                pass
            else:
                output = next_state
            
            # Store output and update state
            outputs.append(output)
            state = next_state
        
        # Stack outputs
        if self.return_sequences:
            outputs_tensor = tensor.stack(outputs, axis=1)
        else:
            outputs_tensor = outputs[-1]
        
        # Return outputs and state if requested
        if self.return_state:
            return outputs_tensor, state
        else:
            return outputs_tensor
    
    def _ode_solver(self, inputs, state, elapsed_time):
        """Solve the ODE for the LTC dynamics."""
        # Implementation of the ODE solver would go here
        return state
    
    def get_config(self):
        """Returns the configuration of the LTC layer."""
        config = super().get_config()
        config.update({
            "neuron_map": self.neuron_map.get_config(),
            "return_sequences": self.return_sequences,
            "return_state": self.return_state
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Creates an LTC layer from its configuration."""
        # Extract neuron_map config
        neuron_map_config = config.pop("neuron_map", {})
        
        # Create neuron_map
        from ember_ml.nn.modules.wiring import NCPMap
        neuron_map = NCPMap.from_config(neuron_map_config)
        
        # Create layer
        return cls(neuron_map=neuron_map, **config)
```

## 5. New Usage Patterns

### Creating and Using LTC
```python
# Create an enhanced NCPMap with all parameters
ncp_map = NCPMap(
    inter_neurons=40,
    command_neurons=20,
    motor_neurons=10,
    sensory_neurons=8,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    ode_unfolds=6,
    epsilon=1e-8,
    implicit_param_constraints=False,
    input_mapping="affine",
    output_mapping="affine",
    sparsity_level=0.5,
    seed=42
)

# Create an LTC layer
ltc_layer = LTC(
    neuron_map=ncp_map,
    return_sequences=True,
    return_state=False
)

# Forward pass
x = tensor.random_normal((32, 10, 8))
y = ltc_layer(x)
```

### Creating and Using CfC
```python
# Create an enhanced NCPMap with all parameters
ncp_map = NCPMap(
    inter_neurons=40,
    command_neurons=20,
    motor_neurons=10,
    sensory_neurons=8,
    time_scale_factor=1.0,
    activation="tanh",
    recurrent_activation="sigmoid",
    mode="default",
    mixed_memory=False,
    sparsity_level=0.5,
    seed=42
)

# Create a CfC layer
cfc_layer = CfC(
    neuron_map=ncp_map,
    return_sequences=True,
    return_state=False
)

# Forward pass
x = tensor.random_normal((32, 10, 8))
y = cfc_layer(x)
```

## 6. Migration Strategy

1. **Enhance NCPMap First**:
   - Add cell-specific parameters to NCPMap
   - Update get_config() and from_config()
   - Ensure backward compatibility

2. **Create New Layer Implementations**:
   - Implement new CfC and LTC classes
   - Keep old implementations temporarily for compatibility

3. **Update Documentation and Tests**:
   - Update API documentation
   - Create migration guides
   - Update tests to use new patterns

4. **Deprecate Old Implementations**:
   - Mark old Cell classes as deprecated
   - Provide warnings for old usage patterns
   - Eventually remove old implementations

## 7. Benefits of New Architecture

1. **Simplified Class Hierarchy**: Flatter hierarchy with clearer responsibilities
2. **Consolidated Parameters**: All parameters in one place (NCPMap)
3. **Consistent API**: All RNN layers follow the same pattern
4. **Reduced Duplication**: No separate cell classes with duplicated parameters
5. **Improved Maintainability**: Easier to add new features or fix bugs
6. **Better Serialization**: Cleaner serialization/deserialization
7. **Alignment with Design Principles**: Better reflects the biological inspiration of NCPs