import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ncps import wirings
import keras
from typing import Union, List, Tuple, Dict, Optional
# LeCun improved tanh activation
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
@keras.utils.register_keras_serializable(package="ncps", name="lecun_tanh")
def lecun_tanh(x):
    return 1.7159 * keras.activations.tanh(0.666 * x)


# Register the custom activation function
try:
    from keras.src.activations import ALL_OBJECTS_DICT
    ALL_OBJECTS_DICT["lecun_tanh"] = lecun_tanh
except ImportError:
    # For older versions of Keras
    pass


@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareWiredCfCCell")
class StrideAwareWiredCfCCell(keras.layers.Layer):
    """A stride-aware CfC cell that properly respects the Wiring architecture.
    
    This implementation doesn't inherit from CfC classes but reimplements the functionality.
    """
    
    def __init__(
            self,
            wiring: wirings.Wiring,
            stride_length=1,
            time_scale_factor=1.0,
            fully_recurrent=True,
            mode="default",
            activation="lecun_tanh",
            **kwargs
    ):
        """Initialize a stride-aware WiredCfCCell.
        
        Args:
            wiring: A Wiring instance that determines the connectivity pattern
            stride_length: Length of the stride this cell handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            fully_recurrent: Whether to use full recurrent connectivity within layers
            mode: CfC operation mode ("default", "pure", or "no_gate")
            activation: Activation function used in the backbone layers
            **kwargs: Additional arguments to pass to the Layer constructor
        """
        super().__init__(**kwargs)
        self.wiring = wiring
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        self.mode = mode
        self._activation = activation
        
        # Get dimensions from wiring
        self.units = wiring.units
        self.input_dim = wiring.input_dim
        self.output_dim = wiring.output_dim
        
        # Activation functions
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get('sigmoid')
        
        # Sparsity masks from wiring
        self.input_mask = tf.constant(wiring.get_input_mask(), dtype=tf.float32)
        self.recurrent_mask = tf.constant(wiring.get_recurrent_mask(fully_recurrent), dtype=tf.float32)
        self.output_mask = tf.constant(wiring.get_output_mask(), dtype=tf.float32)
    
    def build(self, input_shape):
        """Build the cell weights."""
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Extract input dimension
        input_dim = input_shape[-1]
        
        # Create backbone weights
        self.backbone_units = 128  # Default backbone size
        
        # Input weights
        self.kernel = self.add_weight(
            shape=(input_dim, self.backbone_units),
            initializer='glorot_uniform',
            name='kernel'
        )
        
        # Recurrent weights
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.backbone_units),
            initializer='orthogonal',
            name='recurrent_kernel'
        )
        
        # Output projection
        self.backbone_out = self.add_weight(
            shape=(self.backbone_units, self.units),
            initializer='glorot_uniform',
            name='backbone_out'
        )
        
        # Time gate weights
        self.time_kernel = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            name='time_kernel'
        )
        
        # Biases
        self.bias = self.add_weight(
            shape=(self.backbone_units,),
            initializer='zeros',
            name='bias'
        )
        
        self.recurrent_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='recurrent_bias'
        )
        
        # Gate weights (for default and pure modes)
        if self.mode != "no_gate":
            self.gate_kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer='glorot_uniform',
                name='gate_kernel'
            )
            
            self.gate_recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='orthogonal',
                name='gate_recurrent_kernel'
            )
            
            self.gate_bias = self.add_weight(
                shape=(self.units,),
                initializer='ones',  # Initialize with ones for open gates
                name='gate_bias'
            )
        
        self.built = True
    
    @property
    def state_size(self):
        return self.units
    
    @property
    def input_size(self):
        return self.input_dim
    
    @property
    def output_size(self):
        return self.output_dim
    
    def call(self, inputs, states, **kwargs):
        """Apply stride-specific temporal scaling."""
        # Extract state
        h_prev = states[0]
        
        # Handle different input formats
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
        else:
            # Regularly sampled mode
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            # Create a tensor for time
            t = tf.constant(t, dtype=tf.float32)
        
        # Apply input mask from wiring
        masked_inputs = inputs * self.input_mask
        
        # Apply recurrent mask from wiring
        masked_h_prev = h_prev * self.recurrent_mask
        
        # Compute backbone activations
        backbone_in = tf.matmul(masked_inputs, self.kernel) + self.bias
        backbone_rec = tf.matmul(masked_h_prev, self.recurrent_kernel)
        backbone_act = self.activation(backbone_in + backbone_rec)
        
        # Project backbone to hidden state size
        h_candidate = tf.matmul(backbone_act, self.backbone_out) + self.recurrent_bias
        
        # Apply time-scaling
        time_gate = tf.exp(-tf.abs(t) * tf.exp(self.time_kernel))
        
        if self.mode == "no_gate":
            # No gating, just apply time scaling
            h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
        else:
            # Compute update gate
            gate_in = tf.matmul(inputs, self.gate_kernel)
            gate_rec = tf.matmul(h_prev, self.gate_recurrent_kernel)
            gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)
            
            if self.mode == "pure":
                # Pure CfC mode
                h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
            else:
                # Default mode
                h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)
        
        # Apply output mask from wiring
        output = h_new * self.output_mask
        
        return output, [h_new]
    
    def get_config(self):
        config = {
            "wiring": self.wiring,
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
            "activation": self._activation,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareCfC")
class StrideAwareCfC(keras.layers.RNN):
    """A stride-aware CfC RNN layer that properly handles temporal scaling.
    
    This implementation doesn't inherit from CfC classes but reimplements the functionality.
    """
    
    def __init__(
            self,
            units: Union[int, wirings.Wiring],
            stride_length=1,
            time_scale_factor=1.0,
            mixed_memory=False,
            mode="default",
            activation="lecun_tanh",
            backbone_units=None,
            backbone_layers=None,
            backbone_dropout=None,
            sparsity_mask=None,
            fully_recurrent=True,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            zero_output_for_mask=False,
            **kwargs
    ):
        """Initialize a stride-aware CfC RNN layer.
        
        Args:
            units: Number of hidden units or a Wiring instance
            stride_length: Length of the stride this cell handles
            time_scale_factor: Scaling factor for temporal dynamics (multiplied by stride_length)
            mixed_memory: Whether to augment the RNN with a memory cell
            mode: CfC operation mode ("default", "pure", or "no_gate")
            activation: Activation function used in the backbone layers
            backbone_units: Number of hidden units in the backbone layer (default 128)
            backbone_layers: Number of backbone layers (default 1)
            backbone_dropout: Dropout rate in the backbone layers (default 0)
            sparsity_mask: Optional sparsity mask for the cell
            fully_recurrent: Whether to use full recurrent connectivity within layers
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the last state along with the output
            go_backwards: Whether to process the sequence backwards
            stateful: Whether to reuse the last state for the next batch
            unroll: Whether to unroll the RNN
            zero_output_for_mask: Whether to zero out masked timesteps in the output
            **kwargs: Additional arguments to pass to the RNN constructor
        """
        # Create the cell based on whether units is a Wiring instance or an integer
        if isinstance(units, wirings.Wiring):
            if backbone_units is not None:
                raise ValueError("Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError("Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError("Cannot use backbone_dropout in wired mode")
            cell = StrideAwareWiredCfCCell(
                units, 
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                fully_recurrent=fully_recurrent,
                mode=mode, 
                activation=activation
            )
        else:
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            
            # Create a custom CfC cell with stride awareness
            class StrideAwareCfCCell(keras.layers.Layer):
                def __init__(self, units, stride_length=1, time_scale_factor=1.0, 
                             mode="default", activation="lecun_tanh", 
                             backbone_units=128, backbone_layers=1, backbone_dropout=0.0,
                             sparsity_mask=None, **kwargs):
                    super().__init__(**kwargs)
                    self.units = units
                    self.stride_length = stride_length
                    self.time_scale_factor = time_scale_factor
                    self.mode = mode
                    self._activation = activation
                    self.backbone_units = backbone_units
                    self.backbone_layers = backbone_layers
                    self.backbone_dropout = backbone_dropout
                    self.sparsity_mask = sparsity_mask
                    
                    # Activation functions
                    self.activation = keras.activations.get(activation)
                    self.recurrent_activation = keras.activations.get('sigmoid')
                
                def build(self, input_shape):
                    if isinstance(input_shape, list):
                        input_shape = input_shape[0]
                    
                    # Extract input dimension
                    input_dim = input_shape[-1]
                    
                    # Input weights
                    self.kernel = self.add_weight(
                        shape=(input_dim, self.backbone_units),
                        initializer='glorot_uniform',
                        name='kernel'
                    )
                    
                    # Backbone weights for multiple layers
                    self.backbone_kernels = []
                    self.backbone_biases = []
                    
                    for i in range(self.backbone_layers):
                        self.backbone_kernels.append(
                            self.add_weight(
                                shape=(self.backbone_units, self.backbone_units),
                                initializer='glorot_uniform',
                                name=f'backbone_kernel_{i}'
                            )
                        )
                        
                        self.backbone_biases.append(
                            self.add_weight(
                                shape=(self.backbone_units,),
                                initializer='zeros',
                                name=f'backbone_bias_{i}'
                            )
                        )
                    
                    # Output projection
                    self.backbone_out = self.add_weight(
                        shape=(self.backbone_units, self.units),
                        initializer='glorot_uniform',
                        name='backbone_out'
                    )
                    
                    # Recurrent weights
                    self.recurrent_kernel = self.add_weight(
                        shape=(self.units, self.backbone_units),
                        initializer='orthogonal',
                        name='recurrent_kernel'
                    )
                    
                    # Time gate weights
                    self.time_kernel = self.add_weight(
                        shape=(1, self.units),
                        initializer='zeros',
                        name='time_kernel'
                    )
                    
                    # Biases
                    self.bias = self.add_weight(
                        shape=(self.backbone_units,),
                        initializer='zeros',
                        name='bias'
                    )
                    
                    self.recurrent_bias = self.add_weight(
                        shape=(self.units,),
                        initializer='zeros',
                        name='recurrent_bias'
                    )
                    
                    # Gate weights (for default and pure modes)
                    if self.mode != "no_gate":
                        self.gate_kernel = self.add_weight(
                            shape=(input_dim, self.units),
                            initializer='glorot_uniform',
                            name='gate_kernel'
                        )
                        
                        self.gate_recurrent_kernel = self.add_weight(
                            shape=(self.units, self.units),
                            initializer='orthogonal',
                            name='gate_recurrent_kernel'
                        )
                        
                        self.gate_bias = self.add_weight(
                            shape=(self.units,),
                            initializer='ones',  # Initialize with ones for open gates
                            name='gate_bias'
                        )
                    
                    # Apply sparsity mask if provided
                    if self.sparsity_mask is not None:
                        self.sparsity_tensor = tf.constant(self.sparsity_mask, dtype=tf.float32)
                    
                    self.built = True
                
                @property
                def state_size(self):
                    return self.units
                
                def call(self, inputs, states, **kwargs):
                    # Extract state
                    h_prev = states[0]
                    
                    # Handle different input formats
                    if isinstance(inputs, (tuple, list)):
                        # Irregularly sampled mode
                        inputs, t = inputs
                        t = t * self.stride_length * self.time_scale_factor
                    else:
                        # Regularly sampled mode
                        t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
                        # Create a tensor for time
                        t = tf.constant(t, dtype=tf.float32)
                    
                    # Apply sparsity mask if provided
                    if hasattr(self, 'sparsity_tensor'):
                        inputs = inputs * self.sparsity_tensor
                    
                    # Compute backbone activations
                    x = tf.matmul(inputs, self.kernel) + self.bias
                    x = self.activation(x + tf.matmul(h_prev, self.recurrent_kernel))
                    
                    # Apply multiple backbone layers with dropout
                    for i in range(self.backbone_layers):
                        x = tf.matmul(x, self.backbone_kernels[i]) + self.backbone_biases[i]
                        x = self.activation(x)
                        if self.backbone_dropout > 0:
                            x = tf.nn.dropout(x, rate=self.backbone_dropout)
                    
                    # Project backbone to hidden state size
                    h_candidate = tf.matmul(x, self.backbone_out) + self.recurrent_bias
                    
                    # Apply time-scaling
                    time_gate = tf.exp(-tf.abs(t) * tf.exp(self.time_kernel))
                    
                    if self.mode == "no_gate":
                        # No gating, just apply time scaling
                        h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
                    else:
                        # Compute update gate
                        gate_in = tf.matmul(inputs, self.gate_kernel)
                        gate_rec = tf.matmul(h_prev, self.gate_recurrent_kernel)
                        gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)
                        
                        if self.mode == "pure":
                            # Pure CfC mode
                            h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
                        else:
                            # Default mode
                            h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)
                    
                    return h_new, [h_new]
                
                def get_config(self):
                    config = {
                        "units": self.units,
                        "stride_length": self.stride_length,
                        "time_scale_factor": self.time_scale_factor,
                        "mode": self.mode,
                        "activation": self._activation,
                        "backbone_units": self.backbone_units,
                        "backbone_layers": self.backbone_layers,
                        "backbone_dropout": self.backbone_dropout,
                        "sparsity_mask": self.sparsity_mask,
                    }
                    base_config = super().get_config()
                    return {**base_config, **config}
            
            cell = StrideAwareCfCCell(
                units,
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                mode=mode,
                activation=activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                sparsity_mask=sparsity_mask,
            )
        
        # Add mixed memory if requested
        if mixed_memory:
            class MixedMemoryRNN(keras.layers.Layer):
                def __init__(self, cell, **kwargs):
                    super().__init__(**kwargs)
                    self.rnn_cell = cell
                    self.units = cell.units
                    self.state_size = [cell.state_size, cell.units]
                
                def build(self, input_shape):
                    self.rnn_cell.build(input_shape)
                    
                    # Memory gate weights
                    input_dim = input_shape[-1]
                    self.memory_kernel = self.add_weight(
                        shape=(input_dim, self.units),
                        initializer='glorot_uniform',
                        name='memory_kernel'
                    )
                    
                    self.memory_recurrent_kernel = self.add_weight(
                        shape=(self.units, self.units),
                        initializer='orthogonal',
                        name='memory_recurrent_kernel'
                    )
                    
                    self.memory_bias = self.add_weight(
                        shape=(self.units,),
                        initializer='zeros',
                        name='memory_bias'
                    )
                    
                    self.built = True
                
                def call(self, inputs, states, **kwargs):
                    rnn_state = states[0]
                    memory_state = states[1]
                    
                    # Process with RNN cell
                    output, new_rnn_state = self.rnn_cell(inputs, [rnn_state], **kwargs)
                    
                    # Update memory with gating
                    memory_gate = tf.sigmoid(
                        tf.matmul(inputs, self.memory_kernel) +
                        tf.matmul(memory_state, self.memory_recurrent_kernel) +
                        self.memory_bias
                    )
                    
                    new_memory_state = memory_state * memory_gate + output * (1 - memory_gate)
                    
                    return output, [new_rnn_state[0], new_memory_state]
                
                def get_config(self):
                    config = {
                        "cell": keras.layers.serialize(self.rnn_cell),
                    }
                    base_config = super().get_config()
                    return {**base_config, **config}
            
            cell = MixedMemoryRNN(cell)
        
        super(StrideAwareCfC, self).__init__(
            cell,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            unroll,
            zero_output_for_mask,
            **kwargs,
        )
        
        # Store parameters for serialization
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.mixed_memory = mixed_memory
    
    def get_config(self):
        is_mixed_memory = hasattr(self.cell, 'rnn_cell')
        cell = self.cell.rnn_cell if is_mixed_memory else self.cell
        
        config = super().get_config()
        
        # Add stride-specific parameters
        config["stride_length"] = self.stride_length
        config["time_scale_factor"] = self.time_scale_factor
        
        # Add CfC parameters
        if hasattr(cell, 'wiring'):
            config["units"] = cell.wiring
        else:
            config["units"] = cell.units
            
        config["mixed_memory"] = self.mixed_memory
        
        if hasattr(cell, 'fully_recurrent'):
            config["fully_recurrent"] = cell.fully_recurrent
        
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        # The following parameters are recreated by the constructor
        config_copy = config.copy()
        
        if "cell" in config_copy:
            del config_copy["cell"]
            
        if "wiring" in config_copy and isinstance(config_copy["units"], dict):
            from ncps import wirings
            wiring_class = getattr(wirings, config_copy["units"]["class_name"])
            units = wiring_class.from_config(config_copy["units"]["config"])
            del config_copy["wiring"]
        else:
            units = config_copy["units"]
            
        del config_copy["units"]
        return cls(units, **config_copy)


def visualize_stride_temporal_dynamics(time_steps=100, stride_lengths=[1, 3, 5], 
                                       units=16, input_dim=3, seed=42):
    """
    Visualizes how different stride lengths affect the temporal dynamics in CfC neurons.
    
    This visualization creates a rigorous analysis of:
    1. State evolution trajectories across different temporal scales
    2. Information retention characteristics as function of stride length
    3. Comparative phase space analysis of multi-timescale representations
    
    Args:
        time_steps: Total number of time steps to simulate
        stride_lengths: List of stride lengths to compare
        units: Number of hidden units in each CfC cell
        input_dim: Input dimensionality
        seed: Random seed for reproducibility
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.gridspec import GridSpec
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create a simple wiring for each stride cell
    wiring = wirings.AutoNCP(units=units, output_size=units//4, sparsity_level=0.5)
    
    # Generate synthetic input sequence with temporal structure
    # Using sinusoidal patterns with varying frequencies to test multi-scale dynamics
    t = np.linspace(0, 4*np.pi, time_steps)
    frequencies = [1.0, 2.0, 0.5]
    input_signals = []
    for freq in frequencies[:input_dim]:
        signal = np.sin(freq * t) + 0.1 * np.random.randn(time_steps)
        input_signals.append(signal)
    input_sequence = np.stack(input_signals, axis=1).astype(np.float32)
    
    # Create cells for each stride length
    stride_cells = {}
    for stride in stride_lengths:
        # Use the properly implemented StrideAwareWiredCfCCell
        cell = StrideAwareWiredCfCCell(
            wiring=wiring,
            stride_length=stride,
            time_scale_factor=1.0,
            mode="default"
        )
        stride_cells[stride] = cell
        
    # Initialize states for each cell
    states = {stride: [tf.zeros((1, units))] for stride in stride_lengths}
    
    # Track state evolution for each stride
    state_evolution = {stride: np.zeros((time_steps, units)) for stride in stride_lengths}
    
    # Process sequence through each stride-specific cell
    for t_idx in range(time_steps):
        x_t = input_sequence[t_idx:t_idx+1]
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        
        for stride, cell in stride_cells.items():
            # Only process input at stride-specific intervals
            if t_idx % stride == 0:
                # Ensure states have the right shape for the cell
                current_state = states[stride][0]
                # Add batch dimension if needed
                if len(current_state.shape) == 1:
                    current_state = tf.reshape(current_state, [1, -1])
                    states[stride] = [current_state]
                
                output, new_state = cell(x_t, states[stride], time=1.0)
                states[stride] = new_state
            
            # Record state at every time step for all cells
            # Ensure we're getting a numpy array with the right shape
            state_array = states[stride][0].numpy()
            if len(state_array.shape) > 1:
                state_array = state_array[0]  # Take the first batch item
            state_evolution[stride][t_idx] = state_array
    
    # === CREATE MULTI-PANEL ANALYTICAL VISUALIZATION ===
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # 1. Time series evolution of key state components
    ax1 = fig.add_subplot(gs[0, :])
    neurons_to_plot = min(3, units)  # Plot first few neurons
    
    for stride, states in state_evolution.items():
        for n in range(neurons_to_plot):
            ax1.plot(states[:, n], label=f"Stride {stride}, Neuron {n}")
    
    ax1.set_title("Temporal Evolution of Neuronal States Across Strides", fontsize=14)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Activation", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. State space trajectories (3D phase plot)
    if units >= 3:
        ax2 = fig.add_subplot(gs[1, :2], projection='3d')
        
        for stride, states in state_evolution.items():
            ax2.plot3D(
                states[:, 0], 
                states[:, 1], 
                states[:, 2], 
                label=f"Stride {stride}"
            )
            # Mark start and end points
            ax2.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]], 
                        color='green', s=50, label="_start" if stride > stride_lengths[0] else "Start")
            ax2.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]], 
                        color='red', s=50, label="_end" if stride > stride_lengths[0] else "End")
        
        ax2.set_title("Phase Space Trajectory", fontsize=14)
        ax2.set_xlabel("State Dimension 1", fontsize=10)
        ax2.set_ylabel("State Dimension 2", fontsize=10)
        ax2.set_zlabel("State Dimension 3", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)
    
    # 3. Information retention analysis
    ax3 = fig.add_subplot(gs[1, 2:])
    
    # Calculate state change rates for each stride
    change_rates = {}
    for stride, states in state_evolution.items():
        # Compute L2 norm of state differences
        diffs = np.linalg.norm(states[1:] - states[:-1], axis=1)
        change_rates[stride] = diffs
    
    for stride, rates in change_rates.items():
        rate_smoothed = np.convolve(rates, np.ones(5)/5, mode='valid')
        ax3.plot(rate_smoothed, label=f"Stride {stride}")
    
    ax3.set_title("State Change Magnitude Over Time", fontsize=14)
    ax3.set_xlabel("Time Step", fontsize=12)
    ax3.set_ylabel("L2 Norm of State Δ", fontsize=12)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Input sensitivity analysis - how different strides respond to input features
    input_idx = np.arange(0, time_steps, max(stride_lengths))
    ax4 = fig.add_subplot(gs[2, :2])
    
    # Plot input signals
    for i in range(input_dim):
        ax4.plot(input_sequence[:, i], '--', alpha=0.5, label=f"Input {i}")
    
    # Overlay vertical lines at each stride's sampling points
    for stride in stride_lengths:
        for idx in range(0, time_steps, stride):
            ax4.axvline(x=idx, color=f'C{stride_lengths.index(stride)}', 
                        linestyle=':', alpha=0.3)
    
    ax4.set_title("Input Signals with Stride Sampling Points", fontsize=14)
    ax4.set_xlabel("Time Step", fontsize=12)
    ax4.set_ylabel("Input Value", fontsize=12)
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Spectral analysis - frequency domain comparison
    ax5 = fig.add_subplot(gs[2, 2:])
    
    for stride, states in state_evolution.items():
        # Take FFT of first few neurons and average
        fft_magnitudes = []
        for n in range(min(5, units)):
            fft = np.abs(np.fft.rfft(states[:, n]))
            fft_magnitudes.append(fft)
        
        avg_fft = np.mean(np.array(fft_magnitudes), axis=0)
        freqs = np.fft.rfftfreq(time_steps)
        
        ax5.plot(freqs, avg_fft, label=f"Stride {stride}")
    
    ax5.set_title("Frequency Domain Analysis", fontsize=14)
    ax5.set_xlabel("Frequency", fontsize=12)
    ax5.set_ylabel("Magnitude", fontsize=12)
    ax5.legend(loc="upper right", fontsize=10)
    ax5.set_xlim([0, 0.5])  # Only show meaningful frequency range
    ax5.grid(True, alpha=0.3)
    
    # Add title and adjust layout
    plt.suptitle(
        f"Multi-scale CfC Temporal Dynamics Analysis\n"
        f"Comparing Stride Lengths: {stride_lengths}", 
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

if __name__ == "__main__":
    fig = visualize_stride_temporal_dynamics()
    plt.show()