import matplotlib.pyplot as plt
from ncps import wirings
import os
def set_keras_backend(backend_name):
    """
    Sets the Keras backend by setting the KERAS_BACKEND environment variable.
    This should be done *before* importing Keras.
    """
    os.environ['KERAS_BACKEND'] = backend_name
set_keras_backend('numpy')

import keras
from typing import Union, List, Tuple, Dict, Optional

# LeCun improved tanh activation
@keras.utils.register_keras_serializable(package="", name="lecun_tanh")  # Changed package to "ncps"
def lecun_tanh(x):
    return 1.7159 * keras.activations.tanh(0.66666667 * x)  # More precise 2/3

# Binomial Initializer (remains unchanged, but included for completeness)
@keras.utils.register_keras_serializable(package="ncps", name="BinomialInitializer")  # Consistent package name
class BinomialInitializer(keras.initializers.Initializer):
    def __init__(self, probability=0.5, seed=None):
        super().__init__()
        self.probability = probability
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = keras.backend.floatx()
        return keras.ops.cast(
            keras.random.uniform(shape, minval=0.0, maxval=1.0, seed=self.seed) < self.probability,
            dtype=dtype
        )

    def get_config(self):
        return {"probability": self.probability, "seed": self.seed}


@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareWiredCfCCell")
class StrideAwareWiredCfCCell(keras.layers.Layer):

    def __init__(
            self,
            wiring: wirings.Wiring,
            stride_length: int = 1,
            time_scale_factor: float = 1.0,
            fully_recurrent: bool = True,
            mode: str = "default",
            activation = lecun_tanh,
            backbone_units: int = 128,  # Added backbone_units and backbone_layers here
            backbone_layers: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.wiring = wiring
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        self.mode = mode
        self._activation = activation  # Store string for serialization
        self.backbone_units = backbone_units # Use in build
        self.backbone_layers = backbone_layers

        self.units = wiring.units
        self.input_dim = wiring.input_dim
        self.output_dim = wiring.output_dim

        self.activation = lecun_tanh  # Get activation function here
        self.recurrent_activation = keras.activations.get('sigmoid')

        # Masks are now initialized in build()

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        # Input weights
        self.kernel = self.add_weight(
            shape=(input_dim, self.backbone_units),
            initializer='glorot_uniform',
            name='kernel',
            regularizer=keras.regularizers.L2(0.01),  # Use Keras regularizers
            constraint=keras.constraints.MaxNorm(3)  # Use Keras constraints
        )

        # Recurrent weights
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.backbone_units),
            initializer='orthogonal',
            name='recurrent_kernel',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
        )

       # Backbone weights (now handles multiple layers)
        self.backbone_kernels = []
        self.backbone_biases = []
        for i in range(self.backbone_layers):
            backbone_kernel = self.add_weight(
                shape=(self.backbone_units, self.backbone_units),
                initializer="glorot_uniform",
                name=f"backbone_kernel_{i}",
                regularizer=keras.regularizers.L2(0.01),
                constraint=keras.constraints.MaxNorm(3)
            )
            backbone_bias = self.add_weight(
                shape=(self.backbone_units,),
                initializer="zeros",
                name=f"backbone_bias_{i}",
            )
            self.backbone_kernels.append(backbone_kernel)
            self.backbone_biases.append(backbone_bias)

        # Output projection
        self.backbone_out = self.add_weight(
            shape=(self.backbone_units, self.units),
            initializer='glorot_uniform',
            name='backbone_out',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
        )

        # Time gate weights
        self.time_kernel = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            name='time_kernel',
            regularizer=keras.regularizers.L2(0.01),
            constraint=keras.constraints.MaxNorm(3)
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
                name='gate_kernel',
                regularizer=keras.regularizers.L2(0.01),
                constraint=keras.constraints.MaxNorm(3)
            )

            self.gate_recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='orthogonal',
                name='gate_recurrent_kernel',
                regularizer=keras.regularizers.L2(0.01),
                constraint=keras.constraints.MaxNorm(3)
            )

            self.gate_bias = self.add_weight(
                shape=(self.units,),
                initializer='ones',  # Initialize with ones for open gates
                name='gate_bias'
            )

        # Sparsity masks (using the custom initializer, now in build())
        sparsity = self.wiring.get_config()["sparsity_level"]
        self.input_mask = self.add_weight(
            shape=(input_dim,),
            initializer=BinomialInitializer(probability=sparsity, seed=42),
            name='input_mask',
            trainable=False
        )
        self.recurrent_mask = self.add_weight(
            shape=(self.units, self.units),
            initializer=BinomialInitializer(probability=sparsity, seed=43),
            name='recurrent_mask',
            trainable=False
        )
        self.output_mask = self.add_weight(
            shape=(self.units,),
            initializer=BinomialInitializer(probability=sparsity, seed=44),
            name='output_mask',
            trainable=False
        )

        self.built = True

    def _compute_time_scaling(self, inputs, kwargs):
        """Helper function to compute time scaling."""
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = t * self.stride_length * self.time_scale_factor
        else:
            t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
            t = keras.ops.cast(t, dtype=keras.backend.floatx())  # Use keras.ops.cast and keras.backend.floatx()
        return inputs, t


    def call(self, inputs, states, **kwargs):
        h_prev = states[0]
        inputs, t = self._compute_time_scaling(inputs, kwargs)

        masked_inputs = inputs * self.input_mask
        masked_h_prev = h_prev * self.recurrent_mask

        # Backbone computation
        x = keras.ops.matmul(masked_inputs, self.kernel) + self.bias
        x = self.activation(x + keras.ops.matmul(masked_h_prev, self.recurrent_kernel))

        for i in range(self.backbone_layers):
            x = self.activation(keras.ops.matmul(x, self.backbone_kernels[i]) + self.backbone_biases[i])

        h_candidate = keras.ops.matmul(x, self.backbone_out) + self.recurrent_bias
        time_gate = keras.ops.exp(-keras.ops.abs(t) * keras.ops.exp(self.time_kernel))

        if self.mode == "no_gate":
            h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
        else:
            gate_in = keras.ops.matmul(inputs, self.gate_kernel)
            gate_rec = keras.ops.matmul(h_prev, self.gate_recurrent_kernel)
            gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)

            if self.mode == "pure":
                h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
            else:
                h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)

        output = h_new * self.output_mask
        return output, [h_new]

    def get_config(self):
        config = {
            "wiring": self.wiring.get_config(),  # Get the config, not the wiring object
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
            "activation": self._activation,  # Store string for serialization
            "backbone_units": self.backbone_units, # Serialize backbone parameters
            "backbone_layers": self.backbone_layers,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        wiring_config = config.pop("wiring")
        from ncps import wirings  # Local import to avoid circular dependency
        wiring_class = getattr(wirings, wiring_config["class_name"])
        wiring = wiring_class.from_config(wiring_config["config"])
        return cls(wiring=wiring, **config)  # Pass wiring to constructor

    @property
    def state_size(self):
        return self.units

    @property
    def input_size(self):
        return self.input_dim

    @property
    def output_size(self):
        return self.output_dim




@keras.utils.register_keras_serializable(package="ncps", name="StrideAwareCfC")
class StrideAwareCfC(keras.layers.RNN):
    def __init__(
        self,
        units: Union[int, wirings.Wiring],
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        mixed_memory: bool = False,
        mode: str = "default",
        activation = lecun_tanh,
        backbone_units: int = None,  # Add backbone parameters
        backbone_layers: int = None,
        backbone_dropout: float = None,
        fully_recurrent: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        zero_output_for_mask: bool = False,
        **kwargs
    ):
        if isinstance(units, wirings.Wiring):
            if any([backbone_units, backbone_layers, backbone_dropout]):
                raise ValueError("Cannot use backbone parameters with a Wiring object.")
            cell = StrideAwareWiredCfCCell(
                wiring=units,  # Pass the Wiring object directly
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                fully_recurrent=fully_recurrent,
                mode=mode,
                activation=activation,
                backbone_units = 128,
                backbone_layers = 1
            )
        else:
            backbone_units = backbone_units or 128  # Default values
            backbone_layers = backbone_layers or 1
            backbone_dropout = backbone_dropout or 0.0

            class StrideAwareCfCCell(keras.layers.Layer):
                def __init__(self, units, stride_length=1, time_scale_factor=1.0,
                             mode="default", activation=lecun_tanh,
                             backbone_units=128, backbone_layers=1, backbone_dropout=0.0,
                              **kwargs):
                    super().__init__(**kwargs)
                    self.units = units
                    self.stride_length = stride_length
                    self.time_scale_factor = time_scale_factor
                    self.mode = mode
                    self._activation = activation  #For serialization
                    self.backbone_units = backbone_units
                    self.backbone_layers = backbone_layers
                    self.backbone_dropout = backbone_dropout


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
                    self.built = True

                @property
                def state_size(self):
                    return self.units

                def call(self, inputs, states, **kwargs):
                    h_prev = states[0]

                    if isinstance(inputs, (tuple, list)):
                        inputs, t = inputs
                        t = t * self.stride_length * self.time_scale_factor
                    else:
                        t = kwargs.get("time", 1.0) * self.stride_length * self.time_scale_factor
                        t = keras.ops.cast(t, dtype=keras.backend.floatx())  # Use keras.ops

                    # Compute backbone activations
                    x = keras.ops.matmul(inputs, self.kernel) + self.bias
                    x = self.activation(x + keras.ops.matmul(h_prev, self.recurrent_kernel))

                    # Apply multiple backbone layers with dropout
                    for i in range(self.backbone_layers):
                        x = keras.ops.matmul(x, self.backbone_kernels[i]) + self.backbone_biases[i]
                        x = self.activation(x)
                        if self.backbone_dropout > 0:
                            x = keras.ops.dropout(x, rate=self.backbone_dropout)

                    # Project backbone to hidden state size
                    h_candidate = keras.ops.matmul(x, self.backbone_out) + self.recurrent_bias

                    # Apply time-scaling
                    time_gate = keras.ops.exp(-keras.ops.abs(t) * keras.ops.exp(self.time_kernel))

                    if self.mode == "no_gate":
                        h_new = h_prev * time_gate + h_candidate * (1 - time_gate)
                    else:
                        gate_in = keras.ops.matmul(inputs, self.gate_kernel)
                        gate_rec = keras.ops.matmul(h_prev, self.gate_recurrent_kernel)
                        gate = self.recurrent_activation(gate_in + gate_rec + self.gate_bias)

                        if self.mode == "pure":
                            h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
                        else:
                            h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)

                    return h_new, [h_new]

                def get_config(self):
                    config = {
                        "units": self.units,
                        "stride_length": self.stride_length,
                        "time_scale_factor": self.time_scale_factor,
                        "mode": self.mode,
                        "activation": self._activation, #serialize string
                        "backbone_units": self.backbone_units,
                        "backbone_layers": self.backbone_layers,
                        "backbone_dropout": self.backbone_dropout,
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
                    memory_gate = keras.ops.sigmoid(
                        keras.ops.matmul(inputs, self.memory_kernel) +
                        keras.ops.matmul(memory_state, self.memory_recurrent_kernel) +
                        self.memory_bias
                    )

                    new_memory_state = memory_state * memory_gate + output * (1 - memory_gate)

                    return output, [new_rnn_state[0], new_memory_state]

                def get_config(self):
                    config = {
                        "cell": keras.layers.serialize(self.rnn_cell),  # Serialize the cell
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
        self._activation = activation # for serialization
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.fully_recurrent = fully_recurrent


    def get_config(self):
      is_mixed_memory = hasattr(self.cell, 'rnn_cell')
      cell = self.cell.rnn_cell if is_mixed_memory else self.cell

      config = super().get_config()
      config.update({ # Use update for cleaner merging
          "units": cell.units,  # Could be int or Wiring
          "stride_length": self.stride_length,
          "time_scale_factor": self.time_scale_factor,
          "mixed_memory": self.mixed_memory,
          "activation": self._activation,
          "backbone_units": self.backbone_units,
          "backbone_layers": self.backbone_layers,
          "backbone_dropout": self.backbone_dropout,
          "fully_recurrent": self.fully_recurrent,
      })
      if hasattr(cell, 'wiring'): # Check for wiring attribute.
          config["units"] = cell.wiring.get_config() # Save wiring's config

      return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config_copy = config.copy()

        if "cell" in config_copy:  # For MixedMemoryRNN
            del config_copy["cell"]

        if isinstance(config_copy["units"], dict): # It's a wiring config
            from ncps import wirings
            wiring_class = getattr(wirings, config_copy["units"]["class_name"])
            units = wiring_class.from_config(config_copy["units"]["config"])
            del config_copy["units"]
        else:
            units = config_copy.pop("units") # Remove and retrieve units.

        return cls(units, **config_copy) # Pass units and other params

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
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.gridspec import GridSpec

    # Set seeds for reproducibility
    np.random.seed(seed)

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
    states = {stride: [keras.ops.convert_to_tensor(keras.ops.zeros((1, units)), dtype=keras.backend.floatx())] for stride in stride_lengths}

    # Track state evolution for each stride
    state_evolution = {stride: keras.ops.zeros((time_steps, units)) for stride in stride_lengths}

    # Process sequence through each stride-specific cell
    for t_idx in range(time_steps):
        x_t = input_sequence[t_idx:t_idx+1]
        x_t = keras.ops.convert_to_tensor(x_t, dtype=keras.backend.floatx())

        for stride, cell in stride_cells.items():
            # Only process input at stride-specific intervals
            if t_idx % stride == 0:
                # Ensure states have the right shape for the cell
                current_state = states[stride][0]
                # Add batch dimension if needed
                if len(current_state.shape) == 1:
                    current_state = keras.ops.reshape(current_state, [1, -1])
                    states[stride] = [current_state]

                output, new_state = cell(x_t, states[stride], time=1.0)
                states[stride] = new_state

            # Record state at every time step for all cells
            # Ensure we're getting a numpy array with the right shape
            state_array = states[stride][0]  # Already a Keras tensor
            if len(state_array.shape) > 1:
                state_array = state_array[0]
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

        avg_fft = np.mean(np.array(fft_magnitudes), axis=0)  # Corrected line
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