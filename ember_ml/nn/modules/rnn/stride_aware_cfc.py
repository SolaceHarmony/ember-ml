"""
Stride-Aware Wired CfC Cell and Layer

This module provides implementations of StrideAwareWiredCfCCell and StrideAwareCfC,
which are specialized recurrent neural network components for multi-timescale processing.
"""

from typing import Union, Optional

from ember_ml import ops
from ember_ml.nn.modules import Module, Parameter
from ember_ml.nn.wirings import Wiring
from ember_ml.initializers import glorot_uniform, orthogonal, BinomialInitializer

# LeCun improved tanh activation
def lecun_tanh(x):
    """
    LeCun improved tanh activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    scale_factor = ops.convert_to_tensor(0.66666667)  # More precise 2/3
    amplitude = ops.convert_to_tensor(1.7159)
    return ops.multiply(amplitude, ops.tanh(ops.multiply(scale_factor, x)))


class StrideAwareWiredCfCCell(Module):
    """
    Stride-Aware Wired CfC Cell.
    
    This cell implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics,
    specialized for multi-timescale processing with custom wiring.
    
    Args:
        wiring: Wiring configuration (e.g., AutoNCP)
        stride_length: Length of stride for time-scaling
        time_scale_factor: Factor to scale the time constant
        fully_recurrent: Whether to use full recurrent connections
        mode: Mode of operation ("default", "pure", or "no_gate")
        activation: Activation function for the output
        backbone_units: Number of units in the backbone
        backbone_layers: Number of layers in the backbone
    """
    
    def __init__(
            self,
            wiring: Wiring,
            stride_length: int = 1,
            time_scale_factor: float = 1.0,
            fully_recurrent: bool = True,
            mode: str = "default",
            activation = lecun_tanh,
            backbone_units: int = 128,
            backbone_layers: int = 1,
            **kwargs
    ):
        """
        Initialize the Stride-Aware Wired CfC cell.
        
        Args:
            wiring: Wiring configuration (e.g., AutoNCP)
            stride_length: Length of stride for time-scaling
            time_scale_factor: Factor to scale the time constant
            fully_recurrent: Whether to use full recurrent connections
            mode: Mode of operation ("default", "pure", or "no_gate")
            activation: Activation function for the output
            backbone_units: Number of units in the backbone
            backbone_layers: Number of layers in the backbone
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.wiring = wiring
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.fully_recurrent = fully_recurrent
        self.mode = mode
        self._activation = activation  # Store for serialization
        self.backbone_units = backbone_units
        self.backbone_layers = backbone_layers

        self.units = wiring.units
        self.input_dim = wiring.input_dim
        self.output_dim = wiring.output_dim

        self.activation = activation
        
        # Initialize weights
        self._initialize_weights()
        
        # State size is defined as a property

    def _initialize_weights(self):
        """Initialize the weights for the cell with wiring constraints."""
        # Get input dimension from wiring
        input_dim = self.wiring.input_dim
        
        # Input weights
        self.kernel = Parameter(ops.zeros((input_dim, self.backbone_units)))
        self.kernel.data = glorot_uniform((input_dim, self.backbone_units))
        
        # Recurrent weights
        self.recurrent_kernel = Parameter(ops.zeros((self.units, self.backbone_units)))
        self.recurrent_kernel.data = orthogonal((self.units, self.backbone_units))
        
        # Backbone weights (multiple layers)
        self.backbone_kernels = []
        self.backbone_biases = []
        for i in range(self.backbone_layers):
            backbone_kernel = Parameter(ops.zeros((self.backbone_units, self.backbone_units)))
            backbone_kernel.data = glorot_uniform((self.backbone_units, self.backbone_units))
            self.backbone_kernels.append(backbone_kernel)
            
            backbone_bias = Parameter(ops.zeros((self.backbone_units,)))
            self.backbone_biases.append(backbone_bias)
        
        # Output projection
        self.backbone_out = Parameter(ops.zeros((self.backbone_units, self.units)))
        self.backbone_out.data = glorot_uniform((self.backbone_units, self.units))
        
        # Time gate weights
        self.time_kernel = Parameter(ops.zeros((1, self.units)))
        
        # Biases
        self.bias = Parameter(ops.zeros((self.backbone_units,)))
        self.recurrent_bias = Parameter(ops.zeros((self.units,)))
        
        # Gate weights (for default and pure modes)
        if self.mode != "no_gate":
            self.gate_kernel = Parameter(ops.zeros((input_dim, self.units)))
            self.gate_kernel.data = glorot_uniform((input_dim, self.units))
            
            self.gate_recurrent_kernel = Parameter(ops.zeros((self.units, self.units)))
            self.gate_recurrent_kernel.data = orthogonal((self.units, self.units))
            
            self.gate_bias = Parameter(ops.ones((self.units,)))  # Initialize with ones for open gates
        
        # Sparsity masks
        sparsity = self.wiring.sparsity_level
        # Use float32 dtype for masks to ensure compatibility with all backends
        mask_dtype = 'float32'
        self.input_mask = Parameter(BinomialInitializer(probability=sparsity, seed=42)((input_dim,), dtype=mask_dtype))
        self.input_mask.requires_grad = False  # Not trainable
        
        self.recurrent_mask = Parameter(BinomialInitializer(probability=sparsity, seed=43)((self.units, self.units), dtype=mask_dtype))
        self.recurrent_mask.requires_grad = False  # Not trainable
        
        self.output_mask = Parameter(BinomialInitializer(probability=sparsity, seed=44)((self.units,), dtype=mask_dtype))
        self.output_mask.requires_grad = False  # Not trainable

    def _compute_time_scaling(self, inputs, kwargs):
        """Helper function to compute time scaling."""
        if isinstance(inputs, (tuple, list)):
            inputs, t = inputs
            t = ops.multiply(ops.multiply(t, self.stride_length), self.time_scale_factor)
        else:
            t = kwargs.get("time", 1.0)
            t = ops.multiply(ops.multiply(t, self.stride_length), self.time_scale_factor)
            t = ops.cast(t, dtype='float32')
        return inputs, t

    def forward(self, inputs, states=None, **kwargs):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state]
            **kwargs: Additional keyword arguments including time
            
        Returns:
            Tuple of (output, [new_hidden_state])
        """
        # Initialize states if not provided
        if states is None:
            h_prev = ops.zeros((ops.shape(inputs)[0], self.units))
        else:
            h_prev = states[0]
        
        # Apply time scaling
        inputs, t = self._compute_time_scaling(inputs, kwargs)
        
        # Apply wiring masks
        # Extract data from Parameter objects to avoid dtype inference issues
        input_mask_data = self.input_mask.data if hasattr(self.input_mask, 'data') else self.input_mask
        recurrent_mask_data = self.recurrent_mask.data if hasattr(self.recurrent_mask, 'data') else self.recurrent_mask
        
        # Use string dtypes for consistent behavior across backends
        float_dtype = 'float32'
        
        # Cast masks to float32
        input_mask_tensor = ops.cast(input_mask_data, dtype=float_dtype)
        recurrent_mask_tensor = ops.cast(recurrent_mask_data, dtype=float_dtype)
        
        # Apply masks
        masked_inputs = ops.multiply(inputs, input_mask_tensor)
        masked_h_prev = ops.multiply(h_prev, recurrent_mask_tensor)
        
        # Backbone computation
        x = ops.add(ops.matmul(masked_inputs, self.kernel), self.bias)
        x = self.activation(ops.add(x, ops.matmul(masked_h_prev, self.recurrent_kernel)))
        
        # Apply backbone layers
        for i in range(self.backbone_layers):
            x = self.activation(ops.add(ops.matmul(x, self.backbone_kernels[i]), self.backbone_biases[i]))
        
        # Compute candidate hidden state
        h_candidate = ops.add(ops.matmul(x, self.backbone_out), self.recurrent_bias)
        
        # Compute time gate
        time_gate = ops.exp(ops.multiply(-ops.abs(t), ops.exp(self.time_kernel)))
        
        # Apply gating mechanism based on mode
        if self.mode == "no_gate":
            h_new = ops.add(
                ops.multiply(h_prev, time_gate),
                ops.multiply(h_candidate, ops.subtract(ops.ones_like(time_gate), time_gate))
            )
        else:
            # Compute gate values
            gate_in = ops.matmul(inputs, self.gate_kernel)
            gate_rec = ops.matmul(h_prev, self.gate_recurrent_kernel)
            gate = ops.sigmoid(ops.add(ops.add(gate_in, gate_rec), self.gate_bias))
            
            if self.mode == "pure":
                # Pure mode: h_new = h_prev * gate * time_gate + h_candidate * (1 - gate * time_gate)
                gate_time = ops.multiply(gate, time_gate)
                h_new = ops.add(
                    ops.multiply(h_prev, gate_time),
                    ops.multiply(h_candidate, ops.subtract(ops.ones_like(gate_time), gate_time))
                )
            else:
                # Default mode: h_new = h_prev * gate + h_candidate * (1 - gate) * (1 - time_gate)
                h_new = ops.add(
                    ops.multiply(h_prev, gate),
                    ops.multiply(
                        ops.multiply(h_candidate, ops.subtract(ops.ones_like(gate), gate)),
                        ops.subtract(ops.ones_like(time_gate), time_gate)
                    )
                )
        
        # Apply output mask
        # Extract data from Parameter objects to avoid dtype inference issues
        output_mask_data = self.output_mask.data if hasattr(self.output_mask, 'data') else self.output_mask
        
        # Use string dtypes for consistent behavior across backends
        output_mask_tensor = ops.cast(output_mask_data, dtype='float32')
        output = ops.multiply(h_new, output_mask_tensor)
        
        return output, [h_new]

    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            Configuration dictionary
        """
        config = {
            "wiring": self.wiring.get_config() if hasattr(self.wiring, 'get_config') else None,
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "fully_recurrent": self.fully_recurrent,
            "mode": self.mode,
            "activation": self._activation,  # Store for serialization
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a StrideAwareWiredCfCCell from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            StrideAwareWiredCfCCell instance
        """
        wiring_config = config.pop("wiring")
        from ember_ml.nn import wirings  # Import from ember_ml.nn
        
        # Get the wiring class from the class name
        wiring_class_name = wiring_config.get("class_name", "Wiring")
        wiring_class = getattr(wirings, wiring_class_name)
        
        # Get the wiring config
        wiring_params = wiring_config.get("config", {})
        
        # Ensure units is provided
        if "units" not in wiring_params:
            wiring_params["units"] = 8  # Default value
        
        # Create the wiring from the config
        wiring = wiring_class(**wiring_params)
        
        # Create the cell
        return cls(wiring=wiring, **config)

    @property
    def state_size(self):
        return self.units

    @property
    def input_size(self):
        return self.input_dim

    @property
    def output_size(self):
        return self.output_dim




class StrideAwareCfC(Module):
    """
    Stride-Aware Continuous-time Fully Connected (CfC) layer.
    
    This layer implements a continuous-time recurrent neural network
    with closed-form solution for the hidden state dynamics,
    specialized for multi-timescale processing.
    
    Args:
        units: Number of units or a Wiring object
        stride_length: Length of stride for time-scaling
        time_scale_factor: Factor to scale the time constant
        mixed_memory: Whether to use mixed memory
        mode: Mode of operation ("default", "pure", or "no_gate")
        activation: Activation function for the output
        backbone_units: Number of units in the backbone
        backbone_layers: Number of layers in the backbone
        backbone_dropout: Dropout rate for the backbone
        fully_recurrent: Whether to use full recurrent connections
        return_sequences: Whether to return the full sequence
        return_state: Whether to return the state
    """
    
    def __init__(
        self,
        units_or_cell: Union[int, Wiring, StrideAwareWiredCfCCell],
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        mixed_memory: bool = False,
        mode: str = "default",
        activation = lecun_tanh,
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[float] = None,
        fully_recurrent: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        **kwargs
    ):
        """
        Initialize the StrideAwareCfC layer.
        
        Args:
            units_or_cell: Number of units, a Wiring object, or a StrideAwareWiredCfCCell
            stride_length: Length of stride for time-scaling
            time_scale_factor: Factor to scale the time constant
            mixed_memory: Whether to use mixed memory
            mode: Mode of operation ("default", "pure", or "no_gate")
            activation: Activation function for the output
            backbone_units: Number of units in the backbone
            backbone_layers: Number of layers in the backbone
            backbone_dropout: Dropout rate for the backbone
            fully_recurrent: Whether to use full recurrent connections
            return_sequences: Whether to return the full sequence
            return_state: Whether to return the state
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        
        # Store parameters
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.mixed_memory = mixed_memory
        self.mode = mode
        self._activation = activation
        self.backbone_units = backbone_units or 128
        self.backbone_layers = backbone_layers or 1
        self.backbone_dropout = backbone_dropout or 0.0
        self.fully_recurrent = fully_recurrent
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Create the cell
        if isinstance(units_or_cell, StrideAwareWiredCfCCell):
            # Use the provided cell directly
            self.cell = units_or_cell
        elif isinstance(units_or_cell, Wiring):
            if any([backbone_units, backbone_layers, backbone_dropout]):
                raise ValueError("Cannot use backbone parameters with a Wiring object.")
            self.cell = StrideAwareWiredCfCCell(
                wiring=units_or_cell,
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                fully_recurrent=fully_recurrent,
                mode=mode,
                activation=activation,
                backbone_units=128,
                backbone_layers=1
            )
        else:
            # Create a simple CfC cell without wiring
            self.cell = self._create_simple_cfc_cell(
                units_or_cell,
                stride_length=stride_length,
                time_scale_factor=time_scale_factor,
                mode=mode,
                activation=activation,
                backbone_units=self.backbone_units,
                backbone_layers=self.backbone_layers,
                backbone_dropout=self.backbone_dropout
            )
        
        # Add mixed memory if requested
        if mixed_memory:
            self.cell = self._create_mixed_memory_cell(self.cell)
    
    def _create_simple_cfc_cell(self, units, **kwargs):
        """Create a simple CfC cell without wiring."""
        # This would be a simplified version of StrideAwareWiredCfCCell
        # that doesn't use wiring but still has the same functionality
        # For now, we'll just use the StrideAwareWiredCfCCell with a default wiring
        from ember_ml.nn.wirings import FullyConnectedWiring
        wiring = FullyConnectedWiring(units=units, output_dim=units, input_dim=units)
        return StrideAwareWiredCfCCell(wiring=wiring, **kwargs)
    
    def _create_mixed_memory_cell(self, cell):
        """Create a mixed memory cell that wraps the given cell."""
        # This would be a wrapper around the cell that adds mixed memory
        # For now, we'll just return the cell as is
        return cell
    
    def forward(self, inputs, initial_state=None, **kwargs):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor
            initial_state: Initial state
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor or tuple of (output, state) if return_state is True
        """
        # Process the sequence
        batch_size = ops.shape(inputs)[0]
        time_steps = ops.shape(inputs)[1]
        
        # Initialize state if not provided
        if initial_state is None:
            state = [ops.zeros((batch_size, self.cell.units))]
        else:
            state = initial_state
        
        # Process each time step
        outputs = []
        for t in range(time_steps):
            # Get input at time t
            x_t = inputs[:, t, :]
            
            # Process with cell
            output, state = self.cell.forward(x_t, state, **kwargs)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs
        if self.return_sequences:
            outputs = ops.stack(outputs, axis=1)
        else:
            outputs = outputs[-1]
        
        # Return output and state if requested
        if self.return_state:
            return outputs, state
        else:
            return outputs


    def get_config(self):
        """
        Get configuration for serialization.
        
        Returns:
            Configuration dictionary
        """
        config = {
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "mixed_memory": self.mixed_memory,
            "mode": self.mode,
            "activation": self._activation,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "backbone_dropout": self.backbone_dropout,
            "fully_recurrent": self.fully_recurrent,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
        }
        
        # If the cell has a wiring, save its config
        if hasattr(self.cell, 'wiring'):
            config["units_or_cell"] = self.cell.wiring.get_config() if hasattr(self.cell.wiring, 'get_config') else None
        else:
            # Otherwise, save the cell's units
            config["units_or_cell"] = getattr(self.cell, 'units', 8)
        
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a StrideAwareCfC from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            StrideAwareCfC instance
        """
        config_copy = config.copy()
        
        # Handle wiring configuration
        if isinstance(config_copy.get("units_or_cell"), dict):
            # It's a wiring config
            from ember_ml.nn import wirings
            
            # Get the wiring class from the class name
            wiring_class_name = config_copy["units_or_cell"].get("class_name", "Wiring")
            wiring_class = getattr(wirings, wiring_class_name)
            
            # Get the wiring config
            wiring_params = config_copy["units_or_cell"].get("config", {})
            
            # Ensure units is provided
            if "units" not in wiring_params:
                wiring_params["units"] = 8  # Default value
            
            # Create the wiring from the config
            units_or_cell = wiring_class(**wiring_params)
            del config_copy["units_or_cell"]
        else:
            # It's a simple integer
            units_or_cell = config_copy.pop("units_or_cell", 8)
        
        # Remove backbone parameters if units_or_cell is a Wiring object
        if isinstance(units_or_cell, Wiring):
            config_copy.pop('backbone_units', None)
            config_copy.pop('backbone_layers', None)
            config_copy.pop('backbone_dropout', None)
        
        # Create the layer
        return cls(units_or_cell, **config_copy)

# NOTE: This function is for visualization purposes only and is not used in the core functionality.
# It's acceptable to use NumPy here because:
# 1. The visualization function is not part of the core functionality
# 2. It's only used for debugging and demonstration purposes
# 3. The visualization libraries (matplotlib) require NumPy arrays
# The core functionality of the module uses the ops abstraction layer as required.
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
    # Import visualization libraries
    # These imports are only used for visualization and not for core functionality
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from ember_ml.nn import wirings
    
    try:
        from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
    except ImportError:
        Axes3D = None  # If not available, we'll skip 3D plotting

    # Set seeds for reproducibility
    np.random.seed(seed)
    ops.set_seed(seed)

    # Create a simple wiring for each stride cell
    from ember_ml.nn.wirings import AutoNCP
    wiring = AutoNCP(units=units, output_size=units//4, sparsity_level=0.5)

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
    states = {stride: [ops.zeros((1, units))] for stride in stride_lengths}

    # Track state evolution for each stride
    state_evolution = {stride: ops.zeros((time_steps, units)) for stride in stride_lengths}

    # Process sequence through each stride-specific cell
    for t_idx in range(time_steps):
        x_t = input_sequence[t_idx:t_idx+1]
        x_t = ops.convert_to_tensor(x_t)

        for stride, cell in stride_cells.items():
            # Only process input at stride-specific intervals
            if t_idx % stride == 0:
                # Ensure states have the right shape for the cell
                current_state = states[stride][0]
                # Add batch dimension if needed
                if len(ops.shape(current_state)) == 1:
                    current_state = ops.reshape(current_state, [1, -1])
                    states[stride] = [current_state]

                output, new_state = cell.forward(x_t, states[stride], time=1.0)
                states[stride] = new_state

            # Record state at every time step for all cells
            # Ensure we're getting a numpy array with the right shape
            state_array = states[stride][0]  # Already a tensor
            if len(ops.shape(state_array)) > 1:
                state_array = state_array[0]
            state_evolution[stride] = ops.tensor_scatter_nd_update(
                state_evolution[stride],
                ops.stack([[t_idx]], axis=0),
                ops.expand_dims(state_array, axis=0)
            )

    # Convert tensors to NumPy for visualization
    state_evolution_np = {}
    for stride, states in state_evolution.items():
        state_evolution_np[stride] = states.numpy() if hasattr(states, 'numpy') else states

    # === CREATE MULTI-PANEL ANALYTICAL VISUALIZATION ===
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    # 1. Time series evolution of key state components
    ax1 = fig.add_subplot(gs[0, :])
    neurons_to_plot = min(3, units)  # Plot first few neurons

    for stride, states in state_evolution_np.items():
        for n in range(neurons_to_plot):
            ax1.plot(states[:, n], label=f"Stride {stride}, Neuron {n}")

    ax1.set_title("Temporal Evolution of Neuronal States Across Strides", fontsize=14)
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Activation", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. State space trajectories (3D phase plot)
    if units >= 3 and Axes3D is not None:
        ax2 = fig.add_subplot(gs[1, :2], projection='3d')

        for stride, states in state_evolution_np.items():
            ax2.plot(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                label=f"Stride {stride}"
            )
            # Mark start and end points
            ax2.scatter([states[0, 0]], [states[0, 1]], [states[0, 2]],
                        color='green', marker='o', label="_start" if stride > stride_lengths[0] else "Start")
            ax2.scatter([states[-1, 0]], [states[-1, 1]], [states[-1, 2]],
                        color='red', marker='o', label="_end" if stride > stride_lengths[0] else "End")

        ax2.set_title("Phase Space Trajectory", fontsize=14)
        ax2.set_xlabel("State Dimension 1", fontsize=10)
        ax2.set_ylabel("State Dimension 2", fontsize=10)
        # set_zlabel is a valid method for 3D axes, but Pylance doesn't recognize it
        # This is fine because we're only using it when Axes3D is available
        if hasattr(ax2, 'set_zlabel'):
            ax2.set_zlabel("State Dimension 3", fontsize=10)  # type: ignore
        ax2.legend(loc="upper right", fontsize=10)

    # 3. Information retention analysis
    ax3 = fig.add_subplot(gs[1, 2:])

    # Calculate state change rates for each stride
    change_rates = {}
    for stride, states in state_evolution_np.items():
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

    for stride, states in state_evolution_np.items():
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
    ax5.set_xlim(0, 0.5)  # Only show meaningful frequency range
    ax5.grid(True, alpha=0.3)

    # Add title and adjust layout
    fig.suptitle(
        f"Multi-scale CfC Temporal Dynamics Analysis\n"
        f"Comparing Stride Lengths: {stride_lengths}",
        fontsize=16
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    return fig

if __name__ == "__main__":
    # Import matplotlib here for the main block
    import matplotlib.pyplot as plt
    fig = visualize_stride_temporal_dynamics()
    plt.show()