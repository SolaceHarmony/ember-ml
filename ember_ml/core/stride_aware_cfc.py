"""
Stride-Aware Closed-form Continuous-time (CfC) Neural Network

This module provides an implementation of Stride-Aware CfC cells and layers,
which extend the standard CfC with awareness of different stride lengths
for processing temporal data at multiple time scales.
"""

import ember_ml as em
from ember_ml.nn.modules import Module, 
from ember_ml.nn.modules.rnn import RNN
from ember_ml.nn.container import dense, Activation, Lambda, Multiply
from ember_ml.initializers import orthogonal, glorot_uniform, constant
import ember_ml.ops as ops
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

# Import AutoNCP wiring if available
try:
    from ncps import wirings
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False
    print("Warning: ncps package not available. AutoNCP wiring will not be available.")


class StrideAwareCfCCell(Module):
    """
    Stride-Aware Closed-form Continuous-time (CfC) cell.
    
    This cell extends the standard CfC with awareness of stride length,
    allowing it to process temporal data with different time scales.
    """
    
    def __init__(
        self,
        units: int,
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[Any] = None,
        recurrent_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        kernel_constraint: Optional[Any] = None,
        recurrent_constraint: Optional[Any] = None,
        bias_constraint: Optional[Any] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs
    ):
        """
        Initialize the Stride-Aware CfC cell.
        
        Args:
            units: Number of units in the cell
            stride_length: Length of the stride for temporal processing
            time_scale_factor: Factor to scale the time constant
            activation: Activation function for the output
            recurrent_activation: Activation function for the recurrent step
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias
            kernel_regularizer: Regularizer for the kernel weights
            recurrent_regularizer: Regularizer for the recurrent weights
            bias_regularizer: Regularizer for the bias
            kernel_constraint: Constraint for the kernel weights
            recurrent_constraint: Constraint for the recurrent weights
            bias_constraint: Constraint for the bias
            dropout: Dropout rate for the inputs
            recurrent_dropout: Dropout rate for the recurrent step
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.units = units
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))
        
        # State size: [hidden_state, time_state]
        self.state_size = [self.units, self.units]
        self.output_size = self.units
    
    def build(self, input_shape):
        """
        Build the cell weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        input_dim = input_shape[-1]
        
        # Input weights
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        
        # Recurrent weights
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
        )
        
        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        
        # Time-scale parameter (learnable)
        self.time_scale = self.add_weight(
            shape=(self.units,),
            name="time_scale",
            initializer=tf.keras.initializers.Constant(self.time_scale_factor),
            constraint=lambda x: tf.clip_by_value(x, 0.01, 10.0)
        )
        
        # Stride-aware parameter (learnable)
        self.stride_scale = self.add_weight(
            shape=(self.units,),
            name="stride_scale",
            initializer=.initializers.Constant(float(self.stride_length)),
            constraint=lambda x: tf.clip_by_value(x, 0.1, 100.0)
        )
        
        self.built = True
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through the cell.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, time_state]
            training: Whether in training mode
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Previous states
        h_prev, t_prev = states
        
        # Apply dropout
        if training is not None and (self.dropout > 0 or self.recurrent_dropout > 0):
            dp_mask = self._get_dropout_mask_for_cell(inputs, training, self.dropout)
            rec_dp_mask = self._get_recurrent_dropout_mask_for_cell(h_prev, training, self.recurrent_dropout)
        else:
            dp_mask = None
            rec_dp_mask = None
        
        if dp_mask is not None:
            inputs = inputs * dp_mask
        
        if rec_dp_mask is not None:
            h_prev = h_prev * rec_dp_mask
        
        # Compute gates
        z = K.dot(inputs, self.kernel)
        z += K.dot(h_prev, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        
        # Split into gates
        z_i, z_f, z_o, z_c = tf.split(z, 4, axis=-1)
        
        # Apply activations
        i = self.recurrent_activation(z_i)  # Input gate
        f = self.recurrent_activation(z_f)  # Forget gate
        o = self.recurrent_activation(z_o)  # Output gate
        c = self.activation(z_c)            # Cell input
        
        # Apply stride-aware time scaling
        # Scale the time constant based on stride length
        effective_time_scale = self.time_scale * self.stride_scale
        
        # Compute time decay factor
        # For longer strides, the decay is stronger
        decay = ops.exp(-1.0 / effective_time_scale)
        
        # Update time state with stride awareness
        t = f * t_prev + i * c
        
        # Apply time decay to hidden state
        h = o * self.activation(decay * h_prev + (1 - decay) * t)
        
        return h, [h, t]
    
    def get_config(self):
        """
        Get the cell configuration.
        
        Returns:
            Cell configuration dictionary
        """
        config = {
            "units": self.units,
            "stride_length": self.stride_length,
            "time_scale_factor": self.time_scale_factor,
            "activation": tf.keras.activations.serialize(self.activation),
            "recurrent_activation": tf.keras.activations.serialize(self.recurrent_activation),
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": tf.keras.initializers.serialize(self.recurrent_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "recurrent_regularizer": tf.keras.regularizers.serialize(self.recurrent_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(self.recurrent_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    def _get_dropout_mask_for_cell(self, inputs, training, dropout):
        """Create dropout mask for inputs."""
        if 0 < dropout < 1:
            ones = tf.ones_like(K.expand_dims(inputs[:, 0], -1))
            ones = tf.tile(ones, [1, 1, self.units * 4])
            return K.in_train_phase(
                K.dropout(ones, dropout),
                ones,
                training=training
            )
        return None
    
    def _get_recurrent_dropout_mask_for_cell(self, h_prev, training, recurrent_dropout):
        """Create dropout mask for recurrent connection."""
        if 0 < recurrent_dropout < 1:
            ones = ops.ones_like(K.reshape(h_prev, (-1, 1, self.units)))
            ones = tf.tile(ones, [1, 1, self.units * 4])
            return K.in_train_phase(
                K.dropout(ones, recurrent_dropout),
                ones,
                training=training
            )
        return None


class StrideAwareWiredCfCCell(StrideAwareCfCCell):
    """
    Stride-Aware CfC cell with custom wiring.
    
    This cell extends StrideAwareCfCCell with support for custom wiring,
    such as Neural Circuit Policies (NCPs).
    """
    
    def __init__(
        self,
        wiring,
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[Any] = None,
        recurrent_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        kernel_constraint: Optional[Any] = None,
        recurrent_constraint: Optional[Any] = None,
        bias_constraint: Optional[Any] = None,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs
    ):
        """
        Initialize the Stride-Aware Wired CfC cell.
        
        Args:
            wiring: Wiring configuration (e.g., AutoNCP)
            stride_length: Length of the stride for temporal processing
            time_scale_factor: Factor to scale the time constant
            activation: Activation function for the output
            recurrent_activation: Activation function for the recurrent step
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            recurrent_initializer: Initializer for the recurrent weights
            bias_initializer: Initializer for the bias
            kernel_regularizer: Regularizer for the kernel weights
            recurrent_regularizer: Regularizer for the recurrent weights
            bias_regularizer: Regularizer for the bias
            kernel_constraint: Constraint for the kernel weights
            recurrent_constraint: Constraint for the recurrent weights
            bias_constraint: Constraint for the bias
            dropout: Dropout rate for the inputs
            recurrent_dropout: Dropout rate for the recurrent step
            **kwargs: Additional keyword arguments
        """
        self.wiring = wiring
        units = wiring.units
        
        super().__init__(
            units=units,
            stride_length=stride_length,
            time_scale_factor=time_scale_factor,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )
    
    def build(self, input_shape):
        """
        Build the cell weights with wiring constraints.
        
        Args:
            input_shape: Shape of the input tensor
        """
        input_dim = input_shape[-1]
        
        # Build wiring masks
        self.input_mask, self.recurrent_mask, self.output_mask = self.wiring.build()
        
        # Input weights with wiring mask
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        
        # Apply input mask to kernel
        input_mask_4x = tf.tile(tf.expand_dims(self.input_mask, -1), [1, 4])
        self.kernel_mask = tf.reshape(input_mask_4x, [input_dim, self.units * 4])
        
        # Recurrent weights with wiring mask
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
        )
        
        # Apply recurrent mask to recurrent kernel
        recurrent_mask_4x = tf.tile(tf.expand_dims(self.recurrent_mask, -1), [1, 1, 4])
        self.recurrent_kernel_mask = tf.reshape(recurrent_mask_4x, [self.units, self.units * 4])
        
        # Bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        
        # Time-scale parameter (learnable)
        self.time_scale = self.add_weight(
            shape=(self.units,),
            name="time_scale",
            initializer=tf.keras.initializers.Constant(self.time_scale_factor),
            constraint=lambda x: tf.clip_by_value(x, 0.01, 10.0)
        )
        
        # Stride-aware parameter (learnable)
        self.stride_scale = self.add_weight(
            shape=(self.units,),
            name="stride_scale",
            initializer=tf.keras.initializers.Constant(float(self.stride_length)),
            constraint=lambda x: tf.clip_by_value(x, 0.1, 100.0)
        )
        
        self.built = True
    
    def call(self, inputs, states, training=None):
        """
        Forward pass through the cell with wiring constraints.
        
        Args:
            inputs: Input tensor
            states: Previous states [hidden_state, time_state]
            training: Whether in training mode
            
        Returns:
            Tuple of (output, [new_hidden_state, new_time_state])
        """
        # Previous states
        h_prev, t_prev = states
        
        # Apply dropout
        if training is not None and (self.dropout > 0 or self.recurrent_dropout > 0):
            dp_mask = self._get_dropout_mask_for_cell(inputs, training, self.dropout)
            rec_dp_mask = self._get_recurrent_dropout_mask_for_cell(h_prev, training, self.recurrent_dropout)
        else:
            dp_mask = None
            rec_dp_mask = None
        
        if dp_mask is not None:
            inputs = inputs * dp_mask
        
        if rec_dp_mask is not None:
            h_prev = h_prev * rec_dp_mask
        
        # Apply wiring constraints
        masked_kernel = self.kernel * self.kernel_mask
        masked_recurrent_kernel = self.recurrent_kernel * self.recurrent_kernel_mask
        
        # Compute gates with wiring constraints
        z = K.dot(inputs, masked_kernel)
        z += K.dot(h_prev, masked_recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        
        # Split into gates
        z_i, z_f, z_o, z_c = tf.split(z, 4, axis=-1)
        
        # Apply activations
        i = self.recurrent_activation(z_i)  # Input gate
        f = self.recurrent_activation(z_f)  # Forget gate
        o = self.recurrent_activation(z_o)  # Output gate
        c = self.activation(z_c)            # Cell input
        
        # Apply stride-aware time scaling
        effective_time_scale = self.time_scale * self.stride_scale
        decay = tf.exp(-1.0 / effective_time_scale)
        
        # Update time state with stride awareness
        t = f * t_prev + i * c
        
        # Apply time decay to hidden state
        h = o * self.activation(decay * h_prev + (1 - decay) * t)
        
        # Apply output mask
        output = h * self.output_mask
        
        return output, [h, t]
    
    def get_config(self):
        """
        Get the cell configuration.
        
        Returns:
            Cell configuration dictionary
        """
        config = super().get_config()
        config.update({
            "wiring": self.wiring.get_config() if hasattr(self.wiring, "get_config") else None
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Create cell from configuration.
        
        Args:
            config: Cell configuration dictionary
            
        Returns:
            Cell instance
        """
        # Handle wiring configuration
        wiring_config = config.pop("wiring", None)
        if wiring_config is not None and NCPS_AVAILABLE:
            # Recreate wiring from config
            wiring_type = wiring_config.get("type", "")
            if wiring_type == "AutoNCP":
                wiring = wirings.AutoNCP(
                    units=wiring_config.get("units", 0),
                    output_size=wiring_config.get("output_size", 0),
                    sparsity_level=wiring_config.get("sparsity_level", 0.5)
                )
            else:
                # Default to a simple wiring
                wiring = wirings.FullyConnected(
                    units=config.get("units", 0),
                    output_size=config.get("units", 0)
                )
        else:
            # Create a default wiring
            wiring = None
            if NCPS_AVAILABLE:
                wiring = wirings.FullyConnected(
                    units=config.get("units", 0),
                    output_size=config.get("units", 0)
                )
        
        # Create cell with wiring
        if wiring is not None:
            return cls(wiring=wiring, **config)
        else:
            # Fall back to parent class if wiring is not available
            return super(StrideAwareWiredCfCCell, cls).from_config(config)


class StrideAwareCfC(RNN):
    """
    Stride-Aware Closed-form Continuous-time (CfC) RNN layer.
    
    This layer wraps a StrideAwareCfCCell or StrideAwareWiredCfCCell
    to create a recurrent layer that processes temporal data with
    awareness of different stride lengths.
    """
    
    def __init__(
        self,
        cell,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        mixed_memory: bool = False,
        **kwargs
    ):
        """
        Initialize the Stride-Aware CfC layer.
        
        Args:
            cell: StrideAwareCfCCell or StrideAwareWiredCfCCell instance
            return_sequences: Whether to return the full sequence or just the last output
            return_state: Whether to return the final state
            go_backwards: Whether to process the sequence backwards
            stateful: Whether to reuse the last state for the next batch
            unroll: Whether to unroll the RNN or use symbolic loop
            mixed_memory: Whether to use mixed memory for different strides
            **kwargs: Additional keyword arguments
        """
        if not isinstance(cell, (StrideAwareCfCCell, StrideAwareWiredCfCCell)):
            raise ValueError(
                "Cell must be an instance of StrideAwareCfCCell or StrideAwareWiredCfCCell"
            )
        
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.mixed_memory = mixed_memory
        
        super().__init__(
            cell=cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )
    
    def call(self, inputs, initial_state=None, constants=None, training=None, mask=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor
            initial_state: Initial state
            constants: Constants
            training: Whether in training mode
            mask: Mask tensor
            
        Returns:
            Layer output
        """
        # Get input shape
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        
        # Create initial state if not provided
        if initial_state is None:
            initial_state = self.get_initial_state(inputs)
        
        # Apply mixed memory if enabled
        if self.mixed_memory:
            # Process with different memory timescales
            outputs = self._apply_mixed_memory(
                inputs, initial_state, training, mask
            )
        else:
            # Standard RNN processing
            outputs = super().call(
                inputs, initial_state, constants, training, mask
            )
        
        return outputs
    
    def _apply_mixed_memory(self, inputs, initial_state, training, mask):
        """
        Apply mixed memory processing with different timescales.
        
        Args:
            inputs: Input tensor
            initial_state: Initial state
            training: Whether in training mode
            mask: Mask tensor
            
        Returns:
            Layer output
        """
        # Get input shape
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        
        # Get stride length
        stride_length = self.cell.stride_length
        
        # Process with standard memory
        standard_output = super().call(
            inputs, initial_state, None, training, mask
        )
        
        if isinstance(standard_output, list):
            standard_output, *states = standard_output
        
        # If stride length is 1, just return standard output
        if stride_length <= 1:
            return standard_output if not self.return_state else [standard_output] + states
        
        # Create a version with longer memory
        # by subsampling the input sequence
        indices = tf.range(0, time_steps, stride_length)
        subsampled_inputs = tf.gather(inputs, indices, axis=1)
        
        # Process with longer memory
        long_memory_output = super().call(
            subsampled_inputs, initial_state, None, training, mask
        )
        
        if isinstance(long_memory_output, list):
            long_memory_output, *long_states = long_memory_output
        
        # Combine outputs based on return_sequences
        if self.return_sequences:
            # Upsample long memory output to match original sequence length
            repeated_indices = tf.repeat(tf.range(tf.shape(long_memory_output)[1]), stride_length)
            repeated_indices = repeated_indices[:time_steps]
            upsampled_long_output = tf.gather(long_memory_output, repeated_indices, axis=1)
            
            # Combine standard and long memory outputs
            alpha = 0.5  # Mixing factor (could be learned)
            mixed_output = alpha * standard_output + (1 - alpha) * upsampled_long_output
            
            return mixed_output if not self.return_state else [mixed_output] + states
        else:
            # For single output, combine the final outputs
            alpha = 0.5  # Mixing factor (could be learned)
            mixed_output = alpha * standard_output + (1 - alpha) * long_memory_output
            
            return mixed_output if not self.return_state else [mixed_output] + states
    
    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Layer configuration dictionary
        """
        config = {
            "mixed_memory": self.mixed_memory
        }
        base_config = super().get_config()
        
        # Remove 'cell' from base_config as it will be added separately
        if "cell" in base_config:
            cell_config = base_config.pop("cell")
            config["cell"] = {
                "class_name": self.cell.__class__.__name__,
                "config": self.cell.get_config()
            }
        
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Create layer from configuration.
        
        Args:
            config: Layer configuration dictionary
            custom_objects: Custom objects dictionary
            
        Returns:
            Layer instance
        """
        # Extract cell config
        cell_config = config.pop("cell")
        
        # Create cell from config
        if isinstance(cell_config, dict):
            cell_class_name = cell_config.get("class_name", "")
            cell_config = cell_config.get("config", {})
            
            if cell_class_name == "StrideAwareCfCCell":
                cell = StrideAwareCfCCell.from_config(cell_config)
            elif cell_class_name == "StrideAwareWiredCfCCell":
                cell = StrideAwareWiredCfCCell.from_config(cell_config)
            else:
                # Default to standard cell
                cell = StrideAwareCfCCell(**cell_config)
        else:
            # If cell_config is not a dict, assume it's already a cell
            cell = cell_config
        
        # Create layer with cell
        return cls(cell=cell, **config)


class MotorNeuron(Layer):
    """
    Motor neuron layer that outputs a value for triggering deeper exploration.
    
    This layer produces both an output value and a trigger signal that
    can be used to determine when to explore data more deeply.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        activity_regularizer: Optional[Any] = None,
        kernel_constraint: Optional[Any] = None,
        bias_constraint: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the motor neuron layer.
        
        Args:
            threshold: Threshold for triggering deeper exploration
            activation: Activation function to use
            use_bias: Whether to use bias
            kernel_initializer: Initializer for the kernel weights
            bias_initializer: Initializer for the bias
            kernel_regularizer: Regularizer for the kernel weights
            bias_regularizer: Regularizer for the bias
            activity_regularizer: Regularizer for the activity
            kernel_constraint: Constraint for the kernel weights
            bias_constraint: Constraint for the bias
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Create weights for the motor neuron
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(1,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )
        else:
            self.bias = None
        
        self.built = True
    
    def call(self, inputs, training=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Motor neuron output and trigger signal
        """
        # Compute raw output
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        # Apply activation function
        activated_output = self.activation_fn(output)
        
        # In training mode, just return the activated output
        if training:
            return activated_output
        
        # In inference mode, also compute the trigger signal
        trigger = tf.cast(activated_output > self.threshold, tf.float32)
        
        # Return both the activated output and the trigger signal
        return [activated_output, trigger]
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.
        
        Args:
            input_shape: Shape of the input tensor
            
        Returns:
            Output shape
        """
        output_shape = (input_shape[0], 1)
        return [output_shape, output_shape]
    
    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Layer configuration dictionary
        """
        config = {
            "threshold": self.threshold,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return {**base_config, **config}


class AdaptiveExplorationTrigger(Layer):
    """
    Adaptive exploration trigger that adjusts threshold based on recent history.
    
    This layer maintains a history of motor neuron outputs and adjusts
    the threshold for triggering exploration based on recent statistics.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.01,
        history_length: int = 100,
        **kwargs
    ):
        """
        Initialize the adaptive exploration trigger.
        
        Args:
            initial_threshold: Initial threshold value
            adaptation_rate: Rate at which to adapt the threshold
            history_length: Length of history to maintain
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.history_length = history_length
    
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Initialize threshold and history
        self.threshold = self.add_weight(
            name="threshold",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_threshold),
            trainable=False
        )
        
        self.output_history = self.add_weight(
            name="output_history",
            shape=(self.history_length,),
            initializer="zeros",
            trainable=False
        )
        
        self.history_index = self.add_weight(
            name="history_index",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype=tf.int32
        )
        
        self.built = True
    
    def call(self, inputs, training=None):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input tensor (motor neuron output)
            training: Whether in training mode
            
        Returns:
            Trigger signal and updated threshold
        """
        # Update history
        index = self.history_index % self.history_length
        self.output_history = tf.tensor_scatter_nd_update(
            self.output_history,
            [[index]],
            [inputs[0, 0]]  # Assuming scalar input
        )
        self.history_index.assign_add(1)
        
        # Compute trigger signal
        trigger = tf.cast(inputs > self.threshold, tf.float32)
        
        # Adapt threshold if not in training mode
        if not training:
            # Compute mean and standard deviation of recent outputs
            recent_mean = tf.reduce_mean(self.output_history)
            recent_std = tf.math.reduce_std(self.output_history)
            
            # Adjust threshold based on recent statistics
            # Move threshold toward (mean + std) to trigger on unusually high values
            target_threshold = recent_mean + recent_std
            self.threshold.assign(
                self.threshold * (1 - self.adaptation_rate) + 
                target_threshold * self.adaptation_rate
            )
        
        return [trigger, self.threshold]
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.
        
        Args:
            input_shape: Shape of the input tensor
            
        Returns:
            Output shape
        """
        output_shape = input_shape
        return [output_shape, ()]
    
    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Layer configuration dictionary
        """
        config = {
            "initial_threshold": self.initial_threshold,
            "adaptation_rate": self.adaptation_rate,
            "history_length": self.history_length
        }
        base_config = super().get_config()
        return {**base_config, **config}


def create_liquid_network_with_motor_neuron(
    input_dim: int,
    units: int = 64,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    stride_length: int = 1,
    time_scale_factor: float = 1.0,
    threshold: float = 0.5,
    adaptive_threshold: bool = True,
    mixed_memory: bool = True
):
    """
    Create a liquid neural network with motor neuron output.
    
    Args:
        input_dim: Dimension of input features
        units: Number of units in the circuit
        output_dim: Dimension of output (default: 1 for motor neuron)
        sparsity_level: Sparsity level for the connections
        stride_length: Length of the stride for temporal processing
        time_scale_factor: Factor to scale the time constant
        threshold: Initial threshold for triggering exploration
        adaptive_threshold: Whether to use adaptive threshold
        mixed_memory: Whether to use mixed memory for different strides
        
    Returns:
        Configured model with motor neuron output
    """
    if not NCPS_AVAILABLE:
        raise ImportError("ncps package is required for AutoNCP wiring")
    
    # Create AutoNCP wiring
    wiring = wirings.AutoNCP(
        units=units,
        output_size=units // 2,
        sparsity_level=sparsity_level
    )
    
    # Create CfC cell with wiring
    cell = StrideAwareWiredCfCCell(
        wiring=wiring,
        stride_length=stride_length,
        time_scale_factor=time_scale_factor
    )
    
    # Create CfC layer
    cfc_layer = StrideAwareCfC(
        cell=cell,
        return_sequences=False,
        mixed_memory=mixed_memory
    )
    
    # Create model
    inputs = tf.keras.layers.Input(shape=(None, input_dim))
    x = cfc_layer(inputs)
    
    # Add motor neuron
    motor_output = MotorNeuron(threshold=threshold)(x)
    
    # Add adaptive threshold if requested
    if adaptive_threshold:
        trigger_output = AdaptiveExplorationTrigger(
            initial_threshold=threshold
        )(motor_output)
        outputs = [motor_output, trigger_output]
    else:
        outputs = motor_output
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_lstm_gated_liquid_network(
    input_dim: int,
    units: int = 64,
    lstm_units: int = 32,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    stride_length: int = 1,
    time_scale_factor: float = 1.0,
    threshold: float = 0.5,
    adaptive_threshold: bool = True
):
    """
    Create a liquid neural network with LSTM gating.
    
    Args:
        input_dim: Dimension of input features
        units: Number of units in the CfC circuit
        lstm_units: Number of units in the LSTM gating mechanism
        output_dim: Dimension of output (default: 1 for motor neuron)
        sparsity_level: Sparsity level for the connections
        stride_length: Length of the stride for temporal processing
        time_scale_factor: Factor to scale the time constant
        threshold: Initial threshold for triggering exploration
        adaptive_threshold: Whether to use adaptive threshold
        
    Returns:
        Configured liquid neural network model with LSTM gating
    """
    if not NCPS_AVAILABLE:
        raise ImportError("ncps package is required for AutoNCP wiring")
    
    # Create AutoNCP wiring
    wiring = wirings.AutoNCP(
        units=units,
        output_size=units,  # Output all units for gating
        sparsity_level=sparsity_level
    )
    
    # Create CfC cell with wiring
    cfc_cell = StrideAwareWiredCfCCell(
        wiring=wiring,
        stride_length=stride_length,
        time_scale_factor=time_scale_factor
    )
    
    # Create CfC layer
    cfc_layer = StrideAwareCfC(
        cell=cfc_cell,
        return_sequences=True,  # Return sequences for LSTM gating
        mixed_memory=True
    )
    
    # Create model with LSTM gating
    inputs = tf.keras.layers.Input(shape=(None, input_dim))
    
    # CfC processing
    cfc_output = cfc_layer(inputs)
    
    # LSTM gating
    lstm_output = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(cfc_output)
    
    # Gating mechanism
    gate = tf.keras.layers.Dense(units, activation='sigmoid')(lstm_output)
    
    # Apply gate to CfC output (using the last timestep)
    last_cfc_output = Lambda(lambda x: x[:, -1, :])(cfc_output)
    gated_output = Multiply()([last_cfc_output, gate])
    
    # Add motor neuron
    motor_output = MotorNeuron(threshold=threshold)(gated_output)
    
    # Add adaptive threshold if requested
    if adaptive_threshold:
        trigger_output = AdaptiveExplorationTrigger(
            initial_threshold=threshold
        )(motor_output)
        outputs = [motor_output, trigger_output]
    else:
        outputs = motor_output
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_multi_stride_liquid_network(
    input_dim: int,
    stride_perspectives: List[int] = [1, 3, 5],
    units_per_stride: int = 32,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    time_scale_factor: float = 1.0,
    threshold: float = 0.5,
    adaptive_threshold: bool = True
):
    """
    Create a liquid neural network with multiple stride perspectives.
    
    Args:
        input_dim: Dimension of input features
        stride_perspectives: List of stride lengths to use
        units_per_stride: Number of units per stride perspective
        output_dim: Dimension of output (default: 1 for motor neuron)
        sparsity_level: Sparsity level for the connections
        time_scale_factor: Factor to scale the time constant
        threshold: Initial threshold for triggering exploration
        adaptive_threshold: Whether to use adaptive threshold
        
    Returns:
        Configured multi-stride liquid neural network model
    """
    if not NCPS_AVAILABLE:
        raise ImportError("ncps package is required for AutoNCP wiring")
    
    # Create inputs for each stride perspective
    inputs = []
    cfc_outputs = []
    
    for stride in stride_perspectives:
        # Create input for this stride
        input_layer = tf.keras.layers.Input(shape=(None, input_dim), name=f'input_stride_{stride}')
        inputs.append(input_layer)
        
        # Create AutoNCP wiring for this stride
        wiring = wirings.AutoNCP(
            units=units_per_stride,
            output_size=units_per_stride // 2,
            sparsity_level=sparsity_level
        )
        
        # Create CfC cell with wiring
        cell = StrideAwareWiredCfCCell(
            wiring=wiring,
            stride_length=stride,
            time_scale_factor=time_scale_factor
        )
        
        # Create CfC layer
        cfc_layer = StrideAwareCfC(
            cell=cell,
            return_sequences=False,
            mixed_memory=True
        )
        
        # Process input through CfC layer
        cfc_output = cfc_layer(input_layer)
        cfc_outputs.append(cfc_output)
    
    # Merge outputs from all stride perspectives
    if len(cfc_outputs) > 1:
        merged = tf.keras.layers.Concatenate()(cfc_outputs)
    else:
        merged = cfc_outputs[0]
    
    # Add motor neuron
    motor_output = MotorNeuron(threshold=threshold)(merged)
    
    # Add adaptive threshold if requested
    if adaptive_threshold:
        trigger_output = AdaptiveExplorationTrigger(
            initial_threshold=threshold
        )(motor_output)
        outputs = [motor_output, trigger_output]
    else:
        outputs = motor_output
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Check if ncps is available
    if not NCPS_AVAILABLE:
        print("Warning: ncps package not available. Using standard CfC cell.")
        
        # Create standard CfC cell
        cell = StrideAwareCfCCell(
            units=64,
            stride_length=2,
            time_scale_factor=1.0
        )
    else:
        # Create wiring
        wiring = wirings.AutoNCP(
            units=64,
            output_size=32,
            sparsity_level=0.5
        )
        
        # Create wired CfC cell
        cell = StrideAwareWiredCfCCell(
            wiring=wiring,
            stride_length=2,
            time_scale_factor=1.0
        )
    
    # Create CfC layer
    cfc_layer = StrideAwareCfC(
        cell=cell,
        return_sequences=False,
        mixed_memory=True
    )
    
    # Create model
    inputs = tf.keras.layers.Input(shape=(None, 10))
    x = cfc_layer(inputs)
    motor_output = MotorNeuron(threshold=0.5)(x)
    trigger_output = AdaptiveExplorationTrigger(initial_threshold=0.5)(motor_output)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=[motor_output, trigger_output])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())