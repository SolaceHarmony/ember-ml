"""
Stride-aware Continuous-time Fourier Cell (CfC) implementation.

This module provides implementations of stride-aware CfC cells and networks,
which can process data with different stride lengths.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from ember_ml import ops
from ember_ml.nn import tensor
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.rnn import CfC, LSTMCell
from ember_ml.nn.modules import AutoNCP # Updated import path

def create_liquid_network_with_motor_neuron(
    input_dim: int,
    units: int = 128,
    output_dim: int = 1,
    sparsity_level: float = 0.5,
    stride_length: int = 1,
    time_scale_factor: float = 1.0,
    threshold: float = 0.5,
    adaptive_threshold: bool = True,
    mixed_memory: bool = True
) -> Module:
    """
    Create a liquid neural network with a motor neuron.
    
    Args:
        input_dim: Input dimension
        units: Number of units in the liquid network
        output_dim: Output dimension
        sparsity_level: Sparsity level (0.0 to 1.0)
        stride_length: Stride length for temporal processing
        time_scale_factor: Time scale factor for CfC cells
        threshold: Threshold for motor neuron activation
        adaptive_threshold: Whether to use adaptive thresholding
        mixed_memory: Whether to use mixed memory
        
    Returns:
        Module: Liquid neural network with motor neuron
    """
    # Create stride-aware CfC cell
    cfc_cell = StrideAwareCfCCell(
        units=units,
        sparsity_level=sparsity_level,
        stride_length=stride_length,
        time_scale_factor=time_scale_factor
    )
    
    # Create liquid network
    liquid_network = LiquidNetworkWithMotorNeuron(
        cell=cfc_cell,
        input_dim=input_dim,
        output_dim=output_dim,
        threshold=threshold,
        adaptive_threshold=adaptive_threshold,
        mixed_memory=mixed_memory
    )
    
    return liquid_network

def create_lstm_gated_liquid_network(
    input_dim: int,
    cfc_units: int = 128,
    lstm_units: int = 32,
    output_dim: int = 1,
    sparsity_level: float = 0.5
) -> Module:
    """
    Create a liquid neural network with LSTM gating.
    
    Args:
        input_dim: Input dimension
        cfc_units: Number of units in the CfC cell
        lstm_units: Number of units in the LSTM cell
        output_dim: Output dimension
        sparsity_level: Sparsity level (0.0 to 1.0)
        
    Returns:
        Module: LSTM-gated liquid network
    """
    # Create CfC cell
    cfc_cell = CfC(
        units=cfc_units,
        sparsity_level=sparsity_level
    )
    
    # Create LSTM cell
    lstm_cell = LSTMCell(
        units=lstm_units
    )
    
    # Create gated liquid network
    gated_network = LSTMGatedLiquidNetwork(
        cfc_cell=cfc_cell,
        lstm_cell=lstm_cell,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    return gated_network

def create_multi_stride_liquid_network(
    input_dim: int,
    units: int = 128,
    output_dim: int = 1,
    stride_perspectives: List[int] = [1, 3, 5],
    sparsity_level: float = 0.5
) -> Module:
    """
    Create a multi-stride liquid neural network.
    
    Args:
        input_dim: Input dimension
        units: Number of units in each CfC cell
        output_dim: Output dimension
        stride_perspectives: List of stride lengths
        sparsity_level: Sparsity level (0.0 to 1.0)
        
    Returns:
        Module: Multi-stride liquid network
    """
    # Create CfC cells for each stride
    cfc_cells = {}
    for stride in stride_perspectives:
        cfc_cells[stride] = StrideAwareCfCCell(
            units=units,
            sparsity_level=sparsity_level,
            stride_length=stride
        )
    
    # Create multi-stride network
    multi_stride_network = MultiStrideLiquidNetwork(
        cells=cfc_cells,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    return multi_stride_network

class StrideAwareCfCCell(CfC):
    """
    Stride-aware Continuous-time Fourier Cell (CfC).
    
    This cell extends the CfC with stride-aware processing capabilities.
    """
    
    def __init__(
        self,
        units: int,
        sparsity_level: float = 0.5,
        stride_length: int = 1,
        time_scale_factor: float = 1.0,
        **kwargs
    ):
        """
        Initialize the stride-aware CfC cell.
        
        Args:
            units: Number of units
            sparsity_level: Sparsity level (0.0 to 1.0)
            stride_length: Stride length for temporal processing
            time_scale_factor: Time scale factor
            **kwargs: Additional arguments for the CfC cell
        """
        super().__init__(units=units, sparsity_level=sparsity_level, **kwargs)
        self.stride_length = stride_length
        self.time_scale_factor = time_scale_factor
        
        # Stride-aware parameter (learnable)
        self.stride_scale = self.add_weight(
            shape=(self.units,),
            name="stride_scale",
            initializer=ops.initializers.Constant(float(self.stride_length)),
            constraint=lambda x: ops.clip_by_value(x, 0.1, 100.0)
        )
        
        # Time scale parameter (learnable)
        self.time_scale = self.add_weight(
            shape=(self.units,),
            name="time_scale",
            initializer=ops.initializers.Constant(float(self.time_scale_factor)),
            constraint=lambda x: ops.clip_by_value(x, 0.1, 100.0)
        )
    
    def call(self, inputs, states, training=None):
        """
        Call the cell on inputs and states.
        
        Args:
            inputs: Input tensor
            states: State tensors
            training: Whether in training mode
            
        Returns:
            Tuple: (output, new_states)
        """
        # Apply stride scaling to inputs
        scaled_inputs = inputs * tensor.reshape(self.stride_scale, (1, -1))
        
        # Apply time scaling to states
        scaled_states = [state * tensor.reshape(self.time_scale, (1, -1)) for state in states]
        
        # Call parent class with scaled inputs and states
        outputs, new_states = super().call(scaled_inputs, scaled_states, training=training)
        
        return outputs, new_states

class LiquidNetworkWithMotorNeuron(Module):
    """
    Liquid Neural Network with a motor neuron.
    
    This network processes inputs through a liquid network and outputs
    both the processed features and trigger signals from a motor neuron.
    """
    
    def __init__(
        self,
        cell: Module,
        input_dim: int,
        output_dim: int = 1,
        threshold: float = 0.5,
        adaptive_threshold: bool = True,
        mixed_memory: bool = True
    ):
        """
        Initialize the liquid network with motor neuron.
        
        Args:
            cell: RNN cell
            input_dim: Input dimension
            output_dim: Output dimension
            threshold: Threshold for motor neuron activation
            adaptive_threshold: Whether to use adaptive thresholding
            mixed_memory: Whether to use mixed memory
        """
        super().__init__()
        self.cell = cell
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.mixed_memory = mixed_memory
        
        # Input projection
        self.input_projection = self.add_module(
            "input_projection",
            nn.modules.activations.Dense(
                units=cell.units,
                input_shape=(input_dim,)
            )
        )
        
        # Output projection
        self.output_projection = self.add_module(
            "output_projection",
            nn.modules.activations.Dense(
                units=output_dim,
                input_shape=(cell.units,)
            )
        )
        
        # Motor neuron
        self.motor_neuron = self.add_module(
            "motor_neuron",
            MotorNeuron(
                input_dim=cell.units,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold
            )
        )
        
        # Mixed memory
        if mixed_memory:
            self.memory_gate = self.add_module(
                "memory_gate",
                nn.modules.activations.Dense(
                    units=cell.units,
                    input_shape=(cell.units,),
                    activation="sigmoid"
                )
            )
    
    def call(self, inputs, states=None, training=None):
        """
        Call the network on inputs and states.
        
        Args:
            inputs: Input tensor
            states: State tensors
            training: Whether in training mode
            
        Returns:
            Tuple: (outputs, [trigger_signals, threshold_values])
        """
        # Initialize states if not provided
        if states is None:
            batch_size = tensor.shape(inputs)[0]
            time_steps = tensor.shape(inputs)[1]
            states = self.cell.get_initial_state(batch_size)
        
        # Process each time step
        outputs = []
        trigger_signals = []
        threshold_values = []
        
        for t in range(tensor.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Project input
            projected_input = self.input_projection(x_t)
            
            # Process through cell
            cell_output, states = self.cell(projected_input, states, training=training)
            
            # Apply mixed memory if enabled
            if self.mixed_memory:
                memory_gate = self.memory_gate(cell_output)
                if t > 0:
                    cell_output = memory_gate * cell_output + (1 - memory_gate) * outputs[-1]
            
            # Generate motor neuron output and trigger
            motor_output, trigger, threshold = self.motor_neuron(cell_output, training=training)
            
            # Project output
            projected_output = self.output_projection(cell_output)
            
            # Store outputs
            outputs.append(projected_output)
            trigger_signals.append(trigger)
            threshold_values.append(threshold)
        
        # Stack outputs
        outputs = tensor.stack(outputs, axis=1)
        trigger_signals = tensor.stack(trigger_signals, axis=1)
        threshold_values = tensor.stack(threshold_values, axis=1)
        
        return outputs, [trigger_signals, threshold_values]

class MotorNeuron(Module):
    """
    Motor neuron that generates output values and trigger signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.5,
        adaptive_threshold: bool = True
    ):
        """
        Initialize the motor neuron.
        
        Args:
            input_dim: Input dimension
            threshold: Threshold for activation
            adaptive_threshold: Whether to use adaptive thresholding
        """
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        
        # Output projection
        self.output_projection = self.add_module(
            "output_projection",
            nn.modules.activations.Dense(
                units=1,
                input_shape=(input_dim,),
                activation="sigmoid"
            )
        )
        
        # Adaptive threshold
        if adaptive_threshold:
            self.threshold_projection = self.add_module(
                "threshold_projection",
                nn.modules.activations.Dense(
                    units=1,
                    input_shape=(input_dim,),
                    activation="sigmoid",
                    bias_initializer=ops.initializers.Constant(threshold)
                )
            )
    
    def call(self, inputs, training=None):
        """
        Call the motor neuron on inputs.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Tuple: (output, trigger, threshold)
        """
        # Generate output
        output = self.output_projection(inputs)
        
        # Generate threshold
        if self.adaptive_threshold:
            threshold = self.threshold_projection(inputs)
        else:
            threshold = tensor.full_like(output, self.threshold)
        
        # Generate trigger
        trigger = tensor.cast(output > threshold, tensor.float32)
        
        return output, trigger, threshold

class LSTMGatedLiquidNetwork(Module):
    """
    LSTM-gated liquid neural network.
    
    This network uses an LSTM to gate the output of a CfC-based liquid network.
    """
    
    def __init__(
        self,
        cfc_cell: Module,
        lstm_cell: Module,
        input_dim: int,
        output_dim: int = 1
    ):
        """
        Initialize the LSTM-gated liquid network.
        
        Args:
            cfc_cell: CfC cell
            lstm_cell: LSTM cell
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.cfc_cell = cfc_cell
        self.lstm_cell = lstm_cell
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection for CfC
        self.cfc_input_projection = self.add_module(
            "cfc_input_projection",
            nn.modules.activations.Dense(
                units=cfc_cell.units,
                input_shape=(input_dim,)
            )
        )
        
        # Input projection for LSTM
        self.lstm_input_projection = self.add_module(
            "lstm_input_projection",
            nn.modules.activations.Dense(
                units=lstm_cell.units,
                input_shape=(input_dim,)
            )
        )
        
        # Output projection
        self.output_projection = self.add_module(
            "output_projection",
            nn.modules.activations.Dense(
                units=output_dim,
                input_shape=(cfc_cell.units + lstm_cell.units,)
            )
        )
        
        # Gating mechanism
        self.gate = self.add_module(
            "gate",
            nn.modules.activations.Dense(
                units=cfc_cell.units,
                input_shape=(lstm_cell.units,),
                activation="sigmoid"
            )
        )
    
    def call(self, inputs, states=None, training=None):
        """
        Call the network on inputs and states.
        
        Args:
            inputs: Input tensor
            states: State tensors
            training: Whether in training mode
            
        Returns:
            Tensor: Output tensor
        """
        # Initialize states if not provided
        if states is None:
            batch_size = tensor.shape(inputs)[0]
            cfc_states = self.cfc_cell.get_initial_state(batch_size)
            lstm_states = self.lstm_cell.get_initial_state(batch_size)
            states = [cfc_states, lstm_states]
        else:
            cfc_states, lstm_states = states
        
        # Process each time step
        outputs = []
        
        for t in range(tensor.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Project input for CfC
            cfc_input = self.cfc_input_projection(x_t)
            
            # Project input for LSTM
            lstm_input = self.lstm_input_projection(x_t)
            
            # Process through CfC
            cfc_output, cfc_states = self.cfc_cell(cfc_input, cfc_states, training=training)
            
            # Process through LSTM
            lstm_output, lstm_states = self.lstm_cell(lstm_input, lstm_states, training=training)
            
            # Generate gate
            gate = self.gate(lstm_output)
            
            # Apply gate to CfC output
            gated_cfc_output = gate * cfc_output
            
            # Concatenate gated CfC output and LSTM output
            combined_output = tensor.concatenate([gated_cfc_output, lstm_output], axis=-1)
            
            # Project output
            output = self.output_projection(combined_output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs
        outputs = tensor.stack(outputs, axis=1)
        
        return outputs

class MultiStrideLiquidNetwork(Module):
    """
    Multi-stride liquid neural network.
    
    This network processes inputs through multiple CfC cells with different stride lengths.
    """
    
    def __init__(
        self,
        cells: Dict[int, Module],
        input_dim: int,
        output_dim: int = 1
    ):
        """
        Initialize the multi-stride liquid network.
        
        Args:
            cells: Dictionary mapping stride lengths to CfC cells
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.cells = cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projections
        self.input_projections = {}
        for stride, cell in cells.items():
            self.input_projections[stride] = self.add_module(
                f"input_projection_{stride}",
                nn.modules.activations.Dense(
                    units=cell.units,
                    input_shape=(input_dim,)
                )
            )
        
        # Output projection
        total_units = sum(cell.units for cell in cells.values())
        self.output_projection = self.add_module(
            "output_projection",
            nn.modules.activations.Dense(
                units=output_dim,
                input_shape=(total_units,)
            )
        )
    
    def call(self, inputs, states=None, training=None):
        """
        Call the network on inputs and states.
        
        Args:
            inputs: Input tensor
            states: State tensors
            training: Whether in training mode
            
        Returns:
            Tensor: Output tensor
        """
        # Initialize states if not provided
        if states is None:
            batch_size = tensor.shape(inputs)[0]
            states = {}
            for stride, cell in self.cells.items():
                states[stride] = cell.get_initial_state(batch_size)
        
        # Process each time step
        outputs = []
        
        for t in range(tensor.shape(inputs)[1]):
            # Get input at current time step
            x_t = inputs[:, t, :]
            
            # Process through each cell
            cell_outputs = []
            for stride, cell in self.cells.items():
                # Only process at appropriate stride steps
                if t % stride == 0:
                    # Project input
                    projected_input = self.input_projections[stride](x_t)
                    
                    # Process through cell
                    cell_output, states[stride] = cell(projected_input, states[stride], training=training)
                    
                    # Store cell output
                    cell_outputs.append(cell_output)
                elif stride in states and len(states[stride]) > 0:
                    # Use previous output for non-stride steps
                    cell_outputs.append(states[stride][0])
                else:
                    # Initialize with zeros if no previous output
                    cell_outputs.append(tensor.zeros((tensor.shape(x_t)[0], cell.units)))
            
            # Concatenate cell outputs
            combined_output = tensor.concatenate(cell_outputs, axis=-1)
            
            # Project output
            output = self.output_projection(combined_output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs
        outputs = tensor.stack(outputs, axis=1)
        
        return outputs