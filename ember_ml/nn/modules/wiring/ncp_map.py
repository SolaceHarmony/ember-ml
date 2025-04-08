"""
Neural Circuit Policy (NCP) Wiring.

This module provides a wiring configuration for neural circuit policies,
which divides neurons into sensory, inter, and motor neurons.
"""

from typing import Optional, Tuple, List, Dict, Any

from ember_ml.nn import tensor

# Already imports NeuronMap correctly
from ember_ml.nn.modules.wiring.neuron_map import NeuronMap # Explicit path
from ember_ml.nn.tensor import EmberTensor, int32, zeros, ones, random_uniform

class NCPMap(NeuronMap): # Name is already correct
    """
    Neural Circuit Policy (NCP) wiring configuration.
    
    In an NCP wiring, neurons are divided into three groups:
    - Sensory neurons: Receive input from the environment
    - Inter neurons: Process information internally
    - Motor neurons: Produce output to the environment
    
    The connectivity pattern between these groups is defined by the
    sparsity level and can be customized.
    """
    
    def __init__(
        self,
        inter_neurons: int,
        command_neurons: int, # Add command_neurons
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
        units: Optional[int] = None,  # Added for compatibility with from_config
        output_dim: Optional[int] = None,  # Added for compatibility with from_config
        input_dim: Optional[int] = None,  # Added for compatibility with from_config
    ):
        """
        Initialize an NCP wiring configuration.
        
        Args:
            inter_neurons: Number of inter neurons
            command_neurons: Number of command neurons
            motor_neurons: Number of motor neurons
            sensory_neurons: Number of sensory neurons (default: 0)
            sparsity_level: Default sparsity level for all connections (default: 0.5)
            seed: Random seed for reproducibility
            sensory_to_inter_sparsity: Sparsity level for sensory to inter connections
            sensory_to_motor_sparsity: Sparsity level for sensory to motor connections
            inter_to_inter_sparsity: Sparsity level for inter to inter connections
            inter_to_motor_sparsity: Sparsity level for inter to motor connections
            motor_to_motor_sparsity: Sparsity level for motor to motor connections
            motor_to_inter_sparsity: Sparsity level for motor to inter connections
            units: Total number of units (optional, for compatibility)
            output_dim: Output dimension (optional, for compatibility)
            input_dim: Input dimension (optional, for compatibility)
        """
        # If units is provided, use it, otherwise calculate it
        if units is None:
            units = inter_neurons + command_neurons + motor_neurons + sensory_neurons # Update units calculation
        
        # If output_dim is provided, use it, otherwise use motor_neurons
        if output_dim is None:
            output_dim = motor_neurons
        
        # If input_dim is provided, use it, otherwise use units
        if input_dim is None:
            input_dim = units
        
        super().__init__(units, output_dim, input_dim, sparsity_level, seed)
        
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons # Store command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_neurons = sensory_neurons
        
        # Custom sparsity levels
        self.sensory_to_inter_sparsity = sensory_to_inter_sparsity or sparsity_level
        self.sensory_to_motor_sparsity = sensory_to_motor_sparsity or sparsity_level
        self.inter_to_inter_sparsity = inter_to_inter_sparsity or sparsity_level
        self.inter_to_motor_sparsity = inter_to_motor_sparsity or sparsity_level
        self.motor_to_motor_sparsity = motor_to_motor_sparsity or sparsity_level
        self.motor_to_inter_sparsity = motor_to_inter_sparsity or sparsity_level
    
    def build(self, input_dim=None) -> Tuple[EmberTensor, EmberTensor, EmberTensor]:
        """
        Build the NCP wiring configuration.
        
        Args:
            input_dim: Input dimension (optional)
            
        Returns:
            Tuple of (input_mask, recurrent_mask, output_mask)
        """
        # Set input_dim if provided
        if input_dim is not None:
            self.set_input_dim(input_dim)
        # Set random seed for reproducibility
        if self.seed is not None:
            tensor.set_seed(self.seed)
        
        # Create masks
        recurrent_mask = zeros((self.units, self.units), dtype=int32)
        input_mask = ones((self.input_dim,), dtype=int32)
        output_mask = zeros((self.units,), dtype=int32)
        
        # Define neuron group indices based on diagram and original source structure
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = sensory_end + self.inter_neurons
        command_start = inter_end # Add command indices
        command_end = inter_end + self.command_neurons
        motor_start = command_end # Adjust motor indices
        motor_end = command_end + self.motor_neurons
        
        # Create output mask (only motor neurons contribute to output)
        output_mask = zeros((self.units,), dtype=int32)
        
        # Set motor neurons to 1
        output_mask_list = [0] * self.units
        for i in range(motor_start, motor_end):
            output_mask_list[i] = 1
        
        # Create a new tensor with the updated values
        output_mask = EmberTensor(output_mask_list, dtype=int32)
        
        # Create connections using ops functions
        # We'll use a functional approach with lists and convert to tensors
        
        # Initialize a 2D list for the recurrent mask
        recurrent_mask_list = [[0 for _ in range(self.units)] for _ in range(self.units)]
        
        # Sensory to inter connections
        if self.sensory_neurons > 0 and self.inter_neurons > 0:
            for i in range(sensory_start, sensory_end):
                for j in range(inter_start, inter_end):
                    if random_uniform(()) >= self.sensory_to_inter_sparsity:
                        recurrent_mask_list[i][j] = 1
        
        # Sensory to command connections (NEW - Matches diagram & original source logic)
        if self.sensory_neurons > 0 and self.command_neurons > 0:
             for i in range(sensory_start, sensory_end):
                 for j in range(command_start, command_end):
                     # Using sensory_to_inter_sparsity for now, could add specific arg later
                     if random_uniform(()) >= self.sensory_to_inter_sparsity:
                         recurrent_mask_list[i][j] = 1

        # Sensory to motor connections -> REMOVE (Not in diagram or original source build logic)
        # if self.sensory_neurons > 0 and self.motor_neurons > 0:
        #     for i in range(sensory_start, sensory_end):
        #         for j in range(motor_start, motor_end):
        #             if random_uniform(()) >= self.sensory_to_motor_sparsity:
        #                 recurrent_mask_list[i][j] = 1
        
        # Inter to inter connections
        if self.inter_neurons > 0:
            for i in range(inter_start, inter_end):
                for j in range(inter_start, inter_end):
                    if random_uniform(()) >= self.inter_to_inter_sparsity:
                        recurrent_mask_list[i][j] = 1

        # Inter to command connections (NEW - Matches diagram & original source logic)
        if self.inter_neurons > 0 and self.command_neurons > 0:
            for i in range(inter_start, inter_end):
                for j in range(command_start, command_end):
                    # Using inter_to_inter_sparsity for now, could add specific arg later
                    if random_uniform(()) >= self.inter_to_inter_sparsity:
                         recurrent_mask_list[i][j] = 1
        
        # Inter to motor connections
        if self.inter_neurons > 0 and self.motor_neurons > 0:
            for i in range(inter_start, inter_end):
                for j in range(motor_start, motor_end):
                    if random_uniform(()) >= self.inter_to_motor_sparsity:
                        recurrent_mask_list[i][j] = 1
        
        # Command to motor connections (NEW - Matches diagram & original source logic)
        if self.command_neurons > 0 and self.motor_neurons > 0:
            for i in range(command_start, command_end):
                for j in range(motor_start, motor_end):
                     # Using inter_to_motor_sparsity for now, could add specific arg later
                     if random_uniform(()) >= self.inter_to_motor_sparsity:
                         recurrent_mask_list[i][j] = 1

        # Motor to motor connections -> REMOVE (Not in diagram or original source build logic)
        # if self.motor_neurons > 0:
        #     for i in range(motor_start, motor_end):
        #         for j in range(motor_start, motor_end):
        #             if random_uniform(()) >= self.motor_to_motor_sparsity:
        #                 recurrent_mask_list[i][j] = 1

        # Motor to inter connections -> REMOVE (Not in diagram or original source build logic)
        # if self.motor_neurons > 0 and self.inter_neurons > 0:
        #     for i in range(motor_start, motor_end):
        #         for j in range(inter_start, inter_end):
        #             if random_uniform(()) >= self.motor_to_inter_sparsity:
        #                 recurrent_mask_list[i][j] = 1
        
        # Convert the list to a tensor
        recurrent_mask = EmberTensor(recurrent_mask_list, dtype=int32)
        
        self._built = True # Mark map as built
        return input_mask, recurrent_mask, output_mask
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the NCP wiring.
        
        Returns:
            Dictionary containing the configuration
        """
        config = super().get_config()
        config.update({
            "inter_neurons": self.inter_neurons,
            "command_neurons": self.command_neurons, # Add command_neurons
            "motor_neurons": self.motor_neurons,
            "sensory_neurons": self.sensory_neurons,
            "sensory_to_inter_sparsity": self.sensory_to_inter_sparsity,
            "sensory_to_motor_sparsity": self.sensory_to_motor_sparsity,
            "inter_to_inter_sparsity": self.inter_to_inter_sparsity,
            "inter_to_motor_sparsity": self.inter_to_motor_sparsity,
            "motor_to_motor_sparsity": self.motor_to_motor_sparsity,
            "motor_to_inter_sparsity": self.motor_to_inter_sparsity
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NCPMap': # Update return type hint
        """
        Create an NCP wiring configuration from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            NCP wiring configuration
        """
        # Handle the case where 'units' is in the config but not needed by __init__
        # We'll keep it in the config for compatibility with the parent class
        return cls(**config)
    
    def get_neuron_groups(self) -> Dict[str, List[int]]:
        """
        Get the indices of neurons in each group.
        
        Returns:
            Dictionary mapping group names to lists of neuron indices
        """
        # Define start/end indices consistent with build method
        sensory_start = 0
        sensory_end = self.sensory_neurons
        inter_start = sensory_end
        inter_end = inter_start + self.inter_neurons
        command_start = inter_end
        command_end = command_start + self.command_neurons
        motor_start = command_end
        motor_end = self.units # self.units now includes command_neurons

        # Generate index lists
        sensory_idx = list(range(sensory_start, sensory_end))
        inter_idx = list(range(inter_start, inter_end))
        command_idx = list(range(command_start, command_end)) # Add command indices
        motor_idx = list(range(motor_start, motor_end)) # Adjust motor indices

        return {
            "sensory": sensory_idx,
            "inter": inter_idx,
            "command": command_idx, # Add command group
            "motor": motor_idx
        }