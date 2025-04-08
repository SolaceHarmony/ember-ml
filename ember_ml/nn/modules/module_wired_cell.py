"""
Module Wired Cell abstract base class.

This module provides the ModuleWiredCell abstract base class, which defines
the interface for all wired cell types in ember_ml.
"""

from typing import Optional, List, Dict, Any, Union, Tuple

from ember_ml import ops
from ember_ml.ops import stats  # Import stats module for sum operation
import ember_ml.nn.tensor as tensor
from ember_ml.nn.modules.module_cell import ModuleCell # Import ModuleCell
from ember_ml.nn.modules.wiring import NeuronMap # Use renamed base class

class ModuleWiredCell(ModuleCell): # Inherit from ModuleCell
    """
    Abstract base class for wired cell types.
    
    This class defines the interface for all wired cell types, which use
    wiring configurations to define connectivity patterns.
    """
    def __init__(
        self,
        neuron_map: Union[NeuronMap, Dict[str, Any]], # Removed input_size, Allow dict for deserialization
        mode: str = "default",
        **kwargs
    ):
        """
        Initialize a ModuleWiredCell.
        
        Args:
            # input_size removed, will be determined during build
            neuron_map: Neuron map configuration or dictionary
            mode: Mode of operation
            **kwargs: Additional arguments
        """
        # If neuron_map is a dictionary, convert it to a NeuronMap instance
        if isinstance(neuron_map, dict):
            # Get the map class name from the dictionary or default to NeuronMap
            map_class_name = neuron_map.pop("class_name", "NCPMap")
            
            # Import map classes
            from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
            map_classes = {
                "NeuronMap": NeuronMap,
                "NCPMap": NCPMap,
                "FullyConnectedMap": FullyConnectedMap,
                "RandomMap": RandomMap
            }
            
            # Get the map class and create an instance
            map_class = map_classes.get(map_class_name)
            if map_class is None:
                raise ValueError(f"Unknown map class: {map_class_name}")
            
            neuron_map = map_class(**neuron_map)
        # Store the neuron map directly using object.__setattr__ to bypass BaseModule.__setattr__
        # This prevents trying to add neuron_map as a submodule
        object.__setattr__(self, 'neuron_map', neuron_map)
        
        # Build logic moved to build method
            
        # Call ModuleCell's __init__
        # Pass wiring.units as the 'hidden_size' equivalent
        # Pass other relevant parameters if ModuleCell expects them (like activation, use_bias - let's assume defaults or add if needed)
        # Pass only hidden_size derived from map and kwargs to ModuleCell init
        # input_size will be set during build
        # Get input_dim from map if it's already built, otherwise None
        input_dim = getattr(neuron_map, 'input_dim', None)
        super().__init__(
            input_size=input_dim, # Pass map's input_dim or None
            hidden_size=neuron_map.units,
            **kwargs
        )
        
        # Store mode (already handled by ModuleCell or store specifically if needed)
        # self.mode is set by parent init or store specifically if needed
        self.mode = mode
        # self.input_size and self.hidden_size are set by ModuleCell's init
    
    @property
    def state_size(self):
        """Return the size of the cell state."""
        return self.neuron_map.units
        
    @property
    def layer_sizes(self):
        """Return the sizes of each layer."""
        # Default implementation for backward compatibility
        return [self.neuron_map.units]
    
    @property
    def num_layers(self):
        """Return the number of layers."""
        # Default implementation for backward compatibility
        return 1
    
    @property
    def sensory_size(self):
        """Return the sensory size."""
        return self.neuron_map.input_dim
    
    @property
    def motor_size(self):
        """Return the motor size."""
        return self.neuron_map.output_dim
    
    @property
    def output_size(self):
        """Return the output size."""
        return self.motor_size
    
    @property
    def synapse_count(self):
        """Return the number of synapses."""
        adj_matrix = getattr(self.neuron_map, 'adjacency_matrix', None)
        return stats.sum(ops.abs(adj_matrix)) if adj_matrix is not None else 0
    
    @property
    def sensory_synapse_count(self):
        """Return the number of sensory synapses."""
        sensory_matrix = getattr(self.neuron_map, 'sensory_adjacency_matrix', None)
        return stats.sum(ops.abs(sensory_matrix)) if sensory_matrix is not None else 0

    def build(self, input_shape):
        """Builds the cell based on input shape, builds the map, and sets dimensions."""
        # Note: self.built check is handled by BaseModule.__call__

        # Basic input shape validation (adapt as needed)
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 1:
            raise ValueError(f"Invalid input_shape received: {input_shape}")

        input_dim = input_shape[-1]

        # Build the neuron map using the runtime input dimension
        if not self.neuron_map.is_built() or self.neuron_map.input_dim != input_dim:
            self.neuron_map.build(input_dim) # This should set map._built = True

        # Verify map build
        if not self.neuron_map.is_built() or self.neuron_map.input_dim is None:
             raise RuntimeError("NeuronMap failed to build or set input_dim.")

        # Set dimensions on the cell instance *after* map is built
        # These are needed by subclasses during *their* build phase
        self.input_size = self.neuron_map.input_dim
        # self.hidden_size should already be set by ModuleCell.__init__ via map.units
        # self.output_size property delegates to map.output_dim

        # Call parent build method(s) AFTER setting dimensions
        # This assumes ModuleCell build might need input_size/hidden_size
        super().build(input_shape)
        # Note: self.built = True is handled by BaseModule.__call__ after build returns

    def forward(self, input, hx, timespans=None):
        """
        Forward pass of the cell.

        Args:
            input: Input tensor
            hx: Hidden state tensor
            timespans: Time spans for continuous-time dynamics

        Returns:
            Tuple of (output, new_state)
        """
        raise NotImplementedError("Subclasses must implement forward")

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the wired cell."""
        config = super().get_config() # Get config from ModuleCell
        # Remove input_size and hidden_size if they are solely determined by the map
        config.pop('input_size', None)
        config.pop('hidden_size', None)
        # Add map config and mode
        map_config = self.neuron_map.get_config()
        config.update({
            # Save the map's config dictionary
            "neuron_map": map_config,
            # Save the map's class name for reconstruction
            "neuron_map_class": self.neuron_map.__class__.__name__,
            "mode": self.mode,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModuleWiredCell':
        """Creates a wired cell from its configuration."""
        # Check if neuron_map is already in the config and is a NeuronMap instance
        neuron_map = config.get("neuron_map")
        map_config = {}
        
        # If neuron_map is a dict, we need to reconstruct it
        if isinstance(neuron_map, dict) or not isinstance(neuron_map, NeuronMap):
            map_config = config.pop("neuron_map", {})
            map_class_name = config.pop("neuron_map_class", "NeuronMap")
            
            # Import known map classes
            from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
            neuron_map_class_map = {
                "NeuronMap": NeuronMap,
                "NCPMap": NCPMap,
                "FullyConnectedMap": FullyConnectedMap,
                "RandomMap": RandomMap,
            }
            map_class_obj = neuron_map_class_map.get(map_class_name)
            if map_class_obj is None:
                raise ImportError(f"Unknown NeuronMap class '{map_class_name}' specified in config.")
                
            # Reconstruct the map object
            neuron_map = map_class_obj.from_config(map_config)
            
            # Add the reconstructed map object back to config
            config['neuron_map'] = neuron_map

        # Ensure 'input_size' is present in the config dict before calling __init__
        # It might be saved by ModuleCell.get_config or derivable from the map config.
        if 'input_size' not in config:
            input_size_from_map = map_config.get('input_dim')
            if input_size_from_map is None:
                # Check the reconstructed map object itself if it was built during reconstruction
                # This assumes from_config might build the map if needed, which isn't standard.
                # Safer to rely on saved config or require input_shape during load.
                input_size_from_map = getattr(neuron_map, 'input_dim', None)

            if input_size_from_map is None:
                raise ValueError("Cannot reconstruct ModuleWiredCell: 'input_size' key missing "
                               "from config and cannot be inferred from NeuronMap config.")
            config['input_size'] = input_size_from_map

        # BaseModule.from_config calls cls(**config).
        # Ensure config dict ONLY contains valid args for the specific cls.__init__
        # The cls.__init__ for ModuleWiredCell expects 'input_size', 'neuron_map', 'mode', **kwargs
        # The super().from_config call will pass this prepared config dict.
        return super(ModuleWiredCell, cls).from_config(config)