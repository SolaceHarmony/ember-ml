"""
Liquid Time Constant (LTC) neuron implementations, including both Euclidean and chain variants.
"""

from typing import Optional, Union, List, Dict, Any
from ember_ml import ops
from ember_ml.core.base import BaseNeuron, BaseChain
from ember_ml.nn import tensor

class LTCNeuron(BaseNeuron):
    """Standard Euclidean LTC neuron implementation."""
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 cm: float = 1.0):
        """
        Initialize LTC neuron.

        Args:
            neuron_id: Unique identifier for the neuron
            tau: Time constant
            dt: Time step for numerical integration
            gleak: Leak conductance
            cm: Membrane capacitance
        """
        # Validate parameters
        if tau <= 0:
            raise ValueError("Time constant must be positive")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if gleak < 0:
            raise ValueError("Leak conductance must be non-negative")
        if cm <= 0:
            raise ValueError("Membrane capacitance must be positive")
            
        super().__init__(neuron_id, tau, dt)
        self.gleak = gleak
        self.cm = cm
        self.last_prediction = 0.0
        
    def _initialize_state(self) -> float:
        """Initialize neuron state to zero."""
        return 0.0
        
    def update(self, 
               input_signal: float,
               **kwargs) -> float:
        """
        Update LTC neuron state.

        Args:
            input_signal: Input to the neuron
            **kwargs: Additional parameters

        Returns:
            Updated neuron state
        """
        # Calculate state update using ops functions instead of Python operators
        dh = ops.multiply(
            ops.divide(1.0, self.tau),
            ops.subtract(input_signal, self.state)
        )
        dh = ops.subtract(dh, ops.multiply(self.gleak, self.state))
        
        # Update state using ops functions
        state_change = ops.divide(
            ops.multiply(self.dt, dh),
            self.cm
        )
        self.state = ops.add(self.state, state_change)
        
        # Store prediction for next update
        self.last_prediction = self.state
        
        # Update history
        self.history.append(self.state)
        
        return self.state
        
    def save_state(self) -> Dict[str, Any]:
        """Save neuron state and parameters."""
        state_dict = super().save_state()
        state_dict.update({
            'gleak': self.gleak,
            'cm': self.cm,
            'last_prediction': self.last_prediction
        })
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load neuron state and parameters."""
        super().load_state(state_dict)
        self.gleak = state_dict['gleak']
        self.cm = state_dict['cm']
        self.last_prediction = state_dict['last_prediction']

class LTCChain(BaseChain):
    """Chain of LTC neurons with progressive time constants."""
    
    def __init__(self,
                 num_neurons: int,
                 base_tau: float = 1.0,
                 dt: float = 0.01,
                 gleak: float = 0.5,
                 cm: float = 1.0):
        """
        Initialize LTC chain.

        Args:
            num_neurons: Number of neurons in chain
            base_tau: Base time constant
            dt: Time step
            gleak: Leak conductance
            cm: Membrane capacitance
        """
        super().__init__(
            num_neurons=num_neurons,
            neuron_class=LTCNeuron,
            base_tau=base_tau,
            dt=dt
        )
        # Initialize weights for chain connections using ops instead of numpy
        self.weights = tensor.random_uniform(
            shape=(num_neurons,),
            minval=0.5,
            maxval=1.5
        )
        
    def update(self, input_signals):
        """
        Update chain state based on inputs.

        Args:
            input_signals: Input array for the chain

        Returns:
            Updated states of all neurons
        """
        # Create zeros tensor using ops instead of numpy
        states = tensor.zeros(self.num_neurons)
        
        # Update first neuron with external input
        states_0 = self.neurons[0].update(input_signals[0])
        states = tensor.tensor_scatter_nd_update(states, [[0]], [states_0])
        
        # Update subsequent neurons using chain connections
        for i in range(1, self.num_neurons):
            # Each neuron receives weighted input from previous neuron
            # Use ops functions instead of Python operators
            prev_idx = ops.subtract(i, 1)
            chain_input = ops.multiply(self.weights[prev_idx], states[prev_idx])
            states_i = self.neurons[i].update(chain_input)
            states = tensor.tensor_scatter_nd_update(states, [[i]], [states_i])
        
        # Store chain state history
        self.state_history.append(tensor.copy(states))
        
        return states
        
    def save_state(self) -> Dict[str, Any]:
        """Save chain state and parameters."""
        state_dict = super().save_state()
        # Convert weights tensor to list for serialization
        state_dict['weights'] = tensor.to_numpy(self.weights).tolist()
        return state_dict
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load chain state and parameters."""
        super().load_state(state_dict)
        # Convert list back to tensor
        self.weights = tensor.convert_to_tensor(state_dict['weights'])

def create_ltc_chain(num_neurons: int,
                    base_tau: float = 1.0,
                    dt: float = 0.01,
                    gleak: float = 0.5,
                    cm: float = 1.0) -> LTCChain:
    """
    Factory function to create an LTC chain.

    Args:
        num_neurons: Number of neurons in chain
        base_tau: Base time constant
        dt: Time step
        gleak: Leak conductance
        cm: Membrane capacitance

    Returns:
        Configured LTC chain
    """
    return LTCChain(
        num_neurons=num_neurons,
        base_tau=base_tau,
        dt=dt,
        gleak=gleak,
        cm=cm
    )