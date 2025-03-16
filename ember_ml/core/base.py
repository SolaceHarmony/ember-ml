"""
Base classes for neural network implementations.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from ember_ml import ops

class BaseNeuron(ABC):
    """Base class for all neuron implementations."""
    
    def __init__(self,
                 neuron_id: int,
                 tau: float = 1.0,
                 dt: float = 0.01):
        """
        Initialize base neuron.

        Args:
            neuron_id: Unique identifier for neuron
            tau: Time constant
            dt: Time step for numerical integration
        """
        if tau <= 0:
            raise ValueError("Time constant must be positive")
        if dt <= 0:
            raise ValueError("Time step must be positive")
            
        self.neuron_id = neuron_id
        self.tau = tau
        self.dt = dt
        self.state = self._initialize_state()
        self.history: List[Any] = []
        
    @abstractmethod
    def _initialize_state(self) -> Any:
        """Initialize neuron state."""
        pass
    
    @abstractmethod
    def update(self,
               input_signal: Any,
               **kwargs) -> Any:
        """
        Update neuron state.

        Args:
            input_signal: Input to the neuron
            **kwargs: Additional parameters

        Returns:
            Updated neuron state
        """
        pass
    
    def reset(self) -> None:
        """Reset neuron state."""
        self.state = self._initialize_state()
        self.history.clear()
        
    def save_state(self) -> Dict[str, Any]:
        """
        Save neuron state.

        Returns:
            State dictionary
        """
        return {
            'neuron_id': self.neuron_id,
            'tau': self.tau,
            'dt': self.dt,
            'state': self.state,
            'history': self.history
        }
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load neuron state.

        Args:
            state_dict: State dictionary
        """
        self.neuron_id = state_dict['neuron_id']
        self.tau = state_dict['tau']
        self.dt = state_dict['dt']
        self.state = state_dict['state']
        self.history = state_dict['history']

class BaseChain(ABC):
    """Base class for neuron chains."""
    
    def __init__(self,
                 num_neurons: int,
                 neuron_class: type,
                 base_tau: float = 1.0,
                 dt: float = 0.01):
        """
        Initialize neuron chain.

        Args:
            num_neurons: Number of neurons in chain
            neuron_class: Class for chain neurons
            base_tau: Base time constant
            dt: Time step for numerical integration
        """
        if num_neurons <= 0:
            raise ValueError("Number of neurons must be positive")
        if base_tau <= 0:
            raise ValueError("Base time constant must be positive")
            
        self.num_neurons = num_neurons
        self.base_tau = base_tau
        self.dt = dt
        
        # Create neurons with progressive time constants
        self.neurons = []
        for i in range(num_neurons):
            # Progressive increase using ops functions instead of Python operators
            tau = ops.multiply(base_tau, ops.pow(1.5, i))
            neuron = neuron_class(
                neuron_id=i,
                tau=tau,
                dt=dt
            )
            self.neurons.append(neuron)
            
        # State history
        self.state_history: List[Any] = []
        
    @abstractmethod
    def update(self, input_signals: Any) -> Any:
        """
        Update chain state.

        Args:
            input_signals: Input to the chain

        Returns:
            Updated chain state
        """
        pass
    
    def reset(self) -> None:
        """Reset chain state."""
        for neuron in self.neurons:
            neuron.reset()
        self.state_history.clear()
        
    def save_state(self) -> Dict[str, Any]:
        """
        Save chain state.

        Returns:
            State dictionary
        """
        return {
            'num_neurons': self.num_neurons,
            'base_tau': self.base_tau,
            'dt': self.dt,
            'neuron_states': [n.save_state() for n in self.neurons],
            'state_history': self.state_history
        }
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load chain state.

        Args:
            state_dict: State dictionary
        """
        self.num_neurons = state_dict['num_neurons']
        self.base_tau = state_dict['base_tau']
        self.dt = state_dict['dt']
        
        for neuron, state in zip(self.neurons, state_dict['neuron_states']):
            neuron.load_state(state)
            
        self.state_history = state_dict['state_history']