"""Neural network primitives and core components.

This module provides the foundational building blocks for constructing neural
networks with backend-agnostic implementations.

Components:
    Base Modules:
        - Module: Base class for all neural components
        - Parameter: Trainable parameter abstraction
        - TensorInterface: Backend-agnostic tensor interface
        
    Neural Circuit Components:
        - AutoNCP: Automatic Neural Circuit Policy generator
        - Wiring: Base class for neural connectivity patterns
        - RandomWiring: Stochastic connectivity generator
        - NCPWiring: Neural Circuit Policy specific wiring
        
    Advanced Neurons:
        - LTCNeuronWithAttention: Liquid Time-Constant neuron with attention
        - DopamineModulator: Neuromodulatory system implementation

All components maintain strict backend independence through the ops abstraction.
"""

from ember_ml.nn.modulation import DopamineState, DopamineModulator
from ember_ml.nn.specialized import LTCNeuronWithAttention

from ember_ml.nn.modules import Module, Parameter, BaseModule
from ember_ml.nn.wirings import Wiring, FullyConnectedWiring, RandomWiring, NCPWiring, NCP, AutoNCP
from ember_ml.nn.container import Sequential
from ember_ml.nn.tensor import TensorInterface, EmberTensor
from ember_ml.nn.modules.rnn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell, StrideAware, StrideAwareCfC, StrideAwareCell
from ember_ml.nn.modules.rnn import CfC, CfCCell, WiredCfCCell, StrideAwareWiredCfCCell, LTC, LTCCell, RNNCell

__all__ = [
    'Module',
    'Parameter', 
    'BaseModule',
    'Sequential',
    'NCP',
    'AutoNCP',
    'Wiring',
    'FullyConnectedWiring',
    'RandomWiring',
    'NCPWiring',
    'TensorInterface',
    'EmberTensor',
    'LTCNeuronWithAttention',
    'DopamineState',
    'DopamineModulator',
]
