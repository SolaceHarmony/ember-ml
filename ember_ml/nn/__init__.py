"""
Neural network components module.

This module provides neural network components,
including modulation, specialized neurons, RNN layers, and neural circuit policies.
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
