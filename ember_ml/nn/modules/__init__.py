"""This module provides backend-agnostic implementations of neural network modules
using the ops abstraction layer.

Available Modules:
    Base Classes:
        RNNCell: Base recurrent cell
        LSTMCell: Long Short-Term Memory cell
        GRUCell: Gated Recurrent Unit cell
        
    Specialized Cells:
        StrideAware: Stride-aware cell base class
        StrideAwareCfC: Stride-aware Closed-form Continuous-time cell
        StrideAwareCell: Stride-aware general cell
        
    Advanced Modules:
        CfC: Closed-form Continuous-time cell
        WiredCfCCell: CfC with wiring capabilities
        LTC: Liquid Time-Constant cell
        LTCCell: LTC cell implementation
        StrideAwareWiredCfCCell: Stride-aware wired cell implementation

"""

from ember_ml.nn.modules.base_module import BaseModule as Module, BaseModule, Parameter
from ember_ml.nn.modules.module_cell import ModuleCell
from ember_ml.nn.modules.module_wired_cell import ModuleWiredCell
from ember_ml.nn.modules.ncp import NCP
from ember_ml.nn.modules.auto_ncp import AutoNCP
from ember_ml.nn.modules.dense import Dense # Import Dense from its new location
# Import NeuronMap classes from the new wiring sub-package
from ember_ml.nn.modules.wiring import NeuronMap, NCPMap, FullyConnectedMap, RandomMap
# Import RNN modules (keep existing)
from ember_ml.nn.modules.rnn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell, StrideAware, StrideAwareCfC, StrideAwareCell
# Import the separated layer and corrected cell import
from ember_ml.nn.modules.rnn import CfC, CfCCell, WiredCfCCell, StrideAwareWiredCfCCell, LTC, LTCCell
from ember_ml.nn.modules.rnn.stride_aware_cfc_layer import StrideAwareCfC
# Import activation modules
from ember_ml.nn.modules.activations import ReLU, Tanh, Sigmoid, Softmax, Softplus, LeCunTanh, Dropout

__all__ = [
    # Base
    'Module',
    'Parameter',
    'Module',
    'BaseModule',
    'ModuleCell',
    'ModuleWiredCell',
    'Dense', # Add Dense export
    'NCP',
    'AutoNCP', # Layer convenience class
    # NeuronMap exports (imported from .wiring)
    'NeuronMap',
    'NCPMap',
    'FullyConnectedMap',
    'RandomMap',
    # RNN exports (keep existing)
    'RNN',
    'LSTM',
    'GRU',
    'RNNCell',
    'LSTMCell', 
    'GRUCell',
    
    # Stride-aware
    'StrideAware',
    'StrideAwareCell',
    'StrideAwareWiredCfCCell',
    'StrideAwareCfC', # Now correctly imported layer class
    # Advanced modules
    'CfC',
    'CfCCell',
    'WiredCfCCell',
    'LTC',
    'LTCCell',
    # Activations
    'ReLU',
    'Tanh',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'LeCunTanh',
    'Dropout',
]